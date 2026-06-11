import csv
import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


CHIRPS_ERDDAP_URL = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/chirps20GlobalDailyP05.csvp"
CHIRPS_TIF_HTTP_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
CHIRPS_COG_HTTP_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05"
CHIRPS_RESOLUTION_DEG = 0.05
CHIRPS_NCOLS = 7200
CHIRPS_NROWS = 2000


def _parse_date(value):
    text = str(value)
    text = re.sub(r"\.\d+$", "", text)
    return pd.to_datetime(text).date()


def infer_bounds(coords, padding=0.05):
    """Return lon/lat bounds for the landslide area."""
    coords = np.asarray(coords, dtype=np.float64)
    lon_min = float(np.nanmin(coords[:, 0]) - padding)
    lon_max = float(np.nanmax(coords[:, 0]) + padding)
    lat_min = float(np.nanmin(coords[:, 1]) - padding)
    lat_max = float(np.nanmax(coords[:, 1]) + padding)
    return lon_min, lon_max, lat_min, lat_max


def iter_dates(start_date, end_date):
    current = _parse_date(start_date)
    end = _parse_date(end_date)
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_chirps_tif_url(date_value):
    date = _parse_date(date_value)
    filename = f"chirps-v2.0.{date:%Y.%m.%d}.tif.gz"
    return f"{CHIRPS_TIF_HTTP_BASE}/{date:%Y}/{filename}"


def build_chirps_cog_url(date_value):
    date = _parse_date(date_value)
    filename = f"chirps-v2.0.{date:%Y.%m.%d}.cog"
    return f"{CHIRPS_COG_HTTP_BASE}/{date:%Y}/{filename}"


def chirps_crop_window(bounds):
    lon_min, lon_max, lat_min, lat_max = bounds
    res = CHIRPS_RESOLUTION_DEG
    c0 = max(0, int(np.floor((lon_min + 180.0) / res)))
    c1 = min(CHIRPS_NCOLS, int(np.floor((lon_max + 180.0) / res)) + 1)
    r0 = max(0, int(np.floor((50.0 - lat_max) / res)))
    r1 = min(CHIRPS_NROWS, int(np.floor((50.0 - lat_min) / res)) + 1)
    if c0 >= c1 or r0 >= r1:
        raise ValueError(f"Invalid CHIRPS crop window for bounds={bounds}: {(r0, r1, c0, c1)}")
    return r0, r1, c0, c1


def chirps_window_lon_lat(r0, r1, c0, c1):
    cols = np.arange(c0, c1, dtype=np.float64)
    rows = np.arange(r0, r1, dtype=np.float64)
    lon = -180.0 + (cols + 0.5) * CHIRPS_RESOLUTION_DEG
    lat = 50.0 - (rows + 0.5) * CHIRPS_RESOLUTION_DEG
    return lon, lat


def download_file(url, output_path, rate_limit="300K", retries=3, retry_waits=(30, 60, 120)):
    """Download one file with resume support, preferring curl on Windows."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "--ssl-no-revoke",
            "-C",
            "-",
            "--retry",
            str(int(retries)),
            "--limit-rate",
            str(rate_limit),
            url,
            "-o",
            output_path,
        ]
        subprocess.check_call(cmd)
        return output_path

    last_error = None
    for attempt in range(int(retries) + 1):
        try:
            mode = "ab" if os.path.exists(output_path) else "wb"
            headers = {}
            existing = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            if existing:
                headers["Range"] = f"bytes={existing}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, open(output_path, mode) as out:
                shutil.copyfileobj(response, out)
            return output_path
        except Exception as exc:
            last_error = exc
            if attempt < int(retries):
                time.sleep(retry_waits[min(attempt, len(retry_waits) - 1)])
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def gzip_is_readable(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception:
        return False


def read_chirps_tif_crop(tif_path, window):
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to read CHIRPS GeoTIFF files.") from exc

    r0, r1, c0, c1 = window
    with Image.open(tif_path) as image:
        crop = image.crop((c0, r0, c1, r1))
        values = np.array(crop, dtype=np.float32, copy=True)
    values[values < -9000] = np.nan
    return values


def read_chirps_cog_crop(url, window):
    try:
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:
        raise RuntimeError("rasterio is required to read CHIRPS COG files.") from exc

    r0, r1, c0, c1 = window
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".cog"):
        with rasterio.open(url) as dataset:
            values = dataset.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    values[values < -9000] = np.nan
    return values


def write_chirps_crop_outputs(records, output_dir, bounds, crop_bounds, window, source_url_base=CHIRPS_TIF_HTTP_BASE):
    if not records:
        raise ValueError("No CHIRPS crop records to write.")

    os.makedirs(output_dir, exist_ok=True)
    dates = np.array([row["date"] for row in records], dtype=object)
    rain_stack = np.stack([row["rain_mm_grid"] for row in records]).astype(np.float32)
    r0, r1, c0, c1 = window
    lon, lat = chirps_window_lon_lat(r0, r1, c0, c1)

    npz_path = os.path.join(output_dir, "chirps_crop_daily.npz")
    np.savez_compressed(npz_path, dates=dates, lon=lon, lat=lat, rain_mm=rain_stack)

    csv_path = os.path.join(output_dir, "chirps_crop_daily.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "row", "col", "lon", "lat", "rain_mm"])
        for day_idx, row in enumerate(records):
            grid = rain_stack[day_idx]
            for rr in range(grid.shape[0]):
                for cc in range(grid.shape[1]):
                    writer.writerow([
                        row["date"],
                        rr,
                        cc,
                        f"{lon[cc]:.6f}",
                        f"{lat[rr]:.6f}",
                        f"{float(grid[rr, cc]):.6f}" if np.isfinite(grid[rr, cc]) else "",
                    ])

    daily_path = os.path.join(os.path.dirname(output_dir), "chirps_daily.csv")
    daily_rows = []
    for row in records:
        grid = row["rain_mm_grid"]
        daily_rows.append({
            "date": row["date"],
            "rain_mm": float(np.nanmean(grid)),
            "source": "CHIRPS-2.0-global-daily-tif-p05-crop",
            "lon_min": crop_bounds[0],
            "lon_max": crop_bounds[1],
            "lat_min": crop_bounds[2],
            "lat_max": crop_bounds[3],
        })
    pd.DataFrame(daily_rows).to_csv(daily_path, index=False, encoding="utf-8")

    metadata = {
        "source": "CHIRPS-2.0 global_daily tifs p05",
        "source_url_base": source_url_base,
        "resolution_degree": CHIRPS_RESOLUTION_DEG,
        "bbox_landslide": list(map(float, bounds)),
        "bbox_crop": list(map(float, crop_bounds)),
        "window": {"row_start": r0, "row_end": r1, "col_start": c0, "col_end": c1},
        "date_start": records[0]["date"],
        "date_end": records[-1]["date"],
        "days": len(records),
        "outputs": {
            "crop_csv": csv_path,
            "crop_npz": npz_path,
            "daily_mean_csv": daily_path,
            "daily_crop_cache_dir": os.path.join(output_dir, "daily"),
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return {"crop_csv": csv_path, "crop_npz": npz_path, "daily_csv": daily_path}


def fetch_chirps_tif_crop(
    start_date,
    end_date,
    bounds,
    output_dir,
    buffer_degree=0.25,
    rate_limit="300K",
    sleep_seconds=2.0,
    keep_raw=False,
    source="tif",
):
    """Download CHIRPS daily GeoTIFFs, crop around the landslide area, and save compact outputs."""
    lon_min, lon_max, lat_min, lat_max = bounds
    crop_bounds = (
        lon_min - float(buffer_degree),
        lon_max + float(buffer_degree),
        lat_min - float(buffer_degree),
        lat_max + float(buffer_degree),
    )
    window = chirps_crop_window(crop_bounds)
    tmp_dir = os.path.join(output_dir, "_tmp")
    daily_dir = os.path.join(output_dir, "daily")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(daily_dir, exist_ok=True)

    dates = list(iter_dates(start_date, end_date))
    total_days = len(dates)
    records = []
    for day_number, date in enumerate(dates, start=1):
        date_text = date.isoformat()
        daily_cache = os.path.join(daily_dir, f"{date_text}.npz")
        progress = f"[{day_number}/{total_days} {day_number / max(total_days, 1) * 100:.1f}%]"
        if os.path.exists(daily_cache):
            with np.load(daily_cache) as cached:
                records.append({"date": date_text, "rain_mm_grid": cached["rain_mm"].astype(np.float32)})
            print(f">> {progress} CHIRPS {date_text}: cached crop", flush=True)
            continue

        if source == "cog":
            url = build_chirps_cog_url(date)
            print(f">> {progress} CHIRPS {date_text}: cog crop", flush=True)
            crop = read_chirps_cog_crop(url, window)
            np.savez_compressed(daily_cache, rain_mm=crop.astype(np.float32))
            records.append({"date": date_text, "rain_mm_grid": crop})
        elif source == "tif":
            url = build_chirps_tif_url(date)
            gz_path = os.path.join(tmp_dir, os.path.basename(url))
            tif_path = gz_path[:-3]
            print(f">> {progress} CHIRPS {date_text}: download/crop", flush=True)
            try:
                if not gzip_is_readable(gz_path):
                    download_file(url, gz_path, rate_limit=rate_limit)
                with gzip.open(gz_path, "rb") as src, open(tif_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                crop = read_chirps_tif_crop(tif_path, window)
                np.savez_compressed(daily_cache, rain_mm=crop.astype(np.float32))
                records.append({"date": date_text, "rain_mm_grid": crop})
            finally:
                if not keep_raw:
                    for path in (gz_path, tif_path):
                        if os.path.exists(path):
                            os.remove(path)
        else:
            raise ValueError(f"Unsupported CHIRPS source: {source}")
        if sleep_seconds:
            time.sleep(float(sleep_seconds))

    source_url_base = CHIRPS_COG_HTTP_BASE if source == "cog" else CHIRPS_TIF_HTTP_BASE
    outputs = write_chirps_crop_outputs(records, output_dir, bounds, crop_bounds, window, source_url_base=source_url_base)
    if not keep_raw and os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
        os.rmdir(tmp_dir)
    return outputs


def build_chirps_erddap_url(start_date, end_date, bounds):
    """Build an ERDDAP CSV request for CHIRPS daily precipitation."""
    lon_min, lon_max, lat_min, lat_max = bounds
    start = _parse_date(start_date).isoformat() + "T00:00:00Z"
    end = _parse_date(end_date).isoformat() + "T00:00:00Z"
    query = (
        "precip"
        f"[({start}):1:({end})]"
        f"[({lat_min:.4f}):1:({lat_max:.4f})]"
        f"[({lon_min:.4f}):1:({lon_max:.4f})]"
    )
    return CHIRPS_ERDDAP_URL + "?" + urllib.parse.quote(query, safe="[]():,?=./+-")


def fetch_chirps_region_mean(start_date, end_date, bounds, output_csv, timeout=120):
    """Download CHIRPS daily precipitation and save a regional mean table."""
    url = build_chirps_erddap_url(start_date, end_date, bounds)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read().decode("utf-8")

    reader = csv.DictReader(raw.splitlines())
    by_date = {}
    for row in reader:
        time_key = next((key for key in row if key and key.startswith("time")), "time")
        precip_key = next((key for key in row if key and key.startswith("precip")), "precip")
        try:
            date = pd.to_datetime(row[time_key]).date()
        except Exception:
            continue
        value = pd.to_numeric(row.get(precip_key), errors="coerce")
        if np.isfinite(value):
            by_date.setdefault(date, []).append(float(value))

    if not by_date:
        raise ValueError("CHIRPS request returned no precipitation values.")

    lon_min, lon_max, lat_min, lat_max = bounds
    rows = []
    for date in sorted(by_date):
        rows.append({
            "date": date.isoformat(),
            "rain_mm": float(np.mean(by_date[date])),
            "source": "CHIRPS-2.0-global-daily-0.05deg",
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        })
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")
    return output_csv


def load_daily_rainfall(path):
    df = pd.read_csv(path)
    required = {"date", "rain_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Rainfall CSV is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0.0)
    numeric_cols = ["rain_mm"]
    optional_cols = [
        "soil_moisture",
        "surface_soil_moisture",
        "rootzone_soil_moisture",
        "runoff",
        "surface_runoff",
        "subsurface_runoff",
        "evapotranspiration",
        "temperature",
    ]
    for col in optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            numeric_cols.append(col)
    return df.groupby("date", as_index=False)[numeric_cols].mean().sort_values("date")


def _rolling_sum(series, ts, window):
    if window == 0:
        return float(series.get(ts, 0.0))
    start = ts - pd.Timedelta(days=window - 1)
    return float(series.loc[start:ts].sum())


def _effective_antecedent_rain(rain_series, ts, max_days=45, decay=0.90):
    start = ts - pd.Timedelta(days=max_days - 1)
    window = rain_series.loc[start:ts].to_numpy(dtype=np.float64)
    if window.size == 0:
        return 0.0, 0.0
    age = np.arange(window.size - 1, -1, -1, dtype=np.float64)
    exp_weights = decay ** age
    power_weights = 1.0 / np.power(age + 1.0, 1.25)
    return float(np.sum(window * exp_weights)), float(np.sum(window * power_weights))


def _series_anomaly(series, ts, window=30):
    start = ts - pd.Timedelta(days=window - 1)
    values = series.loc[start:ts].to_numpy(dtype=np.float64)
    if values.size == 0:
        return 0.0
    baseline = series.to_numpy(dtype=np.float64)
    std = float(np.nanstd(baseline))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return float((np.nanmean(values) - np.nanmean(baseline)) / std)


def _merge_climate_features(rainfall, climate_csv):
    if not climate_csv:
        return rainfall
    climate = pd.read_csv(climate_csv)
    if "date" not in climate.columns:
        raise ValueError("Climate feature CSV must contain a date column.")
    climate = climate.copy()
    climate["date"] = pd.to_datetime(climate["date"]).dt.date.astype(str)
    rainfall = rainfall.copy()
    rainfall["date"] = pd.to_datetime(rainfall["date"]).dt.date.astype(str)
    climate_cols = [c for c in climate.columns if c != "date"]
    overlap = [c for c in climate_cols if c in rainfall.columns]
    if overlap:
        rainfall = rainfall.drop(columns=overlap)
    return rainfall.merge(climate, on="date", how="left")


def align_rainfall_to_insar(rainfall_csv, insar_dates, lag_windows=(0, 3, 7, 15, 30), climate_csv=None):
    """Create rainfall and hydrologic proxy features aligned to InSAR acquisition dates.

    The CSV must contain date/rain_mm. Optional daily columns such as
    soil_moisture, runoff, surface_runoff, subsurface_runoff, evapotranspiration,
    and temperature are treated as coarse hydrologic state proxies. These are
    not node-level pore-pressure observations.
    """
    rainfall = _merge_climate_features(load_daily_rainfall(rainfall_csv), climate_csv)
    dates = [_parse_date(d) for d in insar_dates]
    if not dates:
        raise ValueError("No InSAR dates were provided.")

    lag_windows = tuple(lag_windows)
    positive = rainfall.loc[rainfall["rain_mm"] > 0, "rain_mm"]
    event_thresholds = {}
    if not positive.empty:
        event_thresholds = {
            "p90": float(positive.quantile(0.90)),
            "p95": float(positive.quantile(0.95)),
        }
    full_index = pd.date_range(min(dates) - timedelta(days=max(lag_windows + (45,))), max(dates), freq="D")
    daily = rainfall.set_index(pd.to_datetime(rainfall["date"]))
    rain_series = daily["rain_mm"].reindex(full_index, fill_value=0.0).astype(float)
    hydro_series = {}
    for col in daily.columns:
        if col in ("date", "rain_mm"):
            continue
        hydro_series[col] = daily[col].reindex(full_index).interpolate(limit_direction="both").fillna(0.0).astype(float)

    rows = []
    prev_date = None
    for date in dates:
        ts = pd.Timestamp(date)
        row = {"date": date.isoformat()}
        for window in lag_windows:
            row[f"rain_{window}d"] = _rolling_sum(rain_series, ts, window)
        for level, threshold in event_thresholds.items():
            for event_window in (3, 7, 15):
                start = ts - pd.Timedelta(days=event_window - 1)
                row[f"rain_event_{level}_{event_window}d"] = float(rain_series.loc[start:ts].max() >= threshold)
        if prev_date is None:
            row["rain_since_last_insar"] = row["rain_0d"]
        else:
            row["rain_since_last_insar"] = float(rain_series.loc[pd.Timestamp(prev_date) + pd.Timedelta(days=1):ts].sum())
        exp_rain, power_rain = _effective_antecedent_rain(rain_series, ts)
        row["hydro_effective_rain_exp45"] = exp_rain
        row["hydro_effective_rain_power45"] = power_rain
        row["hydro_short_long_ratio"] = row.get("rain_7d", 0.0) / max(row.get("rain_30d", 0.0), 1e-6)
        row["hydro_dry_to_wet_transition"] = float(row.get("rain_7d", 0.0) > 0.0 and row.get("rain_30d", 0.0) <= np.nanmedian(rain_series.rolling(30, min_periods=1).sum()))
        runoff_cols = [c for c in ("runoff", "surface_runoff", "subsurface_runoff") if c in hydro_series]
        if runoff_cols:
            row["hydro_runoff_proxy"] = float(sum(_rolling_sum(hydro_series[c], ts, 7) for c in runoff_cols))
        else:
            row["hydro_runoff_proxy"] = float(row.get("rain_3d", 0.0) * row.get("hydro_short_long_ratio", 0.0))
        if "soil_moisture" in hydro_series:
            row["hydro_soil_moisture_anomaly"] = _series_anomaly(hydro_series["soil_moisture"], ts)
        elif "surface_soil_moisture" in hydro_series:
            row["hydro_soil_moisture_anomaly"] = _series_anomaly(hydro_series["surface_soil_moisture"], ts)
        elif "rootzone_soil_moisture" in hydro_series:
            row["hydro_soil_moisture_anomaly"] = _series_anomaly(hydro_series["rootzone_soil_moisture"], ts)
        else:
            row["hydro_soil_moisture_anomaly"] = row["hydro_effective_rain_exp45"] / max(float(np.nanpercentile(rain_series, 95)), 1e-6)
        if "evapotranspiration" in hydro_series:
            row["hydro_water_balance_15d"] = row.get("rain_15d", 0.0) - _rolling_sum(hydro_series["evapotranspiration"], ts, 15)
        else:
            row["hydro_water_balance_15d"] = row.get("rain_15d", 0.0)
        row["hydro_memory_index"] = (
            0.40 * row["hydro_effective_rain_exp45"]
            + 0.25 * row["hydro_soil_moisture_anomaly"]
            + 0.20 * row["hydro_runoff_proxy"]
            + 0.15 * row["rain_since_last_insar"]
        )
        rows.append(row)
        prev_date = date

    return pd.DataFrame(rows)


def normalized_rain_matrix(aligned_df):
    feature_cols = [c for c in aligned_df.columns if c.startswith("rain_") or c.startswith("hydro_")]
    values = aligned_df[feature_cols].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((values - mean) / std).astype(np.float32), feature_cols


def _first_existing(columns, names, required_name):
    for name in names:
        if name in columns:
            return name
    raise ValueError(f"Forecast feature CSV must contain {required_name}.")


def load_forecast_features(forecast_csv):
    """Load issue-date forecast precipitation rows.

    Supported column aliases keep the interface CSV-first:
    issue_date/forecast_date, valid_date/date, and rain_mm/precip_mm/
    precip_mm_mean/precipitation_mm.
    """
    forecast = pd.read_csv(forecast_csv)
    issue_col = _first_existing(forecast.columns, ("issue_date", "forecast_date", "run_date"), "issue_date")
    valid_col = _first_existing(forecast.columns, ("valid_date", "date", "target_date"), "valid_date")
    rain_col = _first_existing(
        forecast.columns,
        ("rain_mm", "precip_mm", "precip_mm_mean", "precipitation_mm"),
        "rain/precipitation amount",
    )
    keep_cols = [issue_col, valid_col, rain_col]
    source_col = "source" if "source" in forecast.columns else None
    if source_col:
        keep_cols.append(source_col)
    out = forecast[keep_cols].copy()
    out.columns = ["issue_date", "valid_date", "forecast_rain_mm"] + (["source"] if source_col else [])
    out["issue_date"] = pd.to_datetime(out["issue_date"], errors="coerce")
    out["valid_date"] = pd.to_datetime(out["valid_date"], errors="coerce")
    out["forecast_rain_mm"] = pd.to_numeric(out["forecast_rain_mm"], errors="coerce").fillna(0.0)
    if "source" not in out.columns:
        out["source"] = ""
    out["source"] = out["source"].fillna("").astype(str)
    out = out.dropna(subset=["issue_date", "valid_date"])
    if out.empty:
        raise ValueError("Forecast feature CSV did not contain any valid issue/valid date rows.")
    return out.sort_values(["issue_date", "valid_date"]).reset_index(drop=True)


def forecast_context_for_starts(forecast_csv, insar_dates, starts, n_his, windows=(1, 3, 7, 15)):
    """Build sample-level future rainfall summaries without using observed future rain."""
    forecast = load_forecast_features(forecast_csv)
    dates = pd.to_datetime([_parse_date(d) for d in insar_dates])

    rows = []
    max_window = max(windows)
    for start in starts:
        origin_date = pd.Timestamp(dates[int(start) + int(n_his) - 1])
        issued = forecast[forecast["issue_date"] <= origin_date]
        row = {
            "origin_date": origin_date.date().isoformat(),
            "forecast_available": 0.0,
            "forecast_is_fallback": 0.0,
            "forecast_issue_age_days": float(max_window),
        }
        if issued.empty:
            for window in windows:
                row[f"forecast_rain_{window}d"] = 0.0
            row["forecast_event_p90_15d"] = 0.0
            row["forecast_event_p95_15d"] = 0.0
            rows.append(row)
            continue
        issue_date = issued["issue_date"].max()
        latest = issued[issued["issue_date"] == issue_date]
        row["forecast_issue_age_days"] = float((origin_date - issue_date).days)
        if row["forecast_issue_age_days"] > max_window:
            for window in windows:
                row[f"forecast_rain_{window}d"] = 0.0
            row["forecast_event_p90_15d"] = 0.0
            row["forecast_event_p95_15d"] = 0.0
            rows.append(row)
            continue
        row["forecast_available"] = 1.0
        source_text = " ".join(latest.get("source", pd.Series(dtype=str)).astype(str).tolist()).lower()
        row["forecast_is_fallback"] = float("fallback" in source_text or "climatology" in source_text)
        for window in windows:
            end_date = origin_date + pd.Timedelta(days=int(window))
            valid = latest[(latest["valid_date"] > origin_date) & (latest["valid_date"] <= end_date)]
            row[f"forecast_rain_{window}d"] = float(valid["forecast_rain_mm"].sum())
        row["forecast_event_p90_15d"] = 0.0
        row["forecast_event_p95_15d"] = 0.0
        rows.append(row)
    context = pd.DataFrame(rows)
    rain_15d = pd.to_numeric(context.get("forecast_rain_15d", 0.0), errors="coerce").fillna(0.0)
    positive = rain_15d[rain_15d > 0.0]
    if not positive.empty:
        p90 = float(positive.quantile(0.90))
        p95 = float(positive.quantile(0.95))
        context["forecast_event_p90_15d"] = (rain_15d >= p90).astype(np.float32)
        context["forecast_event_p95_15d"] = (rain_15d >= p95).astype(np.float32)
    return context


def main():
    parser = argparse.ArgumentParser(description="Download and crop CHIRPS daily GeoTIFF rainfall data.")
    parser.add_argument("--file_path", default=os.path.join("dataset", "inter228_5241.csv"))
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--buffer_degree", type=float, default=0.25)
    parser.add_argument("--output_dir", default=os.path.join("dataset", "rainfall", "chirps_crop"))
    parser.add_argument("--rate_limit", default="300K")
    parser.add_argument("--sleep_seconds", type=float, default=2.0)
    parser.add_argument("--keep_raw", action="store_true")
    parser.add_argument("--source", choices=["tif", "cog"], default="tif")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from data_loader.date_loader import load_and_clean_data

    _, coords, _, time_cols, _ = load_and_clean_data(args.file_path)
    start_date = args.start_date or time_cols[0]
    end_date = args.end_date or time_cols[-1]
    bounds = infer_bounds(coords, padding=0.0)
    outputs = fetch_chirps_tif_crop(
        start_date,
        end_date,
        bounds,
        args.output_dir,
        buffer_degree=args.buffer_degree,
        rate_limit=args.rate_limit,
        sleep_seconds=args.sleep_seconds,
        keep_raw=args.keep_raw,
        source=args.source,
    )
    print(">> Saved CHIRPS crop outputs:")
    for name, path in outputs.items():
        print(f"   {name}: {path}")


if __name__ == "__main__":
    main()
