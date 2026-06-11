import argparse
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import detect_change_points_professional, load_and_clean_data  # noqa: E402
from data_loader.rainfall_loader import infer_bounds  # noqa: E402
from scripts.cpd_split_validate_impl import split_rolling_cpd  # noqa: E402


CHIRPS_GEFS_15DAY_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/15_day/global/data"


def build_15day_url(issue_date):
    return f"{CHIRPS_GEFS_15DAY_BASE}/{issue_date:%Y}/c3g_{issue_date:%Y.%m.%d}.tif"


def download_with_curl(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return False
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()
    cmd = [
        "curl.exe",
        "-L",
        "--ssl-no-revoke",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "-o",
        str(part_path),
        url,
    ]
    last_error = None
    for attempt in range(1, 4):
        try:
            subprocess.check_call(cmd)
            last_error = None
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(20 * attempt)
    if last_error is not None:
        raise last_error
    part_path.replace(output_path)
    return True


def force_download_with_curl(url, output_path):
    if output_path.exists():
        output_path.unlink()
    return download_with_curl(url, output_path)


def geotiff_crop_window(tif_path, bounds):
    import tifffile

    lon_min, lon_max, lat_min, lat_max = bounds
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        n_rows, n_cols = page.shape
        scale_x, scale_y, _ = page.tags["ModelPixelScaleTag"].value
        tiepoint = page.tags["ModelTiepointTag"].value
        origin_x = float(tiepoint[3])
        origin_y = float(tiepoint[4])

    c0 = max(0, int(np.floor((lon_min - origin_x) / scale_x)))
    c1 = min(n_cols, int(np.floor((lon_max - origin_x) / scale_x)) + 1)
    r0 = max(0, int(np.floor((origin_y - lat_max) / scale_y)))
    r1 = min(n_rows, int(np.floor((origin_y - lat_min) / scale_y)) + 1)
    if c0 >= c1 or r0 >= r1:
        raise ValueError(f"Invalid GeoTIFF crop window for bounds={bounds}: {(r0, r1, c0, c1)}")
    return r0, r1, c0, c1


def read_geotiff_crop(tif_path, window):
    import tifffile

    r0, r1, c0, c1 = window
    values = tifffile.imread(tif_path)[r0:r1, c0:c1].astype(np.float32, copy=True)
    values[values < -9000] = np.nan
    return values


def read_forecast_crop_with_repair(tif_path, url, bounds, skip_download):
    try:
        window = geotiff_crop_window(tif_path, bounds)
        return read_geotiff_crop(tif_path, window), "cached"
    except Exception:
        if skip_download:
            raise
        repair_path = tif_path.with_suffix(tif_path.suffix + ".repair")
        force_download_with_curl(url, repair_path)
        window = geotiff_crop_window(repair_path, bounds)
        crop = read_geotiff_crop(repair_path, window)
        try:
            repair_path.replace(tif_path)
        except PermissionError:
            pass
        return crop, "redownloaded"


def cp_rolling_origins(args):
    raw_seq, coords, _, time_cols, _ = load_and_clean_data(args.file_path)
    _, change_points = detect_change_points_professional(
        raw_seq,
        method=args.cpd_method,
    )
    starts = np.arange(0, raw_seq.shape[0] - args.n_his - args.n_pred + 1, dtype=np.int32)
    folds = split_rolling_cpd(starts, change_points, args.n_his, args.n_pred)
    if args.fold_filter:
        requested = {name.strip() for name in args.fold_filter.split(",") if name.strip()}
        folds = [fold for fold in folds if fold[0] in requested]
    if not folds:
        raise ValueError(f"No rolling CPD folds matched fold_filter={args.fold_filter!r}.")

    origins = []
    for _, train_idx, val_idx, test_idx in folds:
        for start in np.concatenate([train_idx, val_idx, test_idx]):
            origins.append(pd.to_datetime(time_cols[int(start) + args.n_his - 1]).date())
    return sorted(set(origins)), coords


def write_forecast_csv(rows, output_csv):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=["issue_date", "valid_date", "precip_mm_mean", "source", "source_url"]
        )
    df = df.sort_values(["issue_date", "valid_date"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    return df


def write_missing_csv(rows, missing_csv):
    missing_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["issue_date", "valid_date", "reason", "source_url"])
    df = df.sort_values(["issue_date", "valid_date"]).reset_index(drop=True)
    df.to_csv(missing_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download CHIRPS-GEFS 15-day forecasts and aggregate them over the project area."
    )
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--cpd_method", default="binseg")
    parser.add_argument("--fold_filter", default="cp_180")
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--cache_dir", default="dataset/forecast/chirps_gefs_15day_cache")
    parser.add_argument("--output_csv", default="dataset/forecast/chirps_gefs_15day_cp180.csv")
    parser.add_argument("--missing_csv", default=None)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit issue dates for a smoke download.")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    origins, coords = cp_rolling_origins(args)
    if args.limit is not None:
        origins = origins[: max(0, int(args.limit))]

    bounds = infer_bounds(coords, padding=args.padding)
    cache_dir = Path(args.cache_dir)
    output_csv = Path(args.output_csv)
    missing_csv = Path(args.missing_csv) if args.missing_csv else output_csv.with_name(f"{output_csv.stem}_missing.csv")
    rows = []
    missing_rows = []

    print(f"Matched {len(origins)} issue dates for fold_filter={args.fold_filter}.")
    print(f"Crop bounds={bounds}.")
    for i, issue_date in enumerate(origins, start=1):
        url = build_15day_url(issue_date)
        tif_path = cache_dir / f"c3g_{issue_date:%Y.%m.%d}.tif"
        try:
            if not args.skip_download:
                downloaded = download_with_curl(url, tif_path)
                status = "downloaded" if downloaded else "cached"
            else:
                status = "cached"
            crop, repaired_status = read_forecast_crop_with_repair(tif_path, url, bounds, args.skip_download)
            if repaired_status == "redownloaded":
                status = repaired_status
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 22:
                raise
            missing_rows.append(
                {
                    "issue_date": issue_date.isoformat(),
                    "valid_date": (issue_date + timedelta(days=15)).isoformat(),
                    "reason": "http_error",
                    "source_url": url,
                }
            )
            write_forecast_csv(rows, output_csv)
            write_missing_csv(missing_rows, missing_csv)
            print(f"[{i}/{len(origins)}] missing {tif_path.name} url={url}")
            continue
        rows.append(
            {
                "issue_date": issue_date.isoformat(),
                "valid_date": (issue_date + timedelta(days=15)).isoformat(),
                "precip_mm_mean": float(np.nanmean(crop)),
                "source": "CHIRPS-GEFS v3 15_day global",
                "source_url": url,
            }
        )
        print(f"[{i}/{len(origins)}] {status} {tif_path.name} mean={rows[-1]['precip_mm_mean']:.3f}")
        write_forecast_csv(rows, output_csv)
        write_missing_csv(missing_rows, missing_csv)

    df = write_forecast_csv(rows, output_csv)
    missing = write_missing_csv(missing_rows, missing_csv)
    print(f"Wrote {len(df)} forecast rows to {output_csv}")
    print(f"Wrote {len(missing)} missing rows to {missing_csv}")


if __name__ == "__main__":
    main()
