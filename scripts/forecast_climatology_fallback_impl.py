import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.date_loader import load_and_clean_data
from data_loader.rainfall_loader import forecast_context_for_starts, load_forecast_features


def load_daily_rainfall(rainfall_csv):
    rain = pd.read_csv(rainfall_csv)
    if "date" not in rain.columns or "rain_mm" not in rain.columns:
        raise ValueError("Rainfall CSV must contain date and rain_mm columns.")
    out = rain[["date", "rain_mm"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["rain_mm"] = pd.to_numeric(out["rain_mm"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("Rainfall CSV did not contain any valid daily rows.")
    return out


def circular_day_distance(a, b):
    diff = np.abs(np.asarray(a, dtype=np.int32) - int(b))
    return np.minimum(diff, 366 - diff)


def historical_daily_climatology(prior_rain, target_date, seasonal_window_days, min_history, trailing_days):
    if prior_rain.empty:
        return 0.0
    target_doy = int(pd.Timestamp(target_date).dayofyear)
    doy = prior_rain["date"].dt.dayofyear.to_numpy(dtype=np.int32)
    seasonal = prior_rain[circular_day_distance(doy, target_doy) <= int(seasonal_window_days)]
    if len(seasonal) >= int(min_history):
        return float(max(seasonal["rain_mm"].mean(), 0.0))

    trailing_start = prior_rain["date"].max() - pd.Timedelta(days=int(trailing_days))
    trailing = prior_rain[prior_rain["date"] >= trailing_start]
    if not trailing.empty:
        return float(max(trailing["rain_mm"].mean(), 0.0))
    return float(max(prior_rain["rain_mm"].mean(), 0.0))


def rainfall_window_sum(rain_by_date, end_date, days):
    total = 0.0
    end = pd.Timestamp(end_date).normalize()
    for offset in range(int(days)):
        total += float(rain_by_date.get(end - pd.Timedelta(days=offset), 0.0))
    return total


def analog_antecedent_vector(rain_by_date, issue_date, windows):
    issue = pd.Timestamp(issue_date).normalize()
    end = issue - pd.Timedelta(days=1)
    return np.asarray([rainfall_window_sum(rain_by_date, end, days) for days in windows], dtype=np.float64)


def analog_lead_sequence(rain_by_date, issue_date, lead_days):
    issue = pd.Timestamp(issue_date).normalize()
    return np.asarray(
        [float(rain_by_date.get(issue + pd.Timedelta(days=lead), 0.0)) for lead in range(1, int(lead_days) + 1)],
        dtype=np.float64,
    )


def historical_analog_forecast(
    prior_rain,
    origin_date,
    lead_days,
    windows,
    neighbors,
    min_candidates,
    seasonal_window_days,
    season_weight,
):
    if prior_rain.empty:
        return None, 0
    origin = pd.Timestamp(origin_date).normalize()
    max_window = max(int(w) for w in windows)
    rain_by_date = {
        pd.Timestamp(row.date).normalize(): float(row.rain_mm)
        for row in prior_rain[["date", "rain_mm"]].itertuples(index=False)
    }
    first_candidate = prior_rain["date"].min() + pd.Timedelta(days=max_window)
    last_candidate = origin - pd.Timedelta(days=int(lead_days) + 1)
    if last_candidate < first_candidate:
        return None, 0

    candidates = prior_rain[(prior_rain["date"] >= first_candidate) & (prior_rain["date"] <= last_candidate)]["date"]
    if len(candidates) < int(min_candidates):
        return None, int(len(candidates))

    target_vec = analog_antecedent_vector(rain_by_date, origin, windows)
    candidate_rows = []
    for candidate in candidates:
        candidate_vec = analog_antecedent_vector(rain_by_date, candidate, windows)
        candidate_rows.append((pd.Timestamp(candidate).normalize(), candidate_vec))
    matrix = np.vstack([row[1] for row in candidate_rows])
    scale = np.nanstd(matrix, axis=0)
    scale[scale < 1e-6] = 1.0
    antecedent_distance = np.mean(np.abs((matrix - target_vec.reshape(1, -1)) / scale), axis=1)
    candidate_doy = np.asarray([row[0].dayofyear for row in candidate_rows], dtype=np.int32)
    season_distance = circular_day_distance(candidate_doy, origin.dayofyear) / max(float(seasonal_window_days), 1.0)
    score = antecedent_distance + float(season_weight) * season_distance
    k = max(1, min(int(neighbors), len(candidate_rows)))
    nearest = np.argsort(score)[:k]
    sequences = [analog_lead_sequence(rain_by_date, candidate_rows[idx][0], lead_days) for idx in nearest]
    return np.mean(np.vstack(sequences), axis=0), int(len(candidate_rows))


def missing_origin_dates(forecast_csv, insar_dates, starts, n_his):
    context = forecast_context_for_starts(forecast_csv, insar_dates, starts, n_his)
    missing = context[context["forecast_available"] <= 0.0].copy()
    return pd.to_datetime(missing["origin_date"], errors="coerce").dropna().drop_duplicates().sort_values()


def build_fallback_rows(
    origin_dates,
    rain,
    lead_days,
    seasonal_window_days,
    min_history,
    trailing_days,
    fallback_method,
    analog_windows,
    analog_neighbors,
    analog_min_candidates,
    analog_season_weight,
):
    rows = []
    for origin_date in origin_dates:
        origin = pd.Timestamp(origin_date).normalize()
        prior = rain[rain["date"] < origin]
        analog_sequence = None
        analog_candidate_count = 0
        if fallback_method == "analog":
            analog_sequence, analog_candidate_count = historical_analog_forecast(
                prior,
                origin,
                lead_days=lead_days,
                windows=analog_windows,
                neighbors=analog_neighbors,
                min_candidates=analog_min_candidates,
                seasonal_window_days=seasonal_window_days,
                season_weight=analog_season_weight,
            )
        for lead in range(1, int(lead_days) + 1):
            valid_date = origin + pd.Timedelta(days=lead)
            if analog_sequence is not None:
                precip = float(max(analog_sequence[lead - 1], 0.0))
                source = "CHIRPS historical analog fallback"
            else:
                precip = historical_daily_climatology(
                    prior,
                    valid_date,
                    seasonal_window_days=seasonal_window_days,
                    min_history=min_history,
                    trailing_days=trailing_days,
                )
                source = "CHIRPS historical climatology fallback"
            rows.append(
                {
                    "issue_date": origin.date().isoformat(),
                    "valid_date": valid_date.date().isoformat(),
                    "precip_mm_mean": precip,
                    "source": source,
                    "source_url": "",
                    "fallback_method": "analog" if analog_sequence is not None else "climatology",
                    "analog_candidates": analog_candidate_count,
                }
            )
    return pd.DataFrame(rows)


def write_report(path, input_rows, fallback_rows, output_rows, origin_dates, args):
    lines = [
        "# Forecast Climatology Fallback",
        "",
        "## Inputs",
        "",
        f"- forecast_features: {args.forecast_features}",
        f"- rainfall_features: {args.rainfall_features}",
        f"- missing origin dates filled: {len(origin_dates)}",
        f"- official forecast rows: {input_rows}",
        f"- fallback rows: {fallback_rows}",
        f"- output rows: {output_rows}",
        "",
        "## Fallback Rule",
        "",
        "Fallback rows are generated only for sample origin dates whose existing forecast_context is unavailable. "
        "For each issue date, lead days 1-15 are estimated from CHIRPS rainfall strictly before the issue date. "
        "The estimator first uses same-season historical daily climatology; if there is not enough history, it "
        "falls back to a trailing historical mean.",
        "",
        "When `--fallback_method analog` is used, each missing issue date is filled from the mean lead-day sequence "
        "of historical issue dates with similar antecedent CHIRPS rainfall. Candidate analog lead windows must be "
        "fully before the issue date, so target-period rainfall is not used.",
        "",
        "This is a non-leaking coverage baseline, not a real weather forecast.",
        "",
    ]
    if len(origin_dates):
        lines.extend(["## Filled Origin Date Range", ""])
        lines.append(f"- start: {origin_dates.min().date().isoformat()}")
        lines.append(f"- end: {origin_dates.max().date().isoformat()}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Augment unavailable forecast_context samples with non-leaking CHIRPS climatology fallback rows."
    )
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--forecast_features", required=True)
    parser.add_argument("--rainfall_features", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--report_path", default=None)
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--lead_days", type=int, default=15)
    parser.add_argument("--seasonal_window_days", type=int, default=30)
    parser.add_argument("--min_history", type=int, default=5)
    parser.add_argument("--trailing_days", type=int, default=90)
    parser.add_argument("--fallback_method", choices=("climatology", "analog"), default="climatology")
    parser.add_argument("--analog_windows", type=int, nargs="+", default=[3, 7, 15, 30])
    parser.add_argument("--analog_neighbors", type=int, default=5)
    parser.add_argument("--analog_min_candidates", type=int, default=20)
    parser.add_argument("--analog_season_weight", type=float, default=0.50)
    args = parser.parse_args()

    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    starts = np.arange(0, raw_seq.shape[0] - args.n_his - args.n_pred + 1, dtype=np.int32)
    origin_dates = missing_origin_dates(args.forecast_features, time_cols, starts, args.n_his)
    official = load_forecast_features(args.forecast_features)
    rain = load_daily_rainfall(args.rainfall_features)
    fallback = build_fallback_rows(
        origin_dates,
        rain,
        lead_days=args.lead_days,
        seasonal_window_days=args.seasonal_window_days,
        min_history=args.min_history,
        trailing_days=args.trailing_days,
        fallback_method=args.fallback_method,
        analog_windows=args.analog_windows,
        analog_neighbors=args.analog_neighbors,
        analog_min_candidates=args.analog_min_candidates,
        analog_season_weight=args.analog_season_weight,
    )

    official_out = pd.read_csv(args.forecast_features)
    output = pd.concat([official_out, fallback], ignore_index=True, sort=False)
    output["issue_date"] = pd.to_datetime(output["issue_date"], errors="coerce")
    output["valid_date"] = pd.to_datetime(output["valid_date"], errors="coerce")
    output = output.dropna(subset=["issue_date", "valid_date"])
    output = output.sort_values(["issue_date", "valid_date", "source"]).reset_index(drop=True)
    output["issue_date"] = output["issue_date"].dt.date.astype(str)
    output["valid_date"] = output["valid_date"].dt.date.astype(str)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    report_path = Path(args.report_path) if args.report_path else out_path.with_suffix(".md")
    write_report(report_path, len(official), len(fallback), len(output), origin_dates, args)
    print(f"Wrote augmented forecast CSV: {out_path}")
    print(f"Wrote fallback report: {report_path}")
    print(f"Filled missing origin dates: {len(origin_dates)}")
    print(f"Fallback rows: {len(fallback)}")


if __name__ == "__main__":
    main()
