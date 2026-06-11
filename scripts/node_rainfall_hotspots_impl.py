import argparse
import csv
import os
import re
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import load_and_clean_data  # noqa: E402
from data_loader.rainfall_loader import fetch_chirps_tif_crop, infer_bounds, load_daily_rainfall  # noqa: E402
from scripts.build_rain_susceptibility_impl import build_susceptibility  # noqa: E402


CPD_PROFILES = {
    "strict": {"min_size": 10, "n_bkps": 5},
    "medium": {"min_size": 6, "n_bkps": 8},
    "loose": {"min_size": 4, "n_bkps": 12},
}


def parse_insar_date(value):
    text = re.sub(r"\.\d+$", "", str(value))
    return pd.to_datetime(text).date()


def detect_node_change_points(raw_seq, profile_name, profile, max_nodes=None):
    """Detect exploratory node-level CPD candidates for one sensitivity profile."""
    try:
        import ruptures as rpt
    except Exception as exc:
        raise RuntimeError("ruptures is required for node-level CPD detection.") from exc

    t_len, n_nodes = raw_seq.shape
    n_nodes_run = n_nodes if max_nodes is None else min(int(max_nodes), n_nodes)
    rows = []
    for node_idx in range(n_nodes_run):
        signal = raw_seq[:, node_idx].astype(np.float64)
        std = signal.std()
        if not np.isfinite(std) or std < 1e-6:
            continue
        signal = ((signal - signal.mean()) / std).reshape(-1, 1)
        try:
            algo = rpt.Binseg(model="l2", min_size=profile["min_size"]).fit(signal)
            bkps = algo.predict(n_bkps=profile["n_bkps"])
        except Exception:
            continue
        for cp in bkps[:-1]:
            if profile["min_size"] <= cp <= t_len - profile["min_size"]:
                rows.append((node_idx, int(cp), profile_name))
    return rows


def build_rain_events(rainfall_csv, start_date, end_date, quantiles=(0.50, 0.75, 0.90, 0.95)):
    rainfall = load_daily_rainfall(rainfall_csv)
    rainfall = rainfall[(rainfall["date"] >= start_date) & (rainfall["date"] <= end_date)].copy()
    positive = rainfall.loc[rainfall["rain_mm"] > 0, "rain_mm"]
    if positive.empty:
        raise ValueError("Rainfall data has no positive rain days in the InSAR time span.")

    thresholds = {f"P{int(q * 100)}": float(positive.quantile(q)) for q in quantiles}
    events = []
    for level, threshold in thresholds.items():
        selected = rainfall[rainfall["rain_mm"] >= threshold]
        for row in selected.itertuples(index=False):
            events.append({
                "event_date": row.date,
                "rain_mm": float(row.rain_mm),
                "threshold_level": level,
                "threshold_mm": threshold,
            })
    return pd.DataFrame(events), thresholds


def make_event_windows(events, lag_days):
    rows = []
    for row in events.itertuples(index=False):
        end_date = row.event_date + timedelta(days=int(lag_days))
        rows.append({
            "threshold_level": row.threshold_level,
            "event_date": row.event_date,
            "rain_mm": row.rain_mm,
            "threshold_mm": row.threshold_mm,
            "lag_days": int(lag_days),
            "window_start": row.event_date,
            "window_end": end_date,
        })
    return pd.DataFrame(rows)


def indices_in_windows(dates, windows):
    selected = set()
    for row in windows.itertuples(index=False):
        for idx, date in enumerate(dates):
            if row.window_start <= date <= row.window_end:
                selected.add(idx)
    return selected


def random_window_indices(dates, n_windows, lag_days, rng):
    selected = set()
    if not dates:
        return selected
    min_date, max_date = min(dates), max(dates)
    span_days = max(1, (max_date - min_date).days - int(lag_days))
    for _ in range(int(n_windows)):
        start = min_date + timedelta(days=int(rng.randint(0, span_days + 1)))
        end = start + timedelta(days=int(lag_days))
        for idx, date in enumerate(dates):
            if start <= date <= end:
                selected.add(idx)
    return selected


def response_for_windows(cpd_df, dates, windows, profile, rng, permutations):
    obs_indices = indices_in_windows(dates, windows)
    profile_cps = cpd_df[cpd_df["profile"] == profile]
    observed = (
        profile_cps[profile_cps["time_index"].isin(obs_indices)]
        .groupby("node_index")
        .size()
    )

    n_nodes = int(cpd_df["node_index"].max()) + 1 if not cpd_df.empty else 0
    observed_arr = np.zeros(n_nodes, dtype=np.float32)
    if not observed.empty:
        observed_arr[observed.index.to_numpy(dtype=np.int32)] = observed.to_numpy(dtype=np.float32)

    baseline = np.zeros((int(permutations), n_nodes), dtype=np.float32)
    for i in range(int(permutations)):
        base_indices = random_window_indices(dates, len(windows), int(windows["lag_days"].iloc[0]), rng)
        base_counts = (
            profile_cps[profile_cps["time_index"].isin(base_indices)]
            .groupby("node_index")
            .size()
        )
        if not base_counts.empty:
            baseline[i, base_counts.index.to_numpy(dtype=np.int32)] = base_counts.to_numpy(dtype=np.float32)

    baseline_mean = baseline.mean(axis=0) if len(baseline) else np.zeros(n_nodes, dtype=np.float32)
    p_values = ((baseline >= observed_arr[None, :]).sum(axis=0) + 1.0) / (int(permutations) + 1.0)
    lift = observed_arr / np.maximum(baseline_mean, 1e-6)
    return observed_arr, baseline_mean, lift, p_values


def summarize_responses(response_df):
    rows = []
    for keys, group in response_df.groupby(["threshold_level", "lag_days", "profile"]):
        level, lag_days, profile = keys
        rows.append({
            "threshold_level": level,
            "lag_days": int(lag_days),
            "profile": profile,
            "responsive_nodes_p05": int((group["p_value"] < 0.05).sum()),
            "responsive_nodes_p10": int((group["p_value"] < 0.10).sum()),
            "nodes_with_observed_cpd": int((group["observed_cpd_count"] > 0).sum()),
            "max_observed_cpd_count": float(group["observed_cpd_count"].max()),
            "max_lift": float(group["lift"].replace([np.inf, -np.inf], np.nan).fillna(0).max()),
        })
    return pd.DataFrame(rows).sort_values(["threshold_level", "lag_days", "profile"])


def add_stability(response_df):
    key_cols = ["node_index", "threshold_level", "lag_days"]
    stable = (
        response_df.assign(hit=response_df["observed_cpd_count"] > 0)
        .groupby(key_cols)["hit"]
        .sum()
        .rename("sensitivity_hits")
        .reset_index()
    )
    return response_df.merge(stable, on=key_cols, how="left")


def save_hotspot_plots(output_dir, response_df, coords):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    best = response_df.sort_values(["sensitivity_hits", "observed_cpd_count", "lift"], ascending=False).head(1)
    if best.empty:
        return
    row = best.iloc[0]
    panel = response_df[
        (response_df["threshold_level"] == row["threshold_level"])
        & (response_df["lag_days"] == row["lag_days"])
        & (response_df["profile"] == row["profile"])
    ].copy()
    values = panel.sort_values("node_index")["observed_cpd_count"].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=8, cmap="inferno")
    ax.set_title(f"Rainfall CPD hotspot | {row['threshold_level']} +{int(row['lag_days'])}d {row['profile']}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(sc, ax=ax, label="Observed CPD count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rain_response_hotspots.png"), dpi=200)
    plt.close(fig)

    lags = sorted(response_df["lag_days"].unique())
    fig, axes = plt.subplots(1, len(lags), figsize=(4 * len(lags), 4), sharex=True, sharey=True)
    if len(lags) == 1:
        axes = [axes]
    vmax = max(1.0, float(response_df["observed_cpd_count"].max()))
    for ax, lag in zip(axes, lags):
        subset = response_df[
            (response_df["threshold_level"] == row["threshold_level"])
            & (response_df["lag_days"] == lag)
            & (response_df["profile"] == row["profile"])
        ].sort_values("node_index")
        vals = subset["observed_cpd_count"].to_numpy(dtype=np.float32)
        ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=5, cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"+{int(lag)}d")
    fig.suptitle(f"Lag panels | {row['threshold_level']} {row['profile']}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rain_response_lag_panels.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Node-level rainfall-triggered CPD hotspot analysis.")
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument("--rainfall_csv", default=os.path.join(project_root, "dataset", "rainfall", "chirps_daily.csv"))
    parser.add_argument("--download_chirps_crop", action="store_true")
    parser.add_argument("--chirps_crop_dir", default=os.path.join(project_root, "dataset", "rainfall", "chirps_crop"))
    parser.add_argument("--chirps_buffer_degree", type=float, default=0.25)
    parser.add_argument("--chirps_rate_limit", default="300K")
    parser.add_argument("--chirps_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--lag_days", default="3,7,15,30")
    parser.add_argument("--quantiles", default="0.50,0.75,0.90,0.95")
    parser.add_argument("--permutations", type=int, default=300)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "node_rainfall_hotspots"))
    parser.add_argument("--rain_susceptibility_output", default=os.path.join(project_root, "output", "rain_susceptibility", "rain_susceptibility.csv"))
    parser.add_argument("--rain_susceptibility_hotspot_quantile", type=float, default=0.75)
    args = parser.parse_args()

    raw_seq, coords, _, time_cols, node_ids = load_and_clean_data(args.file_path)
    dates = [parse_insar_date(d) for d in time_cols]
    lag_days = [int(x.strip()) for x in args.lag_days.split(",") if x.strip()]
    quantiles = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]
    n_nodes_run = raw_seq.shape[1] if args.max_nodes is None else min(args.max_nodes, raw_seq.shape[1])

    if args.download_chirps_crop:
        outputs = fetch_chirps_tif_crop(
            time_cols[0],
            time_cols[-1],
            infer_bounds(coords, padding=0.0),
            args.chirps_crop_dir,
            buffer_degree=args.chirps_buffer_degree,
            rate_limit=args.chirps_rate_limit,
            sleep_seconds=args.chirps_sleep_seconds,
        )
        args.rainfall_csv = outputs["daily_csv"]

    cpd_rows = []
    for profile_name, profile in CPD_PROFILES.items():
        print(f">> Detecting node CPD profile={profile_name}: {profile}")
        cpd_rows.extend(detect_node_change_points(raw_seq, profile_name, profile, max_nodes=n_nodes_run))

    os.makedirs(args.output_dir, exist_ok=True)
    cpd_df = pd.DataFrame(cpd_rows, columns=["node_index", "time_index", "profile"])
    cpd_df["node_id"] = cpd_df["node_index"].map(lambda i: node_ids[int(i)])
    cpd_df["date"] = cpd_df["time_index"].map(lambda i: dates[int(i)].isoformat())
    cpd_df.to_csv(os.path.join(args.output_dir, "node_change_points.csv"), index=False, encoding="utf-8")

    events, thresholds = build_rain_events(args.rainfall_csv, min(dates), max(dates), quantiles=quantiles)
    window_frames = []
    for lag in lag_days:
        window_frames.append(make_event_windows(events, lag))
    event_windows = pd.concat(window_frames, ignore_index=True)
    event_windows.to_csv(os.path.join(args.output_dir, "rain_event_windows.csv"), index=False, encoding="utf-8")

    rng = np.random.RandomState(7)
    response_rows = []
    for (level, lag), windows in event_windows.groupby(["threshold_level", "lag_days"]):
        for profile_name in CPD_PROFILES:
            observed, baseline, lift, p_values = response_for_windows(
                cpd_df,
                dates,
                windows,
                profile_name,
                rng,
                args.permutations,
            )
            for node_idx in range(n_nodes_run):
                response_rows.append({
                    "node_index": node_idx,
                    "node_id": node_ids[node_idx],
                    "longitude": float(coords[node_idx, 0]),
                    "latitude": float(coords[node_idx, 1]),
                    "threshold_level": level,
                    "lag_days": int(lag),
                    "profile": profile_name,
                    "observed_cpd_count": float(observed[node_idx]),
                    "baseline_cpd_count": float(baseline[node_idx]),
                    "lift": float(lift[node_idx]),
                    "p_value": float(p_values[node_idx]),
                })

    response_df = pd.DataFrame(response_rows)
    response_df = add_stability(response_df)
    response_path = os.path.join(args.output_dir, "node_rain_response.csv")
    response_df.to_csv(response_path, index=False, encoding="utf-8")
    summary_df = summarize_responses(response_df)
    summary_df.to_csv(os.path.join(args.output_dir, "rain_response_summary.csv"), index=False, encoding="utf-8")
    save_hotspot_plots(args.output_dir, response_df, coords[:n_nodes_run])
    if n_nodes_run == raw_seq.shape[1]:
        build_susceptibility(
            response_path,
            args.file_path,
            args.rain_susceptibility_output,
            hotspot_quantile=args.rain_susceptibility_hotspot_quantile,
        )
    else:
        print(">> Skipped rain susceptibility export because --max_nodes did not cover all nodes.")

    print(f">> Rain thresholds: {thresholds}")
    print(f">> Saved node rainfall hotspot outputs to: {args.output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
