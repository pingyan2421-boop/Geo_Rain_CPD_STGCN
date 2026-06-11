import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import detect_change_points_professional, load_and_clean_data
from data_loader.cpd_methods import CPD_METHODS
from data_loader.rainfall_loader import (
    align_rainfall_to_insar,
    fetch_chirps_region_mean,
    fetch_chirps_tif_crop,
    infer_bounds,
)


def cluster_nodes(coords, n_regions, seed=7):
    try:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_regions, random_state=seed, n_init=10).fit_predict(coords)
    except TypeError:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_regions, random_state=seed).fit_predict(coords)
    return labels.astype(np.int32)


def detect_region_change_points(raw_seq, labels, n_regions, cpd_args):
    region_cps = {}
    for region_id in range(n_regions):
        node_idx = np.where(labels == region_id)[0]
        if len(node_idx) == 0:
            region_cps[region_id] = []
            continue
        region_seq = raw_seq[:, node_idx]
        _, cps = detect_change_points_professional(
            region_seq,
            penalty=cpd_args.cpd_penalty,
            mode=cpd_args.cpd_mode,
            top_ratio=cpd_args.cpd_top_ratio,
            min_size=cpd_args.cpd_min_size,
            method=cpd_args.cpd_method,
            model=cpd_args.cpd_cost,
            pelt_penalty=cpd_args.pelt_penalty,
            bfast_frequency=cpd_args.bfast_frequency,
        )
        region_cps[region_id] = cps
    return region_cps


def compute_cpd_density(n_time, region_cps, event_window=1):
    density = np.zeros(n_time, dtype=np.float32)
    n_regions = max(1, len(region_cps))
    for cps in region_cps.values():
        for cp in cps:
            start = max(0, int(cp) - event_window)
            end = min(n_time, int(cp) + event_window + 1)
            density[start:end] += 1.0
    return density / float(n_regions)


def spearman_corr(x, y):
    try:
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(x, y)
        return float(corr), float(p_value)
    except Exception:
        return float("nan"), float("nan")


def permutation_p_value(x, y, n_perm=1000, seed=7):
    rng = np.random.RandomState(seed)
    observed, _ = spearman_corr(x, y)
    if not np.isfinite(observed):
        return float("nan")
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(y)
        corr, _ = spearman_corr(x, permuted)
        if np.isfinite(corr) and abs(corr) >= abs(observed):
            count += 1
    return float((count + 1) / (n_perm + 1))


def write_region_cps(path, region_cps):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "change_point_index"])
        for region, cps in region_cps.items():
            for cp in cps:
                writer.writerow([region, cp])


def save_response_plot(path, response_df, summary_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(response_df["date"])
    ax1.bar(dates, response_df["rain_7d"], width=8, color="#4c78a8", alpha=0.35, label="7-day rainfall")
    ax1.set_ylabel("Rainfall (mm)")

    ax2 = ax1.twinx()
    ax2.plot(dates, response_df["cpd_density"], color="#d62728", linewidth=2, label="CPD density")
    ax2.set_ylabel("CPD density")
    ax2.set_ylim(0, max(1.0, float(response_df["cpd_density"].max()) * 1.2))

    best = summary_df.sort_values("permutation_p_value").head(1)
    title = "Rainfall response and regional CPD density"
    if not best.empty:
        row = best.iloc[0]
        title += f" | best={row['feature']} rho={row['spearman_r']:.3f}"
    ax1.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze rainfall response of regional CPD density.")
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument("--rainfall_csv", default=os.path.join(project_root, "dataset", "rainfall", "chirps_daily.csv"))
    parser.add_argument("--download_chirps", action="store_true")
    parser.add_argument("--download_chirps_crop", action="store_true")
    parser.add_argument("--chirps_crop_dir", default=os.path.join(project_root, "dataset", "rainfall", "chirps_crop"))
    parser.add_argument("--chirps_buffer_degree", type=float, default=0.25)
    parser.add_argument("--chirps_rate_limit", default="300K")
    parser.add_argument("--chirps_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--n_regions", type=int, default=12)
    parser.add_argument("--event_window", type=int, default=1)
    parser.add_argument("--cpd_mode", default="mix", choices=["mean", "std", "top", "mix"])
    parser.add_argument("--cpd_method", default="binseg", choices=list(CPD_METHODS))
    parser.add_argument("--cpd_cost", default="l2")
    parser.add_argument("--cpd_penalty", type=float, default=10)
    parser.add_argument("--cpd_top_ratio", type=float, default=0.10)
    parser.add_argument("--cpd_min_size", type=int, default=10)
    parser.add_argument("--pelt_penalty", type=float, default=None)
    parser.add_argument("--bfast_frequency", type=int, default=23)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "rainfall_cpd_response"))
    args = parser.parse_args()

    raw_seq, coords, elevation, time_cols, _ = load_and_clean_data(args.file_path)
    del elevation

    if args.download_chirps_crop:
        bounds = infer_bounds(coords, padding=0.0)
        outputs = fetch_chirps_tif_crop(
            time_cols[0],
            time_cols[-1],
            bounds,
            args.chirps_crop_dir,
            buffer_degree=args.chirps_buffer_degree,
            rate_limit=args.chirps_rate_limit,
            sleep_seconds=args.chirps_sleep_seconds,
        )
        args.rainfall_csv = outputs["daily_csv"]
    elif args.download_chirps or not os.path.exists(args.rainfall_csv):
        bounds = infer_bounds(coords, padding=0.05)
        fetch_chirps_region_mean(time_cols[0], time_cols[-1], bounds, args.rainfall_csv)

    labels = cluster_nodes(coords, args.n_regions)
    region_cps = detect_region_change_points(raw_seq, labels, args.n_regions, args)
    cpd_density = compute_cpd_density(raw_seq.shape[0], region_cps, event_window=args.event_window)
    rain_features = align_rainfall_to_insar(args.rainfall_csv, time_cols)
    rain_features["cpd_density"] = cpd_density

    feature_cols = [c for c in rain_features.columns if c.startswith("rain_")]
    summary_rows = []
    for col in feature_cols:
        corr, scipy_p = spearman_corr(rain_features[col].values, cpd_density)
        perm_p = permutation_p_value(rain_features[col].values, cpd_density, n_perm=args.permutations)
        summary_rows.append({
            "feature": col,
            "spearman_r": corr,
            "scipy_p_value": scipy_p,
            "permutation_p_value": perm_p,
        })
    summary = pd.DataFrame(summary_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    rain_features.to_csv(os.path.join(args.output_dir, "rainfall_cpd_response.csv"), index=False, encoding="utf-8")
    summary.to_csv(os.path.join(args.output_dir, "rainfall_cpd_summary.csv"), index=False, encoding="utf-8")
    write_region_cps(os.path.join(args.output_dir, "regional_change_points.csv"), region_cps)
    save_response_plot(os.path.join(args.output_dir, "rainfall_cpd_response.png"), rain_features, summary)

    print(f">> Saved rainfall CPD response outputs to: {args.output_dir}")
    print(summary.sort_values("permutation_p_value").to_string(index=False))


if __name__ == "__main__":
    main()
