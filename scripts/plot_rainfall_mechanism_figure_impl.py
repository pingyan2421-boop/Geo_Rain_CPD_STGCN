import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import detect_change_points_professional, load_and_clean_data, load_rain_susceptibility  # noqa: E402
from data_loader.cpd_methods import CPD_METHODS  # noqa: E402


def parse_dates(time_cols):
    cleaned = [str(col).split(".")[0] for col in time_cols]
    dates = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(dates).any():
        return np.arange(len(time_cols)), False
    return dates, True


def stage_velocity(raw_seq, cps):
    rows = []
    start = 0
    for stage_id, end in enumerate(cps + [raw_seq.shape[0]]):
        segment = raw_seq[start:end]
        velocity = np.diff(segment, axis=0)
        rows.append((stage_id, float(np.mean(velocity)) if len(velocity) else 0.0))
        start = end
    return pd.DataFrame(rows, columns=["stage_id", "mean_step_velocity"])


def downslope_vector(coords, elevation):
    lon = coords[:, 0].astype(np.float64)
    lat = coords[:, 1].astype(np.float64)
    lat0 = np.deg2rad(np.nanmean(lat))
    x = (lon - np.nanmean(lon)) * 111320.0 * np.cos(lat0)
    y = (lat - np.nanmean(lat)) * 110540.0
    design = np.column_stack([x, y, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(design, elevation.astype(np.float64), rcond=None)
    dx, dy = -float(coef[0]), -float(coef[1])
    norm = math.sqrt(dx * dx + dy * dy) + 1e-12
    return dx / norm, dy / norm, (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def save_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    raw_seq, coords, elevation, time_cols, node_ids = load_and_clean_data(args.file_path)
    _, cps = detect_change_points_professional(
        raw_seq,
        penalty=args.cpd_penalty,
        mode=args.cpd_mode,
        top_ratio=args.cpd_top_ratio,
        min_size=args.cpd_min_size,
        method=args.cpd_method,
        model=args.cpd_cost,
        pelt_penalty=args.pelt_penalty,
        bfast_frequency=args.bfast_frequency,
    )
    rain_susceptibility = load_rain_susceptibility(args.rain_susceptibility, node_ids, raw_seq.shape[1])
    if rain_susceptibility is None:
        rain_susceptibility = np.zeros(raw_seq.shape[1], dtype=np.float32)

    x_dates, has_dates = parse_dates(time_cols)
    rainfall = pd.read_csv(args.rainfall_csv)
    rainfall["date"] = pd.to_datetime(rainfall["date"], errors="coerce")
    rainfall = rainfall.dropna(subset=["date"])

    motion_std = np.std(np.diff(raw_seq, axis=0), axis=0)
    top_motion = motion_std >= np.quantile(motion_std, 0.90)
    active_rain = rain_susceptibility > 0
    rain_threshold = np.quantile(rain_susceptibility[active_rain], 0.60) if np.any(active_rain) else 1.0
    top_rain = rain_susceptibility >= rain_threshold
    overlap = top_motion & top_rain
    dx, dy, azimuth = downslope_vector(coords, elevation)
    velocity_df = stage_velocity(raw_seq, cps)

    os.makedirs(args.output_dir, exist_ok=True)
    fig = plt.figure(figsize=(13, 8.2), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.88, 1.12])

    ax_rain = fig.add_subplot(gs[0, :2])
    ax_rain.bar(rainfall["date"], rainfall["rain_mm"], width=2.0, color="#9fc5d9", edgecolor="none", label="CHIRPS daily rainfall")
    mean_signal = np.mean(raw_seq, axis=1)
    ax_signal = ax_rain.twinx()
    ax_signal.plot(x_dates, mean_signal, color="#263238", linewidth=1.8, label="Mean InSAR displacement")
    for cp in cps:
        ax_signal.axvline(x_dates[cp], color="#c0392b", linewidth=1.2, alpha=0.75)
    ax_rain.set_title("Rainfall forcing and deformation-stage boundaries")
    ax_rain.set_ylabel("Rainfall (mm/day)")
    ax_signal.set_ylabel("Mean displacement (mm)")
    if has_dates:
        fig.autofmt_xdate()
    else:
        ax_rain.set_xlabel("Time index")

    ax_vel = fig.add_subplot(gs[0, 2])
    ax_vel.bar(velocity_df["stage_id"], velocity_df["mean_step_velocity"], color="#5b7f95")
    ax_vel.axhline(0, color="#333333", linewidth=0.8)
    ax_vel.set_title("Stage kinematics")
    ax_vel.set_xlabel("Stage")
    ax_vel.set_ylabel("Mean step velocity")

    ax_map = fig.add_subplot(gs[1, :2])
    sc = ax_map.scatter(coords[:, 0], coords[:, 1], c=elevation, s=4, cmap="terrain", linewidths=0, alpha=0.72)
    ax_map.scatter(coords[top_rain, 0], coords[top_rain, 1], s=9, facecolors="none", edgecolors="#1f77b4", linewidths=0.45, label="Rain-susceptible nodes")
    ax_map.scatter(coords[top_motion, 0], coords[top_motion, 1], s=9, color="#d95f02", alpha=0.55, label="Top motion nodes")
    ax_map.scatter(coords[overlap, 0], coords[overlap, 1], s=18, color="#7b3294", alpha=0.85, label="Overlap")
    center_lon = float(np.mean(coords[:, 0]))
    center_lat = float(np.mean(coords[:, 1]))
    ax_map.arrow(center_lon, center_lat, dx * 0.006, dy * 0.006, width=0.00008, head_width=0.00055, color="#111111", length_includes_head=True)
    ax_map.text(center_lon + dx * 0.0063, center_lat + dy * 0.0063, f"downslope {azimuth:.1f}°", fontsize=9, color="#111111")
    ax_map.set_title("Rain-sensitive zone, high-motion zone and terrain-constrained movement")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.legend(loc="lower left", frameon=True, fontsize=8)
    fig.colorbar(sc, ax=ax_map, label="Elevation", shrink=0.86)

    ax_scatter = fig.add_subplot(gs[1, 2])
    cumulative_delta = raw_seq[-1] - raw_seq[0]
    ax_scatter.scatter(elevation, cumulative_delta, s=5, color="#9e9e9e", alpha=0.28, label="All nodes")
    ax_scatter.scatter(elevation[top_rain], cumulative_delta[top_rain], s=8, color="#1f77b4", alpha=0.55, label="Rain-sensitive")
    ax_scatter.scatter(elevation[top_motion], cumulative_delta[top_motion], s=8, color="#d95f02", alpha=0.55, label="Top motion")
    ax_scatter.set_title("Elevation and cumulative motion")
    ax_scatter.set_xlabel("Elevation")
    ax_scatter.set_ylabel("Cumulative displacement (mm)")
    ax_scatter.legend(loc="best", fontsize=8)

    out_path = os.path.join(args.output_dir, "rainfall_mechanism_overview.png")
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    print(f">> Saved rainfall mechanism figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Create a mechanism-oriented rainfall response figure.")
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument("--rainfall_csv", default=os.path.join(project_root, "dataset", "rainfall", "chirps_daily.csv"))
    parser.add_argument("--rain_susceptibility", default=os.path.join(project_root, "output", "rain_susceptibility", "rain_susceptibility.csv"))
    parser.add_argument("--cpd_method", choices=list(CPD_METHODS), default="binseg")
    parser.add_argument("--cpd_mode", choices=["mean", "std", "top", "mix"], default="mix")
    parser.add_argument("--cpd_cost", default="l2")
    parser.add_argument("--cpd_penalty", type=float, default=10)
    parser.add_argument("--cpd_top_ratio", type=float, default=0.10)
    parser.add_argument("--cpd_min_size", type=int, default=10)
    parser.add_argument("--pelt_penalty", type=float, default=None)
    parser.add_argument("--bfast_frequency", type=int, default=23)
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "rainfall_mechanism_figures"))
    args = parser.parse_args()
    save_figure(args)


if __name__ == "__main__":
    main()
