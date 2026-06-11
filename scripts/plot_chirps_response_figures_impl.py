import argparse
import os
import sys

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


DEFAULT_COMBOS = [
    ("P95", 7, "strict"),
    ("P95", 15, "strict"),
    ("P90", 7, "strict"),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_inputs(args):
    response = pd.read_csv(args.node_response_csv)
    summary = pd.read_csv(args.summary_csv)
    rainfall = pd.read_csv(args.rainfall_csv)
    rainfall["date"] = pd.to_datetime(rainfall["date"])
    return response, summary, rainfall


def plot_rainfall_thresholds(rainfall, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = rainfall.loc[rainfall["rain_mm"] > 0, "rain_mm"]
    thresholds = {f"P{int(q * 100)}": float(positive.quantile(q)) for q in (0.50, 0.75, 0.90, 0.95)}

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(rainfall["date"], rainfall["rain_mm"], width=1.0, color="#8fb9d9", edgecolor="none", alpha=0.75)
    colors = {"P50": "#6f6f6f", "P75": "#4b8f6a", "P90": "#d68a26", "P95": "#bf3d3d"}
    for name, value in thresholds.items():
        ax.axhline(value, color=colors[name], linewidth=1.6, linestyle="--", label=f"{name} = {value:.2f} mm")
    ax.set_title("CHIRPS Daily Rainfall and Event Thresholds")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily rainfall mean in crop window (mm)")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "chirps_rainfall_thresholds.png"), dpi=220)
    plt.close(fig)


def plot_summary_heatmap(summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = ["strict", "medium", "loose"]
    thresholds = ["P50", "P75", "P90", "P95"]
    lags = sorted(summary["lag_days"].unique())

    fig, axes = plt.subplots(1, len(profiles), figsize=(14, 4.8), sharey=True)
    vmax = max(1, int(summary["responsive_nodes_p05"].max()))
    for ax, profile in zip(axes, profiles):
        grid = np.zeros((len(thresholds), len(lags)), dtype=np.float32)
        for i, threshold in enumerate(thresholds):
            for j, lag in enumerate(lags):
                row = summary[
                    (summary["threshold_level"] == threshold)
                    & (summary["lag_days"] == lag)
                    & (summary["profile"] == profile)
                ]
                if not row.empty:
                    grid[i, j] = float(row["responsive_nodes_p05"].iloc[0])
        im = ax.imshow(grid, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(profile)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels([f"+{lag}d" for lag in lags])
        ax.set_yticks(range(len(thresholds)))
        ax.set_yticklabels(thresholds)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color = "white" if grid[i, j] > vmax * 0.35 else "black"
                ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Nodes with p < 0.05", shrink=0.88)
    fig.suptitle("Rainfall-Triggered Node CPD Response Summary")
    fig.savefig(os.path.join(output_dir, "chirps_response_summary_heatmap.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hotspot_panels(response, output_dir, combos=DEFAULT_COMBOS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = response[["node_index", "longitude", "latitude"]].drop_duplicates("node_index").sort_values("node_index")
    fig, axes = plt.subplots(1, len(combos), figsize=(5.2 * len(combos), 5.0), sharex=True, sharey=True)
    if len(combos) == 1:
        axes = [axes]

    vmax = 0.0
    panels = []
    for combo in combos:
        threshold, lag, profile = combo
        panel = response[
            (response["threshold_level"] == threshold)
            & (response["lag_days"] == lag)
            & (response["profile"] == profile)
        ].copy()
        panels.append(panel)
        if not panel.empty:
            vmax = max(vmax, float(panel["lift"].replace([np.inf, -np.inf], np.nan).fillna(0).max()))
    vmax = max(vmax, 1.0)

    scatter = None
    for ax, combo, panel in zip(axes, combos, panels):
        threshold, lag, profile = combo
        ax.scatter(coords["longitude"], coords["latitude"], s=4, color="#d0d0d0", alpha=0.45, linewidths=0)
        sig = panel[panel["p_value"] < 0.05].copy()
        if not sig.empty:
            size = 18 + 16 * sig["observed_cpd_count"].clip(lower=0)
            scatter = ax.scatter(
                sig["longitude"],
                sig["latitude"],
                c=sig["lift"].clip(upper=vmax),
                s=size,
                cmap="inferno",
                vmin=0,
                vmax=vmax,
                alpha=0.92,
                linewidths=0.2,
                edgecolors="black",
            )
        ax.set_title(f"{threshold} +{lag}d {profile}\np<0.05 nodes={len(sig)}")
        ax.set_xlabel("Longitude")
        ax.grid(alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("Latitude")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, label="Lift vs random windows", shrink=0.86)
    fig.suptitle("CHIRPS Rainfall Response Hotspots")
    fig.savefig(os.path.join(output_dir, "chirps_hotspot_combo_panels.png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_top_nodes(response, output_dir, threshold="P95", lag=7, profile="strict", n=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel = response[
        (response["threshold_level"] == threshold)
        & (response["lag_days"] == lag)
        & (response["profile"] == profile)
    ].copy()
    top = (
        panel[panel["p_value"] < 0.05]
        .sort_values(["observed_cpd_count", "lift", "p_value"], ascending=[False, False, True])
        .head(n)
        .copy()
    )
    if top.empty:
        return
    labels = [str(x).replace("Interpolated_", "") for x in top["node_id"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(top))
    bars = ax.barh(y, top["lift"], color="#b04745")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Lift vs random windows")
    ax.set_title(f"Top Rainfall-Responsive Nodes | {threshold} +{lag}d {profile}")
    for bar, count, p_value in zip(bars, top["observed_cpd_count"], top["p_value"]):
        ax.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2, f"cp={count:.0f}, p={p_value:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, max(top["lift"].max() * 1.25, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "chirps_top_responsive_nodes.png"), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create CHIRPS rainfall response visualization figures.")
    parser.add_argument("--node_response_csv", default=os.path.join(project_root, "output", "node_rainfall_hotspots_chirps", "node_rain_response.csv"))
    parser.add_argument("--summary_csv", default=os.path.join(project_root, "output", "node_rainfall_hotspots_chirps", "rain_response_summary.csv"))
    parser.add_argument("--rainfall_csv", default=os.path.join(project_root, "dataset", "rainfall", "chirps_daily.csv"))
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "chirps_response_figures"))
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    response, summary, rainfall = load_inputs(args)
    plot_rainfall_thresholds(rainfall, args.output_dir)
    plot_summary_heatmap(summary, args.output_dir)
    plot_hotspot_panels(response, args.output_dir)
    plot_top_nodes(response, args.output_dir)
    print(f">> Saved CHIRPS response figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
