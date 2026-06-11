import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import load_and_clean_data  # noqa: E402
from data_loader.rainfall_loader import align_rainfall_to_insar  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Screen lagged rainfall/hydro candidates for mechanism-MoE gating."
    )
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument("--rainfall_csv", default=os.path.join(project_root, "dataset", "rainfall", "chirps_daily.csv"))
    parser.add_argument(
        "--rain_susceptibility",
        default=os.path.join(project_root, "output", "rain_susceptibility", "rain_susceptibility.csv"),
    )
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "mechanism_moe_causal_lag"))
    parser.add_argument("--max_lag_steps", type=int, default=6)
    parser.add_argument("--top_quantile", type=float, default=0.60)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def safe_spearman(x, y):
    try:
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(x, y)
        return float(corr), float(p_value)
    except Exception:
        x_rank = pd.Series(x).rank().to_numpy(dtype=np.float64)
        y_rank = pd.Series(y).rank().to_numpy(dtype=np.float64)
        if np.nanstd(x_rank) < 1e-9 or np.nanstd(y_rank) < 1e-9:
            return float("nan"), float("nan")
        return float(np.corrcoef(x_rank, y_rank)[0, 1]), float("nan")


def permutation_p_value(x, y, observed, permutations, rng):
    if not np.isfinite(observed):
        return float("nan")
    count = 0
    for _ in range(int(permutations)):
        permuted = rng.permutation(y)
        corr, _ = safe_spearman(x, permuted)
        if np.isfinite(corr) and abs(corr) >= abs(observed):
            count += 1
    return float((count + 1.0) / (int(permutations) + 1.0))


def load_susceptibility(path, n_nodes, top_quantile):
    if not path or not os.path.exists(path):
        susceptibility = np.zeros(n_nodes, dtype=np.float32)
    else:
        df = pd.read_csv(path)
        if "rain_susceptibility" not in df.columns:
            raise ValueError(f"{path} is missing rain_susceptibility column.")
        susceptibility = df["rain_susceptibility"].to_numpy(dtype=np.float32)
        if susceptibility.shape[0] != n_nodes:
            raise ValueError(f"rain_susceptibility length={susceptibility.shape[0]} does not match n_nodes={n_nodes}.")

    active = susceptibility > 0
    top = np.zeros_like(active, dtype=bool)
    if active.any():
        threshold = float(np.quantile(susceptibility[active], float(top_quantile)))
        top = susceptibility >= threshold
    return susceptibility, active, top


def response_series(raw_seq, node_mask):
    increments = np.diff(raw_seq, axis=0)
    if node_mask is not None:
        increments = increments[:, node_mask]
    abs_response = np.mean(np.abs(increments), axis=1)
    signed_response = np.mean(increments, axis=1)
    acceleration = np.diff(abs_response, prepend=abs_response[0])
    return {
        "abs_increment": abs_response.astype(np.float64),
        "signed_increment": signed_response.astype(np.float64),
        "abs_increment_change": acceleration.astype(np.float64),
    }


def aligned_feature_frame(rainfall_csv, time_cols):
    aligned = align_rainfall_to_insar(rainfall_csv, time_cols)
    feature_cols = [c for c in aligned.columns if c.startswith("rain_") or c.startswith("hydro_")]
    if not feature_cols:
        raise ValueError("No rain_* or hydro_* features were generated.")
    features = aligned[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return aligned[["date"]].join(features), feature_cols


def compute_lag_rows(feature_df, feature_cols, responses, max_lag_steps, permutations, seed):
    rows = []
    rng = np.random.RandomState(seed)
    feature_values = feature_df[feature_cols].to_numpy(dtype=np.float64)

    # responses are defined for intervals ending at acquisition index 1..T-1.
    for scope, response_map in responses.items():
        for target_name, target in response_map.items():
            for lag in range(int(max_lag_steps) + 1):
                end_indices = np.arange(1, feature_values.shape[0], dtype=np.int32)
                feature_indices = end_indices - int(lag)
                valid = feature_indices >= 0
                if valid.sum() < 8:
                    continue
                y = target[valid]
                for col_idx, feature in enumerate(feature_cols):
                    x = feature_values[feature_indices[valid], col_idx]
                    if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
                        corr, scipy_p, perm_p = float("nan"), float("nan"), float("nan")
                    else:
                        corr, scipy_p = safe_spearman(x, y)
                        perm_p = permutation_p_value(x, y, corr, permutations, rng)

                    high = x >= np.nanquantile(x, 0.90)
                    low = x <= np.nanquantile(x, 0.50)
                    high_mean = float(np.nanmean(y[high])) if high.any() else float("nan")
                    low_mean = float(np.nanmean(y[low])) if low.any() else float("nan")
                    delta = high_mean - low_mean if np.isfinite(high_mean) and np.isfinite(low_mean) else float("nan")
                    if np.isfinite(high_mean) and np.isfinite(low_mean) and low_mean > 1e-9:
                        lift = high_mean / low_mean
                    else:
                        lift = float("nan")

                    rows.append({
                        "scope": scope,
                        "target": target_name,
                        "feature": feature,
                        "lag_steps": int(lag),
                        "samples": int(valid.sum()),
                        "spearman_r": corr,
                        "scipy_p_value": scipy_p,
                        "permutation_p_value": perm_p,
                        "high_p90_response_mean": high_mean,
                        "low_p50_response_mean": low_mean,
                        "high_low_delta": delta,
                        "high_low_lift": lift,
                    })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["abs_spearman_r"] = summary["spearman_r"].abs()
    return summary.sort_values(
        ["permutation_p_value", "abs_spearman_r", "high_low_lift"],
        ascending=[True, False, False],
    )


def write_report(path, summary, feature_cols, max_lag_steps, top_n=20):
    top = summary[summary["target"] == "abs_increment"].head(top_n)
    if top.empty:
        top = summary.head(top_n)
    lines = [
        "# Rainfall Causal-Lag Screening Report",
        "",
        "This report screens lagged rainfall/hydro candidates for mechanism-MoE gating.",
        "It is a lagged association screen, not proof of pore-pressure causality or 3D movement direction.",
        "",
        f"- Features screened: {len(feature_cols)}",
        f"- Max acquisition lag steps: {int(max_lag_steps)}",
        "",
        "## Top Lagged Candidates",
        "",
        "| rank | scope | target | feature | lag_steps | rho | perm_p | lift | delta | samples |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            "| {rank} | {scope} | {target} | {feature} | {lag} | {rho:.3f} | {perm:.3f} | {lift:.3f} | {delta:.3f} | {samples} |".format(
                rank=rank,
                scope=row["scope"],
                target=row["target"],
                feature=row["feature"],
                lag=int(row["lag_steps"]),
                rho=float(row["spearman_r"]) if np.isfinite(row["spearman_r"]) else float("nan"),
                perm=float(row["permutation_p_value"]) if np.isfinite(row["permutation_p_value"]) else float("nan"),
                lift=float(row["high_low_lift"]) if np.isfinite(row["high_low_lift"]) else float("nan"),
                delta=float(row["high_low_delta"]) if np.isfinite(row["high_low_delta"]) else float("nan"),
                samples=int(row["samples"]),
            )
        )

    top_features = (
        top
        .groupby("feature")
        .size()
        .sort_values(ascending=False)
    )
    lines.extend([
        "",
        "## Candidate Gate Features",
        "",
    ])
    for feature, count in top_features.items():
        lines.append(f"- `{feature}` appears {int(count)} time(s) in the top {top_n} lagged rows.")
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "- Use these rows to choose candidate MoE gate features and lag windows.",
        "- Confirm later with prediction residuals once per-sample CPD validation predictions are persisted.",
        "- Keep mechanism language at rainfall/hydro state proxy level; do not call these node-level pore-pressure observations.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    susceptibility, active_nodes, top_nodes = load_susceptibility(args.rain_susceptibility, raw_seq.shape[1], args.top_quantile)
    del susceptibility

    feature_df, feature_cols = aligned_feature_frame(args.rainfall_csv, time_cols)
    responses = {
        "all_nodes": response_series(raw_seq, None),
        "rain_sensitive_nodes": response_series(raw_seq, active_nodes),
        "rain_top_nodes": response_series(raw_seq, top_nodes),
    }
    summary = compute_lag_rows(
        feature_df,
        feature_cols,
        responses,
        args.max_lag_steps,
        args.permutations,
        args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "causal_lag_summary.csv"
    report_md = output_dir / "causal_lag_report.md"
    feature_df.to_csv(output_dir / "aligned_rain_hydro_features.csv", index=False, encoding="utf-8")
    summary.to_csv(summary_csv, index=False, encoding="utf-8")
    write_report(report_md, summary, feature_cols, args.max_lag_steps)

    print(summary.head(20).to_string(index=False))
    print(f">> Saved causal lag summary: {summary_csv}")
    print(f">> Saved causal lag report: {report_md}")


if __name__ == "__main__":
    main()
