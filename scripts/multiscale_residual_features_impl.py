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

from data_loader.date_loader import detect_change_points_professional, load_and_clean_data  # noqa: E402
from data_loader.rainfall_loader import align_rainfall_to_insar  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build multi-scale response features for mechanism-MoE gating."
    )
    parser.add_argument("--file_path", default=os.path.join(project_root, "dataset", "inter228_5241.csv"))
    parser.add_argument("--rainfall_csv", default=os.path.join(project_root, "dataset", "rainfall", "chirps_daily.csv"))
    parser.add_argument(
        "--rain_susceptibility",
        default=os.path.join(project_root, "output", "rain_susceptibility", "rain_susceptibility.csv"),
    )
    parser.add_argument(
        "--causal_lag_summary",
        default=os.path.join(project_root, "output", "mechanism_moe_causal_lag", "causal_lag_summary.csv"),
    )
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "mechanism_moe_multiscale"))
    parser.add_argument("--top_quantile", type=float, default=0.60)
    parser.add_argument("--trend_window", type=int, default=6)
    parser.add_argument("--short_window", type=int, default=2)
    parser.add_argument("--near_cpd_window", type=int, default=3)
    parser.add_argument("--max_gate_features", type=int, default=8)
    parser.add_argument("--cpd_method", default="binseg")
    parser.add_argument("--cpd_penalty", type=float, default=10)
    parser.add_argument("--cpd_mode", default="mix")
    parser.add_argument("--cpd_top_ratio", type=float, default=0.10)
    parser.add_argument("--cpd_min_size", type=int, default=10)
    parser.add_argument("--cpd_cost", default="l2")
    return parser.parse_args()


def parse_insar_date(value):
    return pd.to_datetime(str(value).split(".")[0]).date()


def load_susceptibility(path, n_nodes, top_quantile):
    if not path or not os.path.exists(path):
        susceptibility = np.zeros(n_nodes, dtype=np.float32)
    else:
        df = pd.read_csv(path)
        susceptibility = df["rain_susceptibility"].to_numpy(dtype=np.float32)
        if susceptibility.shape[0] != n_nodes:
            raise ValueError(f"rain_susceptibility length={susceptibility.shape[0]} does not match n_nodes={n_nodes}.")
    active = susceptibility > 0
    top = np.zeros_like(active, dtype=bool)
    if active.any():
        top = susceptibility >= float(np.quantile(susceptibility[active], float(top_quantile)))
    return {"all_nodes": None, "rain_sensitive_nodes": active, "rain_top_nodes": top}


def trailing_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - int(window) + 1)
        out[idx] = float(np.nanmean(values[start:idx + 1]))
    return out


def robust_z(values):
    values = np.asarray(values, dtype=np.float64)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 1e-9 else float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return (values - median) / scale


def build_stage_features(n_intervals, cps, near_window):
    cps = sorted(int(cp) for cp in cps if 0 < int(cp) < n_intervals + 1)
    cp_array = np.array(cps, dtype=np.int32)
    rows = []
    for interval_idx in range(n_intervals):
        acquisition_idx = interval_idx + 1
        prev_cps = cp_array[cp_array <= acquisition_idx]
        next_cps = cp_array[cp_array > acquisition_idx]
        prev_cp = int(prev_cps.max()) if prev_cps.size else 0
        next_cp = int(next_cps.min()) if next_cps.size else n_intervals + 1
        distance = min(abs(acquisition_idx - prev_cp), abs(next_cp - acquisition_idx))
        rows.append({
            "stage_id": int(np.searchsorted(cp_array, acquisition_idx, side="right")),
            "since_last_cp": int(acquisition_idx - prev_cp),
            "until_next_cp": int(next_cp - acquisition_idx),
            "cp_distance": int(distance),
            "near_cpd": int(distance <= int(near_window)),
        })
    return pd.DataFrame(rows)


def response_components(raw_seq, node_mask, trend_window, short_window):
    increments = np.diff(raw_seq, axis=0)
    if node_mask is not None:
        increments = increments[:, node_mask]
    signed = np.mean(increments, axis=1).astype(np.float64)
    magnitude = np.mean(np.abs(increments), axis=1).astype(np.float64)
    trend = trailing_mean(magnitude, trend_window)
    short_baseline = trailing_mean(magnitude, short_window)
    disturbance = magnitude - trend
    shock = magnitude - short_baseline
    return pd.DataFrame({
        "signed_increment": signed,
        "abs_increment": magnitude,
        "long_trend": trend,
        "short_baseline": short_baseline,
        "rain_disturbance": disturbance,
        "rain_disturbance_z": robust_z(disturbance),
        "short_shock": shock,
        "short_shock_z": robust_z(shock),
    })


def select_gate_specs(causal_lag_summary, max_gate_features):
    if not causal_lag_summary or not os.path.exists(causal_lag_summary):
        return [
            ("hydro_dry_to_wet_transition", 6),
            ("rain_event_p90_15d", 6),
            ("rain_30d", 6),
            ("hydro_memory_index", 6),
        ]
    df = pd.read_csv(causal_lag_summary)
    df = df[df["target"] == "abs_increment"].copy()
    if df.empty:
        return []
    df["abs_spearman_r"] = df["spearman_r"].abs()
    df = df.sort_values(["permutation_p_value", "abs_spearman_r"], ascending=[True, False])
    specs = []
    seen = set()
    for row in df.itertuples(index=False):
        key = (str(row.feature), int(row.lag_steps))
        feature_seen = str(row.feature)
        if feature_seen in seen:
            continue
        specs.append(key)
        seen.add(feature_seen)
        if len(specs) >= int(max_gate_features):
            break
    return specs


def add_gate_features(frame, aligned_rain, gate_specs):
    feature_cols = [c for c in aligned_rain.columns if c.startswith("rain_") or c.startswith("hydro_")]
    rain_values = aligned_rain[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    n_intervals = len(frame)
    for feature, lag in gate_specs:
        if feature not in rain_values.columns:
            continue
        values = []
        for interval_idx in range(n_intervals):
            acquisition_idx = interval_idx + 1
            source_idx = acquisition_idx - int(lag)
            values.append(float(rain_values.iloc[source_idx][feature]) if source_idx >= 0 else 0.0)
        frame[f"gate_{feature}_lag{int(lag)}"] = values
    return frame


def safe_corr(x, y):
    if np.nanstd(x) < 1e-9 or np.nanstd(y) < 1e-9:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(x, y)
        return float(corr), float(p_value)
    except Exception:
        return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]), float("nan")


def summarize_gate_links(features):
    gate_cols = [c for c in features.columns if c.startswith("gate_")]
    component_cols = ["abs_increment", "long_trend", "rain_disturbance", "rain_disturbance_z", "short_shock_z"]
    rows = []
    for scope, group in features.groupby("scope"):
        for component in component_cols:
            for gate in gate_cols:
                corr, p_value = safe_corr(group[gate].to_numpy(dtype=np.float64), group[component].to_numpy(dtype=np.float64))
                high = group[gate] >= group[gate].quantile(0.90)
                low = group[gate] <= group[gate].quantile(0.50)
                high_mean = float(group.loc[high, component].mean()) if high.any() else float("nan")
                low_mean = float(group.loc[low, component].mean()) if low.any() else float("nan")
                rows.append({
                    "scope": scope,
                    "component": component,
                    "gate_feature": gate,
                    "spearman_r": corr,
                    "p_value": p_value,
                    "high_p90_component_mean": high_mean,
                    "low_p50_component_mean": low_mean,
                    "high_low_delta": high_mean - low_mean if np.isfinite(high_mean) and np.isfinite(low_mean) else float("nan"),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_spearman_r"] = out["spearman_r"].abs()
    return out.sort_values(["abs_spearman_r", "high_low_delta"], ascending=[False, False])


def write_report(path, feature_df, summary_df, gate_specs, cps):
    top_summary = summary_df.head(20)
    lines = [
        "# Multi-Scale Response Feature Report",
        "",
        "This report builds local mechanism features for later post-hoc MoE gating.",
        "It decomposes observed InSAR response into trend, stage-transition, and short disturbance terms.",
        "",
        f"- Intervals: {feature_df['interval_index'].nunique()}",
        f"- Scopes: {', '.join(sorted(feature_df['scope'].unique()))}",
        f"- CPD boundaries: {', '.join(map(str, cps))}",
        f"- Gate specs: {', '.join([f'{f}@lag{l}' for f, l in gate_specs])}",
        "",
        "## Strongest Gate-Component Links",
        "",
        "| rank | scope | component | gate_feature | rho | delta |",
        "| ---: | --- | --- | --- | ---: | ---: |",
    ]
    for rank, (_, row) in enumerate(top_summary.iterrows(), start=1):
        lines.append(
            "| {rank} | {scope} | {component} | {gate} | {rho:.3f} | {delta:.3f} |".format(
                rank=rank,
                scope=row["scope"],
                component=row["component"],
                gate=row["gate_feature"],
                rho=float(row["spearman_r"]) if np.isfinite(row["spearman_r"]) else float("nan"),
                delta=float(row["high_low_delta"]) if np.isfinite(row["high_low_delta"]) else float("nan"),
            )
        )
    lines.extend([
        "",
        "## MoE Use",
        "",
        "- `long_trend` is the slow deformation state candidate.",
        "- `near_cpd`, `stage_id`, `since_last_cp`, and `until_next_cp` are stage-transition candidates.",
        "- `rain_disturbance_z` and `short_shock_z` are short-term disturbance candidates.",
        "- `gate_*` columns carry the lagged rainfall/hydro candidates selected by Step 2.",
        "",
        "## Boundary",
        "",
        "These are observed response features, not model prediction residuals yet. Later MoE work should reuse the same decomposition on persisted per-sample prediction residuals when available.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    _, cps = detect_change_points_professional(
        raw_seq,
        penalty=args.cpd_penalty,
        mode=args.cpd_mode,
        top_ratio=args.cpd_top_ratio,
        min_size=args.cpd_min_size,
        method=args.cpd_method,
        model=args.cpd_cost,
    )
    scope_masks = load_susceptibility(args.rain_susceptibility, raw_seq.shape[1], args.top_quantile)
    aligned_rain = align_rainfall_to_insar(args.rainfall_csv, time_cols)
    gate_specs = select_gate_specs(args.causal_lag_summary, args.max_gate_features)
    stage_df = build_stage_features(raw_seq.shape[0] - 1, cps, args.near_cpd_window)

    frames = []
    dates = [parse_insar_date(d) for d in time_cols]
    for scope, mask in scope_masks.items():
        components = response_components(raw_seq, mask, args.trend_window, args.short_window)
        frame = pd.DataFrame({
            "scope": scope,
            "interval_index": np.arange(raw_seq.shape[0] - 1, dtype=np.int32),
            "start_date": [d.isoformat() for d in dates[:-1]],
            "end_date": [d.isoformat() for d in dates[1:]],
        })
        frame = pd.concat([frame, stage_df.reset_index(drop=True), components.reset_index(drop=True)], axis=1)
        frame = add_gate_features(frame, aligned_rain, gate_specs)
        frames.append(frame)

    feature_df = pd.concat(frames, ignore_index=True)
    summary_df = summarize_gate_links(feature_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "multiscale_response_features.csv"
    summary_path = output_dir / "gate_component_summary.csv"
    report_path = output_dir / "multiscale_response_report.md"
    feature_df.to_csv(feature_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    write_report(report_path, feature_df, summary_df, gate_specs, cps)

    print(summary_df.head(20).to_string(index=False))
    print(f">> Saved multiscale response features: {feature_path}")
    print(f">> Saved gate-component summary: {summary_path}")
    print(f">> Saved multiscale report: {report_path}")


if __name__ == "__main__":
    main()
