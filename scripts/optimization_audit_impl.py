import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize overnight optimization experiment outputs.")
    parser.add_argument("--output_dir", default="output/overnight_optimization_audit")
    return parser.parse_args()


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def add_summary(rows, method, scope, metrics_df, model_name, baseline_name="selected_model", note=""):
    if metrics_df is None or metrics_df.empty:
        return
    target = metrics_df[metrics_df["model"] == model_name]
    baseline = metrics_df[metrics_df["model"].isin([baseline_name, "selected"])]
    if target.empty or baseline.empty:
        return
    target_row = target.iloc[0]
    baseline_row = baseline.iloc[0]
    rows.append(
        {
            "method": method,
            "scope": scope,
            "seeds": "",
            "global_mae": float(target_row["global_mae"]),
            "rain_sensitive_mae": float(target_row["rain_sensitive_mae"]),
            "rain_top_mae": float(target_row["rain_top_mae"]),
            "rain_sensitive_delta_vs_selected": float(
                target_row["rain_sensitive_mae"] - baseline_row["rain_sensitive_mae"]
            ),
            "status": note,
        }
    )


def add_mean_summary(rows, method, scope, summary_df, model_name, baseline_name="selected_model", note=""):
    if summary_df is None or summary_df.empty:
        return
    target = summary_df[summary_df["model"] == model_name]
    baseline = summary_df[summary_df["model"] == baseline_name]
    if target.empty or baseline.empty:
        return
    target_row = target.iloc[0]
    baseline_row = baseline.iloc[0]
    rows.append(
        {
            "method": method,
            "scope": scope,
            "seeds": "0,7,23",
            "global_mae": float(target_row["global_mae_mean"]),
            "rain_sensitive_mae": float(target_row["rain_sensitive_mae_mean"]),
            "rain_top_mae": float(target_row["rain_top_mae_mean"]),
            "rain_sensitive_delta_vs_selected": float(
                target_row["rain_sensitive_mae_mean"] - baseline_row["rain_sensitive_mae_mean"]
            ),
            "status": note,
        }
    )


def source_switch_summary(rows):
    df = read_csv("output/forecast_gefs15_climfallback_flag_source_switch_bestpair_audit/source_switch_bestpair_summary.csv")
    if df is None or df.empty:
        return
    row = df.iloc[0]
    rows.append(
        {
            "method": "source_switch_bestpair",
            "scope": "posthoc source-aware MoE",
            "seeds": "0,7,23",
            "global_mae": float(row["global_mae_mean"]),
            "rain_sensitive_mae": float(row["rain_sensitive_mae_mean"]),
            "rain_top_mae": float(row["rain_top_mae_mean"]),
            "rain_sensitive_delta_vs_selected": np.nan,
            "status": "continue: only method below 0.8",
        }
    )


def analog_summaries(rows):
    add_mean_summary(
        rows,
        "analog_residual_fixed",
        "response_only k3 shrink0.25",
        read_csv("output/forecast_gefs15_climfallback_flag_analog_correct_audit/analog_residual_correction_summary.csv"),
        "analog_residual_corrected",
        note="continue as weak distillation signal",
    )
    add_mean_summary(
        rows,
        "analog_residual_auto",
        "LOO k/shrink",
        read_csv("output/forecast_gefs15_climfallback_flag_analog_correct_auto_audit/analog_residual_correction_summary.csv"),
        "analog_residual_corrected",
        note="continue as weak distillation signal",
    )
    add_mean_summary(
        rows,
        "analog_residual_gated_auto",
        "LOO k/shrink/distance",
        read_csv("output/forecast_gefs15_climfallback_flag_analog_correct_gated_auto_audit/analog_residual_correction_summary.csv"),
        "analog_residual_corrected",
        note="stop: no gain over auto",
    )


def single_seed_summaries(rows):
    add_summary(
        rows,
        "analog_fallback",
        "seed7 source-switch",
        read_csv("output/forecast_gefs15_analogfallback_source_switch_seed7/moe_metrics.csv"),
        "source_switch_posthoc_moe",
        note="stop: slightly worse than bestpair",
    )
    add_summary(
        rows,
        "displacement_projection",
        "seed7 selected",
        read_csv("output/forecast_gefs15_climfallback_flag_project_seed7_selected_rain_auto/displacement_projection_metrics.csv"),
        "displacement_projected",
        baseline_name="selected",
        note="stop: no change",
    )
    add_summary(
        rows,
        "expert_stacking",
        "seed7 rain_sensitive",
        read_csv("output/forecast_gefs15_climfallback_flag_stack_seed7_rain_s010/expert_stacking_metrics.csv"),
        "expert_stacked",
        note="stop: selected weight 1.0",
    )
    add_summary(
        rows,
        "quantile_mapping",
        "seed7 rain_sensitive",
        read_csv("output/forecast_gefs15_climfallback_flag_quantile_seed7_selected_rain_auto/quantile_mapping_metrics.csv"),
        "quantile_mapped",
        baseline_name="selected",
        note="stop: validation mapping does not transfer",
    )


def write_report(path, df):
    lines = [
        "# Overnight Optimization Audit",
        "",
        "This report summarizes existing experiment outputs and does not train or modify predictions.",
        "",
        "| method | scope | seeds | global | rain_sensitive | rain_top | delta_vs_selected | status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in df.itertuples(index=False):
        delta = "nan" if not np.isfinite(row.rain_sensitive_delta_vs_selected) else f"{row.rain_sensitive_delta_vs_selected:.3f}"
        lines.append(
            f"| {row.method} | {row.scope} | {row.seeds} | {row.global_mae:.3f} | "
            f"{row.rain_sensitive_mae:.3f} | {row.rain_top_mae:.3f} | {delta} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Continue source-aware posthoc MoE and distillation from its sample-level assignments.",
            "- Keep analog residual correction as a weak residual-direction soft label.",
            "- Stop global distribution calibration, global convex stacking, and simple displacement projection.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rows = []
    source_switch_summary(rows)
    analog_summaries(rows)
    single_seed_summaries(rows)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No optimization outputs were found to summarize.")
    df = df.sort_values(["rain_sensitive_mae", "global_mae"], ascending=True).reset_index(drop=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "overnight_optimization_audit.csv", index=False)
    write_report(output_dir / "overnight_optimization_audit.md", df)
    print(df.to_string(index=False))
    print(f">> Saved overnight optimization audit to: {output_dir}")


if __name__ == "__main__":
    main()
