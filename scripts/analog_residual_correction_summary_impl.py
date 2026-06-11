import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize analog residual correction metrics across seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 23])
    parser.add_argument(
        "--metrics_template",
        default=(
            "output/forecast_gefs15_climfallback_flag_analog_correct_seed{seed}"
            "_response_k3_s025/analog_residual_correction_metrics.csv"
        ),
    )
    parser.add_argument("--output_dir", default="output/forecast_gefs15_climfallback_flag_analog_correct_audit")
    return parser.parse_args()


def read_seed_metrics(seed, template):
    path = Path(template.format(seed=seed))
    if not path.exists():
        raise FileNotFoundError(f"Missing analog correction metrics for seed {seed}: {path}")
    df = pd.read_csv(path)
    df.insert(0, "seed", seed)
    df.insert(1, "metrics_path", str(path))
    return df


def write_report(path, summary_df, seed_df):
    lines = [
        "# Analog Residual Correction Seed Summary",
        "",
        "This report only summarizes existing analog-correct outputs; it does not retrain CPD-STGCN.",
        "",
        "## Summary",
        "",
        "| model | global_mean | rain_sensitive_mean | rain_top_mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.global_mae_mean:.3f} | "
            f"{row.rain_sensitive_mae_mean:.3f} | {row.rain_top_mae_mean:.3f} |"
        )
    lines.extend(["", "## Seed Metrics", ""])
    for row in seed_df.itertuples(index=False):
        lines.append(
            f"- seed {row.seed}, {row.model}: "
            f"{row.global_mae:.3f}/{row.rain_sensitive_mae:.3f}/{row.rain_top_mae:.3f}"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    seed_df = pd.concat([read_seed_metrics(seed, args.metrics_template) for seed in args.seeds], ignore_index=True)
    summary_df = (
        seed_df.groupby("model", as_index=False)
        .agg(
            global_mae_mean=("global_mae", "mean"),
            global_mae_std=("global_mae", "std"),
            global_mae_max=("global_mae", "max"),
            rain_sensitive_mae_mean=("rain_sensitive_mae", "mean"),
            rain_sensitive_mae_std=("rain_sensitive_mae", "std"),
            rain_sensitive_mae_max=("rain_sensitive_mae", "max"),
            rain_top_mae_mean=("rain_top_mae", "mean"),
            rain_top_mae_std=("rain_top_mae", "std"),
            rain_top_mae_max=("rain_top_mae", "max"),
        )
        .sort_values("rain_sensitive_mae_mean")
        .reset_index(drop=True)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(output_dir / "analog_residual_correction_seed_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "analog_residual_correction_summary.csv", index=False)
    write_report(output_dir / "analog_residual_correction_summary.md", summary_df, seed_df)
    print(summary_df.to_string(index=False))
    print(f">> Saved analog residual correction summary to: {output_dir}")


if __name__ == "__main__":
    main()
