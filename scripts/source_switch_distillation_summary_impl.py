import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize source-switch distillation target exports.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 7, 23])
    parser.add_argument(
        "--distill_dir_template",
        default="output/forecast_gefs15_climfallback_flag_distill_seed{seed}",
    )
    parser.add_argument("--output_dir", default="output/forecast_gefs15_climfallback_flag_distill_audit")
    return parser.parse_args()


def read_seed(seed, template):
    root = Path(template.format(seed=seed))
    metrics_path = root / "distillation_metrics.csv"
    sample_path = root / "distillation_sample_metrics.csv"
    if not metrics_path.exists() or not sample_path.exists():
        raise FileNotFoundError(f"Missing distillation outputs for seed {seed}: {root}")
    metrics = pd.read_csv(metrics_path)
    samples = pd.read_csv(sample_path)
    metrics.insert(0, "seed", seed)
    samples.insert(0, "seed", seed)
    return metrics, samples


def write_report(path, summary, sample_summary):
    lines = [
        "# Source-Switch Distillation Summary",
        "",
        "This report summarizes exported teacher targets; it does not train CPD-STGCN.",
        "",
        "## Metric Summary",
        "",
        "| model | global | rain_sensitive | rain_top |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.global_mae:.3f} | {row.rain_sensitive_mae:.3f} | {row.rain_top_mae:.3f} |"
        )
    lines.extend(["", "## Sample Gain", ""])
    for row in sample_summary.itertuples(index=False):
        lines.append(
            f"- seed {row.seed}: positive_gain={row.positive_gain_samples}/{row.samples}, "
            f"mean_rain_sensitive_gain={row.mean_rain_sensitive_gain:.3f}"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    metric_frames = []
    sample_frames = []
    for seed in args.seeds:
        metrics, samples = read_seed(seed, args.distill_dir_template)
        metric_frames.append(metrics)
        sample_frames.append(samples)
    metrics_df = pd.concat(metric_frames, ignore_index=True)
    samples_df = pd.concat(sample_frames, ignore_index=True)
    summary = (
        metrics_df.groupby("model", as_index=False)[["global_mae", "rain_sensitive_mae", "rain_top_mae"]]
        .mean()
        .sort_values("rain_sensitive_mae")
        .reset_index(drop=True)
    )
    sample_summary = (
        samples_df.assign(positive_gain=samples_df["teacher_gain_rain_sensitive_mae"] > 0.0)
        .groupby("seed", as_index=False)
        .agg(
            samples=("sample_index", "count"),
            positive_gain_samples=("positive_gain", "sum"),
            mean_rain_sensitive_gain=("teacher_gain_rain_sensitive_mae", "mean"),
            mean_global_gain=("teacher_gain_global_mae", "mean"),
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "distillation_seed_metrics.csv", index=False)
    samples_df.to_csv(output_dir / "distillation_sample_metrics_all.csv", index=False)
    summary.to_csv(output_dir / "distillation_summary.csv", index=False)
    sample_summary.to_csv(output_dir / "distillation_sample_summary.csv", index=False)
    write_report(output_dir / "source_switch_distillation_summary.md", summary, sample_summary)
    print(summary.to_string(index=False))
    print(sample_summary.to_string(index=False))
    print(f">> Saved distillation summary to: {output_dir}")


if __name__ == "__main__":
    main()
