import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize post-hoc mechanism MoE grid outputs.")
    parser.add_argument(
        "--search_root",
        default=os.path.join(project_root, "output"),
        help="Directory containing post-hoc MoE output folders.",
    )
    parser.add_argument(
        "--pattern",
        default="mechanism_moe_posthoc*/moe_metrics.csv",
        help="Glob pattern under --search_root for moe_metrics.csv files.",
    )
    parser.add_argument(
        "--anchor_results",
        default=os.path.join(project_root, "output", "cpd_split_validation_binseg_cp180_rain_res015", "cpd_split_results.csv"),
    )
    parser.add_argument(
        "--artifact_results",
        default=os.path.join(project_root, "output", "mechanism_moe_prediction_artifact_full", "cpd_split_results.csv"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(project_root, "output", "mechanism_moe_posthoc_audit"),
    )
    return parser.parse_args()


def read_anchor(path, label):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "label": label,
        "global_mae": float(row.get("model_mae", float("nan"))),
        "rain_sensitive_mae": float(row.get("rain_sensitive_mae", float("nan"))),
        "rain_top_mae": float(row.get("rain_top_mae", float("nan"))),
    }


def collect_moe_rows(search_root, pattern):
    rows = []
    root = Path(search_root)
    for metrics_path in root.glob(pattern):
        df = pd.read_csv(metrics_path)
        moe = df[df["model"] == "posthoc_mechanism_moe"]
        if moe.empty:
            continue
        row = moe.iloc[0]
        rows.append({
            "experiment": metrics_path.parent.name,
            "global_mae": float(row["global_mae"]),
            "rain_sensitive_mae": float(row["rain_sensitive_mae"]),
            "rain_top_mae": float(row["rain_top_mae"]),
            "metrics_path": str(metrics_path),
        })
    return pd.DataFrame(rows)


def seed_stability(grid):
    if grid.empty:
        return pd.DataFrame()
    rows = []
    patterns = [
        re.compile(r"mechanism_moe_posthoc_anchorlike_seed(\d+)_local_k6_t060"),
        re.compile(r"refactor_validation_posthoc_seed(\d+)_local_k6_t060"),
    ]
    for row in grid.itertuples(index=False):
        match = next((pat.fullmatch(str(row.experiment)) for pat in patterns if pat.fullmatch(str(row.experiment))), None)
        if not match:
            continue
        rows.append({
            "seed": int(match.group(1)),
            "global_mae": float(row.global_mae),
            "rain_sensitive_mae": float(row.rain_sensitive_mae),
            "rain_top_mae": float(row.rain_top_mae),
            "experiment": str(row.experiment),
        })
    return pd.DataFrame(rows).sort_values("seed") if rows else pd.DataFrame()


def feature_group_stability(grid):
    if grid.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    pattern = re.compile(r"mechanism_moe_posthoc_anchorlike_seed(\d+)_local_k6_t060_([A-Za-z0-9_]+)")
    for row in grid.itertuples(index=False):
        match = pattern.fullmatch(str(row.experiment))
        if not match:
            continue
        rows.append({
            "seed": int(match.group(1)),
            "feature_group": match.group(2),
            "global_mae": float(row.global_mae),
            "rain_sensitive_mae": float(row.rain_sensitive_mae),
            "rain_top_mae": float(row.rain_top_mae),
            "experiment": str(row.experiment),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    detail = detail.sort_values(["feature_group", "seed"])
    summary = detail.groupby("feature_group", as_index=False).agg(
        global_mae_mean=("global_mae", "mean"),
        global_mae_std=("global_mae", "std"),
        rain_sensitive_mae_mean=("rain_sensitive_mae", "mean"),
        rain_sensitive_mae_std=("rain_sensitive_mae", "std"),
        rain_top_mae_mean=("rain_top_mae", "mean"),
        rain_top_mae_std=("rain_top_mae", "std"),
        n=("seed", "count"),
    )
    return detail, summary.sort_values(["rain_sensitive_mae_mean", "global_mae_mean"])


def metric_summary(frame):
    if frame.empty:
        return pd.DataFrame()
    metrics = ["global_mae", "rain_sensitive_mae", "rain_top_mae"]
    rows = []
    for metric in metrics:
        rows.append({
            "metric": metric,
            "mean": float(frame[metric].mean()),
            "std": float(frame[metric].std(ddof=1)) if len(frame) > 1 else 0.0,
            "min": float(frame[metric].min()),
            "max": float(frame[metric].max()),
            "n": int(len(frame)),
        })
    return pd.DataFrame(rows)


def write_report(path, grid, anchor, artifact, seed_frame, seed_summary, feature_summary):
    lines = [
        "# 后验机制 MoE 审计报告",
        "",
        "本报告汇总本地后验机制 MoE 网格输出，并与当前 cp_180 锚点结果对比。",
        "",
        "## 基线",
        "",
        "| 标签 | global_mae | rain_sensitive_mae | rain_top_mae |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in [anchor, artifact]:
        if item is None:
            continue
        lines.append(
            f"| {item['label']} | {item['global_mae']:.3f} | {item['rain_sensitive_mae']:.3f} | {item['rain_top_mae']:.3f} |"
        )

    lines.extend([
        "",
        "## 最优 MoE 结果",
        "",
        "| 排名 | 实验 | global_mae | rain_sensitive_mae | rain_top_mae |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    if not grid.empty:
        ranked = grid.sort_values(["rain_sensitive_mae", "global_mae"]).head(10)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"| {rank} | {row['experiment']} | {row['global_mae']:.3f} | {row['rain_sensitive_mae']:.3f} | {row['rain_top_mae']:.3f} |"
            )
    lines.extend([
        "",
        "## Seed 稳定性",
        "",
    ])
    if seed_summary.empty:
        lines.append("未发现 seed 稳定性结果。")
    else:
        lines.extend([
            "| metric | mean | std | min | max | n |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in seed_summary.itertuples(index=False):
            lines.append(f"| {row.metric} | {row.mean:.3f} | {row.std:.3f} | {row.min:.3f} | {row.max:.3f} | {row.n} |")
        lines.extend([
            "",
            "| seed | global_mae | rain_sensitive_mae | rain_top_mae |",
            "| ---: | ---: | ---: | ---: |",
        ])
        for row in seed_frame.itertuples(index=False):
            lines.append(f"| {row.seed} | {row.global_mae:.3f} | {row.rain_sensitive_mae:.3f} | {row.rain_top_mae:.3f} |")
    lines.extend([
        "",
        "## 特征组消融",
        "",
    ])
    if feature_summary.empty:
        lines.append("未发现特征组消融结果。")
    else:
        lines.extend([
            "| feature_group | global_mean | global_std | rain_sensitive_mean | rain_sensitive_std | rain_top_mean | rain_top_std | n |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in feature_summary.itertuples(index=False):
            lines.append(
                f"| {row.feature_group} | {row.global_mae_mean:.3f} | {row.global_mae_std:.3f} | "
                f"{row.rain_sensitive_mae_mean:.3f} | {row.rain_sensitive_mae_std:.3f} | "
                f"{row.rain_top_mae_mean:.3f} | {row.rain_top_mae_std:.3f} | {row.n} |"
            )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 这些输出是后验融合结果，不改变 CPD-STGCN 训练图。",
        "- MoE 可以改善较弱的完整 prediction artifact，但仍可能在全局 MAE 上弱于旧 cp_180 锚点。",
        "- 正式结论需要由同等锚点质量的训练结果保存 prediction artifact，并至少完成 3 个 seed 验证。",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = collect_moe_rows(args.search_root, args.pattern)
    if not grid.empty:
        grid = grid.sort_values(["rain_sensitive_mae", "global_mae"])
    grid_path = output_dir / "posthoc_moe_grid_summary.csv"
    grid.to_csv(grid_path, index=False, encoding="utf-8")
    seed_frame = seed_stability(grid)
    seed_summary = metric_summary(seed_frame)
    feature_detail, feature_summary = feature_group_stability(grid)
    seed_path = output_dir / "posthoc_moe_seed_stability.csv"
    seed_summary_path = output_dir / "posthoc_moe_seed_stability_summary.csv"
    feature_path = output_dir / "posthoc_moe_feature_group_stability.csv"
    feature_summary_path = output_dir / "posthoc_moe_feature_group_summary.csv"
    seed_frame.to_csv(seed_path, index=False, encoding="utf-8")
    seed_summary.to_csv(seed_summary_path, index=False, encoding="utf-8")
    feature_detail.to_csv(feature_path, index=False, encoding="utf-8")
    feature_summary.to_csv(feature_summary_path, index=False, encoding="utf-8")

    anchor = read_anchor(args.anchor_results, "cp_180_res015_anchor")
    artifact = read_anchor(args.artifact_results, "full_prediction_artifact_selected_model")
    report_path = output_dir / "posthoc_moe_audit_report.md"
    write_report(report_path, grid, anchor, artifact, seed_frame, seed_summary, feature_summary)

    if grid.empty:
        print("No post-hoc MoE metrics found.")
    else:
        print(grid.head(10).to_string(index=False))
    print(f">> Saved post-hoc MoE grid summary: {grid_path}")
    print(f">> Saved post-hoc MoE seed stability: {seed_path}")
    print(f">> Saved post-hoc MoE seed stability summary: {seed_summary_path}")
    print(f">> Saved post-hoc MoE feature group stability: {feature_path}")
    print(f">> Saved post-hoc MoE feature group summary: {feature_summary_path}")
    print(f">> Saved post-hoc MoE audit report: {report_path}")


if __name__ == "__main__":
    main()
