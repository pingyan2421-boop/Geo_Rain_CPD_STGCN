import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = [
    "experiment",
    "fold",
    "anchor_role",
    "moe_candidate",
    "model_mae",
    "rain_sensitive_mae",
    "rain_top_mae",
    "p95_15d_mae",
    "p90_15d_mae",
    "persistence_mae",
    "rain_sensitive_mae_gain_pct",
    "rain_top_mae_gain_pct",
    "dynamic_rain_graph",
    "hydro_flow_gate",
    "temporal_encoder",
    "residual_scale",
    "selection_metric",
    "cpd_method",
    "source_csv",
]


PRIMARY_ANCHOR = "cpd_split_validation_binseg_cp180_rain_res015"
REFERENCE_EXPERIMENTS = {
    PRIMARY_ANCHOR: "primary_anchor",
    "fullflow_cp180_baseline_no_rain": "no_rain_reference",
    "fullflow_cp180_old_rain_dynamic": "old_dynamic_reference",
    "fullflow_cp180_hydro_film": "hydro_film_reference",
    "fullflow_cp180_tcn_attention": "tcn_attention_reference",
    "cpd_split_validation_binseg_cp180_rain_res020": "residual_scale_020_ablation",
    "cpd_split_validation_binseg_cp180_rain_dynamic": "dynamic_graph_reference",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize CPD split validation outputs as mechanism-MoE anchor experiments."
    )
    parser.add_argument("--output_root", default="output")
    parser.add_argument("--output_dir", default=os.path.join("output", "mechanism_moe_anchor"))
    parser.add_argument("--top_k", type=int, default=12)
    return parser.parse_args()


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def discover_result_csvs(output_root):
    root = Path(output_root)
    if not root.exists():
        return []
    return sorted(root.glob("**/cpd_split_results.csv"))


def normalize_results(csv_path):
    df = pd.read_csv(csv_path)
    experiment = csv_path.parent.name
    df["experiment"] = experiment
    df["source_csv"] = str(csv_path)
    df["anchor_role"] = REFERENCE_EXPERIMENTS.get(experiment, "comparison")

    if "hydro_flow_gate" not in df.columns:
        df["hydro_flow_gate"] = np.nan
    if "temporal_encoder" not in df.columns:
        df["temporal_encoder"] = "legacy"
    if "p95_15d_mae" not in df.columns:
        df["p95_15d_mae"] = np.nan
    if "p90_15d_mae" not in df.columns:
        df["p90_15d_mae"] = np.nan

    rain_mae = df["rain_sensitive_mae"].apply(as_float) if "rain_sensitive_mae" in df.columns else np.nan
    df["moe_candidate"] = (
        (df.get("fold", "") == "cp_180")
        & np.isfinite(rain_mae)
        & (rain_mae > 0)
    ).astype(int)

    for col in KEY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[KEY_COLUMNS]


def rank_results(df):
    ranked = df.copy()
    ranked["rank_metric"] = ranked["rain_sensitive_mae"].apply(as_float)
    fallback = ranked["model_mae"].apply(as_float)
    ranked["rank_metric"] = ranked["rank_metric"].where(np.isfinite(ranked["rank_metric"]), fallback + 100.0)
    ranked = ranked.sort_values(
        ["moe_candidate", "rank_metric", "model_mae"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked.drop(columns=["rank_metric"])


def fmt(value):
    value = as_float(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def write_markdown(path, ranked, top_k):
    top = ranked.head(top_k)
    primary = ranked[ranked["anchor_role"] == "primary_anchor"]
    if primary.empty:
        primary_text = "Primary anchor not found in current output directory."
    else:
        row = primary.iloc[0]
        primary_text = (
            f"`{row['experiment']}` / `{row['fold']}`: "
            f"global MAE={fmt(row['model_mae'])}, "
            f"rain_sensitive_mae={fmt(row['rain_sensitive_mae'])}, "
            f"rain_top_mae={fmt(row['rain_top_mae'])}."
        )

    lines = [
        "# Mechanism MoE Anchor Experiment Summary",
        "",
        "This report is generated from existing `output/**/cpd_split_results.csv` files.",
        "It does not rerun training and should be used as the first fixed reference before causal-lag and MoE work.",
        "",
        "## Primary Anchor",
        "",
        primary_text,
        "",
        "## Ranked Candidate Experiments",
        "",
        "| rank | experiment | fold | role | MoE | model_mae | rain_sensitive_mae | rain_top_mae | p95_15d_mae | encoder | dynamic |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| {rank} | {experiment} | {fold} | {role} | {moe} | {model} | {rain} | {top_mae} | {p95} | {encoder} | {dynamic} |".format(
                rank=int(row["rank"]),
                experiment=row["experiment"],
                fold=row["fold"],
                role=row["anchor_role"],
                moe=int(row["moe_candidate"]),
                model=fmt(row["model_mae"]),
                rain=fmt(row["rain_sensitive_mae"]),
                top_mae=fmt(row["rain_top_mae"]),
                p95=fmt(row["p95_15d_mae"]),
                encoder=row["temporal_encoder"],
                dynamic=row["dynamic_rain_graph"],
            )
        )

    lines.extend([
        "",
        "## Interpretation Rules",
        "",
        "- `primary_anchor` is the current fixed reference for later mechanism-MoE work.",
        "- `moe_candidate=1` means the row has a finite `cp_180` rain-sensitive metric and can enter the first expert pool.",
        "- Lower `rain_sensitive_mae` is the main prediction target, but later stages must still explain rainfall lag, stage state, and rain-sensitive node behavior.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    csvs = discover_result_csvs(args.output_root)
    if not csvs:
        raise SystemExit(f"No cpd_split_results.csv files found under {args.output_root}")

    frames = [normalize_results(path) for path in csvs]
    ranked = rank_results(pd.concat(frames, ignore_index=True))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "anchor_experiment_summary.csv"
    md_path = output_dir / "anchor_experiment_summary.md"
    ranked.to_csv(csv_path, index=False, encoding="utf-8")
    write_markdown(md_path, ranked, args.top_k)

    print(ranked.head(args.top_k).to_string(index=False))
    print(f">> Saved anchor summary CSV: {csv_path}")
    print(f">> Saved anchor summary report: {md_path}")


if __name__ == "__main__":
    main()
