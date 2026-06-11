import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae
from scripts.source_switch_posthoc_moe_impl import source_switch_mask


def parse_seeds(value):
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise argparse.ArgumentTypeError("Expected at least one seed.")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search source-aware post-hoc MoE pairs over persisted posthoc predictions."
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,7,23"))
    parser.add_argument(
        "--prediction_dir_template",
        default="output/forecast_gefs15_climfallback_flag_cp180_seed{seed}/cp_180/predictions",
    )
    parser.add_argument(
        "--posthoc_grid_dir",
        default="output/forecast_gefs15_climfallback_flag_posthoc_grid",
    )
    parser.add_argument(
        "--forecast_context_by_sample",
        default="output/forecast_context_diagnostics_cp180_climfallback_flag/forecast_context_by_sample.csv",
    )
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--switch_col", default="forecast_is_fallback")
    parser.add_argument("--output_dir", default="output/forecast_gefs15_climfallback_flag_source_switch_grid")
    parser.add_argument("--top_n", type=int, default=20)
    return parser.parse_args()


def config_dirs(posthoc_grid_dir, seed):
    root = Path(posthoc_grid_dir)
    prefix = f"seed{seed}_"
    dirs = {}
    for path in sorted(root.glob(f"{prefix}*")):
        if path.is_dir() and (path / "moe_test_pred_real.npy").exists():
            dirs[path.name[len(prefix) :]] = path
    if not dirs:
        raise FileNotFoundError(f"No posthoc prediction directories found under {root} for seed={seed}.")
    return dirs


def load_seed_pack(args, seed):
    pred_dir = artifacts.validate_predictions_dir(args.prediction_dir_template.format(seed=seed))
    y_true = artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL)
    indices = artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32)
    switch = source_switch_mask(args.forecast_context_by_sample, indices, args.switch_col)
    return {
        "seed": seed,
        "y_true": y_true,
        "switch": switch,
        "n_nodes": y_true.shape[2],
        "switched_samples": int(switch.sum()),
        "total_samples": int(len(switch)),
    }


def load_posthoc_preds(seed_dirs):
    preds = {}
    for config_name, path in seed_dirs.items():
        preds[config_name] = np.load(path / "moe_test_pred_real.npy")
    return preds


def evaluate_pair(seed_pack, preds, primary_config, fallback_config, rain_mask, rain_top_mask):
    primary_pred = preds[primary_config]
    fallback_pred = preds[fallback_config]
    if primary_pred.shape != fallback_pred.shape:
        raise ValueError(
            f"Posthoc prediction shape mismatch for {primary_config} and {fallback_config}: "
            f"{primary_pred.shape} vs {fallback_pred.shape}"
        )
    switched = primary_pred.copy()
    switched[seed_pack["switch"]] = fallback_pred[seed_pack["switch"]]
    y_true = seed_pack["y_true"]
    return {
        "seed": seed_pack["seed"],
        "primary_config": primary_config,
        "fallback_config": fallback_config,
        "switched_samples": seed_pack["switched_samples"],
        "total_samples": seed_pack["total_samples"],
        "global_mae": mae(y_true, switched),
        "rain_sensitive_mae": mae(y_true, switched, rain_mask),
        "rain_top_mae": mae(y_true, switched, rain_top_mask),
    }


def markdown_table(df):
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.3f}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path, summary_df, seed_df, args):
    top = summary_df.head(args.top_n).copy()
    best = top.iloc[0]
    best_seed = seed_df[
        (seed_df["primary_config"] == best["primary_config"])
        & (seed_df["fallback_config"] == best["fallback_config"])
    ].copy()
    lines = [
        "# Source-Switch Pair Grid",
        "",
        "This report exhaustively combines primary and fallback post-hoc MoE configs using forecast source metadata.",
        "",
        "## Configuration",
        "",
        f"- seeds: `{','.join(str(seed) for seed in args.seeds)}`",
        f"- posthoc_grid_dir: `{args.posthoc_grid_dir}`",
        f"- switch_col: `{args.switch_col}`",
        "",
        "## Top Pairs",
        "",
        markdown_table(
            top[
                [
                    "primary_config",
                    "fallback_config",
                    "global_mae",
                    "rain_sensitive_mae",
                    "rain_top_mae",
                ]
            ]
        ),
        "",
        "## Best Pair Seed Metrics",
        "",
        markdown_table(
            best_seed[
                [
                    "seed",
                    "switched_samples",
                    "total_samples",
                    "global_mae",
                    "rain_sensitive_mae",
                    "rain_top_mae",
                ]
            ]
        ),
        "",
        "## Boundary",
        "",
        "This is a post-hoc search over persisted predictions. It does not retrain CPD-STGCN.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    seed_dirs = {seed: config_dirs(args.posthoc_grid_dir, seed) for seed in args.seeds}
    common_configs = sorted(set.intersection(*(set(dirs) for dirs in seed_dirs.values())))
    if not common_configs:
        raise ValueError("No common posthoc configs are available across all requested seeds.")

    seed_packs = [load_seed_pack(args, seed) for seed in args.seeds]
    n_nodes = seed_packs[0]["n_nodes"]
    for pack in seed_packs:
        if pack["n_nodes"] != n_nodes:
            raise ValueError(f"Seed {pack['seed']} has n_nodes={pack['n_nodes']}, expected {n_nodes}.")
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, n_nodes)
    seed_preds = {seed: load_posthoc_preds(seed_dirs[seed]) for seed in args.seeds}

    seed_rows = []
    for primary_config in common_configs:
        for fallback_config in common_configs:
            for pack in seed_packs:
                seed_rows.append(
                    evaluate_pair(
                        pack,
                        seed_preds[pack["seed"]],
                        primary_config,
                        fallback_config,
                        rain_mask,
                        rain_top_mask,
                    )
                )

    seed_df = pd.DataFrame(seed_rows)
    summary_df = (
        seed_df.groupby(["primary_config", "fallback_config"], as_index=False)[
            ["global_mae", "rain_sensitive_mae", "rain_top_mae"]
        ]
        .mean()
        .sort_values(["rain_sensitive_mae", "rain_top_mae", "global_mae"], ascending=True)
        .reset_index(drop=True)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "source_switch_pair_grid.csv", index=False)
    seed_df.to_csv(output_dir / "source_switch_pair_seed_metrics.csv", index=False)
    write_report(output_dir / "source_switch_grid_report.md", summary_df, seed_df, args)

    print(summary_df.head(args.top_n).to_string(index=False))
    print(f">> Saved source-switch pair grid outputs to: {output_dir}")


if __name__ == "__main__":
    main()
