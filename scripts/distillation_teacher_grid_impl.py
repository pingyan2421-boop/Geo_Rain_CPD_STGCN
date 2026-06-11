import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.posthoc_mechanism_moe_impl import (
    FEATURE_GROUPS,
    build_gates,
    combine_experts,
    expert_stack,
    forecast_gate_features,
    load_node_masks,
    load_prediction_pack,
    mae,
    sample_expert_mae,
    sample_features,
)


def parse_csv_ints(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search posthoc train teachers without reading test labels."
    )
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("0,7,23"))
    parser.add_argument(
        "--prediction_dir_template",
        default="output/forecast_gefs15_climfallback_flag_train_artifact_full_seed{seed}/cp_180/predictions",
    )
    parser.add_argument(
        "--multiscale_features",
        default="output/mechanism_moe_multiscale/multiscale_response_features.csv",
    )
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--forecast_context_by_sample", default=None)
    parser.add_argument("--scope", default="rain_sensitive_nodes")
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--feature_groups", type=parse_csv_strings, default=parse_csv_strings("all,response_only,no_rain_gate"))
    parser.add_argument("--neighbors", type=parse_csv_ints, default=parse_csv_ints("1,3,5"))
    parser.add_argument("--temperatures", type=parse_csv_floats, default=parse_csv_floats("0.25,0.35,0.60,0.90"))
    parser.add_argument("--global_gate_weights", type=parse_csv_floats, default=parse_csv_floats("0.00,0.25,0.50,0.75"))
    parser.add_argument("--output_dir", default="output/forecast_gefs15_climfallback_flag_train_teacher_grid")
    parser.add_argument("--top_n", type=int, default=20)
    return parser.parse_args()


def load_seed_context(args, seed):
    pack = load_prediction_pack(args.prediction_dir_template.format(seed=seed))
    if "train_true" not in pack:
        raise FileNotFoundError(f"Seed {seed} predictions do not include train arrays.")
    n_nodes = pack["train_true"].shape[2]
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, n_nodes)
    val_names, val_experts = expert_stack(pack, "val")
    train_names, train_experts = expert_stack(pack, "train")
    if val_names != train_names:
        raise ValueError(f"Validation and train expert names differ for seed {seed}.")
    val_losses = sample_expert_mae(pack["val_true"], val_experts, node_mask=rain_mask)
    selected = pack["train_selected"]
    baseline = {
        "global_mae": mae(pack["train_true"], selected),
        "rain_sensitive_mae": mae(pack["train_true"], selected, rain_mask),
        "rain_top_mae": mae(pack["train_true"], selected, rain_top_mask),
    }
    return {
        "seed": seed,
        "pack": pack,
        "rain_mask": rain_mask,
        "rain_top_mask": rain_top_mask,
        "expert_names": train_names,
        "val_experts": val_experts,
        "train_experts": train_experts,
        "val_losses": val_losses,
        "baseline": baseline,
    }


def feature_matrices(args, ctx, feature_group):
    pack = ctx["pack"]
    val_feature_df, feature_cols = sample_features(
        args.multiscale_features,
        args.scope,
        pack["val_indices"],
        args.n_his,
        args.n_pred,
        feature_group,
    )
    train_feature_df, _ = sample_features(
        args.multiscale_features,
        args.scope,
        pack["train_indices"],
        args.n_his,
        args.n_pred,
        feature_group,
    )
    val_x = val_feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train_x = train_feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    val_forecast_x, val_forecast_cols = forecast_gate_features(args.forecast_context_by_sample, pack["val_indices"])
    train_forecast_x, train_forecast_cols = forecast_gate_features(args.forecast_context_by_sample, pack["train_indices"])
    if val_forecast_cols or train_forecast_cols:
        if val_forecast_cols != train_forecast_cols:
            raise ValueError("Validation and train forecast gate feature columns do not match.")
        val_x = pd.concat([val_x.reset_index(drop=True), val_forecast_x.reset_index(drop=True)], axis=1)
        train_x = pd.concat([train_x.reset_index(drop=True), train_forecast_x.reset_index(drop=True)], axis=1)
        feature_cols = feature_cols + val_forecast_cols
    return val_x, train_x, feature_cols


def evaluate_config(args, ctx, feature_group, neighbors, temperature, global_gate_weight):
    val_x, train_x, feature_cols = feature_matrices(args, ctx, feature_group)
    gates = build_gates(
        val_x,
        train_x,
        ctx["val_losses"],
        neighbors,
        temperature,
        global_gate_weight,
    )
    teacher = combine_experts(ctx["train_experts"], gates)
    pack = ctx["pack"]
    metrics = {
        "teacher_global_mae": mae(pack["train_true"], teacher),
        "teacher_rain_sensitive_mae": mae(pack["train_true"], teacher, ctx["rain_mask"]),
        "teacher_rain_top_mae": mae(pack["train_true"], teacher, ctx["rain_top_mask"]),
    }
    metrics["global_gain"] = ctx["baseline"]["global_mae"] - metrics["teacher_global_mae"]
    metrics["rain_sensitive_gain"] = ctx["baseline"]["rain_sensitive_mae"] - metrics["teacher_rain_sensitive_mae"]
    metrics["rain_top_gain"] = ctx["baseline"]["rain_top_mae"] - metrics["teacher_rain_top_mae"]
    metrics["mean_selected_weight"] = float(gates[:, ctx["expert_names"].index("selected_model")].mean())
    metrics["feature_count"] = len(feature_cols)
    return metrics


def main():
    args = parse_args()
    invalid = [name for name in args.feature_groups if name not in FEATURE_GROUPS]
    if invalid:
        raise ValueError(f"Unknown feature_groups: {invalid}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_contexts = [load_seed_context(args, seed) for seed in args.seeds]
    rows = []
    for ctx in seed_contexts:
        for feature_group in args.feature_groups:
            for neighbors in args.neighbors:
                for temperature in args.temperatures:
                    for global_gate_weight in args.global_gate_weights:
                        row = {
                            "seed": ctx["seed"],
                            "feature_group": feature_group,
                            "neighbors": neighbors,
                            "temperature": temperature,
                            "global_gate_weight": global_gate_weight,
                            "baseline_global_mae": ctx["baseline"]["global_mae"],
                            "baseline_rain_sensitive_mae": ctx["baseline"]["rain_sensitive_mae"],
                            "baseline_rain_top_mae": ctx["baseline"]["rain_top_mae"],
                        }
                        row.update(
                            evaluate_config(
                                args,
                                ctx,
                                feature_group,
                                neighbors,
                                temperature,
                                global_gate_weight,
                            )
                        )
                        rows.append(row)
    seed_df = pd.DataFrame(rows)
    summary_df = (
        seed_df.groupby(["feature_group", "neighbors", "temperature", "global_gate_weight"], as_index=False)[
            ["global_gain", "rain_sensitive_gain", "rain_top_gain"]
        ]
        .mean()
        .sort_values(["rain_sensitive_gain", "global_gain", "rain_top_gain"], ascending=False)
        .reset_index(drop=True)
    )
    seed_df.to_csv(output_dir / "train_teacher_grid_seed_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "train_teacher_grid_summary.csv", index=False)
    print(summary_df.head(args.top_n).to_string(index=False))
    print(f">> Saved train teacher grid outputs to: {output_dir}")


if __name__ == "__main__":
    main()
