import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.posthoc_mechanism_moe_impl import expert_stack, load_node_masks, load_prediction_pack, mae


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-selected convex stacking over persisted prediction experts.")
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--metric_scope", choices=("all", "rain_sensitive", "rain_top"), default="rain_sensitive")
    parser.add_argument("--step", type=float, default=0.10)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def scope_mask(scope, rain_mask, rain_top_mask, n_nodes):
    if scope == "all":
        return np.ones(n_nodes, dtype=bool)
    if scope == "rain_top":
        return rain_top_mask
    return rain_mask


def simplex_grid(n_experts, step):
    units = int(round(1.0 / float(step)))
    if units <= 0:
        raise ValueError("step must be positive.")
    for combo in product(range(units + 1), repeat=n_experts):
        if sum(combo) == units:
            yield np.asarray(combo, dtype=np.float32) / float(units)


def combine(experts, weights):
    return np.einsum("e,e...->...", weights.astype(np.float32), experts.astype(np.float32))


def select_weights(val_true, val_experts, metric_mask, step):
    rows = []
    best = None
    for weights in simplex_grid(val_experts.shape[0], step):
        pred = combine(val_experts, weights)
        score = mae(val_true, pred, metric_mask)
        row = {"val_mae": score, **{f"weight_{idx}": float(weight) for idx, weight in enumerate(weights)}}
        rows.append(row)
        if best is None or score < best["val_mae"]:
            best = row
    selected = np.asarray([best[f"weight_{idx}"] for idx in range(val_experts.shape[0])], dtype=np.float32)
    return selected, pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)


def write_report(path, rows, weight_rows, args):
    lines = [
        "# Expert Stacking",
        "",
        "This command selects a global convex expert combination on validation data and applies it to test predictions.",
        "It is a post-hoc forecast-combination probe and does not retrain CPD-STGCN.",
        "",
        "## Configuration",
        "",
        f"- predictions_dir: `{args.predictions_dir}`",
        f"- metric_scope: `{args.metric_scope}`",
        f"- step: `{args.step}`",
        "",
        "## Selected Weights",
        "",
        "| expert | weight |",
        "| --- | ---: |",
    ]
    for row in weight_rows:
        lines.append(f"| {row['expert']} | {row['weight']:.3f} |")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| model | global_mae | rain_sensitive_mae | rain_top_mae |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['global_mae']:.3f} | {row['rain_sensitive_mae']:.3f} | {row['rain_top_mae']:.3f} |"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pack = load_prediction_pack(args.predictions_dir)
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, pack["test_true"].shape[2])
    metric_mask = scope_mask(args.metric_scope, rain_mask, rain_top_mask, pack["test_true"].shape[2])
    val_names, val_experts = expert_stack(pack, "val")
    test_names, test_experts = expert_stack(pack, "test")
    if val_names != test_names:
        raise ValueError("Validation and test expert names do not match.")

    weights, grid_df = select_weights(pack["val_true"], val_experts, metric_mask, args.step)
    stacked = combine(test_experts, weights)
    rows = []
    for name, pred in zip(test_names, test_experts):
        rows.append(
            {
                "model": name,
                "global_mae": mae(pack["test_true"], pred),
                "rain_sensitive_mae": mae(pack["test_true"], pred, rain_mask),
                "rain_top_mae": mae(pack["test_true"], pred, rain_top_mask),
            }
        )
    rows.append(
        {
            "model": "expert_stacked",
            "global_mae": mae(pack["test_true"], stacked),
            "rain_sensitive_mae": mae(pack["test_true"], stacked, rain_mask),
            "rain_top_mae": mae(pack["test_true"], stacked, rain_top_mask),
        }
    )
    weight_rows = [{"expert": name, "weight": float(weight)} for name, weight in zip(test_names, weights)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "expert_stacked_test_pred_real.npy", stacked.astype(np.float32))
    pd.DataFrame(rows).to_csv(output_dir / "expert_stacking_metrics.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(output_dir / "expert_stacking_weights.csv", index=False)
    grid_df.to_csv(output_dir / "expert_stacking_weight_grid.csv", index=False)
    write_report(output_dir / "expert_stacking_report.md", rows, weight_rows, args)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(weight_rows).to_string(index=False))
    print(f">> Saved expert stacking outputs to: {output_dir}")


if __name__ == "__main__":
    main()
