import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project displacement forecasts toward simple persistence-anchored motion constraints."
    )
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--model", choices=("selected", "raw", "node_calibrated"), default="selected")
    parser.add_argument("--apply_scope", choices=("all", "rain_sensitive", "rain_top"), default="rain_sensitive")
    parser.add_argument("--auto_select", action="store_true")
    parser.add_argument("--candidate_alpha", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_prediction(pred_dir, split, model):
    if split == "val":
        if model == "raw":
            return artifacts.load_array(pred_dir, artifacts.VAL_PRED_RAW_REAL)
        if model == "node_calibrated":
            return artifacts.load_array(pred_dir, artifacts.VAL_PRED_CAL_REAL)
        return artifacts.load_array(pred_dir, artifacts.VAL_PRED_REAL)
    if model == "raw":
        return artifacts.load_array(pred_dir, artifacts.TEST_PRED_RAW_REAL)
    if model == "node_calibrated":
        return artifacts.load_array(pred_dir, artifacts.TEST_PRED_CAL_REAL)
    return artifacts.load_array(pred_dir, artifacts.TEST_PRED_REAL)


def correction_mask(scope, rain_mask, rain_top_mask, n_nodes):
    if scope == "all":
        return np.ones(n_nodes, dtype=bool)
    if scope == "rain_top":
        return rain_top_mask
    return rain_mask


def monotone_endpoint_projection(pred, persistence, apply_mask, alpha):
    projected = pred.copy()
    base = persistence[:, :1, :, :]
    delta = pred - base
    endpoint = delta[:, -1:, :, :]
    sign = np.sign(endpoint)
    signed_delta = sign * delta
    signed_delta = np.maximum(signed_delta, 0.0)
    signed_delta = np.maximum.accumulate(signed_delta, axis=1)
    constrained = base + sign * signed_delta
    blended = (1.0 - float(alpha)) * pred + float(alpha) * constrained
    projected[:, :, apply_mask, :] = blended[:, :, apply_mask, :]
    return projected.astype(np.float32)


def masked_mae(y_true, y_pred, apply_mask):
    return float(np.mean(np.abs(y_true[:, :, apply_mask, :] - y_pred[:, :, apply_mask, :])))


def auto_select_alpha(val_true, val_pred, val_persistence, apply_mask, args):
    rows = []
    best = None
    for alpha in args.candidate_alpha:
        projected = monotone_endpoint_projection(val_pred, val_persistence, apply_mask, alpha)
        score = masked_mae(val_true, projected, apply_mask)
        row = {"alpha": float(alpha), "val_apply_scope_mae": score}
        rows.append(row)
        if best is None or score < best["val_apply_scope_mae"]:
            best = row
    return best, pd.DataFrame(rows).sort_values("val_apply_scope_mae").reset_index(drop=True)


def write_report(path, rows, selected_alpha, args):
    lines = [
        "# Displacement Projection",
        "",
        "This command applies a persistence-anchored monotone endpoint projection to forecast trajectories.",
        "It is a post-hoc physical-constraint probe and does not retrain CPD-STGCN.",
        "",
        "## Configuration",
        "",
        f"- predictions_dir: `{args.predictions_dir}`",
        f"- model: `{args.model}`",
        f"- apply_scope: `{args.apply_scope}`",
        f"- auto_select: `{args.auto_select}`",
        f"- selected_alpha: `{selected_alpha}`",
        "",
        "## Metrics",
        "",
        "| model | global_mae | rain_sensitive_mae | rain_top_mae |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['global_mae']:.3f} | {row['rain_sensitive_mae']:.3f} | {row['rain_top_mae']:.3f} |"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pred_dir = artifacts.validate_predictions_dir(args.predictions_dir)
    val_true = artifacts.load_array(pred_dir, artifacts.VAL_TRUE_REAL)
    test_true = artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL)
    val_pred = load_prediction(pred_dir, "val", args.model)
    test_pred = load_prediction(pred_dir, "test", args.model)
    val_persistence = artifacts.load_array(pred_dir, artifacts.VAL_PERSISTENCE_REAL)
    test_persistence = artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL)
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, test_true.shape[2])
    apply_mask = correction_mask(args.apply_scope, rain_mask, rain_top_mask, test_true.shape[2])

    selected_alpha = float(args.alpha)
    selection_df = pd.DataFrame()
    if args.auto_select:
        best, selection_df = auto_select_alpha(val_true, val_pred, val_persistence, apply_mask, args)
        selected_alpha = float(best["alpha"])
    projected = monotone_endpoint_projection(test_pred, test_persistence, apply_mask, selected_alpha)

    rows = [
        {
            "model": args.model,
            "global_mae": mae(test_true, test_pred),
            "rain_sensitive_mae": mae(test_true, test_pred, rain_mask),
            "rain_top_mae": mae(test_true, test_pred, rain_top_mask),
        },
        {
            "model": "displacement_projected",
            "global_mae": mae(test_true, projected),
            "rain_sensitive_mae": mae(test_true, projected, rain_mask),
            "rain_top_mae": mae(test_true, projected, rain_top_mask),
        },
        {
            "model": "persistence",
            "global_mae": mae(test_true, test_persistence),
            "rain_sensitive_mae": mae(test_true, test_persistence, rain_mask),
            "rain_top_mae": mae(test_true, test_persistence, rain_top_mask),
        },
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "displacement_projected_test_pred_real.npy", projected)
    pd.DataFrame(rows).to_csv(output_dir / "displacement_projection_metrics.csv", index=False)
    if not selection_df.empty:
        selection_df.to_csv(output_dir / "displacement_projection_auto_selection.csv", index=False)
    write_report(output_dir / "displacement_projection_report.md", rows, selected_alpha, args)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved displacement projection outputs to: {output_dir}")


if __name__ == "__main__":
    main()
