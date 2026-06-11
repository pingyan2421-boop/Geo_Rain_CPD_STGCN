import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-only quantile mapping calibration for persisted predictions.")
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


def scope_mask(scope, rain_mask, rain_top_mask, n_nodes):
    if scope == "all":
        return np.ones(n_nodes, dtype=bool)
    if scope == "rain_top":
        return rain_top_mask
    return rain_mask


def fit_quantile_map(reference_pred, reference_true, mask):
    pred_values = np.sort(reference_pred[:, :, mask, :].reshape(-1).astype(np.float64))
    true_values = np.sort(reference_true[:, :, mask, :].reshape(-1).astype(np.float64))
    if pred_values.size < 2 or true_values.size < 2:
        raise ValueError("Not enough values to fit quantile mapping.")
    quantiles = np.linspace(0.0, 1.0, pred_values.size)
    mapped_true = np.interp(quantiles, np.linspace(0.0, 1.0, true_values.size), true_values)
    return pred_values, mapped_true


def apply_quantile_map(pred, mask, pred_quantiles, true_quantiles, alpha):
    corrected = pred.copy()
    values = corrected[:, :, mask, :].reshape(-1).astype(np.float64)
    mapped = np.interp(values, pred_quantiles, true_quantiles, left=true_quantiles[0], right=true_quantiles[-1])
    blended = (1.0 - float(alpha)) * values + float(alpha) * mapped
    scoped = corrected[:, :, mask, :]
    scoped[...] = blended.reshape(scoped.shape).astype(np.float32)
    corrected[:, :, mask, :] = scoped
    return corrected.astype(np.float32)


def masked_mae(y_true, y_pred, mask):
    return float(np.mean(np.abs(y_true[:, :, mask, :] - y_pred[:, :, mask, :])))


def select_alpha(val_true, val_pred, mask, pred_quantiles, true_quantiles, args):
    rows = []
    best = None
    for alpha in args.candidate_alpha:
        corrected = apply_quantile_map(val_pred, mask, pred_quantiles, true_quantiles, alpha)
        score = masked_mae(val_true, corrected, mask)
        row = {"alpha": float(alpha), "val_apply_scope_mae": score}
        rows.append(row)
        if best is None or score < best["val_apply_scope_mae"]:
            best = row
    return float(best["alpha"]), pd.DataFrame(rows).sort_values("val_apply_scope_mae").reset_index(drop=True)


def write_report(path, rows, selected_alpha, args):
    lines = [
        "# Quantile Mapping Calibration",
        "",
        "This command fits empirical quantile mapping on validation predictions and applies it to test predictions.",
        "It is a post-hoc distribution calibration probe and does not retrain CPD-STGCN.",
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
    test_persistence = artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL)
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, test_true.shape[2])
    mask = scope_mask(args.apply_scope, rain_mask, rain_top_mask, test_true.shape[2])
    pred_quantiles, true_quantiles = fit_quantile_map(val_pred, val_true, mask)

    selected_alpha = float(args.alpha)
    selection_df = pd.DataFrame()
    if args.auto_select:
        selected_alpha, selection_df = select_alpha(val_true, val_pred, mask, pred_quantiles, true_quantiles, args)
    corrected = apply_quantile_map(test_pred, mask, pred_quantiles, true_quantiles, selected_alpha)

    rows = [
        {
            "model": args.model,
            "global_mae": mae(test_true, test_pred),
            "rain_sensitive_mae": mae(test_true, test_pred, rain_mask),
            "rain_top_mae": mae(test_true, test_pred, rain_top_mask),
        },
        {
            "model": "quantile_mapped",
            "global_mae": mae(test_true, corrected),
            "rain_sensitive_mae": mae(test_true, corrected, rain_mask),
            "rain_top_mae": mae(test_true, corrected, rain_top_mask),
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
    np.save(output_dir / "quantile_mapped_test_pred_real.npy", corrected)
    pd.DataFrame(rows).to_csv(output_dir / "quantile_mapping_metrics.csv", index=False)
    if not selection_df.empty:
        selection_df.to_csv(output_dir / "quantile_mapping_auto_selection.csv", index=False)
    pd.DataFrame({"pred_quantile": pred_quantiles, "true_quantile": true_quantiles}).to_csv(
        output_dir / "quantile_mapping_curve.csv",
        index=False,
    )
    write_report(output_dir / "quantile_mapping_report.md", rows, selected_alpha, args)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved quantile mapping outputs to: {output_dir}")


if __name__ == "__main__":
    main()
