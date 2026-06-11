import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae
from scripts.source_switch_posthoc_moe_impl import source_switch_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply validation-only source-aware residual calibration to persisted predictions."
    )
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--forecast_context_by_sample", required=True)
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--switch_col", default="forecast_is_fallback")
    parser.add_argument("--base_model", choices=["selected", "raw", "node_calibrated"], default="selected")
    parser.add_argument("--bias_scope", choices=["global", "node", "node_horizon"], default="node_horizon")
    parser.add_argument("--shrinkage", type=float, default=12.0)
    parser.add_argument("--max_abs_bias", type=float, default=1.0)
    parser.add_argument("--apply_unseen_source", action="store_true")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_base_prediction(pred_dir, split, model_name):
    if split == "val":
        if model_name == "selected":
            return artifacts.load_array(pred_dir, artifacts.VAL_PRED_REAL)
        if model_name == "raw":
            return artifacts.load_array(pred_dir, artifacts.VAL_PRED_RAW_REAL)
        return artifacts.load_array(pred_dir, artifacts.VAL_PRED_CAL_REAL)
    if model_name == "selected":
        return artifacts.load_array(pred_dir, artifacts.TEST_PRED_REAL)
    if model_name == "raw":
        return artifacts.load_array(pred_dir, artifacts.TEST_PRED_RAW_REAL)
    return artifacts.load_array(pred_dir, artifacts.TEST_PRED_CAL_REAL)


def source_values(context_csv, indices, switch_col):
    return source_switch_mask(context_csv, indices, switch_col).astype(np.int32)


def mean_bias(residual, scope):
    if scope == "global":
        return np.full((1, 1, 1, 1), np.mean(residual, dtype=np.float64), dtype=np.float32)
    if scope == "node":
        return np.mean(residual, axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    return np.mean(residual, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)


def calibrate(val_true, val_pred, val_source, test_pred, test_source, args):
    calibrated = test_pred.copy()
    global_bias = mean_bias(val_true - val_pred, args.bias_scope)
    global_weight = len(val_source) / (len(val_source) + max(args.shrinkage, 0.0))
    global_bias = global_weight * global_bias
    seen_sources = set()

    for source in (0, 1):
        val_mask = val_source == source
        test_mask = test_source == source
        if not np.any(test_mask):
            continue
        if np.any(val_mask):
            seen_sources.add(source)
            bias = mean_bias(val_true[val_mask] - val_pred[val_mask], args.bias_scope)
            weight = int(val_mask.sum()) / (int(val_mask.sum()) + max(args.shrinkage, 0.0))
            bias = weight * bias
        elif args.apply_unseen_source:
            bias = global_bias
        else:
            continue
        bias = np.clip(bias, -abs(args.max_abs_bias), abs(args.max_abs_bias))
        calibrated[test_mask] = calibrated[test_mask] + bias

    return calibrated, sorted(seen_sources)


def metrics_rows(y_true, predictions, rain_mask, rain_top_mask):
    rows = []
    for name, pred in predictions.items():
        rows.append(
            {
                "model": name,
                "global_mae": mae(y_true, pred),
                "rain_sensitive_mae": mae(y_true, pred, rain_mask),
                "rain_top_mae": mae(y_true, pred, rain_top_mask),
            }
        )
    return rows


def write_report(path, rows, seen_sources, args):
    lines = [
        "# Source-Aware Residual Calibration",
        "",
        "This calibration learns residual bias from validation predictions only and applies it by forecast source.",
        "",
        "## Configuration",
        "",
        f"- base_model: `{args.base_model}`",
        f"- switch_col: `{args.switch_col}`",
        f"- bias_scope: `{args.bias_scope}`",
        f"- shrinkage: `{args.shrinkage}`",
        f"- max_abs_bias: `{args.max_abs_bias}`",
        f"- validation sources observed: `{','.join(map(str, seen_sources))}`",
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
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is validation-only post-training calibration, not CPD-STGCN retraining.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pred_dir = artifacts.validate_predictions_dir(args.predictions_dir)
    val_true = artifacts.load_array(pred_dir, artifacts.VAL_TRUE_REAL)
    test_true = artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL)
    val_pred = load_base_prediction(pred_dir, "val", args.base_model)
    test_pred = load_base_prediction(pred_dir, "test", args.base_model)
    persistence = artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL)
    val_indices = artifacts.load_array(pred_dir, artifacts.VAL_INDICES).astype(np.int32)
    test_indices = artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32)
    val_source = source_values(args.forecast_context_by_sample, val_indices, args.switch_col)
    test_source = source_values(args.forecast_context_by_sample, test_indices, args.switch_col)
    calibrated, seen_sources = calibrate(val_true, val_pred, val_source, test_pred, test_source, args)

    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, test_true.shape[2])
    rows = metrics_rows(
        test_true,
        {
            args.base_model: test_pred,
            "source_residual_calibrated": calibrated,
            "persistence": persistence,
        },
        rain_mask,
        rain_top_mask,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "source_residual_calibrated_test_pred_real.npy", calibrated.astype(np.float32))
    pd.DataFrame(rows).to_csv(output_dir / "source_residual_calibration_metrics.csv", index=False)
    pd.DataFrame(
        {
            "sample_index": np.arange(len(test_indices), dtype=np.int32),
            "sample_start": test_indices,
            args.switch_col: test_source,
        }
    ).to_csv(output_dir / "source_residual_calibration_assignments.csv", index=False)
    write_report(output_dir / "source_residual_calibration_report.md", rows, seen_sources, args)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved source-aware residual calibration outputs to: {output_dir}")


if __name__ == "__main__":
    main()
