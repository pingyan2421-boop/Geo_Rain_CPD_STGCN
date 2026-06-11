import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae


def parse_args():
    parser = argparse.ArgumentParser(
        description="Switch between two post-hoc MoE predictions using forecast source metadata."
    )
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--primary_posthoc_dir", required=True)
    parser.add_argument("--fallback_posthoc_dir", required=True)
    parser.add_argument("--forecast_context_by_sample", required=True)
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--switch_col", default="forecast_is_fallback")
    parser.add_argument("--target_split", choices=["test", "train", "val"], default="test")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_posthoc_prediction(path, target_split):
    filename = "moe_test_pred_real.npy" if target_split == "test" else f"moe_{target_split}_pred_real.npy"
    pred_path = Path(path) / filename
    if not pred_path.exists():
        # Fallback to general split naming for val
        if target_split == "val":
            pred_path = Path(path) / "moe_val_pred_real.npy"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing posthoc prediction: {pred_path}")
    return np.load(pred_path)


def source_switch_mask(context_csv, indices, switch_col):
    df = pd.read_csv(context_csv)
    required = {"sample_start", switch_col}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise ValueError(f"forecast_context_by_sample is missing columns: {sorted(missing_cols)}")
    by_start = df.drop_duplicates("sample_start").set_index("sample_start")
    missing = [int(idx) for idx in indices if int(idx) not in by_start.index]
    if missing:
        raise ValueError(f"forecast_context_by_sample is missing sample_start values: {missing[:5]}")
    return by_start.loc[np.asarray(indices, dtype=np.int32), switch_col].to_numpy(dtype=float) > 0.0


def write_report(path, metrics, switch_count, total_count, args):
    lines = [
        "# Source-Switch Post-hoc MoE",
        "",
        "## Configuration",
        "",
        f"- primary_posthoc_dir: `{args.primary_posthoc_dir}`",
        f"- fallback_posthoc_dir: `{args.fallback_posthoc_dir}`",
        f"- switch_col: `{args.switch_col}`",
        f"- switched_samples: {switch_count} / {total_count}",
        "",
        "## Metrics",
        "",
        "| model | global_mae | rain_sensitive_mae | rain_top_mae |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['model']} | {row['global_mae']:.3f} | {row['rain_sensitive_mae']:.3f} | {row['rain_top_mae']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a post-hoc source-aware switch over already generated predictions. "
            "It does not retrain CPD-STGCN.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pred_dir = artifacts.validate_predictions_dir(args.predictions_dir)
    if args.target_split == "test":
        y_true = artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL)
        selected = artifacts.load_array(pred_dir, artifacts.TEST_PRED_REAL)
        persistence = artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL)
        indices = artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32)
    elif args.target_split == "val":
        y_true = artifacts.load_array(pred_dir, artifacts.VAL_TRUE_REAL)
        selected = artifacts.load_array(pred_dir, artifacts.VAL_PRED_REAL)
        persistence = artifacts.load_array(pred_dir, artifacts.VAL_PERSISTENCE_REAL)
        indices = artifacts.load_array(pred_dir, artifacts.VAL_INDICES).astype(np.int32)
    else:
        missing = artifacts.missing_required_files(pred_dir, artifacts.OPTIONAL_TRAIN_ARRAYS)
        if missing:
            raise FileNotFoundError(
                f"Prediction artifacts do not contain train arrays: {', '.join(missing)}"
            )
        y_true = artifacts.load_array(pred_dir, artifacts.TRAIN_TRUE_REAL)
        selected = artifacts.load_array(pred_dir, artifacts.TRAIN_PRED_REAL)
        persistence = artifacts.load_array(pred_dir, artifacts.TRAIN_PERSISTENCE_REAL)
        indices = artifacts.load_array(pred_dir, artifacts.TRAIN_INDICES).astype(np.int32)
    primary_pred = load_posthoc_prediction(args.primary_posthoc_dir, args.target_split)
    fallback_pred = load_posthoc_prediction(args.fallback_posthoc_dir, args.target_split)
    if primary_pred.shape != fallback_pred.shape:
        raise ValueError(f"Posthoc prediction shapes do not match: {primary_pred.shape} vs {fallback_pred.shape}")

    switch = source_switch_mask(args.forecast_context_by_sample, indices, args.switch_col)
    switched = primary_pred.copy()
    switched[switch] = fallback_pred[switch]

    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, y_true.shape[2])
    rows = [
        {
            "model": "selected_model",
            "global_mae": mae(y_true, selected),
            "rain_sensitive_mae": mae(y_true, selected, rain_mask),
            "rain_top_mae": mae(y_true, selected, rain_top_mask),
        },
        {
            "model": "persistence",
            "global_mae": mae(y_true, persistence),
            "rain_sensitive_mae": mae(y_true, persistence, rain_mask),
            "rain_top_mae": mae(y_true, persistence, rain_top_mask),
        },
        {
            "model": "source_switch_posthoc_moe",
            "global_mae": mae(y_true, switched),
            "rain_sensitive_mae": mae(y_true, switched, rain_mask),
            "rain_top_mae": mae(y_true, switched, rain_top_mask),
        },
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "moe_test_pred_real.npy" if args.target_split == "test" else f"moe_{args.target_split}_pred_real.npy"
    np.save(output_dir / output_name, switched.astype(np.float32))
    pd.DataFrame(rows).to_csv(output_dir / "moe_metrics.csv", index=False)
    pd.DataFrame(
        {
            "sample_index": np.arange(len(indices), dtype=np.int32),
            "sample_start": indices,
            "use_fallback_posthoc": switch.astype(np.int32),
        }
    ).to_csv(output_dir / "source_switch_assignments.csv", index=False)
    write_report(output_dir / "source_switch_posthoc_moe_report.md", rows, int(switch.sum()), len(switch), args)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved source-switch post-hoc MoE outputs to: {output_dir}")


if __name__ == "__main__":
    main()
