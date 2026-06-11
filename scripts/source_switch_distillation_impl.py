import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae


def parse_args():
    parser = argparse.ArgumentParser(description="Export source-switch posthoc predictions as distillation targets.")
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--source_switch_dir", required=True)
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--target_split", choices=["test", "train"], default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--event_calibrator_path", default=None, help="Path to event calibrator JSON model.")
    parser.add_argument("--event_context_features", default="output/rainfall_event_catalog/rainfall_event_catalog.csv")
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--forecast_context_by_sample", default="output/forecast_context_diagnostics_cp180_climfallback_flag/forecast_context_by_sample.csv")
    parser.add_argument("--source_col", default="forecast_is_fallback")
    return parser.parse_args()


def sample_mae(y_true, y_pred, node_mask=None):
    if node_mask is not None:
        y_true = y_true[:, :, node_mask, :]
        y_pred = y_pred[:, :, node_mask, :]
    return np.mean(np.abs(y_true - y_pred), axis=(1, 2, 3))


def write_report(path, rows, args):
    lines = [
        "# Source-Switch Distillation Targets",
        "",
        "This artifact exports posthoc source-switch predictions as teacher targets for later model distillation.",
        "It does not retrain CPD-STGCN.",
        "",
        "## Inputs",
        "",
        f"- predictions_dir: `{args.predictions_dir}`",
        f"- source_switch_dir: `{args.source_switch_dir}`",
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
            "## Use",
            "",
            "- `teacher_pred_real.npy`: posthoc teacher prediction.",
            "- `teacher_delta_real.npy`: teacher minus selected_model prediction.",
            "- `distillation_sample_metrics.csv`: sample-level improvement and source segment labels.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pred_dir = artifacts.validate_predictions_dir(args.predictions_dir)
    switch_dir = Path(args.source_switch_dir)
    teacher_name = "moe_test_pred_real.npy" if args.target_split == "test" else f"moe_{args.target_split}_pred_real.npy"
    teacher_path = switch_dir / teacher_name
    assignment_path = switch_dir / "source_switch_assignments.csv"
    if not assignment_path.exists():
        event_assignment_path = switch_dir / "event_source_switch_assignments.csv"
        if event_assignment_path.exists():
            assignment_path = event_assignment_path
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing source-switch teacher prediction: {teacher_path}")

    if args.target_split == "test":
        y_true = artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL)
        selected = artifacts.load_array(pred_dir, artifacts.TEST_PRED_REAL)
        persistence = artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL)
        indices = artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32)
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
    teacher = np.load(teacher_path)
    assignments = pd.read_csv(assignment_path) if assignment_path.exists() else None
    if teacher.shape != selected.shape:
        raise ValueError(f"Teacher shape {teacher.shape} does not match selected shape {selected.shape}.")

    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, y_true.shape[2])
    selected_sample = sample_mae(y_true, selected, rain_mask)
    teacher_sample = sample_mae(y_true, teacher, rain_mask)
    sample_df = pd.DataFrame(
        {
            "sample_index": np.arange(len(indices), dtype=np.int32),
            "sample_start": indices,
            "selected_rain_sensitive_mae": selected_sample,
            "teacher_rain_sensitive_mae": teacher_sample,
            "teacher_gain_rain_sensitive_mae": selected_sample - teacher_sample,
            "selected_global_mae": sample_mae(y_true, selected),
            "teacher_global_mae": sample_mae(y_true, teacher),
            "teacher_gain_global_mae": sample_mae(y_true, selected) - sample_mae(y_true, teacher),
        }
    )
    if assignments is not None:
        sample_df = sample_df.merge(assignments, on=["sample_index", "sample_start"], how="left")
        sample_df["teacher_source"] = "source_switch"
    else:
        sample_df["teacher_source"] = "posthoc"

    if args.event_calibrator_path and Path(args.event_calibrator_path).exists():
        from scripts.train_event_calibrator_impl import load_calibrator
        from scripts.event_source_switch_posthoc_moe_impl import load_event_features, source_switch_mask
        print(f">> Loading event calibrator from {args.event_calibrator_path}...")
        calibrator = load_calibrator(args.event_calibrator_path)
        load_cols = list(calibrator.features)
        features = load_event_features(
            args.event_context_features,
            args.file_path,
            indices,
            args.n_his,
            args.n_pred,
            load_cols
        )
        features_df = pd.DataFrame(features)
        probs = calibrator.predict_proba(features_df)
        sample_df["event_calibrated_probability"] = probs
        
        fallback = source_switch_mask(args.forecast_context_by_sample, indices, args.source_col)
        sample_df["event_fallback_calibrated_weight"] = probs * fallback
        print(f">> Added calibrated weight 'event_fallback_calibrated_weight' to sample metrics (mean={(probs * fallback).mean():.4f}).")
    rows = [
        {
            "model": "selected_model",
            "global_mae": mae(y_true, selected),
            "rain_sensitive_mae": mae(y_true, selected, rain_mask),
            "rain_top_mae": mae(y_true, selected, rain_top_mask),
        },
        {
            "model": "source_switch_teacher",
            "global_mae": mae(y_true, teacher),
            "rain_sensitive_mae": mae(y_true, teacher, rain_mask),
            "rain_top_mae": mae(y_true, teacher, rain_top_mask),
        },
        {
            "model": "persistence",
            "global_mae": mae(y_true, persistence),
            "rain_sensitive_mae": mae(y_true, persistence, rain_mask),
            "rain_top_mae": mae(y_true, persistence, rain_top_mask),
        },
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "teacher_pred_real.npy", teacher.astype(np.float32))
    np.save(output_dir / "selected_pred_real.npy", selected.astype(np.float32))
    np.save(output_dir / "teacher_delta_real.npy", (teacher - selected).astype(np.float32))
    sample_df.to_csv(output_dir / "distillation_sample_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "distillation_metrics.csv", index=False)
    write_report(output_dir / "source_switch_distillation_report.md", rows, args)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved source-switch distillation targets to: {output_dir}")


if __name__ == "__main__":
    main()
