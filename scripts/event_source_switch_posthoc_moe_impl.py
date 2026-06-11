import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.date_loader import load_and_clean_data
from scripts.common import prediction_artifacts as artifacts
from scripts.cpd_split_validate_impl import event_context_for_starts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae
from scripts.source_switch_posthoc_moe_impl import load_posthoc_prediction, source_switch_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Switch among official, fallback-event, and fallback-nonevent posthoc MoE predictions."
    )
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--official_posthoc_dir", required=True)
    parser.add_argument("--fallback_event_posthoc_dir", required=True)
    parser.add_argument("--fallback_noevent_posthoc_dir", required=True)
    parser.add_argument("--forecast_context_by_sample", required=True)
    parser.add_argument("--event_context_features", required=True)
    parser.add_argument("--event_calibrator_path", default=None, help="Path to event calibrator JSON model.")
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--source_col", default="forecast_is_fallback")
    parser.add_argument("--event_col", default="event_historical_trigger_probability")
    parser.add_argument("--event_threshold", type=float, default=0.25)
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--target_split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--continuous_gate", action="store_true")
    parser.add_argument("--gate_coef_prob", type=float, default=10.0)
    parser.add_argument("--gate_coef_wetness", type=float, default=0.0)
    parser.add_argument("--gate_bias", type=float, default=-2.5)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_split_arrays(pred_dir, target_split):
    if target_split == "test":
        return (
            artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL),
            artifacts.load_array(pred_dir, artifacts.TEST_PRED_REAL),
            artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL),
            artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32),
        )
    elif target_split == "val":
        return (
            artifacts.load_array(pred_dir, artifacts.VAL_TRUE_REAL),
            artifacts.load_array(pred_dir, artifacts.VAL_PRED_REAL),
            artifacts.load_array(pred_dir, artifacts.VAL_PERSISTENCE_REAL),
            artifacts.load_array(pred_dir, artifacts.VAL_INDICES).astype(np.int32),
        )
    missing = artifacts.missing_required_files(pred_dir, artifacts.OPTIONAL_TRAIN_ARRAYS)
    if missing:
        raise FileNotFoundError(f"Prediction artifacts do not contain train arrays: {', '.join(missing)}")
    return (
        artifacts.load_array(pred_dir, artifacts.TRAIN_TRUE_REAL),
        artifacts.load_array(pred_dir, artifacts.TRAIN_PRED_REAL),
        artifacts.load_array(pred_dir, artifacts.TRAIN_PERSISTENCE_REAL),
        artifacts.load_array(pred_dir, artifacts.TRAIN_INDICES).astype(np.int32),
    )


def load_event_features(event_csv, file_path, indices, n_his, n_pred, cols):
    raw_seq, _, _, time_cols, _ = load_and_clean_data(file_path)
    starts = np.arange(0, raw_seq.shape[0] - n_his - n_pred + 1, dtype=np.int32)
    context = event_context_for_starts(event_csv, time_cols, starts, n_his)
    res = {}
    for col in cols:
        if col not in context.columns:
            res[col] = np.zeros(len(indices), dtype=np.float32)
        else:
            res[col] = context.iloc[np.asarray(indices, dtype=np.int32)][col].to_numpy(dtype=float)
            res[col] = np.nan_to_num(res[col], nan=0.0)
    return res


def event_mask(event_csv, file_path, indices, n_his, n_pred, event_col, event_threshold=0.0):
    raw_seq, _, _, time_cols, _ = load_and_clean_data(file_path)
    starts = np.arange(0, raw_seq.shape[0] - n_his - n_pred + 1, dtype=np.int32)
    context = event_context_for_starts(event_csv, time_cols, starts, n_his)
    if event_col not in context.columns:
        raise ValueError(f"event_context is missing column: {event_col}")
    values = context.iloc[np.asarray(indices, dtype=np.int32)][event_col].to_numpy(dtype=float)
    if event_threshold > 0.0:
        return values >= float(event_threshold)
    return values > 0.0


def write_report(path, metrics, counts, args):
    lines = [
        "# Event-Source Switch Post-hoc MoE",
        "",
        "## Configuration",
        "",
        f"- official_posthoc_dir: `{args.official_posthoc_dir}`",
        f"- fallback_event_posthoc_dir: `{args.fallback_event_posthoc_dir}`",
        f"- fallback_noevent_posthoc_dir: `{args.fallback_noevent_posthoc_dir}`",
        f"- source_col: `{args.source_col}`",
        f"- event_col: `{args.event_col}`",
        f"- event_threshold: `{args.event_threshold}`",
        f"- official_samples: {counts['official']}",
        f"- fallback_event_samples: {counts['fallback_event']}",
        f"- fallback_noevent_samples: {counts['fallback_noevent']}",
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
            "This is a post-hoc diagnostic switch over already generated predictions. "
            "It does not retrain CPD-STGCN and should be validated without test-set configuration selection before being treated as a formal model.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pred_dir = artifacts.validate_predictions_dir(args.predictions_dir)
    y_true, selected, persistence, indices = load_split_arrays(pred_dir, args.target_split)

    official_pred = load_posthoc_prediction(args.official_posthoc_dir, args.target_split)
    fallback_event_pred = load_posthoc_prediction(args.fallback_event_posthoc_dir, args.target_split)
    fallback_noevent_pred = load_posthoc_prediction(args.fallback_noevent_posthoc_dir, args.target_split)
    for pred in (fallback_event_pred, fallback_noevent_pred):
        if pred.shape != official_pred.shape:
            raise ValueError(f"Posthoc prediction shapes do not match: {official_pred.shape} vs {pred.shape}")

    fallback = source_switch_mask(args.forecast_context_by_sample, indices, args.source_col)

    # Determine probabilities and antecedent wetness index
    if args.event_calibrator_path and Path(args.event_calibrator_path).exists():
        from scripts.train_event_calibrator_impl import load_calibrator
        print(f">> Loading event calibrator from {args.event_calibrator_path}...")
        calibrator = load_calibrator(args.event_calibrator_path)
        load_cols = list(set(calibrator.features + ["event_antecedent_wetness_index"]))
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
        wetness = features["event_antecedent_wetness_index"]
        print(f">> Computed calibrated probabilities: mean={probs.mean():.4f}, max={probs.max():.4f}")
    else:
        features = load_event_features(
            args.event_context_features,
            args.file_path,
            indices,
            args.n_his,
            args.n_pred,
            [args.event_col, "event_antecedent_wetness_index"]
        )
        probs = features[args.event_col]
        wetness = features["event_antecedent_wetness_index"]

    active_event = probs >= args.event_threshold

    if args.continuous_gate:
        linear_val = args.gate_coef_prob * probs + args.gate_coef_wetness * wetness + args.gate_bias
        w = 1.0 / (1.0 + np.exp(-linear_val))
        
        w_expanded = w.reshape(-1, 1, 1, 1)
        fallback_expanded = fallback.reshape(-1, 1, 1, 1)
        
        switched = (1.0 - fallback_expanded) * official_pred + fallback_expanded * (
            w_expanded * fallback_event_pred + (1.0 - w_expanded) * fallback_noevent_pred
        )
    else:
        switched = official_pred.copy()
        fallback_event = fallback & active_event
        fallback_noevent = fallback & ~active_event
        switched[fallback_event] = fallback_event_pred[fallback_event]
        switched[fallback_noevent] = fallback_noevent_pred[fallback_noevent]

    fallback_event = fallback & active_event
    fallback_noevent = fallback & ~active_event

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
            "model": "event_source_switch_posthoc_moe",
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
            "use_fallback_event_posthoc": fallback_event.astype(np.int32),
            "use_fallback_noevent_posthoc": fallback_noevent.astype(np.int32),
            "event_active": active_event.astype(np.int32),
            "forecast_is_fallback": fallback.astype(np.int32),
        }
    ).to_csv(output_dir / "event_source_switch_assignments.csv", index=False)
    counts = {
        "official": int((~fallback).sum()),
        "fallback_event": int(fallback_event.sum()),
        "fallback_noevent": int(fallback_noevent.sum()),
    }
    write_report(output_dir / "event_source_switch_posthoc_moe_report.md", rows, counts, args)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved event-source-switch post-hoc MoE outputs to: {output_dir}")


if __name__ == "__main__":
    main()
