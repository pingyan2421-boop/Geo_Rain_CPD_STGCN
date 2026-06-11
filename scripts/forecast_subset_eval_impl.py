import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts


def parse_named_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH.")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=PATH with non-empty name and path.")
    return name, path


def load_node_masks(path, n_nodes, top_quantile=0.60):
    df = pd.read_csv(path)
    if "rain_susceptibility" not in df.columns:
        raise ValueError("rain_susceptibility CSV must contain a rain_susceptibility column.")
    susceptibility = df["rain_susceptibility"].to_numpy(dtype=np.float32)
    if susceptibility.shape[0] != n_nodes:
        raise ValueError(f"rain_susceptibility length={susceptibility.shape[0]} != n_nodes={n_nodes}")
    active = susceptibility > 0.0
    top = np.zeros_like(active, dtype=bool)
    if active.any():
        top = susceptibility >= float(np.quantile(susceptibility[active], top_quantile))
    return active, top


def mae(y_true, y_pred, sample_mask=None, node_mask=None):
    true = y_true
    pred = y_pred
    if sample_mask is not None:
        true = true[sample_mask]
        pred = pred[sample_mask]
    if node_mask is not None:
        true = true[:, :, node_mask, :]
        pred = pred[:, :, node_mask, :]
    if true.size == 0:
        return np.nan
    return float(np.mean(np.abs(true - pred)))


def load_prediction_models(predictions_dir, posthoc_dir=None):
    pred_dir = artifacts.validate_predictions_dir(predictions_dir)
    y_true = artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL)
    indices = artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32)
    models = {
        "selected_model": artifacts.load_array(pred_dir, artifacts.TEST_PRED_REAL),
        "raw_model": artifacts.load_array(pred_dir, artifacts.TEST_PRED_RAW_REAL),
        "node_calibrated": artifacts.load_array(pred_dir, artifacts.TEST_PRED_CAL_REAL),
        "persistence": artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL),
    }
    if posthoc_dir:
        posthoc_path = Path(posthoc_dir) / "moe_test_pred_real.npy"
        if not posthoc_path.exists():
            raise FileNotFoundError(f"Missing posthoc prediction file: {posthoc_path}")
        models["posthoc_mechanism_moe"] = np.load(posthoc_path)
    return y_true, indices, models


def subset_masks(indices, diagnostics_df):
    test_diag = diagnostics_df[diagnostics_df["split"] == "test"].copy()
    if test_diag.empty:
        raise ValueError("Forecast diagnostics CSV does not contain split=test rows.")
    if "sample_start" not in test_diag.columns or "forecast_available" not in test_diag.columns:
        raise ValueError("Forecast diagnostics CSV must contain sample_start and forecast_available columns.")
    by_start = test_diag.set_index("sample_start")
    missing = [int(i) for i in indices if int(i) not in by_start.index]
    if missing:
        raise ValueError(f"Prediction test indices are absent from diagnostics CSV: {missing[:5]}")
    available = by_start.loc[indices, "forecast_available"].to_numpy(dtype=float) > 0.0
    return {
        "all": np.ones(len(indices), dtype=bool),
        "forecast_available": available,
        "forecast_missing": ~available,
    }


def evaluate_run(run_name, predictions_dir, posthoc_dir, diagnostics_df, rain_mask, rain_top_mask):
    y_true, indices, models = load_prediction_models(predictions_dir, posthoc_dir=posthoc_dir)
    masks = subset_masks(indices, diagnostics_df)
    rows = []
    for subset_name, sample_mask in masks.items():
        for model_name, y_pred in models.items():
            rows.append(
                {
                    "run": run_name,
                    "model": model_name,
                    "subset": subset_name,
                    "samples": int(sample_mask.sum()),
                    "global_mae": mae(y_true, y_pred, sample_mask=sample_mask),
                    "rain_sensitive_mae": mae(y_true, y_pred, sample_mask=sample_mask, node_mask=rain_mask),
                    "rain_top_mae": mae(y_true, y_pred, sample_mask=sample_mask, node_mask=rain_top_mask),
                }
            )
    return rows, y_true.shape[2]


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


def write_report(path, rows_df):
    keep = [
        "run",
        "model",
        "subset",
        "samples",
        "global_mae",
        "rain_sensitive_mae",
        "rain_top_mae",
    ]
    lines = [
        "# Forecast Subset Evaluation",
        "",
        "Metrics are computed from persisted prediction artifacts, split by forecast availability for test samples.",
        "",
        markdown_table(rows_df[keep]),
        "",
        "Interpretation boundary: this report diagnoses existing predictions only; it does not retrain the model or prove forecast skill.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction artifacts by forecast availability subsets.")
    parser.add_argument(
        "--predictions",
        action="append",
        type=parse_named_path,
        required=True,
        help="Named prediction artifact directory, formatted as NAME=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--posthoc",
        action="append",
        type=parse_named_path,
        default=[],
        help="Optional named posthoc output directory, formatted as NAME=PATH. Name must match --predictions.",
    )
    parser.add_argument(
        "--forecast_diagnostics",
        default="output/forecast_context_diagnostics_cp180/forecast_context_by_sample.csv",
    )
    parser.add_argument(
        "--rain_susceptibility",
        default="output/rain_susceptibility/rain_susceptibility.csv",
    )
    parser.add_argument("--output_dir", default="output/forecast_subset_eval_cp180")
    args = parser.parse_args()

    diagnostics_df = pd.read_csv(args.forecast_diagnostics)
    posthoc_by_name = dict(args.posthoc)

    first_true, _, _ = load_prediction_models(args.predictions[0][1])
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, first_true.shape[2])

    rows = []
    for run_name, predictions_dir in args.predictions:
        run_rows, n_nodes = evaluate_run(
            run_name,
            predictions_dir,
            posthoc_by_name.get(run_name),
            diagnostics_df,
            rain_mask,
            rain_top_mask,
        )
        if n_nodes != first_true.shape[2]:
            raise ValueError(f"Run {run_name} has n_nodes={n_nodes}, expected {first_true.shape[2]}.")
        rows.extend(run_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(output_dir / "forecast_subset_metrics.csv", index=False)
    write_report(output_dir / "forecast_subset_evaluation.md", rows_df)
    print(f"Wrote forecast subset evaluation to {output_dir}")
    print(rows_df[["run", "model", "subset", "samples", "rain_sensitive_mae"]].to_string(index=False))


if __name__ == "__main__":
    main()
