import argparse
import os

import numpy as np


def metrics(y_true, y_pred, sample_mask=None, node_mask=None):
    true = y_true
    pred = y_pred
    if sample_mask is not None:
        true = true[sample_mask]
        pred = pred[sample_mask]
    if node_mask is not None:
        true = true[:, :, node_mask, :]
        pred = pred[:, :, node_mask, :]
    rmse = float(np.sqrt(np.mean(np.square(true - pred))))
    mae = float(np.mean(np.abs(true - pred)))
    return rmse, mae


def add_row(rows, name, y_true, y_pred, y_base, sample_mask=None, node_mask=None):
    if sample_mask is not None and int(sample_mask.sum()) == 0:
        return
    if node_mask is not None and int(node_mask.sum()) == 0:
        return
    model_rmse, model_mae = metrics(y_true, y_pred, sample_mask, node_mask)
    base_rmse, base_mae = metrics(y_true, y_base, sample_mask, node_mask)
    rows.append({
        "scope": name,
        "samples": int(sample_mask.sum()) if sample_mask is not None else int(y_true.shape[0]),
        "nodes": int(node_mask.sum()) if node_mask is not None else int(y_true.shape[2]),
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "persistence_rmse": base_rmse,
        "persistence_mae": base_mae,
        "rmse_gain_pct": 100.0 * (base_rmse - model_rmse) / max(base_rmse, 1e-6),
        "mae_gain_pct": 100.0 * (base_mae - model_mae) / max(base_mae, 1e-6),
    })


def main():
    parser = argparse.ArgumentParser(description="Evaluate rainfall-window and rain-susceptible-node prediction gains.")
    parser.add_argument("--processed_npz", default=os.path.join("dataset", "processed", "preprocessed_data.npz"))
    parser.add_argument("--pred", default=os.path.join("output", "y_test_pred_real.npy"))
    parser.add_argument("--true", default=os.path.join("output", "y_test_true_real.npy"))
    parser.add_argument("--persistence", default=os.path.join("output", "y_test_persistence_real.npy"))
    parser.add_argument("--output_csv", default=os.path.join("output", "rainfall_focus_metrics.csv"))
    args = parser.parse_args()

    pack = np.load(args.processed_npz, allow_pickle=True)
    y_pred = np.load(args.pred)
    y_true = np.load(args.true)
    y_base = np.load(args.persistence)
    rain = pack["test_rain"]
    rain_cols = [str(c) for c in pack["rain_feature_cols"]]
    susceptibility = pack["rain_susceptibility"].astype(np.float32)

    event_cols = [i for i, col in enumerate(rain_cols) if col.startswith("rain_event_p9")]
    event_samples = rain[:, :, event_cols].max(axis=(1, 2)) > 0 if event_cols else np.zeros(y_true.shape[0], dtype=bool)
    active_nodes = susceptibility > 0
    top_nodes = np.zeros_like(active_nodes, dtype=bool)
    if active_nodes.any():
        top_nodes = susceptibility >= np.quantile(susceptibility[active_nodes], 0.60)

    rows = []
    add_row(rows, "global", y_true, y_pred, y_base)
    add_row(rows, "rain_event_samples", y_true, y_pred, y_base, sample_mask=event_samples)
    add_row(rows, "rain_active_nodes", y_true, y_pred, y_base, node_mask=active_nodes)
    add_row(rows, "rain_top_nodes", y_true, y_pred, y_base, node_mask=top_nodes)
    add_row(rows, "event_active_nodes", y_true, y_pred, y_base, sample_mask=event_samples, node_mask=active_nodes)
    add_row(rows, "event_top_nodes", y_true, y_pred, y_base, sample_mask=event_samples, node_mask=top_nodes)

    import pandas as pd
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f">> Saved rainfall focus metrics: {args.output_csv}")


if __name__ == "__main__":
    main()
