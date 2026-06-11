import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.common import prediction_artifacts as artifacts  # noqa: E402


DEFAULT_FEATURES = [
    "stage_id",
    "near_cpd",
    "since_last_cp",
    "until_next_cp",
    "long_trend",
    "rain_disturbance_z",
    "short_shock_z",
    "gate_hydro_dry_to_wet_transition_lag6",
    "gate_rain_event_p90_15d_lag6",
    "gate_rain_30d_lag6",
    "gate_hydro_memory_index_lag6",
]

FEATURE_GROUPS = {
    "all": DEFAULT_FEATURES,
    "stage_only": ["stage_id", "near_cpd", "since_last_cp", "until_next_cp"],
    "response_only": ["long_trend", "rain_disturbance_z", "short_shock_z"],
    "rain_gate_only": [
        "gate_hydro_dry_to_wet_transition_lag6",
        "gate_rain_event_p90_15d_lag6",
        "gate_rain_30d_lag6",
        "gate_hydro_memory_index_lag6",
    ],
    "no_rain_gate": [
        "stage_id",
        "near_cpd",
        "since_last_cp",
        "until_next_cp",
        "long_trend",
        "rain_disturbance_z",
        "short_shock_z",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc mechanism MoE over persisted fold prediction artifacts.")
    parser.add_argument(
        "--predictions_dir",
        default=os.path.join(project_root, "output", "mechanism_moe_prediction_artifact_smoke", "cp_180", "predictions"),
    )
    parser.add_argument(
        "--multiscale_features",
        default=os.path.join(project_root, "output", "mechanism_moe_multiscale", "multiscale_response_features.csv"),
    )
    parser.add_argument(
        "--rain_susceptibility",
        default=os.path.join(project_root, "output", "rain_susceptibility", "rain_susceptibility.csv"),
    )
    parser.add_argument(
        "--forecast_context_by_sample",
        default=None,
        help="Optional forecast_context diagnostics CSV; numeric forecast_* columns are appended to MoE gate features.",
    )
    parser.add_argument("--scope", default="rain_sensitive_nodes")
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--feature_group", choices=sorted(FEATURE_GROUPS), default="all")
    parser.add_argument("--target_split", choices=["test", "val", "train"], default="test")
    parser.add_argument(
        "--global_gate_weight",
        type=float,
        default=0.50,
        help="Blend weight for global validation-set expert reliability; the remainder uses local mechanism neighbors.",
    )
    parser.add_argument("--output_dir", default=os.path.join(project_root, "output", "mechanism_moe_posthoc"))
    return parser.parse_args()


def load_prediction_pack(predictions_dir):
    pred_dir = artifacts.validate_predictions_dir(os.path.abspath(predictions_dir))
    pack = {
        "val_true": artifacts.load_array(pred_dir, artifacts.VAL_TRUE_REAL),
        "val_selected": artifacts.load_array(pred_dir, artifacts.VAL_PRED_REAL),
        "val_raw": artifacts.load_array(pred_dir, artifacts.VAL_PRED_RAW_REAL),
        "val_cal": artifacts.load_array(pred_dir, artifacts.VAL_PRED_CAL_REAL),
        "val_persistence": artifacts.load_array(pred_dir, artifacts.VAL_PERSISTENCE_REAL),
        "test_true": artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL),
        "test_selected": artifacts.load_array(pred_dir, artifacts.TEST_PRED_REAL),
        "test_raw": artifacts.load_array(pred_dir, artifacts.TEST_PRED_RAW_REAL),
        "test_cal": artifacts.load_array(pred_dir, artifacts.TEST_PRED_CAL_REAL),
        "test_persistence": artifacts.load_array(pred_dir, artifacts.TEST_PERSISTENCE_REAL),
        "val_indices": artifacts.load_array(pred_dir, artifacts.VAL_INDICES).astype(np.int32),
        "test_indices": artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32),
    }
    train_missing = artifacts.missing_required_files(pred_dir, artifacts.OPTIONAL_TRAIN_ARRAYS)
    if not train_missing:
        pack.update(
            {
                "train_true": artifacts.load_array(pred_dir, artifacts.TRAIN_TRUE_REAL),
                "train_selected": artifacts.load_array(pred_dir, artifacts.TRAIN_PRED_REAL),
                "train_raw": artifacts.load_array(pred_dir, artifacts.TRAIN_PRED_RAW_REAL),
                "train_cal": artifacts.load_array(pred_dir, artifacts.TRAIN_PRED_CAL_REAL),
                "train_persistence": artifacts.load_array(pred_dir, artifacts.TRAIN_PERSISTENCE_REAL),
                "train_indices": artifacts.load_array(pred_dir, artifacts.TRAIN_INDICES).astype(np.int32),
            }
        )
    return pack


def load_node_masks(path, n_nodes, top_quantile=0.60):
    if not path or not os.path.exists(path):
        active = np.ones(n_nodes, dtype=bool)
        return active, active
    df = pd.read_csv(path)
    susceptibility = df["rain_susceptibility"].to_numpy(dtype=np.float32)
    if susceptibility.shape[0] != n_nodes:
        raise ValueError(f"rain_susceptibility length={susceptibility.shape[0]} != n_nodes={n_nodes}")
    active = susceptibility > 0
    top = np.zeros_like(active, dtype=bool)
    if active.any():
        top = susceptibility >= float(np.quantile(susceptibility[active], top_quantile))
    return active, top


def expert_stack(pack, split):
    names = ["selected_model", "raw_model", "node_calibrated", "persistence"]
    arrays = [
        pack[f"{split}_selected"],
        pack[f"{split}_raw"],
        pack[f"{split}_cal"],
        pack[f"{split}_persistence"],
    ]
    # Remove exact duplicates to keep the gate interpretable.
    unique_names = []
    unique_arrays = []
    for name, arr in zip(names, arrays):
        if any(np.allclose(arr, existing) for existing in unique_arrays):
            continue
        unique_names.append(name)
        unique_arrays.append(arr)
    return unique_names, np.stack(unique_arrays, axis=0)


def mae(y_true, y_pred, node_mask=None):
    true = y_true
    pred = y_pred
    if node_mask is not None:
        true = true[:, :, node_mask, :]
        pred = pred[:, :, node_mask, :]
    return float(np.mean(np.abs(true - pred)))


def sample_expert_mae(y_true, experts, node_mask=None):
    if node_mask is not None:
        y_true = y_true[:, :, node_mask, :]
        experts = experts[:, :, :, node_mask, :]
    return np.mean(np.abs(experts - y_true[None, ...]), axis=(2, 3, 4)).T


def sample_features(multiscale_path, scope, starts, n_his, n_pred, feature_group):
    df = pd.read_csv(multiscale_path)
    scoped = df[df["scope"] == scope].copy()
    if scoped.empty:
        scoped = df[df["scope"] == "all_nodes"].copy()
    scoped = scoped.set_index("interval_index")
    rows = []
    for start in starts:
        interval_idx = int(start) + int(n_his) + int(n_pred) - 2
        if interval_idx not in scoped.index:
            interval_idx = int(scoped.index.max())
        rows.append(scoped.loc[interval_idx])
    features = pd.DataFrame(rows).reset_index(drop=True)
    requested = FEATURE_GROUPS.get(feature_group, DEFAULT_FEATURES)
    available = [c for c in requested if c in features.columns]
    return features, available


def forecast_gate_features(path, starts):
    if not path:
        return None, []
    df = pd.read_csv(path)
    if "sample_start" not in df.columns:
        raise ValueError("forecast_context_by_sample must contain sample_start.")
    forecast_cols = [
        c
        for c in df.columns
        if c.startswith("forecast_") and c not in ("forecast_available_pct",)
    ]
    forecast_cols = [c for c in forecast_cols if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    if not forecast_cols:
        return pd.DataFrame(index=np.arange(len(starts))), []
    by_start = df.drop_duplicates("sample_start").set_index("sample_start")
    missing = [int(start) for start in starts if int(start) not in by_start.index]
    if missing:
        raise ValueError(f"forecast_context_by_sample is missing sample_start values: {missing[:5]}")
    rows = by_start.loc[np.asarray(starts, dtype=np.int32), forecast_cols].reset_index(drop=True)
    rows = rows.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    rows = rows.add_prefix("ctx_")
    return rows, list(rows.columns)


def standardize(train_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def softmax(scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores / max(float(temperature), 1e-6))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def build_gates(val_features, test_features, val_losses, neighbors, temperature, global_gate_weight):
    val_x = val_features.to_numpy(dtype=np.float64)
    test_x = test_features.to_numpy(dtype=np.float64)
    val_x, test_x = standardize(val_x, test_x)
    global_loss = val_losses.mean(axis=0)
    global_scores = -global_loss.reshape(1, -1)
    global_gate = np.repeat(softmax(global_scores, temperature), test_x.shape[0], axis=0)
    global_weight = float(np.clip(global_gate_weight, 0.0, 1.0))

    gates = []
    k = max(1, min(int(neighbors), val_x.shape[0]))
    for row in test_x:
        dist = np.linalg.norm(val_x - row.reshape(1, -1), axis=1)
        nn = np.argsort(dist)[:k]
        local_loss = val_losses[nn].mean(axis=0)
        local_gate = softmax(-local_loss.reshape(1, -1), temperature)[0]
        gates.append(global_weight * global_gate[0] + (1.0 - global_weight) * local_gate)
    gates = np.asarray(gates, dtype=np.float32)
    gates = gates / np.maximum(gates.sum(axis=1, keepdims=True), 1e-12)
    return gates.astype(np.float32)


def combine_experts(experts, gates):
    return np.einsum("se,es...->s...", gates, experts)


def write_report(path, metrics, expert_names, gates, feature_cols, args):
    mean_gate = gates.mean(axis=0)
    dominant = np.argmax(gates, axis=1)
    lines = [
        "# Post-hoc Mechanism MoE Report",
        "",
        "This is a local prototype using persisted fold prediction artifacts.",
        "It tests whether mechanism gate features can choose among available prediction experts.",
        "",
        "## Configuration",
        "",
        f"- predictions_dir: `{args.predictions_dir}`",
        f"- scope: `{args.scope}`",
        f"- neighbors: `{args.neighbors}`",
        f"- temperature: `{args.temperature}`",
        f"- global_gate_weight: `{args.global_gate_weight}`",
        f"- feature_group: `{args.feature_group}`",
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
    lines.extend([
        "",
        "## Mean Gate Weights",
        "",
        "| expert | mean_weight | dominant_samples |",
        "| --- | ---: | ---: |",
    ])
    for idx, name in enumerate(expert_names):
        lines.append(f"| {name} | {mean_gate[idx]:.3f} | {int(np.sum(dominant == idx))} |")
    lines.extend([
        "",
        "## Gate Features",
        "",
    ])
    for col in feature_cols:
        lines.append(f"- `{col}`")
    lines.extend([
        "",
        "## Boundary",
        "",
        "- This run is post-hoc and does not change the CPD-STGCN model.",
        "- If the input predictions come from a 1 epoch smoke run, use this only to verify plumbing and interpretation outputs.",
        "- Formal claims require full training artifacts and multi-seed validation.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pack = load_prediction_pack(args.predictions_dir)
    target_split = args.target_split
    required_target = f"{target_split}_true"
    if required_target not in pack:
        raise FileNotFoundError(
            f"Prediction artifacts do not contain {target_split} arrays. "
            "Regenerate them with scripts/cpd.py validate --save_fold_predictions."
        )
    n_nodes = pack[required_target].shape[2]
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, n_nodes)

    val_names, val_experts = expert_stack(pack, "val")
    target_names, target_experts = expert_stack(pack, target_split)
    if val_names != target_names:
        raise ValueError(f"Validation and {target_split} expert names do not match.")

    val_feature_df, feature_cols = sample_features(
        args.multiscale_features,
        args.scope,
        pack["val_indices"],
        args.n_his,
        args.n_pred,
        args.feature_group,
    )
    target_feature_df, _ = sample_features(
        args.multiscale_features,
        args.scope,
        pack[f"{target_split}_indices"],
        args.n_his,
        args.n_pred,
        args.feature_group,
    )
    val_x = val_feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target_x = target_feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    val_forecast_x, val_forecast_cols = forecast_gate_features(args.forecast_context_by_sample, pack["val_indices"])
    target_forecast_x, target_forecast_cols = forecast_gate_features(
        args.forecast_context_by_sample,
        pack[f"{target_split}_indices"],
    )
    if val_forecast_cols or target_forecast_cols:
        if val_forecast_cols != target_forecast_cols:
            raise ValueError(f"Validation and {target_split} forecast gate feature columns do not match.")
        val_x = pd.concat([val_x.reset_index(drop=True), val_forecast_x.reset_index(drop=True)], axis=1)
        target_x = pd.concat([target_x.reset_index(drop=True), target_forecast_x.reset_index(drop=True)], axis=1)
        feature_cols = feature_cols + val_forecast_cols
    val_losses = sample_expert_mae(pack["val_true"], val_experts, node_mask=rain_mask)
    gates = build_gates(val_x, target_x, val_losses, args.neighbors, args.temperature, args.global_gate_weight)
    moe_pred = combine_experts(target_experts, gates)

    rows = []
    for name, pred in zip(target_names, target_experts):
        rows.append({
            "model": name,
            "global_mae": mae(pack[required_target], pred),
            "rain_sensitive_mae": mae(pack[required_target], pred, rain_mask),
            "rain_top_mae": mae(pack[required_target], pred, rain_top_mask),
        })
    rows.append({
        "model": "posthoc_mechanism_moe",
        "global_mae": mae(pack[required_target], moe_pred),
        "rain_sensitive_mae": mae(pack[required_target], moe_pred, rain_mask),
        "rain_top_mae": mae(pack[required_target], moe_pred, rain_top_mask),
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "moe_test_pred_real.npy" if target_split == "test" else f"moe_{target_split}_pred_real.npy"
    np.save(output_dir / output_name, moe_pred.astype(np.float32))
    pd.DataFrame(rows).to_csv(output_dir / "moe_metrics.csv", index=False, encoding="utf-8")
    gate_df = pd.DataFrame(gates, columns=[f"weight_{name}" for name in target_names])
    gate_df.insert(0, "sample_index", np.arange(gates.shape[0], dtype=np.int32))
    gate_df["dominant_expert"] = [target_names[idx] for idx in np.argmax(gates, axis=1)]
    gate_df.to_csv(output_dir / "moe_gate_weights.csv", index=False, encoding="utf-8")
    target_x.to_csv(output_dir / "moe_gate_features.csv", index=False, encoding="utf-8")
    write_report(output_dir / "posthoc_mechanism_moe_report.md", rows, target_names, gates, feature_cols, args)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved post-hoc MoE outputs to: {output_dir}")


if __name__ == "__main__":
    main()
