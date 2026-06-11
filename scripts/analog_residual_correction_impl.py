import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.common import prediction_artifacts as artifacts
from scripts.posthoc_mechanism_moe_impl import (
    FEATURE_GROUPS,
    forecast_gate_features,
    load_node_masks,
    load_prediction_pack,
    mae,
    sample_features,
    standardize,
)
from data_loader.date_loader import load_and_clean_data
from scripts.cpd_split_validate_impl import event_context_for_starts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply validation-only analog residual correction using mechanism-state nearest neighbors."
    )
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument(
        "--multiscale_features",
        default="output/mechanism_moe_multiscale/multiscale_response_features.csv",
    )
    parser.add_argument("--forecast_context_by_sample", default=None)
    parser.add_argument("--event_context_features", default=None)
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--scope", default="rain_sensitive_nodes")
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--shrink", type=float, default=0.25)
    parser.add_argument("--auto_select", action="store_true")
    parser.add_argument("--candidate_neighbors", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--candidate_shrink", type=float, nargs="+", default=[0.0, 0.10, 0.25, 0.50, 0.75, 0.90])
    parser.add_argument("--candidate_distance_quantiles", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.50, 1.0])
    parser.add_argument("--max_abs_correction", type=float, default=0.50)
    parser.add_argument("--feature_group", choices=sorted(FEATURE_GROUPS), default="all")
    parser.add_argument("--apply_scope", choices=("all", "rain_sensitive", "rain_top"), default="rain_sensitive")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def build_feature_frames(args, pack):
    val_features, feature_cols = sample_features(
        args.multiscale_features,
        args.scope,
        pack["val_indices"],
        args.n_his,
        args.n_pred,
        args.feature_group,
    )
    test_features, _ = sample_features(
        args.multiscale_features,
        args.scope,
        pack["test_indices"],
        args.n_his,
        args.n_pred,
        args.feature_group,
    )
    train_features, _ = sample_features(
        args.multiscale_features,
        args.scope,
        pack.get("train_indices", []),
        args.n_his,
        args.n_pred,
        args.feature_group,
    ) if "train_indices" in pack else (pd.DataFrame(), [])

    val_x = val_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test_x = test_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train_x = train_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if not train_features.empty else pd.DataFrame()

    val_forecast_x, val_forecast_cols = forecast_gate_features(args.forecast_context_by_sample, pack["val_indices"])
    test_forecast_x, test_forecast_cols = forecast_gate_features(args.forecast_context_by_sample, pack["test_indices"])
    train_forecast_x, train_forecast_cols = forecast_gate_features(args.forecast_context_by_sample, pack.get("train_indices", [])) if "train_indices" in pack else (pd.DataFrame(), [])

    if val_forecast_cols or test_forecast_cols:
        if val_forecast_cols != test_forecast_cols:
            raise ValueError("Validation and test forecast feature columns do not match.")
        val_x = pd.concat([val_x.reset_index(drop=True), val_forecast_x.reset_index(drop=True)], axis=1)
        test_x = pd.concat([test_x.reset_index(drop=True), test_forecast_x.reset_index(drop=True)], axis=1)
        if not train_x.empty:
            train_x = pd.concat([train_x.reset_index(drop=True), train_forecast_x.reset_index(drop=True)], axis=1)
        feature_cols = feature_cols + val_forecast_cols

    if args.event_context_features:
        _, _, _, time_cols, _ = load_and_clean_data(args.file_path)
        val_event = event_context_for_starts(args.event_context_features, time_cols, pack["val_indices"], args.n_his)
        test_event = event_context_for_starts(args.event_context_features, time_cols, pack["test_indices"], args.n_his)
        
        val_event = val_event.apply(pd.to_numeric, errors="coerce").fillna(0.0).add_prefix("eventctx_")
        test_event = test_event.apply(pd.to_numeric, errors="coerce").fillna(0.0).add_prefix("eventctx_")
        
        val_x = pd.concat([val_x.reset_index(drop=True), val_event.reset_index(drop=True)], axis=1)
        test_x = pd.concat([test_x.reset_index(drop=True), test_event.reset_index(drop=True)], axis=1)
        
        if not train_x.empty:
            train_event = event_context_for_starts(args.event_context_features, time_cols, pack["train_indices"], args.n_his)
            train_event = train_event.apply(pd.to_numeric, errors="coerce").fillna(0.0).add_prefix("eventctx_")
            train_x = pd.concat([train_x.reset_index(drop=True), train_event.reset_index(drop=True)], axis=1)
            
        feature_cols = feature_cols + list(val_event.columns)
        
    return val_x, test_x, train_x, feature_cols


def causal_nearest_neighbor_weights(pool_x, query_x, pool_times, query_times, neighbors):
    pool_arr = pool_x.to_numpy(dtype=np.float64)
    query_arr = query_x.to_numpy(dtype=np.float64)
    pool_arr, query_arr = standardize(pool_arr, query_arr)
    
    rows = []
    weights = []
    mean_distances = []
    
    for row_idx, row in enumerate(query_arr):
        q_time = query_times[row_idx]
        
        # Valid pool indices: strictly before query time
        valid_mask = pool_times < q_time
        valid_indices = np.where(valid_mask)[0]
        
        target_k = int(neighbors)
        
        if len(valid_indices) == 0:
            # Fallback if no history available
            rows.append(np.zeros(target_k, dtype=np.int32))
            weights.append(np.zeros(target_k, dtype=np.float32))
            mean_distances.append(np.inf)
            continue
            
        k = min(target_k, len(valid_indices))
        
        valid_pool = pool_arr[valid_indices]
        dist = np.linalg.norm(valid_pool - row.reshape(1, -1), axis=1)
        
        nn_local = np.argsort(dist)[:k]
        nn_global = valid_indices[nn_local]
        
        inv = 1.0 / np.maximum(dist[nn_local], 1e-6)
        inv = inv / np.maximum(inv.sum(), 1e-12)
        
        # Pad if fewer than target_k neighbors
        if k < target_k:
            pad_size = target_k - k
            nn_global = np.pad(nn_global, (0, pad_size), 'constant', constant_values=nn_global[-1])
            inv = np.pad(inv, (0, pad_size), 'constant', constant_values=0.0)
            
        rows.append(nn_global.astype(np.int32))
        weights.append(inv.astype(np.float32))
        mean_distances.append(float(np.mean(dist[nn_local])))
        
    return np.vstack(rows), np.vstack(weights), np.asarray(mean_distances, dtype=np.float32)


def correction_mask(scope, rain_mask, rain_top_mask, n_nodes):
    if scope == "all":
        return np.ones(n_nodes, dtype=bool)
    if scope == "rain_top":
        return rain_top_mask
    return rain_mask


def apply_analog_residual(
    base_pred,
    val_residual,
    neighbor_idx,
    neighbor_weight,
    shrink,
    max_abs_correction,
    apply_mask,
    sample_gate=None,
):
    corrected = base_pred.copy()
    if sample_gate is None:
        sample_gate = np.ones(corrected.shape[0], dtype=bool)
    for sample_idx in range(corrected.shape[0]):
        if not bool(sample_gate[sample_idx]):
            continue
        residual = np.sum(
            val_residual[neighbor_idx[sample_idx]] * neighbor_weight[sample_idx, :, None, None, None],
            axis=0,
        )
        residual = float(shrink) * residual
        residual = np.clip(residual, -abs(max_abs_correction), abs(max_abs_correction))
        sample_pred = corrected[sample_idx]
        sample_pred[:, apply_mask, :] = sample_pred[:, apply_mask, :] + residual[:, apply_mask, :]
        corrected[sample_idx] = sample_pred
    return corrected.astype(np.float32)


def masked_mae(y_true, y_pred, apply_mask):
    return float(np.mean(np.abs(y_true[:, :, apply_mask, :] - y_pred[:, :, apply_mask, :])))


def auto_select_params(pack, global_x, global_times, val_x, val_times, global_residual, apply_mask, args):
    rows = []
    best = None
    for neighbors in args.candidate_neighbors:
        neighbor_idx, neighbor_weight, mean_distance = causal_nearest_neighbor_weights(
            global_x,
            val_x,
            global_times,
            val_times,
            neighbors,
        )
        for shrink in args.candidate_shrink:
            for quantile in args.candidate_distance_quantiles:
                q = float(np.clip(quantile, 0.0, 1.0))
                threshold = float(np.quantile(mean_distance[mean_distance != np.inf], q)) if np.any(mean_distance != np.inf) else np.inf
                sample_gate = mean_distance <= threshold
                corrected = apply_analog_residual(
                    pack["val_selected"],
                    global_residual,
                    neighbor_idx,
                    neighbor_weight,
                    shrink,
                    args.max_abs_correction,
                    apply_mask,
                    sample_gate=sample_gate,
                )
                score = masked_mae(pack["val_true"], corrected, apply_mask)
                row = {
                    "neighbors": int(neighbors),
                    "shrink": float(shrink),
                    "distance_quantile": q,
                    "distance_threshold": threshold,
                    "corrected_samples": int(np.sum(sample_gate)),
                    "val_apply_scope_mae": score,
                }
                rows.append(row)
                if best is None or score < best["val_apply_scope_mae"]:
                    best = row
    return best, pd.DataFrame(rows).sort_values("val_apply_scope_mae").reset_index(drop=True)


def analog_correct(pack, args, apply_mask):
    val_x, test_x, train_x, feature_cols = build_feature_frames(args, pack)
    
    # Build global pool to strictly enforce causal timeline
    pool_frames = []
    pool_residuals = []
    pool_times = []
    
    if "train_true" in pack and "train_selected" in pack and not train_x.empty:
        pool_frames.append(train_x)
        pool_residuals.append(pack["train_true"] - pack["train_selected"])
        pool_times.append(np.array(pack["train_indices"]))
        
    pool_frames.append(val_x)
    pool_residuals.append(pack["val_true"] - pack["val_selected"])
    pool_times.append(np.array(pack["val_indices"]))
    
    global_x = pd.concat(pool_frames, axis=0).reset_index(drop=True)
    global_residual = np.concatenate(pool_residuals, axis=0)
    global_times = np.concatenate(pool_times, axis=0)
    
    val_times = np.array(pack["val_indices"])
    test_times = np.array(pack["test_indices"])
    train_times = np.array(pack["train_indices"]) if "train_indices" in pack else np.array([])
    
    selection_df = pd.DataFrame()
    neighbors = args.neighbors
    shrink = args.shrink
    distance_threshold = np.inf
    if args.auto_select:
        best, selection_df = auto_select_params(pack, global_x, global_times, val_x, val_times, global_residual, apply_mask, args)
        neighbors = int(best["neighbors"])
        shrink = float(best["shrink"])
        distance_threshold = float(best["distance_threshold"])
        
    test_neighbor_idx, test_neighbor_weight, test_mean_distance = causal_nearest_neighbor_weights(
        global_x, test_x, global_times, test_times, neighbors
    )
    test_sample_gate = test_mean_distance <= distance_threshold
    test_corrected = apply_analog_residual(
        pack["test_selected"],
        global_residual,
        test_neighbor_idx,
        test_neighbor_weight,
        shrink,
        args.max_abs_correction,
        apply_mask,
        sample_gate=test_sample_gate,
    )
    
    val_neighbor_idx, val_neighbor_weight, val_mean_distance = causal_nearest_neighbor_weights(
        global_x, val_x, global_times, val_times, neighbors
    )
    val_sample_gate = val_mean_distance <= distance_threshold
    val_corrected = apply_analog_residual(
        pack["val_selected"],
        global_residual,
        val_neighbor_idx,
        val_neighbor_weight,
        shrink,
        args.max_abs_correction,
        apply_mask,
        sample_gate=val_sample_gate,
    )
    
    train_corrected = pack["train_selected"].copy() if "train_selected" in pack else None
    if not train_x.empty and "train_selected" in pack:
        train_neighbor_idx, train_neighbor_weight, train_mean_distance = causal_nearest_neighbor_weights(
            global_x, train_x, global_times, train_times, neighbors
        )
        train_sample_gate = train_mean_distance <= distance_threshold
        train_corrected = apply_analog_residual(
            pack["train_selected"],
            global_residual,
            train_neighbor_idx,
            train_neighbor_weight,
            shrink,
            args.max_abs_correction,
            apply_mask,
            sample_gate=train_sample_gate,
        )

    return (
        test_corrected,
        val_corrected,
        train_corrected,
        test_neighbor_idx,
        test_neighbor_weight,
        test_mean_distance,
        test_sample_gate,
        feature_cols,
        neighbors,
        shrink,
        distance_threshold,
        selection_df,
    )


def write_report(path, rows, feature_cols, selected_neighbors, selected_shrink, distance_threshold, corrected_samples, args):
    lines = [
        "# Analog Residual Correction",
        "",
        "This command applies validation-only nearest-neighbor residual correction in mechanism feature space.",
        "It is a post-training bridge toward distillation, not CPD-STGCN retraining.",
        "",
        "## Configuration",
        "",
        f"- predictions_dir: `{args.predictions_dir}`",
        f"- feature_group: `{args.feature_group}`",
        f"- forecast_context_by_sample: `{args.forecast_context_by_sample}`",
        f"- event_context_features: `{args.event_context_features}`",
        f"- neighbors: `{args.neighbors}`",
        f"- shrink: `{args.shrink}`",
        f"- auto_select: `{args.auto_select}`",
        f"- selected_neighbors: `{selected_neighbors}`",
        f"- selected_shrink: `{selected_shrink}`",
        f"- selected_distance_threshold: `{distance_threshold}`",
        f"- corrected_samples: `{corrected_samples}`",
        f"- max_abs_correction: `{args.max_abs_correction}`",
        f"- apply_scope: `{args.apply_scope}`",
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
    lines.extend(["", "## Features", ""])
    for col in feature_cols:
        lines.append(f"- `{col}`")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    pack = load_prediction_pack(args.predictions_dir)
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, pack["test_true"].shape[2])
    apply_mask = correction_mask(args.apply_scope, rain_mask, rain_top_mask, pack["test_true"].shape[2])
    (
        test_corrected,
        val_corrected,
        train_corrected,
        neighbor_idx,
        neighbor_weight,
        mean_distance,
        sample_gate,
        feature_cols,
        selected_neighbors,
        selected_shrink,
        distance_threshold,
        selection_df,
    ) = analog_correct(
        pack,
        args,
        apply_mask,
    )
    rows = [
        {
            "model": "selected_model",
            "global_mae": mae(pack["test_true"], pack["test_selected"]),
            "rain_sensitive_mae": mae(pack["test_true"], pack["test_selected"], rain_mask),
            "rain_top_mae": mae(pack["test_true"], pack["test_selected"], rain_top_mask),
        },
        {
            "model": "analog_residual_corrected",
            "global_mae": mae(pack["test_true"], test_corrected),
            "rain_sensitive_mae": mae(pack["test_true"], test_corrected, rain_mask),
            "rain_top_mae": mae(pack["test_true"], test_corrected, rain_top_mask),
        },
        {
            "model": "persistence",
            "global_mae": mae(pack["test_true"], pack["test_persistence"]),
            "rain_sensitive_mae": mae(pack["test_true"], pack["test_persistence"], rain_mask),
            "rain_top_mae": mae(pack["test_true"], pack["test_persistence"], rain_top_mask),
        },
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "test_pred_real.npy", test_corrected)
    np.save(output_dir / "val_pred_real.npy", val_corrected)
    if train_corrected is not None:
        np.save(output_dir / "train_pred_real.npy", train_corrected)
        
    np.save(output_dir / "analog_residual_corrected_test_pred_real.npy", test_corrected)
    pd.DataFrame(rows).to_csv(output_dir / "analog_residual_correction_metrics.csv", index=False)
    if not selection_df.empty:
        selection_df.to_csv(output_dir / "analog_residual_auto_selection.csv", index=False)
    neighbor_df = pd.DataFrame(neighbor_idx, columns=[f"neighbor_{idx}" for idx in range(neighbor_idx.shape[1])])
    for idx in range(neighbor_weight.shape[1]):
        neighbor_df[f"weight_{idx}"] = neighbor_weight[:, idx]
    neighbor_df["mean_neighbor_distance"] = mean_distance
    neighbor_df["corrected_sample"] = sample_gate.astype(np.int32)
    neighbor_df.insert(0, "sample_index", np.arange(neighbor_idx.shape[0], dtype=np.int32))
    neighbor_df.to_csv(output_dir / "analog_residual_neighbors.csv", index=False)
    write_report(
        output_dir / "analog_residual_correction_report.md",
        rows,
        feature_cols,
        selected_neighbors,
        selected_shrink,
        distance_threshold,
        int(np.sum(sample_gate)),
        args,
    )

    print(pd.DataFrame(rows).to_string(index=False))
    print(f">> Saved analog residual correction outputs to: {output_dir}")


if __name__ == "__main__":
    main()
