import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.date_loader import load_and_clean_data
from scripts.common import prediction_artifacts as artifacts
from scripts.cpd_split_validate_impl import event_context_for_starts
from scripts.posthoc_mechanism_moe_impl import load_node_masks, mae
from scripts.source_switch_posthoc_moe_impl import source_switch_mask


def parse_seeds(value):
    seeds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise argparse.ArgumentTypeError("Expected at least one seed.")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-search event-conditioned source-switch posthoc MoE configs."
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,7,23"))
    parser.add_argument(
        "--prediction_dir_template",
        default="output/forecast_gefs15_climfallback_flag_cp180_seed{seed}/cp_180/predictions",
    )
    parser.add_argument(
        "--posthoc_grid_dir",
        default="output/forecast_gefs15_climfallback_flag_posthoc_grid",
    )
    parser.add_argument(
        "--source_pair_grid",
        default="output/forecast_gefs15_climfallback_flag_source_switch_grid_repro/source_switch_pair_grid.csv",
    )
    parser.add_argument(
        "--forecast_context_by_sample",
        default="output/forecast_context_diagnostics_cp180_climfallback_flag/forecast_context_by_sample.csv",
    )
    parser.add_argument("--event_context_features", default="output/rainfall_event_catalog/rainfall_event_catalog.csv")
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--rain_susceptibility", default="output/rain_susceptibility/rain_susceptibility.csv")
    parser.add_argument("--source_col", default="forecast_is_fallback")
    parser.add_argument("--event_col", default="event_historical_trigger_probability")
    parser.add_argument("--event_threshold", type=float, default=0.25)
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--candidate_pairs", type=int, default=30)
    parser.add_argument("--max_official_configs", type=int, default=12)
    parser.add_argument("--max_fallback_configs", type=int, default=12)
    parser.add_argument("--target_split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--continuous_gate", action="store_true")
    parser.add_argument("--gate_coef_prob", type=float, default=10.0)
    parser.add_argument("--gate_coef_wetness", type=float, default=0.0)
    parser.add_argument("--gate_bias", type=float, default=-2.5)
    parser.add_argument("--event_calibrator_path", default=None, help="Path to event calibrator JSON model.")
    parser.add_argument("--output_dir", default="output/forecast_gefs15_climfallback_flag_event_source_switch_grid")
    parser.add_argument("--top_n", type=int, default=20)
    return parser.parse_args()


def config_dirs(posthoc_grid_dir, seed):
    root = Path(posthoc_grid_dir)
    prefix = f"seed{seed}_"
    dirs = {}
    for path in sorted(root.glob(f"{prefix}*")):
        if path.is_dir() and (path / "moe_test_pred_real.npy").exists():
            dirs[path.name[len(prefix) :]] = path
    if not dirs:
        raise FileNotFoundError(f"No posthoc prediction directories found under {root} for seed={seed}.")
    return dirs


def candidate_configs(args, common_configs):
    pair_path = Path(args.source_pair_grid)
    if pair_path.exists():
        pair_df = pd.read_csv(pair_path).head(args.candidate_pairs)
        official_values = list(pair_df.get("primary_config", pd.Series(dtype=str)).astype(str))
        fallback_values = list(pair_df.get("fallback_config", pd.Series(dtype=str)).astype(str))
    else:
        official_values = list(common_configs)
        fallback_values = list(common_configs)

    def select(values, limit):
        selected = []
        for value in values:
            if value in common_configs and value not in selected:
                selected.append(value)
            if len(selected) >= limit:
                break
        return selected

    official = select(official_values, args.max_official_configs)
    fallback = select(fallback_values, args.max_fallback_configs)
    if not official:
        official = list(common_configs)[: args.max_official_configs]
    if not fallback:
        fallback = list(common_configs)[: args.max_fallback_configs]
    return official, fallback


def all_configs(*groups):
    selected = []
    for values in groups:
        for value in values:
            if value not in selected:
                selected.append(value)
    return selected


def event_mask_for_indices(args, indices):
    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    starts = np.arange(0, raw_seq.shape[0] - args.n_his - args.n_pred + 1, dtype=np.int32)
    event_df = event_context_for_starts(args.event_context_features, time_cols, starts, args.n_his)
    if args.event_col not in event_df.columns:
        raise ValueError(f"event_context is missing column: {args.event_col}")
    values = event_df.iloc[np.asarray(indices, dtype=np.int32)][args.event_col].to_numpy(dtype=float)
    if args.event_threshold > 0.0:
        return values >= float(args.event_threshold)
    return values > 0.0


def load_event_feature_array(args, indices, col):
    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    starts = np.arange(0, raw_seq.shape[0] - args.n_his - args.n_pred + 1, dtype=np.int32)
    event_df = event_context_for_starts(args.event_context_features, time_cols, starts, args.n_his)
    if col not in event_df.columns:
        return np.zeros(len(indices), dtype=np.float32)
    values = event_df.iloc[np.asarray(indices, dtype=np.int32)][col].to_numpy(dtype=float)
    return np.nan_to_num(values, nan=0.0)


def load_seed_pack(args, seed):
    pred_dir = artifacts.validate_predictions_dir(args.prediction_dir_template.format(seed=seed))
    if args.target_split == "test":
        y_true = artifacts.load_array(pred_dir, artifacts.TEST_TRUE_REAL)
        indices = artifacts.load_array(pred_dir, artifacts.TEST_INDICES).astype(np.int32)
    elif args.target_split == "val":
        y_true = artifacts.load_array(pred_dir, artifacts.VAL_TRUE_REAL)
        indices = artifacts.load_array(pred_dir, artifacts.VAL_INDICES).astype(np.int32)
    else:
        y_true = artifacts.load_array(pred_dir, artifacts.TRAIN_TRUE_REAL)
        indices = artifacts.load_array(pred_dir, artifacts.TRAIN_INDICES).astype(np.int32)
    fallback = source_switch_mask(args.forecast_context_by_sample, indices, args.source_col)
    
    event_wetness = load_event_feature_array(args, indices, "event_antecedent_wetness_index")
    
    if args.event_calibrator_path and Path(args.event_calibrator_path).exists():
        from scripts.train_event_calibrator_impl import load_calibrator
        from scripts.event_source_switch_posthoc_moe_impl import load_event_features
        print(f">> [Seed {seed}] Loading event calibrator from {args.event_calibrator_path}...")
        calibrator = load_calibrator(args.event_calibrator_path)
        features = load_event_features(
            args.event_context_features,
            args.file_path,
            indices,
            args.n_his,
            args.n_pred,
            calibrator.features
        )
        features_df = pd.DataFrame(features)
        event_probs = calibrator.predict_proba(features_df)
    else:
        event_probs = load_event_feature_array(args, indices, args.event_col)
        
    event_active = event_probs >= args.event_threshold
    
    return {
        "seed": seed,
        "y_true": y_true,
        "indices": indices,
        "fallback": fallback,
        "event_active": event_active,
        "fallback_event": fallback & event_active,
        "fallback_noevent": fallback & ~event_active,
        "event_probs": event_probs,
        "event_wetness": event_wetness,
        "n_nodes": y_true.shape[2],
    }


def load_posthoc_preds(seed_dirs, configs, target_split):
    preds = {}
    filename = "moe_test_pred_real.npy" if target_split == "test" else f"moe_{target_split}_pred_real.npy"
    for config_name in configs:
        path = seed_dirs[config_name] / filename
        preds[config_name] = np.load(path)
    return preds


def evaluate_combo(args, pack, preds, official_config, fallback_event_config, fallback_noevent_config, rain_mask, rain_top_mask, coef_prob=10.0, coef_wetness=0.0, bias=-2.5, precomputed_w=None):
    if args.continuous_gate:
        if precomputed_w is not None:
            w_expanded = precomputed_w
        else:
            probs = pack["event_probs"]
            wetness = pack["event_wetness"]
            linear_val = coef_prob * probs + coef_wetness * wetness + bias
            w = 1.0 / (1.0 + np.exp(-linear_val))
            w_expanded = w.reshape(-1, 1, 1, 1)
        
        fallback_expanded = pack["fallback"].reshape(-1, 1, 1, 1)
        
        switched = (1.0 - fallback_expanded) * preds[official_config] + fallback_expanded * (
            w_expanded * preds[fallback_event_config] + (1.0 - w_expanded) * preds[fallback_noevent_config]
        )
    else:
        switched = preds[official_config].copy()
        switched[pack["fallback_event"]] = preds[fallback_event_config][pack["fallback_event"]]
        switched[pack["fallback_noevent"]] = preds[fallback_noevent_config][pack["fallback_noevent"]]
        
    y_true = pack["y_true"]
    return {
        "seed": pack["seed"],
        "official_config": official_config,
        "fallback_event_config": fallback_event_config,
        "fallback_noevent_config": fallback_noevent_config,
        "coef_prob": coef_prob,
        "coef_wetness": coef_wetness,
        "bias": bias,
        "official_samples": int((~pack["fallback"]).sum()),
        "fallback_event_samples": int(pack["fallback_event"].sum()),
        "fallback_noevent_samples": int(pack["fallback_noevent"].sum()),
        "global_mae": mae(y_true, switched),
        "rain_sensitive_mae": mae(y_true, switched, rain_mask),
        "rain_top_mae": mae(y_true, switched, rain_top_mask),
    }


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


def write_report(path, summary_df, seed_df, configs, args):
    top = summary_df.head(args.top_n).copy()
    columns_to_show = [
        "official_config",
        "fallback_event_config",
        "fallback_noevent_config",
    ]
    if args.continuous_gate:
        columns_to_show.extend(["coef_prob", "coef_wetness", "bias"])
    columns_to_show.extend(["global_mae", "rain_sensitive_mae", "rain_top_mae"])

    lines = [
        "# Event-Source Switch Grid",
        "",
        "This report grid-searches a three-way posthoc switch: official forecast samples, fallback samples inside an event window, and fallback samples outside an event window.",
        "",
        "## Configuration",
        "",
        f"- seeds: `{','.join(str(seed) for seed in args.seeds)}`",
        f"- event_col: `{args.event_col}`",
        f"- event_threshold: `{args.event_threshold}`",
        f"- candidate_configs: `{', '.join(configs)}`",
        f"- continuous_gate: `{args.continuous_gate}`",
        "",
        "## Top Configs",
        "",
        markdown_table(top[columns_to_show]),
        "",
        "## Boundary",
        "",
        "This is a posthoc test-label audit over persisted predictions. Use it to design a validation-selected gate or distillation target, not as a standalone formal model selection result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    seed_dirs = {seed: config_dirs(args.posthoc_grid_dir, seed) for seed in args.seeds}
    common_configs = sorted(set.intersection(*(set(dirs) for dirs in seed_dirs.values())))
    if not common_configs:
        raise ValueError("No common posthoc configs are available across all requested seeds.")
    official_configs, fallback_configs = candidate_configs(args, common_configs)
    configs = all_configs(official_configs, fallback_configs)
    seed_packs = [load_seed_pack(args, seed) for seed in args.seeds]
    n_nodes = seed_packs[0]["n_nodes"]
    for pack in seed_packs:
        if pack["n_nodes"] != n_nodes:
            raise ValueError(f"Seed {pack['seed']} has n_nodes={pack['n_nodes']}, expected {n_nodes}.")
    rain_mask, rain_top_mask = load_node_masks(args.rain_susceptibility, n_nodes)
    seed_preds = {seed: load_posthoc_preds(seed_dirs[seed], configs, args.target_split) for seed in args.seeds}

    seed_rows = []
    if args.continuous_gate:
        coef_prob_list = [5.0, 10.0, 15.0]
        coef_wetness_list = [0.0, 1.0, 3.0, 5.0]
        bias_list = [-3.5, -2.5, -1.5, -0.5]
    else:
        coef_prob_list = [args.gate_coef_prob]
        coef_wetness_list = [args.gate_coef_wetness]
        bias_list = [args.gate_bias]

    precomputed_ws = {}
    if args.continuous_gate:
        for coef_prob in coef_prob_list:
            for coef_wetness in coef_wetness_list:
                for bias in bias_list:
                    key = (coef_prob, coef_wetness, bias)
                    precomputed_ws[key] = {}
                    for pack in seed_packs:
                        probs = pack["event_probs"]
                        wetness = pack["event_wetness"]
                        linear_val = coef_prob * probs + coef_wetness * wetness + bias
                        w = 1.0 / (1.0 + np.exp(-linear_val))
                        precomputed_ws[key][pack["seed"]] = w.reshape(-1, 1, 1, 1)

    for official_config in official_configs:
        # Precompute the non-fallback error statistics for this official_config
        non_fb_stats = {}
        for pack in seed_packs:
            seed = pack["seed"]
            preds = seed_preds[seed]
            official_pred = preds[official_config]
            y_true = pack["y_true"]
            fb = pack["fallback"]
            not_fb = ~fb
            
            if np.any(not_fb):
                err_non_fb_global = np.sum(np.abs(y_true[not_fb] - official_pred[not_fb]))
                err_non_fb_rain = np.sum(np.abs(y_true[not_fb] - official_pred[not_fb])[:, :, rain_mask, :])
                err_non_fb_top = np.sum(np.abs(y_true[not_fb] - official_pred[not_fb])[:, :, rain_top_mask, :])
                n_non_fb_global = y_true[not_fb].size
                n_non_fb_rain = y_true[not_fb][:, :, rain_mask, :].size
                n_non_fb_top = y_true[not_fb][:, :, rain_top_mask, :].size
            else:
                err_non_fb_global = err_non_fb_rain = err_non_fb_top = 0.0
                n_non_fb_global = n_non_fb_rain = n_non_fb_top = 0
                
            non_fb_stats[seed] = {
                "err_global": err_non_fb_global,
                "err_rain": err_non_fb_rain,
                "err_top": err_non_fb_top,
                "n_global": n_non_fb_global,
                "n_rain": n_non_fb_rain,
                "n_top": n_non_fb_top,
            }

        for fallback_event_config in fallback_configs:
            for fallback_noevent_config in fallback_configs:
                # Pre-slice arrays for the current configuration pair for all seeds
                config_fb_stats = {}
                for pack in seed_packs:
                    seed = pack["seed"]
                    preds = seed_preds[seed]
                    y_true = pack["y_true"]
                    fb = pack["fallback"]
                    
                    if np.any(fb):
                        y_true_fb = y_true[fb]
                        pred_evt_fb = preds[fallback_event_config][fb]
                        pred_noevt_fb = preds[fallback_noevent_config][fb]
                        
                        y_true_fb_rain = y_true_fb[:, :, rain_mask, :]
                        pred_evt_fb_rain = pred_evt_fb[:, :, rain_mask, :]
                        pred_noevt_fb_rain = pred_noevt_fb[:, :, rain_mask, :]
                        
                        y_true_fb_top = y_true_fb[:, :, rain_top_mask, :]
                        pred_evt_fb_top = pred_evt_fb[:, :, rain_top_mask, :]
                        pred_noevt_fb_top = pred_noevt_fb[:, :, rain_top_mask, :]
                        
                        active_fb = pack["event_active"][fb]
                        
                        n_fb_global = y_true_fb.size
                        n_fb_rain = y_true_fb_rain.size
                        n_fb_top = y_true_fb_top.size
                    else:
                        y_true_fb = pred_evt_fb = pred_noevt_fb = None
                        y_true_fb_rain = pred_evt_fb_rain = pred_noevt_fb_rain = None
                        y_true_fb_top = pred_evt_fb_top = pred_noevt_fb_top = None
                        active_fb = None
                        n_fb_global = n_fb_rain = n_fb_top = 0
                        
                    config_fb_stats[seed] = {
                        "y_true_fb": y_true_fb,
                        "pred_evt_fb": pred_evt_fb,
                        "pred_noevt_fb": pred_noevt_fb,
                        "y_true_fb_rain": y_true_fb_rain,
                        "pred_evt_fb_rain": pred_evt_fb_rain,
                        "pred_noevt_fb_rain": pred_noevt_fb_rain,
                        "y_true_fb_top": y_true_fb_top,
                        "pred_evt_fb_top": pred_evt_fb_top,
                        "pred_noevt_fb_top": pred_noevt_fb_top,
                        "active_fb": active_fb,
                        "n_fb_global": n_fb_global,
                        "n_fb_rain": n_fb_rain,
                        "n_fb_top": n_fb_top,
                    }

                for coef_prob in coef_prob_list:
                    for coef_wetness in coef_wetness_list:
                        for bias in bias_list:
                            key = (coef_prob, coef_wetness, bias)
                            for pack in seed_packs:
                                seed = pack["seed"]
                                fb = pack["fallback"]
                                stats = non_fb_stats[seed]
                                cfg_stats = config_fb_stats[seed]
                                
                                if np.any(fb):
                                    if args.continuous_gate:
                                        w_fb = precomputed_ws[key][seed][fb]
                                        moe_pred_fb = w_fb * cfg_stats["pred_evt_fb"] + (1.0 - w_fb) * cfg_stats["pred_noevt_fb"]
                                        moe_pred_fb_rain = w_fb * cfg_stats["pred_evt_fb_rain"] + (1.0 - w_fb) * cfg_stats["pred_noevt_fb_rain"]
                                        moe_pred_fb_top = w_fb * cfg_stats["pred_evt_fb_top"] + (1.0 - w_fb) * cfg_stats["pred_noevt_fb_top"]
                                    else:
                                        active_fb = cfg_stats["active_fb"]
                                        moe_pred_fb = np.zeros_like(cfg_stats["y_true_fb"])
                                        moe_pred_fb[active_fb] = cfg_stats["pred_evt_fb"][active_fb]
                                        moe_pred_fb[~active_fb] = cfg_stats["pred_noevt_fb"][~active_fb]
                                        
                                        moe_pred_fb_rain = np.zeros_like(cfg_stats["y_true_fb_rain"])
                                        moe_pred_fb_rain[active_fb] = cfg_stats["pred_evt_fb_rain"][active_fb]
                                        moe_pred_fb_rain[~active_fb] = cfg_stats["pred_noevt_fb_rain"][~active_fb]
                                        
                                        moe_pred_fb_top = np.zeros_like(cfg_stats["y_true_fb_top"])
                                        moe_pred_fb_top[active_fb] = cfg_stats["pred_evt_fb_top"][active_fb]
                                        moe_pred_fb_top[~active_fb] = cfg_stats["pred_noevt_fb_top"][~active_fb]
                                        
                                    err_fb_global = np.sum(np.abs(cfg_stats["y_true_fb"] - moe_pred_fb))
                                    err_fb_rain = np.sum(np.abs(cfg_stats["y_true_fb_rain"] - moe_pred_fb_rain))
                                    err_fb_top = np.sum(np.abs(cfg_stats["y_true_fb_top"] - moe_pred_fb_top))
                                else:
                                    err_fb_global = err_fb_rain = err_fb_top = 0.0
                                    
                                global_mae = (stats["err_global"] + err_fb_global) / (stats["n_global"] + cfg_stats["n_fb_global"])
                                rain_sensitive_mae = (stats["err_rain"] + err_fb_rain) / (stats["n_rain"] + cfg_stats["n_fb_rain"])
                                rain_top_mae = (stats["err_top"] + err_fb_top) / (stats["n_top"] + cfg_stats["n_fb_top"])
                                
                                seed_rows.append({
                                    "seed": seed,
                                    "official_config": official_config,
                                    "fallback_event_config": fallback_event_config,
                                    "fallback_noevent_config": fallback_noevent_config,
                                    "coef_prob": coef_prob,
                                    "coef_wetness": coef_wetness,
                                    "bias": bias,
                                    "official_samples": int((~fb).sum()),
                                    "fallback_event_samples": int((fb & pack["event_active"]).sum()),
                                    "fallback_noevent_samples": int((fb & ~pack["event_active"]).sum()),
                                    "global_mae": global_mae,
                                    "rain_sensitive_mae": rain_sensitive_mae,
                                    "rain_top_mae": rain_top_mae,
                                })
    seed_df = pd.DataFrame(seed_rows)
    group_cols = ["official_config", "fallback_event_config", "fallback_noevent_config", "coef_prob", "coef_wetness", "bias"]
    summary_df = (
        seed_df.groupby(group_cols, as_index=False)[
            ["global_mae", "rain_sensitive_mae", "rain_top_mae"]
        ]
        .mean()
        .sort_values(["rain_sensitive_mae", "rain_top_mae", "global_mae"], ascending=True)
        .reset_index(drop=True)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "event_source_switch_grid.csv", index=False)
    seed_df.to_csv(output_dir / "event_source_switch_seed_metrics.csv", index=False)
    write_report(output_dir / "event_source_switch_grid_report.md", summary_df, seed_df, configs, args)
    print(summary_df.head(args.top_n).to_string(index=False))
    print(f">> Saved event-source-switch grid outputs to: {output_dir}")


if __name__ == "__main__":
    main()
