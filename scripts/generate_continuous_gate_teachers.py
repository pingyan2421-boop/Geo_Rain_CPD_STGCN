import argparse
import os
import numpy as np
import pandas as pd
import subprocess
import sys
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scripts.train_event_calibrator_impl import load_calibrator
from scripts.event_source_switch_posthoc_moe_impl import load_event_features
from data_loader.date_loader import load_and_clean_data
from data_loader.rainfall_loader import forecast_context_for_starts

def forecast_mask(forecast_csv, file_path, indices, n_his, n_pred, switch_col):
    raw_seq, _, _, time_cols, _ = load_and_clean_data(file_path)
    starts = np.arange(0, raw_seq.shape[0] - n_his - n_pred + 1, dtype=np.int32)
    context = forecast_context_for_starts(forecast_csv, time_cols, starts, n_his)
    values = context.iloc[np.asarray(indices, dtype=np.int32)][switch_col].to_numpy(dtype=float)
    return values > 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def run_posthoc(base_dir, fold, feature_group, shrink, max_abs, output_prefix):
    """Run analog residual correction for a specific posthoc expert."""
    out_dir = os.path.join(base_dir, fold, f"posthoc_{output_prefix}")
    if os.path.exists(os.path.join(out_dir, "train_pred_real.npy")):
        return out_dir
    cmd = [
        sys.executable, "-m", "scripts.analog_residual_correction_impl",
        "--predictions_dir", os.path.join(base_dir, fold, "predictions"),
        "--file_path", "dataset/inter228_5241.csv",
        "--shrink", str(shrink),
        "--max_abs_correction", str(max_abs),
        "--feature_group", feature_group,
        "--output_dir", out_dir
    ]
    subprocess.run(cmd, check=True)
    return out_dir

def generate_teachers(base_dir_official, base_dir_fallback, output_dir, args, folds=["cp_160", "cp_180", "cp_200"]):
    calibrator = load_calibrator("output/event_calibrator.json")
    
    # Pre-calculate node indices for rain sensitivity
    rain_susceptibility = pd.read_csv("output/rain_susceptibility/rain_susceptibility.csv")["rain_susceptibility"].values
    p90 = np.percentile(rain_susceptibility[rain_susceptibility > 0] if np.any(rain_susceptibility > 0) else [0], 90)
    node_weights = np.ones((len(rain_susceptibility), 1)) * 0.1  # Penalize safe nodes
    node_weights[rain_susceptibility > 0] = 3.0  # Massive boost to rain sensitive
    node_weights[rain_susceptibility >= p90] = 5.0 # Massive boost to top 10% rain sensitive
    
    # Pre-calculate horizon weights
    horizon_weights = np.ones((18, 1, 1)) * 0.5
    horizon_weights[2:] = 3.0  # Huge boost to H+3 to H+18

    gate_summary_rows = []

    # --- Phase 1: Global Feature Collection ---
    print("=== Phase 1: Global Feature Collection for Unified Gate ===")
    global_X_train = []
    global_y_train = []
    
    for fold in folds:
        val_idx_path = os.path.join(base_dir_official, fold, "predictions", "val_indices.npy")
        if not os.path.exists(val_idx_path):
            continue
            
        try:
            cp_idx = int(fold.split('_')[1])
            val_cpd_stage = cp_idx / 200.0
        except:
            val_cpd_stage = 1.0
            
        val_idx = np.load(val_idx_path)
        val_is_fallback = forecast_mask("dataset/forecast/chirps_gefs_15day_cp180_climfallback.csv", "dataset/inter228_5241.csv", val_idx, 12, 5, "forecast_is_fallback")
        
        val_off_pred = np.load(os.path.join(base_dir_official, fold, "predictions", "val_pred_real.npy"))
        val_fall_pred = np.load(os.path.join(base_dir_fallback, fold, "predictions", "val_pred_real.npy"))
        val_true = np.load(os.path.join(base_dir_official, fold, "predictions", "val_true_real.npy"))
        
        feat = load_event_features("output/rainfall_event_catalog/rainfall_event_catalog.csv", "dataset/inter228_5241.csv", val_idx, 12, 5, list(calibrator.features))
        val_probs = calibrator.predict_proba(pd.DataFrame(feat))
        val_wetness = feat["antecedent_wetness_index"]
        
        val_cpd_feat = np.full_like(val_probs, val_cpd_stage)
        
        val_off_err = np.mean(np.abs(val_off_pred - val_true), axis=(1,2,3))
        val_fall_err = np.mean(np.abs(val_fall_pred - val_true), axis=(1,2,3))
        y_val = (val_fall_err < val_off_err).astype(int)
        
        train_mask = val_is_fallback > 0.5
        X_val = np.column_stack([val_probs, val_wetness, val_cpd_feat])
        
        if train_mask.sum() > 0:
            global_X_train.append(X_val[train_mask])
            global_y_train.append(y_val[train_mask])

    # --- Phase 2: Global LR Training ---
    print("=== Phase 2: Training Global Unified Gate ===")
    scaler = StandardScaler()
    lr = LogisticRegression(class_weight='balanced', C=10.0, max_iter=1000)
    used_default = False
    
    if len(global_X_train) > 0:
        X_all = np.concatenate(global_X_train, axis=0)
        y_all = np.concatenate(global_y_train, axis=0)
        print(f"Global training data size: {len(X_all)}, Positive ratio: {np.mean(y_all):.3f}")
        if len(X_all) > 5 and len(np.unique(y_all)) > 1:
            X_all_scaled = scaler.fit_transform(X_all)
            lr.fit(X_all_scaled, y_all)
            # Rescale coefficients back to original scale to match the inference formula:
            # logit = coef * x + bias
            # scaled_x = (x - mean) / std
            # coef_scaled * scaled_x + bias_scaled = (coef_scaled / std) * x + (bias_scaled - coef_scaled * mean / std)
            orig_coef = lr.coef_[0] / scaler.scale_
            orig_bias = lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_)
            
            coef_prob, coef_wetness, coef_cpd = orig_coef
            bias = orig_bias
            if np.sum(np.abs(orig_coef)) < 1e-4:
                print("Global LR yielded near-zero coefficients (likely due to zero variance in validation features). Falling back.")
                used_default = True
            else:
                print(f"Global LR Fitted: probs={coef_prob:.3f}, wetness={coef_wetness:.3f}, cpd_stage={coef_cpd:.3f}, bias={bias:.3f}")
        else:
            used_default = True
    else:
        used_default = True
        
    if used_default:
        print("Insufficient global variations. Falling back to default unified params.")
        coef_prob, coef_wetness, coef_cpd, bias = 5.0, 5.0, 2.0, -3.5
        lr.coef_ = np.array([[coef_prob, coef_wetness, coef_cpd]])
        lr.intercept_ = np.array([bias])

    gate_summary_rows.append({
        "fold": "GLOBAL",
        "coef_prob": coef_prob,
        "coef_wetness": coef_wetness,
        "coef_cpd": coef_cpd,
        "bias": bias,
        "used_default_params": used_default
    })

    # --- Phase 3: Generation & Weight Assignment ---
    for fold in folds:
        print(f"=== Generating Teachers & Weights for fold {fold} ===")
        val_idx_path = os.path.join(base_dir_official, fold, "predictions", "val_indices.npy")
        if not os.path.exists(val_idx_path):
            continue
            
        try:
            cp_idx = int(fold.split('_')[1])
            val_cpd_stage = cp_idx / 200.0
        except:
            val_cpd_stage = 1.0

        print(f"[{fold}] Generating Official Expert (response_only/k=3/t=0.90/g=200.0)...")
        off_expert_dir = run_posthoc(base_dir_official, fold, "response_only", 0.90, 200.0, "off")
        
        print(f"[{fold}] Generating Fallback Low-risk Expert (all/k=3/t=0.35/g=40.0)...")
        fall_low_expert_dir = run_posthoc(base_dir_fallback, fold, "all", 0.35, 40.0, "fall_low")
        
        print(f"[{fold}] Generating Fallback High-risk Expert (all/k=3/t=0.90/g=200.0)...")
        fall_high_expert_dir = run_posthoc(base_dir_fallback, fold, "all", 0.90, 200.0, "fall_high")

        for split in ["train", "val", "test"]:
            idx_path = os.path.join(base_dir_official, fold, "predictions", f"{split}_indices.npy")
            if not os.path.exists(idx_path):
                continue
            idx = np.load(idx_path)
            
            off_pred = np.load(os.path.join(off_expert_dir, f"{split}_pred_real.npy"))
            fall_high_pred = np.load(os.path.join(fall_high_expert_dir, f"{split}_pred_real.npy"))
            
            feat = load_event_features("output/rainfall_event_catalog/rainfall_event_catalog.csv", "dataset/inter228_5241.csv", idx, 12, 5, list(calibrator.features))
            probs = calibrator.predict_proba(pd.DataFrame(feat))
            wetness = feat["antecedent_wetness_index"]
            
            logits = coef_prob * probs + coef_wetness * wetness + coef_cpd * val_cpd_stage + bias
            w = sigmoid(logits)
            w_reshaped = w.reshape(-1, 1, 1, 1)
            
            is_fallback = forecast_mask("dataset/forecast/chirps_gefs_15day_cp180_climfallback.csv", "dataset/inter228_5241.csv", idx, 12, 5, "forecast_is_fallback")
            is_fallback_reshaped = is_fallback.reshape(-1, 1, 1, 1)
            
            fallback_teacher = fall_high_pred
            teacher_pred = (1 - is_fallback_reshaped) * off_pred + is_fallback_reshaped * fallback_teacher
            
            sample_weight = np.ones(len(idx)) * 1.0
            sample_weight[is_fallback] = 0.5 # Downweight fallback noise
            sample_weight[~is_fallback] = 2.0 # Boost official high-risk events
            
            horizon_weights_5 = horizon_weights[:5].reshape(1, 5, 1, 1)
            w_sample = sample_weight.reshape(-1, 1, 1, 1) if args.enable_sample_weight else 1.0
            w_horizon = horizon_weights_5 if args.enable_horizon_weight else 1.0
            w_node = node_weights.reshape(1, 1, -1, 1) if args.enable_node_weight else 1.0
            
            weight_3d = w_sample * w_horizon * w_node
            weight_3d = np.broadcast_to(weight_3d, (len(idx), 5, len(node_weights), 1)).copy()
            weight_3d = weight_3d / np.mean(weight_3d)
            weight_3d = np.clip(weight_3d, 0.0, 500.0)
            weight_3d = weight_3d / np.mean(weight_3d)

            out_path = os.path.join(output_dir, fold, "predictions")
            os.makedirs(out_path, exist_ok=True)
            
            if split == "train":
                np.save(os.path.join(out_path, "train_distill_weights.npy"), weight_3d)
                df = pd.DataFrame({
                    "sample_start": idx,
                    "gate_weight": w,
                    "event_risk": probs,
                    "cpd_stage": val_cpd_stage,
                    "is_fallback": is_fallback.astype(int)
                })
                df.to_csv(os.path.join(out_path, "distillation_sample_metrics.csv"), index=False)
                teacher_file = "train_teacher_pred_real.npy"
            else:
                teacher_file = f"{split}_teacher_pred_real.npy"
                
            np.save(os.path.join(out_path, teacher_file), teacher_pred)
            
            true_path = os.path.join(base_dir_official, fold, "predictions", f"{split}_true_real.npy")
            if os.path.exists(true_path):
                shutil.copy2(true_path, os.path.join(out_path, f"{split}_true_real.npy"))
            shutil.copy2(idx_path, os.path.join(out_path, f"{split}_indices.npy"))
            
            print(f"Saved {fold} {split} teacher predictions to {out_path} (HighRisk: {np.sum((is_fallback) & (w >= 0.5))})")

    # Save summary
    if gate_summary_rows:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(gate_summary_rows).to_csv(os.path.join(output_dir, "gate_fit_summary.csv"), index=False)
        print("Saved gate_fit_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir_official", type=str, default="output/pipeline_gefs15")
    parser.add_argument("--base_dir_fallback", type=str, default="output/pipeline_event")
    parser.add_argument("--output_dir", type=str, default="output/pipeline_teacher")
    parser.add_argument("--enable_sample_weight", action="store_true")
    parser.add_argument("--enable_horizon_weight", action="store_true")
    parser.add_argument("--enable_node_weight", action="store_true")
    args = parser.parse_args()
    
    generate_teachers(args.base_dir_official, args.base_dir_fallback, args.output_dir, args)
