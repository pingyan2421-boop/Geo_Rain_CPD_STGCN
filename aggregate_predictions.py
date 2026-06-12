import numpy as np
import os
import pandas as pd
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

def bootstrap_mae(data, n_resamples=1000):
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    if len(data) < 5:
        return np.mean(data), np.mean(data), np.mean(data)
    means = []
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return np.mean(data), np.percentile(means, 5), np.percentile(means, 95)

def calculate_metrics():
    seeds = [0, 7, 23]
    methods = [
        "pipeline_gefs15", "pipeline_gefs15_nosar", 
        "pipeline_distilled_nosar", "pipeline_distilled_noweight",
        "pipeline_distilled", "pipeline_hybrid"
    ]
    folds = ["cp_160", "cp_180"]
    
    calibrator = load_calibrator("output/event_calibrator.json")
    
    # Load susceptibility to filter rain sensitive nodes
    susceptibility_path = "output/rain_susceptibility/rain_susceptibility.csv"
    if not os.path.exists(susceptibility_path):
        print("Susceptibility file not found.")
        return
        
    susc_df = pd.read_csv(susceptibility_path)
    rain_susceptibility = susc_df['rain_susceptibility'].to_numpy()
    rain_sensitive_mask = rain_susceptibility > 0
    
    q90 = np.percentile(rain_susceptibility[rain_susceptibility > 0] if np.any(rain_susceptibility > 0) else [0], 90)
    rain_top_mask = rain_susceptibility >= q90

    results = []
    segment_results = []

    for seed in seeds:
        gate_summary_path = f"output/pipeline_teacher_seed{seed}/gate_fit_summary.csv"
        gate_summary = None
        if os.path.exists(gate_summary_path):
            gate_summary = pd.read_csv(gate_summary_path)

        for fold in folds:
            # Get Gate LR parameters
            coef_prob, coef_wetness, coef_cpd, bias = 5.0, 5.0, 2.0, -3.5
            if gate_summary is not None:
                row = gate_summary[gate_summary["fold"] == fold]
                if row.empty: # Fallback to GLOBAL if fold not found
                    row = gate_summary[gate_summary["fold"] == "GLOBAL"]
                if not row.empty:
                    coef_prob = row.iloc[0]["coef_prob"]
                    coef_wetness = row.iloc[0]["coef_wetness"]
                    if "coef_cpd" in row.columns:
                        coef_cpd = row.iloc[0]["coef_cpd"]
                    bias = row.iloc[0]["bias"]

            test_idx_path = f"output/pipeline_gefs15_seed{seed}/{fold}/predictions/test_indices.npy"
            if not os.path.exists(test_idx_path):
                continue
            test_idx = np.load(test_idx_path)
            
            try:
                cp_idx = int(fold.split('_')[1])
                val_cpd_stage = cp_idx / 200.0
            except:
                val_cpd_stage = 1.0
            
            # Predict Event Context Features
            feat = load_event_features("output/rainfall_event_catalog/rainfall_event_catalog.csv", "dataset/inter228_5241.csv", test_idx, 12, 5, list(calibrator.features))
            probs = calibrator.predict_proba(pd.DataFrame(feat))
            wetness = feat["antecedent_wetness_index"]
            w = sigmoid(coef_prob * probs + coef_wetness * wetness + coef_cpd * val_cpd_stage + bias)
            is_fallback = forecast_mask("dataset/forecast/chirps_gefs_15day_cp180_climfallback.csv", "dataset/inter228_5241.csv", test_idx, 12, 5, "forecast_is_fallback")
            
            gefs15_pred_file = f"output/pipeline_gefs15_seed{seed}/{fold}/predictions/test_pred_real.npy"
            distilled_pred_file = f"output/pipeline_distilled_seed{seed}/{fold}/predictions/test_pred_real.npy"
            true_file = f"output/pipeline_gefs15_seed{seed}/{fold}/predictions/test_true_real.npy"
            
            hybrid_dir = f"output/pipeline_hybrid_seed{seed}/{fold}/predictions"
            os.makedirs(hybrid_dir, exist_ok=True)
            
            if os.path.exists(gefs15_pred_file) and os.path.exists(distilled_pred_file) and os.path.exists(true_file):
                gefs15_pred = np.load(gefs15_pred_file)
                distilled_pred = np.load(distilled_pred_file)
                true_data = np.load(true_file)
                
                hybrid_pred = np.copy(gefs15_pred)
                hybrid_pred[~is_fallback] = distilled_pred[~is_fallback]
                
                np.save(f"{hybrid_dir}/test_pred_real.npy", hybrid_pred)
                np.save(f"{hybrid_dir}/test_true_real.npy", true_data)

            # Segments
            seg_masks = {
                "official": ~is_fallback,
                "fallback_low_prob": is_fallback & (w < 0.5),
                "fallback_high_prob": is_fallback & (w >= 0.5),
                "event_active_3d": feat["spell_days"] <= 3,
                "event_active_7d": (feat["spell_days"] > 3) & (feat["spell_days"] <= 7),
                "event_active_15d": (feat["spell_days"] > 7) & (feat["spell_days"] <= 15),
                "delayed_response_event": (feat["spell_days"] > 0) & (wetness > 0.5)
            }

            for method in methods:
                base_dir = f"output/{method}_seed{seed}"
                pred_file = f"{base_dir}/{fold}/predictions/test_pred_real.npy"
                true_file = f"{base_dir}/{fold}/predictions/test_true_real.npy"
                
                if not (os.path.exists(pred_file) and os.path.exists(true_file)):
                    continue
                    
                pred = np.load(pred_file) # (samples, 18, 228, 1)
                true = np.load(true_file)
                
                mae_all = np.abs(pred - true) # (samples, 18, 228, 1)
                
                def compute_sub_metrics(m_all, mask=None):
                    if mask is not None:
                        m_all = m_all[mask]
                    if len(m_all) == 0:
                        return np.nan, np.nan, np.nan, np.nan, np.nan
                    global_mae = np.mean(m_all)
                    rain_sensitive_mae = np.mean(m_all[:, :, rain_sensitive_mask, :])
                    rain_top_mae = np.mean(m_all[:, :, rain_top_mask, :])
                    h3_rain_sensitive_mae = np.mean(m_all[:, 2, rain_sensitive_mask, :]) if m_all.shape[1] > 2 else np.nan
                    h5_rain_sensitive_mae = np.mean(m_all[:, 4, rain_sensitive_mask, :]) if m_all.shape[1] > 4 else np.nan
                    return global_mae, rain_sensitive_mae, rain_top_mae, h3_rain_sensitive_mae, h5_rain_sensitive_mae

                # Full Test Set
                g_mae, rs_mae, rt_mae, h3_rs_mae, h5_rs_mae = compute_sub_metrics(mae_all)
                
                results.append({
                    "method": method,
                    "seed": seed,
                    "fold": fold,
                    "global_mae": g_mae,
                    "rain_sensitive_mae": rs_mae,
                    "rain_top_mae": rt_mae,
                    "h3_rain_sensitive_mae": h3_rs_mae,
                    "h5_rain_sensitive_mae": h5_rs_mae,
                    "official_samples": seg_masks["official"].sum(),
                    "fallback_samples": is_fallback.sum(),
                    "fallback_low_risk_samples": seg_masks["fallback_low_prob"].sum(),
                    "fallback_high_risk_samples": seg_masks["fallback_high_prob"].sum()
                })
                
                # Segmented Test Set (5.7)
                for seg_name, mask in seg_masks.items():
                    if mask.sum() == 0:
                        continue
                    # Compute mean and 95% CI using bootstrap for rain_sensitive_mae and h5_rain_sensitive_mae
                    sample_means = np.mean(mae_all[mask], axis=(1,2,3))
                    rs_sample_means = np.mean(mae_all[mask][:, :, rain_sensitive_mask, :], axis=(1,2,3))
                    h5_rs_sample_means = np.mean(mae_all[mask][:, 4, rain_sensitive_mask, :], axis=(1,2)) if mae_all.shape[1] > 4 else []
                    
                    g_mean, g_p05, g_p95 = bootstrap_mae(sample_means)
                    rs_mean, rs_p05, rs_p95 = bootstrap_mae(rs_sample_means)
                    h5_rs_mean, h5_rs_p05, h5_rs_p95 = bootstrap_mae(h5_rs_sample_means)
                    
                    segment_results.append({
                        "method": method,
                        "seed": seed,
                        "fold": fold,
                        "segment": seg_name,
                        "sample_count": mask.sum(),
                        "global_mae": g_mean,
                        "global_mae_ci": f"[{g_p05:.3f}, {g_p95:.3f}]" if not np.isnan(g_p05) else "",
                        "rain_sensitive_mae": rs_mean,
                        "rain_sensitive_mae_ci": f"[{rs_p05:.3f}, {rs_p95:.3f}]" if not np.isnan(rs_p05) else "",
                        "h5_rain_sensitive_mae": h5_rs_mean,
                        "h5_rain_sensitive_mae_ci": f"[{h5_rs_p05:.3f}, {h5_rs_p95:.3f}]" if not np.isnan(h5_rs_p05) else "",
                    })

    if results:
        df = pd.DataFrame(results)
        df.to_csv("full_metrics_audit.csv", index=False)
        print("Generated full_metrics_audit.csv")
        
        agg = df.groupby('method').agg({
            'global_mae': ['mean', 'std'],
            'rain_sensitive_mae': ['mean', 'std'],
            'rain_top_mae': ['mean', 'std'],
            'h3_rain_sensitive_mae': ['mean', 'std'],
            'h5_rain_sensitive_mae': ['mean', 'std']
        }).round(4)
        print("\n=== Global Metrics Summary ===")
        print(agg.to_string())
        
    if segment_results:
        seg_df = pd.DataFrame(segment_results)
        seg_df.to_csv("segmented_metrics_audit.csv", index=False)
        print("\nGenerated segmented_metrics_audit.csv")

if __name__ == "__main__":
    calculate_metrics()
