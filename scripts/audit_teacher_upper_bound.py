import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def mae(y_t, y_p, mask=None):
    if mask is not None:
        return np.mean(np.abs(y_t[:, :, mask, :] - y_p[:, :, mask, :]))
    return np.mean(np.abs(y_t - y_p))

def h_mae(y_t, y_p, h, mask=None):
    if mask is not None:
        return np.mean(np.abs(y_t[:, h, mask, :] - y_p[:, h, mask, :]))
    return np.mean(np.abs(y_t[:, h, :, :] - y_p[:, h, :, :]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv('output/rain_susceptibility/rain_susceptibility.csv')
    sus = df['rain_susceptibility'].values
    mask_rain = (sus > 0.0)
    q_top = np.quantile(sus[sus > 0], 0.6)
    mask_top = (sus > q_top)

    records = []
    
    for seed in [0, 7, 23]:
        for fold in ["cp_160", "cp_180"]:
            base_dir_off = f"output/pipeline_gefs15_seed{seed}/{fold}/predictions"
            if not Path(f"{base_dir_off}/test_true_real.npy").exists():
                continue
                
            y_true = np.load(f"{base_dir_off}/test_true_real.npy")
            y_gefs = np.load(f"{base_dir_off}/test_pred_real.npy")
            
            # 1. Baseline GEFS15
            records.append({
                'method': 'Baseline GEFS15', 'seed': seed, 'fold': fold,
                'global_mae': mae(y_true, y_gefs),
                'rain_sensitive_mae': mae(y_true, y_gefs, mask_rain),
                'rain_top_mae': mae(y_true, y_gefs, mask_top),
                'h3_rain_sensitive_mae': h_mae(y_true, y_gefs, 2, mask_rain),
                'h5_rain_sensitive_mae': h_mae(y_true, y_gefs, 4, mask_rain)
            })

            # 2. Mixed KNN Teacher (LR Gate) - The actual distillation target
            teacher_path = f"output/pipeline_teacher_seed{seed}_model_I/{fold}/predictions/test_teacher_pred_real.npy"
            if Path(teacher_path).exists():
                y_teacher = np.load(teacher_path)
                records.append({
                    'method': 'Mixed KNN Teacher (LR Gate)', 'seed': seed, 'fold': fold,
                    'global_mae': mae(y_true, y_teacher),
                    'rain_sensitive_mae': mae(y_true, y_teacher, mask_rain),
                    'rain_top_mae': mae(y_true, y_teacher, mask_top),
                    'h3_rain_sensitive_mae': h_mae(y_true, y_teacher, 2, mask_rain),
                    'h5_rain_sensitive_mae': h_mae(y_true, y_teacher, 4, mask_rain)
                })

            # 3. Model E
            model_e_path = f"output/pipeline_distilled_seed{seed}/{fold}/predictions/test_pred_real.npy"
            if Path(model_e_path).exists():
                y_e = np.load(model_e_path)
                records.append({
                    'method': 'Model E', 'seed': seed, 'fold': fold,
                    'global_mae': mae(y_true, y_e),
                    'rain_sensitive_mae': mae(y_true, y_e, mask_rain),
                    'rain_top_mae': mae(y_true, y_e, mask_top),
                    'h3_rain_sensitive_mae': h_mae(y_true, y_e, 2, mask_rain),
                    'h5_rain_sensitive_mae': h_mae(y_true, y_e, 4, mask_rain)
                })

            # 4. source-switch best-pair (Oracle)
            event_path = f"output/pipeline_event_seed{seed}/{fold}/predictions/test_pred_real.npy"
            if Path(event_path).exists():
                y_fallback = np.load(event_path)
                err_official = np.abs(y_true - y_gefs)
                err_fallback = np.abs(y_true - y_fallback)
                err_oracle = np.minimum(err_official, err_fallback)
                records.append({
                    'method': 'source-switch best-pair', 'seed': seed, 'fold': fold,
                    'global_mae': np.mean(err_oracle),
                    'rain_sensitive_mae': np.mean(err_oracle[:, :, mask_rain, :]),
                    'rain_top_mae': np.mean(err_oracle[:, :, mask_top, :]),
                    'h3_rain_sensitive_mae': np.mean(err_oracle[:, 2, mask_rain, :]),
                    'h5_rain_sensitive_mae': np.mean(err_oracle[:, 4, mask_rain, :])
                })



    df = pd.DataFrame(records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "teacher_upper_bound_audit.csv", index=False)
    
    # Save markdown summary
    md_lines = ["# Teacher Upper Bound Audit\n\n"]
    for fold in ["cp_160", "cp_180"]:
        md_lines.append(f"## {fold}\n")
        fold_df = df[df['fold'] == fold]
        if not fold_df.empty:
            mean_df = fold_df.groupby('method').mean(numeric_only=True).reset_index()
            # Sort by rain_sensitive_mae
            mean_df = mean_df.sort_values("rain_sensitive_mae", ascending=False)
            
            # Format markdown table
            md_lines.append("| method | global_mae | rain_sensitive_mae | rain_top_mae | h3_rain_sensitive_mae | h5_rain_sensitive_mae |")
            md_lines.append("|---|---|---|---|---|---|")
            for _, row in mean_df.iterrows():
                md_lines.append(f"| {row['method']} | {row['global_mae']:.3f} | {row['rain_sensitive_mae']:.3f} | {row['rain_top_mae']:.3f} | {row['h3_rain_sensitive_mae']:.3f} | {row['h5_rain_sensitive_mae']:.3f} |")
        md_lines.append("\n")
        
    (out_dir / "teacher_upper_bound_audit_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("Audit generated at", out_dir)

if __name__ == "__main__":
    main()
