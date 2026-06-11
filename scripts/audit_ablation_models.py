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
    
    models_to_eval = [
        ("Baseline GEFS15", "output/pipeline_gefs15_seed0/cp_180/predictions/test_pred_real.npy"),
        ("Model E (Old Full 3D)", "output/pipeline_distilled_seed0/cp_180/predictions/test_pred_real.npy"),
        ("Model F (Sample Weight)", "output/pipeline_model_F_seed0/cp_180/predictions/test_pred_real.npy"),
        ("Model G (Horizon Weight)", "output/pipeline_model_G_seed0/cp_180/predictions/test_pred_real.npy"),
        ("Model H (Node Weight)", "output/pipeline_model_H_seed0/cp_180/predictions/test_pred_real.npy"),
        ("Model I (New Full 3D)", "output/pipeline_model_I_seed0/cp_180/predictions/test_pred_real.npy"),
        ("Model J (Final Distilled)", "output/pipeline_distilled_final/cp_180/predictions/test_pred_real.npy"),
        ("Model K (Final Source-Switch Posthoc)", "output/pipeline_source_switch_posthoc_seed0/cp_180/predictions/test_pred_real.npy"),
    ]
    
    y_true_path = "output/pipeline_gefs15_seed0/cp_180/predictions/test_true_real.npy"
    if not Path(y_true_path).exists():
        print("Test true data not found.")
        return
        
    y_true = np.load(y_true_path)

    for name, path in models_to_eval:
        if Path(path).exists():
            y_p = np.load(path)
            records.append({
                'model': name,
                'global_mae': mae(y_true, y_p),
                'rain_sensitive_mae': mae(y_true, y_p, mask_rain),
                'rain_top_mae': mae(y_true, y_p, mask_top),
                'h3_rain_sensitive_mae': h_mae(y_true, y_p, 2, mask_rain),
                'h5_rain_sensitive_mae': h_mae(y_true, y_p, 4, mask_rain)
            })

    df = pd.DataFrame(records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_audit.csv", index=False)
    
    # Save markdown summary
    md_lines = ["# 3D Weight Ablation Audit (cp_180)\n\n"]
    if not df.empty:
        # Format markdown table
        md_lines.append("| model | global_mae | rain_sensitive_mae | rain_top_mae | h3_rain_sensitive_mae | h5_rain_sensitive_mae |")
        md_lines.append("|---|---|---|---|---|---|")
        for _, row in df.iterrows():
            md_lines.append(f"| {row['model']} | {row['global_mae']:.3f} | {row['rain_sensitive_mae']:.3f} | {row['rain_top_mae']:.3f} | {row['h3_rain_sensitive_mae']:.3f} | {row['h5_rain_sensitive_mae']:.3f} |")
    md_lines.append("\n")
        
    (out_dir / "ablation_audit_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("Ablation audit generated at", out_dir)

if __name__ == "__main__":
    main()
