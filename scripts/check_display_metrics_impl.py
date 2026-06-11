from pathlib import Path
import numpy as np

base = Path(r"C:\Users\lenovo\Desktop\STGCNa\STGCN-Landslide-Prediction\output")
pred = np.load(base / "y_test_pred_real.npy")[:, :, :, 0]
true = np.load(base / "y_test_true_real.npy")[:, :, :, 0]

for h in range(pred.shape[1]):
    e = pred[:, h] - true[:, h]
    print(f"H+{h+1}: RMSE={np.sqrt(np.mean(e * e)):.4f}, MAE={np.mean(np.abs(e)):.4f}")

e = pred - true
print(f"All horizons: RMSE={np.sqrt(np.mean(e * e)):.4f}, MAE={np.mean(np.abs(e)):.4f}")

mean_pred = pred.mean(axis=(0, 1))
mean_true = true.mean(axis=(0, 1))
e = mean_pred - mean_true
print(f"Per-node mean map: RMSE={np.sqrt(np.mean(e * e)):.4f}, MAE={np.mean(np.abs(e)):.4f}")
