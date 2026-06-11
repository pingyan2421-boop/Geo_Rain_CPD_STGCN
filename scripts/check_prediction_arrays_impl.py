from pathlib import Path
import numpy as np

base = Path(r"C:\Users\lenovo\Desktop\STGCNa\STGCN-Landslide-Prediction\output")
for name in [
    "y_test_pred_real.npy",
    "y_test_true_real.npy",
    "y_test_persistence_real.npy",
    "pred_array.npy",
    "y_test.npy",
]:
    path = base / name
    if path.exists():
        arr = np.load(path)
        print(name, arr.shape, float(np.nanmin(arr)), float(np.nanmax(arr)))
    else:
        print(name, "missing")
