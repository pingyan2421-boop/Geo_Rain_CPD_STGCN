import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Passing.*as a synonym of type is deprecated")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.compat.v1.disable_eager_execution()

current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
data_loader_dir = os.path.join(project_root, "data_loader")

if data_loader_dir not in sys.path:
    sys.path.insert(0, data_loader_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_loader.date_loader import load_preprocessed
from models.base_model import build_model
from models.tester import direct_pred, inverse_transform, evaluate_metrics


N_HIS = 21
N_PRED = 5
N_ROUTE = 245
KS = 3
KT = 3
BLOCKS = [[1, 32, 64], [64, 32, 64]]
BATCH_SIZE = 64


def damped_velocity_baseline(history_real, n_pred, alpha, k):
    last = history_real[:, -1:, :, :]
    recent = np.mean(np.diff(history_real[:, -k:, :, :], axis=1), axis=1, keepdims=True)
    steps = np.arange(1, n_pred + 1, dtype=np.float32).reshape(1, n_pred, 1, 1)
    return last - alpha * steps * recent


def node_gain(pred, true, base):
    pred_delta = pred - base
    true_delta = true - base
    numerator = np.sum(pred_delta * true_delta, axis=(0, 1, 3), keepdims=True)
    denominator = np.sum(np.square(pred_delta), axis=(0, 1, 3), keepdims=True) + 1e-6
    return np.clip(numerator / denominator, -1.5, 1.5)


def make_folds(n_samples, n_folds=4):
    fold_size = n_samples // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        valid_start = train_end
        valid_end = min(valid_start + fold_size, n_samples)
        if train_end > 0 and valid_end > valid_start:
            folds.append((np.arange(0, train_end), np.arange(valid_start, valid_end)))
    return folds


def main():
    processed_dir = os.path.join(project_root, "dataset", "processed")
    dataset = load_preprocessed(save_path=processed_dir)
    tf.compat.v1.add_to_collection("graph_kernel", tf.constant(dataset.get_adj(), dtype=tf.float32))

    x = tf.compat.v1.placeholder(tf.float32, [None, N_HIS, N_ROUTE, 1], name="data_input")
    stage_input = tf.compat.v1.placeholder(tf.int32, [None, N_HIS], name="stage_input")
    keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
    y_pred, _ = build_model(x, stage_input, N_HIS, KS, KT, BLOCKS, keep_prob, n_pred=N_PRED)

    val_data = dataset.get_data("val")
    val_stage = dataset.get_stage("val")
    test_data = dataset.get_data("test")
    test_stage = dataset.get_stage("test")
    stats = dataset.get_stats()

    saver = tf.compat.v1.train.Saver()
    model_dir = os.path.join(project_root, "output", "models")
    ckpt = os.path.join(model_dir, "best_model")

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, ckpt)
        val_raw = inverse_transform(direct_pred(sess, y_pred, val_data, val_stage, BATCH_SIZE, N_HIS), stats)
        test_raw = inverse_transform(direct_pred(sess, y_pred, test_data, test_stage, BATCH_SIZE, N_HIS), stats)

    val_true = inverse_transform(val_data[:, N_HIS:N_HIS + N_PRED, :, :], stats)
    val_hist = inverse_transform(val_data[:, :N_HIS, :, :], stats)
    test_true = inverse_transform(test_data[:, N_HIS:N_HIS + N_PRED, :, :], stats)
    test_hist = inverse_transform(test_data[:, :N_HIS, :, :], stats)

    folds = make_folds(len(val_data), n_folds=4)
    alphas = np.round(np.linspace(0.55, 1.05, 11), 3)
    ks = [6, 10, 15, 21]
    shrinks = np.round(np.linspace(0.0, 0.2, 9), 3)

    rows = []
    for k in ks:
        for alpha in alphas:
            val_base_all = damped_velocity_baseline(val_hist, N_PRED, alpha, k)
            test_base = damped_velocity_baseline(test_hist, N_PRED, alpha, k)
            fold_scores = []
            for fit_idx, score_idx in folds:
                gain = node_gain(val_raw[fit_idx], val_true[fit_idx], val_base_all[fit_idx])
                for shrink in shrinks:
                    pred = val_base_all[score_idx] + shrink * gain * (val_raw[score_idx] - val_base_all[score_idx])
                    rmse, mae = evaluate_metrics(val_true[score_idx], pred)
                    fold_scores.append((shrink, rmse, mae))

            for shrink in shrinks:
                selected = [s for s in fold_scores if s[0] == shrink]
                cv_rmse = float(np.mean([s[1] for s in selected]))
                cv_mae = float(np.mean([s[2] for s in selected]))

                full_gain = node_gain(val_raw, val_true, val_base_all)
                test_pred = test_base + shrink * full_gain * (test_raw - test_base)
                test_rmse, test_mae = evaluate_metrics(test_true, test_pred)
                rows.append([
                    k, alpha, shrink, cv_rmse, cv_mae,
                    float(test_rmse), float(test_mae), float(np.mean(shrink * full_gain))
                ])

    rows.sort(key=lambda r: (r[3], r[4]))
    out_path = os.path.join(project_root, "output", "rolling_param_validation.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k", "alpha", "shrink", "CV_RMSE", "CV_MAE",
            "Test_RMSE", "Test_MAE", "Mean_Gain"
        ])
        writer.writerows(rows)

    print(f"Saved rolling validation results to: {out_path}")
    print("Top 10 by rolling CV:")
    for row in rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
