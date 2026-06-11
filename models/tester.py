# tester.py —— 论文级测试与评估脚本 (完整可执行版)

# ====================== 第一道防线：最严格的警告屏蔽 ======================
import os
import sys
import warnings
import csv
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Passing.*as a synonym of type is deprecated")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_eager_execution()

# ====================== 路径注入核心逻辑 ======================
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


# ===============================
# 学术级评价指标计算器与日志
# ===============================
def evaluate_metrics(y_true, y_pred):
    """计算 RMSE 和 MAE"""
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae


def print_horizon_metrics(y_true, y_pred, persistence, damped):
    """Print per-horizon metrics for recursive multi-step prediction."""
    print("📏 分步预测误差:")
    rows = []
    for h in range(y_true.shape[1]):
        h_true = y_true[:, h:h + 1, :, :]
        h_pred = y_pred[:, h:h + 1, :, :]
        h_base = persistence[:, h:h + 1, :, :]
        h_damped = damped[:, h:h + 1, :, :]
        rmse, mae = evaluate_metrics(h_true, h_pred)
        b_rmse, b_mae = evaluate_metrics(h_true, h_base)
        d_rmse, d_mae = evaluate_metrics(h_true, h_damped)
        rows.append((h + 1, rmse, mae, b_rmse, b_mae, d_rmse, d_mae))
        print(f"   H+{h + 1}: Model RMSE {rmse:.4f} / MAE {mae:.4f} | Persistence RMSE {b_rmse:.4f} / MAE {b_mae:.4f} | Damped RMSE {d_rmse:.4f} / MAE {d_mae:.4f}")
    return rows


def inverse_transform(data, stats):
    """反归一化：转换回真实的物理位移值"""
    return data * stats['std'] + stats['mean']


def calibrate_residual_gain(val_pred, val_true, val_last):
    """Estimate node-wise residual gates on validation data only."""
    pred_delta = val_pred - val_last
    true_delta = val_true - val_last
    numerator = np.sum(pred_delta * true_delta, axis=(0, 1, 3), keepdims=True)
    denominator = np.sum(np.square(pred_delta), axis=(0, 1, 3), keepdims=True) + 1e-6
    gain = 0.5 * numerator / denominator
    return np.clip(gain, -1.5, 1.5)


def calibrate_global_blend(val_pred, val_true, val_base):
    """Select one validation-only residual weight for stable test reporting."""
    best_alpha = 0.0
    best_score = float("inf")
    for alpha in np.linspace(0.0, 1.0, 41):
        candidate = val_base + alpha * (val_pred - val_base)
        rmse, mae = evaluate_metrics(val_true, candidate)
        score = rmse + 0.25 * mae
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def calibrate_node_residual_gain(val_pred, val_true, val_base, shrink=0.5):
    """Validation-calibrated node-wise residual reliability.

    Each node gets a closed-form least-squares residual gate, clipped to the
    physically conservative range [0, 1] and shrunk toward persistence.
    """
    pred_delta = val_pred - val_base
    true_delta = val_true - val_base
    numerator = np.sum(pred_delta * true_delta, axis=(0, 1, 3), keepdims=True)
    denominator = np.sum(np.square(pred_delta), axis=(0, 1, 3), keepdims=True) + 1e-6
    gain = np.clip(numerator / denominator, 0.0, 1.0)
    return shrink * gain


def direct_pred(sess, y_pred_tensor, seq, stage_seq, batch_size, n_his, rain_seq=None, rain_tensor_name='rain_input:0'):
    n_samples = len(seq)
    n_batches = int(np.ceil(n_samples / batch_size))
    all_preds = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        x_batch = seq[start_idx:end_idx, 0:n_his, :, :]
        stage_b = stage_seq[start_idx:end_idx]
        feed = {
            'data_input:0': x_batch,
            'stage_input:0': stage_b,
            'keep_prob:0': 1.0
        }
        if rain_seq is not None:
            feed[rain_tensor_name] = rain_seq[start_idx:end_idx]
        pred = sess.run(y_pred_tensor, feed_dict=feed)
        if isinstance(pred, list):
            pred = np.array(pred[0])
        all_preds.append(pred)

    return np.concatenate(all_preds, axis=0)


def damped_velocity_baseline(history_real, n_pred, alpha=0.50, k=12):
    last = history_real[:, -1:, :, :]
    recent = np.mean(np.diff(history_real[:, -k:, :, :], axis=1), axis=1, keepdims=True)
    steps = np.arange(1, n_pred + 1, dtype=np.float32).reshape(1, n_pred, 1, 1)
    return last + alpha * steps * recent


def graph_smooth(pred_real, adj_matrix, smooth_weight=0.05):
    """Light spatial consistency smoothing over the terrain graph."""
    row_sum = np.sum(adj_matrix, axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    smooth_kernel = adj_matrix / row_sum
    values = pred_real[:, :, :, 0]
    smoothed = np.einsum('ij,bhj->bhi', smooth_kernel, values)
    blended = (1.0 - smooth_weight) * values + smooth_weight * smoothed
    return blended[:, :, :, np.newaxis]


def log_experiment_results(overall_rmse, overall_mae, stage_metrics, filepath="output/experiment_logs.csv"):
    """自动将实验结果追加到 CSV 报表中"""
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 如果文件不存在，先写表头
        if not file_exists:
            writer.writerow(
                ['实验时间', '全局 RMSE', '全局 MAE', 'Stage 6 RMSE', 'Stage 7 RMSE', 'Stage 8 RMSE', '备注'])

        # 提取各个 Stage 的 RMSE (如果没有该 Stage，则填 -)
        s6_rmse = f"{stage_metrics.get(6, '-'):.4f}" if isinstance(stage_metrics.get(6), float) else "-"
        s7_rmse = f"{stage_metrics.get(7, '-'):.4f}" if isinstance(stage_metrics.get(7), float) else "-"
        s8_rmse = f"{stage_metrics.get(8, '-'):.4f}" if isinstance(stage_metrics.get(8), float) else "-"

        # 写入当前实验数据
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([current_time, f"{overall_rmse:.4f}", f"{overall_mae:.4f}", s6_rmse, s7_rmse, s8_rmse,
                         "STGCN+CPD+FiLM (74%稀疏度)"])
    print(f"📝 本次实验指标已自动记录至: {filepath}")


# ===============================
# 多步预测引擎 (保持你的原版逻辑不变)
# ===============================
def multi_pred(sess, y_pred_tensor, seq, stage_seq, batch_size, n_his, n_pred):
    n_samples = len(seq)
    n_batches = int(np.ceil(n_samples / batch_size))
    all_preds = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        x_batch = seq[start_idx:end_idx]
        test_seq = np.copy(x_batch[:, 0:n_his, :, :])

        stage_b = stage_seq[start_idx:end_idx]

        batch_step_preds = []

        for j in range(n_pred):
            pred = sess.run(y_pred_tensor, feed_dict={
                'data_input:0': test_seq,
                'stage_input:0': stage_b,
                'keep_prob:0': 1.0
            })
            if isinstance(pred, list): pred = np.array(pred[0])

            batch_step_preds.append(pred)

            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1:n_his, :, :] = pred

        batch_step_preds = np.concatenate(batch_step_preds, axis=1)
        all_preds.append(batch_step_preds)

    final_pred_array = np.concatenate(all_preds, axis=0)
    return final_pred_array


# ===============================
# 测试主流程 (修复重复循环 + 加入双轨保存逻辑)
# ===============================
def model_test_academic(sess, pred_tensor, dataset, batch_size, n_his, n_pred):
    x_test = dataset.get_data('test')
    x_val = dataset.get_data('val')
    stage_test = dataset.get_stage('test')
    stage_val = dataset.get_stage('val')
    rain_test = dataset.get_rain('test')
    rain_val = dataset.get_rain('val')
    stats = dataset.get_stats()

    print("\n" + "=" * 50)
    print("🚀 正在执行测试集推理与反归一化评估...")

    y_val_pred = direct_pred(sess, pred_tensor, x_val, stage_val, batch_size, n_his, rain_seq=rain_val)
    y_val_true = x_val[:, n_his:n_his + n_pred, :, :]
    y_val_last = x_val[:, n_his - 1:n_his, :, :]

    y_val_pred_real = inverse_transform(y_val_pred, stats)
    y_val_true_real = inverse_transform(y_val_true, stats)
    y_val_last_real = inverse_transform(y_val_last, stats)
    y_val_hist_real = inverse_transform(x_val[:, :n_his, :, :], stats)
    y_val_base_real = np.repeat(y_val_last_real, n_pred, axis=1)
    node_gain = calibrate_node_residual_gain(y_val_pred_real, y_val_true_real, y_val_base_real, shrink=0.4)

    y_test_pred = direct_pred(sess, pred_tensor, x_test, stage_test, batch_size, n_his, rain_seq=rain_test)
    y_test_true = x_test[:, n_his:n_his + n_pred, :, :]

    y_test_pred_raw_real = inverse_transform(y_test_pred, stats)
    y_test_true_real = inverse_transform(y_test_true, stats)
    y_test_last_real = inverse_transform(x_test[:, n_his - 1:n_his, :, :], stats)
    y_test_hist_real = inverse_transform(x_test[:, :n_his, :, :], stats)
    y_test_base_real = np.repeat(y_test_last_real, n_pred, axis=1)
    y_test_damped_real = damped_velocity_baseline(y_test_hist_real, n_pred)
    y_test_damped_smooth_real = graph_smooth(y_test_damped_real, dataset.get_adj(), smooth_weight=0.05)
    y_test_pred_real = y_test_base_real + node_gain * (y_test_pred_raw_real - y_test_base_real)

    overall_rmse, overall_mae = evaluate_metrics(y_test_true_real, y_test_pred_real)
    raw_rmse, raw_mae = evaluate_metrics(y_test_true_real, y_test_pred_raw_real)
    base_rmse, base_mae = evaluate_metrics(y_test_true_real, y_test_base_real)
    damped_rmse, damped_mae = evaluate_metrics(y_test_true_real, y_test_damped_real)
    smooth_rmse, smooth_mae = evaluate_metrics(y_test_true_real, y_test_damped_smooth_real)
    print("-" * 50)
    print(f"🌍 全局测试集物理误差 | RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f}")
    print(f"🧭 未校准模型输出 | RMSE: {raw_rmse:.4f} | MAE: {raw_mae:.4f}")
    print(f"📌 Persistence基线 | RMSE: {base_rmse:.4f} | MAE: {base_mae:.4f}")
    print(f"🧱 速度阻尼基线 | RMSE: {damped_rmse:.4f} | MAE: {damped_mae:.4f}")
    print(f"🧩 图平滑阻尼基线 | RMSE: {smooth_rmse:.4f} | MAE: {smooth_mae:.4f}")
    print(f"🎚️ 节点级残差可信度 | mean: {np.mean(node_gain):.4f} | min: {np.min(node_gain):.4f} | max: {np.max(node_gain):.4f}")
    print("-" * 50)
    horizon_rows = print_horizon_metrics(y_test_true_real, y_test_pred_real, y_test_base_real, y_test_damped_real)

    # 统一计算各 Stage 的误差并收集
    stage_metrics = {}
    stage_eval = stage_test[:, -1] if stage_test.ndim == 2 else stage_test
    unique_stages = np.unique(stage_eval)
    for s in unique_stages:
        idx = np.where(stage_eval == s)[0]
        if len(idx) == 0: continue

        stage_true = y_test_true_real[idx]
        stage_pred = y_test_pred_real[idx]

        s_rmse, s_mae = evaluate_metrics(stage_true, stage_pred)
        stage_metrics[s] = s_rmse  # 收集每个 stage 的 RMSE
        print(f"📊 演化 Stage {s} 评估 (样本数: {len(idx):4d}) | RMSE: {s_rmse:.4f} | MAE: {s_mae:.4f}")

    # 🌟 调用日志记录器 🌟
    log_experiment_results(overall_rmse, overall_mae, stage_metrics,
                           filepath=os.path.join(project_root, "output", "experiment_logs.csv"))

    # ==========================================
    # 🌟 统一数据保存逻辑 (适配所有后续脚本) 🌟
    # ==========================================
    os.makedirs(os.path.join(project_root, 'output'), exist_ok=True)

    # 1. 保存全量数据 —— 供 evaluator.py (算 SSIM/EVar) 使用
    np.save(os.path.join(project_root, 'output', 'y_test_pred_real.npy'), y_test_pred_real)
    np.save(os.path.join(project_root, 'output', 'y_test_true_real.npy'), y_test_true_real)
    np.save(os.path.join(project_root, 'output', 'y_test_persistence_real.npy'), y_test_base_real)
    np.save(os.path.join(project_root, 'output', 'y_test_damped_real.npy'), y_test_damped_real)
    np.save(os.path.join(project_root, 'output', 'y_test_damped_smooth_real.npy'), y_test_damped_smooth_real)
    with open(os.path.join(project_root, 'output', 'horizon_metrics.csv'), mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Horizon', 'Model_RMSE', 'Model_MAE', 'Persistence_RMSE', 'Persistence_MAE', 'Damped_RMSE', 'Damped_MAE'])
        writer.writerows(horizon_rows)
    print(f"   [OK] 全量预测数据已存至: output/y_test_pred_real.npy")

    # 2. 提取并保存精简版数据 —— 供 plotter.py (画论文对比曲线) 使用
    node_errors = np.mean(np.abs(y_test_true_real - y_test_pred_real), axis=(0, 1, 3))
    best_nodes = np.argsort(node_errors)[:3]
    worst_nodes = np.argsort(node_errors)[-3:]

    plot_data = {
        'best_nodes': best_nodes,
        'worst_nodes': worst_nodes,
        'true_best': y_test_true_real[:, :, best_nodes, :],
        'pred_best': y_test_pred_real[:, :, best_nodes, :],
        'true_worst': y_test_true_real[:, :, worst_nodes, :],
        'pred_worst': y_test_pred_real[:, :, worst_nodes, :]
    }
    np.save(os.path.join(project_root, 'output', 'paper_plot_data.npy'), plot_data)
    print(f"   [OK] 精简画图数据已存至: output/paper_plot_data.npy")

    print("=" * 50)
    print("✅ 真实尺度预测结果已全部分发完毕，可直接运行 evaluator.py 和 plotter.py！")


# ===============================
# 🚀 执行入口：加载模型与跑测试
# ===============================
if __name__ == '__main__':
    # 1. 超参数 (必须与 trainer 保持一致)
    n_his = 12
    n_route = 5241
    Ks, Kt = 3, 3
    blocks = [[1, 8, 16], [16, 8, 16]]
    batch_size = 4
    n_pred = 5

    # 2. 加载数据与注入图矩阵
    print(">> 加载测试数据与拓扑结构...")
    processed_dir = os.path.join(project_root, "dataset", "processed")
    dataset = load_preprocessed(save_path=processed_dir)
    adj_matrix = dataset.get_adj()
    tf.compat.v1.add_to_collection('graph_kernel', tf.constant(adj_matrix, dtype=tf.float32))

    # 3. 搭建计算图 (完全保留你的参数结构)
    print(">> 构建模型并准备加载权重...")
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his, n_route, 1], name='data_input')
    stage_input = tf.compat.v1.placeholder(tf.int32, [None, n_his], name='stage_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    rain_test = dataset.get_rain('test')
    rain_input = None
    if rain_test is not None:
        rain_input = tf.compat.v1.placeholder(tf.float32, [None, n_his, rain_test.shape[2]], name='rain_input')
        print(f">> 已启用降雨条件特征: {rain_test.shape[2]} 个滞后变量")
    rain_susceptibility = dataset.get_rain_susceptibility()
    rain_susceptibility_tensor = None
    hydro_flow_weight = dataset.get_hydro_flow_weight()
    hydro_flow_weight_tensor = None
    dynamic_rain_graph = False
    if rain_test is not None and rain_susceptibility is not None:
        rain_susceptibility_tensor = tf.constant(rain_susceptibility, dtype=tf.float32, name='rain_susceptibility')
        dynamic_rain_graph = True
        print(f">> 已启用雨致动态图门控: rain_susceptibility={rain_susceptibility.shape}")
        if hydro_flow_weight is not None:
            print(f">> 已检测到下坡水文传播权重: hydro_flow_weight={hydro_flow_weight.shape}，固定测试默认不启用粗门控")

    y_pred, _ = build_model(
        x,
        stage_input,
        n_his,
        Ks,
        Kt,
        blocks,
        keep_prob,
        n_pred=n_pred,
        rain_features=rain_input,
        rain_susceptibility=rain_susceptibility_tensor,
        hydro_flow_weight=hydro_flow_weight_tensor,
        dynamic_rain_graph=dynamic_rain_graph,
    )

    # 4. 加载最佳模型权重并测试
    saver = tf.compat.v1.train.Saver()
    model_dir = os.path.join(project_root, "output", "models")

    with tf.compat.v1.Session() as sess:
        # Prefer the copied project's local best_model. TensorFlow checkpoint
        # files may contain absolute paths from the original PyCharm project.
        local_best = os.path.join(model_dir, "best_model")
        ckpt = local_best if os.path.exists(local_best + ".index") else tf.train.latest_checkpoint(model_dir)
        if ckpt:
            print(f">> 成功找到检查点: {ckpt}，正在恢复权重...")
            saver.restore(sess, ckpt)
            # 开始进行学术级测试
            model_test_academic(sess, y_pred, dataset, batch_size, n_his, n_pred)
        else:
            print(f"❌ 未能在 {model_dir} 找到模型检查点，请确认是否成功运行过 trainer2.py！")
