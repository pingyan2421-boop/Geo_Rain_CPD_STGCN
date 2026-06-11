import numpy as np
import argparse
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from skimage.metrics import structural_similarity as ssim


def calculate_global_metrics(preds, trues, mape_threshold=0.1):
    # 展平矩阵以计算全局的回归指标
    preds_flat = preds.flatten()
    trues_flat = trues.flatten()

    mse = mean_squared_error(trues_flat, preds_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues_flat, preds_flat)
    # Plain MAPE is kept for compatibility, but it is unstable when true
    # displacement is close to 0. Use Masked MAPE / sMAPE for paper reporting.
    mape = np.mean(np.abs((trues_flat - preds_flat) / (trues_flat + 1e-5))) * 100
    valid = np.abs(trues_flat) > mape_threshold
    masked_mape = np.nan
    if np.any(valid):
        masked_mape = np.mean(np.abs((trues_flat[valid] - preds_flat[valid]) / trues_flat[valid])) * 100
    smape = np.mean(2.0 * np.abs(preds_flat - trues_flat) /
                    (np.abs(trues_flat) + np.abs(preds_flat) + 1e-5)) * 100
    evar = explained_variance_score(trues_flat, preds_flat)

    # SSIM (结构相似性) 计算
    # 按照每个时间步的 1D 空间节点信号计算结构相似度，然后取平均
    ssim_values = []
    if preds.ndim == 3:
        iterator = (
            (trues[i, h], preds[i, h])
            for i in range(preds.shape[0])
            for h in range(preds.shape[1])
        )
    else:
        iterator = ((trues[i], preds[i]) for i in range(preds.shape[0]))

    for t_i, p_i in iterator:
        data_range = t_i.max() - t_i.min()
        if data_range == 0:
            data_range = 1e-5
        s = ssim(t_i, p_i, data_range=data_range, win_size=7)
        ssim_values.append(s)

    mean_ssim = np.mean(ssim_values)

    return rmse, mae, mape, masked_mape, smape, evar, mean_ssim


def process_files(path):
    print(f">> 📂 正在加载测试数据 (目标路径: {path}) ...")

    pred_path = os.path.join(path, 'y_test_pred_real.npy')
    true_path = os.path.join(path, 'y_test_true_real.npy')

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print(f"❌ 找不到数据文件！请确认 {path} 文件夹下是否已生成 y_test_pred_real.npy 和 y_test_true_real.npy")
        return

    # 1. 加载数据
    preds = np.load(pred_path)
    trues = np.load(true_path)

    # 2. 去除多余维度: single-step -> [Batch, Nodes], multi-step -> [Batch, Horizon, Nodes]
    preds = np.squeeze(preds)
    trues = np.squeeze(trues)

    # 3. 计算全部性能指标
    rmse, mae, mape, masked_mape, smape, evar, mean_ssim = calculate_global_metrics(preds, trues)

    # 4. 保存每个节点在每个时间步的绝对误差矩阵 (可用于画 3D 误差热力图)
    abs_error = np.abs(preds - trues)
    if abs_error.ndim == 3:
        # Flatten sample and horizon dimensions so each row is one predicted
        # horizon snapshot and each column is a node.
        error_for_csv = abs_error.reshape(-1, abs_error.shape[-1])
    else:
        error_for_csv = abs_error
    error_df = pd.DataFrame(error_for_csv)

    # 将列名替换为真实的节点序号 (Node_0, Node_1...)
    error_df.columns = [f'Node_{i}' for i in range(error_for_csv.shape[1])]
    error_csv_path = os.path.join(path, 'node_metrics_error.csv')
    error_df.to_csv(error_csv_path, index=False)

    # 5. 打印并保存全局评估指标 (论文里的 Table)
    print("=" * 50)
    print("🚀 全局测试集物理误差评估 (可直接填入论文表格):")
    print("-" * 50)
    print(f"RMSE (均方根误差) : {rmse:.4f} mm")
    print(f"MAE  (平均绝对误差) : {mae:.4f} mm")
    print(f"MAPE (原始平均相对误差) : {mape:.4f} %")
    print(f"Masked MAPE (|真实值|>0.1mm) : {masked_mape:.4f} %")
    print(f"sMAPE (对称平均相对误差) : {smape:.4f} %")
    print(f"EVar (解释方差)     : {evar:.4f} (越接近1越好)")
    print(f"SSIM (结构相似度)   : {mean_ssim:.4f} (越接近1越好)")
    print("=" * 50)

    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'MAE', 'MAPE', 'Masked_MAPE_abs_true_gt_0.1mm', 'sMAPE', 'EVar', 'SSIM'],
        'Value': [rmse, mae, mape, masked_mape, smape, evar, mean_ssim]
    })
    metrics_csv_path = os.path.join(path, 'performance_metrics_final.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    if preds.ndim == 3:
        horizon_rows = []
        for h in range(preds.shape[1]):
            h_rmse, h_mae, h_mape, h_masked_mape, h_smape, h_evar, h_ssim = calculate_global_metrics(
                preds[:, h, :],
                trues[:, h, :]
            )
            horizon_rows.append([h + 1, h_rmse, h_mae, h_mape, h_masked_mape, h_smape, h_evar, h_ssim])
        horizon_df = pd.DataFrame(
            horizon_rows,
            columns=['Horizon', 'RMSE', 'MAE', 'MAPE', 'Masked_MAPE_abs_true_gt_0.1mm', 'sMAPE', 'EVar', 'SSIM']
        )
        horizon_csv_path = os.path.join(path, 'performance_metrics_by_horizon.csv')
        horizon_df.to_csv(horizon_csv_path, index=False)
        print(f"✅ 分步指标报表已保存至: {horizon_csv_path}")

    print(f"✅ 节点误差矩阵已保存至: {error_csv_path}")
    print(f"✅ 全局指标报表已保存至: {metrics_csv_path}")


if __name__ == "__main__":
    # 🌟 核心修复：自动推导项目的绝对根目录，并指向 output 文件夹
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'models' else current_dir
    default_output_path = os.path.join(project_root, "output")

    parser = argparse.ArgumentParser(description='Evaluate STGCN metrics for papers.')
    # 将默认路径改为我们刚刚算出来的绝对路径
    parser.add_argument('--path', type=str, default=default_output_path, help='.npy 文件所在的路径')
    args = parser.parse_args()

    process_files(args.path)
