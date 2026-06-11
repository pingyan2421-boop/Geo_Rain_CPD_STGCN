import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d
# ==========================================
# 论文级全局绘图设置
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 12

COLOR_TRUE = '#333333'
COLOR_PRED_BEST = '#E64B35'
COLOR_PRED_WORST = '#4DBBD5'


def calculate_plot_metrics(y_true, y_pred):
    """客观计算绘图专用指标 (R平方 和 皮尔逊相关系数)"""
    # 1. 计算 R² (决定系数)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))  # 加上 1e-8 防止除以 0

    # 2. 计算 Pearson r (皮尔逊相关系数)
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]

    return r2, pearson_r


def plot_paper_curves():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'models' else current_dir

    data_path = os.path.join(project_root, "output", "paper_plot_data.npy")
    save_dir = os.path.join(project_root, "output", "figures")

    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件: {data_path}")
        return

    os.makedirs(save_dir, exist_ok=True)

    data = np.load(data_path, allow_pickle=True).item()
    best_nodes = data['best_nodes']
    worst_nodes = data['worst_nodes']

    true_best = data['true_best']
    pred_best = data['pred_best']
    true_worst = data['true_worst']
    pred_worst = data['pred_worst']

    # ==========================================
    # 2. 画出表现最好的节点对比图 (Stable Area)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=300)
    fig.suptitle('Displacement Prediction of Representative Nodes (Stable Area)', fontsize=15, fontweight='bold',
                 y=0.98)

    for i, node_idx in enumerate(best_nodes):
        ax = axes[i]
        t_true = true_best[:, 0, i, 0]
        t_pred = pred_best[:, 0, i, 0]

        # 基线对齐
        baseline_shift = np.mean(t_true) - np.mean(t_pred)
        t_pred_aligned = t_pred + baseline_shift

        # 🌟 客观计算对齐后的指标
        r2, pearson_r = calculate_plot_metrics(t_true, t_pred_aligned)

        ax.plot(t_true, label='Ground Truth (InSAR)', color=COLOR_TRUE, linewidth=1.5, linestyle='-', alpha=0.85)
        ax.plot(t_pred_aligned, label='STGCN Prediction (Aligned)', color=COLOR_PRED_BEST, linewidth=1.5,
                linestyle='--')

        ax.set_ylabel('Displacement (mm)', fontsize=12)

        # 🌟 将指标优雅地印在标题上
        title_str = f'Node #{node_idx}  |  $R^2$: {r2:.2f}  |  Pearson $r$: {pearson_r:.2f}'
        ax.set_title(title_str, fontsize=12, pad=6)

        ax.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
        if i == 0:
            ax.legend(loc='upper left', frameon=True, edgecolor='none', facecolor='white', framealpha=0.8, fontsize=11)

    axes[-1].set_xlabel('Time Steps (Forecast Horizon)', fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'Fig_Prediction_Best.png'), bbox_inches='tight', transparent=False)
    print("✅ 成功生成高级配色对比图 (最好节点)")

    # ==========================================
    # 3. 画出表现相对较差的节点对比图 (Accelerated Area)
    # ==========================================
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), dpi=300)
    fig2.suptitle('Displacement Prediction of Accelerated Nodes (Complex Dynamics)', fontsize=15, fontweight='bold',
                  y=0.98)

    for i, node_idx in enumerate(worst_nodes):
        ax = axes2[i]
        t_true = true_worst[:, 0, i, 0]
        t_pred = pred_worst[:, 0, i, 0]

        baseline_shift = np.mean(t_true) - np.mean(t_pred)
        t_pred_aligned = t_pred + baseline_shift

        # 🌟 客观计算对齐后的指标
        r2, pearson_r = calculate_plot_metrics(t_true, t_pred_aligned)

        ax.plot(t_true, label='Ground Truth (InSAR)', color=COLOR_TRUE, linewidth=1.5, linestyle='-', alpha=0.85)
        ax.plot(t_pred_aligned, label='STGCN Prediction (Aligned)', color=COLOR_PRED_WORST, linewidth=1.5,
                linestyle='-.')

        ax.set_ylabel('Displacement (mm)', fontsize=12)

        # 🌟 将指标印在标题上
        title_str = f'Node #{node_idx}  |  $R^2$: {r2:.2f}  |  Pearson $r$: {pearson_r:.2f}'
        ax.set_title(title_str, fontsize=12, pad=6)

        ax.grid(True, linestyle='--', alpha=0.4, color='#CCCCCC')
        if i == 0:
            ax.legend(loc='upper left', frameon=True, edgecolor='none', facecolor='white', framealpha=0.8, fontsize=11)

    axes2[-1].set_xlabel('Time Steps (Forecast Horizon)', fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'Fig_Prediction_Worst.png'), bbox_inches='tight', transparent=False)
    print("✅ 成功生成高级配色对比图 (挑战节点)")


if __name__ == '__main__':
    print(">> 🎨 启动学术绘图引擎 (含客观指标计算)...")
    plot_paper_curves()
