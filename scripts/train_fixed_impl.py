 # trainer2.py —— 终极学术优化版 (包含物理趋势约束与阶段自适应学习率)

# ====================== 第一道防线：最严格的警告屏蔽 ======================
import os
import shutil
import sys
import warnings
import time
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 针对 numpy 1.20+ 和 TF 1.15 的底层 C 接口警告强力屏蔽
warnings.filterwarnings("ignore", message="Passing.*as a synonym of type is deprecated")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 第二道防线：屏蔽 TensorFlow 内部算子的警告日志
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

try:
    from data_loader.date_loader import load_preprocessed
    from models.base_model import build_model, model_save

    print("✅ 所有核心模块导入成功！")
except ModuleNotFoundError as e:
    print(f"❌ 导入失败！报错信息: {e}")
    sys.exit(1)

# ========================= 数据加载与拓扑注入 =========================
print(">> 正在加载数据集与邻接矩阵...")
processed_dir = os.path.join(project_root, "dataset", "processed")
dataset = load_preprocessed(save_path=processed_dir)

train_data = dataset.get_data('train')
train_stage = dataset.get_stage('train')
train_rain = dataset.get_rain('train')
rain_susceptibility = dataset.get_rain_susceptibility()
hydro_flow_weight = dataset.get_hydro_flow_weight()
val_data = dataset.get_data('val')
val_stage = dataset.get_stage('val')
val_rain = dataset.get_rain('val')

# 【关键修复点】：提取邻接矩阵，并注入到 TensorFlow 全局集合中
adj_matrix = dataset.get_adj()
tf.compat.v1.add_to_collection('graph_kernel', tf.constant(adj_matrix, dtype=tf.float32))
print("✅ 空间拓扑邻接矩阵注入成功！")

# ========================= 构建计算图 =========================
print(">> 正在构建基于 CPD 阶段感知的 STGCN 模型...")

# 超参数配置 (请与你的测试集配置保持严格一致)
n_his = 12
n_pred = 5
n_route = 5241
Ks, Kt = 3, 3
blocks = [[1, 8, 16], [16, 8, 16]]
batch_size = 4
epoch = 60
initial_lr = 0.001
copy_loss_weight = 0.5

x = tf.compat.v1.placeholder(tf.float32, [None, n_his, n_route, 1], name='data_input')
y_true = tf.compat.v1.placeholder(tf.float32, [None, n_pred, n_route, 1], name='data_label')
stage_input = tf.compat.v1.placeholder(tf.int32, [None, n_his], name='stage_input')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
rain_input = None
if train_rain is not None:
    rain_input = tf.compat.v1.placeholder(tf.float32, [None, n_his, train_rain.shape[2]], name='rain_input')
    print(f">> 已启用降雨条件特征: {train_rain.shape[2]} 个滞后变量")
rain_susceptibility_tensor = None
hydro_flow_weight_tensor = None
dynamic_rain_graph = False
if train_rain is not None and rain_susceptibility is not None:
    rain_susceptibility_tensor = tf.constant(rain_susceptibility, dtype=tf.float32, name='rain_susceptibility')
    dynamic_rain_graph = True
    print(f">> 已启用雨致动态图门控: rain_susceptibility={rain_susceptibility.shape}")
    if hydro_flow_weight is not None:
        print(f">> 已检测到下坡水文传播权重: hydro_flow_weight={hydro_flow_weight.shape}，固定训练默认不启用粗门控")

# 🌟 新增：由外部传入的动态学习率占位符
lr_placeholder = tf.compat.v1.placeholder(tf.float32, name='dynamic_lr')

# 调用 build_model
y_pred, c_loss = build_model(
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

# CPD-aware sample weights: if the historical window crosses a detected
# change point, the residual learner receives more gradient on that sample.
stage_float = tf.cast(stage_input, tf.float32)
stage_span = tf.reduce_max(stage_float, axis=1) - tf.reduce_min(stage_float, axis=1)
has_cpd_transition = tf.cast(stage_span > 0.0, tf.float32)
stage_level = tf.reduce_mean(stage_float, axis=1) / 5.0
cpd_sample_weight = 1.0 + 0.7 * has_cpd_transition + 0.3 * stage_level
if rain_input is not None:
    rain_intensity = tf.reduce_mean(tf.nn.relu(rain_input), axis=[1, 2])
    cpd_sample_weight += 0.2 * tf.minimum(rain_intensity, 3.0)
cpd_sample_weight = tf.reshape(cpd_sample_weight, [-1, 1, 1, 1])

# 🌟🌟🌟 核心优化 1：构建物理约束联合损失函数 🌟🌟🌟
# 1. Robust residual displacement loss.
# The model predicts y(t+1)=y(t)+delta, so this loss directly supervises the
# learned deformation increment rather than only the absolute displacement.
last_step_x = x[:, -1:, :, :]
diff_pred = y_pred - last_step_x
diff_true = y_true - last_step_x
residual_error = diff_pred - diff_true
abs_residual_error = tf.abs(residual_error)
quadratic = tf.minimum(abs_residual_error, 1.0)
huber_per_value = 0.5 * tf.square(quadratic) + (abs_residual_error - quadratic)
huber_loss = tf.reduce_mean(cpd_sample_weight * huber_per_value)

# Keep the absolute error as a secondary anchor for paper-friendly MAE/RMSE.
mse_loss = tf.reduce_mean(cpd_sample_weight * tf.square(y_pred - y_true))

# 2. 基础 MAE (L1)：对极端异常值/环境噪声更鲁棒
mae_loss = tf.reduce_mean(cpd_sample_weight * tf.abs(y_pred - y_true))

# 3. 物理趋势损失 (Trend/Direction Penalty)：惩罚反向预测，解决滞后问题
dir_penalty = tf.reduce_mean(cpd_sample_weight * tf.nn.relu(-(diff_pred * diff_true)))

# Emphasize active deformation samples so the network can improve over the
# persistence baseline on meaningful movement instead of over-smoothing.
motion_weight = 1.0 + 2.0 * tf.minimum(tf.abs(diff_true), 2.0)
weighted_residual_mae = tf.reduce_mean(cpd_sample_weight * motion_weight * tf.abs(residual_error))

# 最终总损失 = residual Huber(主导) + weighted residual MAE + absolute MAE + Trend + small delta regularization
total_loss = huber_loss + 0.2 * mse_loss + 0.5 * weighted_residual_mae + 0.2 * mae_loss + 0.05 * dir_penalty + copy_loss_weight * c_loss

optimizer = tf.compat.v1.train.AdamOptimizer(lr_placeholder)
train_op = optimizer.minimize(total_loss)

# ========================= 执行训练 =========================
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 限制只保留最新的 1 个最佳模型
saver = tf.compat.v1.train.Saver(max_to_keep=1)
model_path = os.path.join(project_root, "output", "models")

# 自动清理旧实验残留
if os.path.exists(model_path):
    print(f"\n>> 🧹 正在清理上一次实验的旧模型文件...")
    shutil.rmtree(model_path)
os.makedirs(model_path, exist_ok=True)

print(f"\n>> 开始训练，样本数: {len(train_data)}")

best_val_loss = float('inf')
patience = 8  # 早停容忍度
wait_epochs = 0
num_batch = int(np.ceil(len(train_data) / batch_size))

for i in range(epoch):
    t1 = time.time()
    epoch_loss = 0.0
    order = np.random.permutation(len(train_data))

    # 训练循环
    for j in range(num_batch):
        start = j * batch_size
        end = min(start + batch_size, len(train_data))
        batch_idx = order[start:end]

        batch_x = train_data[batch_idx, :n_his, :, :]
        # Train the residual model as a stable one-step predictor. During
        # testing, tester.py rolls this one-step model forward recursively for
        # multi-horizon evaluation.
        batch_y = train_data[batch_idx, n_his:n_his + n_pred, :, :]

        batch_s = train_stage[batch_idx]
        batch_rain = train_rain[batch_idx] if train_rain is not None else None

        # 🌟🌟🌟 核心优化 2：阶段自适应课程学习 (Stage-Adaptive LR) 🌟🌟🌟
        # 1. 基础衰减：每 10 个 Epoch，基础学习率打 8 折，保证后期收敛平稳
        base_lr = initial_lr * (0.8 ** (i // 10))

        # 2. 突变感知激活：如果当前批次数据包含 Stage 7 或 Stage 8 (通常是加速或突变期)
        # 计算当前batch平均stage
        stage_mean = np.mean(batch_s)

        # 基础学习率衰减
        base_lr = initial_lr * (0.8 ** (i // 10))

        # 🌟 Stage-aware 平滑调度（核心）
        if stage_mean < 1.5:
            lr_scale = 0.8  # 平稳期 → 小学习率（抗噪）
        elif stage_mean < 2.5:
            lr_scale = 1.0  # 过渡期 → 正常
        else:
            lr_scale = 1.2  # 变形期 → 稍微放大

        current_batch_lr = base_lr * lr_scale

        feed = {
            x: batch_x,
            y_true: batch_y,
            stage_input: batch_s,
            keep_prob: 0.8,
            lr_placeholder: current_batch_lr  # 注入动态计算出的学习率
        }
        if rain_input is not None:
            feed[rain_input] = batch_rain
        _, loss_val = sess.run([train_op, total_loss], feed_dict=feed)
        epoch_loss += loss_val

    # 验证环节
    val_feed = {
        x: val_data[:, :n_his, :, :],
        y_true: val_data[:, n_his:n_his + n_pred, :, :],
        stage_input: val_stage,
        keep_prob: 1.0,
        lr_placeholder: base_lr  # 验证集不更新权重，传基础值即可
    }
    if rain_input is not None:
        val_feed[rain_input] = val_rain
    val_loss = sess.run(total_loss, feed_dict=val_feed)

    print(
        f"Epoch {i + 1:3d}/{epoch} | Train Loss: {epoch_loss / num_batch:.4f} | Val Loss: {val_loss:.4f} | Base LR: {base_lr:.6f} | Time: {time.time() - t1:.1f}s")

    # 🌟 核心修复 3：早停与安全覆盖机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait_epochs = 0  # 破记录，重置耐心值

        # 兼容你的原生保存逻辑
        try:
            model_save(sess, saver, model_path, i + 1)
        except Exception:
            pass

        save_path_name = os.path.join(model_path, 'best_model')
        # 覆盖保存为 'best_model'，供 tester.py 固定读取
        saver.save(sess, save_path_name)
        print(f">> 🌟 模型已在 Epoch {i + 1} 刷新最佳记录，覆盖保存成功！")
    else:
        wait_epochs += 1
        if wait_epochs >= patience:
            print(f"\n⚠️ 验证集 Loss 已经连续 {patience} 轮未下降，触发 Early Stopping (早停) 机制！")
            print(f">> 最终保留的最佳模型 Val Loss: {best_val_loss:.4f}")
            break

print("\n🎉 训练流程完成,运行 tester.py 看看优化后的结果")
