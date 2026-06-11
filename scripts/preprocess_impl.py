# main2.py —— CPD-STGCN 数据预处理执行脚本
# 位置建议：STGCN-Landslide-Prediction/models/main2.py

import os
import sys
import warnings
import argparse
import inspect
import numpy as np

# ==================== 屏蔽无关警告 ====================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ==================== 路径配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))

# 如果 main2.py 放在 models 文件夹，则项目根目录是上一层
# 如果 main2.py 放在项目根目录，则项目根目录就是当前目录
if os.path.basename(current_dir).lower() == "models":
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir

data_loader_path = os.path.join(project_root, "data_loader")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

if data_loader_path not in sys.path:
    sys.path.insert(0, data_loader_path)


# ==================== 导入 data_gen ====================
try:
    # 优先按项目包结构导入
    from data_loader.date_loader import data_gen
    from data_loader.cpd_methods import CPD_METHODS
except ModuleNotFoundError:
    # 兼容你之前的写法
    from date_loader import data_gen
    from cpd_methods import CPD_METHODS


# ==================== 参数配置 ====================
parser = argparse.ArgumentParser(description="CPD-STGCN 数据预处理脚本")

# ---------- 基础窗口参数 ----------
parser.add_argument(
    "--n_his",
    type=int,
    default=12,
    help="历史窗口长度"
)

parser.add_argument(
    "--n_pred",
    type=int,
    default=5,
    help="预测步长"
)

# ---------- CPD 参数 ----------
parser.add_argument(
    "--cpd_mode",
    type=str,
    default="mix",
    choices=["mean", "std", "top", "mix"],
    help="CPD信号类型：mean/std/top/mix"
)

parser.add_argument(
    "--cpd_method",
    type=str,
    default="binseg",
    choices=list(CPD_METHODS),
    help="变化点检测方法：binseg为当前默认，pelt/bfast用于敏感性对照"
)

parser.add_argument(
    "--cpd_cost",
    type=str,
    default="l2",
    help="ruptures代价模型，binseg/pelt默认使用l2"
)

parser.add_argument(
    "--cpd_penalty",
    type=float,
    default=10,
    help="CPD惩罚系数，越大检测到的变化点越少"
)

parser.add_argument(
    "--cpd_top_ratio",
    type=float,
    default=0.10,
    help="CPD强变形节点比例，例如0.10表示取前10%%强变形节点"
)

parser.add_argument(
    "--cpd_min_size",
    type=int,
    default=10,
    help="CPD最小阶段长度，避免阶段切分过碎"
)

# ---------- 邻接矩阵参数 ----------
parser.add_argument(
    "--adj_threshold",
    type=float,
    default=0.23,
    help="邻接矩阵过滤阈值，越大图越稀疏"
)

# ---------- 数据切分参数 ----------
parser.add_argument(
    "--n_train",
    type=int,
    default=160,
    help="训练集时间长度"
)

parser.add_argument(
    "--n_val",
    type=int,
    default=40,
    help="验证集时间长度"
)

parser.add_argument(
    "--n_test",
    type=int,
    default=46,
    help="测试集时间长度"
)

# ---------- 文件路径参数 ----------
parser.add_argument(
    "--file_path",
    type=str,
    default=None,
    help="原始CSV数据路径；默认使用 dataset/inter228_5241a.csv"
)

parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="预处理结果保存路径；默认使用 dataset/processed"
)

parser.add_argument(
    "--pelt_penalty",
    type=float,
    default=None,
    help="PELT惩罚项；默认复用--cpd_penalty"
)

parser.add_argument(
    "--bfast_frequency",
    type=int,
    default=23,
    help="BFAST时间序列频率；12天InSAR约为每年23期"
)

parser.add_argument(
    "--rainfall_features",
    type=str,
    default=None,
    help="CHIRPS日降雨CSV路径；提供后会生成与InSAR日期对齐的降雨滞后特征"
)

parser.add_argument(
    "--rain_susceptibility",
    type=str,
    default=None,
    help="节点雨敏指数CSV路径；提供后写入预处理包用于雨致动态图门控"
)

args = parser.parse_args()


# ==================== 文件路径配置 ====================
if args.file_path is None:
    file_path = os.path.join(project_root, "dataset", "inter228_5241.csv")
else:
    file_path = args.file_path

if args.save_dir is None:
    processed_dir = os.path.join(project_root, "dataset", "processed")
else:
    processed_dir = args.save_dir

os.makedirs(processed_dir, exist_ok=True)

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"\n❌ 找不到输入数据文件:\n{file_path}\n"
        f"请检查 inter228_5241a.csv 是否在 dataset 文件夹下，"
        f"或使用 --file_path 手动指定路径。"
    )


# ==================== 数据切分配置 ====================
data_config = (args.n_train, args.n_val, args.n_test)


# ==================== 打印实验配置 ====================
print("\n" + "=" * 60)
print("🚀 启动 CPD-STGCN 数据预处理流程")
print("=" * 60)
print(f"项目根目录       : {project_root}")
print(f"输入文件         : {file_path}")
print(f"输出目录         : {processed_dir}")
print(f"降雨数据         : {args.rainfall_features or '未启用'}")
print(f"雨敏节点指数     : {args.rain_susceptibility or '未启用'}")
print("-" * 60)
print(f"n_his            : {args.n_his}")
print(f"n_pred           : {args.n_pred}")
print(f"data_config      : {data_config}")
print("-" * 60)
print(f"cpd_mode         : {args.cpd_mode}")
print(f"cpd_method       : {args.cpd_method}")
print(f"cpd_cost         : {args.cpd_cost}")
print(f"cpd_penalty      : {args.cpd_penalty}")
print(f"pelt_penalty     : {args.pelt_penalty}")
print(f"bfast_frequency  : {args.bfast_frequency}")
print(f"cpd_top_ratio    : {args.cpd_top_ratio}")
print(f"cpd_min_size     : {args.cpd_min_size}")
print(f"adj_threshold    : {args.adj_threshold}")
print("=" * 60)


# ==================== 兼容式调用 data_gen ====================
# 说明：
# 如果你的 date_loader.py 已经改成新版 data_gen，则会传入 CPD 和邻接矩阵参数；
# 如果 date_loader.py 还是旧版 data_gen，则只传入旧参数，避免 TypeError。
base_kwargs = {
    "file_path": file_path,
    "data_config": data_config,
    "save_path": processed_dir,
    "n_his": args.n_his,
    "n_pred": args.n_pred
}

advanced_kwargs = {
    "cpd_penalty": args.cpd_penalty,
    "cpd_mode": args.cpd_mode,
    "cpd_method": args.cpd_method,
    "cpd_cost": args.cpd_cost,
    "pelt_penalty": args.pelt_penalty,
    "bfast_frequency": args.bfast_frequency,
    "cpd_top_ratio": args.cpd_top_ratio,
    "cpd_min_size": args.cpd_min_size,
    "adj_threshold": args.adj_threshold,
    "rainfall_features_path": args.rainfall_features,
    "rain_susceptibility_path": args.rain_susceptibility,
}

data_gen_signature = inspect.signature(data_gen)
supported_params = set(data_gen_signature.parameters.keys())

final_kwargs = dict(base_kwargs)
unsupported_params = []

for key, value in advanced_kwargs.items():
    if key in supported_params:
        final_kwargs[key] = value
    else:
        unsupported_params.append(key)

if unsupported_params:
    print("\n⚠️ 注意：当前 date_loader.py 的 data_gen 函数还不支持以下参数：")
    print(f"   {unsupported_params}")
    print("   本次会先按旧版 data_gen 运行，CPD模式和邻接矩阵阈值参数暂时不会生效。")
    print("   如果你想让 --cpd_mode、--adj_threshold 真正生效，下一步需要同步修改 date_loader.py。\n")


# ==================== 执行数据生成 ====================
dataset = data_gen(**final_kwargs)

npz_file = os.path.join(processed_dir, "preprocessed_data.npz")


# ==================== 输出结果检查 ====================
print("\n" + "=" * 60)
print("✅ 数据预处理与 CPD 阶段划分完成")
print("=" * 60)

try:
    print(f"Train shape      : {dataset.get_data('train').shape}")
    print(f"Val shape        : {dataset.get_data('val').shape}")
    print(f"Test shape       : {dataset.get_data('test').shape}")
except Exception as e:
    print(f"⚠️ 数据 shape 打印失败: {e}")

print("-" * 60)


# ==================== 阶段分布统计 ====================
def print_stage_distribution(name, stage_data):
    stage_data = np.asarray(stage_data)
    unique, counts = np.unique(stage_data, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"{name} Stage 分布 : {dist}")


try:
    train_stage = dataset.get_stage("train")
    val_stage = dataset.get_stage("val")
    test_stage = dataset.get_stage("test")

    print_stage_distribution("训练集", train_stage)
    print_stage_distribution("验证集", val_stage)
    print_stage_distribution("测试集", test_stage)

except Exception as e:
    print(f"⚠️ Stage 分布打印失败: {e}")

try:
    train_rain = dataset.get_rain("train")
    if train_rain is not None:
        print(f"训练集降雨特征 : {train_rain.shape}")
    rain_susceptibility = dataset.get_rain_susceptibility()
    if rain_susceptibility is not None:
        print(f"雨敏节点指数     : {rain_susceptibility.shape}, range=({rain_susceptibility.min():.3f}, {rain_susceptibility.max():.3f})")
except Exception as e:
    print(f"⚠️ 降雨特征打印失败: {e}")


# ==================== 保存文件检查 ====================
print("-" * 60)

if os.path.exists(npz_file):
    print(f"预处理文件       : {npz_file}")
    print(f"输出文件大小     : {os.path.getsize(npz_file) / (1024 * 1024):.2f} MB")
else:
    print(f"⚠️ 没有找到预处理文件: {npz_file}")

print("=" * 60)
print(">> 下一步：运行训练脚本")
print("   python .\\scripts\\train.py train-fixed")
print("=" * 60)
