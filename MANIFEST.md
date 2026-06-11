# 文件材料清单 (Manifest)

> 📁 **降雨事件动态蒸馏模型 (Event-Distilled Model) 目录结构全解**

本目录（`release_event_distilled_model`）是一个**完全自包含（Self-contained）**的深度学习端到端训练包。所有的核心代码、原始数据、一键运行脚本以及先验输出均已备齐，确保拷贝到任何电脑均可直接开跑。

## 📜 核心运行脚本

- **`run_full_distillation_pipeline.ps1`**: 🚀 **最核心的一键启动器**。它编排了多 Seed 下的基线训练、教师预测、验证集动态门控学习以及带动态事件权重的蒸馏训练全过程。支持 `-Python` 参数跨电脑指定运行环境。
- **`requirements.txt`**: 本运行环境所需的 Python 依赖项（包含了必须的 `scikit-learn` 与 `tensorflow`）。
- **`aggregate_predictions.py` & `metric.py`**: 全流程跑到终点后，自动汇总各个切片与各个 Seed 误差的核心算分组件。

## 🧠 模型与逻辑代码

- **`scripts/`**: 存放所有的运行时逻辑。
  - `cpd_split_validate_impl.py`: 模型训练执行引擎（包含提取降雨特征并动态放大样本 `distill_weight` 损失权重的核心代码）。
  - `generate_continuous_gate_teachers.py`: 利用气象变量 `[probs, wetness]` 拟合逻辑回归门控，计算软路由权重的核心代码。
- **`models/`**: 包含 STGCN 底层网络结构定义（`base_model.py`, `layers.py`）。
- **`data_loader/`**: 负责处理 InSAR 位移矩阵、突变点检测（CPD）以及 CHIRPS 降雨序列。
- **`utils/`**: 通用工具类库。

## 📊 数据与先验特征 (Data & Pre-computed Features)

为了保证开箱即用，免去极度耗时的历史雨量全局扫描，本包预置了必要的先验特征：

- **`dataset/`**:
  - `inter228_5241.csv`: 原始 InSAR 地表位移坐标与时序数据。
  - `rainfall/chirps_daily.csv`: 对齐研究区的日降雨数据。
  - `forecast/chirps_gefs_15day_cp180_climfallback.csv`: 带有气候态均值填补的 GEFS 15日气象预报。
- **`output/`**:
  - `rain_susceptibility/rain_susceptibility.csv`: 基于历史形变关系预计算的静态“雨敏节点图网络权重”。
  - `rainfall_event_catalog/rainfall_event_catalog.csv`: 全局范围扫描提取的“降雨事件名录”（包含事件概率 `probs` 与湿度 `wetness` 等触发机制前提）。

## 📝 文档资产 (Documentation)

- **`README.md`**: 项目整体说明与极简运行指南。
- **`ENVIRONMENT.md`**: 详细的依赖项说明与可复现性提示。
- **`METHODOLOGY.md`**: 详细解释为何要引入降雨事件机制，以及解决长步长衰减的逻辑流程。
- **`ALGORITHMS.md`**: 从数学与代码层面拆解降雨事件特征提取、逻辑回归门控与动态蒸馏损失的实现。
- **`OPTIMIZATION_RESULTS_REPORT.md` & `CPD_STGCN_PROGRESS_REPORT.md`**: 保留了您进行事件机制优化的原始思考脉络、实验纪要与历史详细成绩单。

---
*注：本结构经过精心剪裁与补全，剔除了不相关的干扰试验，为您后续专心深耕“降雨事件网络优化”提供了一个干净、强健且立等可运行的专属沙盒。*
