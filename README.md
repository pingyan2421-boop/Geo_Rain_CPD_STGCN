# 降雨事件动态蒸馏模型 (Event-Distilled Model)

> 🌧️ **利用降雨事件发生概率与气象风险的端到端混合预测网络**

本目录是专门为您之前设计的**“降雨事件”历史最优模型**单独切出的一份**纯净独立版本 (Clean Environment)**。在这里，所有与事件机制相关的特征工程、Loss 动态加权、以及基于验证集预测误差动态学习门控的方法都被完整保留。

## 🌟 核心特色 (Key Features)

与 `release_source_switch_posthoc_moe`（纯后处理硬切换）不同，本仓库属于**端到端深度学习优化**：

1. **降雨事件特征提取**: 动态追踪 `event_active_3d`, `event_active_7d`, 以及 `antecedent_wetness_index` (前期影响雨量)。
2. **气象风险动态门控**: 根据气象变量 `[probs, wetness]`，系统在验证集上主动学习并生成能够判定高低风险的门控（Gate）权重。
3. **样本级动态蒸馏惩罚**: 
   - 处于降雨高风险（季风雨季且门控 $w \ge 0.5$）的样本，蒸馏损失比重翻倍至 $2.0$。
   - 处于气候平稳期（非降雨季），降低拟合惩罚至 $0.5$。

## 📊 历史最高性能 (cp_180 阶段测试)

相比于未加优化的纯端到端网络，本方案实现了极强的抗衰减与雨季泛化能力：

| 评估指标 | MAE 成绩 | 提升描述 |
| :--- | :--- | :--- |
| **全局预测 (Global MAE)** | 1.0740 | 相比基线下降 7.5% |
| **雨敏点预测 (Rain Sensitive)** | 0.9771 | 相比基线暴降 12.9% |
| **H+3 中长步长 (Rain Sensitive)** | 0.9623 | 相比基线下降近 10% |
| **H+5 极长步长 (Rain Sensitive)** | 1.5707 | 相比基线缩减高达 15.2% |

## 🚀 启动指引 (Out-of-the-Box Execution)

本目录已经过**深度整合与环境补全**，它不再依赖上级或原始工作区，完全做到了**开箱即用**：
- 补全了 `aggregate_predictions.py` 等评估脚本。
- 预置了 `rain_susceptibility` 与 `rainfall_event_catalog` 等耗时的先验气象统计特征（位于 `output/` 目录）。
- 重构了启动脚本以支持跨电脑参数化运行。

1. **安装基础依赖库**（推荐具有 TensorFlow 的环境）：
   ```powershell
   pip install -r requirements.txt
   ```
2. **运行一键端到端实验**（自动执行教师预判、门控学习、动态加权蒸馏与全局误差统计）：
   ```powershell
   .\run_full_distillation_pipeline.ps1
   
   # 如果需要在别人电脑或特定环境中运行，可直接指定 Python 路径：
   # .\run_full_distillation_pipeline.ps1 -Python "D:\Your\Path\To\python.exe"
   ```

*注意：本目录保留了完整的 `dataset` 与 `model` 架构。您可以独立地在这里对“降雨事件”机制进行任何改良或再次训练，而不必受纯净对比基线库的限制。详细的实验复盘和原理论述见 `OPTIMIZATION_RESULTS_REPORT.md`，目录结构说明详见 `MANIFEST.md`。*
