# 环境配置指南 (Environment Setup Guide)

> 🛠️ **降雨事件动态蒸馏模型 训练与运行环境说明**

本仓库（`release_event_distilled_model`）是一个完整的**端到端深度学习训练工程**。由于包含了神经网络的前向推理、反向传播以及基于 Scikit-Learn 的动态逻辑回归门控学习，本环境对依赖库有较强要求。

## 📦 核心依赖项

为了确保特征提取、图构建以及知识蒸馏的顺利执行，请确保您的 Python 环境具备以下核心依赖：

- **Python** 3.x (需兼容遗留 TensorFlow 代码架构)
- `numpy` & `pandas` (用于处理时间序列位移、降雨特征与验证集度量)
- `tensorflow` (通常为 1.x 版本或 2.x 的 `compat.v1` 运行模式，承载底层 STGCN 运算)
- `scikit-learn` (用于动态学习 `probs` 与 `wetness` 等气象特征，生成连续门控函数)
- `ruptures` (用于前期突变点检测/CPD)

## 🚀 安装指引

1. **进入专属运行目录:**
   ```powershell
   cd D:\文档\CPD-STGCN\release_event_distilled_model
   ```
2. **执行依赖安装:**
   ```powershell
   pip install -r requirements.txt
   ```

## 🧠 一键端到端实验运行 (Out-of-the-Box Pipeline)

不同于需要繁琐前置步骤的早期工作流，本目录已经整合为**完全自包含（Self-contained）**的版本，并预置了必要的先验特征（位于 `output/` 中）。

我们提供了一个高度自动化的 PowerShell 管道脚本，支持跨电脑、跨环境一键执行完整的闭环：

```powershell
.\run_full_distillation_pipeline.ps1

# 该脚本已经过优化，如果您在其他电脑上存在多个虚拟环境，可显式指定：
# .\run_full_distillation_pipeline.ps1 -Python "C:\Path\To\python.exe"
```

### 管道脚本背后执行的四大步骤：
脚本启动后，后台将依次经历以下计算阶段（总体耗时较长，请耐心等待）：
1. **教师预判 (Teacher Generation):** 在 `cp_180` 等时序切分点下，分别评估 `Official` 基线和 `Fallback` 基线的误差分布。
2. **门控学习 (Gate Learning):** 在 Validation 集上，通过逻辑回归动态算出基于事件风险的软路由权重。
3. **标签混合 (Distillation Target Synthesis):** 将两个独立专家的预测结果，按动态门控比率融合，并生成相应的样本蒸馏权重。
4. **学生重训 (Student Retrain):** 加载事件惩罚权重（在长步长 H+3/H+5 阶段重点发力），重新训练最终的 Distilled 模型网络。

## ⚠️ 训练与可复现性提示

由于该过程涉及**真实且深度的神经网络迭代训练**，以下因素可能导致各次运行产生的指标结果与报告中（如 `0.977`）存在极其细微的波动：
- GPU / CPU 的底层张量计算指令差异。
- TensorFlow 的伪随机数种子容差。
- 操作系统对并发进程的调度机制差异。

这属于动态蒸馏网络训练中的正常现象，本项目的管道脚本已经通过强制固定 `SEEDS = @(0, 7, 23)` 最大程度上平抑了这种随机扰动。
