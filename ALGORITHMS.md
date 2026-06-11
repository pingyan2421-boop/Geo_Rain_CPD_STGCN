# 核心算法全解 (Event-Distilled Algorithms)

> 💡 **本文档详细拆解了“降雨事件蒸馏模型”工程中所运用的特有算法，重点聚焦于如何通过数学与机器学习手段将气象风险融合进深度图神经网络。**

本目录下的模型不再局限于传统的时序预测，而是将**“降雨事件特征提取”**、**“可解释性机器学习”**与**“深度知识蒸馏”**进行了三位一体的结合。

---

## 1. 降雨事件特征工程算法 (Rainfall Event Feature Engineering)

**算法原理：**
传统的日降雨量（Daily Rainfall）含有大量随机噪声，无法直接反映地质体的真实滑动风险。该算法旨在通过物理与统计规律将原始降雨转化为“风险特征”。
- **活跃事件衰减 (Event Active Constraint):** 算法不再只看当天的降水，而是通过设定 3天/7天/15天的窗口，提取诸如 `event_active_3d`。如果距离上一次极端降雨事件越近，该衰减系数越高。
- **前期影响雨量 (Antecedent Wetness Index, AWI / Wetness):** 这是一个水文学算法。考虑到土壤中水分的累积和蒸发特性，通过衰减因子（例如 `alpha = 0.8`）计算过去几十天降水的指数衰减累加值，以表征山体当前的“饱和度/湿润度”。
- **历史触发概率 (Historical Trigger Probability / Probs):** 利用贝叶斯先验统计，计算出在历史上相同的“湿度等级”和“降雨等级”下，真实诱发显著位移事件的概率。

---

## 2. 气象风险驱动的动态门控算法 (Dynamic Gating via Logistic Regression)

**算法原理：**
在融合 Official（常规）与 Fallback（后备）两套基础专家网络时，如果不加以控制，深度学习经常会因为“软投票”导致特征坍塌。为此引入了外部的可解释机器学习监督器。
- **目标设定 (Target):** 算法在验证集（Validation Set）上扫描，将预测误差更小的那个模型记为优胜者（1），误差大的记为（0）。
- **模型选择:** 摒弃了黑盒的全连接层，选用了 Scikit-Learn 的**逻辑回归模型 (Logistic Regression)**。
- **特征绑定:** 强制将特征矩阵限定为刚才提取的外部气象风险特征 `X = [probs, wetness]`。
- **产出:** 通过 `model.fit(X, y)`，算法学习到了气象风险与哪个专家网络更匹配的回归系数组合。随后在预测时输出连续的门控权重 $w \in (0,1)$，生成最终的混合教师 (Composite Teacher)。

---

## 3. 基于事件风险的动态加权蒸馏算法 (Risk-Aware Distillation Loss)

**算法原理：**
传统的知识蒸馏 (Knowledge Distillation, KD) 在整个训练集上赋予统一的 Teacher 权重系数，导致模型把注意力浪费在不重要的平稳期噪声上。
- **动态阈值惩罚 (Adaptive Sample Weighting):**
  算法设计了一套分段的惩罚函数：
  - **低风险平稳期 (Risk w < 0.5 & Non-monsoon):** 样本学习权重 `distill_weight` 被手动调低至 $0.5$。
  - **高风险季风期 (Risk w >= 0.5):** 样本学习权重被猛烈提升至 $2.0$。
- **损失函数融合:** 在长步长（如 H+3, H+4, H+5）的反向传播中，最终的损失函数为：
  $Loss = Loss_{GroundTruth} + \lambda_{dynamic} \times Loss_{Teacher}$
  使得模型在雨季时被“强迫”严格服从经过验证的专家混合输出，彻底逆转了长步长退化问题。

---

## 4. 底层支撑骨架 (Base Architecture)

除了上述针对降雨事件独创的算法外，底层依旧依托了坚实的 CPD-STGCN 架构支撑：
- **二元分割突变点检测 (Binseg CPD):** 利用 `ruptures` 对历史形变序列执行时间切片提取稳态序列。
- **时空图卷积 (STGCN):** 基于 InSAR 空间邻接矩阵提取空间形变协同效应，利用 Gated 1D-CNN 提取时序记忆。
- **物理先验预测残差 (Physics Prior Residual):** 强制剥离绝对值，网络结构只专注于预测未来几天由气象引起的微小位移残差 $\Delta y$。
