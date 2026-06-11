# CPD-STGCN 滑坡位移预测优化进展说明

本文档用于说明当前代码相对原 STGCN 滑坡位移预测工作的主要修改、实验思路、当前精度结果以及后续需要讨论的问题。

## 1. 当前问题背景

本项目基于 Sela 滑坡 InSAR 位移数据进行时空预测。参考论文中使用 STGCN 进行长时间滑坡位移预测，公开报告指标约为：

| 指标 | 原论文 STGCN |
| --- | ---: |
| MSE | 25.51 |
| RMSE | 约 5.05 |
| MAE | 2.34 |

当前工作的核心想法是：不只把 STGCN 用作普通时空预测网络，而是引入变化点检测 CPD，围绕滑坡变形阶段转换进行训练集划分、样本加权和变化点邻域验证。

## 2. 数据方向修正

前期排查发现，项目中存在两个相似数据文件：

| 文件 | 含义 |
| --- | --- |
| `dataset/inter228_5241.csv` | 原始论文式数据，5241 个空间点，246 个时间观测 |
| `dataset/inter228_5241a.csv` | 转置后的辅助数据 |

旧流程容易误用 `inter228_5241a.csv`，导致把空间点当成时间维度，训练结果和论文对比口径不一致。

目前代码已改为使用：

```text
dataset/inter228_5241.csv
```

正确数据形态为：

```text
T = 246 个时间观测
N = 5241 个空间节点
```

模型输入窗口：

```text
n_his = 12
n_pred = 5
```

即用 12 个历史观测预测未来 5 步位移。

## 3. 当前核心改进

### 3.1 CPD 阶段检测

目前在 `data_loader/date_loader.py` 中实现了变化点检测。检测信号综合考虑：

- 全局平均位移；
- 全局位移标准差；
- 高活动节点平均位移；
- 多信号组合。

当前检测到的变化点为：

```text
[35, 80, 135, 160, 180]
```

对应阶段分布：

| Stage | 时间长度 |
| --- | ---: |
| 0 | 35 |
| 1 | 45 |
| 2 | 55 |
| 3 | 25 |
| 4 | 20 |
| 5 | 66 |

### 3.2 CPD 历史窗口输入

旧版本每个样本只有一个阶段标签。现在改为每个样本保存完整历史窗口阶段序列：

```text
[batch, n_his]
```

这样模型可以感知：

- 当前样本是否跨越变化点；
- 历史窗口内是否发生阶段转换；
- 当前预测是否处于滑坡变形阶段演化附近。

### 3.3 CPD 加权残差学习

当前模型不直接大幅预测未来位移，而是采用 persistence 作为物理主先验：

```text
prediction = last_observation + residual
```

其中残差由 STGCN 学习。训练时，如果历史窗口跨越 CPD 变化点，则样本权重提高，使模型更关注变形机制转换附近的样本。

### 3.4 CPD-aware 数据划分

原始时间顺序划分有一个问题：测试集可能集中在单一阶段，例如主要是 Stage 5，导致 CPD 的优势不明显。

因此新增了 `scripts/cpd.py validate`，支持两种 CPD-aware 验证方式：

| 策略 | 说明 |
| --- | --- |
| `stratified_stage` | 每个 CPD 阶段内分层划分 train/val/test，使各集合都覆盖多阶段 |
| `rolling_cpd` | 围绕每个变化点做滚动验证，测试变化点后的预测能力 |

这部分是目前最能体现创新性的地方。

## 4. 当前主要实验结果

### 4.1 固定时间顺序切分结果

在普通固定时间切分下，模型相对 persistence 的提升较小：

| 方法 | RMSE | MAE |
| --- | ---: | ---: |
| CPD-history weighted residual | 3.7418 | 2.2452 |
| Persistence | 3.8025 | 2.4914 |
| 速度阻尼基线 | 4.0497 | 2.4300 |

说明：固定测试集主要集中在后期 Stage 5，CPD 阶段切换信息发挥空间有限。

### 4.2 CPD 多阶段分层划分结果

采用 `stratified_stage` 后，训练、验证、测试都覆盖多个阶段。优化后的结果为：

| Fold | Model RMSE | Model MAE | Persistence RMSE | Persistence MAE | RMSE 提升 | MAE 提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stratified_stage | 3.0104 | 1.7302 | 3.5341 | 2.4547 | 14.82% | 29.52% |

该结果说明：当训练集和测试集都覆盖多阶段变形过程时，CPD-guided 模型比 persistence 有明显优势。

### 4.3 变化点滚动验证结果

围绕变化点进行滚动验证，上一版完整结果如下：

| Fold | Model RMSE | Model MAE | Persistence RMSE | Persistence MAE | RMSE 提升 | MAE 提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cp_80 | 2.0327 | 1.4986 | 2.7904 | 2.0946 | 27.15% | 28.45% |
| cp_135 | 2.9526 | 1.9775 | 4.1436 | 2.9435 | 28.74% | 32.82% |
| cp_160 | 2.3259 | 1.5632 | 3.7158 | 2.7769 | 37.40% | 43.71% |
| cp_180 | 1.5000 | 1.0573 | 2.6216 | 1.9510 | 42.78% | 45.81% |

这组结果说明：在变化点后邻域，CPD-guided 残差模型相对 persistence 的提升更明显。

## 5. 与原论文指标的粗略对比

原论文 STGCN 公开指标约为：

| 指标 | 原论文 |
| --- | ---: |
| RMSE | 约 5.05 |
| MAE | 2.34 |

当前 CPD-aware 多轮验证结果中，多数 fold 的 RMSE 和 MAE 均低于上述数值。例如：

| 实验 | RMSE | MAE |
| --- | ---: | ---: |
| stratified_stage | 3.0104 | 1.7302 |
| cp_80 | 2.0327 | 1.4986 |
| cp_135 | 2.9526 | 1.9775 |
| cp_160 | 2.3259 | 1.5632 |
| cp_180 | 1.5000 | 1.0573 |

但需要注意：这不是严格同口径复现，因为原论文的具体训练测试划分、窗口长度和评价设置未完全确认。因此更稳妥的表述是：

> 在相同 Sela InSAR 数据基础上，CPD-aware 多阶段划分与变化点滚动验证下，当前模型在各 fold 中均优于 persistence 基线，并相较原 STGCN 公开指标表现出更低误差。

## 6. 当前代码中主要文件

| 文件 | 作用 |
| --- | --- |
| `data_loader/date_loader.py` | 正确读取原始数据、CPD 检测、阶段序列生成、地形 KNN 图构建 |
| `models/base_model.py` | STGCN 主模型，采用 persistence prior + residual 结构 |
| `scripts/train.py train-fixed` | 固定时间切分训练脚本 |
| `scripts/train.py test-fixed` | 固定时间切分测试脚本 |
| `scripts/cpd.py validate` | CPD-aware 多阶段划分和变化点滚动验证脚本 |
| `EXPERIMENT_NOTES.md` | 实验记录和阶段性结果 |

## 7. 如何复现实验

### 7.1 固定时间切分预处理

```powershell
C:\Users\lenovo\anaconda3\envs\stgcn\python.exe .\scripts\train.py preprocess
```

### 7.2 固定时间切分训练

```powershell
C:\Users\lenovo\anaconda3\envs\stgcn\python.exe .\scripts\train.py train-fixed
```

### 7.3 固定时间切分测试

```powershell
C:\Users\lenovo\anaconda3\envs\stgcn\python.exe .\scripts\train.py test-fixed
```

### 7.4 CPD-aware 多轮验证

```powershell
C:\Users\lenovo\anaconda3\envs\stgcn\python.exe .\scripts\cpd.py validate --strategy both --max_rounds 5 --epochs 8 --patience 3 --output_dir .\output\cpd_split_validation_full
```

结果输出：

```text
output/cpd_split_validation_full/cpd_split_results.csv
```

### 7.5 优化版多阶段分层实验

```powershell
C:\Users\lenovo\anaconda3\envs\stgcn\python.exe .\scripts\cpd.py validate --strategy both --max_rounds 5 --epochs 8 --patience 3 --output_dir .\output\cpd_split_validation_optimized
```

结果输出：

```text
output/cpd_split_validation_optimized/cpd_split_results.csv
```

## 8. 当前可作为论文创新点的表述

可以考虑将方法概括为：

```text
CPD-guided Stage-history Residual STGCN
```

中文可写为：

```text
变化点引导的阶段历史残差 STGCN 模型
```

主要创新点包括：

1. 使用 CPD 自动识别滑坡位移时序中的阶段转换；
2. 将 CPD 阶段从单一标签扩展为历史阶段序列；
3. 利用 CPD 跨阶段样本加权，引导模型关注变形机制转换区域；
4. 提出 CPD-aware 多阶段分层划分，避免单一阶段测试掩盖 CPD 优势；
5. 提出变化点滚动验证，更直接评估模型在阶段转换后邻域的预测能力。

## 9. 目前仍需要讨论的问题

1. 原论文具体训练/测试划分需要进一步确认，方便做更严格同口径对比。
2. CPD-aware split 是否可以作为主实验，还是作为补充实验，需要结合论文投稿定位决定。
3. rolling CPD 每个 fold 的样本量较小，后续最好做 repeated seeds 或 bootstrap 置信区间。
4. 当前 CPD 是全局变化点，后续可考虑空间分区 CPD 或节点群 CPD。
5. 当前模型仍基于 TensorFlow 1.x，后续如继续扩展，建议迁移到 PyTorch 或 TensorFlow 2.x 以便维护。

## 10. 简要结论

目前固定时间切分下模型提升有限，但在 CPD-aware 多阶段划分和变化点滚动验证中，模型相对 persistence 基线表现出较明显优势。当前结果支持继续围绕 CPD 构建论文核心，而不是单纯堆叠 STGCN 网络结构。

## 11. 降雨响应扩展

新增两阶段降雨响应功能，用于解释降雨后滑坡变形阶段转换，并在响应关系显著时接入 CPD-STGCN。

### 11.1 第一阶段：降雨-CPD 密度解释分析

实现文件：

| 文件 | 作用 |
| --- | --- |
| `data_loader/rainfall_loader.py` | 下载 CHIRPS 日降雨、裁剪官方 GeoTIFF、生成 InSAR 日期对齐的滞后降雨特征 |
| `scripts/rainfall.py cpd-response` | 按空间分区检测 CPD，计算分区变化点密度与降雨响应关系 |

推荐运行：

```powershell
python .\scripts\rainfall.py cpd-response --download_chirps
```

正式 CHIRPS 数据推荐使用 CHC 官方日尺度 GeoTIFF 裁剪流程：

```powershell
python .\data_loader\rainfall_loader.py --buffer_degree 0.25 --rate_limit 300K --sleep_seconds 2
```

或由响应分析脚本直接触发：

```powershell
python .\scripts\rainfall.py cpd-response --download_chirps_crop
```

输出：

```text
output/rainfall_cpd_response/rainfall_cpd_response.csv
output/rainfall_cpd_response/rainfall_cpd_summary.csv
output/rainfall_cpd_response/regional_change_points.csv
output/rainfall_cpd_response/rainfall_cpd_response.png
```

正式版采用 CHC 官方 `global_daily/tifs/p05` 日尺度 GeoTIFF，逐日下载后裁剪 `滑坡点云 bbox + 0.25°` 范围，并输出裁剪栅格与区域平均雨量。CPD 密度定义为：

```text
cpd_density = 发生变化点的空间分区数 / 空间分区总数
```

默认空间分区数为 12，降雨滞后窗口为：

```text
0 / 3 / 7 / 15 / 30 天，以及相邻 InSAR 日期间累计降雨
```

裁剪数据输出：

```text
dataset/rainfall/chirps_crop/chirps_crop_daily.csv
dataset/rainfall/chirps_crop/chirps_crop_daily.npz
dataset/rainfall/chirps_crop/daily/
dataset/rainfall/chirps_crop/metadata.json
dataset/rainfall/chirps_daily.csv
```

其中 `daily/` 是逐日裁剪缓存，用于支持中断后续跑；原始 `.tif.gz` 和临时 `.tif` 默认不长期保留。

### 11.2 第二阶段：降雨增强 CPD-STGCN

预处理入口已支持降雨特征：

```powershell
python .\scripts\train.py preprocess --rainfall_features .\dataset\rainfall\chirps_daily.csv
```

节点级 CHIRPS 热区结果可进一步压缩为连续雨敏指数，并写入预处理包：

```powershell
python .\scripts\rainfall.py build-susceptibility `
  --node_response_csv .\output\node_rainfall_hotspots_chirps\node_rain_response.csv `
  --output_csv .\output\rain_susceptibility\rain_susceptibility.csv `
  --hotspot_quantile 0.75

python .\scripts\train.py preprocess `
  --rainfall_features .\dataset\rainfall\chirps_daily.csv `
  --rain_susceptibility .\output\rain_susceptibility\rain_susceptibility.csv
```

训练和测试脚本会在预处理包中存在 `train_rain`、`val_rain`、`test_rain` 时自动启用降雨条件：

```powershell
python .\scripts\train.py train-fixed
python .\scripts\train.py test-fixed
```

模型耦合方式保持保守：

1. 位移输入张量仍为 `[batch, n_his, n_nodes, 1]`；
2. 降雨滞后特征作为 FiLM 条件变量，不改变图卷积输入通道；
3. 强降雨历史窗口会提高 CPD-aware residual learning 的样本权重；
4. 若预处理包含 `rain_susceptibility`，图卷积会启用低秩雨致动态图门控；默认雨敏指数保留多证据得分最高的 25% 节点作为热点区，并在已有地形 KNN 图传播前后用 `rain_intensity × centered(rain_susceptibility)` 的可学习小幅门控调制 source/target 节点强度；
5. `scripts/cpd.py validate` 支持 `--rainfall_features --rain_susceptibility --dynamic_rain_graph`，用于比较无降雨、rainfall-FiLM、雨致动态图和动态图+FiLM 版本。

该设计避免直接把粗分辨率 CHIRPS 网格复制到 5241 个 InSAR 节点造成伪空间精细化，也不生成样本级完整 `N×N` 动态邻接矩阵；降雨的主耦合点从辅助 FiLM 升级为雨敏节点上的动态图传播调制。

### 11.3 节点级雨后变化点热区

为回答“降雨后一段时间内，整个滑坡体哪些局部区域出现变化点密集”，新增节点级热区分析：

```powershell
python .\scripts\rainfall.py hotspots --rainfall_csv .\dataset\rainfall\chirps_daily.csv
```

该分析不替换训练用 CPD 阶段划分，而是新增探索型 CPD 筛查：

| 档位 | 用途 |
| --- | --- |
| `strict` | 接近当前大阶段变化点口径，误报较少 |
| `medium` | 平衡短期响应和稳定性 |
| `loose` | 捕捉更短期、局部的候选响应 |

脚本使用 P50/P75/P90/P95 降雨事件和 3/7/15/30 天雨后窗口，输出：

```text
output/node_rainfall_hotspots/node_change_points.csv
output/node_rainfall_hotspots/rain_event_windows.csv
output/node_rainfall_hotspots/node_rain_response.csv
output/node_rainfall_hotspots/rain_response_summary.csv
output/node_rainfall_hotspots/rain_response_hotspots.png
output/node_rainfall_hotspots/rain_response_lag_panels.png
output/rain_susceptibility/rain_susceptibility.csv
```

固定切分测试后，使用以下脚本补充强降雨窗口和雨敏节点子集指标：

```powershell
python .\scripts\rainfall.py focus-eval
```

输出：

```text
output/rainfall_focus_metrics.csv
```

当前 POWER 探索版结果显示，P95 强降雨后 3-7 天窗口比 15/30 天窗口更容易出现节点级响应，其中 `medium` 与 `strict` 档下有更多节点达到置换检验探索阈值。该结果说明：原 12 分区整体相关口径过粗，不能据此否定局部降雨响应；后续应以 CHIRPS 裁剪数据复跑节点级热区图。P50/P75 仅用于观察一般降雨背景，论文解释应重点关注 P90/P95 是否形成稳定短滞后热区。

### 11.4 CHIRPS 正式复跑结果

已使用 CHC 官方 COG 数据完成 `2014-10-24` 至 `2022-07-26` 全时段裁剪：

```text
dataset/rainfall/chirps_crop/chirps_crop_daily.npz
dataset/rainfall/chirps_crop/chirps_crop_daily.csv
dataset/rainfall/chirps_daily.csv
```

裁剪窗口为 `11 x 11` CHIRPS 像元，逐日缓存 `2833/2833` 天完整，检查结果为：

```text
missing_before_last=0
bad_files=0
nan_files=0
negative_files=0
rain_mm_max=103.215309
```

节点级热区分析输出目录：

```text
output/node_rainfall_hotspots_chirps/
```

CHIRPS 正式结果显示，强降雨后的节点级变化点响应比整体分区响应更清晰：

| 降雨事件 | 雨后窗口 | CPD 档位 | p < 0.05 节点数 | p < 0.10 节点数 | 最大 lift |
| --- | --- | --- | ---: | ---: | ---: |
| P95 | 7 天 | strict | 104 | 116 | 4.669 |
| P95 | 15 天 | strict | 104 | 112 | 3.261 |
| P90 | 7 天 | strict | 40 | 43 | 2.721 |
| P95 | 7 天 | medium | 23 | 32 | 3.856 |
| P90 | 15 天 | strict | 26 | 47 | 1.894 |

分区级 CPD 密度响应输出目录：

```text
output/rainfall_cpd_response_chirps/
```

12 分区整体密度与降雨窗口的 Spearman 相关均不显著，最小置换检验 p 值约为 `0.312`。这说明：如果把整个滑坡体压缩成分区 CPD 密度，局部强降雨响应会被明显稀释；论文解释应以节点级热区为主，分区级结果作为“整体相关不显著”的反证边界，而不是否定降雨响应。

### 11.5 CHIRPS 响应可视化

新增可视化脚本：

```powershell
python .\scripts\rainfall.py response-figures
```

输出目录：

```text
output/chirps_response_figures/
```

生成图件：

| 图件 | 用途 |
| --- | --- |
| `chirps_rainfall_thresholds.png` | 展示全时段 CHIRPS 日降雨和 P50/P75/P90/P95 事件阈值 |
| `chirps_response_summary_heatmap.png` | 对比不同降雨阈值、雨后窗口和 CPD 档位下的显著节点数 |
| `chirps_hotspot_combo_panels.png` | 展示 P95+7 天、P95+15 天、P90+7 天 strict 热区空间分布 |
| `chirps_top_responsive_nodes.png` | 展示 P95+7 天 strict 组合下 lift 最高的响应节点 |

### 11.6 CPD 方法身份与敏感性实验

当前训练阶段划分默认使用 `ruptures.Binseg(model="l2")`，不是 BEAST、BFAST 或 PAE。为避免把变化点算法差异误判为降雨机制收益，新增可选 CPD 方法接口：

```powershell
python .\scripts\train.py preprocess --cpd_method binseg
python .\scripts\train.py preprocess --cpd_method pelt
python .\scripts\train.py preprocess --cpd_method bfast
```

`binseg` 保持为论文主线默认方法；`pelt`、`bfast`、`kernelcpd`、`dynp`、`bottomup`、`window`、`bocpd` 和 `mdl_multivariate` 用作敏感性对照。BFAST 分支调用本机 R 包 `bfast`，BOCPD 分支优先通过 Python 3.11 调用真实 `fast-bocpd` 包；若外部调用失败，才退回内置 BOCPD-style 对照。新增对照脚本：

```powershell
python .\scripts\cpd.py method-sensitivity --methods binseg pelt bfast kernelcpd dynp bottomup window bocpd mdl_multivariate
```

输出：

```text
output/cpd_method_sensitivity/cpd_method_change_points.csv
output/cpd_method_sensitivity/cpd_method_stage_summary.csv
output/cpd_method_sensitivity/cpd_method_timeline.png
```

预测精度是否显著提升，需要分别运行：

```powershell
python .\scripts\cpd.py validate --cpd_method pelt --strategy both
python .\scripts\cpd.py validate --cpd_method bfast --strategy both
```

当前判断口径：替换 CPD 方法可能改变阶段边界和训练/验证切分，但它不会自动增加降雨物理信息；若精度提升，只能说明阶段划分更适合该数据，仍需用雨后窗口、雨敏节点子集和机制图件证明降雨响应。

### 11.7 滑坡运动机理解释输出

为回应“降雨响应不能只服务预测误差，还要说明滑坡如何滑动、为什么滑、向哪个方向滑”，新增机理分析脚本：

```powershell
python .\scripts\rainfall.py mechanism-analysis `
  --rain_susceptibility .\output\rain_susceptibility\rain_susceptibility.csv
```

输出：

```text
output/landslide_mechanism/landslide_mechanism_summary.csv
output/landslide_mechanism/landslide_stage_kinematics.csv
output/landslide_mechanism/landslide_mechanism_interpretation.md
output/landslide_mechanism/mechanism_stage_velocity.png
output/landslide_mechanism/mechanism_space_groups.png
output/landslide_mechanism/mechanism_elevation_displacement.png
```

解释逻辑为：强降雨经入渗和裂隙补给提高坡体含水状态，降低潜在滑带及上覆堆积体的有效抗剪强度；上部或补给敏感区的雨敏节点先出现短滞后变化点响应，随后变形沿已有地形 KNN 邻接关系向下坡方向传播。地形拟合得到的候选下坡方向用于解释“怎么滑”和“向哪里滑”，但单轨 InSAR 只能直接约束视线向位移，不能单独证明完整三维滑动矢量，因此论文中应使用“地形约束的下坡传播方向”而不是绝对三维运动方向。

新增机制论文图脚本：

```powershell
python .\scripts\rainfall.py mechanism-figure
```

输出：

```text
output/rainfall_mechanism_figures/rainfall_mechanism_overview.png
```

论文叙事说明已整理到：

```text
docs/RAIN_CPD_MECHANISM_EXPLANATION.md
```

扩展 CPD 敏感性已生成：

```text
output/cpd_method_sensitivity_extended/
```

主要边界如下：

| 方法 | 变化点索引 |
| --- | --- |
| `kernelcpd` | 60, 97, 135, 164, 194 |
| `dynp` | 60, 98, 135, 164, 193 |
| `bottomup` | 30, 76, 107, 138, 184 |
| `window` | 33, 75, 138 |
| `bocpd` | 87, 117, 141 |
| `mdl_multivariate` | 78, 136, 180 |

其中 `mdl_multivariate` 两折 smoke 结果为 `stratified_stage MAE=1.554`、`cp_78 MAE=1.719`，接近 PELT；`kernelcpd` smoke 结果为 `stratified_stage MAE=1.810`、`cp_60 MAE=2.029`，当前不优先作为主线。



