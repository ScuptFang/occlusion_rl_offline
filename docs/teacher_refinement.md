# Teacher 统一监督与最近表面距离实验

本实验实现两项改动：ranking/calibration 共享姿态生成和质量标签，calibration 保留排序与分数差监督；将最近下颌顶点距离替换为最近三角网格表面距离。仍使用原上颌顶点、顶点法向与拓扑曲率，4096 个均匀候选顶点 → 128 点接触优先加权 FPS，3 层、宽度 64 的编码器。不生成体积 SDF。

## 训练目标

新入口 `teacher.trainer=aligned_pair`，实现位于 `occlusion_rl_offline/aligned_teacher.py`。两个阶段统一使用 SO(3) 旋转角和绕上颌全网格中心的平移：

`q(R,t) = -angle(R)/8 - norm(t)/4`

姿态用旋转向量生成；局部比较使用 `R₂ = ΔR @ R₁`、`t₂ = t₁ + Δt`，与 agent 的增量执行方式一致。排序标签完全由实际 q 差产生，不依赖扰动档位名字。q 是参考姿态恢复的代理目标，不是临床咬合标签。

总损失由以下三项组成：

- 分数回归：`Huber(s,q)`，beta=0.5。
- 排序：`relu(0.5*abs(Δq) - sign(Δq)*Δs)`，不对近似同质量状态强制固定 0.5 分差，abs(Δq)≤1e-5 的配对不进入排序项。
- 分数差回归：`Huber(Δs,Δq)`，beta=0.05。

ranking 的三项权重为 0.1 / 1 / 1；calibration 为 1 / 0.5 / 1。两个阶段均为 32 轮、batch=16 对、学习率 1e-4。每阶段按验证集 `MAE + delta_MAE + 0.5*(1-local_accuracy)` 选模；calibration 从 ranking 的最佳权重继续训练。测试指标不参与选模。

每个训练病例固定缓存 128 对姿态，共 896 对，每轮打乱顺序；两个阶段使用同一姿态库。验证每例 128 对，测试每例 256 对。缓存记录配置、病例文件 SHA-256 与种子，不兼容缓存会报错。固定缓存重复训练不等于每轮新增独立姿态。

姿态档位概率为零姿态 10%、近区 30%、初始范围 30%、更宽可达范围 30%；每轴旋转/平移范围分别为 0/0、1.5°/0.5mm、8°/4mm、20°/8mm。半数为局部动作对，半数为同档独立方向对。局部步长轮换 0.05°/0.01mm、0.25°/0.05mm、0.5°/0.1mm、2°/0.4mm，并包含单轴与多轴动作。

## 几何特征与兼容性

`model.distance_method` 可选 `nearest_vertex`（旧默认）与 `nearest_surface`。新方法使用 libigl 缓存 AABB 三角形索引查询精确最近位置，再用重心坐标插值**顶点法向**并归一化，以点积符号乘欧氏距离。没有使用面片法向替代网络法向，也没有改变上颌采样来源。

预处理 NPZ 增加 `lower_faces`，原始上下颌顶点与法向继续保留。旧 NPZ 可继续用于最近顶点模式；最近表面模式遇到没有拓扑的旧文件会要求重新 prepare。退化三角形仅从查询索引中剔除，不修改原顶点；面片绕序不一致会报错。开放边界、非流形边界、插值法向抵消及近切向方向被标为低置信度，本次不将置信度输入网络或改变训练权重。符号是局部表面两侧，不能当作封闭物体的内外真值。

距离选项同时传入 teacher 数据构造、FPS 权重、PoseTask 批量观测及推理。旧模型选项默认保持不变，已有 checkpoint 的配置一致性检查会拒绝误用新几何输入。新 teacher 不能直接替换进旧 agent 当作经过验证的组合，后续 agent 需要匹配配置重训。两个新增配置的 agent 输出路径独立，不引用旧 agent 权重。

## 复现命令

在仓库根目录执行，默认 CPU；可通过 CLI 的 `--device cuda` 或 `OCCLUSION_DEVICE=cuda` 使用已配置的 GPU 环境。几何查询仍在 CPU。

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m occlusion_rl_offline.cli --config configs/teacher_aligned_surface.json prepare-batch --input-dir ../data --out runs/teacher_refinement/prepared
.venv/bin/python -m occlusion_rl_offline.cli --config configs/teacher_aligned_vertex.json train-teacher --manifest runs/teacher_refinement/prepared/manifest.json
.venv/bin/python -m occlusion_rl_offline.cli --config configs/teacher_aligned_surface.json train-teacher --manifest runs/teacher_refinement/prepared/manifest.json
.venv/bin/python scripts/audit_surface_distance.py
.venv/bin/python scripts/evaluate_teacher_refinement.py
.venv/bin/python -m unittest discover -s tests
```

环境依赖增加 `libigl>=2.6,<3`；固定环境清单使用 2.6.3。本地实测 Python 3.9 / PyTorch 2.8 CPU；本轮未验证 CUDA 性能。

## 对照范围

病例划分保留 train=003/002/009/005/008/004/007、val=006/001、test=010/011。两组新训练初始化、姿态库、优化预算与选模规则相同，唯一实验差别是距离方法（它也影响 FPS 选点）。旧 128 点模型作为历史基线，在相同测试姿态重新评分；其训练姿态库、损失与预算不同，因此新旧差异不能全部归因于某一个损失项。

评估同时报告 ranking 与 calibration 最佳权重。MAE、Spearman、局部排序、近区局部排序、分数差误差及“分数升高而误差变差”的局部对比例分别统计。该比例的分母为所有局部比较对，不是 agent 轨迹，也不是仅限分数升高的动作。测试病例曾用于历史实验，属于固定留出病例复测，不代表全新外部临床泛化。当前实验没有重新训练 PPO/diffusion agent，teacher 指标改善不能直接等同于 agent 恢复改善。

所有模型、病例、缓存和原始结果保存在忽略的 `runs/teacher_refinement/` 下。

## 几何与工程验证

- 11 例重新准备的原始顶点、顶点法向和曲率与旧数据逐元素完全一致；病例划分保持不变。
- 每例随机检查 512 个上颌原始顶点，所有表面距离有限且不大于最近顶点距离。平均绝对距离减少约 0.002084 mm；符号不同占约 0.0533%。这是均匀顶点抽样统计，不能代表接触区的全部误差。
- 下颌查询索引排除 009 的 2 个、007 的 1 个、011 的 1 个退化三角形。几何审计记录在 `geometry_audit.json`。
- 21 项单元/回归测试通过，包含解析平面的有符号距离、顶点法向插值、网格重新剖分不改变表面距离、退化面排除、无效索引拒绝、质量标签与损失、训练/推理输入一致性以及扰动文件保留拓扑。
- 统一 CLI 的最近表面 `prepare` 和 `inspect-data` 已在真实病例上验证。

`aligned_pair.selection_weights` 可设置选模的 MAE、分数差误差和局部错序率权重，默认 `[1,1,0.5]`。评估中的“近区”包含原始姿态。

## 本轮实测结果

两组均完成 ranking 32 轮和 calibration 32 轮。验证选出的权重为：统一目标＋最近顶点 ranking 第 29 轮、calibration 第 21 轮；统一目标＋最近表面 ranking 第 29 轮、calibration 第 25 轮。所有权重 SHA-256 在测试评分前写入 `evaluation/locked_checkpoints.json`。测试姿态的几何输入提前生成以减少等待，但模型选取没有使用测试预测。

测试为 010、011 各 256 对，共 512 对、1024 次姿态评分；其中局部对 256 对，排除质量差≤1e-5 后用于方向准确率的有 255 对，近区局部有效对 91 对。以下全部是这次相同测试集上的复测值，不能与旧报告不同姿态分布上的 MAE 直接比较。

### Ranking 阶段

| 模型 | MAE↓ | Spearman↑ | 分数差 MAE↓ | 局部排序↑ | 近区局部排序↑ |
|---|---:|---:|---:|---:|---:|
| 旧 128 点 teacher | 6.224 | 0.597 | 2.712 | 54.5% | 60.4% |
| 统一目标＋最近顶点 | 0.963 | 0.903 | 0.234 | 53.7% | 54.9% |
| 统一目标＋最近表面 | 1.003 | 0.906 | 0.230 | 62.4% | 63.7% |

旧 ranking 没有校准到 q 的绝对尺度，因此它的 MAE 不应单独作为 ranking 失败的证据。局部排序与 Spearman 更适合该阶段比较。最近表面组的 ranking 方向判断在本轮更好。

### Calibration 阶段

| 模型 | MAE↓ | Spearman↑ | 分数差 MAE↓ | 局部排序↑ | 近区局部排序↑ | 升分但误差变差↓ |
|---|---:|---:|---:|---:|---:|---:|
| 旧 128 点 teacher | 0.514 | 0.919 | 0.336 | 60.4% | 60.4% | 21.5% |
| 统一目标＋最近顶点 | 0.457 | 0.928 | 0.273 | 58.8% | 57.1% | 19.5% |
| 统一目标＋最近表面 | 0.515 | 0.924 | 0.293 | 60.4% | 57.1% | 23.8% |

最后一列以全部 256 个局部对为分母。若只看预测升分的有效局部对，其中真实误差变差的比例分别为 50.0%、52.1%、50.0%；不能把最后一列降低解释为每个正奖励都更可信。

- 统一目标＋最近顶点的 MAE 改善 11.2%，分数差 MAE 改善 18.6%，但局部及近区方向准确率未改善。历史基线训练预算与姿态分布不同，这不是对损失函数单项贡献的严格因果估计。
- 最近表面组的全体分数差 MAE 比旧模型改善 12.8%，局部分数差 MAE 从 0.137 降到 0.126，但整体 MAE 和局部方向准确率基本持平，近区与错误正奖励比例更差。
- 最近表面组从 ranking 到 calibration，局部方向准确率从 62.4% 降至 60.4%，近区从 63.7% 降至 57.1%。保留排序和差分损失没有完全避免校准对局部判断的损害。
- 原始姿态的分数绝对误差为旧模型 0.335、统一目标＋最近顶点 0.371、统一目标＋最近表面 0.597，表面组的零点偏差更大。
- 病例间效果不同：表面组在 010 的错误升分比例从 23.4% 降至 19.5%，在 011 却从 19.5% 升至 28.1%。只有两个留出病例、一个训练种子，不能宣称稳定泛化收益。

本轮不替换默认 teacher，也没有重训 agent。两项功能已经实现并验证可运行；当前结果支持“部分回归指标改善”，不支持“agent 的奖励方向已经更可靠”。下一步优先针对 calibration 的近区/零点偏差和局部排序遗忘做独立验证集消融，再评估匹配重训的 agent。

### 训练稳定性与文件

两组训练损失均持续下降，但验证 MAE 后期反弹：最近顶点 calibration 最佳选模轮次 MAE=0.505，末轮=0.570；最近表面对应为 0.473 和 0.492。不能仅凭训练损失下降认定已稳定收敛。训练曲线与最佳权重全部保留，没有按测试结果另选轮次。

- `runs/teacher_refinement/convergence.json`：训练首尾损失、选中轮次、验证指标。
- `runs/teacher_refinement/evaluation/comparison.json`：六组测试汇总和逐病例指标。
- `runs/teacher_refinement/evaluation/*_predictions.npy`：逐对预测；同目录 NPZ 保存姿态、标签、病例 ID 和输入特征。
- `runs/teacher_refinement/aligned_surface/teacher/calibration/best.pt`：第 25 轮新表面距离权重。
- `runs/teacher_refinement/aligned_vertex/teacher/calibration/best.pt`：第 21 轮统一目标消融权重。

统一 CLI 已成功加载新表面模型给 010 原始姿态评分，输出 -0.700543，与离线评估一致；推理输入仍为 128 个点，不需要正确姿态作为网络输入。
