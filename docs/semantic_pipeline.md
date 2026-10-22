# 独立语义区域 / 简面 / Teacher / Diffusion pipeline

入口：`python -m occlusion_rl_offline.semantic_pipeline`，配置：`configs/semantic_pipeline.json`。不依赖旧实验 `runs/` 中的配置或模型；只复用仓库内的基础网络、teacher 损失和 diffusion 训练实现。当前输出写入 `runs/semantic_pipeline_all_vertices`，不覆盖旧实验。用户确认第三阶段采用 **先模仿预训练，再混合奖励 RL**。

当前数据划分固定为：训练 002、003、004、005、007、010；验证 006；测试仅 011。001、008、009 继续排除。分割训练 96 轮。011 不参与任何训练、验证选模或专家轨迹采集。旧模型不作为本轮新训练的初始化。

## 1. 原始咬合标签、体素 PointTransformer 与反映射

- 在原始正确咬合姿态，分别计算上下颌每个原顶点到对颌三角面表面的最近距离，距离 ≤5 mm 标为正类。非封闭网格不生成体积 SDF。
- 标签附着于原顶点；增强时不根据扰动后的上下颌距离重新生成标签。单颌输入为坐标、原顶点法向、原完整网格曲率、上下颌标识，共 8 维。
- 每个占据体素选一个最接近体素内顶点质心的原顶点。法向和曲率直接取该原顶点值，不平均为新的虚拟点。体素原点相对点云均值固定，避免纯平移改变网格。
- 默认体素边长 1 mm；若占据体素超过 2048，自动增大边长直到满足内存预算。不会随机丢弃体素。每次记录实际边长、代表原顶点 ID、所有原顶点到体素的 inverse map。
- 训练使用均匀 SO(3) 随机旋转及平移，坐标和法向一起转，曲率和标签不变；**旋转后重新体素采样**。变长体素点云逐例训练，不用重复点填充。
- 模型为已有紧凑局部 PointTransformer，按验证集全原顶点 IoU 选择 checkpoint。不同旋转下的鲁棒性需用独立测试验证，数据增强本身不构成保证。
- 每个体素预测概率反映射给该体素包含的所有原顶点。三个顶点均为正类的原三角面组成区域；不创建新的区域面片，也不使用参考标签补齐预测。

可视化包含病例、上下颌、三种实际输入旋转，以及原顶点预测、5 mm 标签、错分/漏分、实际网络体素点、简面网格五种模式。IoU/Dice 基于全部原顶点；页面绘制最多 12000 个原顶点作为显示子集。完整原网格彩色 PLY 和原顶点映射 NPZ 另存，保留全部顶点/面片。

## 2. 保留网格特征的简面和五档 Teacher

采用 [VTK DecimatePro](https://vtk.org/doc/nightly/html/classvtkDecimatePro.html) 的三角网格顶点删减，开启拓扑保护，关闭拆边，以相邻面法向夹角区分尖锐边，默认 feature angle 为 30°。保留边界拓扑，允许沿边界删除冗余顶点；`lock_boundary_vertices=true` 可锁住边界顶点，但可能无法满足预算。不是随机点下采样，也不补洞。

循环以实际顶点数检查 **≤4095（严格小于 4096）**。若特征/拓扑约束阻止继续简化则失败，不静默关闭约束。通过坐标核对保留顶点仍是原 STL 顶点，法向和曲率保持原完整网格值；简面三角连接关系会变化，不能把简化后的三角面冒充为原面片。

先在输入姿态预测区域并简面，然后固定该子网格，在上颌施加扰动；下颌固定。一次轨迹中不重新分割/简面，避免每步改变几何身份。teacher 和 agent 直接编码简面全部上颌顶点，不再进行 FPS 或近接触点筛选；下颌全部简面三角面用于距离查询。表征仍为坐标、顶点法向、曲率、对下颌三角面的局部有符号距离（8 维）。变长批次用 NaN 行填充，编码前逐例移除填充行，不参与注意力或 pooling。编码器的距离加权 pooling 保留；取消的是距离加权 FPS。符号来自最近三角面处插值的原顶点法向。

| 扰动档 | 旋转向量各分量范围 | 平移各分量范围 | 概率 |
|---|---:|---:|---:|
| 正确姿态 | 0° | 0 mm | 10% |
| 近区 | ±1.5° | ±0.5 mm | 25% |
| 较小 | ±4° | ±2 mm | 25% |
| 中等 | ±8° | ±4 mm | 25% |
| 较大 | ±20° | ±8 mm | 15% |

这些是嵌套的采样范围，并非互斥球壳；总旋转角可大于各分量限值。teacher 同时比较独立位姿对和局部动作对，使用统一的 SO(3) 角度/平移质量标签、ranking、回归和分差损失。当前仅训练 ranking 96 轮，每病例 512 对，不执行 calibration；agent 使用 ranking 原始输出的逐步差值。ranking 阶段原有回归、排序、分差联合损失保留。GT 位姿仅用于监督标签和评估，不拼进网络观测。

## 3. Diffusion 模仿预训练与混合奖励 RL

先按 bounded expert 生成动作块，进行 4200 次 diffusion 噪声预测更新，再运行 150 轮随机去噪链 PPO。继承已有实现：冻结参考策略及编码器，训练末尾去噪步骤；每次执行预测动作块的第一步，下一步重新规划。当前正式配置和小预算模式均固定执行 **16 次姿态调整**，统一由 `horizon` 控制；diffusion 去噪次数仍由 `inference_steps` 独立控制。每轴平移步长上限仍为 0.4 mm，因此 16 步最多覆盖每轴 6.4 mm，五档中的 ±8 mm 大扰动不保证完全恢复。

专家轨迹奖励在 **learner 实际到达的训练状态** 重新生成专家下一步 waypoint，避免偏离初始示教路径后奖励失效。专家动作是有步长限制、不超调的真实位姿逆变换。用执行动作前后到同一个 waypoint 的归一化距离改善作为奖励：

```
D(state, expert_next) = 0.5 × [SO3角度/(√3×旋转步长) + 平移距离/(√3×平移步长)]
r_expert = expert_scale × [D(before, expert_next) − D(after, expert_next)]
r_teacher = teacher_scale × [score(after) − score(before)]
r = w × r_expert + (1−w) × r_teacher
```

默认专家权重从 1 降到 0，teacher 权重从 0 升到 1；前 10% 训练保持专家权重，10%–80% 用 smoothstep 平滑切换，后 20% 为纯 teacher 分差奖励。权重在整条 rollout 内固定，按原始训练总预算计算，恢复训练不会重启衰减。scale 默认 1/2。没有额外 BC loss，轨迹信息只进入奖励，不进入策略输入。这里不是“总误差变小”的旧 GT 奖励，而是到专家下一轨迹点的改善。

记录每轮专家奖励、teacher 分差、两项权重、总奖励、降分动作数、PPO 更新数。所有合法降分动作照常执行，固定 horizon，不设置成功阈值提前停止。验证选模型，测试不参与选择。预训练奖励和 RL 的参考策略 KL 是不同项：前者衰减，后者由 `reference_kl_coef` 控制。

## 运行

```bash
.venv/bin/python -m pip install -e '.[semantic]'
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m occlusion_rl_offline.semantic_pipeline
# CUDA 环境下，先按照 docs 中现有 GPU 环境说明安装对应 PyTorch：
.venv/bin/python -m occlusion_rl_offline.semantic_pipeline --device cuda
```

依次执行 `prepare segment segment-test regions configure teacher imitate rl test end-to-end visualize`。可用 `--stages` 指定尚未执行的阶段；已完成阶段拒绝覆盖。所有阶段使用同一份配置快照，更改参数需换输出目录，避免混用旧缓存。

```bash
# 小预算流程联调，单独输出到 runs/semantic_pipeline_all_vertices_smoke_h16
.venv/bin/python -m occlusion_rl_offline.semantic_pipeline --smoke
# 只生成可视化（需要该 run 已有分割 checkpoint）
.venv/bin/python -m occlusion_rl_offline.semantic_pipeline --stages visualize
# 本地显示工具，不依赖 CDN、登录或联网
.venv/bin/python -m http.server 8770 --bind 127.0.0.1 --directory runs/semantic_pipeline_all_vertices/viewer
```

打开 http://127.0.0.1:8770/ 。新 16 步小预算运行的 viewer 位于 `runs/semantic_pipeline_smoke_h16/viewer`。模型、准备数据、PLY/STL、可视化载荷均在忽略的 runs 目录，不提交。

直接处理未知姿态原 STL 对：

```bash
.venv/bin/python -m occlusion_rl_offline.semantic_pipeline.infer \
  --run runs/semantic_pipeline \
  --upper /path/to/observed_upper.stl --lower /path/to/lower.stl \
  --out runs/semantic_observed
```

该接口从当前观测分割、反映射、简面后执行完整 agent 轨迹，输出 `trajectory.json`、变换矩阵和完整原上颌的 `upper_recovered.stl`，不需要参考位姿或标签。

当前 manifest：训练 003、002、005、004、007、010；验证 006；测试仅 011，排除 001、008、009。新数据接口只需序号病例目录下 `upper.stl/lower.stl`，并在 source_manifest 中明确三组不重叠 ID；prepare 重新从原 STL 生成全部标签和几何。

`agent_test.json` 的常规对照是在固定预测子网格上扰动，隔离策略表现；真实未知输入还会产生分割/简面分布变化，应同时检查完整 STL 输入的端到端结果。不要把固定子网格测试等同于全流程泛化精度。

全流程测试 `end_to_end/summary.json` 在测试病例的完整网格上先施加扰动，再重新分割/简面和推理。失败病例保留在分母，误差均值标为仅对成功输入的条件均值。可单独运行 `python -m occlusion_rl_offline.semantic_pipeline.end_to_end --run runs/semantic_pipeline`。

## 本次实现验证（2026-10-20）

全部 11 个阶段已在现有 8 个病例上完成小预算联调，输出 `runs/semantic_pipeline_smoke`。分割 32 轮、512 体素上限、24 通道/2 层；teacher ranking/calibration 各 2 轮、每例 16 对；diffusion 模仿更新 16 次，RL 3 轮，每轮 2 条环境、每条 3 步。**这不是正式预算训练，teacher/agent 尚未收敛。** 当时正式配置是 2048 体素上限、48 通道/3 层、分割 48 轮、teacher 各 32 轮、模仿 2800 次、RL 100 轮；当前配置已改为 16 步。

- 分割根据验证集选 epoch 28；独立测试原姿态及随机 SO(3) 旋转，平均 IoU **0.7684**、Dice **0.8674**、precision **0.8812**、recall **0.8582**。输出翻转比例平均 **0.0964**（含原始姿态的零翻转项），不能理解为完全旋转不变。
- 16 个上下颌区域全部满足顶点预算，简面后 **3556–3949** 顶点。保留顶点坐标/法向/曲率均映射至原网格。
- 简面距离审计每颌采样最多 2048 个原区域顶点，到简面三角面的距离 p95 范围 **0.077–0.543 mm**；病例 002 下颌的采样最大误差约 **3.61 mm**，存在局部细节损失。此采样统计不是完整 Hausdorff 距离。
- 6/16 网格的 Euler 数发生变化。VTK 的拓扑保护选项并不能保证这些复杂/非流形原网格完全保持拓扑，因此不得称为无损简面。`regions/audit.json` 逐病例记录变化；正式使用前应检查这些病例，必要时修复源网格或调整简面约束，并重新准备数据。当前代码不会暗中修补原网格。
- RL 的实际权重依次为 `(1,0)`、`(0.393586,0.606414)`、`(0,1)`，每轮均完成 2 次优化更新。日志保留专家/teacher 奖励和降分动作。
- 小预算 teacher 测试局部排序准确率 ranking **43.75%**、calibration **31.25%**。小预算 agent 在固定子网格测试中误差未改善，不是可部署的恢复模型。完整 STL 输入的 4 个测试均能完成分割→简面→3 步推理，无失败，但恢复效果同样不佳。这里的“完成”仅指算法流程未报错。
- 38 项测试通过，包含 checkpoint 实际权重加载回归测试。可视化 48 个病例/牙颌/旋转组合、5 种模式、拖动/重置和移动布局通过浏览器测试，无 JavaScript 错误。

开发中曾发现分割推理入口漏载 checkpoint 权重，现已修复并增加回归测试；此前 47.5% 的测试 IoU 和对应区域输出无效，当前上述结果全部由修复后的加载器重新生成。8 轮调试目录及错误结果只留作本地排错，不参与正式输出。

下一步正式实验应先确认分割的旋转稳定性、审查简面几何损失，再按独立阶段执行 teacher 与 agent 的完整预算；不要根据这次小预算 agent 的误差决定算法优劣。

独立未知 STL 推理 CLI 也已执行：扰动后的 010 上颌与固定下颌输入，不提供目标标签；原始 275307 个上颌面片全部保留，导出顶点与恢复矩阵变换结果一致。正式预算配置生成及解析检查通过。

## Teacher 与 Agent 可视化

三页共用本地 `viewer/` 目录，页面顶部相互导航：

- `index.html`：语义分割。
- `teacher.html`：当前仅显示 ranking 实际评分（旧导出页面保留历史 calibration 对照）；Rx/Ry/Rz、Tx/Ty/Tz 六个单轴扫描（各 21 帧）及五档固定种子的预设位姿。显示三角网格、简面后的全部实际输入顶点、旋转/平移误差和监督质量。监督质量仅为对照，不冒充 teacher 输出；分数不是概率。显示的是逐帧实际推理结果，不支持任意六轴组合的在线打分。
- `agent.html`：固定预测区域的配对随机起点恢复，支持模仿基线与 RL 切换；基线采用与 RL 一致的随机推理方式。可播放、逐步、滑动或点击图表定位，查看 teacher 分差、实际动作增量、角度/平移误差、输入点和目标对照。另提供“完整 STL”入口，回放先对扰动 STL 重新分割/简面后的端到端结果。页面单独绘制训练轮次上的奖励权重，避免与推理步数混淆。

历史小预算结果：测试病例 010、011，318 个 teacher 位姿、56 条 agent 轨迹（52 条固定区域配对轨迹＋4 条完整 STL 轨迹），共 168 个动作，74 个降分动作全部保留。只按固定随机种子与轨迹序号取样，不按成功与否选择。端到端记录已用当前 checkpoint 重放动作、重算分数和最终误差，检查不一致会阻止导出。

完整流程的 `visualize` 阶段会自动生成已有模型对应的页面。已有实验可独立重新生成，无需重训、无需重跑分割：

```bash
.venv/bin/python -m occlusion_rl_offline.semantic_pipeline.recovery_viewer \
  --run runs/semantic_pipeline_smoke
# 正式结果换成 --run runs/semantic_pipeline；也可仅生成一页：
# --kind teacher 或 --kind agent
# 固定区域每档每例随机起点数：--episodes 3（正确位姿仍只 1 个）
# 单轴扫描帧数：--sweep-points 21（必须为 >=3 的奇数）
.venv/bin/python -m http.server 8770 --bind 127.0.0.1 \
  --directory runs/semantic_pipeline_smoke/viewer
```

打开 http://127.0.0.1:8770/teacher.html 或 http://127.0.0.1:8770/agent.html 。数据载荷与截图留在忽略的 runs 目录，源 HTML/JS/CSS 随 Python 包安装，不依赖 CDN。40 项测试通过；两个页面的病例/位姿切换、配对策略切换、播放、步进、网格/输入点开关、端到端模式、相机和移动布局通过浏览器检查，无 JavaScript 错误。本段为历史联调记录；本次正式实验结果见实验记录。

## 轨迹步数调整

当前 `horizon=16` 同步传递到模仿轨迹采集、RL rollout、验证/测试、端到端推理和可视化；小预算模式不再覆盖成 3 步。新小预算输出目录带步数后缀 `_smoke_h16`，避免与旧 3 步实验混用。旧 `runs/semantic_pipeline_smoke` 中的 checkpoint、结果和页面保留历史的 3 步，不修改为伪造的 16 步记录。已完成独立 16 步小预算运行，见下方记录。


### 2026-10-20：16 步实际训练与验证

输出 `runs/semantic_pipeline_smoke_h16`。对比配置确认仅 out/horizon 改变，复用旧运行的准备数据、分割、简面及 teacher；来源和 checkpoint SHA256 记录于 `reuse_audit.json`。重新采集 5 个训练病例的 160 个专家状态，进行 16 次模仿更新及 3 轮混合奖励 RL，每轮 2×16=32 次交互。验证选择第 1 轮 RL。该预算仅用于流程联调，不代表正式预算训练完成。

测试病例仍为 010、011，验证为 006。固定区域测试的非零扰动宏平均误差：初始 4.036° / 1.928 mm，模仿后 12.903° / 3.512 mm，RL 后 12.905° / 3.512 mm。16 步流程正常，但恢复效果变差；teacher 分差与真实误差仍存在不一致。完整 STL 的 4 个输入均完成重新分割、简面和 16 步推理，这里的完成不等于恢复成功。

可视化含 56 条轨迹，每条 17 帧（起点加 16 个动作），共 896 个动作，其中 443 个降分动作照常保留。teacher 页面含 318 个预设位姿。浏览器校验确认末帧为 16/16、完整 STL 模式可切换、移动端无横向溢出、无 JS 错误；40 项单元测试通过。

```bash
.venv/bin/python -m http.server 8771 --bind 127.0.0.1 --directory runs/semantic_pipeline_smoke_h16/viewer
```

打开 `http://127.0.0.1:8771/agent.html` 查看新轨迹，`teacher.html` 查看评分。历史 3 步 demo 保留在旧目录。


### 当前修改验证与历史结果说明

以上历史实验使用 256 点 FPS 和 calibration，不能作为当前全顶点、ranking-only 配置的结果。本次实际训练、测试已经完成：分割 96 轮，teacher 原定 96 轮、执行 57 轮后人工早停，模仿 4200 次更新，RL 150 轮；详见 [实验记录](experiments/semantic_all_vertices_20261021.md)。运行 `.venv/bin/python -m occlusion_rl_offline.semantic_pipeline` 会从新目录执行全部阶段；不要复用旧 256 点 checkpoint。内部共享训练器保留 `fps_points=4095` 作为检查点预算元数据，当前 all_mesh 路径不会据此筛点或补点；实际点数是简面顶点数。小预算配置仅用于检查运行，不代表正式预算。

本轮修改验证：42 项单元测试通过；真实病例 002/003 的 3637/3618 个简面顶点全部进入网络，变长配对 bank、ranking 损失反传及 diffusion 去噪损失反传均通过有限梯度检查。未使用 011 做梯度验证。
