# occlusion_rl_offline

离线咬合姿态恢复实验仓库。输入是已经处于正确咬合关系的上、下颌 STL；训练时随机扰动上颌，冻结的 teacher 只提供咬合分数，PPO agent 学习逐步旋转和平移上颌。推理时输入**已偏移的上颌**及固定下颌的 SDF，不需要原始姿态标签。

## 安装与数据

需要 Python 3.9+。在本目录运行：

```bash
python -m venv .venv
.venv/bin/pip install -e .
source .venv/bin/activate
occlusion-rl inspect-data
```

`examples/data/` 包含本项目当前的一例对齐上下颌 STL 和预处理好的 `case.npz`（SDF 间距 0.2 mm）；`examples/checkpoints/` 包含该例的初始、排序、校准 teacher 和 PPO agent 权重。默认配置位于 `configs/default.json`。命令中的相对路径均相对于仓库根目录，`--config` 可以指定完整的替代 JSON 配置。`runs/` 保存运行结果且不纳入 Git。

对新的对齐 STL，先改配置中的 `paths.upper_stl`、`paths.lower_stl`，然后执行：

```bash
occlusion-rl prepare --out runs/my_case/case.npz
occlusion-rl inspect-data --case runs/my_case/case.npz
```

`prepare` 从上颌咬合面带采样点，提取坐标、面法线、局部法线变化量（曲率代理值），并由下颌网格生成 SDF。当前 STL 不是封闭网格，SDF 使用下颌表面采样点的最近邻距离和对应面法线估计符号；`*.audit.json` 会记录参数和抽样统计。新病例应检查法线方向与 SDF 符号，再重新训练权重；附带的权重只在附带病例上验证过。

## 训练与评估

```bash
occlusion-rl train-teacher --case runs/my_case/case.npz --out runs/my_case/teacher
occlusion-rl train-agent --case runs/my_case/case.npz \
  --teacher runs/my_case/teacher/calibration/best.pt --out runs/my_case/agent
occlusion-rl evaluate-agent --case runs/my_case/case.npz \
  --teacher runs/my_case/teacher/calibration/best.pt \
  --agent runs/my_case/agent/agent.pt --out runs/my_case/test
```

`train-teacher` 先进行近/远姿态排序训练，再进行分数校准；两阶段都使用对齐病例及其随机扰动。`train-agent` 从 teacher 编码器初始化策略编码器，但 teacher 本身冻结。每步唯一的环境奖励为 `reward_scale × (teacher_score_after - teacher_score_before)`；姿态误差只用于离线评估，**不进入奖励**。两网络读取相同的 8 通道点特征：上颌 `xyz`、法线、曲率、下颌 SDF。先固定随机种子取 `candidate_points` 个点，再在网络内执行 SDF 加权 FPS 取 `fps_points` 个点；采样会偏向接触和穿透区域，同时保留空间覆盖。配置要求 teacher 与 agent 使用相同的点数与模型采样规则。

`configs/default.json` 集中管理影响准确度的变量，包括 SDF 栅格间距与采样密度、咬合面带、FPS 权重和尺度、编码器宽度及邻域、teacher 的姿态分布与损失、agent 的初始扰动范围、动作步长、步数、策略宽度、探索方差、PPO 优化和评估阈值。修改模型结构参数后，须重新训练对应权重；直接加载旧权重可能因尺寸不匹配而失败，或因采样规则变化而使分数失效。每次训练保存 `config.snapshot.json`。

默认 `evaluate-agent` 从对齐样本生成新的随机初始姿态，用训练以外的随机种子报告旋转、平移、teacher 分数和严格成功率。该指标度量*已知正确姿态的同一病例*，不代表跨病例泛化。

## 推理与轨迹

可先用已知扰动创建可复现的推理输入：

```bash
occlusion-rl perturb --out-case runs/demo/case.npz \
  --rotation-rvec-deg 4 -3 2 --translation-mm 1 -1 1
occlusion-rl infer-agent --case runs/demo/case.npz \
  --upper-stl runs/demo/upper_perturbed.stl --out runs/demo/recovered
```

`infer-agent` 输出每一步的上颌姿态矩阵、动作和 teacher 分数到 `trajectory.json`，以及恢复后的 `upper_recovered.stl`。这些文件可以直接作为可视化 demo 的轨迹与网格输入。实际扫描输入需先准备与训练时相同规则的上颌点特征，并保留下颌 SDF；上颌 STL 与 `case.npz` 中的点应处于同一初始坐标系。`score --case ...` 可单独查看某个姿态的 teacher 分数。

`demo/index.html` 附带基于当前病例和权重预先生成的浏览器回放，展示上颌每步运动、teacher 分数以及 SDF 加权 FPS 选点。可直接用浏览器打开；其 `data.js` 是固定示例，单次 `infer-agent` 的结果另存于上述 `trajectory.json`。

默认附带权重在 256 个新扰动上的复核结果：平均旋转误差 7.64°→2.38°、平移误差 3.72→0.49 mm；按旋转 <1° 且平移 <0.5 mm 计算，严格成功 3/256。分数改善不保证完全恢复原始姿态，因为奖励只来自 teacher 分数。
