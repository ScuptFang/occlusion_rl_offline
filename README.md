# occlusion_rl_offline

新增 **5 mm 咬合候选区域 PointTransformer 分割 → teacher → 模仿学习与 diffusion RL**，使用原始顶点标签、随机位姿训练及预测区域推理，见[实验配置与说明](docs/region_point_transformer.md)。

区域流程也支持跳过 teacher 第二阶段校准，重新训练 agent，见 [ranking teacher 对照实验](docs/region_ranking_agent.md)。

新增 **模仿学习 → diffusion 强化学习**：在预训练扩散策略的最后几步去噪上进行 teacher 差分奖励 PPO 微调，见 [配置、算法与对照实验](docs/diffusion_rl.md)。

Teacher 新实验：统一 ranking/calibration 标签、保留分数差监督，并支持最近下颌三角网格表面距离；配置与复测结果见 [teacher 改进实验](docs/teacher_refinement.md)。


新增 **ReAgent 启发的模仿学习 → 位姿/教师混合奖励 PPO**：使用 `configs/reagent_curriculum.json`，见 [流程、命令与测试结果](docs/reagent_curriculum.md)。该模式允许真实位姿仅在训练阶段提供专家标签和衰减奖励；下文 teacher-only 的奖励描述仅适用于旧模式。本次新增内容不包含 demo、病例数据或权重；现有分支历史中的文件不重复上传。
当前状态与问题分析见[训练链路审计](docs/current_pipeline_review.md)，实验顺序及比较指标见[测试与优化计划](docs/recovery_test_plan.md)，已执行结果见[连续误差优化报告](docs/optimization_20261001_report.md)。优化目标为角度和平移误差越小越好，agent 执行完整动作预算、不提前停止；1.5°/0.5 mm 仅作历史参考。

离线咬合姿态恢复实验仓库。输入是已经处于正确咬合关系的上、下颌 STL；训练时随机扰动上颌，冻结的 teacher 只提供咬合分数，PPO agent 学习逐步旋转和平移上颌。推理时输入**已偏移的上颌**及固定下颌的原始顶点/顶点法向，计算带符号最近顶点距离，不需要原始姿态标签或体积 SDF。

## 安装与数据

需要 Python 3.9+。在本目录运行：

```bash
python -m venv .venv
.venv/bin/pip install -e .
source .venv/bin/activate
occlusion-rl inspect-data
```

`examples/data/` 包含一例对齐上下颌 STL 和旧版预处理 `case.npz`（SDF 间距 0.2 mm）；`examples/checkpoints/` 是旧版低点数、2 层编码器权重，与当前配置不兼容。默认配置位于 `configs/default.json`；当前不生成下颌距离网格。命令中的相对路径均相对于仓库根目录，`--config` 可以指定完整的替代 JSON 配置。`runs/` 保存运行结果且不纳入 Git。新规则下先从 STL 重新 `prepare`，再训练 teacher 和 agent；未显式传 `train-teacher --source` 时 teacher 从随机初始化开始。

对新的对齐 STL，先改配置中的 `paths.upper_stl`、`paths.lower_stl`，然后执行：

```bash
occlusion-rl prepare --out runs/my_case/case.npz
occlusion-rl inspect-data --case runs/my_case/case.npz
```

多病例训练可用 `prepare-batch` 从 `case_id/upper.stl`、`case_id/lower.stl` 目录批量建立数据与 train/val/test 清单；已有 `case_id/case.npz` 则用 `index-cases`。完整目录格式、训练命令和病例划分规则见 [大量病例数据输入](docs/multiple_cases.md)。本地编号 STL 的对应关系见[数据目录说明](docs/data_layout.md)。

10 例免登录公开病例及演示病例的来源、坐标规范化见 [公开病例试验](docs/public_cohort_experiment.md)。当前全网格 128 点的 010 单病例训练与可视化见 [010 全网格实验](docs/case010_mesh128_report.md)，ranking/calibration 打分及 FPS/pooling 消融见 [teacher 诊断](docs/teacher_sampling_diagnostics_010.md)。上一版最近顶点距离、256/64 点且 PPO 不拒绝降分动作的训练结果见 [新规则重训报告](docs/retraining_nearest256_report.md)；历史 0.6 mm 网格、1024 FPS 点结果见 [旧规则重训报告](docs/retraining_sdf06_report.md)。逐档测试协议见 [恢复测试计划](docs/recovery_test_plan.md)。

当前四档 ranking 姿态和 4096 候选点 / 64 点加权 FPS 规则见 [采样与比较规则](docs/ranking_bands_candidate4096.md)，010 单病例使用 `configs/case010_candidate4096_fps64.json`。上述实验结果均为各自历史配置下的结果。

新采样规则下 010 病例的 ranking、calibration 训练与独立可用性测试见 [teacher 可用性报告](docs/teacher_candidate4096_usability_010.md)。

使用 `data/001`–`011` 全部病例扩量训练 teacher，并在留出的 010、011 病例逐例测试，见 [11 例 teacher 训练报告](docs/data11_teacher_more_training.md)。

`prepare` 保留上颌网格的全部原始顶点，不在三角面上生成新点；每个点使用该顶点的平滑顶点法向。曲率由三角面的顶点、边、面积及余切权重计算离散平均曲率，按每例 99% 分位归一化到 0–1。下颌保留原始网格顶点及顶点法向；每次计算上颌点特征时查找最近的下颌顶点，以欧氏距离为绝对值，以位移和该顶点法向的点积判断符号。无需封闭网格，也不生成距离网格。`*.audit.json` 会记录顶点来源和 SDF 抽样统计。更改上颌点、法向与曲率规则后，须从 STL 重新运行 `prepare` 或 `prepare-batch`（不要使用 `--skip-existing`），并重新训练 teacher 和 agent；已有 `.npz` 和权重仍采用旧规则。

## 训练与评估

```bash
occlusion-rl train-teacher --case runs/my_case/case.npz --out runs/my_case/teacher
occlusion-rl train-agent --case runs/my_case/case.npz \
  --teacher runs/my_case/teacher/calibration/best.pt --out runs/my_case/agent
occlusion-rl evaluate-agent --case runs/my_case/case.npz \
  --teacher runs/my_case/teacher/calibration/best.pt \
  --agent runs/my_case/agent/agent.pt --out runs/my_case/test
```

`train-teacher` 先进行四档姿态排序训练，再进行分数校准；两阶段都使用对齐病例及其随机扰动。ranking 在“接近正确、较好、较差、明显错误”四档之间取有序姿态对，覆盖相邻档和跨档比较；具体幅度见 `teacher.ranking_pose_bands`。`train-agent` 从 teacher 编码器初始化策略编码器，但 teacher 本身冻结。每步唯一的环境奖励为 `reward_scale × (teacher_score_after - teacher_score_before)`；真实姿态误差**不直接进入 agent 奖励**。两网络读取相同的 8 通道点特征：上颌网格顶点 `xyz`、顶点法向、网格曲率、下颌带符号最近点距离。每个病例固定从原上颌网格均匀抽取 4096 个候选顶点，每次姿态变化后计算候选点距离，再用距离加权 FPS 取 64 个点。teacher、agent 与推理使用相同的候选点和选点规则。原有权重使用其他采样规则，需要重新训练。

分数校准本身使用已知正确姿态构造监督目标，因此姿态误差信息会间接进入 teacher；它不改变 agent 的分数差奖励公式。排序版和校准版的单病例对照见 [校准消融](docs/calibration_ablation.md)。

`configs/default.json` 集中管理影响准确度的变量，包括 FPS 点数、权重和尺度、编码器宽度、层数及邻域、teacher 的姿态分布与损失、agent 的初始扰动范围、动作步长、步数、策略宽度、探索方差、PPO 优化和评估阈值。当前编码器使用 3 层局部注意力。修改模型结构参数后，须重新训练对应权重；直接加载旧权重可能因尺寸不匹配而失败，或因采样规则变化而使分数失效。每次训练保存 `config.snapshot.json`。

默认 `evaluate-agent` 从对齐样本生成新的随机初始姿态，用训练以外的随机种子报告旋转、平移、teacher 分数和严格成功率。该指标度量*已知正确姿态的同一病例*，不代表跨病例泛化。

## 推理与轨迹

新规则训练完成后，可先用已知扰动创建可复现的推理输入（以下权重路径由新训练生成）：

```bash
occlusion-rl --config configs/public_cohort_optimized.json perturb \
  --case runs/my_128/prepared/cases/011/case.npz \
  --upper-stl ../data/011/upper.stl --out-case runs/demo/case.npz \
  --rotation-rvec-deg 4 -3 2 --translation-mm 1 -1 1
occlusion-rl --config configs/public_cohort_optimized.json infer-agent \
  --case runs/demo/case.npz --upper-stl runs/demo/upper_perturbed.stl \
  --teacher runs/my_128/teacher/calibration/best.pt \
  --agent runs/my_128/agent/agent.pt --out runs/demo/recovered
```

`infer-agent` 输出每一步的上颌姿态矩阵、动作和 teacher 分数到 `trajectory.json`，以及恢复后的 `upper_recovered.stl`。这些文件可以直接作为可视化 demo 的轨迹与网格输入。实际扫描输入需先准备与训练时相同规则的上颌点特征，并保留下颌原始顶点和顶点法向；上颌 STL 与 `case.npz` 中的点应处于同一初始坐标系。`score --case ...` 可单独查看某个姿态的 teacher 分数。

`demo/index.html` 现在使用[010 全网格 128 点实验](docs/case010_mesh128_report.md)的 agent 和 010 STL，包含 3 条可切换的 8 步随机扰动轨迹，可直接在浏览器打开。页面逐步显示教师分数、旋转和平移误差、动作及从全部上颌原网格顶点选出的 128 个输入点。PPO 不拒绝降分动作；回放忠实显示分数下降和未达标轨迹。在仓库根目录执行 `.venv/bin/python demo/generate.py` 可复现。

新训练的多随机位姿 agent 回放见 [demo/multi_pose.html](demo/multi_pose.html)：010、011 各有近、中、远三档，每档 32 条，共 192 条 10 步测试轨迹。可按病例、扰动档位和位姿编号筛选、播放或逐步拖动，并查看对应上下颌、4096 个均匀候选点、64 个网络输入点、教师分数及姿态误差。运行 `.venv/bin/python demo/generate_multi_pose.py` 可重新生成 `demo/multi_pose_data.js`。训练与测试结果见 [多随机位姿 agent 报告](docs/data11_agent_multi_pose_report.md)。

新的[接触区优先 FPS 诊断](docs/contact_fps_sampling_diagnostics.md)在 64 点预算中保留 32 个全颌覆盖点、另外选取 32 个近下颌点；配置见 `configs/data11_contact_fps.json`，默认配置也采用这套规则。已用新采样重训 teacher 和 agent；训练、独立测试及旧采样对比见[接触区优先采样重训报告](docs/data11_contact_fps_retraining.md)。历史 192 条回放仍对应改动前的采样。

2026-10-01 的[连续误差优化实验](docs/optimization_20261001_report.md)包含 teacher 联合排序/回归与采样一致性、固定顶点 ID 消融、新 PPO、三训练种子及同起点测试。新栈保留有符号 teacher 分差作为唯一奖励，完整执行动作预算，没有 STOP 或降分回退。首轮减少了部分漂移，但平移误差和种子稳定性仍未解决；原默认配置未被替换为实验模型。

本轮权重可用 `configs/optimization_selected.json` 加载（本地权重位于 `runs/optimization_20261001/agent_control/agent.pt`，teacher 路径已配对）。[持续动作演示](demo/optimized.html)包含 010/011 共 50 条 40 步轨迹，展示零/近/中/远固定起点、64 个真实输入顶点、每步分数和两项误差。演示使用同一 10 步训练权重，在推理时加入粗细动作阶段；它是长轨迹消融，不应与独立 10 步测试混同。HTML 与旁边的 `optimized_data.js` 可直接在浏览器打开。

旧版附带权重在旧规则下的 256 个扰动上曾得到旋转误差 7.64°→2.38°、平移误差 3.72→0.49 mm，严格成功 3/256。当前 11 例重训的独立测试结果见[新规则重训报告](docs/retraining_nearest256_report.md)；分数改善不保证完全恢复原始姿态，因为奖励只来自 teacher 分数。

Teacher 位姿评分可视化：[打开评分视图](demo/teacher_scores.html)。直接用浏览器打开即可，支持 001–011 病例、六个单轴旋转/平移扫描和随机组合姿态；拖动姿态滑条或点击分数曲线可查看对应上下颌及网络实际输入点。「原始姿态」「本组最高分」可用于检查 teacher 是否偏好偏离原始咬合的姿态。曲线是离线推理，不是任意姿态的实时预测；不运行 agent。

重新生成评分数据（默认使用 `optimization_selected.json` 中与选中 agent 配套的 teacher；可用 `--teacher` 指定匹配配置的其他模型）：

```bash
.venv/bin/python demo/generate_teacher_scores.py
# 只导出指定病例
.venv/bin/python demo/generate_teacher_scores.py --cases 010 011
```

每病例包含 279 个姿态：每轴旋转 −10° 到 +10°（0.5° 间隔），每轴平移 −4 到 +4 mm（0.2 mm 间隔），以及原始姿态加 32 个固定随机种子的组合姿态。旋转使用绕上颌顶点中心的旋转向量，组合姿态每分量范围 ±8°、±4 mm。采样沿用配置的 4096 个均匀候选顶点 → 距离加权 FPS 64 点，保留顶点法向和网格曲率；带符号距离由下颌最近顶点法向判断，不生成体积 SDF。背景表面点仅用于显示，不作为网络输入。模型路径和 SHA256 显示在界面中。train/val/test 标签沿用现有 manifest；010/011 曾用于探索，不代表全新独立测试集。

最新训练/推理栈与 NVIDIA GPU 环境见 [GPU 使用说明](docs/gpu_training.md)。从头训练配置为 `configs/latest_gpu.json`；支持统一 `--device auto|cpu|cuda|cuda:N`，数据准备、teacher、PPO、评估、推理及 demo 的完整命令均在说明中。

模仿学习课程的[128 点 / PPO 宽度 512 扩容配置与测试](docs/reagent_capacity.md)使用 `configs/reagent_curriculum_large.json`，保留原 64 点 / 宽度 128 配置作同起点对照。

[逐步 Diffusion Policy 实验](docs/diffusion_policy.md)使用 `configs/diffusion_policy.json`：从当前点云去噪预测 4 步动作，每次只执行首步再重规划，保存完整逐步轨迹，并与相同 128 点输入的 PPO 对照。
