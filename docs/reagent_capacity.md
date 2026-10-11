# 输入点数和 PPO 容量扩展

配置：`configs/reagent_curriculum_large.json`。保留 `configs/reagent_curriculum.json` 作为 64 点基线。两个配置使用相同的模仿预训练、位姿/教师奖励退火、动作幅度、训练轮数和病例划分。

| 项目 | 基线 | 扩容 |
|---|---:|---:|
| 原网格均匀候选顶点 | 4096 | 4096 |
| teacher / agent 实际输入点 | 64 | 128 |
| 每点输入特征 | 8 | 8 |
| encoder 层数 / 宽度 | 3 / 64 | 3 / 64 |
| PPO 共享隐藏层宽度 `agent.policy_width` | 128 | 512 |
| 策略/价值部分参数（含 log_std） | 42,765 | 171,021 |
| agent 总参数（含不参与 agent 训练的 teacher head） | 119,761 | 248,017 |
| agent 可训练参数 | 98,896 | 227,152 |

PPO 结构为 `326 → policy_width → actor(6) / critic(1)`，共享隐藏层使用 Tanh。326 维由 encoder 的 324 维汇总特征与 2 维时间上下文构成。本次加宽 PPO，不改变 teacher encoder 的结构。点数增加本身不增加参数量，但会增加局部注意力、距离矩阵、采样和反向传播的计算与内存开销。

实际输入为 `[batch, 128, 8]`：XYZ、顶点法向、网格拓扑曲率、到下颌的有符号最近顶点距离。距离符号仍由下颌最近顶点法向判断，不生成 SDF 体。接触优先 FPS 比例保持 0.5，对应 64 个覆盖点和 64 个接触优先点；接触候选不足时沿用原有回退规则，并不保证所有这些点均在 0.6mm 以内。每步随姿态重新选择顶点。

Teacher 和 agent 使用相同的 128 点密度与采样规则。64 点 teacher checkpoint 会被拒绝，不能直接作为 128 点奖励模型。Agent 评估与推理还明确检查 checkpoint 的输入点数、PPO 宽度和时间上下文，防止点数不改变权重形状时静默误载。历史 checkpoint 缺失相关元数据时仍沿用兼容加载。

## 运行

复用本机的原网格缓存（缓存保存全部顶点，并非固定的 64 点输入）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m occlusion_rl_offline \
  --config configs/reagent_curriculum_large.json train-teacher \
  --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json

.venv/bin/python -m occlusion_rl_offline \
  --config configs/reagent_curriculum_large.json train-agent \
  --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json

.venv/bin/python scripts/evaluate_capacity.py \
  --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json \
  --baseline-teacher runs/data11_contact_fps/teacher/calibration/best.pt \
  --baseline-agent runs/reagent_curriculum/agent/agent.pt
```

新机器先使用同一配置的 `prepare-batch --input-dir ../data --out runs/reagent_curriculum_large/prepared`，再将以上 manifest 改为生成的路径。GPU 可在统一 CLI 子命令前添加 `--device cuda:0`，评估脚本也接受 `--device cuda:0`。本机仅验证 CPU。

扩容配置保留 ranking/calibration 各 32 轮、BC 140 轮、PPO 168 轮。所有权重与结果保存到忽略目录 `runs/reagent_curriculum_large/`；不会覆盖基线产物。`agent.pt` 只按验证集误差选择，`latest.pt` 保存课程最终模型。

## 验证原则

自动测试覆盖新增网络的有限梯度、encoder/actor/critic 更新、保存加载一致性以及错误密度/宽度拒绝加载，并保留原有课程、PPO 恢复与动作行为测试。实际输入另外检查 11 个病例，每例原始位姿和扰动位姿各一组：128 个互不重复的原网格顶点、8 维有限特征、单位顶点法向和对应的原始网格曲率。本轮 11 例原始位姿的平均 0.6mm 内点比例为 59.7%；施加旋转向量 `[3,-2,1]°` 与平移 `[0.5,-0.4,0.3]mm` 后为 50.2%。这些是采样诊断，不是训练或恢复精度。

同起点评估固定种子 426201，010/011 各三档、每档 32 个随机起点，加两条零误差轨迹，全部执行 10 步。比较几何角度和平移误差；不同 teacher 的分差不能直接比较。两项改动同时实施且 teacher 重新训练，因此联合对照不能单独证明点数或 PPO 宽度的因果收益。测试病例此前已有实验使用，属于回归测试，不代表全新外部泛化。

## 本轮 teacher 结果

2026-10-12，CPU / PyTorch 2.8，保持原 7 训练 / 2 验证 / 2 测试病例划分。128 点 teacher 从头完成 ranking 32 轮和 calibration 32 轮，按原规则分别选择 checkpoint。

Ranking 验证最佳为第 1 轮，64 对比较的准确率 81.25%。最后一轮训练准确率 94.64%，验证准确率 73.44%，说明训练集拟合并不等于排序泛化改善。Calibration 选中第 30 轮：

| 验证指标 | 原 64 点 teacher | 新 128 点 teacher |
|---|---:|---:|
| 质量分数 MAE | 0.63861 | 0.59606 |
| 分数与目标相关系数 | 0.85403 | 0.87079 |

MAE 约降低 6.7%。最终第 32 轮 MAE 回升至 0.89550，因此实际使用第 30 轮；不能声称训练已稳定收敛。这里是基于合成位姿质量标签的验证结果，不是临床咬合标签评估。

## 本轮 agent 结果

训练种子 61001，BC 140 轮（11,200 个动作标签）、PPO 168 轮（13,440 个环境动作），与原课程预算一致。Teacher 在 agent 阶段冻结，encoder 可训练。BC 后验证误差为 3.1339° / 1.2808mm；PPO 最佳验证综合误差在第 56 轮，为 2.2533° / 1.2273mm、E=0.29424（原模型验证最佳 E=0.31338）。据此锁定第 56 轮，再运行测试，没有根据测试集更换 checkpoint。

课程最终第 168 轮的验证误差为 2.8644° / 1.4724mm、E=0.36307，明显差于验证最优。第 56 轮 α=0.6724、β=0.3362；第 168 轮 α=β=0。加大网络没有消除退火到纯教师奖励后的性能退化。

测试共 192 条随机起点轨迹 + 2 条原始咬合轨迹，固定种子 426201，每条 10 步；以下为非零起点、病例和档位等权均值：

| 配置（验证选中轮次） | 角度误差 ° | 平移误差 mm | E |
|---|---:|---:|---:|
| 初始姿态 | 4.3929 | 2.0526 | 0.53113 |
| 64 点 / PPO 128（第 42 轮） | 2.9602 | 1.3561 | 0.35452 |
| 128 点 / PPO 512（第 56 轮） | 3.2097 | 1.1308 | 0.34196 |

扩容后平移误差下降 **16.6%**，角度误差增加 **8.4%**，综合误差下降 **3.5%**。因此不能称为全面精度提升；原配置保留为基线，新配置独立提供。两例、单训练种子的结果不足以证明稳定收益。

| 测试档位 | 原模型 ° / mm | 扩容模型 ° / mm |
|---|---:|---:|
| zero | 0.6924 / 0.5113 | 1.5704 / 0.6079 |
| near | 0.8999 / 0.5777 | 1.5275 / 0.6614 |
| medium | 2.6984 / 1.0535 | 2.8306 / 0.8586 |
| wide | 5.2823 / 2.4371 | 5.2711 / 1.8723 |

收益主要来自 medium/wide 的平移恢复；near 及原始咬合的漂移更严重。随机起点的旋转改善比例由 85.9% 降至 66.7%；“教师分数上升但几何 E 恶化”的比例由 9.9% 升至 20.8%。尽管 teacher 的校准 MAE 更好，其分数局部方向仍不足以可靠指导姿态恢复。下一步宜在训练/验证集分别比较仅增加点数、仅增加 PPO 宽度，并测试保留少量末端位姿/BC 约束；本次没有改变奖励或为了修饰结果而加 STOP / 降分回退。

13 项自动测试全部通过。另用 010 施加固定扰动后只传入观测点云进行独立推理：10/10 步执行，其中 3 步教师分数下降，仍照常执行。推理不传真实位姿。新增代码通过编译与 diff 空白检查；CUDA 未在本机实测。

完整本机产物：`runs/reagent_curriculum_large/input_audit.json`、`teacher/{ranking,calibration}/history.json`、`agent/{imitation_history,history,evaluation}.json`、`comparison.json`、`inference_smoke/trajectory.json`。它们和权重均被 Git 忽略；本次仅提交文本代码、配置、测试、文档，不提交 demo、数据或权重。
