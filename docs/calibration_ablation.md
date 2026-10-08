# Teacher 分数校准消融（单病例）

这里的“矫正”指 `train-teacher` 的第二阶段 `calibrate_teacher.py`。第一阶段只要求近姿态分数高于远姿态，并使用软锚点限制 logit；第二阶段以已知正确姿态为基准，给随机旋转、平移姿态设置连续目标分数 `-rotation_error_deg/8 - translation_error_mm/4`，再拟合 teacher 的输出。Agent 在两组实验中始终只拿到前后 teacher 分数差作为奖励；真实姿态误差不直接进入 agent 的奖励。

## 设计

- 以下是历史实验：同一附带病例，同一 `256` 个候选点、`64` 个 SDF 加权 FPS 点，2 层编码器和点种子不变。当前配置已调整为 `4096` 个候选点、`1024` 个 FPS 点和 3 层编码器，这些旧指标不可直接用于新配置。
- 比较附带的排序阶段权重 `teacher_ranking.pt` 与后续校准权重 `teacher.pt`。
- Teacher 使用固定的 512 个未参与训练的合成姿态和 512 个近/远配对；指标分别是姿态质量目标相关系数、绝对误差和配对准确率。
- 每种 teacher 各冻结训练一个新 PPO agent。随机种子 42，60 次迭代、8 个并行环境、12 步 horizon、每轮 2 个 PPO epoch；从相应 teacher 编码器初始化。模型选择按 64 个验证扰动的 teacher 分数增益；最终在另 256 个扰动上评估姿态误差。
- 再测试排序 teacher 的 `reward_scale=0.1`，排查分数尺度不同是否是主要原因。

## 结果

| Teacher | 连续姿态目标相关系数 ↑ | 目标 MAE ↓ | 近/远配对准确率 ↑ |
| --- | ---: | ---: | ---: |
| 仅排序 | 0.708 | 6.212 | 98.24% |
| 排序后校准 | 0.978 | 0.208 | 97.46% |

测试扰动的初始平均误差均为 **7.639°、3.720 mm**：

| Agent 奖励来源 | Reward scale | 最终旋转误差 ↓ | 最终平移误差 ↓ | Teacher 分数变化 | 严格成功率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 仅排序 | 1.0 | 15.335° | 5.651 mm | -8.730 → -6.396 | 0/256 |
| 仅排序 | 0.1 | 12.163° | 6.851 mm | -8.730 → -5.600 | 0/256 |
| 排序后校准 | 1.0 | 3.245° | 3.472 mm | -1.758 → -1.137 | 0/256 |

排序 teacher 虽然在训练目标的近/远配对上略高，但 agent 可以靠远离原始姿态来提高它的分数；校准后，分数与姿态质量相关性明显增强，agent 的旋转误差随之下降。缩小排序分数的奖励尺度没有消除该现象，表明问题不只是 logit 数值范围。对当前单病例、当前 teacher 和短 PPO 预算，**校准阶段对得到有用的奖励方向很重要**。它仍未让 agent 在此预算内达到严格恢复阈值，也不能证明所有病例或更长训练中都必须采用这种基于姿态误差的校准。

这个校准将原始正确姿态作为监督信息传入 teacher。若要求整个训练链都不能使用姿态误差标签，需要换成只依赖几何/接触的 teacher 训练方案，并重新做对照。

## 复现

从仓库根目录运行，示例权重和数据已包含在仓库中：

```bash
python scripts/calibration_ablation.py
occlusion-rl --config configs/calibration_ablation.json train-agent --teacher examples/checkpoints/teacher_ranking.pt --out runs/calibration_ablation/ranking_agent
occlusion-rl --config configs/calibration_ablation.json train-agent --teacher examples/checkpoints/teacher.pt --out runs/calibration_ablation/calibrated_agent
occlusion-rl --config configs/calibration_ablation_rank_scaled.json train-agent --teacher examples/checkpoints/teacher_ranking.pt --out runs/calibration_ablation/ranking_scaled_agent
occlusion-rl --config configs/calibration_ablation.json evaluate-agent --teacher examples/checkpoints/teacher_ranking.pt --agent runs/calibration_ablation/ranking_agent/agent.pt --out runs/calibration_ablation/ranking_test
occlusion-rl --config configs/calibration_ablation.json evaluate-agent --teacher examples/checkpoints/teacher.pt --agent runs/calibration_ablation/calibrated_agent/agent.pt --out runs/calibration_ablation/calibrated_test
occlusion-rl --config configs/calibration_ablation_rank_scaled.json evaluate-agent --teacher examples/checkpoints/teacher_ranking.pt --agent runs/calibration_ablation/ranking_scaled_agent/agent.pt --out runs/calibration_ablation/ranking_scaled_test
```

原始运行输出与检查点在本地 `runs/calibration_ablation/`，该目录按仓库 `.gitignore` 不提交。
