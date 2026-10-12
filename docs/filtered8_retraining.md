# 剔除 001、008、009 后重新训练

本轮固定训练病例为 003、002、005、004、007，验证病例为 006，测试病例为 010、011。原始 STL 保留，历史实验结果不覆盖。独立 manifest 位于 `runs/filtered8_surface_diffusion/manifest.json`，其中路径仍引用原始网格缓存；teacher 姿态库和 agent 轨迹均重新建立。

配置：`configs/filtered8_surface_diffusion.json`。

- Teacher 从随机初始化训练，统一 SO(3) 位姿标签，ranking/calibration 各 32 轮。最近三角面距离及插值顶点法向定义符号，不生成体积 SDF。
- 输入沿用 4096 个原始网格候选顶点 → 接触优先 FPS 128 点，顶点法向、网格曲率、三层编码器。
- Agent 使用 diffusion policy，新 teacher 初始化编码器；每个训练病例 64 条专家/随机混合轨迹，每条 10 步，共 3200 个观测样本。训练 2800 次更新，batch 64。这是监督模仿学习，teacher 用于编码器初始化和评分，不是 PPO 奖励。
- 全程执行固定 10 步，不拒绝 teacher 降分动作，不按误差阈值提前停止。仅使用验证病例选择 checkpoint。

复现命令（仓库根目录）：

```bash
.venv/bin/python -m occlusion_rl_offline.cli --config configs/filtered8_surface_diffusion.json train-teacher --manifest runs/filtered8_surface_diffusion/manifest.json
.venv/bin/python -m occlusion_rl_offline.cli --config configs/filtered8_surface_diffusion.json train-agent --manifest runs/filtered8_surface_diffusion/manifest.json
.venv/bin/python scripts/evaluate_filtered_cohort.py --config configs/filtered8_surface_diffusion.json --manifest runs/filtered8_surface_diffusion/manifest.json --stage teacher
.venv/bin/python scripts/evaluate_filtered_cohort.py --config configs/filtered8_surface_diffusion.json --manifest runs/filtered8_surface_diffusion/manifest.json --stage agent
```

新机器需要先准备网格，再按上述病例固定划分写入 manifest；不要直接对 8 例重新随机分组。训练运行中不要修改 manifest 或配置。再次从零训练请使用新的 run_dir 及对应 packaged_teacher/packaged_agent 路径。

Teacher 测试复用上一轮相同的 512 对位姿（seed 242042）；agent 测试为 010/011 各三档 32 个随机起点，加每例一个原始位姿，执行 10 步。测试位姿 seed 426201，diffusion 采样种子 91001、91002、91003 全部报告，不择优。

只有一例验证和两例历史上已检查的测试病例，不能作为充分的临床泛化结论。与旧 diffusion 的比较同时改变了 teacher 训练方式、几何距离和病例集，不能把差异全部归因于剔除病例。和上一轮 surface teacher 的比较保持 teacher 算法和每例样本数不变，但训练总样本数及验证集随病例剔除而改变。

## 本轮实际结果（2026-10-12，CPU）

Teacher 完成 ranking 32 轮和 calibration 32 轮，分别选中第 30、28 轮。训练损失分别从 0.5427 → 0.2248、0.7964 → 0.2394。以下均为固定 010/011 的 512 对测试姿态：

| Teacher | 阶段 | MAE | 局部排序准确率 | 近区局部排序准确率 |
|---|---|---:|---:|---:|
| 原 surface / 7 例训练 | ranking | 1.0033 | 62.35% | 63.74% |
| 新 surface / 5 例训练 | ranking | 1.0567 | 55.29% | 47.25% |
| 原 surface / 7 例训练 | calibration | 0.5149 | 60.39% | 57.14% |
| 新 surface / 5 例训练 | calibration | 0.5207 | 57.25% | 53.85% |

Teacher 训练损失下降，但剔除病例后测试排序能力没有改善，反而下降。不能以训练 loss 下降宣称测试评分更准确。

Agent 完成 2800 次更新，选中第 2800 次更新的 EMA 权重。去噪 MSE 从 0.25535 降至 0.07823；验证综合误差从 0.71577 降至 0.21120，最终验证平均 1.9783° / 0.7005 mm。验证曲线末尾仍在改善，尚无充分收敛证据。

| Agent 测试 | 角度误差 ° | 平移误差 mm | 综合误差 E |
|---|---:|---:|---:|
| 初始姿态 | 4.3929 | 2.0526 | 0.53113 |
| 新模型 / seed 91001 | 2.9760 | 1.0172 | 0.31315 |
| 新模型 / seed 91002 | 2.9699 | 1.0276 | 0.31407 |
| 新模型 / seed 91003 | 2.7669 | 0.9981 | 0.29769 |
| 新模型 / 三种子平均 | 2.9043 | 1.0143 | 0.30831 |
| 原 diffusion / 三种子平均 | 2.9608 | 0.9468 | 0.30340 |

角度误差较旧模型下降约 1.9%，平移误差增加约 7.1%，综合误差增加约 1.6%。测试起点与旧实验逐项核对一致；teacher 分数的绝对增量不跨模型直接比较。新模型在验证病例上的改善没有转化为总体测试改善，保留为独立实验，不替换旧默认模型。

新模型三种子平均分档：near 2.2920° / 0.7694 mm，medium 2.6036° / 0.8798 mm，wide 3.8172° / 1.3937 mm。原始正确姿态执行 10 步后的漂移为 1.6684° / 0.7606 mm（2 病例 × 3 种子），说明近目标动作稳定性仍是问题。后续可增加近目标状态的动作监督、延长训练并按验证选择，但不引入 STOP 或误差阈值终止；扩大独立验证病例后再判断泛化效果。

所有测试共执行 5820 个动作，其中 2525 个使 teacher 降分，均未拒绝。已验证每步变换、预测动作片段的首动作执行，以及新旧测试初始位姿一致。21 项自动回归测试通过；CPU 实训和测试完成，未进行 GPU 实测。

输出目录：`runs/filtered8_surface_diffusion/`。

- `teacher/{ranking,calibration}/best.pt`：新教师。
- `agent/agent.pt`：验证最优 agent；`agent/latest.pt`：可恢复训练状态。
- `test/teacher/comparison.json`、`test/agent/comparison.json`：完整测试指标。
- `test/agent/policy_*_trajectories.json`：三个种子的逐步轨迹。
- `summary.json`：新旧比较、分档结果、逐步平均和动作检查统计。

模型、网格缓存和轨迹保存在忽略目录，未提交 GitHub。
