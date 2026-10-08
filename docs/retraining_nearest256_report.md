# 最近顶点距离与 256/64 点重训结果

使用 `../data/001`–`011` 的 11 对 STL，从原始网格重新准备病例。上颌保留原始顶点、顶点法向和三角网格曲率；下颌保留原始顶点及顶点法向。网络输入的距离是最近下颌顶点的欧氏距离，以该顶点法向决定正负，不生成体素 SDF。teacher 与 agent 均使用 256 个候选点、64 个带符号距离加权 FPS 点、3 层编码器。

本次使用 `configs/public_cohort_optimized.json`，其中 `agent.score_gated_actions=false`。PPO 执行分数降低的动作，并直接获得负的教师分数差奖励。训练目录为 `runs/retrain_nearest256_20260930/`，已被 Git 忽略。病例划分与上次一致：003、002、009、005、008、004、007 训练；006、001 验证；010、011 测试。

teacher 从随机初始化训练：排序阶段 40 epoch，最佳验证配对准确率 0.78125（epoch 39）；校准阶段 40 epoch，最佳验证 MAE 0.5810、相关系数 0.8543（epoch 30）。PPO 训练 200 次迭代，按验证分数增益选择 iteration 150 的权重。其验证集 48 次扰动中 0 次同时达到 1.5° 和 0.5 mm；初始/最终平均旋转误差 8.01°/6.94°，平移误差 4.10/6.46 mm。

独立测试病例 010、011 各 64 次，共 128 次扰动：初始/最终平均旋转误差 **7.85°/9.28°**，平移误差 **3.80/6.92 mm**，teacher logit **−1.721/−1.501**，同时满足 1.5° 和 0.5 mm 的成功率 **0/128**。当前 teacher 分数上升并未对应准确姿态恢复；不能把该权重视为合格的恢复模型。

重现命令：

```bash
.venv/bin/python -m occlusion_rl_offline.cli --config configs/public_cohort_optimized.json prepare-batch --input-dir ../data --out runs/retrain_nearest256_20260930/prepared
.venv/bin/python -m occlusion_rl_offline.cli --config configs/public_cohort_optimized.json train-teacher --manifest runs/retrain_nearest256_20260930/prepared/manifest.json --out runs/retrain_nearest256_20260930/teacher
.venv/bin/python -m occlusion_rl_offline.cli --config configs/public_cohort_optimized.json train-agent --manifest runs/retrain_nearest256_20260930/prepared/manifest.json --teacher runs/retrain_nearest256_20260930/teacher/calibration/best.pt --out runs/retrain_nearest256_20260930/agent
.venv/bin/python -m occlusion_rl_offline.cli --config configs/public_cohort_optimized.json evaluate-agent --manifest runs/retrain_nearest256_20260930/prepared/manifest.json --teacher runs/retrain_nearest256_20260930/teacher/calibration/best.pt --agent runs/retrain_nearest256_20260930/agent/agent.pt --out runs/retrain_nearest256_20260930/test
```

指标原件见 `teacher/calibration/validation.json`、`agent/evaluation.json` 和 `test/test_evaluation.json`。历史 0.6 mm 网格实验采用不同的输入规则，结果不可直接作为同配置消融比较。
