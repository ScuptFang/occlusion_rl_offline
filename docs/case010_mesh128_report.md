# 010 单病例：全网格 128 点训练与演示

使用 `../data/010/upper.stl` 与 `lower.stl`，按 `configs/case010_mesh128.json` 从头训练 teacher 和 PPO agent。上颌保留 138,093 个原网格顶点、顶点法向及网格曲率；下颌保留 111,189 个原网格顶点和顶点法向。每次姿态变化后先在全部上颌顶点上计算最近下颌顶点的带符号距离，再直接执行距离加权 FPS，选出 128 个 8 通道点；不存在候选点预抽样。PPO 执行降分动作，以 teacher 分数差作为唯一奖励。

本次使用计算量受限的独立配置：teacher 排序 8 epoch、校准 8 epoch；PPO 40 次迭代、每回合 8 步。排序验证最佳配对准确率为 0.9375（每轮 16 对）；校准验证最佳 MAE 1.1262、相关系数 0.7038。agent 依据验证 teacher 分数增益选取第 15 次迭代权重。

010 同病例、不同随机种子的 32 次测试：初始/最终平均旋转误差 **8.53°/13.30°**，平移误差 **3.74/3.73 mm**，teacher logit **−1.949/−1.928**，同时达到 1.5° 和 0.5 mm 的轨迹 **0/32**。这份模型未学会恢复原始姿态；teacher 分数的轻微上升不对应角度改善。单病例测试不衡量跨病例泛化。

`demo/index.html` 与 `demo/data.js` 包含 010 的 3 条真实 8 步轨迹。页面展示当前上颌、下颌、原始上颌目标、每步角度和平移误差、teacher 分数，以及实际输入网络的 128 个顶点。三条轨迹的角度误差均增大；轨迹未筛选或修饰。重新生成：

```bash
.venv/bin/python demo/generate.py
```

权重和评估记录保存在 Git 忽略的 `runs/case010_mesh128_20260930/`，主要文件为 `teacher/calibration/best.pt`、`agent/agent.pt`、`agent/evaluation.json`、`test/test_evaluation.json`。
