# 010 单病例训练与演示

只使用 `../data/010/upper.stl`、`lower.stl`，按当前最近下颌原始顶点距离、256 候选点、64 加权 FPS 点、3 层编码器规则重新准备 `runs/case010_only_20260930/case.npz`。teacher 从随机初始化训练排序 40 epoch、校准 40 epoch；最佳校准验证 MAE 为 0.3608、相关系数为 0.9485。PPO 从该 teacher 初始化策略编码器，训练 200 次迭代，**执行所有有效动作，包括降低教师分数的动作**。

同一病例、不同随机扰动种子的 64 次测试中，旋转误差均值从 8.00° 降至 5.01°，平移误差从 3.85 mm 增至 5.80 mm；teacher logit 从 −1.885 升至 −0.870，同时达到 1.5° 和 0.5 mm 的轨迹为 **0/64**。单病例实验不能衡量跨病例泛化，分数上涨也不能证明姿态恢复。

`demo/index.html` 和 `demo/data.js` 使用选中的 agent 权重生成 010 的 3 条真实随机轨迹，每条 16 步。每步展示上颌姿态、教师分数、相对原始姿态的角度和平移误差、256 个候选点中的 64 个实际网络输入点。轨迹中降分动作也被执行。重新生成命令：

```bash
.venv/bin/python demo/generate.py
```

训练产物位于 Git 忽略的 `runs/case010_only_20260930/`，其中 `teacher/calibration/best.pt`、`agent/agent.pt` 是本次选中权重，`agent/evaluation.json` 和 `test/test_evaluation.json` 保存评估结果。
