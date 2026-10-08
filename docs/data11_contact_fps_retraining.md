# 接触区优先 FPS：teacher 与 agent 重训

使用 `configs/data11_contact_fps.json` 和原有 11 例清单重新训练两网络。训练/验证/测试划分保持不变：训练 003、002、009、005、008、004、007；验证 006、001；测试 010、011。每例固定从原上颌网格均匀抽取 4096 个候选顶点，每个姿态选 64 点：32 点做全颌距离加权 FPS，另外 32 点从距下颌 0.6 mm 内的候选点做距离加权 FPS；接触候选不足时扩展到最近的至少 128 点。两网络均使用该新采样，旧权重未混用。下颌为非封闭网格，距离符号由原下颌最近顶点法向决定。

teacher 的 ranking 和 calibration 各训练 32 轮。ranking 验证集最佳准确率为 **55/64（85.9%，第 18 轮）**，旧采样为 49/64（76.6%）；calibration 最佳 MAE 为 **0.639（第 30 轮）**，旧采样为 0.652。测试病例采用相同随机种子、每例 96 个排序对、128 个校准姿态和 64 个局部修正姿态：

| 测试例 | 采样 | 排序准确率 | 校准 Spearman | 校准 MAE | 近姿态朝原始方向得分提高 | 原始姿态高于近邻姿态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 010 | 旧 | 81.2% | 0.822 | 0.631 | 54.7% | 9.4% |
| 010 | 新 | 85.4% | 0.859 | 0.599 | 67.2% | 96.9% |
| 011 | 旧 | 83.3% | 0.830 | 0.744 | 51.6% | 93.8% |
| 011 | 新 | 79.2% | 0.857 | 0.543 | 68.8% | 39.1% |

Agent 使用新 calibration teacher 的分数差作为唯一奖励，PPO 不拒绝降分动作；84 轮训练、每轮 6 个独立随机起点、每条 10 步。按验证集 teacher 分数增益选择第 56 轮权重。在 010、011 共 64 个宽范围测试起点上，新 agent 的平均旋转误差为 **8.062°→6.698°**，平移误差为 **3.820→3.875 mm**，teacher 分数为 **-2.052→-1.394**，严格 1.5°/0.5 mm 成功率仍为 **0/64**。相同起点的旧 agent 为 8.062°→8.018°、3.820→3.817 mm、成功 0/64。新采样改善了宽范围角度修正，但未改善平移或严格恢复。

独立种子 `20261003` 下，每个测试病例、每档 32 个随机起点：

| 病例 | 初始扰动范围 | 新 agent 旋转误差 | 新 agent 平移误差 | 最终严格成功 |
| --- | --- | --- | --- | ---: |
| 010 | ±1.5° / ±0.5 mm | 1.392°→4.397° | 0.485→0.966 mm | 0/32 |
| 010 | ±4° / ±2 mm | 4.012°→5.366° | 2.063→1.950 mm | 0/32 |
| 010 | ±8° / ±4 mm | 7.709°→6.981° | 3.500→3.709 mm | 0/32 |
| 011 | ±1.5° / ±0.5 mm | 1.526°→4.039° | 0.456→0.837 mm | 0/32 |
| 011 | ±4° / ±2 mm | 3.725°→4.716° | 1.999→2.201 mm | 0/32 |
| 011 | ±8° / ±4 mm | 7.430°→6.809° | 4.051→3.929 mm | 0/32 |

近目标区 64 条轨迹均未改善旋转误差，原本达到严格阈值的 17 条在 10 步后全部失去达标状态。teacher 的近邻分数方向仍不稳定，尤其 011 的原始姿态局部最高分比例仅 39.1%。当前模型适合继续研究远姿态粗调，不能作为可靠的咬合精调模型。下一轮应优先改善 teacher 的局部排序和原始姿态峰值，再调整 agent 的近目标动作尺度并用同一测试协议复测。

复现命令：

```bash
.venv/bin/python -m occlusion_rl_offline.cli --config configs/data11_contact_fps.json train-teacher --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json
.venv/bin/python -m occlusion_rl_offline.cli --config configs/data11_contact_fps.json train-agent --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json --teacher runs/data11_contact_fps/teacher/calibration/best.pt --out runs/data11_contact_fps/agent
.venv/bin/python -m occlusion_rl_offline.cli --config configs/data11_contact_fps.json evaluate-agent --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json --teacher runs/data11_contact_fps/teacher/calibration/best.pt --agent runs/data11_contact_fps/agent/agent.pt --out runs/data11_contact_fps/agent/test
.venv/bin/python scripts/evaluate_agent_pose_bands.py --config configs/data11_contact_fps.json --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json --teacher runs/data11_contact_fps/teacher/calibration/best.pt --agent runs/data11_contact_fps/agent/agent.pt --out runs/data11_contact_fps/agent/pose_band_test.json --episodes 32 --seed 20261003
```

完整指标在 `runs/data11_contact_fps/teacher/usability_010.json`、`usability_011.json`、`agent/test/test_evaluation.json` 和 `agent/pose_band_test.json`。运行产物位于 `runs/data11_contact_fps/`，默认被 `.gitignore` 忽略。
