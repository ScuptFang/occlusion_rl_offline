# data 11 例 teacher 扩量训练

使用 `../data/001`–`011` 的全部成对 STL，重新执行当前的网格预处理。上颌保留原始合并网格顶点、顶点法向和 cotangent 网格曲率；下颌保留顶点及顶点法向。每例固定均匀选 4096 个上颌候选点，姿态改变后计算到原下颌顶点的带符号最近点距离，再做距离加权 FPS 取 64 点。ranking 使用四档姿态、六类有序比较；calibration 使用合成的角度与平移质量目标。

固定 seed 42，按病例分组：训练 `003,002,009,005,008,004,007`，验证 `006,001`，测试 `010,011`。完整路径记录在 `runs/data11_candidate4096_fps64_more/prepared/manifest.json`，同一病例不会跨组。配置为 `configs/data11_candidate4096_fps64_more.json`，所有实验输出在 `runs/data11_candidate4096_fps64_more/`，未覆盖单病例权重。

本次 ranking 从头训练 32 轮，每轮每训练病例 24 对，共 168 对/轮、5376 对训练样本；每验证病例固定 32 对。最佳权重为第 32 轮，验证排序 49/64（76.6%），验证损失 0.582；末轮训练排序 149/168（88.7%）。calibration 从该 ranking 权重继续训练 32 轮，每轮 256 姿态，共 8192 个训练姿态，固定验证 128 姿态。最佳权重为第 29 轮，验证 MAE 0.652，Pearson 相关 0.853。后期相关性约 0.85，但 MAE 仍随预测整体偏移波动；模型没有完全稳定收敛。

独立测试使用 `scripts/evaluate_teacher_usability.py`，随机种子 20261002。每个测试病例取 96 对全新四档排序姿态、128 个全新校准姿态、64 个接近原始姿态的扰动，以及各 64 个较大与接近目标的朝目标/背离目标步骤。朝目标步骤将当前姿态扰动缩小到 0.8 倍，背离目标放大到 1.2 倍。

| 测试病例 | ranking 排序 | calibration 排序 | calibration 质量 Spearman / MAE | 原始分数高于近邻扰动 | 较大扰动：朝目标高于背离目标 / 当前 | 近目标：朝目标高于当前 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 010 | 79/96 | 78/96 | 0.822 / 0.631 | 6/64 | 61/64 / 51/64 | 35/64 |
| 011 | 85/96 | 80/96 | 0.830 / 0.744 | 60/64 | 61/64 / 51/64 | 33/64 |

两例合计，calibration 在四档排序中正确 158/192（82.3%），在较大扰动时“朝目标高于背离目标”正确 122/128（95.3%）。但近目标“朝目标高于当前”仅 68/128（53.1%），与随机方向接近；010 的正确原始姿态尤其常被近邻扰动超过。这套 teacher 可用于粗姿态判别，尚不能作为要求恢复到 1.5°、0.5 mm 以内的 agent 的可靠唯一奖励。这里的测试病例与训练/选权重病例完全分离，但仅有 2 例，且原 STL 的临床咬合姿态未独立复核。

可复现命令（仓库根目录）：

```bash
.venv/bin/python -m occlusion_rl_offline.cli --config configs/data11_candidate4096_fps64_more.json prepare-batch --input-dir ../data --out runs/data11_candidate4096_fps64_more/prepared
.venv/bin/python -m occlusion_rl_offline.cli --config configs/data11_candidate4096_fps64_more.json train-teacher --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json
.venv/bin/python scripts/evaluate_teacher_usability.py --config configs/data11_candidate4096_fps64_more.json --case runs/data11_candidate4096_fps64_more/prepared/cases/010/case.npz --out runs/data11_candidate4096_fps64_more/teacher/usability_010.json
.venv/bin/python scripts/evaluate_teacher_usability.py --config configs/data11_candidate4096_fps64_more.json --case runs/data11_candidate4096_fps64_more/prepared/cases/011/case.npz --out runs/data11_candidate4096_fps64_more/teacher/usability_011.json
```

最终权重在 `runs/data11_candidate4096_fps64_more/teacher/ranking/best.pt` 和 `runs/data11_candidate4096_fps64_more/teacher/calibration/best.pt`，逐轮训练日志及逐例完整指标在同一 `teacher/` 目录。
