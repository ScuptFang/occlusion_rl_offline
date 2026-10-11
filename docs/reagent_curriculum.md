# 模仿学习与位姿/教师奖励课程

本流程借鉴 [ReAgent 论文](https://openaccess.thecvf.com/content/CVPR2021/html/Bauer_ReAgent_Point_Cloud_Registration_Using_Imitation_and_Reinforcement_Learning_CVPR_2021_paper.html)及[官方训练实现](https://github.com/dornik/reagent/blob/master/registration/train.py)。ReAgent 从专家动作学习初始策略，再结合 BC 与 PPO。这里沿用项目的连续六维动作；以真实位姿误差奖励逐步让位给咬合教师奖励，是本项目新增设计，并非复现原论文的奖励公式或离散动作分类。

## 训练过程

1. 冻结现有咬合 teacher，复制其 encoder 初始化 agent。输入继续是原上颌网格顶点的 XYZ、顶点法向、网格曲率、下颌最近顶点法向决定符号的距离；4096 候选顶点→接触优先加权 FPS 64 点、3 层 encoder。teacher 与 agent 规则相同。
2. BC 预训练默认 140 轮，每轮 8 条 10 步轨迹，3 次优化遍历。使用训练病例，以 50% 概率执行专家动作，其余执行当前策略的确定性动作。在每个访问状态重新计算专家标签；不会把真实姿态作为网络输入。BC 使用 `MSE(tanh(policy_mean), expert_action)`。
3. PPO 默认 168 轮，全部动作来自当前策略采样；专家动作只作为辅助 BC 标签，不替换 PPO 动作，因此 PPO 概率比对应实际执行的策略。每轮奖励权重固定，轮间逐渐切换。
4. 最前 10% PPO 轮次使用纯位姿奖励；10%–80% 线性切换；最后 20% 为纯教师奖励，BC 系数也降为零。完整跑完课程，不因验证平台期提前退出。
5. 仅用验证集选择 `agent.pt`；保存最终 `latest.pt`、单独的 `imitation.pt`。测试集只用于最后对比。推理无需真实位姿、专家或课程模块，仍兼容原有 `infer-agent`。

环境以原上颌顶点均值为旋转中心，更新为 `R_next = ΔR @ R`、`t_next = t + Δt`。专家取 `rotvec(R^-1)` 与 `-t`。旋转整体缩放到允许的单步范围，保持旋转轴，避免逐轴裁剪改变纠正方向；平移逐轴限制到动作范围。微小误差可用微小动作纠正。没有 STOP 动作、精度阈值终止或拒绝教师降分动作；零误差状态的专家标签为零，但轨迹仍执行全部步数。

## 奖励与配置

令 `E = 0.5 × (角度误差/8° + 平移误差/4mm)`，默认：

```text
r_gt      = gt_reward_scale × (E_before − E_after)
r_teacher = reward_scale × (score_after − score_before)
r         = α × r_gt + (1 − α) × r_teacher
loss      = PPO_loss + value_coef × value_loss − entropy_coef × entropy
            + β × BC_loss
```

负向改善保留为负奖励，不裁零。奖励使用 teacher 的原始分数差。两个奖励的量级可能不同，不能仅把 α 理解为实际梯度贡献占比；日志同时记录两项有符号均值与绝对值均值，以检查量级。γ=1 时，单条固定 α 轨迹上的奖励差可望远镜求和。

所有新增超参位于 `agent.imitation`：

| 参数 | 默认值 | 用途 |
|---|---:|---|
| iterations / epochs / batch_size | 140 / 3 / 40 | BC 轮数、遍历数、批量 |
| learning_rate / encoder_learning_rate | 3e-4 / 3e-5 | BC 策略与 encoder 学习率 |
| expert_rollout_probability | 0.5 | BC 收集阶段执行专家动作的概率 |
| expert_action_limit | 0.999 | 专家动作上限，避免 tanh 边界不可达 |
| gt_reward_scale | 2.0 | 位姿误差改善缩放 |
| gt_weight_start / gt_weight_end | 1.0 / 0.0 | α 的起点与终点 |
| decay_start / decay_end | 0.1 / 0.8 | 按 PPO 总轮数的切换进度 |
| bc_weight_start / bc_weight_end | 0.5 / 0.0 | PPO 阶段 β 的起点与终点 |

`agent.error_scales` 为角度/平移归一化尺度；`agent.reward_scale` 为教师奖励缩放；原有随机位姿分布、探索方差、动作幅度、horizon、PPO 超参仍有效。新配置默认不冻结 agent encoder；teacher 始终冻结。要保留最终弱位姿约束，可把 `gt_weight_end` 配为非零值，但那就不是纯教师收尾。

## 运行

CPU / CUDA 环境安装见 [GPU 文档](gpu_training.md)。新配置默认 `runtime.device=auto`；统一 CLI 的 `--device cuda:0` 必须放在子命令之前。本机测试为 CPU，CUDA 数值与性能尚未实测。

已有病例缓存及本项目现有 teacher：

```bash
occlusion-rl --config configs/reagent_curriculum.json train-agent \
  --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json \
  --teacher runs/data11_contact_fps/teacher/calibration/best.pt \
  --out runs/reagent_curriculum/agent
```

新机器从外部 STL 开始，不能依赖 Git 中的权重：

```bash
occlusion-rl --config configs/reagent_curriculum.json prepare-batch \
  --input-dir ../data --out runs/reagent_curriculum/prepared
occlusion-rl --config configs/reagent_curriculum.json --device cuda:0 train-teacher \
  --manifest runs/reagent_curriculum/prepared/manifest.json
occlusion-rl --config configs/reagent_curriculum.json --device cuda:0 train-agent \
  --manifest runs/reagent_curriculum/prepared/manifest.json
occlusion-rl --config configs/reagent_curriculum.json --device cuda:0 evaluate-agent \
  --manifest runs/reagent_curriculum/prepared/manifest.json
```

续训在同一训练命令后增加 `--checkpoint runs/reagent_curriculum/agent/latest.pt`。PPO 阶段恢复模型、优化器、随机状态和课程位置，跳过 BC。必须保留原 `agent` 配置和原总轮数；修改总轮数会改变退火进度，因此本模式明确拒绝这种恢复。BC 阶段尚无中途 checkpoint，若在 BC 中断，需要重新开始 BC。跨设备不保证逐位复现。保持原病例划分；更换 teacher 或采样配置需重新训练。

查看 `imitation_history.json`、`imitation_evaluation.json`、`history.json`、`evaluation.json`；模型和完整轨迹都留在忽略的 `runs/`。`agent.pt` 按验证误差选择，可能来自课程中段；`latest.pt` 才一定经历完整教师奖励收尾，不应混淆二者。

## 测试

执行 `python -m unittest discover -s tests -v`。新增测试验证专家在实际环境变换约定下单调纠错并接近恒等姿态、奖励正负与权重边界、退火单调性、BC 有限梯度，以及小型真实网络 BC→PPO→中断恢复→推理。CPU 中断恢复与不中断训练的参数及历史逐项完全相同；修改课程总预算会被拒绝。还覆盖 encoder 冻结到解冻的 PPO 路径。实际 010 病例独立推理也已运行：人为施加 `[3,-2,1]°` 旋转向量和 `[0.5,-0.4,0.3]mm` 平移后，只传入扰动点云，10/10 个动作均执行，其中 7 个动作降低教师分数，仍未被拒绝。此检查仅验证推理链路，不用于选模型。

实际病例训练与对照结果见下方实验记录。历史上 010/011 已被用于分析，010 还曾用于其他单病例训练，因此此处只能视为按当前划分留出的回归实验，不能宣称是完全未触及的外部泛化测试。

固定起点评估命令（先完成所有训练，再执行一次）：

```bash
python scripts/evaluate_imitation_curriculum.py \
  --manifest runs/data11_candidate4096_fps64_more/prepared/manifest.json \
  --teacher runs/data11_contact_fps/teacher/calibration/best.pt \
  --baseline runs/reagent_curriculum/teacher_only/agent.pt \
  --episodes 32 --seed 426201
```

纯教师对照配置由新配置复制：`agent.trainer="continuous"`，删除 `agent.imitation`，设 `agent.early_stopping_checks=100000`，其余 agent 参数不变；用同一个 manifest、teacher 和 `--out runs/reagent_curriculum/teacher_only` 训练。比较使用同一 PPO 轮数；新流程还额外包含 BC 交互，不是等总计算量的消融。

## 实际病例实验：2026-10-12

CPU / PyTorch 2.8.0，训练种子 61001。保持既有病例划分：训练 003、002、009、005、008、004、007；验证 006、001；测试 010、011。使用现有冻结 teacher `runs/data11_contact_fps/teacher/calibration/best.pt`，SHA256 为 `90a6342da7bc3da7bb7a6e1721ed5e928f1ea2f200481cccc2d177a55c957258`，本轮没有重新训练 teacher。

新流程完成 140 轮 BC（11,200 个环境动作）和 168 轮 PPO（13,440 个动作）；纯教师对照完成 168 轮 PPO。两组使用相同的采样、随机初始位姿分布、网络和 PPO 参数，均从第 1 轮更新 agent encoder；对照不是历史上冻结 encoder 28 轮的旧 checkpoint。新方法有额外 BC 预算，不能把全部收益归因于奖励混合。

验证种子 426101，每个非零档位每例 8 个初始姿态，外加原始位姿。新模型在第 42 轮取得最低验证 E=0.31338（2.3963° / 1.3088mm），当时 α=0.792、β=0.396；它作为 `agent.pt` 锁定。纯教师对照选中第 98 轮，验证 E=0.47780。未用测试集选轮次。

测试种子 426201；010、011 各有 near、medium、wide 三档，每档 32 个随机位姿，合计 **192 条非零轨迹 + 2 条零误差轨迹**，每条完整执行 10 步。下表为非零档位的病例/档位等权均值：

| 阶段 | 角度误差 ° | 平移误差 mm | E | 教师分数提升 |
|---|---:|---:|---:|---:|
| 初始姿态 | 4.3929 | 2.0526 | 0.53113 | — |
| 纯教师 PPO，验证最优第 98 轮 | 3.9641 | 2.2294 | 0.52644 | 0.5260 |
| 仅 BC 预训练完成 | 3.0937 | 1.4786 | 0.37818 | 0.6738 |
| 新课程，验证最优第 42 轮 | **2.9602** | 1.3561 | 0.35452 | 0.7960 |
| 新课程，最终第 168 轮，α=β=0 | 3.0105 | **1.3104** | **0.35195** | 0.8456 |

新课程验证最优模型相对纯教师对照，平均角度误差降低约 25.3%，平移降低约 39.2%，E 降低约 32.7%。相对仅 BC，E 进一步降低约 6.3%。最终模型在这批测试起点的 E 略低，但仍保留验证集选中的第 42 轮为默认 `agent.pt`，不根据测试结果更换选择。

验证最优模型的分档结果：

| 档位 | 初始 ° / mm | 恢复后 ° / mm |
|---|---:|---:|
| 原始咬合（2 条） | 0 / 0 | 0.6924 / 0.5113 |
| near（64 条） | 1.5448 / 0.4957 | 0.8999 / 0.5777 |
| medium（64 条） | 3.8131 / 1.9042 | 2.6984 / 1.0535 |
| wide（64 条） | 7.8209 / 3.7578 | 5.2823 / 2.4371 |

**改善有效，但仍有近正确姿态漂移。** near 档的平均平移比初始值略差；零误差输入也会移动。最终课程模型的零误差漂移为 1.2522° / 0.6268mm，较验证最优模型更大。新课程最优模型有 85.9% 的随机起点旋转误差下降、75.5% 平移误差下降，仍有 9.9% 的轨迹出现“教师分数上升而 E 变差”；纯教师对照对应为 42.2%。不能把教师分数提升直接等同于几何姿态恢复。

退火过程并非单调改善：验证平移在第 56 轮为 1.2607mm，第 126 轮升到 1.8264mm，最终回到 1.2851mm；第 140/154 轮纯教师阶段也有较大误差。这表明需保留验证选模，不能假设课程末端必然最好。BC 在不同随机状态上的逐轮损失有波动，前/后 14 轮均值约 0.2090/0.1892；本轮没有证明 BC 或 PPO 已充分收敛。

第 28 轮记录的位姿/教师奖励绝对均值分别约 0.0762/0.5550，说明两个信号量级明显不同。下一轮应只在训练/验证集上比较 `gt_reward_scale`、退火速度、非零 `gt_weight_end` 与轻量残余 `bc_weight_end`，并加入更多近零状态；保持完整动作预算，不用强制 STOP 掩盖漂移。随后至少复验 3 个训练种子和更长的 20/40 步轨迹。当前结论限于单种子、两例历史上已分析过的测试病例，不能证明外部病例泛化或长期稳定性。

完整结果保留在本机忽略目录 `runs/reagent_curriculum/comparison.json`、`agent/history.json`、`teacher_only/history.json`。本次 Git 新增内容仅代码、配置、测试和文字报告，不新增 demo、病例数据、模型权重或完整轨迹；继承的 Git 历史不重写。12 项自动测试全部通过，另完成实际病例 140+168 轮训练、168 轮对照与上述独立评估。
