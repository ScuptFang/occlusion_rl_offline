# 未校准 teacher 的 agent 实验

从最近的 `region_point_transformer` 实验读取 ranking 最佳 checkpoint，跳过 calibration 阶段，重新训练 diffusion 模仿学习和 diffusion RL。原实验的模型及结果保留。

这里的“未校准”不是完全取消姿态监督：目前 ranking 损失仍包括质量回归、排序、分数差三项，权重为 `0.1 / 1 / 1`；本实验只取消第二阶段 `1 / 0.5 / 1` 的继续训练。

## 对照设置

- 训练：003、002、005、004、007；验证：006；测试：010、011。继续排除 001、008、009。
- 冻结相同的 PointTransformer 区域分割器，均匀候选 4096 点，再按接触距离加权 FPS 取 128 点。顶点法向、网格曲率及最近三角面距离保持相同。
- 使用各自 teacher 的编码器初始化 agent，模仿学习 2800 次更新，再进行 50 轮 diffusion RL，每轮 8 个环境、10 步，共 4000 次交互。
- 模仿学习使用真实姿态生成专家动作；强化学习奖励只有 `2 × (score_next - score_previous)`。不拒绝降分动作，不使用 STOP 或按精度提前停止。
- 测试起点种子 426201，推理种子 91001、91002、91003。每个种子共 194 条轨迹：两个病例，每例 near/medium/wide 各 32 个起点，以及一个原始正确位姿。
- 额外评估未强化学习、但使用同样随机去噪尾部的模仿基线，区分 RL 更新与推理方式变化的影响。
- checkpoint 只按验证集选择；综合误差 `E = 0.5 × (角度/8° + 平移/4 mm)`，主表不含零位姿，零位姿用于检查漂移。

这是完整训练栈对照，同时改变 teacher 奖励和初始编码器，并非只替换奖励的因果消融。仅一个训练种子、两个历史测试病例，三个推理种子不能替代多次独立训练。

## 复现

先完成 `docs/region_point_transformer.md` 中的区域实验，保留其本地模型及评估结果。在仓库根目录执行：

```bash
.venv/bin/python -m pip install -e ".[report]"
.venv/bin/python scripts/run_ranking_agent_ablation.py
```

入口配置：`configs/region_ranking_agent.json`。脚本从原实验配置继承全部超参，只更换 teacher 和输出路径，生成两份完整配置：

- `runs/region_ranking_agent/teacher_agent_config.json`
- `runs/region_ranking_agent/diffusion_rl_config.json`

所有模型、专家样本、轨迹和日志都位于已忽略的 `runs/region_ranking_agent/`。新实验请修改 `out` 使用不同目录；脚本拒绝覆盖已有训练。若训练阶段已经完成，可继续执行后续阶段：

```bash
.venv/bin/python scripts/run_ranking_agent_ablation.py --stages rl evaluate report
# 或仅重新汇总已有测试
.venv/bin/python scripts/run_ranking_agent_ablation.py --stages report
```

训练中断后的恢复由原训练模块的 `--resume` 接口完成；本脚本不会自动覆盖或恢复部分训练。

完成后的本地结果见 `runs/region_ranking_agent/report.md` 和 `summary.json`；后者包括分档误差、零位姿漂移、teacher 奖励方向与真实误差改善的一致率、降分动作执行数及 checkpoint 审计。

## 本次实际结果

使用区域 teacher 的 ranking 第 29 epoch checkpoint，SHA256 为 `a10fc12f1cc21cea14526fa495f16f06c2ddfbd0d346cca4f5b00645e9968286`。模仿学习完成 2800 次更新，噪声 MSE 从 0.2537 降至 0.0765；验证选中第 2800 次更新。RL 完成 50 轮，验证选择第 30 轮，验证误差 2.1722° / 0.5821 mm。第 40 轮曾退化，最后一轮并非最佳模型。

在相同测试起点、相同三个推理种子下：

| Teacher | Agent | 旋转误差 ° ↓ | 平移误差 mm ↓ | E ↓ |
|---|---|---:|---:|---:|
| calibration | 模仿学习 | 3.6938 | 0.8618 | 0.33859 |
| calibration | 随机推理模仿基线 | 3.7390 | 0.8676 | 0.34214 |
| calibration | diffusion RL | 3.6216 | 0.8475 | 0.33229 |
| ranking | 模仿学习 | 3.4601 | 0.8474 | 0.32218 |
| ranking | 随机推理模仿基线 | 3.4559 | 0.8507 | 0.32234 |
| ranking | diffusion RL | **3.3257** | **0.8345** | **0.31217** |

跳过 calibration 后的完整训练栈，相比 calibration RL：旋转误差降低 8.17%，平移降低 1.54%，综合误差降低 6.06%。ranking RL 相对其自身随机推理模仿基线，综合误差降低 3.15%。模仿阶段已经产生差异，不能把全部收益归因于 ranking 奖励更准确。

| 测试病例 | calibration RL 角度 / 平移 | ranking RL 角度 / 平移 |
|---|---:|---:|
| 010 | 5.0731° / 1.1701 mm | 4.1256° / 1.1716 mm |
| 011 | 2.1701° / 0.5250 mm | 2.5259° / 0.4973 mm |

整体收益主要来自 010 的旋转改善；011 的旋转反而变差。因此不能据此认为未校准 teacher 对所有病例都更好。

ranking RL 的 near / medium / wide 平均误差分别为 2.4753° / 0.6308 mm、2.9918° / 0.7320 mm、4.5100° / 1.1407 mm。原本正确的零位姿在执行完整 10 步后仍有 **2.7241° / 0.6820 mm** 漂移，虽低于 calibration RL 的 2.9198° / 0.7361 mm，但尚未解决保持正确位姿的问题。

在各自策略轨迹上，ranking teacher 的逐步奖励方向与真实综合误差改善的一致率为 **54.28%**，calibration 为 **56.03%**；ranking 正奖励动作中仍有 **34.87%** 使真实误差变差。两项一致率来自不同策略轨迹，只能描述各自实际奖励质量，不能单独构成相同姿态上的 teacher 优劣判定。

### 审计与文件

- 3200 条专家样本的点云、上下文、动作数组与 calibration 实验逐项完全一致；所有测试起始旋转和平移也逐项核对一致。
- RL 的 144 个冻结张量未改变，38 个策略张量改变；共 197 次优化器更新。
- 训练实际执行 4000 次动作，其中 1773 次降分；选中 RL 策略的测试实际执行 5820 次动作，其中 2648 次降分。
- diffusion RL 的两项现有回归测试通过，新增入口语法检查及 `git diff --check` 通过。
- 模型：`runs/region_ranking_agent/rl/agent/agent.pt`；模仿模型：`runs/region_ranking_agent/agent/agent.pt`。
- 配对结果、分病例/分档指标及审计：`runs/region_ranking_agent/summary.json`；验证曲线：`runs/region_ranking_agent/training_curves.png`。

此次结果支持继续保留 ranking teacher 作为候选方案。下一项更严格的对照应固定同一个模仿模型和初始策略，只切换 RL 奖励 teacher，并重复多个训练种子，区分编码器初始化与奖励本身的影响。

## 本地 Demo：未参与训练病例的恢复轨迹

页面：`demo/ranking_recovery.html`。展示 010、011 的完整测试轨迹，推理种子 91001；每例 near/medium/wide 各 32 个随机起点，外加一个正确位姿，共 194 条轨迹，每条 10 步。不按最终效果筛选样本。这两个病例未参与 teacher、分割器及 agent 的训练，但已用于历史测试，并非全新外部数据。

```bash
# 从已保存的真实测试轨迹导出，不重新运行策略或修改轨迹
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/export_recovery_demo.py
.venv/bin/python -m http.server 8768 --bind 127.0.0.1 --directory demo
```

打开 `http://127.0.0.1:8768/ranking_recovery.html`。可切换病例、扰动档位和随机起点，播放/暂停、逐步查看、调整速度、拖动时间轴，或切换正面/侧面/咬合面/斜视。支持查看原始上颌对照、4096 个候选顶点及每一步实际输入的 128 个顶点。旋转角和平移误差、teacher 原始分数及本步动作随回放更新。扰动档位标注的是旋转向量/平移的各轴范围，总旋转角可能大于单轴上限。

导出器核对模型和 manifest 哈希，重建所有输入顶点及距离特征，重新评分，并检查每一步动作与实际姿态变换是否一致。本次 1940 次动作包含 880 次降分动作，重算 teacher 分数与记录的最大差为 0。灰色原始上颌仅用于可视化对照，不输入策略。背景点云同样来自原网格顶点。

浏览器检查已覆盖病例/档位/轨迹切换、播放、单步、时间轴、输入点开关、视角切换和手机宽度布局，无 JavaScript 错误。Demo 页面、数据与截图保留在本地忽略目录，不提交到 GitHub。
