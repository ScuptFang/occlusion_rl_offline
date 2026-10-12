# 模仿学习 → diffusion 强化学习

本流程采用 [DPPO](https://diffusion-ppo.github.io/) 的两层决策过程思路，在环境动作之外，对扩散去噪转移做策略梯度更新。实现为本项目点云/MLP 的精简版本，不是官方代码的完整复现。

阶段一复用 `filtered8_surface_diffusion` 中完成的 2800 次监督扩散训练。阶段二仍用 diffusion 生成动作片段，不切换普通高斯动作网络：20 步采样中的前 15 步由冻结模仿模型确定性执行，最后 5 步为带正方差的高斯去噪转移，接受 PPO 更新。随机初始噪声保持不变的标准高斯先验；最后一步也保留非零噪声。

每个环境步记录最后五步去噪的输入 latent、未经裁剪的输出 latent 和旧策略 log-prob。概率按整个 `4×6` 动作片段的独立高斯联合密度求和。PPO 的新旧概率比针对每个去噪转移分别计算，不将它冒充最终动作的边缘概率，不对裁剪后的动作计算高斯密度。环境只执行最终片段的第一个动作，归一化裁剪到 [-1,1] 后按原步长转换。尾部噪声标准差在采样和概率计算中完全一致。

强化学习奖励只有 `2 × (teacher_score_next − teacher_score_current)`。奖励包含负数，不使用真实位姿误差奖励。真实位姿只用于第一阶段的专家标签及验证/测试误差统计，不进入策略输入或 RL reward。critic 只读当前点云和时间上下文，不读取真实姿态。

环境级 GAE 使用 gamma=1、lambda=.95、固定十步终点零 bootstrap。各去噪步共享该环境步 advantage，去噪折扣为 1；这是使用环境状态 value baseline 的简化两层 PPO。编码器冻结；去噪 actor 可训练；冻结参考 actor 同时用于前缀和 KL 正则。KL 正则仅约束当前访问 latent 处的尾部高斯核，不是最终动作边缘分布 KL。critic 使用独立 head，其梯度不回传 actor。

## 配置和命令

`configs/filtered8_diffusion_rl.json` 中的 `agent.diffusion_rl` 提供尾部步数、eta、噪声下限、训练轮数、并行轨迹数、PPO epochs/minibatch/clip、KL 上限、参考 KL 权重、actor/critic 学习率、value 权重、gamma/GAE、验证间隔等超参。默认 50 轮 × 8 轨迹 × 10 步 = 4000 次交互，每 10 轮验证。PPO 的 KL 超限只停止当前优化轮次，不会停止或拒绝环境动作。

```bash
.venv/bin/python -m occlusion_rl_offline.cli --config configs/filtered8_diffusion_rl.json train-agent --manifest runs/filtered8_surface_diffusion/manifest.json
.venv/bin/python scripts/evaluate_filtered_cohort.py --config configs/filtered8_diffusion_rl.json --manifest runs/filtered8_surface_diffusion/manifest.json --stage agent
```

前提是完成第一阶段，配置的 `paths.imitation_agent` 指向其 `agent.pt`。新机器先按 `docs/filtered8_retraining.md` 准备数据和完成 teacher、diffusion 模仿训练；不从远端下载本地权重。启动时校验模仿模型的 teacher/manifest 哈希、采样方式、动作尺度和扩散配置。

续训：相同命令追加 `--checkpoint runs/filtered8_diffusion_rl/agent/latest.pt`，必须保持原配置和原输出目录；恢复 optimizer、RNG、验证最优指标和历史。验证点之间的更新会重跑。CUDA 使用统一 CLI 的 `--device cuda:0`，本次只实测 CPU。

## 对照和模型选择

- `agent/imitation.pt`：初始化后的随机尾部模型，尚未 RL，作为同采样规则的对照。
- `agent/agent.pt`：仅在 RL 后 checkpoint 中按验证几何误差选最优。
- `agent/best_including_imitation.pt`：把上述初始化模型也纳入选择。如果 RL 没有改善验证指标，该文件保持 iteration 0，明确保留回退依据。
- `agent/latest.pt`：最后验证点，可续训。

测试固定 010/011、同样 192 个非零起点与 2 个零姿态、三种采样噪声种子、每条 10 步。先锁定 checkpoint 再评估，不根据测试选择种子。原确定性尾部 DDIM 的模仿模型是额外历史参照；因采样规则变化，判断 RL 效果应首先比较 `imitation.pt` 与 RL 权重。

数据仍为 5 例训练、006 验证、010/011 测试，001/008/009 不参与。仍然存在单病例验证、历史测试集、单训练种子的限制。Teacher 局部排序准确率不高，因此即使 teacher 累计奖励提高，也不能直接认定几何精度提高。

新机器可使用提交的轻量 manifest 模板 `configs/filtered8_manifest.json`，它只包含病例 ID 和相对路径，不包含数据。先将外部 STL 准备至 `runs/teacher_refinement/prepared`，再把 teacher、模仿训练和 RL 命令中的 `--manifest` 全部指定为此模板。三阶段必须使用同一个 manifest 文件，以保持 checkpoint 哈希校验一致。当前本地实验使用 `runs/filtered8_surface_diffusion/manifest.json`；不要把已训练模型直接与字节不同的模板混用。

## 实际训练与测试结果

完成 50 轮、4000 次环境交互、198 次优化更新。1738 个 teacher 降分动作全部执行。checkpoint 审计确认参考模型与编码器的 144 个参数/缓冲张量不变，去噪 actor 的 38 个参数张量接受了更新。只有 002、003、004、005、007 参与 RL。

| 验证 checkpoint | 角度 ° | 平移 mm | E |
|---|---:|---:|---:|
| RL 前，同随机去噪规则 | 1.6191 | 0.6565 | 0.18326 |
| 10 轮 | 1.7201 | 0.6517 | 0.18896 |
| 20 轮 | 1.6303 | 0.6261 | 0.18015 |
| **30 轮，选中** | **1.5478** | **0.6422** | **0.17702** |
| 40 轮 | 1.7345 | 0.6477 | 0.18937 |
| 50 轮 | 1.6993 | 0.6618 | 0.18893 |

验证 E 改善约 3.4%，后续轮数退化，不能宣称训练稳定收敛。按预定验证选择第 30 轮后锁定权重，再执行测试。

010/011 每例 near/medium/wide 各 32 个随机起点，另加原始姿态，三个采样种子均完整测试。以下整体均值不含 zero 档：

| 模型 | 角度 ° | 平移 mm | E |
|---|---:|---:|---:|
| 同采样规则的模仿基线 | 2.9697 | 1.0038 | 0.31109 |
| RL / 91001 | 2.9173 | 0.9899 | 0.30607 |
| RL / 91002 | 2.9926 | 0.9899 | 0.31078 |
| RL / 91003 | 3.0114 | 0.9717 | 0.30968 |
| **RL 三种子平均** | **2.9738** | **0.9838** | **0.30884** |

平移误差改善约 2.0%，角度增加约 0.14%（基本持平），综合 E 改善约 0.72%。三种子均值标准差为 0.0406° / 0.0086 mm；这是采样重复，不是独立训练重复或泛化置信区间。收益较小，不足以断言稳定提高。

| 档位 | 模仿 ° / mm | 模仿 + RL ° / mm |
|---|---:|---:|
| zero（原始姿态漂移） | 1.9664 / 0.8238 | 2.0091 / 0.8046 |
| near | 2.2167 / 0.7682 | 2.2118 / 0.7464 |
| medium | 2.7087 / 0.8624 | 2.7734 / 0.8395 |
| wide | 3.9839 / 1.3810 | 3.9361 / 1.3656 |

原始姿态仍明显漂移，RL 没有解决这一问题。teacher 分数提高但最终几何误差更差的比例为 16.84%，模仿基线为 16.15%。下一轮优先提高 teacher 在近正确姿态区间的动作排序可靠性，再评估 RL；不通过拒绝降分动作或提前停止掩盖问题。

RL 前后各执行 5820 个测试动作，分别包含 2551、2534 个降分动作，全部通过环境变换回放和“只执行计划首动作”检查。两组测试的初始旋转、平移、病例、teacher 哈希与 manifest 哈希一致。自动测试 23 项通过，覆盖概率密度重算、裁剪前 latent、冻结权重、真实 RL 更新、中断续训逐位复现、模型保存加载和无目标位姿推理。仅 CPU 实测。

详细指标与轨迹保存在忽略目录：

- `runs/filtered8_diffusion_rl/training_audit.json`
- `runs/filtered8_diffusion_rl/test/imitation/`
- `runs/filtered8_diffusion_rl/test/agent/`
- `runs/filtered8_diffusion_rl/comparison_summary.json`

代码、配置、测试和本报告可提交；上述模型、数据缓存及轨迹不提交。模型仍作为独立实验，不修改默认策略。
