# 最新训练与推理栈 / GPU

分支 `feature/training-inference-gpu` 收录数据准备、ranking → calibration teacher、联合 teacher 对照、连续 PPO、连续误差评估和两个可视化界面。推荐从 `configs/latest_gpu.json` 开始重新训练；历史验证选中模型配置为 `configs/optimization_selected.json`，它引用本机 `runs/`，并不随 Git 分发。

## 环境

Linux + NVIDIA GPU 使用 Python 3.10 或 3.11；先确认 `nvidia-smi` 正常。本项目只需 torch，不需 torchvision/torchaudio。固定环境使用 PyTorch 2.5.1 CUDA 12.4 wheel，安装方式来自 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。GPU/驱动必须支持此 wheel；新 GPU 架构可能需要更新 PyTorch，不能保证这套固定版本覆盖所有显卡。

```bash
PYTHON_BIN=python3.10 bash environments/setup.sh cu124
source .venv-cu124/bin/activate
python scripts/check_environment.py --device cuda:0
```

CPU（含 macOS）环境：`PYTHON_BIN=python3.10 bash environments/setup.sh cpu`。不会改动已有 `.venv`。设置 `OCCLUSION_VENV` 可以指定新环境目录。

设备优先级：命令行 `--device` > 环境变量 `OCCLUSION_DEVICE` > config 的 `runtime.device` > `auto`。`auto` 有 CUDA 则使用 CUDA，否则 CPU。显式要求 CUDA 时，不可用就报错，不悄悄退回 CPU。支持 `cuda:N` 单卡，不含 DDP、AMP 或 MPS。统一 CLI 的 `--device` 放在子命令前；独立训练/评估模块及新 demo 导出脚本也支持此选项。

## 从数据重新运行

在仓库根目录执行。将配对 STL 放在 `../data/001/upper.stl`、`lower.stl` 等数字病例目录；数据和 `runs/` 不包含在该代码分支中。至少三个病例，以病例为单位划分训练/验证/测试。数据准备无需 GPU。

```bash
occlusion-rl --config configs/latest_gpu.json --device cpu prepare-batch --input-dir ../data --out runs/latest_gpu/prepared
occlusion-rl --config configs/latest_gpu.json --device cuda:0 train-teacher --manifest runs/latest_gpu/prepared/manifest.json
occlusion-rl --config configs/latest_gpu.json --device cuda:0 train-agent --manifest runs/latest_gpu/prepared/manifest.json
occlusion-rl --config configs/latest_gpu.json --device cuda:0 evaluate-agent --manifest runs/latest_gpu/prepared/manifest.json
occlusion-rl --config configs/latest_gpu.json --device cuda:0 infer-agent --case runs/latest_gpu/prepared/cases/010/case.npz --upper-stl ../data/010/upper.stl
```

最后一条从原始咬合姿态运行，适合检查漂移；恢复测试先用 `perturb` 生成扰动 case，再把该 case 交给 `infer-agent`。教师从头训练 ranking，再 calibration；PPO 使用同一次训练的 teacher。切换 teacher 后需要重新训练 agent，不能直接配用另一个 teacher 的已有 agent。

联合 teacher 对照（warm start 来自上述 calibration）：

```bash
python -m occlusion_rl_offline.joint_teacher --config configs/latest_gpu.json --device cuda:0 --manifest runs/latest_gpu/prepared/manifest.json --source runs/latest_gpu/teacher/calibration/best.pt --out runs/latest_gpu/teacher_joint --mode consistent
```

连续 PPO 恢复：在配置中增加 `agent.iterations`，然后加 `train-agent --checkpoint runs/latest_gpu/agent/latest.pt`。checkpoint 保存 optimizer、CPU RNG 和所选 CUDA 设备 RNG；跨 CPU/GPU 恢复不保证逐位复现。恢复时 teacher 哈希和采样配置必须匹配。

## 推理展示

```bash
python demo/generate_teacher_scores.py --config configs/latest_gpu.json --manifest runs/latest_gpu/prepared/manifest.json --device cuda:0
python demo/generate_multi_pose.py --config configs/latest_gpu.json --manifest runs/latest_gpu/prepared/manifest.json --teacher runs/latest_gpu/teacher/calibration/best.pt --agent runs/latest_gpu/agent/agent.pt --device cuda:0 --out demo/optimized_data.js --examples-per-band 8
```

打开 `demo/teacher_scores.html` 或 `demo/optimized.html`。导出脚本会覆盖对应展示数据；页面显示实际步数，GPU 配置默认 10 步。`generate_multi_pose.py` 通过 `--data-root` 指定 STL 目录。

## 计算位置与实验约束

GPU 执行 teacher/agent 前向、反向、PPO 优化和模型评估。STL 处理、网格曲率、SciPy 最近点查询、4096 顶点候选及加权 FPS 仍在 CPU；小点云小批量不保证 GPU 总耗时更快。输入为 64×8，保留顶点法向/曲率，不生成体积 SDF。奖励仅为 teacher 分数差，降分动作照常执行，无 STOP 或精度早停。训练早停只结束后续训练迭代，不截断已开始的动作轨迹。

CUDA 环境需在目标机器执行 `check_environment.py`，它真实运行 3 层 encoder 前向和反向。本次开发机器没有 CUDA，不能将 CPU 验证当作 GPU 性能或数值一致性验证。

## 本次验证记录（2026-10-08）

本机 CPU / PyTorch 2.8.0：7 项单元测试通过；3 层 teacher 前向/反向检查通过；ranking → calibration 各 1 epoch、联合 teacher 1 epoch、PPO 冻结编码器 1 轮及恢复后解冻 1 轮、独立 score 和 2 步 agent 推理均完成。检查使用现有病例，输出位于被 Git 忽略的 `runs/gpu_stack_smoke/`，仅验证运行链路，不是新精度实验。固定安装环境（Python 3.10/3.11、PyTorch 2.5.1 CUDA 12.4）未在本机安装或实测；目标 GPU 上先执行安装脚本的真实前后向检查。
