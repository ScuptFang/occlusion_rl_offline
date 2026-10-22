# 全顶点 / Ranking / Diffusion 实验（2026-10-21）

当前 pipeline 的实验归档。使用语义分割后简面全部上颌顶点（最多 4095），不做 FPS；下颌简面三角面用于最近表面带符号距离。特征为坐标、顶点法向、原网格曲率、距离共 8 维，3 层 64 通道局部注意力，保留距离加权 pooling。Teacher 仅 ranking，无 calibration。先模仿预训练，再混合专家轨迹奖励与 ranking 分差奖励；固定 16 步，降分动作不拒绝。

## 数据隔离

- 训练：002、003、004、005、007、010。
- 验证与选模：006。
- 测试：011；不参与训练、验证、专家轨迹采集或模型选择。
- 排除：001、008、009。

原 STL 应放在仓库同级 `data/<编号>/upper.stl` 与 `lower.stl`。source manifest 中只读取划分与 ID；历史 case 缓存路径无需存在。数据、模型、生成页面载荷不在本分支文件树中；从历史分支继承的旧二进制仍存在于 Git 历史，本次不改写历史。

## 实际预算与选模

| 阶段 | 实际预算 | 说明 |
|---|---:|---|
| 分割 | 96 epochs | 2048 体素代表点上限，48 通道、3 层 |
| Teacher ranking | 57 epochs | 原上限 96；连续 49 轮未刷新验证综合指标后人工早停；选第 8 轮 |
| 模仿学习 | 4200 updates | 从原 8400 减半；每训练病例 96 条专家轨迹 |
| RL | 150 iterations | 从原 300 减半；每轮 8 环境 × 16 步；选第 60 轮 |

Teacher 选模指标为 `MAE + delta_MAE + 0.5*(1-local_accuracy)`：最佳 0.5648517，最后 0.8858294。最优模型 SHA256：`2bc1ab899ba2a241135c72483003ffc8f18361c94c5cf7575d5785c86c48f4b1`。

`configs/semantic_pipeline.json` 保留 teacher 原定 96 轮上限。`configs/semantic_all_vertices_20261021.json` 使用实际执行的 57 轮及最终 agent 预算，方便重跑；这是固定预算复现，不宣称实现了自动早停。重跑按验证集重新选最优，跨平台不保证逐位一致。

## 环境与运行

实测 macOS arm64、Python 3.9、PyTorch 2.8.0、CPU、VTK 9.4.2，torch 线程数 2。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[semantic]'
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m occlusion_rl_offline.semantic_pipeline \
  --config configs/semantic_all_vertices_20261021.json
```

NVIDIA GPU 使用匹配驱动的 CUDA PyTorch，再加 `--device cuda`；同一运行中设备和配置快照必须一致。流水线顺序包含准备、分割、分割测试、简面、配置、ranking、模仿、RL、测试、端到端测试与可视化。已完成阶段拒绝覆盖，继续执行时用 `--stages` 指定剩余阶段。CPU 完整运行耗时较长。

```bash
.venv/bin/python -m http.server 8772 --bind 127.0.0.1 \
  --directory runs/semantic_all_vertices_20261021/viewer
```

打开 `/agent.html`、`/teacher.html`、`/index.html`。可视化工具源代码随 Python 包发布，生成数据只保存在 runs。

## 011 测试结果

固定分割简面区域，非零测试档宏平均（每档 32 个起点；零扰动另记）：

| 模型 | 旋转误差 | 平移误差 |
|---|---:|---:|
| 初始姿态 | 4.2539° | 2.1724 mm |
| 模仿 | 4.4821° | 1.9089 mm |
| RL | 4.4052° | 1.9032 mm |

RL 相比模仿略有改善，但相对初始姿态平均旋转误差增大，不能认定已经稳定恢复咬合。测试采用共享评估器的测试档位，并非训练五档宽度完全一致。完整 STL 在扰动后重新分割和简面的五个起点均完成处理，最终平均 4.2981° / 3.8194 mm；样本很少，流程完成不等于恢复成功。

独立 demo 快照展示 011 五档共 26 条固定区域轨迹（模仿/RL 配对），每条 16 动作；使用最终同一最佳 checkpoint。全流水线另生成包含端到端输入的页面。原始目标只用于误差和视觉对照，不进入网络输入。

验证：42 项单元测试通过；002/003 真实全顶点前向反向检查通过；demo 播放到第 16 步、同起点切换、移动端布局和 ranking-only 曲线检查通过。
