# 大量病例数据输入

原有 `PoseDataset` 能接收多个 `.npz` 路径，但旧 CLI 只传单个病例，agent 环境也只加载一例。现在通过病例清单（manifest）把批量准备、teacher 训练、agent 训练和独立测试连起来。`score`、`infer-agent` 仍以一个病例为单位运行；批量推理可对清单中的病例逐个调用。

## 目录格式

原始 STL：

```text
my_raw_cases/
  patient_001/upper.stl
  patient_001/lower.stl
  patient_002/upper.stl
  patient_002/lower.stl
  ...
```

每对 STL **必须已经在正确咬合状态，并且使用相同的坐标系和 mm 单位**。目录名是唯一病例 ID；文件名可通过 `configs/default.json` 的 `cohort.upper_filename` 和 `cohort.lower_filename` 修改。不完整的上下颌文件对会报错。当前 `prepare` 使用上下颌 STL 的面和下颌近邻法线估算 SDF，非封闭网格需要检查生成的 `case.audit.json` 与法线方向。

已预处理数据也可直接输入：

```text
my_prepared_cases/
  patient_001/case.npz
  patient_002/case.npz
  ...
```

`case.npz` 必须包含 `xyz [N,3]`、`normal [N,3]`、`curvature [N]`、`sdf_grid [X,Y,Z]`、`origin [3]`、`spacing [3]`；坐标、SDF 和网格间距均以 mm 为单位。上颌必须对应正确咬合姿态，随机偏移由训练环境生成。建议所有病例采用同一预处理配置和 SDF 符号规则。

## 建清单

从 STL 批量准备，顺序处理每例并输出 `case.npz`、审计文件和 manifest：

```bash
occlusion-rl prepare-batch --input-dir my_raw_cases --out runs/cohort
```

中断后可加 `--skip-existing` 继续；已存在的 `.npz` 会验证格式后跳过。从已有 `.npz` 目录直接索引：

```bash
occlusion-rl index-cases --input-dir my_prepared_cases --out runs/cohort/manifest.json
```

manifest 使用相对路径引用病例文件，移动整个目录仍可使用。它将病例 ID 按 `cohort.split_seed` 随机分为 `train`、`val`、`test`，比例由 `cohort.validation_fraction` 和 `cohort.test_fraction` 控制；至少需要 3 例且每组非空。**分组单位是病例，不是采样点或随机扰动。**如果同一个患者有多个扫描，应先在外部给它们统一患者级分组，避免同患者跨组泄漏。不要在比较实验之间重新生成不同划分；复用同一个 manifest。

## 训练和测试

```bash
occlusion-rl train-teacher --manifest runs/cohort/manifest.json --out runs/cohort/teacher
occlusion-rl train-agent --manifest runs/cohort/manifest.json \
  --teacher runs/cohort/teacher/calibration/best.pt --out runs/cohort/agent
occlusion-rl evaluate-agent --manifest runs/cohort/manifest.json \
  --teacher runs/cohort/teacher/calibration/best.pt \
  --agent runs/cohort/agent/agent.pt --out runs/cohort/test
```

Teacher 排序和分数校准只用 `train` 病例优化，用 `val` 病例选权重；agent 在 `train` 病例上轮流生成随机偏移，并使用 `val` 选择策略权重；`evaluate-agent` 只用 `test` 病例。输出的 `cases` 是参与评估的病例数，`episodes` 是总扰动数，指标对病例取平均。Agent 训练的奖励仍只由冻结 teacher 的前后分数差给出。

使用 `--manifest` 且未指定 `--source` 时，排序 teacher 从随机初始化开始，避免默认的单病例示例权重污染多病例测试。若有来自**完全独立病例集**的预训练权重，可以显式传 `train-teacher --source path/to/pretrained.pt`。

大量病例时，`teacher.ranking_pairs_per_epoch` 是**每例每 epoch**的配对数，`teacher.ranking_validation_pairs_per_case` 是每个验证病例的配对数；`teacher.calibration_poses_per_epoch` 与 `calibration_validation_poses` 则分别是整个划分的总数，至少要覆盖各自划分内的全部病例。`agent.iterations` 至少应等于训练病例数；`agent.validation_episodes` 和 `test_episodes` 是**每例**扰动数。可调整这些参数控制训练时长。数据集按病例块打乱，每个数据集对象最多缓存两个完整 SDF 网格，避免一次加载所有病例。

当前附带的 teacher 和 agent 权重来自单病例，只用于演示接口；多病例性能需要在新 cohort 上重新训练和测试。更换点密度或 SDF 规则时，teacher 与 agent 的配置必须保持一致。
