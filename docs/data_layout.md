# 本地 STL 数据目录

仓库同级的 `../data` 仅存放上下颌 STL：`001/upper.stl`、`001/lower.stl`，依次到 `011/`。每个编号是一对；没有清单、说明或预处理文件混放其中。编号与来源、SHA-256、三角面数的对应关系在 [data_catalog.json](data_catalog.json)。

| 编号 | 病例 ID | 用途 |
| --- | --- | --- |
| 001–002 | `github_zoliqua_01`–`02` | 训练候选 |
| 003–006 | `osf8a53z_I`–`IV` | 训练候选 |
| 007–010 | `zenodo8027553_00`–`03` | 训练候选 |
| 011 | `demo_01` | 演示数据，默认与其他 10 对一起构建 |

先前 SDF 审计异常的两对 Zenodo 11280343 T0 已删除，不在清单中。其余 10 对只通过文件完整性和几何初筛，尚无逐例临床咬合姿态复核。`007` 的上下颌已同时从原始 Y 竖直轴旋转到 Z 轴；当前编号目录内全部 STL 均可由训练构建脚本直接按现有朝向读取。`001`–`010` 是上次训练输入的 STL 重新导出副本，原来源哈希保留在清单的 `original_source_sha256` 字段；`011` 与仓库演示 STL 逐字节一致。

在仓库根目录运行：

```bash
python3 scripts/audit_public_data.py
.venv/bin/python scripts/build_public_cohort.py --out runs/public_cohort_all11/raw
```

第一条校验目录结构、22 个 STL 的二进制长度、面数和 SHA-256；第二条默认生成全部 11 对病例，包括演示数据。后续 `prepare-batch`、训练和评估命令见 [多病例输入](multiple_cases.md)与[公开病例试验](public_cohort_experiment.md)。此前的 teacher/agent 指标来自未包含演示数据的 10 例实验，不能直接视作 11 例结果。
