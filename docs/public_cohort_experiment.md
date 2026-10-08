# 免登录公开病例试验

本地 `../data/001`–`010` 的来源：Zenodo [8027553](https://zenodo.org/records/8027553) 的 `FusedData/00`–`03` 四例（每例 `IOS/U.stl` 和 `IOS/L.stl`）；[OSF 8a53z](https://osf.io/8a53z) 的四例；[Dental-CAD-Designer](https://github.com/ZoliQua/Dental-CAD-Designer) 的两例。合计 10 对，每例只用一对。编号、来源路径、大小及当前/原始 SHA-256 见 [数据清单](data_catalog.json)；目录格式见[本地 STL 数据目录](data_layout.md)。STL 文件未放入 Git 仓库。

Zenodo 8027553 的 00 例原始竖直轴为 Y；编号 `007` 的上下颌已共同旋转到 Z 轴，未改变两颌相对姿态。先前 [Zenodo 11280343](https://zenodo.org/records/11280343) 的两名患者 T0 预处理审计显示上颌采样点负 SDF 占比约为 0.44 和 0.62，明显高于其余病例的 0.00–0.03；这两对已从本地源数据和运行副本中删除。几何邻近性和 SDF 审计只能排查明显错配，不能证明临床咬合正确。

以下是当前 11 对数据的运行命令；本页末尾的既有指标来自先前的 10 例实验。按最新 0.6 mm SDF、1024 FPS 点与 3 层编码器完成的 11 例重训和测试结果见 [0.6 mm SDF 重训报告](retraining_sdf06_report.md)：

```bash
.venv/bin/python scripts/build_public_cohort.py --out runs/public_cohort_all11/raw
.venv/bin/occlusion-rl --config configs/public_cohort_pilot.json prepare-batch \
  --input-dir runs/public_cohort_all11/raw --out runs/public_cohort_all11/prepared
.venv/bin/occlusion-rl --config configs/public_cohort_pilot.json train-teacher \
  --manifest runs/public_cohort_all11/prepared/manifest.json --out runs/public_cohort_all11/teacher
.venv/bin/occlusion-rl --config configs/public_cohort_extended.json train-agent \
  --manifest runs/public_cohort_all11/prepared/manifest.json \
  --teacher runs/public_cohort_all11/teacher/calibration/best.pt --out runs/public_cohort_all11/agent_extended
.venv/bin/occlusion-rl --config configs/public_cohort_extended.json evaluate-agent \
  --manifest runs/public_cohort_all11/prepared/manifest.json \
  --teacher runs/public_cohort_all11/teacher/calibration/best.pt \
  --agent runs/public_cohort_all11/agent_extended/agent.pt --out runs/public_cohort_all11/test_extended
```

`configs/public_cohort_pilot.json` 当前配置：SDF 栅格间距 0.6 mm、上颌保留原网格全部顶点及其顶点法向、网络候选点 4096、SDF 加权 FPS 取 1024 点、3 层局部注意力编码器；teacher 和 agent 使用完全相同的点规则。训练/验证/测试按病例分组。下方既有指标使用旧版上颌面采样和面法向、128 个候选点、32 个 FPS 点与 2 层编码器，不能视为当前规则的评估结果。所有病例需重新预处理并重训后才能比较新规则下的表现。

本次实际运行的清单为 `runs/public_cohort/manifest.json`，训练/验证/测试分别有 6/2/2 例。Teacher 产物位于 `runs/public_cohort/teacher_v2`。排序验证准确率从首轮 0.609 上升到第 40 轮 0.703；校准阶段最佳验证 MAE 为 0.628，预测与目标的相关系数为 0.868。

Agent 先用 100 次迭代做快速试验：独立测试的 2 例、64 次随机扰动中，平均旋转误差 8.06°→8.42°，平移误差 3.82→5.89 mm。随后使用 `configs/public_cohort_extended.json` 的 500 次迭代、8 个并行环境及 16 步轨迹重新训练。其权重位于 `runs/public_cohort/agent_extended/agent.pt`；在**同一测试集和扰动种子**上，平均旋转误差 8.06°→4.33°，平移误差 3.82→3.42 mm，教师分数 -1.785→-0.621，但严格成功率仍为 0/64。延长训练带来部分改进，**agent 仍未学会可靠恢复正确姿态**，不能将该权重用于实际矫正。较少病例、较粗 SDF 和只依据教师分数变化的奖励均可能影响结果，需要进一步实验才能定位原因。
