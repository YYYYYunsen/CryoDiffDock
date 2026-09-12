# Canonical174 Global versus cascade

比较同一 canonical Top-20 来源的 Global-only 与现有 cascade 输出；不提供真实口袋进行预测，
不包含 oracle-centred DiffDock-Pocket。reference 仅在预测/排序完成后用于 RMSD 评估。

## 同步两份代码

在本机项目根目录运行：

```bash
rsync -avz --relative \
  evaluation/compare_global_cascade.py \
  scripts/run_compare_global_cascade174.sh \
  victoria:/Scr/yunsenz2/data/DiffDock-Pocket-EM/
```

服务器应已有 evaluation/evaluate_ligand_accuracy.py（复用其 symmetry_rmsd）。

## 服务器运行

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM
conda activate diffdock-pocket-em
bash scripts/run_compare_global_cascade174.sh
```

默认输出：

```text
artifacts/paper_test/global_to_local_cascade_top20_canonical192_v1/
  05_metrics_druglike174_v1/global_vs_cascade174_v1/
    REPORT.md
    success_summary.csv
    paired_comparisons.csv
    per_parent_rmsd.csv
    per_parent_transitions.csv
    global_top1_audit.csv
    analysis_manifest.json
```

脚本只读取现有结果和结构文件，不训练、不推理、不重新选择 Local 输出。
依赖 pandas、numpy；默认结构评估另需 RDKit 与 spyrmsd（项目环境通常已包含）。
不需要 GPU。输出目录必须为空，避免把不同版本分析混在一起。

参考配体现在由运行脚本明确从 `01_prepared/evaluation_manifest.csv` 读取，
不再假定 `00_source/cascade_source_reveal.csv` 含有 reference_ligand。
可设置 REFERENCE_MANIFEST 指向另一份具有完整174例reference的评估清单。
按 parent_complex_name、pdb_id 或 complex_name（支持 __global_rankXX 后缀）关联，
不按row_index配对；同一parent的不同分支必须指向相同reference文件。
缺列、缺parent、路径冲突或reference文件不存在会在逐例RMSD计算之前明确报错，
不能用 --allow-global-evaluation-failures 绕过这些配置错误。
若真实reference列名不同，用 --reference-column 显式指定；禁止用预测ligand列代替。

## 先看这两个文件

- success_summary.csv：Global、rank1 cascade、multiscale NCC、basin-q25 NCC、oracle best5；
  <2/<5 Å 成功数、固定174百分率、95% bootstrap CI、有效数量、中位RMSD。
- paired_comparisons.csv：Global→各cascade端点、rank1→跨分支端点；
  gained/lost、delta_pp、配对95%CI、exact McNemar p 与 Holm p。

主对比为 Global→rank1；跨分支 selector 默认 exploratory。
只有存在验证集冻结证据，才使用 --selector-status validation_frozen。
oracle 不计算 p 值，只报告候选覆盖差距。
Holm family 为两个阈值下所有非oracle对比，CI为未作多重校正的parent bootstrap CI。
缺失计失败；中位RMSD只在有效结果上计算，不据此取代固定分母成功率。

## 数据核验

1. 174唯一ID；每个parent恰好20个候选；重复候选、非法分数会停止运行。
2. combined_at_best_weights降序，original_rank升序（若存在），剩余tie保持输入顺序，
   与build_cascade_predicted_center_manifest.py一致。
3. 重新选出的Global Top1路径必须与source reveal的rank1 global_pose解析后一致；否则停止。
4. 只在选完Global Top1之后评估RMSD；symmetry-corrected，不做质心平移或刚体叠合。
5. 两份parent表中的rank1与oracle结果必须一致；读取并保留cross summary的missing policy。
6. 计算源文件hash与全部参数；不把60/174、64/174等历史数值硬编码为结果。

原始192行cohort之外的数据按174名单过滤；canonical pool缺失属于输入完整性问题，
默认停止而不是用旧池补齐。Global结构评估失败先写入global_top1_audit.csv并停止，
修复路径/环境后使用新输出目录重跑；只有确认为技术失败时显式添加
--allow-global-evaluation-failures，在固定174分母计失败。

```bash
OUTPUT_DIR=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/global_vs_cascade174_review_v2 \
  bash scripts/run_compare_global_cascade174.sh
```

如果已确认Global score表中存在同口径、未叠合、对称校正的逐pose RMSD字段，可免去SDF重算：

```bash
bash scripts/run_compare_global_cascade174.sh --global-rmsd-column YOUR_VERIFIED_COLUMN
```

字段名需显式提供，脚本不会猜测或把centroid distance当RMSD。
默认路径核验要求source与Global score中的pose解析为同一文件；镜像搬迁导致路径差异时，
先核验并统一输入路径，不跳过来源校验。

## 本地验证范围

2026-09 氢原子修复：服务器日志表明6O9G、7LO6、8XB3、9CF5的reference各有一个
在默认RemoveHs之后仍残留的H，导致与prediction的坐标数组形状不一致。
evaluate_ligand_accuracy.py现按原子序数显式删除所有H节点（仅内存副本），
不对齐/优化重原子坐标、不修改原始SDF。无需重跑docking。
测试 tests/test_ligand_heavy_atom_rmsd.py 需在有RDKit/spyrmsd的服务器环境运行，
之后用diagnose_global_rmsd_failures.py重放四例，成功后再运行完整统计。
已有cascade指标不会自动重写；若其失败审计存在同类残留H错误，须另行按相同策略复评，
不能直接假定此前所有技术缺失都是该原因。

tests/test_compare_global_cascade.py使用小型合成表检查排序、配对、固定分母、阈值、
oracle标记及跨来源错配拦截。服务器真实174结果尚未由本脚本运行验证。
