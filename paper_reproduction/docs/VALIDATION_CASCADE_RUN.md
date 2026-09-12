# Validation drug-like cascade staged runner

服务器检查已确认：val298、test298、train2378，ID无交叉。
无ID交叉不等于蛋白同源/化学空间独立。已有oracle-local val实验不作为本次预测中心cascade结果。

## 交付范围

scripts/run_validation_cascade.py 支持：
draft-cohort → lock → global → prepare → local → merge → ncc → v2 → select → evaluate。
cohort、所有输出和最终统计的N均从validation配置读取，不使用174常量。
本轮还没有服务器Global冻结重排命令与完整validation full-polymer source。
因此可先运行draft-cohort，不能把本包当作已配置完毕的一键GPU任务。
global阶段通过显式adapter接入Global仓库，不猜checkpoint、confidence权重或density参数。

## 1. 同步与生成化学审查表（CPU）

从本地项目根运行：

```bash
rsync -avz --relative \
  scripts/run_validation_cascade.py \
  configs/validation_cascade.example.json \
  visualizations/figure2_global/input/raw/chemistry/chemistry_manifest.csv \
  paper_design/cohort_definitions/authoritative_druglike174_v1/chemistry_exclusions.csv \
  victoria:/Scr/yunsenz2/data/DiffDock-Pocket-EM/
```

服务器：

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM
conda activate diffdock-pocket-em
cp -n configs/validation_cascade.example.json configs/validation_cascade.json
python scripts/run_validation_cascade.py draft-cohort --config configs/validation_cascade.json
```

输出artifacts/paper_validation/global_to_local_druglike_v1/00_cohort_review/druglike_review.csv。
对所有初始drug_like记录填写review_decision=include/exclude；exclude填写review_reason。
matches_previous_excluded_ccd提示与test已排除CCD相同的条目，但不是全部化学审查的替代。
按test的结构/身份标准审查所有条目；不能按是否完成推理、RMSD或NCC删除行。
此处审查必要性来自test曾对drug_like标签做过18例化学校正，不能把原标签当同口径最终名单。

## 2. 配置Global与source输入，冻结之前完成

config的global_adapter应指向已核实的bash脚本，它接收：
VAL_COHORT、VAL_GLOBAL_OUTPUT、VAL_GLOBAL_SCORES环境变量。
任务是Global生成 → 与test相同的confidence排序 → 固定参数residual-shape重排，
最终输出每例20行，字段complex/complex_name/pdb_id、sdf_path、
combined_at_best_weights、original_rank。不可通过validation RMSD挑选排序权重。

本地已找到Global通用入口run_evaluation_single_slurm.sh val_inference，
支持VAL_SPLIT、VAL_RESULTS_DIR、FINETUNED_MODEL_DIR等环境变量，
但仅有这个入口还不等于完成canonical Top20重排。不要把SAMPLES直接改20：
Top20是送入density重排的池大小，原始Global采样量必须与test保持一致。

source_manifest要求每个parent唯一，含complex_name/pdb_id、experimental_protein、
ligand、map_path、mask_structure、reference_ligand，路径使用绝对路径。
可从现有contact_entity_val_all_source.csv整理完整输入并补full-polymer mask路径；
不能以source_full_polymer100.csv的100例取代完整validation名单。
不能使用已知中心跑Local：prepare阶段会用Global pose重建center与flexible residues。
reference_ligand只写入独立evaluation manifest，不传入Local inference。

配置文件必须在lock前填好。运行：

```bash
python scripts/run_validation_cascade.py lock --config configs/validation_cascade.json
```

freeze后不允许改cohort、config或已锁定代码。global adapter应使用同一测试protocol，
Global/Local模型的checkpoint及参数还应随研究记录单独存档，勿把wrapper hash当模型hash。

## 3. 分阶段运行

每步成功后才运行下一步。GPU阶段需在已有可用GPU资源的会话中执行。

```bash
python scripts/run_validation_cascade.py global --config configs/validation_cascade.json
python scripts/run_validation_cascade.py prepare --config configs/validation_cascade.json
python scripts/run_validation_cascade.py local --config configs/validation_cascade.json
python scripts/run_validation_cascade.py merge --config configs/validation_cascade.json
python scripts/run_validation_cascade.py ncc --config configs/validation_cascade.json
python scripts/run_validation_cascade.py v2 --config configs/validation_cascade.json
python scripts/run_validation_cascade.py select --config configs/validation_cascade.json
python scripts/run_validation_cascade.py evaluate --config configs/validation_cascade.json
```

global已有确切同协议结果时可跳过global阶段，prepare会验证每例20个候选。
local阶段某GPU失败时先查看日志；merge是独立阶段，可对现有输出做SDF语义审计并
生成available分支清单。缺失parent仍留在原N分母。不要删除cohort ID来通过完整性检查。
prepare阶段缺Global/source先停止并报告，不静默截取交集；若无法恢复须另做固定N缺失路径适配。
每阶段成功后写完成标记，拒绝重复运行已完成阶段；失败阶段可能有部分文件，需查明后处理，不自动删除。

select运行冻结的三个启发式，selection-only先封存无GT的选择。evaluate之后才读reference与RMSD。
主方法为预先指定的union_rank_fusion_guard，其他两种规则只作敏感性分析；
当前通用统计表仍给出全部对比的Holm校正，不把最好的validation规则自动提升为主方法。

结果：05_metrics/global_comparison与06_selection/REPORT.md。
验证的是既定cascade规则在validation上的复现性，不宣称validation从未参与任何上游调参。
