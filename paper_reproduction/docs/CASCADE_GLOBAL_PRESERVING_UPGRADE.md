# Global-preserving cascade selection experiment

这是一个新的探索性selection实验，不覆盖正式Global、Local V2或既有cascade结果。
复用Global Top1与每个分支已生成的一个Local V2输出，不生成新pose、不训练或重跑docking。
默认重新打分最多1044个候选；需CPU、numpy、pandas、RDKit，密度打分不需要GPU。

## 修改内容

旧跨分支规则只在Local五个输出中选择，新规则把Global Top1一起保留。
不能直接把Global combined score与各分支的NCC混比，因此对最多六个候选重新打分。
每个parent固定读取五个**预测中心**的density crop，用同样的10 Å掩膜、
1.0/1.5 Å Gaussian sigma，拼接五个crop的目标值和每个pose渲染值后计算Pearson NCC。
所有候选看到完全相同的目标数组。重叠crop区域作为重复观测保留，对每个候选一致；
这是实验性多crop评分，并非空间去重后的全图NCC，也不是原生V2 multiscale_ncc。
原crop可能包含Global pose定义的mask，这一来源偏好并没有因共享目标而消失。

不读取真实配体位置构建crop，不用native RMSD、reference或中心误差选pose。
只使用manifest白名单字段与ranking中的candidate identity/basin分数。
三个规则一并报告，不根据174例结果自动选冠军或调参数：

| 输出方法 | 规则 |
|---|---|
| union_shared_ncc | Global + Local候选按共享目标平均NCC排序；同分保留Global |
| union_multiscale_guard | Local在两个sigma均优于Global才允许替换；合格者按平均NCC排序 |
| union_rank_fusion_guard | 同样的多尺度替换条件；合格者按Global site rank、原branch basin rank、共享NCC rank的等权倒数排名和排序 |

rank fusion使用已有Global rank 1..5，并非跨尺度不可比原始分数的相加。
只比较排名，不拟合阈值/权重。规则是未在独立验证集确认的启发式，不能保证提升。
五条Local分支不齐全时，三个新规则都返回Global。报告complete_top5与回退原因，
避免把恢复技术缺失的收益全部解释成selector精度提升。

## 同步与运行

本地项目根：

```bash
rsync -avz --relative \
  evaluation/upgrade_cascade_selection.py \
  evaluation/compare_global_cascade.py \
  evaluation/evaluate_ligand_accuracy.py \
  scripts/run_upgrade_cascade174.sh \
  tests/test_upgrade_cascade_selection.py \
  victoria:/Scr/yunsenz2/data/DiffDock-Pocket-EM/
```

服务器：

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM
conda activate diffdock-pocket-em
python tests/test_upgrade_cascade_selection.py
bash scripts/run_upgrade_cascade174.sh --preflight-only
bash scripts/run_upgrade_cascade174.sh --materialize-selected
```

正式运行自身也执行完整preflight；preflight-only不创建结果目录。
预期为174 parents、172 complete_top5、2个parent返回Global（以实际文件审计为准）。
preflight按branch名字查找V2输出，不用不同清单间的row_index强行关联；
检查所有crop可读取、mask有方差、候选SDF重原子组成匹配，坏链接立即报错。
缺失selected目录作为已缺分支记录；存在但损坏的目录报错，不静默回退。

结果默认在正式cascade根下：
05_metrics_druglike174_v1/global_preserving_selection_exploratory_v1/

## 输出与读取顺序

- experiment_manifest.json：参数、代码hash、来源、selection封存hash及状态。
- input_file_hashes.csv：所有打分结构、crop的hash。
- candidate_features_blind.csv：每parent最多6个候选、公共评分、branch身份与pose路径。
- selected_poses_blind.csv：各规则真正输出的pose及替换/回退原因；在读取RMSD前保存并hash。
- selected_results/：加--materialize-selected时复制各规则选中的SDF，原始结果不变。
- success_summary.csv：Global、rank1、旧basin、3个新规则及oracle覆盖的<2/<5 Å成功率、CI。
- paired_comparisons.csv：相对Global、rank1和旧basin的gained/lost、配对delta CI、McNemar/Holm。
- per_parent_rmsd.csv / per_parent_transitions.csv：逐体系结果与转移。
- availability_effects.csv：完整Top5与不完整回退两组对固定174增幅的贡献，区分缺失恢复与排序改变。
- REPORT.md：成功率和相对旧basin规则的配对结果摘要。

oracle_union6的定义：完整Top5 parent取Global+5 Local的事后最佳RMSD；
不完整parent仅保留Global，与新规则候选策略一致。不将oracle作可部署方法或计算其p值。
旧oracle_best5仍保留作为参照，missing policy不同在文中可见。
所有成功率固定174；评估只读取已完成的hfix Global指标和branch指标，不重新评估RMSD。
评估检查Global pose路径、Local candidate_id、parent/rank身份及rank1指标一致性。

若评分已完成、selection已封存，后续评估输入需要修正，可避免重复评分：

```bash
bash scripts/run_upgrade_cascade174.sh --evaluate-only
```

此模式核验封存CSV的hash及cohort hash，直接重建统计，不重新选pose。
--selection-only支持只运行盲态打分与选择；未封存或修改过CSV会拒绝evaluate-only。

## 解释边界

这是在已经查看过的174 test cohort上提出的探索，不是确认性提升。
所有新规则并列报告，不能把最高test分数对应规则直接称作validation-frozen。
升级若有效，下一步在独立验证集确定规则并冻结。若无改善，结果仍可区分
“保留Global的候选覆盖空间”与“当前密度评分能否识别正确候选”。
重复中心的空间去重需要重新定义/运行不同Local branches，本轮不改候选生成。
