# 自动 validation cascade smoke

运行器 scripts/run_validation_cascade_auto.py 将Global、confidence、冻结density公式、
prepare、Local、merge、NCC、V2同步串行衔接。各计算阶段内部按GPU数量并行。
没有SLURM后台提交后提前进入下一步的问题。不要在没有GPU资源的登录节点运行。

默认mode=smoke，使用已审定validation名单中字母序第一例，或--case指定其中一例。
smoke仍保留Global40、confidence Top20、Global5个预测中心、每branch Local40+40、完整V2。
每个阶段都成功才进入下一个。smoke不计算RMSD、不选择好成绩的case、不因文件缺失自动换case。
仅1个GPU，默认编号0；Local/V2现有脚本也按0编号使用GPU。

Global使用conda run -n diffdock（可--global-env更改），Local/V2使用当前Python环境。
需要conda命令可用，并已激活diffdock-pocket-em。两个环境分开，避免Global子脚本切换环境
造成随后confidence运行在错误环境。checkpoint默认路径仍需与test核对。

source由processed protein/ligand、map CSV和完整CIF的文件路径构建，不调用oracle中心计算。
完整CIF root从已有val100 full-polymer审计清单统一推断，仅借用root，不限制为100例；
无法唯一推断时--raw-root明确指定。reference只核对文件存在，不读取坐标做推理。

运行：

```bash
python scripts/run_validation_cascade_auto.py --config configs/validation_cascade.json --mode smoke
```

默认输出artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_v1。
每个阶段独立日志：global.log、prepare.log、local.log、merge.log、ncc.log、v2.log。
运行失败会终止并保留日志和中间产物；不自动重跑/删除。指定新--output-root用于修复后的重试。
--preflight-only只检查初始文件和Global checkpoint路径，不创建输出目录、不启动推理。
此预检查不是分子解析/GPU模型运行验证，后续各阶段仍可能发现环境或输入错误。

成功后必须出现AUTO_SUCCESS.json，且requested_branches=available_branches=selected_v2_branches=5。
smoke完整检查通过后，才可运行：

```bash
python scripts/run_validation_cascade_auto.py --config configs/validation_cascade.json --mode full
```

full默认4卡（配置num_gpus），输出auto_full_v1，不覆盖smoke。
使用自定义smoke目录时--smoke-result指向它的AUTO_SUCCESS.json。
full也保留原名单固定分母，不能删失败体系；任何Local GPU进程失败会停在local，
需先检查日志，不会自动假定所有未完成的分支都是合理技术失败。
本轮到V2为止；未自动执行rank-fusion或RMSD评估，防止smoke提前查看验证成绩。

本地验证了预检查不启动计算、不创建输出，以及失败case不被替换。
真正GPU smoke需在服务器运行，尚未在本地验证。
# Resume after confidence failed for missing ESM

The adapter now passes `--esm_embeddings_path`, checks requested case coverage
and tensor width (1280), and requires all 40 confidence outputs before density
ranking. The default ESM file follows Global's `run_analysis_dtai.sh`:
`$GLOBAL_ROOT/data/esm_embeddings.pt`. Override with `ESM_EMBEDDINGS_PATH` if
needed. Coverage checks do not prove chain sequence identity; use the embeddings
prepared for the same processed receptors as the Global analysis workflow.

Run from the server project, after syncing the updated auto runner, adapter and
`scripts/check_global_confidence_inputs.py`:

```bash
python scripts/run_validation_cascade_auto.py \
  --config configs/validation_cascade.json --mode smoke --case 6DZW \
  --reuse-global-from "$PWD/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_v1" \
  --output-root "$PWD/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_esmfix_v1"
```

Add `--preflight-only` to check before creating the new directory. Reuse requires
the same cohort, source manifest, Global input CSV identities, complete raw poses
and matching checkpoint hashes in the prior log. It copies the 40 raw poses into
a new run, records their hashes, and restarts confidence. Old output is preserved.
It then runs density, prepare, Local, merge, NCC and V2 automatically.
Use the new run's `AUTO_SUCCESS.json` as `--smoke-result` for a later full run.

## Dynamic A6000 parent sharding

`scripts/submit_validation_cascade_a6000_dynamic.sh` queries distinct idle nodes in the
`A6000` partition, freezes that count as the parent-shard count, preflights all
187 reviewed cases once, then calls `sbatch` once per shard. Despite the legacy
filename, the count is dynamic. Eleven idle nodes produce eleven disjoint shards
of seventeen parents each. The scheduler chooses nodes; the submission does not
pin stale host names. The idle-node snapshot, exact shard count and job IDs are
saved under the run root.
Each parent-sharded A6000 job explicitly freezes `gpus_per_job=1`. Both the
SLURM wrapper and Global adapter compare that request with PyTorch's visible GPU
count before inference, preventing a single-GPU allocation from spawning GPU
1--3 workers from the four-GPU repository default.

```bash
bash scripts/submit_validation_cascade_a6000_dynamic.sh --dry-run
bash scripts/submit_validation_cascade_a6000_dynamic.sh
```

Set `VALCAS_MAX_SHARDS` to cap the number of submissions or
`VALCAS_WALLTIME` to override the default `48:00:00`. `--dry-run` creates the
submission plan and log directory but no shard output, so the real submission
can safely reuse the same run root.

Set `VALCAS_SHARD_COUNT` to keep a fixed parent partition even when fewer nodes
are idle at submission time. Slurm starts the currently schedulable jobs and
queues the remainder. This is the recovery-safe choice after a cancelled run,
because it preserves the original parent balance without collapsing all parents
into one long single-GPU job.

For H100, `scripts/submit_validation_cascade_h100.sh` freezes ten parent shards,
submits every shard to partition `h100`, and requests `gpu:h100:1` per job. It
never falls back to A6000. Jobs enter the H100 queue when no card is immediately
idle. The worker still asserts that PyTorch sees exactly one allocated GPU.
