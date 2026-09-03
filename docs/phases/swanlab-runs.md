# SwanLab run 索引

阶段报告按 [`../experiment_plan.md`](../experiment_plan.md) 第 16.5 节链接到对应训练 run。本文件只是跨阶段的汇总视图；每个实验的权威录入处仍是 `experiments/<stage>/<id>/swanlab-url.txt` 和 `experiments/registry.csv` 的 `swanlab_url`，两者不一致时以实验目录为准并当天修正本表。

Workspace 固定为 `@richliu0153`。完整 URL 形如 `https://swanlab.cn/@richliu0153/<project>/runs/<run_id>`。

| 实验 | run 角色 | project | run_id | URL 出处 |
|---|---|---|---|---|
| E00-l20-baseline-20260823 | 无云端 run（只验环境不训练） | - | `n/a-no-cloud-run` | `swanlab-url.txt`（含登录时间线） |
| E01-tokenizer-dataset-20260823 | 不适用（不训练模型） | - | `n/a-no-training` | `swanlab-url.txt` 说明 |
| E02-model-probe-20260823 | probe seed42（step1–50 与 step51–100 resume 共用同一 run） | MiniMind-Lab | `iq14wfm1nc1ca8iigdbop` | `swanlab-url.txt` |
| E02-model-probe-20260823 | probe seed43 | MiniMind-Lab | `is8yx09hw3341ar8qvvfa` | `swanlab-url.txt` |
| E02-model-probe-20260823 | probe seed44 | MiniMind-Lab | `kngkosspce6sfuzkoztzm` | `swanlab-url.txt` |
| P01-dense-pretrain-mini-20260824 | formal | MiniMind-Lab-Stage3 | `7iochx9kfe75qa2pt6d1u` | `swanlab-url.txt` |
| P02-dense-pretrain-full-20260824 | formal | MiniMind-Lab | `3i1muwq039fpfv89fq4ru` | `git show stage5/p02-dense-pretrain-full:experiments/01-pretrain/P02-dense-pretrain-full-20260824/swanlab-url.txt` |
| P03-dense-pretrain-v1-1b28-20260901 | probe（100 step） | MiniMind-Lab | `d9is4iayxaw41ba95u92s` | `/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/probe-b32-a1-step100/attempts/20260901T080420Z-540642/driver.log` |
| P03-dense-pretrain-v1-1b28-20260901 | formal（1 epoch） | MiniMind-Lab | `qdpjh47fjt98184oos4bl` | `/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/formal-b32-a1-epoch1/attempts/20260901T080746Z-542788/driver.log` |
| P03-dense-pretrain-v1-1b28-20260901 | eval logging（`P03-Eval-V1-1B28-64M-Seq768`） | MiniMind-Lab | `k9st16wqu3i7ijy2d7q9h` | `swanlog/run-20260901_110827-k9st16wqu3i7ijy2d7q9h/backup.swanlab` |
| S01-dense-sft-mini-20260825（invalidated） | formal | MiniMind-Lab | `p2ttzc7ycn5tpaegt4odo` | `swanlab-url.txt` |
| S01R1-dense-sft-mini-20260825 | formal | MiniMind-Lab | `s2zj3jb9n8uh9v7raemx5` | `swanlab-url.txt` |

## 待修正的录入缺口

- ~~E00 的机器可读证据与报告矛盾~~（2026-09-03 已修）：`metrics.csv` 拆为 `swanlab_login_first_check`（blocked, 04:22:35Z）与 `swanlab_login`（pass, 05:18:03Z），`environment.json` 新增 `swanlab_login_recheck`，`swanlab-url.txt` 改记时间线与首批云端 run（E02 三 seed）。
- ~~`registry.csv` 中 E00/E01 的 `swanlab_url` 为空~~（2026-09-03 已修）：已写入 `n/a-no-cloud-run` 与 `n/a-no-training`。
- P02 的 `swanlab-url.txt` 只存在于 `stage5/p02-dense-pretrain-full`，`main` 的 `registry.csv` 尚无 P02 行；Phase 1 收口时补齐。
- P03 目录（`data/v1` 分支）没有 `swanlab-url.txt`，`registry.csv` 也无 P03 行；probe / formal / eval 三个 run 目前只能从 artifacts 日志和本地 `swanlog/` 回读。Phase 1 收口必须把三条 URL 写入实验目录与 registry，再由 `phase1-pretrain.md` 引用。
- P01 使用 project `MiniMind-Lab-Stage3`，与其余 run 的 `MiniMind-Lab` 不同；跨实验对比时不要假设同一 project 内可直接比较。
