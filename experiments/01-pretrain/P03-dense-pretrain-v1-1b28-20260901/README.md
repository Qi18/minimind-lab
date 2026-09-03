# P03 Dense Pretrain V1 1B28（8×L20）

状态：训练前方案已冻结；尚未启动 100-step probe 或正式训练。

## 目的与结论边界

本实验使用已验收的 `final-remix-v1` 数据，从随机权重训练 64M Dense Base 模型，并记录独立 validation NLL/PPL、吞吐、显存、GPU 利用率和正式七项 benchmark。它是新的“质量—成本基线”，不是相对 P01/P02 的单变量数据 A/B：数据构成、validation 协议、训练入口和 micro-batch 拆分均发生变化。

## 冻结输入

- 数据：`/data/datasets/minimind-lab/data-v1/pretrain-v1-1b28/final-remix-v1`，`_SUCCESS.status=accepted`；
- fingerprint：`cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd`；
- 训练：40 shards、2,313,483 rows、1,280,000,000 loss-target tokens；
- 验证：1 shard、11,525 rows、6,400,000 loss-target tokens；
- 模型：Dense 63,912,192 parameters，hidden 768，8 layers，vocab 6400；
- 固定项：seq 768、BF16、AdamW、lr 5e-4、grad clip 1、seed 42、8×L20、1 epoch；
- 全局 sequence batch 固定为 256。

启动脚本在训练前校验 `_SUCCESS`、manifest SHA、fingerprint、行数和 token 预算，并逐一重算 41 个数据 shard 的 size/SHA-256（总计 3,804,930,498 bytes）；随后单进程验证 Arrow cache，避免 8 rank 并发建 cache。

## Probe 矩阵

| 顺序 | 每卡 batch | accumulation | 全局 batch | 完整 micro-steps | 完整 optimizer steps |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 1 | 256 | 9,038 | 9,038 |
| 2（仅 32 OOM） | 16 | 2 | 256 | 18,075 | 9,038 |
| 3（仅 16 OOM） | 8 | 4 | 256 | 36,149 | 9,038 |

`--max_steps=100` 只在第 100 个 optimizer update 后正常退出、做 final validation 并保存结果；cosine LR 分母始终是完整一轮的 9,038 updates，所以 probe 采样正式 run 的前 100 步。

首选 probe 的唯一命令：

```bash
cd /data/projects/minimind-lab-data-v1
RUN_KIND=probe FROM_RESUME=0 DRY_RUN=0 \
PER_GPU_BATCH_SIZE=32 ACCUMULATION_STEPS=1 MAX_STEPS=100 \
bash experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/command.sh
```

若明确 CUDA OOM，保留失败目录并立即使用下一级独立输出：

```bash
RUN_KIND=probe FROM_RESUME=0 DRY_RUN=0 PER_GPU_BATCH_SIZE=16 ACCUMULATION_STEPS=2 MAX_STEPS=100 \
bash experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/command.sh

# 仅当 16×2 仍 OOM：
RUN_KIND=probe FROM_RESUME=0 DRY_RUN=0 PER_GPU_BATCH_SIZE=8 ACCUMULATION_STEPS=4 MAX_STEPS=100 \
bash experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/command.sh
```

probe 必须 exit 0、`optimizer_step=100`、无 NaN/Inf/OOM/NCCL 错误，并留下独立 final checkpoint、resume state、`metrics.jsonl`，以及 `attempts/<attempt-id>/` 下独立的 `driver.log`、`nvidia-smi.csv`、runtime manifest 和 exit status。exit status 记录 launcher 端到端 wall time；正式 profile 只从通过的最前序配置中选择。

只读门禁：

```bash
RUN_KIND=probe FROM_RESUME=0 DRY_RUN=1 PER_GPU_BATCH_SIZE=32 ACCUMULATION_STEPS=1 MAX_STEPS=100 \
bash experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/command.sh
```

## 正式训练

probe 通过后代入所选 profile。例如首选正式命令：

```bash
cd /data/projects/minimind-lab-data-v1
RUN_KIND=formal FROM_RESUME=0 DRY_RUN=0 \
PER_GPU_BATCH_SIZE=32 ACCUMULATION_STEPS=1 MAX_STEPS=0 \
bash experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/command.sh
```

launcher 会先验证同 profile probe 的 exit 0、`optimizer_step=100`、9,038-step scheduler 分母、精确 final validation、checkpoint SHA，以及 dataset/trainer/protocol/command SHA；任一不符都会阻断 formal。

probe/formal 的目录、checkpoint 和 SwanLab run name 相互隔离，也不覆盖 P01/P02。正式 run 每 250 optimizer steps 保存可续训状态，每 1,000 步精确 validation，结束时必做 final validation/checkpoint。纯推理 checkpoint 保持 FP16；resume state 以原始训练 dtype 保存模型主权重并包含 optimizer/scaler，使用临时文件加原子替换。续训必须使用同一 profile、同一目录并显式设 `FROM_RESUME=1`；resume contract 会额外固定 dataset fingerprint、trainer SHA 和 protocol SHA，拒绝 world size、batch、accumulation、seq、epoch 或完整 schedule 变化。每次续训写入新的 attempt 子目录，不覆盖既有 driver/runtime/GPU/exit 日志。

预估纯训练 78–84 分钟；计入 validation、导出和七项评测后约 90–105 分钟，以 probe step time 修正。

## Validation、记录与产物

validation 不使用会补齐样本的 `DistributedSampler`。各 rank 按 stride 处理互斥索引，使用 unwrapped raw model 无梯度前向；all-reduce NLL sum、valid token count 和 row count。必须严格得到 11,525 rows/6,400,000 targets。epoch 尾部不足 accumulation 的窗口按实际 remainder 归一化，checkpoint 只能在 optimizer step 后保存。

SwanLab project 固定为 `MiniMind-Lab`；run names 区分 `Probe` 与 `Full`，artifact root 为 `/data/artifacts/minimind-lab/P03-dense-pretrain-v1-1b28-20260901/`。同 profile 的 stale 目录会阻断全新启动，避免把重跑结果拼接。指标同时报告 active-training throughput、训练循环 wall throughput 和 launcher 端到端 wall；checkpoint、validation、SwanLab 等等待只会进入相应 wall 口径，不污染 active window。

## 正式评测与验收

固定 Base 协议：lm-evaluation-harness 0.4.12（commit `6d642546f4688648fced259eb3302efd36ece5af`）、0-shot、无 chat template、batch 16、单卡 L20、seed 42；七项 C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag、Social IQA，共 29,638 samples/112,919 requests。导出必须 strict load 并记录 checkpoint/export SHA；另做 5 个固定 greedy continuation。

1. 技术完成：exit 0、9,038 updates、无 NaN/Inf/OOM/NCCL failure、strict export+SHA、validation 11,525 rows/6.4M targets、七项 29,638 samples/112,919 requests。
2. 实验接受：同一新 validation NLL ≤ P02；七项 macro ≥ 30.91%；任一任务相对 P02 不退化超过 2pp。P02 新 validation NLL 尚待同协议补测，写入前不能判定该门。
3. Release candidate：macro ≥ 31.44%；5 个 greedy 至少 3 个非空且无明显重复退化；median GPU util ≥ 95%。P02 的 379,311.32 token-slots/s 使用 `6,504,059,136 / 17,147.02s` 的 launcher 端到端口径，90% 为 341,380.19；P03 当前把 active、training-loop wall 和 launcher wall 分开报告，因此该数值只作参考，不作为“同口径”硬门，直至两者按同一 wall-clock 边界重算。

机器可读冻结项和 P01/P02 基线见 `config.json`。
