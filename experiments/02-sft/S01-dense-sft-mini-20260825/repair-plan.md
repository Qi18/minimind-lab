# S01 修复与重跑方案

## 1. 失败定义

`S01-dense-sft-mini-20260825` 的进程以 exit code 0 完成，但实验验收失败：没有固定 validation split 和 validation loss，瞬时 batch loss 方差较大，无法证明泛化收敛或选择 best checkpoint。原 checkpoint 只保留为诊断证据。

## 2. 先修观测口径

重跑前修改训练与数据代码：

1. 从 905,718 条 SFT 数据中用 seed 42 固定保留 10,000 条 validation；保存索引、行数和 manifest，训练集为 895,718 条。
2. validation 禁用随机 system prompt 注入和随机空 think 标签处理，确保相同 checkpoint 重复评测得到相同结果。
3. 跨 8 个 rank 聚合 `NLL sum / valid assistant label tokens`，同时记录有效 label token 数，避免把单个 batch loss 当成趋势。
4. 每 100 optimizer steps 记录 token-weighted train loss 均值、gradient norm、learning rate、有效 tokens/s 和峰值显存。
5. 每 500 optimizer steps 计算固定 validation loss；保存 `best_val` 与 `last` 两个 checkpoint，不再只覆盖一个文件。
6. SwanLab 统一记录 train/validation loss、有效 label tokens、gradient norm、吞吐、显存和 checkpoint step。

## 3. Batch probe

在完整重跑前，单独建立 batch probe，不使用 S01 正式 run：

| 每卡 batch | 8 卡 global batch | 目的 |
|---:|---:|---|
| 4 | 32 | 低风险参照 |
| 8 | 64 | 吞吐候选 |
| 16 | 128 | 首选正式候选，对齐当前源码默认 batch 值 |
| 32 | 256 | 只做 OOM/吞吐边界探针 |

每组固定 seq_len 768、BF16、P01 初始化和相同样本预算；预热 50 steps，正式测量 200 steps。选择条件：

- 无 OOM、NaN、Inf 和 NCCL 错误；
- 峰值显存不超过 38 GiB，保留至少约 8 GiB 安全余量；
- 相比更小 batch 的有效 tokens/s 有实际提升；
- gradient norm 和短程 validation loss 无异常。

旧 S01 每卡 batch 2 的峰值显存只有 3,815 MiB，因此 batch 16 是首选候选，但必须通过 probe 后冻结，不能直接假设安全。

## 4. 正式重跑候选配置

实验 ID 使用新目录，禁止覆盖原 S01：`S01R1-dense-sft-mini-20260825`。

```text
initialization       P01-dense-pretrain-mini-20260824
world_size           8
per_gpu_batch        16（以 batch probe 结果为准）
global_batch         128
accumulation_steps   1
epochs               2
max_seq_len          768
learning_rate        1e-5
dtype                bfloat16
seed                 42
train_rows           895718
validation_rows      10000
eval_interval        500 optimizer steps
checkpoint_policy    best_val + last
```

如果 batch 16 不满足显存或吞吐门槛，回退到每卡 batch 8；不通过减少 validation、缩短 seq_len 或修改数据来掩盖问题。

按 batch 16 计算，每 epoch 约 6,998 个 optimizer steps，两轮约 13,996 steps。它是本项目明确选择的 8×L20 配置，不宣称为官方发布模型的严格训练配方。

## 5. 训练验收门

训练完成必须同时满足：

1. 100-step token-weighted train loss 趋势下降且无 NaN/Inf；
2. `best_val_loss` 低于训练前 P01 在同一 validation 集上的 loss；
3. 最终 validation loss 相对 best 不反弹超过 5%，否则选择 best checkpoint 并说明过拟合；
4. checkpoint SHA-256、SwanLab URL、Lab commit、MiniMind commit 和数据 split manifest 完整；
5. 固定 Chat/Tool 集相对 P01 有目标能力提升；
6. 带 chat template 的官方七项全部执行并报告相对 P01 的回归；
7. 只有上述验收完成，Stage 4 才标记 `completed`。

## 6. 对比边界

- 原 S01：仅作为失败诊断，不进入主结果表。
- S01R1：与 P01 做目标能力和通用能力回归对比。
- MiniMind 官方：只按 README 的七项任务和调用口径横向比较；不因为使用源码默认 batch 就声称严格复现官方发布模型。
