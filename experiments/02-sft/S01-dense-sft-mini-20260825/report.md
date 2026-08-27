# S01-dense-sft-mini-20260825

## Status

`invalidated`。训练进程正常完成，但配置和观测口径不足以证明收敛。本 checkpoint 仅保留为诊断证据，不进入发布、官方对比主表或后续阶段初始化。

## Objective

从 P01 的 64M Dense Base checkpoint 出发，用 `sft_t2t_mini.jsonl` 运行一次自定义 8×L20 全参数 SFT。每卡 batch 2、1 epoch 是本项目当时选择的保守配置，并非 MiniMind 官方训练配方。

## Direct baseline

- `P01-dense-pretrain-mini-20260824`
- Base checkpoint SHA-256：`71efd40d9fcd494bc5472891b66ea7f17167ae27ac341968bcd258a5a24b94e9`
- P01 官方七项任务口径宏平均：31.4449；该结果不是官方权重的严格复现。

## Configuration

- Hardware：8×NVIDIA L20 46GB
- Model：64M Dense，hidden 768，8 layers，seq_len 768
- Dataset：905,718 rows，SHA-256 `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`
- Training：1 epoch，每卡 batch 2，累积 1，global sequence batch 16，BF16，lr 1e-5
- Completed：56,608 microsteps / optimizer updates
- SwanLab：project `MiniMind-Lab`，run `S01-SFT-Mini-64M-Seq768`
- SwanLab URL：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/p2ttzc7ycn5tpaegt4odo>

## Training result

- Started：`2026-08-25T13:05:26Z`
- Finished：`2026-08-25T14:10:51Z`
- Exit code：0
- Wall clock：3,925 seconds（65m25s）
- GPU-hours：8.7222
- Mean GPU utilization：97.069%
- Peak GPU memory：3,815 MiB / 46,068 MiB
- 前 100 个日志点 loss：mean 2.0330，std 0.3406
- 最后 100 个日志点 loss（排除最后异常单点）：mean 1.6496，std 0.4055
- 分段均值：1–10k 为 1.8650；10–20k 为 1.7356；20–30k 为 1.6840；30–40k 为 1.6873；40–50k 为 1.6571；50k–结束为 1.6470
- 最后单 batch loss 0.0254 来自尾部小批次，不作为收敛或模型质量结论

## Evaluation result

未进入正式模型质量评测。缺少预先冻结的 validation split 和 validation loss，无法选择 best checkpoint，也无法判断后段训练是否泛化改善。

## Cost and system metrics

详见 `metrics.csv` 和 CPFS 原始日志。checkpoint SHA-256 为 `50859557d6fb03604a005c7e3aca6d8ab7ce41780f0c94fb7691cc18ede70419`，角色标记为 `diagnostic-invalidated-not-for-release`。

## Failures and anomalies

1. 每卡 batch 2 只占峰值 3,815 MiB，未合理利用 L20 显存，单 batch loss 方差较大。
2. 训练只记录瞬时 batch loss，没有 100-step token-weighted 均值、有效 label token、gradient norm 或 validation loss。
3. SFTDataset 的训练期随机 system prompt / think 标签处理没有与确定性的 validation 路径分离。
4. 1 epoch / batch 2 的来源曾被错误描述为官方历史配方；该说法没有官方依据，现已撤回。

## Conclusion

聚合 train loss 确实从约 1.865 降至 1.647，但后半程改善有限且噪声没有下降；更关键的是没有 validation loss，因此不能确认泛化收敛。按项目验收决定，本次训练作废。修复方案见 `repair-plan.md`。
