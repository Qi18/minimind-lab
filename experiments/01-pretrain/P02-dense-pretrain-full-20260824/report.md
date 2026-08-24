# P02-dense-pretrain-full-20260824

## Objective

在 8×NVIDIA L20 上从随机权重训练同一个 64M Dense 模型，只把 P01 的 `pretrain_t2t_mini` 替换为完整 `pretrain_t2t`，测量数据规模对 Base 能力、生成退化和单位成本的影响。

## Direct baseline

直接基线是 `P01-dense-pretrain-mini-20260824`。P01 七项 Base 宏平均为 31.44，训练耗时 45.83 分钟，固定续写仍存在立即 EOS 和重复字符。

## Configuration

- 模型、随机初始化、seed、dtype、batch、梯度累积、seq 768、学习率、epoch 和 8×L20 均与 P01 相同；
- 唯一主要变量：数据从 1,270,238 行 mini 扩大到 8,468,827 行 full，行数为 6.67 倍；
- 虽然源码文档对 full 数据建议 `max_seq_len≈380`，本次保留 P01 的 768，以避免同时改变数据规模和截断长度；
- 预计 264,651 个 micro-steps、约 33,082 次 optimizer update；按 P01 线性外推约 305.5 分钟。

## Data verification

- revision：`312afb4f76391145c6902f765bb51691c09a12f5`；
- 文件大小：8,275,074,893 bytes；
- 行数：8,468,827；
- SHA-256：`31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d`，与官方 LFS oid 一致；
- 首尾记录字段均为 `text`。

## Training result

待正式训练后填写。训练启动前必须通过 L20 preflight、GPU 空闲检查、单实例锁、数据大小/行数/SHA256 校验和 dry-run。

## Evaluation result

训练结束后完全复用 P01 的固定 Base 续写与七项 0-shot 评测协议，并逐项报告 P02-P01 差值；不使用 chat template。

## Cost and system metrics

待从 `driver.log`、SwanLab 和 10 秒 `nvidia-smi` 采样汇总 wall-clock、GPU-hours、loss、GPU 利用率、显存、功耗和 token-slot 吞吐。

## Failures and anomalies

首次正式启动在进入第一个训练 step 前失败：8 个 rank 使用默认 `/root/.cache/huggingface/datasets` 并发生成 full JSON 的 Arrow cache，rank 7 报 `No space left on device`。GPU 尚未进入训练，未生成 checkpoint。

修复为强制 cache 位于 CPFS `/data/cache/huggingface/datasets`，并在 torchrun 前通过单进程预构建并验证 8,468,827 行共享 Arrow cache。只删除了本次失败产生的 217,068,495-byte 临时 cache，历史 cache 未动。

## Conclusion

待训练和评测完成后填写。P02 只有在 checkpoint、SwanLab、固定续写、七项评测和报告全部闭环后才标记为 completed。
