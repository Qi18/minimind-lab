# 训练前准备实验

这里统一保存正式训练前的环境、数据和模型运行时验证。它们回答“能否可信地开始训练”，不作为模型能力结果。

| 顺序 | 实验 | 目的 | 完成门槛 |
|---|---|---|---|
| Stage0 | [E00 L20 baseline](E00-l20-baseline-20260823/report.md) | 验证 8×L20、BF16、NCCL、CPFS、`/dev/shm` 和 SwanLab | 环境 smoke test 通过 |
| Stage1 | [E01 Tokenizer/Dataset](E01-tokenizer-dataset-20260823/report.md) | 固定 Tokenizer，审计数据格式、规模、截断和 padding | 数据 manifest 与质量报告完成 |
| Stage2 | [E02 Model Probe](E02-model-probe-20260823/report.md) | 验证 Dense/MoE、DDP、BF16、SwanLab 和断点恢复 | 100-step 三 seed probe 与 resume 通过 |

目录约定：

```text
00-preparation/
├── E00-l20-baseline-20260823/
├── E01-tokenizer-dataset-20260823/
└── E02-model-probe-20260823/
```

准备阶段通过后，正式训练从 [`../01-pretrain/`](../01-pretrain/) 的 P01 开始。
