# Phase 2：SFT 阶段报告

## 当前判定

**状态：accepted with limitations（2026-09-07）。** Phase 2 已收口，release checkpoint 为 `S08-ifeval-curriculum-v4-20260907`。它在 S07R2 已通过行为门禁的基础上，将独立 IFEval prompt strict 从 11.09% 提升到 17.38%（+6.29pp），七项通用能力 macro 保持在 33.00%，并继续通过 Chat、格式、重复和 Tool 门槛。

“有局限”表示：IFEval 绝对分数仍低，提升集中在可检测内容、格式和标点；语言切换没有改善，关键词和长度约束略有回退。S08 相对 S07R2 的固定行为集也从 Chat 10/10、格式 6/6、重复 0/10 轻微回退到 8/10、5/6、1/10，但仍满足预注册门槛。不能把本阶段写成“通用对话能力全面提升”。

## 实验演进

| 模型 | 数据与预算 | Chat | 格式 | 复读异常 | Tool E2E | 七项 macro |
|---|---|---:|---:|---:|---:|---:|
| P03 | Pretrain base | 0/10 | 0/6 | 9/10 | 0/8 | 31.5228% |
| S01-formal | 官方完整 SFT，约 2.4007B assistant targets | 2/10 | 0/6 | 7/10 | 5/8 | 31.8668% |
| S05A | 官方 accepted pilot，8M | 0/10 | 0/6 | 10/10 | 0/8 | 31.7673% |
| S05B | 自建 SFT-v1 pilot，8M | 0/10 | 0/6 | 7/10 | 1/8 | 31.6290% |
| S07 | 低唯一性能力修复 v1 | 2/10 | 0/6 | 3/10 | 6/8 | 未跑 |
| S07R1 | 多样化能力修复 v2 | 3/10 | 1/6 | 3/10 | 5/8 | 未跑 |
| S07R2 step 400 | 目标概念改写 + 原生工具轨迹 | **10/10** | **6/6** | **0/10** | **7/8** | 32.9926%（chat template） |
| **S08 release** | **41,720 条 verifier-backed IFEval 课程** | **8/10** | **5/6** | **1/10** | **7/8** | **33.0024%（chat template）** |

S07R2 与 S08 的七项数值使用同一 `--apply_chat_template` 协议；P03、S01、S05 的历史数字来自各自阶段记录，不能把不同协议下的小数点差异解释为模型收益。

## 独立 IFEval 验收

统一协议：lm-eval 0.4.12、IFEval v4、541 prompts、0-shot、seed 42、greedy、`--apply_chat_template`。

| 模型 | Prompt strict | Prompt loose | Instruction strict | Instruction loose |
|---|---:|---:|---:|---:|
| P03 | 10.54% | 11.28% | 23.02% | 23.98% |
| S01 official full | 10.72% | 12.20% | 22.30% | 23.86% |
| S07R2 | 11.09% | 11.83% | 22.30% | 23.98% |
| **S08** | **17.38%** | **19.78%** | **29.98%** | **32.97%** |

S08 相对 S07R2 新增 34 条 prompt strict 通过（60/541 → 94/541）。训练、validation、test 与冻结 IFEval 的 exact/normalized prompt overlap 均为 0；但数据使用了相同 verifier 语义，因此这是 benchmark-family-targeted 泛化证据，不是开放域 Chat 盲测。

## Release 资产

- checkpoint：`/data/artifacts/minimind-lab/S08-ifeval-curriculum-v4-20260907/checkpoints/s08_best_val_768.pth`
- SHA-256：`46aeab66795795aa77d703f08d71b952fe98461040f4560e1301021540710131`
- 训练 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/bxdob3rh>
- 评测 SwanLab：<https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/encs5zuk>
- 原始训练与逐题评测：`/data/artifacts/minimind-lab/S08-ifeval-curriculum-v4-20260907/`
- 复现配置、命令和报告：`experiments/02-sft/S08-ifeval-curriculum-v4-20260907/`

## 结论

- 官方全量 SFT 的 loss 收敛没有转化为足够的对话与格式能力，说明同分布 validation loss 不能替代能力评测。
- S05A/S05B 的等预算对照表明，当前自建通用 pilot 没有优于官方 pilot；真正有效的是后续对失败能力进行可验证、可审计的定向课程设计。
- S07R2 证明短答案、格式、重复抑制和工具行为可被定向教会；S08 又在零 prompt 重叠的 IFEval 上取得 +6.29pp strict 提升，同时七项 macro 稳定。
- 后续 Full FT vs LoRA、DPO、GRPO/CISPO 和 Agentic RL 默认从 S08 release 分叉。若专门追求更高 IFEval，再做 S08R1，但它不是进入下一阶段的阻塞项。
- 所有训练和评测均记录在 SwanLab 项目 `MiniMind-Lab`，原始产物保留在 CPFS。

详细证据见 `experiments/02-sft/S08-ifeval-curriculum-v4-20260907/report.md` 与 `experiments/02-sft/S07R2-targeted-chat-curriculum-20260907/report.md`；S05 对照见 `experiments/02-sft/S05A-dense-sft-official-pilot-8m-20260904/report.md` 和 `experiments/02-sft/S05B-dense-sft-custom-pilot-8m-20260904/report.md`。
