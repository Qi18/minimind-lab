# MiniMind Lab 最终实验报告

## 1. 项目背景与目标

待填写。

## 2. 环境与数据

本节先记录已经闭环的 P02/P03 预训练数据对比。两次实验均训练 63,912,192 参数的 64M Dense 模型，使用 8×NVIDIA L20、bfloat16、seq 768、全局 sequence batch 256 和 1 epoch。P03 同时改变了数据组成、tokenizer 对齐切块、独立 validation、训练入口和 micro-batch 分解，因此它是“新数据管线/新基线”对 P02 的整体比较，不是只改变一个变量的严格数据 A/B。

| 维度 | P02：官方 full 单文件 | P03：pretrain-v1-1b28 | P02 / P03 |
|---|---:|---:|---:|
| train 行数 | 8,468,827 | 2,313,483 | 3.66× |
| shifted loss targets | 2.021B | 1.280B | 1.58× |
| padded compute slots | 6.504B | 1.777B | 3.66× |
| 平均 targets / row | 238.61 | 553.28 | 0.43× |
| target 利用率 | 31.07% | 72.04% | 0.43× |
| optimizer updates | 33,082 | 9,038 | 3.66× |
| 估算 targets / update | 61.1K | 141.6K | 0.43× |
| training-visible exact duplicates | 1,433 | 0 | — |
| benchmark query containment | 115 | 0 | — |
| 训练期独立 validation | 无 | 11,525 行 / 6.4M targets | — |
| 行级来源与处理 provenance | 无 | 完整 sidecar | — |

P02 的 8,468,827 行并不等价于 3.66 倍有效训练数据：其有效 target 只是 P03 的 1.58 倍，而且逐行 padding 使 68.93% 的固定计算槽位不参与 loss；P03 经 tokenizer 对齐切块后，target 利用率从 31.07% 提升到 72.04%。P02 另有 331,996 行被截断，丢弃 188,086,250 个 raw tail tokens。P03 的 1.280B targets 来自 ChineseWebText2 45%、FineWeb-Edu 30%、FineMath 10%、Wikipedia zh/en 各 5%、Stack v3 permissive code 5%，并保留来源和处理 sidecar。

污染扫描使用相同的 7 项 benchmark、29,322 个唯一 query，以及 `NFKC + lowercase + Unicode alnum` 归一化。P02 检出 115 个 query-in-document containment pattern-document pair，其中 ARC-Easy 72、CMMLU 8、OpenBookQA 21、PIQA 14；其中既包含较明确的题干级重合，也有通用短语碰撞，不能把 115 个命中全部断言为确定泄漏。P02 与 P03 的共享 validation 均无 exact tokenized-chunk overlap。near duplicate 仅对两者各抽样 20,480 行，样本中均为 0，不能外推为全量 0。

数据证据边界：P02 审计固定原始 revision `312afb4f76391145c6902f765bb51691c09a12f5`，原始文件 SHA-256 为 `31efc9a6fa7430769c0e78cde1c8ec0273ac7bbad20614c0ee58bccef327cc9d`；P03 dataset fingerprint 为 `cd018f6d0a047284f5f77d240d2583a1673c9d9a923536e9da7e4b1e4ead70bd`，共享 validation SHA-256 为 `0a7e8503f01bc185740b3e26e26326c43ca00309452ae1eec081d2ac2d9105cb`。

## 3. 模型结构

待填写参数量、关键模块、Dense/MoE 边界和源码证据。

## 4. 训练方案

待填写 Pretrain、SFT/LoRA、DPO、GRPO/CISPO 和 Agentic RL。

## 5. 对照实验

P02/P03 均正常跑完 1 epoch，但成本差异显著：

| 指标 | P02 | P03 | P03 相对 P02 |
|---|---:|---:|---:|
| wall-clock | 285.78 min | 46.37 min | -83.8% |
| 8 卡合计 GPU-hours | 38.10 | 6.18 | -83.8% |
| optimizer updates | 33,082 | 9,038 | -72.7% |
| 最后 100 个日志点 train loss 均值 | 1.7157 | 2.6080 | 不可直接按 step 比较 |
| padded token-slot throughput 上界 | 379,311/s | 638,745/s | +68.4%（仅参考） |
| GPU 利用率中位数 | 99% | 100% | +1 pp |

P02/P03 的 loss 横轴承载的有效 target 不同，training loss 也来自不同训练分布，因此 P02 更低的末段 training loss 不能证明泛化更好。吞吐也只统计含 padding 的 token slots，且 P02 使用端到端 wall、P03 使用独立计时边界；`+68.4%` 只能作为系统效率信号，不能当作严格同协议测速结论。

## 6. 统一评测结果

### 6.1 共享 validation

两份 checkpoint 均在 P03 固定的 11,525 行、6.4M target validation 上，以相同 MiniMind shifted next-token CE 口径回溯评测：

| Checkpoint | validation NLL | PPL | 相对 P02 |
|---|---:|---:|---:|
| P02 | 3.19096 | 24.3117 | 基线 |
| P03 | **2.60432** | **13.5221** | NLL -0.58664（-18.4%）；PPL -44.4% |

P03 在少 36.7% 有效 train targets、少 72.7% optimizer updates 的情况下取得更低的共享 validation NLL，说明本轮最明确的收益来自数据利用率与数据质量，而不是“见过更多行”或更低 training loss。该 validation 在 P02 训练后才建立，因此这是回溯式 checkpoint 对比，不能补写成 P02 当时的训练期 validation。

### 6.2 七项 Base 0-shot benchmark

统一协议为 `lm-evaluation-harness 0.4.12`、commit `6d642546f4688648fced259eb3302efd36ece5af`、0-shot、batch 16、seed 42、单张 L20，不使用 chat template；任务提供 `acc_norm` 时使用 `acc_norm`，否则使用 `acc`。

| 任务 | P02 | P03 | P03 - P02（百分点） |
|---|---:|---:|---:|
| C-Eval | 23.03 | 23.18 | +0.15 |
| CMMLU | 25.13 | 25.63 | +0.51 |
| ARC-Easy | 28.91 | 31.65 | +2.74 |
| PIQA | 51.47 | 51.90 | +0.44 |
| OpenBookQA | 26.60 | 26.00 | -0.60 |
| HellaSwag | 28.40 | 28.00 | -0.40 |
| Social IQA | 32.80 | 34.29 | +1.48 |
| 七项宏平均 | 30.91 | **31.52** | **+0.62** |

P03 在 5/7 个任务上提升，最大增益来自 ARC-Easy（+2.74 pp）和 Social IQA（+1.48 pp）；OpenBookQA 与 HellaSwag 分别回退 0.60/0.40 pp。宏平均提升说明整体方向有效，但不能掩盖单项回退，也不能仅凭七项均值判断生成质量。

### 6.3 固定续写

P02 的 5 个 greedy Base 续写全部首 token 输出 EOS；P03 为 4/5 非空、0 个立即 EOS，消除了“立即结束”退化，但 4 个非空样例仍出现词语或空白重复。因此 P03 改善了可生成性，却尚未形成可用的对话/指令能力，后续仍需 SFT 并单独评测。

### 6.4 Phase 2 SFT 收口

Phase 2 先建立官方数据基线，再通过等预算 A/B、行为修复课程和独立 IFEval 收口。官方完整 SFT 实际约含 2.4007B assistant targets，并非原计划的 32M；虽然 loss 收敛，但固定集只有 Chat 2/10、格式 0/6、重复异常 7/10、Tool E2E 5/8，说明同分布 loss 不能替代能力评测。

S03/S04 在 8M assistant-target 预算下比较官方与自建 pilot，两者均未过行为门，当前自建通用数据没有独立收益。S07/S08 依次修复低唯一性与模板多样性，S09 step 400 才通过固定行为门；随后 S10 使用 41,720 条、4.163M assistant targets 的 verifier-backed 指令课程，在与冻结 IFEval prompt 零 exact/normalized overlap 的前提下进行独立验收。

| Checkpoint | IFEval prompt strict | instruction strict | Chat | 格式 | 重复异常 | Tool E2E | 七项 macro（chat template） |
|---|---:|---:|---:|---:|---:|---:|---:|
| P03 | 10.54% | 23.02% | 0/10 | 0/6 | 9/10 | 0/8 | 未按同协议纳入此表 |
| S09 step 400 | 11.09% | 22.30% | **10/10** | **6/6** | **0/10** | **7/8** | 32.9926% |
| **S10 release** | **17.38%** | **29.98%** | **8/10** | **5/6** | **1/10** | **7/8** | **33.0024%** |

S10 相对 S09 的 IFEval prompt strict 提升 6.29pp（60/541 → 94/541），instruction strict 提升 7.68pp，七项 macro 基本不变；因此选择 S10 为 S★。其状态是 `accepted with limitations`：提升集中在 detectable content、format、punctuation 和 case，language 无提升，keywords 与 length constraints 分别回退 3.1pp 和 3.5pp，固定行为也较 S09 有轻微回退但仍过门。

结论边界：S10 证明 verifier-backed 课程可以提升同类可验证指令遵循，不能宣称开放域对话或所有通用能力全面提升。后续 LoRA、DPO、GRPO/CISPO 与 Agentic RL 默认从 S10 独立分叉。详细记录见 `docs/phases/phase2-sft.md` 和 `experiments/02-sft/S10-ifeval-curriculum-v4-20260907/report.md`。

## 7. 失败实验与问题排查

待填写失败现象、根因、证据和改进。

## 8. 核心结论

1. P02 的“行数更大、training loss 更低”没有转化为更好的泛化：其有效 target 利用率仅 31.07%，且存在 exact duplicate 与 benchmark containment 命中。
2. P03 用 1.280B targets 和 9,038 updates，将共享 validation NLL 从 3.19096 降到 2.60432，七项宏平均从 30.91% 提升到 31.52%，训练 wall 从 285.78 分钟缩短到 46.37 分钟。
3. 目前最可信的结论是“P03 整体数据管线优于 P02 基线”，不能归因到某一个数据源或单一超参数；要做因果归因，需固定训练入口与 token budget，对 source mix、packing 和去重分别消融。
4. P03 的 Base 生成仍有明显重复，但 Phase 2 已用 S10 完成 SFT 收口：IFEval prompt strict 较 S09 提升 6.29pp，七项 macro 稳定，行为门继续通过；该结论仅适用于可验证指令遵循，不能外推为开放域 Chat 全面提升。

证据位置：P02 正式结果与数据审计位于 `experiments/01-pretrain/P02-dense-pretrain-full-20260824/`；P03 正式结果位于 `experiments/01-pretrain/P03-dense-pretrain-v1-1b28-20260901/`，registry 状态为 `completed`；Phase 2 权威报告位于 `docs/phases/phase2-sft.md`，S10 的配置、命令、checkpoint manifest 和报告位于 `experiments/02-sft/S10-ifeval-curriculum-v4-20260907/`，完整原始日志和逐题评测保留在 CPFS artifacts。

## 9. 发布资产

待填写 GitHub commit、SwanLab、Hugging Face、博客和复现命令。

## 10. 简历项目描述

仅使用能够链接到上述实验证据的指标。


## Phase 3：领域适配对照收尾（2026-09-08）

| 指标 | L00：S10 基线 | L01：Full FT | L02：LoRA r16 |
|---|---:|---:|---:|
| MBPP pass@1 | 0.40% (2/500) | 0.00% (0/500) | 0.00% (0/500) |
| 七项 macro | 33.0024% | 32.4759% | 32.6742% |
| IFEval prompt strict | 17.3752% | 17.9298% | 19.0388% |
| IFEval instruction strict | 29.9760% | 30.2158% | 31.4149% |
| Chat | 8/10 | 7/10 | 8/10 |
| 格式 | 5/6 | 3/6 | 5/6 |
| Tool E2E | 87.5% | 75.0% | 75.0% |
| 代码提取为空 | 336/500 | 424/500 | 394/500 |
| 领域 validation NLL（训练 DDP 口径） | 0.974555 | 0.757733 | 0.837637 |
| 可训练参数 | 0 | 63,912,192 | 393,216（含 adapter 总参数的 0.6115%） |
| 训练 wall | — | 94.65 s | 63.46 s |
| 训练 GPU-hours（6 卡） | — | 0.15775 | 0.10576 |
| 峰值 allocated 显存 | — | 6,936.51 MiB | 4,464.24 MiB |
| 实际 assistant targets | — | 4,820,231 | 4,820,231 |

LoRA本次耗时/显存分别降低约33%/36%，但两组MBPP均无提升；保留S10。不同LR与单种子限制不支持纯方法因果结论。数据构建、训练、评测和失败分析见 [Phase 3报告](phases/phase3-lora.md)。
