# MiniMind 完整训练与源码学习计划

更新时间：2026-08-21

## 0. 文档定位与当前事实

这份计划不是功能清单，而是 MiniMind Lab 的执行主线。每个阶段都必须同时完成源码阅读、最小观察实验、正式训练、统一评测和文档沉淀，满足阶段门后才进入下一阶段。

当前可确认的事实：

- 实验仓库：`Qi18/minimind-lab`。
- L20 权威工作区：`/data/projects/minimind-lab`。
- MiniMind 源码位于普通目录 `minimind/`，初始上游 commit 为 `393e387e9ad99f0f04c296e4c5e7353f4444629f`。
- 目标硬件为 `8 × NVIDIA L20`；每次正式运行前仍需重新确认 GPU 空闲、CUDA、NCCL、CPFS、`/dev/shm` 和网络状态。
- 训练脚本参数名仍叫 `--use_wandb` / `--wandb_project`，但源码实际执行 `import swanlab as wandb`，因此本项目统一使用 SwanLab。
- 2026-08-08 的 Official Zero 记录只是历史快照：当时 Pretrain 曾中断并从 step 4000 恢复，不能据此宣称训练已经完成或当前仍在运行。
- 旧记录中的数据文件、checkpoint、SwanLab run 和日志必须重新盘点后才能登记到本仓库。

所有配置分为两类：

1. **代码默认值**：固定在当前 MiniMind 源码中的 argparse 默认值。
2. **Lab 实验值**：为 8×L20 对照实验主动选择的值，必须写入 `config.json` 和 `command.sh`。

不得把代码默认值、历史运行值和新实验值混为一谈。

## 1. 项目目标

### 1.1 核心目标

- 从随机初始化完成 64M Dense 的 Pretrain → Full SFT 闭环。
- 比较 mini/full 数据对 Base 与 Chat 能力的真实影响。
- 在同一 Full SFT 基线上完成 DPO、PPO、GRPO、CISPO 和 Agentic RL 对照。
- 完成 Dense 与 198M-A64M MoE 的结构、速度和效果对照。
- 完成一次白盒蒸馏实验，理解 `CE + KL`、温度和 teacher/student 组合。
- 建立统一评测、SwanLab、checkpoint、HF 和报告证据链。
- 将训练过程与源码阅读同步沉淀为博客和可量化的简历项目。

### 1.2 暂不作为主线的目标

- 不在 64M 基线闭环前启动 1B 训练。
- 不为了覆盖功能而串联所有后训练权重。
- 不把训练 reward、loss 下降或主观样例当成完整评测。
- 不在没有领域数据与验收集时宣称完成垂域模型。
- 不自动同步 MiniMind 上游最新 commit；源码升级必须单独评审。

## 2. 模型路线

| 路线 | 规模 | 结构 | 用途 | 优先级 |
|---|---:|---|---|---|
| Dense 主线 | 约 64M | 8 层、hidden 768、8 Q heads、4 KV heads | 完整训练、后训练和评测主线 | 必做 |
| MoE 对照 | 约 198M-A64M | 4 experts、top-1，无 shared expert | Router、aux loss、激活参数和调度开销 | 必做扩展 |
| 1B 扩展 | 约 1B | 待单独设计 | 验证规模放大 | 远期门控 |

Dense 与 MoE 共同使用官方 `len_vocab=6400`、`max_pos=32768`、`rope_theta=1e6` 的结构基线。实验使用的 `max_seq_len` 是训练截断长度，不等于模型最大位置长度。

官方源码说明原生 PyTorch 的 4 experts/top-1 MoE 经验上约比 Dense 慢 50%；该数字只能作为参考，必须在 L20 上重新测量。

## 3. 数据路线

| 文件 | 源码文档规模 | 用途 | 进入阶段 |
|---|---:|---|---|
| `pretrain_t2t_mini.jsonl` | 约 1.2GB | 快速 Pretrain 基线 | P01 |
| `pretrain_t2t.jsonl` | 约 10GB | 完整 Dense/MoE Pretrain | P02/M01 |
| `sft_t2t_mini.jsonl` | 约 1.6GB | 快速 SFT，含部分 Tool Call | S01 |
| `sft_t2t.jsonl` | 约 14GB | 完整主线 SFT，含 Tool Call | S02/M02 |
| `dpo.jsonl` | 约 53MB | chosen/rejected 偏好优化 | D01 |
| `rlaif.jsonl` | 约 24MB | PPO、GRPO、CISPO | R01–R03 |
| `agent_rl.jsonl` | 约 86MB | 多轮 Tool-Use Agentic RL | A01–A02 |
| `agent_rl_math.jsonl` | 约 18MB | 带最终校验目标的 Agent RL/RLVR | A03 |

历史计划记录 `pretrain_t2t_mini` 约 1,270,238 条、`sft_t2t_mini` 约 905,718 条；这些行数必须对 L20 的实际文件重新统计，并与文件 SHA-256 一起写入 data manifest。

数据分析必须记录：

- 文件大小、行数、SHA-256、来源 URL 和许可证；
- 字符长度、Token 长度、截断率和 padding 比例；
- 有效 label token 数量，而不是只计算 `样本数 × max_seq_len`；
- 重复样本、空样本、格式错误、超长样本和 train/eval 污染；
- Pretrain、SFT、DPO、RLAIF、Agent RL 的样本结构抽查。

`pretrain_t2t.jsonl` 是完整训练同一 64M Dense/MoE 主线的数据，不是专门给 1B 使用的数据。

## 4. 实验依赖图

```text
Tokenizer / Data Audit
          ↓
Model Forward Probe
          ↓
P01 Dense Pretrain mini ──→ S01 SFT mini ──→ Zero baseline eval
          ↓
P02 Dense Pretrain full ──→ S02 SFT full ───────────────┐
                                                   ├─→ Domain Full FT vs LoRA
                                                   ├─→ D01 DPO
                                                   ├─→ R01 PPO
                                                   ├─→ R02 GRPO
                                                   ├─→ R03 CISPO
                                                   ├─→ A01/A02 Agentic RL
                                                   └─→ Service / Unified Eval

M01 MoE Pretrain ──→ M02 MoE SFT ──→ K01 MoE teacher → Dense student distillation
```

DPO、PPO、GRPO、CISPO 和 Agentic RL 默认从同一个 `S02 Full SFT` checkpoint 独立分支。源码阅读顺序可以是 PPO → GRPO → CISPO → Agentic RL，但正式训练不能不加控制地串成一条权重链。

## 5. 全局实验规范

### 5.1 单变量原则

每个对照实验只主动改变一个主要变量：

- mini vs full：只改变数据规模；
- Dense vs MoE：保持数据、训练预算和评测一致；
- GRPO vs CISPO：固定初始化、数据、seed、rollout 和 reward；
- Torch vs SGLang：固定模型、prompt、生成参数和样本量；
- Full FT vs LoRA：固定领域数据、初始化权重和评测集。

### 5.2 Batch 口径

```text
global_sequence_batch = per_gpu_batch × world_size × accumulation_steps
tokens_per_optimizer_step ≈ global_sequence_batch × 有效平均token数
```

GPU 数量和梯度累积不会增加唯一训练数据量。报告 Token/parameter 时使用有效训练 Token，而不是最大序列槽位。

### 5.3 Seed

- 全量 Pretrain 至少固定 seed 42，并对 100-step probe 做 3 个 seed。
- DPO、GRPO/CISPO、Agentic RL 的关键结论尽量运行 seed 42/43/44。
- 资源不足只能单 seed 时，报告中必须明确，不给出稳定性结论。

### 5.4 实验完成定义

一次实验只有同时满足以下条件才是 `completed`：

1. 命令、配置、Lab commit、MiniMind source commit 和数据 manifest 完整；
2. SwanLab run 可访问且曲线连续；
3. checkpoint manifest 完整；
4. 目标评测和通用能力回归完成；
5. `report.md` 解释收益、成本、失败和适用边界。

## 6. Stage 0：环境、仓库与历史基线盘点

### 6.1 源码阅读

- 每个 `train_*.py` 的 argparse、DDP 初始化、optimizer、scheduler、checkpoint 和恢复逻辑；
- `trainer_utils.py` 的权重保存、resume step 和 SwanLab run id；
- `README.md` 中官方训练顺序和数据说明。

### 6.2 执行任务

- 验证 L20 8 张 GPU、CUDA、BF16、NCCL all-reduce、CPFS 和 `/dev/shm`；
- 验证 SwanLab 登录，但不把 API key 放入命令、日志或仓库；
- 盘点是否仍存在旧 mini 路线的数据、日志、checkpoint 和 SwanLab run；
- 为每个正式训练脚本加入 GPU 空闲门禁、单实例锁和 SIGTERM 安全退出；
- 建立 `scripts/launch/`、`scripts/sync/` 和实验目录模板。

### 6.3 旧内部试跑记录

旧笔记曾记录以下参数，但没有官方来源，不能称为官方配置或官方复现：

| 项目 | 旧内部试跑值 |
|---|---|
| 模型 | Dense 约 63.91M，hidden 768，8 层 |
| 硬件 | 8×L20 |
| dtype | BF16 |
| Pretrain | 每卡 batch 4，累积 8，seq 768，1 epoch |
| SFT | 每卡 batch 2，累积 1，seq 768，1 epoch |
| 数据 | `pretrain_t2t_mini` + `sft_t2t_mini` |

这些值只能作为内部试跑线索；每次正式实验必须重新说明配置依据并完成独立验收。

### 6.4 阶段门

- 环境 smoke test 通过；
- 当前 GPU 占用和其他训练任务已确认；
- 数据与历史资产盘点完成；
- 实验模板能够从 run id 定位到源码、曲线和 checkpoint。

## 7. Stage 1：Tokenizer 与 Dataset

### 7.1 源码阅读

- `minimind/trainer/train_tokenizer.py`；
- `minimind/model/tokenizer.json` 与 `tokenizer_config.json`；
- `minimind/dataset/lm_dataset.py` 中 PretrainDataset、SFTDataset、DPODataset、RLAIFDataset、AgentRLDataset；
- chat template、`<think>`、`<tool_call>`、assistant-only label mask。

### 7.2 观察实验

- 对中文、英文、代码、数学、新闻和领域数据计算压缩率；
- 抽取每类数据各 20 条，显示原始 JSON → template → input_ids → labels；
- 统计被截断和 padding 的 token；
- 验证 SFT 只在目标回答区域计算 loss；
- 检查 Agent 样本中的 messages、tools 和 gt。

### 7.3 Tokenizer 训练边界

官方 Tokenizer 是主线固定组件。`train_tokenizer.py` 只用于理解训练过程和做隔离实验；一旦更换 Tokenizer，现有权重、embedding、LM head 和所有阶段对照都失去可比性，因此不能直接替换主线 Tokenizer。

### 7.4 产物与阶段门

- `docs/source_reading/01-data-tokenizer.md`；
- data manifest 和数据质量报告；
- 能解释一条样本如何变为 `input_ids`、`labels`、mask 和有效 loss token。

## 8. Stage 2：模型结构与最小训练探针

### 8.1 源码阅读

- `minimind/model/model_minimind.py`；
- Embedding、RMSNorm、RoPE、GQA Attention、MLP、Residual、LM Head；
- Dense/MoE 分支、Router、top-k、aux loss；
- KV Cache 与训练/生成路径。

### 8.2 观察实验

- 打印每层关键 Tensor shape；
- 按模块拆解参数量；
- 运行单卡 forward/backward；
- 运行 8 卡 100-step probe；
- 记录 loss、tokens/s、samples/s、GPU 利用率和峰值显存；
- 检查一次 resume 后 loss、step、optimizer 和 SwanLab 是否连续。

### 8.3 阶段门

- 100-step probe 无 NaN、Inf、OOM、NCCL 错误；
- 能从 Token 输入解释到 logits/loss；
- 能说明 Dense 与 MoE 的实际计算差异；
- 完成 `docs/source_reading/02-model-architecture.md`。

## 9. Stage 3：Dense Pretrain mini（P01）

### 9.1 代码默认与 Lab 基线

当前代码默认是 `epochs=2`、每进程 `batch_size=32`、`accumulation_steps=8`、`max_seq_len=340`、`lr=5e-4`。P01 是自定义 8×L20 基线：通过每卡 batch 4 和累积 8 保持 global sequence batch 256，但 epoch 1 和 seq_len 768 均不同于源码默认值：

```bash
cd /data/projects/minimind-lab/minimind/trainer
torchrun --nproc_per_node=8 train_pretrain.py \
  --epochs 1 \
  --batch_size 4 \
  --accumulation_steps 8 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --dtype bfloat16 \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --from_weight none \
  --use_wandb \
  --wandb_project MiniMind-Lab
```

正式脚本还必须补充唯一实验 ID、日志路径、锁、GPU 空闲检查和退出处理，不能只复制上述命令。

### 9.2 评测

- train/validation loss 与 perplexity；
- 固定续写样例；
- C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag、Social IQA；
- wall-clock、tokens/s、GPU 利用率、显存、checkpoint 大小；
- padding 比例和实际训练 Token。

### 9.3 阶段门

- loss 曲线稳定且可恢复；
- 生成不再完全退化；
- Base 评测、系统指标和 checkpoint manifest 完成；
- 选出 `P01 release candidate` 后才能进入 S01。

## 10. Stage 4：Dense SFT mini（S01）与 MiniMind Zero 路线验收

### 10.1 训练基线

当前代码默认 `epochs=2/batch=16/seq=768/lr=1e-5`。在 8 卡 DDP 中，`batch` 是每进程值，直接使用默认值会得到 global batch 128。

`S01-dense-sft-mini-20260825` 使用的每卡 batch 2、累积 1、1 epoch 是本项目自定义配置，不是官方历史配方；该运行因缺少 validation loss、无法可靠判断收敛而作废。后续必须先运行 batch probe，再按实验目录中的 `repair-plan.md` 建立新实验目录重跑。

初始化必须是 P01 选中的 `pretrain` 权重，不能从随机权重训练 SFT。

### 10.2 源码阅读

- SFTDataset 的 chat template 与 label mask；
- `train_full_sft.py` 的 loss、resume 和保存；
- Tool Call 样本如何混入 SFT 主线；
- 为什么 full_sft 已具备基础工具调用格式，不再默认单独做 Tool SFT。

### 10.3 评测

- 中文/英文问答、多轮对话、格式遵循；
- `<think>` 与 `<tool_call>` 标签正确率；
- `scripts/eval_toolcall.py` 固定任务；
- Base 七项基准回归；
- P01 vs S01 的 loss、Chat 指标和遗忘对比。

### 10.4 阶段门

- 形成可交互 Zero 模型；
- Tool Call 基础格式可用；
- 明确 SFT 带来的目标收益和 Base 回归；
- 只有 mini Pretrain + mini SFT 的训练、validation、Chat/Tool 和通用回归全部通过，MiniMind Zero 路线才能标记为完成。

## 11. Stage 5：完整数据 Dense（P02/S02）

### 11.1 目标

用相同 64M Dense 验证 `pretrain_t2t` + `sft_t2t` 相比 mini 数据的收益。该阶段回答“数据规模是否值得”，而不是改变模型规模。

### 11.2 配置原则

- `pretrain_t2t` 源码文档推荐 `max_seq_len≈380`；先用 100-step probe 冻结每卡 batch 和累积；
- 尽量对齐 P01/P02 的 optimizer、学习率曲线和有效 Token 统计；
- S02 使用 `sft_t2t`，初始化为 P02；
- 不将 P01 后继续追加 full 数据与“从头 P02”混为同一个实验。

### 11.3 必做对照

| 对照 | 固定项 | 变量 | 结论 |
|---|---|---|---|
| P01 vs P02 | 模型、优化器、评测 | Pretrain 数据规模 | Base 能力/成本 |
| S01 vs S02 | 模型、评测口径 | SFT 数据规模 | Chat/Tool/遗忘 |
| mini/full Token 效率 | 硬件、dtype | 数据与 seq | tokens/s、有效 Token |

### 11.4 阶段门

S02 完成统一评测并成为后训练共同基线。若 full 数据没有稳定收益，先排查截断、学习率、训练 Token 和数据质量，不直接进入 1B。

## 12. Stage 6：领域 Full FT 与 LoRA（可选但建议）

LoRA 不能直接拿来与主线 Full SFT 比，因为两者训练目标不同。正确对照是：从同一个 S02 权重出发，在同一领域数据上比较全参数继续 SFT 与 LoRA。

进入条件：

- 明确领域，例如医疗、个人知识或业务问答；
- 有 train/validation/test 切分；
- 有通用回放数据或遗忘控制；
- 有领域验收指标。

记录可训练参数、显存、训练时间、权重大小、领域收益和通用回归。没有领域数据时只阅读 `model_lora.py` 和做 smoke test，不宣称完成垂域适配。

## 13. Stage 7：DPO（D01）

### 13.1 源码阅读

- DPODataset 的 chosen/rejected 构造；
- policy/ref log-prob；
- DPO loss、beta 和长度偏差；
- 为什么 DPO 是静态 off-policy 偏好优化，不等于在线探索。

### 13.2 代码默认基线

- `epochs=1`；
- 每卡 `batch_size=4`；
- `lr=4e-8`；
- `max_seq_len=1024`；
- `beta=0.15`；
- 初始化 `full_sft`。

先使用默认值跑 smoke，再决定是否做 beta `{0.05, 0.15, 0.3}` 小消融。

### 13.3 评测与阶段门

- preference accuracy/win rate；
- chosen/rejected margin；
- 输出长度和格式偏差；
- Chat/Tool 目标评测；
- Base 七项基准回归。

只有证明收益不是长度投机，并报告遗忘后，D01 才完成。

## 14. Stage 8：PPO → GRPO → CISPO（R01–R03）

### 14.1 学习顺序

```text
Reward Model / Reference / Critic
→ PPO clipped objective + GAE
→ GRPO group-relative advantage
→ CISPO token-level ratio clipping
→ Rollout Engine（Torch / SGLang）
```

### 14.2 PPO（R01）

阅读 `train_ppo.py` 中 Actor、Critic、Reference、Reward Model、GAE、value clipping 和多轮 update。代码默认：

- actor lr `3e-7`，critic lr `5e-7`；
- batch 2，mini-batch 2，update iters 2；
- `clip_epsilon=0.2`、`vf_coef=0.5`、`kl_coef=0.02`；
- `gamma=1.0`、`lambda=0.95`；
- `early_stop_kl=0.25`；
- `rlaif.jsonl`、Full SFT 初始化。

必须记录 reward、KL_ref、approx_KL、clipfrac、critic_loss、actor/critic lr 和平均输出长度。

### 14.3 GRPO/CISPO（R02/R03）

固定 Full SFT、rlaif、seed、num_generations、reward model、rollout 和生成参数，只改变 `loss_type`。

代码默认：batch 2、lr `3e-7`、num_generations 6、beta 0.1、epsilon 0.2、epsilon_high 5.0。`train_grpo.py` 默认 `loss_type=cispo`，所以 GRPO 必须显式传 `--loss_type grpo`。

记录 reward、KL_ref、advantages mean/std、policy loss、平均输出长度和真实任务准确率。

### 14.4 Torch vs SGLang

先用 Torch rollout 验证算法正确性，再把完全相同的模型和 prompt 切到 SGLang。比较：

- rollout samples/s；
- 训练 step wall-clock；
- GPU 显存；
- policy 更新可见性；
- 输出与 log-prob 一致性。

### 14.5 阶段门

- reward 上升必须对应真实任务指标上升；
- KL、长度和 reward std 不出现持续退化；
- PPO、GRPO、CISPO 的收益、稳定性和成本可以横向解释；
- Base 与 SFT 能力回归已经量化。

## 15. Stage 9：MoE（M01/M02）

### 15.1 源码阅读

- Router logits、top-1 expert selection、token dispatch；
- aux load-balancing loss；
- 总参数量与激活参数量；
- 为什么原生 PyTorch MoE 可能比 Dense 更慢。

### 15.2 实验

- mini 数据先完成 M01/M02 smoke 和正式基线；
- 与 P01/S01 固定数据、seq、评测和硬件；
- 记录 expert token 分布、负载不均、aux loss；
- 再决定是否在 full 数据上运行 MoE。

### 15.3 阶段门

报告 Dense/MoE 的质量、吞吐、显存、总参数、激活参数和训练稳定性。不能只用“参数更多”或“输出更流畅”作为结论。

## 16. Stage 10：知识蒸馏（K01）

### 16.1 路线

- 黑盒蒸馏：教师生成数据后做 SFT，主线 SFT 数据已经包含此类信号；
- 白盒蒸馏：`train_distillation.py` 使用 teacher token 分布，优化 `alpha × CE + (1-alpha) × KL`。

### 16.2 推荐实验

- Teacher：M02 MoE Full SFT；
- Student：S02 Dense Full SFT；
- 数据：固定 SFT validation/training 子集；
- 默认 alpha 0.5、temperature 1.5；
- 消融：CE-only、alpha 0.5、temperature `{1.0, 1.5, 2.0}`。

记录 loss、ce_loss、distill_loss、aux_loss、训练成本，以及学生在统一评测上的变化。

### 16.3 阶段门

蒸馏后学生必须在至少一组目标指标上优于未蒸馏 S02，并量化通用能力、速度和权重大小；否则结论是“当前 teacher/data/温度组合无有效收益”。

## 17. Stage 11：Agentic RL（A01–A03）

### 17.1 源码阅读

- AgentRLDataset 的 messages、tools 和 gt；
- `train_agent.py` 的 `rollout_single` / `rollout_batch`；
- `<tool_call>` 解析、工具执行、环境反馈和多轮上下文；
- response mask、old log-prob、延迟奖励和未完成轨迹；
- 工具数量对齐、GT 命中、格式奖励、重复惩罚和 reward clipping。

### 17.2 实验分支

- A00：S02 Full SFT 工具调用基线；
- A01：Full SFT → Agentic RL + GRPO；
- A02：Full SFT → Agentic RL + CISPO；
- A03：`agent_rl_math` RLVR 补充实验；
- A04：Torch vs SGLang rollout 性能对比；
- 奖励消融：格式、工具正确性、GT、未完成惩罚。

代码默认：batch 2、lr `3e-7`、num_generations 4、max_seq_len 1024、max_gen_len 768、max_total_len 2500、beta 0.1、CISPO、Full SFT 初始化。

### 17.3 评测

- 工具选择准确率；
- 参数 JSON/Schema 正确率；
- 工具执行成功率；
- 最终答案准确率；
- 端到端任务成功率；
- 平均工具调用数、平均轮数、未完成率；
- 无效标签、重复调用和 reward hacking；
- reward、KL、group reward std、policy loss 和平均响应长度；
- Base/SFT 通用能力回归。

### 17.4 阶段门

Agentic RL 必须提升端到端任务成功率，而不仅是训练 reward。报告必须说明工具环境范围、任务是否与训练集重叠、通用能力损失和推理成本。

## 18. Stage 12：统一评测、推理与服务

### 18.1 Checkpoint 矩阵

| Checkpoint | Base | Chat/Instruction | Preference/RL | Tool/Agent | 系统性能 |
|---|---:|---:|---:|---:|---:|
| P01/P02 | 必测 | 抽样 | - | - | 必测 |
| S01/S02 | 回归 | 必测 | 基线 | Tool 基线 | 必测 |
| LoRA/Full FT | 回归 | 领域必测 | - | 可选 | 必测 |
| DPO | 回归 | 必测 | 必测 | 回归 | 必测 |
| PPO/GRPO/CISPO | 回归 | 必测 | 必测 | 回归 | 必测 |
| MoE/Distill | 必测 | 必测 | 可选 | 回归 | 必测 |
| Agentic RL | 回归 | 回归 | 必测 | 必测 | 必测 |

### 18.2 任务

- 官方七项：C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag、Social IQA；
- 指令遵循：优先补充 IFEval 或固定可判分指令集；
- 数学：GSM8K 或自建严格答案集；
- 代码：HumanEval，仅在生成能力足够时作为诊断；
- Tool：官方 20 题只能作为 smoke，最终应建立 train-disjoint 的扩展集；
- Agent：工具、参数、执行、最终答案和完整轨迹五层指标。

评测数据版本、few-shot、模板、decoding、max tokens 和 seed 必须固定。选择题分数接近随机下界时也要如实报告。

### 18.3 服务

- `eval_llm.py`：本地交互与样例；
- `convert_model.py`：导出 Hugging Face 格式；
- `serve_openai_api.py`：OpenAI-compatible API；
- `web_demo.py` / `chat_api.py`：多轮、Thinking 和 Tool Call；
- 记录首 Token 延迟、生成吞吐、峰值显存和并发边界。

完整口径见 [`evaluation_protocol.md`](evaluation_protocol.md)。

## 19. SwanLab 记录规范

### 19.1 项目与实验身份

- 统一项目建议：`MiniMind-Lab`；
- 每个 run 必须绑定 `experiment_id`；
- 记录 Lab commit、MiniMind source commit、数据 checksum、seed、硬件、dtype、world size 和完整解析配置；
- run URL 写入实验目录的 `swanlab-url.txt`；
- checkpoint 中的 run id 用于断点恢复，不创建断裂曲线。

### 19.2 阶段指标

| 阶段 | 必须记录 |
|---|---|
| Pretrain/SFT/LoRA | loss、logits_loss、aux_loss、lr、ETA、tokens/s、显存 |
| DPO | loss、dpo_loss、margin/preference accuracy、lr、长度 |
| PPO | reward、KL_ref、approx_KL、clipfrac、critic_loss、响应长度、actor/critic lr |
| GRPO/CISPO | reward、KL_ref、advantage mean/std、policy_loss、响应长度、真实准确率 |
| Distill | loss、CE、KL/distill loss、aux loss、lr |
| Agentic RL | reward、KL、group reward std、policy loss、响应长度、任务成功率 |

源码未自动记录的吞吐、GPU 利用率、峰值显存和评测指标，通过外部采集或评测脚本补入 `metrics.csv/eval.json`，不得假设 SwanLab 自动包含。

## 20. Checkpoint、恢复与清理

每阶段最多长期保留：

- `last`：断点恢复；
- `best_target`：目标任务最佳；
- `best_retention`：兼顾通用能力；
- `release`：最终 HF 发布候选。

每个 checkpoint manifest 记录路径、step、epoch、指标、大小、SHA-256、初始化权重、SwanLab run 和保留理由。

删除前必须：

1. 确认训练完成或不再恢复；
2. 完成目标评测；
3. 选出 release；
4. 如需公开，确认 HF 上传与 Model Card；
5. 只删除 manifest 中明确列出的冗余文件。

GitHub 不保存权重、优化器状态、原始数据和完整日志。

## 21. 故障与停止条件

### 21.1 启动前

- 8 张 GPU 没有未知计算进程；
- 没有同一实验的重复锁；
- 数据可读且 checksum 正确；
- 输出目录与初始化权重明确；
- SwanLab 登录可用；
- 磁盘、CPFS 和 `/dev/shm` 足够。

### 21.2 自动停止或人工介入

- loss/metric 出现 NaN、Inf；
- OOM 或 NCCL 子进程退出；
- reward 上升但真实准确率下降；
- KL、输出长度或重复率持续失控；
- GPU 被其他任务抢占，吞吐明显异常；
- 数据为空、有效 label token 为 0；
- checkpoint 无法恢复或 SwanLab 曲线断裂。

收到 SIGTERM 时优先保存可恢复 checkpoint、停止串联任务并记录原因。禁止为了恢复本实验终止其他未知训练进程。

## 22. 8 周执行安排

| 周 | 训练主线 | 源码阅读 | 交付物 |
|---|---|---|---|
| 1 | 环境、历史盘点、数据探针 | Dataset、Tokenizer、checkpoint | manifest、数据报告、smoke |
| 2 | P01 + S01 mini 路线 | 模型结构、Pretrain、SFT | MiniMind Zero 基线、源码笔记 |
| 3 | P02 完整 Pretrain | optimizer、DDP、吞吐 | mini/full Base 对照 |
| 4 | S02 + Tool baseline | chat template、LoRA | Full SFT、领域方案 |
| 5 | DPO + PPO smoke | DPO、Reward/Critic、GAE | preference/RL 基线 |
| 6 | GRPO vs CISPO + rollout | GRPO、CISPO、SGLang | 算法与性能对照 |
| 7 | MoE + Distillation | Router、aux loss、CE+KL | Dense/MoE/KD 报告 |
| 8 | Agentic RL、统一评测、服务 | 多轮 rollout、工具环境 | final report、HF、博客、简历 |

时间表服从阶段门，不为赶周数跳过评测。Full 数据或 MoE 未完成时，顺延后续阶段，不并行制造多个不可归因结果。

## 23. 源码阅读与博客计划

| 顺序 | 源码笔记 | 对应博客 |
|---|---|---|
| 1 | Data/Tokenizer | 数据格式、压缩率与 label mask |
| 2 | Model Architecture | 64M Qwen3 风格结构与 Tensor 流 |
| 3 | Pretrain | DDP、梯度累积、lr、checkpoint |
| 4 | SFT/LoRA | Chat template、遗忘与参数高效微调 |
| 5 | DPO/PPO | Preference、Critic、GAE 与 KL |
| 6 | GRPO/CISPO | 组内优势、ratio clipping 与 rollout |
| 7 | MoE/Distill | Router、激活参数与 CE+KL |
| 8 | Agentic RL | 多轮工具环境、延迟奖励与任务成功率 |
| 9 | Eval/Serving | lm-eval、Tool eval、API 与性能 |

源码笔记必须包含模块职责、调用链、关键 Tensor shape、观察实验、失败问题和对应 commit。博客中的数字必须链接到实验报告。

## 24. 最终项目与简历验收

最终至少形成三条可量化证据：

1. **数据规模实验**：mini vs full 对 Base/Chat 的提升、训练时间和有效 Token 成本；
2. **后训练实验**：DPO/PPO/GRPO/CISPO 的目标收益、KL、遗忘和训练稳定性；
3. **Agent 实验**：Full SFT vs Agentic RL 的工具执行与端到端任务成功率，以及推理成本。

加分证据：

- Dense vs MoE 的质量/吞吐/显存对照；
- Full FT vs LoRA 的领域收益/成本；
- MoE teacher → Dense student 蒸馏收益；
- Torch vs SGLang rollout 加速。

最终发布资产：

- GitHub README 结果总表；
- SwanLab 对比项目；
- Hugging Face release 权重与 Model Card；
- `docs/final_report.md` 完整报告；
- 6–9 篇博客系列；
- 一段不夸大、每个数字可追溯的简历项目描述。

## 25. 1B 模型进入门槛

只有同时满足以下条件才评估 1B：

- 64M Dense 的 Pretrain、SFT、至少一种 RL 和统一评测已经闭环；
- 准备至少 20B 高质量 Pretrain tokens，目标 20–30B；
- 完成 1B 参数配置、Attention heads、序列长度和 tokenizer 容量评审；
- 完成显存估算、optimizer state、activation、checkpoint 和通信预算；
- 评估 DDP 是否足够，是否需要 FSDP/ZeRO、activation checkpointing、Flash Attention 和更高效数据管线；
- 明确训练预算、停止阈值、评测目标和相比 64M 的预期收益。

`pretrain_t2t.jsonl` 的 10GB 级数据适合完整训练官方 64M/MoE 主线，但不足以单独支撑可信的 1B 从零训练结论。

## 26. 下一步执行顺序

1. 在 L20 盘点旧 mini 路线资产和当前数据；
2. 建立 Stage 0 环境与实验模板；
3. 完成 Data/Tokenizer 报告；
4. 运行 100-step 模型与恢复探针；
5. 决定旧内部试跑是否仅保留为诊断证据，正式配置必须重新冻结；
6. P01/S01 通过统一评测后，再启动完整数据 P02/S02。

在第 5 步完成前，不把 MiniMind Zero 路线标记为完成，也不启动新的全量后训练。
