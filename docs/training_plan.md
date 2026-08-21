# MiniMind 训练与源码学习规划

## 总目标

以 MiniMind 64M Dense 为主线，完成“源码理解—训练实验—统一评测—报告输出”的闭环。198M-A64M MoE、蒸馏和更复杂推理引擎作为主线完成后的扩展，不与基础阶段并行展开。

每个阶段同时包含四项工作：

1. 阅读对应源码；
2. 运行最小观察实验；
3. 完成正式训练或评测；
4. 写出可验证结论并通过阶段门。

## 阶段 0：环境与可复现基线

### 阅读

- 官方 README、训练入口和参数解析；
- 分布式初始化、dtype、梯度累积、保存与恢复逻辑；
- SwanLab 初始化和指标命名。

### 实验

- 在 L20 执行单卡 smoke test；
- 验证 8 卡 DDP、NCCL、BF16、数据挂载和 checkpoint 恢复；
- 建立 `run.json`、`metrics.csv` 与 SwanLab 对应关系。

### 产物

- 环境清单；
- smoke test 记录；
- 可复现启动脚本。

### 阶段门

相同配置能够恢复训练，且 Git commit、SwanLab run、checkpoint 和指标能够互相追溯。

## 阶段 1：数据与 Tokenizer

### 阅读

- `minimind/dataset/lm_dataset.py`；
- Pretrain、SFT、DPO、RLAIF、Agent RL 样本结构；
- padding、截断、label mask 与 chat template。

### 实验

- 统计 mini/full 数据的行数、长度分布、有效 Token 和 padding 比例；
- 随机解码样本，验证输入、标签和 loss mask；
- 比较不同领域文本的 Tokenizer 压缩率。

### 产物

- 数据 manifest 与 checksum；
- 数据质量报告；
- Token/parameter 的实际口径，而不是只用最大序列槽位估算。

### 阶段门

能够从一条 JSONL 样本解释到最终 `input_ids`、`labels` 和有效 loss token。

## 阶段 2：模型结构与前向传播

### 阅读

- `minimind/model/model_minimind.py`；
- Embedding、RMSNorm、RoPE、Attention、MLP、Residual 和 LM Head；
- Dense 与 MoE 的结构差异；
- KV Cache 和生成路径。

### 实验

- 打印关键 Tensor shape；
- 核对参数量和显存组成；
- 跑 100-step probe，观察 loss、吞吐和显存。

### 产物

- 模型数据流图；
- 参数量拆解；
- 前向和生成流程笔记。

### 阶段门

能够解释一批 Token 如何经过模型形成 logits、loss，以及训练和推理路径的差异。

## 阶段 3：Pretrain

### 阅读

- `minimind/trainer/train_pretrain.py`；
- 优化器参数组、学习率调度、梯度累积、AMP/DDP、日志与 checkpoint。

### 实验

1. mini 数据基线；
2. full 数据基线；
3. 必要时做 batch size、序列长度或学习率的小型消融。

### 评测

- validation loss / perplexity；
- C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag、Social IQA；
- 吞吐、GPU 利用率、峰值显存和总训练时间。

### 核心问题

- mini 与 full 数据对 Base 能力的实际影响是什么？
- 有效 Token/parameter 是否足够？
- 训练是否受数据、计算、通信或 padding 限制？

### 阶段门

选出一个 Base checkpoint，并用统一评测证明选择依据。

## 阶段 4：Full SFT 与 LoRA

### 阅读

- `minimind/trainer/train_full_sft.py`；
- `minimind/trainer/train_lora.py`；
- `minimind/model/model_lora.py`；
- assistant-only label mask 和 LoRA 注入位置。

### 实验

- Full SFT；
- LoRA；
- Full SFT 与 LoRA 的成本、目标能力和遗忘对照；
- 必要时做 mask 或学习率消融。

### 评测

- 指令遵循与领域任务；
- Base 基准回归；
- 训练时间、显存、可训练参数和权重大小。

### 阶段门

选出统一的 Full SFT 基线，供 DPO、GRPO/CISPO 和 Agentic RL 独立初始化。

## 阶段 5：DPO

### 阅读

- `minimind/dataset/lm_dataset.py` 中的 `DPODataset`；
- `minimind/trainer/train_dpo.py`；
- chosen/rejected、reference policy、log-ratio 和 beta。

### 实验

- 从 Full SFT 初始化 DPO；
- 比较训练前后的 preference accuracy；
- 做一个 beta 或数据量消融。

### 评测

目标偏好胜率、输出长度变化、格式变化，以及统一通用能力回归。

### 阶段门

能够证明偏好指标的提升不是由输出长度或模板投机造成。

## 阶段 6：GRPO 与 CISPO

### 阅读

- `minimind/trainer/train_grpo.py`；
- `minimind/trainer/rollout_engine.py`；
- 多样本 rollout、组内标准化 advantage、KL、ratio clipping；
- `loss_type=grpo` 与 `loss_type=cispo` 的实现差异；
- Torch 与 SGLang rollout 边界。

### 实验

- Full SFT → GRPO；
- Full SFT → CISPO；
- 固定数据与预算比较 reward、KL、稳定性和任务准确率；
- 可选比较 Torch 与 SGLang rollout 性能。

### 阶段门

同时报告目标任务收益和通用能力回归，并解释 reward 与真实准确率是否一致。

## 阶段 7：Agentic RL

### 阅读

- `minimind/dataset/lm_dataset.py` 中的 `AgentRLDataset`；
- `minimind/trainer/train_agent.py`；
- 多轮 rollout、工具调用解析、环境反馈、response mask 和延迟奖励；
- 工具格式、调用数量、GT、未完成惩罚和 reward clipping。

### 实验

- Full SFT 工具调用基线；
- Full SFT → Agentic RL + GRPO；
- Full SFT → Agentic RL + CISPO；
- 奖励消融：工具正确性、GT、未完成惩罚；
- 可选比较 Torch 与 SGLang rollout。

### 评测

- 工具选择准确率；
- 参数格式正确率；
- 工具执行成功率；
- 最终答案准确率；
- 完整任务成功率；
- 平均调用次数、平均轮数和未完成率；
- reward、KL、response length 和通用能力回归。

### 阶段门

Agentic RL 必须提升端到端任务成功率，而不只是训练 reward；同时明确通用能力损失和推理成本。

## 阶段 8：统一评测、服务与总结

### 阅读

- `minimind/eval_llm.py`；
- `minimind/scripts/eval_toolcall.py`；
- `minimind/scripts/convert_model.py`；
- `minimind/scripts/serve_openai_api.py` 和 `web_demo.py`。

### 工作

- 用同一评测协议覆盖所有阶段 checkpoint；
- 测试生成延迟、吞吐、显存和服务兼容性；
- 将最终权重发布到 Hugging Face；
- 完成 final report、博客系列和简历描述。

### 阶段门

README 中每个数字都能链接到实验目录；最终项目能够一键定位源码、配置、曲线、权重和评测证据。

## 建议执行顺序

```text
先完成：0 → 1 → 2 → 3 → 4
再分支：DPO / GRPO-CISPO / Agentic RL
最后完成：统一评测 → 服务 → 报告 → 简历
```

不要为了覆盖功能而顺序串联所有后训练权重。源码阅读可以按 PPO → GRPO → CISPO → Agentic RL 的知识顺序推进，但正式训练应从共同基线分支以保持归因。
