# 统一评测协议

## 目标

所有阶段使用一致的任务、生成参数和数据版本，区分“目标能力提升”和“通用能力回归”。任何只报告训练 loss 或 reward 的实验都不算完成。

## Checkpoint 行

- Pretrain mini；
- Pretrain full；
- Full SFT；
- LoRA；
- DPO；
- GRPO；
- CISPO；
- Agentic RL + GRPO；
- Agentic RL + CISPO。

## Base 能力

- validation loss / perplexity；
- C-Eval；
- CMMLU；
- ARC-Easy；
- PIQA；
- OpenBookQA；
- HellaSwag；
- Social IQA。

## 指令与领域能力

- 固定指令集成功率；
- 格式遵循率；
- 领域任务准确率；
- 人工抽样中的正确性、相关性和重复率。

## 强化学习指标

- task accuracy；
- average reward；
- KL to reference；
- group reward std；
- response length；
- policy loss；
- 训练中 NaN、OOM、异常梯度和退化样本数量。

## Agentic RL 指标

- tool selection accuracy；
- argument validity；
- execution success；
- final answer accuracy；
- end-to-end task success；
- average tool calls / turns；
- unfinished rate；
- invalid tool-call tag rate。

## 系统指标

- wall-clock time；
- tokens/s；
- samples/s；
- GPU utilization；
- peak allocated/reserved VRAM；
- checkpoint size；
- 推理首 Token 延迟、生成吞吐和峰值显存。

## 固定条件

每次对照实验必须固定或明确记录：

- checkpoint 初始化；
- 数据版本与样本范围；
- seed；
- decoding 参数；
- max sequence / generation length；
- dtype；
- GPU 数量；
- 评测代码 commit。

## 结果判定

每个实验至少回答：

1. 目标指标是否提升，绝对值和相对值分别是多少？
2. 通用能力下降多少？
3. 训练和推理成本增加多少？
4. 多个 seed 或样本抽查是否支持同一结论？
5. 是否存在 reward hacking、长度偏差或格式投机？
