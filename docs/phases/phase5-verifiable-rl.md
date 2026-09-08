# Phase 5：GRPO/CISPO 可验证强化学习

状态：进行中；启动日期 2026-09-08。Phase4 的 D03 未晋级，本阶段按总计划从 S10 独立分叉。

## 预注册问题

1. 在程序可验证的 train-disjoint 算术选择任务上，GRPO/CISPO 能否提升 held-out pass@1，而不只是提高训练 reward？
2. 与相同 prompt batch 和 optimizer update 数的 SFT continuation 相比，RL 是否有独立收益？
3. GRPO 和 CISPO 在复用同一 rollout 做第二次 policy update、importance ratio 不再恒为 1 时，稳定性和泛化是否不同？

## 数据与 verifier

verifiable-math-v2 为确定性生成的四选一加减乘算术：train 1,000、validation 200、test 400；task key 和 normalized prompt 跨 split 均为 0 重合。四个答案位置在 train/validation/test 分别严格均衡为每类 250/50/100，防止恒答某个位置获得超过 25% 的捷径；操作和四种中英文模板也分层平衡。

正确性 reward 只接受一个明确且不歧义的首选项标签，可兼容 A、A. 14、The correct answer is A.14；多个选项标签或无明确标签均为 0。严格“只输出字母”另记 format，不混入 correctness reward。程序重新计算运算结果并核对 option/label。审计只证明 Phase5 内部分割隔离，不能证明全部历史 S10 语料从未出现同类表达式。

这是一项窄域多选算术实验，不是 GSM8K/自由文本数学推理。若通过，只能声明 verifier RL 管线和该任务上的泛化结果。

## 对照与固定项

| 实验 | 方法 | 目的 |
|---|---|---|
| R02A | S10 不训练 | 初始策略基线 |
| R02B | SFT continuation | 直接监督正确选项的控制组 |
| R02C | GRPO | 对称 PPO-style ratio clipping |
| R02D | CISPO | detached upper-clipped importance weight |

R02B/C/D 固定初始权重 S10、train 顺序、seed 42、LR 3e-6、1,000 prompts、batch prompts 4、每外层 batch 2 次 policy update，共 500 optimizer updates。R02C/D 固定 16 rollouts/prompt、temperature 0.8、top-p 0.95、max new tokens 8、beta 0.02。GRPO epsilon 0.2；CISPO upper clip 2.0。

SFT 与 RL 只配平 prompt batch 和 optimizer update 数：SFT 使用一个真值 completion，RL 每 prompt 采样 16 条且还有 frozen reference forward，因此不宣称 FLOPs、target tokens 或 wall-clock 等预算。GRPO/CISPO 复用每批 rollout 做 2 次 policy update；第一次 ratio=1，第二次才实际检验 clip 分支差异。若每 rollout 只更新一次，二者在本实现中梯度等价，不能做方法对比。

参数以 FP32 保存和更新、BF16 autocast 前向；不再使用会抹除小幅后训练更新的 FP16 checkpoint 保存。

## v1 无效实验与协议修复

首轮 R01 使用的 v1 只检查了答案位置、模板的各自边际均衡，却用同一个递增 index 分配二者，导致 template_id 能确定性预测 answer_position。R01C/D 训练 reward 接近 1，但 validation 出现答案整体错位一格、pass@1=0%；这是明确的 reward hacking。

test 当时仍封闭。R01 run 和失败报告原样保留，不参与方法结论。v2 将答案位置按 index%4、模板按 (index//4)%4 构成交叉联合均衡；auditor 新增 16 个 answer×template 单元的全支持和计数差检查。R02 除数据联合分配外沿用相同超参数与门槛。

## 训练前探针

- 数据审计通过：train/validation/test task 和 prompt overlap 均为 0，答案位置完全平衡。
- 数值 smoke：SFT/GRPO/CISPO 均完成 2 outer steps / 4 updates，finite loss/grad，保存权重可加载。
- R02A validation：pass@1 24.0%，choice-valid 95.5%，strict format 0%；sample accuracy 20.5%，empirical pass@4 35.5%。
- R02A 几乎恒选 A：validation 200 条中 A 185、C 6、invalid 9；所以必须报告按 answer position 分桶，不能只看总体 reward。

正式训练前不打开 test。

## 指标与验收

训练记录 reward、group reward std、degenerate/all-zero group rate、pass@group、approx KL、entropy、completion length、clip fraction、gradient norm、rollout wall、总 wall、显存和 token 数。test 固定 greedy pass@1 为主，另报 fixed-sampling accuracy/pass@4、choice-valid、strict format，以及 operation/template/answer-position 分桶。

配对 bootstrap 95% CI 用同一 400 test task。R02C/R02D 只有同时满足以下条件才晋级：

- 相对 R02A 与 R02B 的 pass@1 差值区间下界均大于 0；
- 收益不是恒答某个位置：按 answer position 报告，choice distribution 不塌缩；
- training reward 上升同时 held-out pass@1 上升；
- 七项 macro 相对 S10 不退超过 1pp，IFEval prompt strict 不退超过 2pp；Chat 至少 7/10、格式至少 4/6、Tool E2E 至少 6/8、重复异常最多 2/10。

若 reward 升而 held-out 不升，或只靠答案位置/格式捷径，按负结果收口并保留 S10。test 只在三组训练全部完成后一次性打开，不用来调 LR、beta、clip 或步数。
