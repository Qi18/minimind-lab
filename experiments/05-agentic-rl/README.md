# Phase6：Agent SFT 与 Agentic RL

当前：A02/A03探索性RL对照完成，Phase6未收口，S10继续保留。

- v1 pilot：validation239/240，结果保留。
- v2：独立test217/240，但既有Tool回归7/8→1/8、Chat8/10→7/10。
- 采样32组×4条全部成功，组内奖励方差0；当时A02/A03未训练。
- 下一步先修复A01遗忘，不能将专项SFT提升写成RL或通用Agent提升。

完整过程见[Phase6 v2实验报告](phase6-v2-report.md)；测试冻结登记见[a01-v2-test-freeze.md](a01-v2-test-freeze.md)。

## A01-v3混合回放修复

Chat恢复8/10，旧Tool仍5/8未过7/8门槛。新graph验证95.0%–98.125%、多工具91.25%–95.625%。
未选候选，未开v3 test，当时A02/A03未训练，S10保留。见[完整v3报告](phase6-v3-report.md)。

## A02/A03探索性Agentic RL

用户要求在A01保持门禁未过时进行探索性对照：固定末步、三臂冻结test。
A01与GRPO均451/480，CISPO461/480（+2.08pp，配对95%CI +0.83至+3.54）。
旧Chat8/10、旧Tool5/8；工具选择子指标下降，IFEval/七项尚未跑，不晋级。
见[训练、评测、失败与统计报告](phase6-rl-v2-report.md)。汇总仅本地，上传待确认。

## 通用回归补充（2026-09-08，随后评测，保留上文历史状态）

本轮S10/A01/GRPO/CISPO全部完成IFEval541题与七项全量评测，协议与样本哈希一致。

| 模型 | IFEval strict | 七项macro |
|---|---:|---:|
| S10 | 94/541 | 33.0024% |
| A01 | 86/541 | 33.2222% |
| A02 | 86/541 | 33.1674% |
| A03 | 83/541 | 33.2176% |

完整逐项变化与配对区间见[通用回归报告](phase6-general-regression-report.md)。
不临时添加下降容差；旧Tool仍5/8，模型不晋级。完整原始证据保存在CPFS，仅轻量结果入仓库，未上传、未commit/push。

## 正式阶段验收

用户同意按completed-not-promoted收尾，详见[正式Phase6报告](../../docs/phases/phase6-agentic-rl.md)。实验执行状态与模型晋级分开；registry状态整理的历史映射见phase6-acceptance.json。S10继续保留。
