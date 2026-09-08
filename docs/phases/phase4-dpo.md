# Phase 4：DPO 偏好优化

状态：进行中；启动日期2026-09-08。Phase3已推送main，S10仍是唯一输入基线。

## 预注册对照

| 实验 | 方法 | 基线 | LR / beta | 预算 |
|---|---|---|---|---|
| D01 | 不训练，计算参考偏好与生成基线 | S10 | — | 同一held-out |
| D02 | chosen-only CE | S10 | 4e-8 / 评测beta0.15 | 1epoch、global batch32 |
| D03 | reference-relative DPO | S10，ref冻结 | 4e-8 / 0.15 | 同D02 chosen曝光与更新次数 |

首轮采用本仓库上游train_dpo默认LR/beta，不预先声称最优。两组相同种子42、seq1024、同模板、同样本顺序、同cosine日程，初始权重也参与最低validation DPO loss选模。D02只监督chosen；D03额外使用rejected，因此配平chosen曝光/更新次数，不声称总targets或FLOPs完全相同。首轮各使用1张L20（GPU2/3）；micro batch8、累积4次，global32。D01在GPU4。

## 数据

复用官方dpo.jsonl，来源MiniMind数据revision312afb4f76391145c6902f765bb51691c09a12f5；SHA ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768，上游说明其抽样自llamafactory/DPO-En-Zh-20k。

原始17,166对，去掉过短缺上下文389、超1024 token的432、相同答案27、精确重复77、近似重复47。保留16,194对，按提示hash(seed42)划分train14,194 / validation1,000 / test1,000。train chosen targets4,765,435、rejected4,375,948。不截断回答；chosen/rejected共享完整prompt，只监督assistant回答及结束标记；空think块在两侧确定性移除，关闭随机模板处理。

构建器对IFEval及固定Chat/Tool提示做近重复过滤；独立auditor另核验全量文件SHA、跨split exact/近似重复、token长度和监督边界，通过后才生成_SUCCESS。该检查不保证S10全部历史数据从未见过测试提示。

## 训练前验收

DPO公式与梯度必须与上游dpo_loss一致；policy=ref时loss=log(2)、reference无梯度；真实batch smoke检查finite loss/grad、参数更新及保存。正式训练只读取train与validation，test不用于LR或checkpoint选择。

共用driver使用FP32 log_softmax计算log概率、BF16模型前向，DPO对assistant log概率求和，D02按整个更新批次的chosen tokens加权。独立输出best/last及optimizer/RNG状态；resume文件已保存但本轮不宣称精确恢复测试完成。

## 最终评测与判定（尚未执行）

- preference test 1,000对：reference-relative reward accuracy（平局0.5）、margin、DPO loss、chosen/rejected NLL、长度归一化偏好率、按chosen长短分桶；D01隐式reward恒为0，不能把该50%基线当作模型天然偏好正确率。
- 与D01和D02作配对比较及bootstrap95%区间；只有D03相对两者的偏好准确率差值区间下界都>0，才声明稳定偏好收益。若本轮无收益，保留负结果，不能直接扩大预算后沿用同test选模。
- 生成盲评：从test固定hash抽200条，对三模型greedy输出隐藏模型身份；报告win/tie/loss、长度分桶、拒答率及重复。人工或独立judge标注尚未落实；无该证据时不能声明回答更好或阶段通过。
- 通用回归：七项macro相对S10不退超过1pp、单项不退超过4pp；IFEval prompt strict不退超过2pp；Chat至少7/10、格式至少4/6、Tool E2E至少6/8、重复异常最多2/10。
- 同时记录输出长度与reference差异。离线数据的logratio不是on-policy KL，不把它标作精确KL；如报告KL，需明确上下文分布与计算方式。
- 盲评长度控制优先采用预注册的长度分桶胜率；未训练AlpacaEval式校正模型，不把分桶结果冒称官方LC win rate。

本阶段不以train loss或偏好margin单独验收。阶段报告须补齐实际训练、独立生成评测、SwanLab链接和checkpoint SHA后才收尾。

## 正式run

- D01-s10-preference-baseline-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/os17qekn
- D02-chosen-only-sft-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/51vdt97x
- D03-dpo-official-20260908: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3w1au7dk

Smoke：chosen-only已通过（dj3ajwap），DPO已通过（uc0ssf3w）；参考模型无梯度，初始DPO loss=log(2)，保存权重发生实际更新。
