# E02-model-probe-20260823

## 状态

Stage2 running。正式结论等待三组 8 卡 100-step probe 和 resume 验证完成。

## 目标

1. 从输入 token 解释 Dense/MoE 模型结构、Tensor shape、logits 与 loss；
2. 验证 Dense/MoE 单卡 BF16 forward/backward；
3. 验证 64M Dense 在 8×L20 上连续运行 100 optimizer steps；
4. 验证 seed42 从 step50 恢复至 step100，optimizer、LR 与 SwanLab run 连续；
5. 用 seed42/43/44 观察短探针稳定性。

## 口径

- 模型：hidden 768、8 层、Dense、约 64M；
- 训练：BF16、8×L20、每卡 batch 4、seq 128、100 optimizer steps；
- 数据：由固定 revision 的官方 pretrain mini 前 8192 条生成预分词 probe 数据；
- 指标：loss、grad norm、tokens/s、samples/s、GPU 利用率、峰值显存；
- 边界：该实验只验证结构和运行时，不用于模型质量或收敛结论。

## 结果

待实验完成后填写。

## 结论

待实验完成后填写。
