# S01-dense-sft-mini-20260825

## Objective

从 P01 的 64M Dense Base checkpoint 出发，用 `sft_t2t_mini.jsonl` 完成一次历史 Official Zero 口径的全参数 SFT，建立后续 S02 和后训练阶段的 Chat/Tool 快速基线。

## Direct baseline

- `P01-dense-pretrain-mini-20260824`
- Base checkpoint SHA-256：`71efd40d9fcd494bc5472891b66ea7f17167ae27ac341968bcd258a5a24b94e9`
- P01 官方七项宏平均：31.4449；固定续写仍有退化，因此 S01 的首要问题是能否形成可交互模型，而不是只比较 loss。

## Configuration

- Hardware：8×NVIDIA L20 46GB
- Model：64M Dense，hidden 768，8 layers，seq_len 768
- Dataset：905,718 rows，SHA-256 `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`
- Training：1 epoch，每卡 batch 2，累积 1，global sequence batch 16，BF16，lr 1e-5
- Expected：56,608 microsteps / optimizer updates
- SwanLab：project `MiniMind-Lab`，run `S01-SFT-Mini-64M-Seq768`

本次先复现历史 SFT 参数，不同时更改 epoch、batch、学习率或数据规模。代码默认的 2 epoch / batch 16 留作后续消融候选。

## Training result

待训练完成后填写 loss、耗时、GPU-hours、吞吐、显存和 checkpoint 哈希。

## Evaluation result

待完成以下同协议比较：

1. 固定中文/英文/多轮与格式遵循集；
2. Tool Call 工具选择、参数合法性和最终答案正确率；
3. 带 chat template 的官方七项；
4. 相对 P01 的目标能力收益与 Base 能力回归。

## Cost and system metrics

待训练完成后由原始日志和 `nvidia-smi.csv` 聚合。

## Failures and anomalies

当前无；若出现数据 cache、OOM、NCCL、NaN、resume 或异常梯度问题，在此保留原始证据。

## Conclusion

状态：ready。只有训练、SwanLab、checkpoint manifest、Chat/Tool 和官方七项全部完成后，才可标记为 Official Zero completed。
