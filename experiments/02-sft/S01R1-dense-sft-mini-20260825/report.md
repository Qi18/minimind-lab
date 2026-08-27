# S01R1-dense-sft-mini-20260825

## 目标

修复已作废的 S01，使用可验证的 validation split、token-weighted 多卡指标、batch 探测和 best-validation checkpoint，在 P01 64M Base 上完成一次可复现的 mini SFT，并同时验收知识回归、Chat、Tool 和系统性能。

## 配置

- 模型：Dense 63,912,192 parameters，hidden 768，8 layers，vocab 6400；
- 初始化：P01 checkpoint，SHA-256 `71efd40d9fcd494bc5472891b66ea7f17167ae27ac341968bcd258a5a24b94e9`；
- 数据：`sft_t2t_mini.jsonl` 905,718 行，固定 revision `312afb4`，SHA-256 `abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518`；
- 切分：seed 42 的 10,000 条 validation，关闭数据增强；
- 训练：2 epochs，每卡 batch 16，全局 batch 128，seq 768，lr 1e-5，BF16；
- 硬件：8×NVIDIA L20 46GB；
- 训练 commit：`276d8f14c5cf43479b696805f9970b373c681613`；
- SwanLab：[S01R1-SFT-Mini-64M-B16x8](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s2zj3jb9n8uh9v7raemx5)。

每卡 batch 16 来自 4/8/16/32 的 8 卡实测：候选必须无 CUDA/NCCL/数据错误、显存低于 38 GiB，并带来真实吞吐收益。batch 32 虽可运行，但吞吐已趋平，因此选择 16，而不是为了伪造官方配置。

## 训练结果

训练从 2026-08-25 15:56:01 UTC 开始，16:31:17 UTC 正常完成，退出码 0。完成 `13,994/13,994` optimizer steps：

- P01 step-0 validation loss：2.469205；
- step-500 validation loss：1.934163；
- final / best validation loss：1.702404，较基线下降 31.06%；
- 最终 train loss：1.714777，grad norm 0.460633；
- 有效目标 token 吞吐中位数：366,142/s；
- 峰值 allocated 显存：6,936.51 MiB/卡；
- wall-clock：35.27 分钟，8 卡合计 4.70 GPU-hours。

best 与 last checkpoint 都有 91 个 tensor key 且逐 tensor 完全相同，文件哈希不同来自序列化元数据。正式评测选择 best：

- 路径：`/data/artifacts/minimind-lab/S01R1-dense-sft-mini-20260825/checkpoints/s01r1_best_val_768.pth`；
- 大小：137,684,570 bytes；
- SHA-256：`239d48e4a8e7a5abab02a72549b34b4996f2a3c9439da3df0c98952dc7f9e24a`。

## 评测结果

使用 `lm-evaluation-harness` 0.4.12（commit `6d642546`），0-shot、启用 chat template、batch 16、单张 L20、seed 42。主指标在任务提供时取 `acc_norm`，否则取 `acc`。

| 任务 | S01R1 | P01 Base | 差值（百分点） |
|---|---:|---:|---:|
| C-Eval | 23.33 | 23.40 | -0.07 |
| CMMLU | 25.14 | 25.07 | +0.07 |
| ARC-Easy | 29.88 | 28.03 | +1.85 |
| PIQA | 53.70 | 51.85 | +1.85 |
| OpenBookQA | 29.40 | 29.00 | +0.40 |
| HellaSwag | 27.82 | 28.88 | -1.06 |
| Social IQA | 35.01 | 33.88 | +1.13 |
| 七项宏平均 | 32.04 | 31.44 | +0.60 |

七项知识能力没有整体回退，但固定行为集暴露出严重问题：Chat 通过 1/10、格式约束 0/6、重复异常 8/10；Tool 选择 50.0%、参数有效 50.0%、端到端成功 37.5%。典型错误包括 `17 + 25 = 35`、无法只输出 JSON/`READY`、多段重复，以及选对工具后仍给出错误最终答案。

系统评测中，首 token 延迟中位数 3.86 ms，64-token 总延迟中位数 235.98 ms，解码吞吐中位数 271.21 token/s。原始结果、固定样例与完整条件见 [`eval/`](eval/)。

## 异常与修复

- 评测口径在 commit `a41170b` 先冻结；首次启动因 tokenizer 返回模型不接收的 `token_type_ids`，在生成任何分数前退出。
- commit `0001250` 只移除了该未使用输入，没有修改样本、评分规则或阈值；随后从头完整重跑。
- `requests` 有 urllib3/chardet 版本告警，但不影响模型导出、18 条行为样例、112,919 个 log-likelihood 请求和结果落盘。
- 训练配置中的 dataset SHA 曾有一处抄写错误，本报告按 L20 对源文件重新计算的 SHA 修正。

## 结论

S01R1 的训练工程闭环和 loss 收敛验收通过，七项宏平均也没有相对 P01 回退；但核心目标“得到稳定可用的对话模型”没有完成。该 run 应以“实验完成、发布验收失败”归档，保留 best checkpoint 与完整证据，不上传为公开 release model。

下一次 SFT 应优先验证数据模板与 assistant loss mask、缩短到可快速迭代的代表性子集，并把 Chat 重复率和格式通过率作为早停/验收指标，而不是继续依靠更长训练或只看 validation loss。
