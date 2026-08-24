# P01-dense-pretrain-mini-20260824

## Objective

在 8×NVIDIA L20 上按历史 Official Zero 配置完成 64M Dense 的 mini 数据预训练，建立后续 P02、S01 和后训练实验的首个正式 Base checkpoint。

## Direct baseline

这是本 Lab 的首个正式 Pretrain，因此没有可用于单变量结论的直接自身基线。历史 Official Zero 只作为配置参照；P01 完成后将成为 P02/S01 的直接基线。

## Configuration

- 模型：Dense 63,912,192 parameters，hidden 768，8 layers，vocab 6400；
- 硬件：8×NVIDIA L20 46GB，BF16；
- 训练：1 epoch，每卡 batch 4，累积 8，全局 sequence batch 256，seq 768，lr 5e-4；
- 数据：`pretrain_t2t_mini.jsonl`，1,270,238 行，1,241,043,656 bytes；
- 数据 SHA-256：`6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`；
- 日志：每 10 micro-steps 写训练日志/SwanLab，每 1,000 micro-steps 覆盖保存可恢复 checkpoint；
- 权重与完整硬件日志只保存在 `/data/artifacts`，不进入 Git。

## Training result

状态：运行中。2026-08-24 03:16:40 UTC 从随机权重启动，训练 commit `64d2908c8300985357993a66011e2b6114104fd1`，SwanLab run `7iochx9kfe75qa2pt6d1u`。

早期窗口已通过：step 10→100→470 的 loss 为 8.3476→7.2536→5.9830；step 480 日志 ETA 约 45 分钟。8 张卡利用率约 98–99%，单卡显存约 3,492 MiB，未观察到 OOM、NaN 或训练中 NCCL 错误。该窗口只证明训练健康，不能提前判定收敛。

## Evaluation result

训练完成后执行固定续写、Base 七项 benchmark 和系统指标；Base 评测不使用 chat template。上游训练脚本不生成独立 validation split，因此不能把 training loss 标成 validation loss。

## Cost and system metrics

待训练完成后从 `driver.log`、SwanLab 和 10 秒间隔的 `nvidia-smi.csv` 汇总 wall-clock、吞吐、GPU 利用率、峰值显存、功耗与 checkpoint 大小。

## Failures and anomalies

待填写。启动前由 `preflight_l20.py`、GPU 空闲检查和单实例锁阻止错误环境或重复任务。

## Conclusion

待训练和评测完成后填写，未完成前不得标记为 release candidate。
