# S01R1 run record

Status: completed at 2026-08-25T16:31:17Z; release acceptance failed after evaluation at 2026-08-27T08:18:57Z.

This is the repaired successor to invalidated S01. It adds a deterministic 10,000-example validation split, a P01 validation baseline at step 0, token-weighted 8-rank metrics, gradient norm and throughput telemetry, and best-validation plus last checkpoints.

The formal per-GPU batch size was selected after 8-GPU probes at 4, 8, 16, and 32. Batch 16 was selected because it passed the 38 GiB memory guard while throughput had already flattened by batch 32.

## Training acceptance

- Completed 13,994/13,994 optimizer steps with exit code 0.
- P01 step-0 validation loss: 2.469205 on 4,120,531 effective target tokens.
- Final and best validation loss: 1.702404, a 31.06% relative improvement.
- Final train loss 1.714777; final grad norm 0.460633.
- Median effective target-token throughput 366,142/s; peak allocated memory 6,936.51 MiB per GPU.
- Wall-clock 35.27 minutes; 4.70 total GPU-hours.
- Best and last checkpoints contain identical tensors; evaluation used best-validation SHA `239d48e4`.
- SwanLab: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s2zj3jb9n8uh9v7raemx5

## Evaluation acceptance

- Official seven-task macro average: 32.04, versus P01 31.44 (+0.60 percentage points).
- Chat success: 10.0%; constrained-format success: 0/6; repetition anomalies: 8/10.
- Tool selection: 50.0%; Tool end-to-end success: 37.5%.
- Inference on one L20: 3.86 ms median first-token latency and 271.21 token/s median decode throughput.

Training convergence passed, but Chat and Tool behavior did not. This run is a completed, reproducible negative result rather than a release candidate. See [`report.md`](report.md) and [`eval/summary.md`](eval/summary.md).
