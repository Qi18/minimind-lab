# S01R1 run record

Status: running since 2026-08-25T15:56:01Z.

This is the repaired successor to invalidated S01. It adds a deterministic 10,000-example validation split, a P01 validation baseline at step 0, token-weighted 8-rank metrics, gradient norm and throughput telemetry, and best-validation plus last checkpoints.

The formal per-GPU batch size is selected only after 8-GPU probes at 4, 8, 16, and 32. A candidate passes when it completes without CUDA/NCCL/data errors, stays below 38 GiB peak memory, and provides a real throughput improvement. The largest passing batch is not selected automatically when throughput has already flattened.

Acceptance requires best validation loss below the P01 step-0 baseline. If final validation loss is more than 5% above best, downstream evaluation uses the best-validation checkpoint.

## Start acceptance

- Selected batch: 16 per GPU, 128 global, from the recorded 4/8/16/32 probe.
- P01 step-0 validation loss: 2.469205 on 4,120,531 effective target tokens.
- Step-500 validation loss: 1.934163, a 21.67% relative improvement over P01.
- Step-1000 validation loss: 1.881176; best-validation and last checkpoints were both created.
- Throughput around 36万 effective target tokens/s; peak allocated memory about 6.94 GiB per GPU.
- SwanLab: https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s2zj3jb9n8uh9v7raemx5
