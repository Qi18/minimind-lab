# S01R1 run record

Status: prepared.

This is the repaired successor to invalidated S01. It adds a deterministic 10,000-example validation split, a P01 validation baseline at step 0, token-weighted 8-rank metrics, gradient norm and throughput telemetry, and best-validation plus last checkpoints.

The formal per-GPU batch size is selected only after 8-GPU probes at 4, 8, 16, and 32. A candidate passes when it completes without CUDA/NCCL/data errors, stays below 38 GiB peak memory, and provides a real throughput improvement. The largest passing batch is not selected automatically when throughput has already flattened.

Acceptance requires best validation loss below the P01 step-0 baseline. If final validation loss is more than 5% above best, downstream evaluation uses the best-validation checkpoint.
