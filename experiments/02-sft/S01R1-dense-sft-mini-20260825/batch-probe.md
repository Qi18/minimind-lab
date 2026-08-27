# S01R1 batch-size probe

Environment: 8x NVIDIA L20 46GB, BF16, sequence length 768, P01 initialization. Each candidate used 50 warm-up steps followed by 200 measured optimizer steps.

| Per-GPU batch | Global batch | Sequences/s | Valid tokens/s | Peak memory/GPU | Result |
|---:|---:|---:|---:|---:|---|
| 4 | 32 | 441.54 | 179,930 | 2,600.9 MiB | pass |
| 8 | 64 | 759.67 | 308,519 | 4,044.6 MiB | pass |
| 16 | 128 | **903.79** | **366,488** | 6,940.8 MiB | selected |
| 32 | 256 | 887.94 | 361,457 | 12,666.2 MiB | pass, not selected |

Decision: use per-GPU batch 16. Batch 32 consumed 82.5% more peak memory than batch 16 while sequences/s fell by 1.75% and valid tokens/s fell by 1.37%, so the throughput curve had already flattened.
