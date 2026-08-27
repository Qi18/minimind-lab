# S01R1 evaluation freeze

The evaluation rules below were committed before observing S01R1 evaluation outputs.

- Official benchmark: `lm-evaluation-harness` 0.4.12, commit `6d642546f4688648fced259eb3302efd36ece5af`.
- Tasks: `ceval-valid`, `cmmlu`, `arc_easy`, `piqa`, `openbookqa`, `hellaswag`, `social_iqa`.
- Main metric: `acc_norm` when supplied by the task, otherwise `acc`.
- Official run: 0-shot, batch 16, FP16-exported checkpoint, one L20, seed 42, chat template enabled.
- Chat: ten deterministic, automatically scored harmless prompts; greedy decoding; success, format, repetition and refusal anomalies are recorded.
- Tool: the eight prompts and tool schemas are derived from MiniMind `scripts/eval_toolcall.py`; greedy decoding; tool selection, argument validity, mock execution, final answer and end-to-end success are scored.
- System benchmark: one fixed chat prompt, five warm-ups and twenty measured greedy runs, max 64 generated tokens; median/P95 first-token and total latency plus throughput and peak VRAM are recorded.
- P01 comparison uses the already completed P01 result under the same harness commit. P01 correctly omits the chat template because it is a Base checkpoint.

No pass threshold is introduced after results are observed. Results are reported as absolute scores and deltas against P01 and the MiniMind README reference table.
