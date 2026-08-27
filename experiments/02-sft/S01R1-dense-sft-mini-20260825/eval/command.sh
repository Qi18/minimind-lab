#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
EXPERIMENT_ID="S01R1-dense-sft-mini-20260825"
EXP_DIR="$ROOT_DIR/experiments/02-sft/$EXPERIMENT_ID"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/artifacts/minimind-lab/$EXPERIMENT_ID}"
CHECKPOINT="$ARTIFACT_DIR/checkpoints/s01r1_best_val_768.pth"
EXPORT_DIR="$ARTIFACT_DIR/exported-chat"
EVAL_ARTIFACT_DIR="$ARTIFACT_DIR/eval"
OFFICIAL_DIR="$EVAL_ARTIFACT_DIR/official"
EVAL_VENV="${EVAL_VENV:-/data/venvs/minimind-eval}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data/cache/huggingface}"
EXPECTED_CHECKPOINT_SHA256="239d48e4a8e7a5abab02a72549b34b4996f2a3c9439da3df0c98952dc7f9e24a"

[[ -x "$EVAL_VENV/bin/python" ]] || { echo "missing eval python" >&2; exit 1; }
[[ -x "$EVAL_VENV/bin/lm-eval" ]] || { echo "missing lm-eval" >&2; exit 1; }
[[ -r "$CHECKPOINT" ]] || { echo "missing checkpoint" >&2; exit 1; }
actual_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$actual_sha256" == "$EXPECTED_CHECKPOINT_SHA256" ]] || { echo "checkpoint sha256 mismatch" >&2; exit 1; }

mkdir -p "$EXP_DIR/eval" "$OFFICIAL_DIR" "$EVAL_ARTIFACT_DIR/behavior"
export HF_HOME="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cd "$ROOT_DIR"

"$EVAL_VENV/bin/python" scripts/eval/export_minimind_base.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR" \
  --minimind-dir minimind \
  --hidden-size 768 \
  --num-hidden-layers 8 \
  --dtype float16 \
  2>&1 | tee "$EVAL_ARTIFACT_DIR/export.log"

CUDA_VISIBLE_DEVICES=0 "$EVAL_VENV/bin/python" scripts/eval/eval_sft_behavior.py \
  --model "$EXPORT_DIR" \
  --output-dir "$EVAL_ARTIFACT_DIR/behavior" \
  --device cuda:0 \
  --seed 42 \
  --max-new-tokens 192 \
  2>&1 | tee "$EVAL_ARTIFACT_DIR/behavior.log"

CUDA_VISIBLE_DEVICES=0 "$EVAL_VENV/bin/lm-eval" run \
  --model hf \
  --model_args "pretrained=$EXPORT_DIR" dtype=auto trust_remote_code=True \
  --tasks ceval-valid cmmlu arc_easy piqa openbookqa hellaswag social_iqa \
  --batch_size 16 \
  --device cuda:0 \
  --trust_remote_code \
  --apply_chat_template \
  --seed 42 \
  --output_path "$OFFICIAL_DIR" \
  2>&1 | tee "$EVAL_ARTIFACT_DIR/official.log"

raw_result="$(find "$OFFICIAL_DIR" -type f -name 'results_*.json' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$raw_result" && -s "$raw_result" ]] || { echo "missing official result" >&2; exit 1; }
cp "$raw_result" "$EXP_DIR/eval/official_benchmarks.json"
cp "$EVAL_ARTIFACT_DIR/behavior/task_eval.json" "$EXP_DIR/eval/task_eval.json"
cp "$EVAL_ARTIFACT_DIR/behavior/system_metrics.json" "$EXP_DIR/eval/system_metrics.json"
cp "$EVAL_ARTIFACT_DIR/behavior/samples.jsonl" "$EXP_DIR/eval/samples.jsonl"

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  printf 'experiment_id=%s\n' "$EXPERIMENT_ID"
  printf 'started_at=%s\n' "$started_at"
  printf 'finished_at=%s\n' "$finished_at"
  printf 'evaluation_lab_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'checkpoint_sha256=%s\n' "$actual_sha256"
  printf 'official_result=%s\n' "$raw_result"
  printf 'exit_code=0\n'
} > "$EVAL_ARTIFACT_DIR/runtime-manifest.txt"

echo "evaluation_finished_at=$finished_at"
