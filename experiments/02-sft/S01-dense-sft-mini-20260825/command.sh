#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_ID="S01-dense-sft-mini-20260825"
EXPERIMENT_DIR="$ROOT_DIR/experiments/02-sft/$EXPERIMENT_ID"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
DATA_PATH="${DATA_PATH:-/data/datasets/minimind/312afb4f76391145c6902f765bb51691c09a12f5/sft_t2t_mini.jsonl}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/artifacts/minimind-lab/P01-dense-pretrain-mini-20260824/checkpoints/p01_pretrain_768.pth}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/artifacts/minimind-lab/$EXPERIMENT_ID}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data/cache/huggingface}"
FROM_RESUME="${FROM_RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXPECTED_DATA_SHA256="abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518"
EXPECTED_BASE_SHA256="71efd40d9fcd494bc5472891b66ea7f17167ae27ac341968bcd258a5a24b94e9"
EXPECTED_DATA_ROWS="905718"
SAVE_WEIGHT="s01_full_sft"
MODEL_INPUT_PATH="$ROOT_DIR/minimind/out/pretrain_768.pth"
RESUME_PATH="$ROOT_DIR/minimind/checkpoints/${SAVE_WEIGHT}_768_resume.pth"
HARDWARE_LOG="$ARTIFACT_DIR/logs/nvidia-smi.csv"
TRAIN_LOG="$ARTIFACT_DIR/logs/train.log"
RUNTIME_MANIFEST="$ARTIFACT_DIR/runtime-manifest.txt"
MONITOR_PID=""

fail() {
  printf 'error=%s\n' "$1" >&2
  exit 1
}

cleanup() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

[[ -x "$PYTHON_BIN" ]] || fail "python_not_executable:$PYTHON_BIN"
[[ -r "$DATA_PATH" ]] || fail "dataset_not_readable:$DATA_PATH"
[[ -r "$BASE_CHECKPOINT" ]] || fail "base_checkpoint_not_readable:$BASE_CHECKPOINT"
[[ "$FROM_RESUME" == "0" || "$FROM_RESUME" == "1" ]] || fail "FROM_RESUME_must_be_0_or_1"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN_must_be_0_or_1"

if [[ "$FROM_RESUME" == "1" ]]; then
  [[ -r "$RESUME_PATH" ]] || fail "resume_checkpoint_not_found:$RESUME_PATH"
elif [[ -e "$RESUME_PATH" ]]; then
  fail "stale_resume_checkpoint_exists:$RESUME_PATH"
fi

actual_data_sha256="$(sha256sum "$DATA_PATH" | awk '{print $1}')"
actual_base_sha256="$(sha256sum "$BASE_CHECKPOINT" | awk '{print $1}')"
actual_data_rows="$(wc -l < "$DATA_PATH" | tr -d ' ')"
[[ "$actual_data_sha256" == "$EXPECTED_DATA_SHA256" ]] || fail "dataset_sha256_mismatch:$actual_data_sha256"
[[ "$actual_base_sha256" == "$EXPECTED_BASE_SHA256" ]] || fail "base_checkpoint_sha256_mismatch:$actual_base_sha256"
[[ "$actual_data_rows" == "$EXPECTED_DATA_ROWS" ]] || fail "dataset_row_count_mismatch:$actual_data_rows"

mkdir -p "$ARTIFACT_DIR/checkpoints" "$ARTIFACT_DIR/logs" "$ROOT_DIR/minimind/out" "$HF_CACHE_ROOT/datasets" "$HF_CACHE_ROOT/hub"
ln -sfn "$BASE_CHECKPOINT" "$MODEL_INPUT_PATH"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$HF_CACHE_ROOT"
export HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets"
export HF_HUB_CACHE="$HF_CACHE_ROOT/hub"

if [[ "$DRY_RUN" == "0" ]]; then
  "$PYTHON_BIN" "$EXPERIMENT_DIR/prepare_dataset.py" \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$ROOT_DIR/minimind/model" \
    --max-seq-len 768 \
    --expected-rows "$EXPECTED_DATA_ROWS"
fi

{
  printf 'experiment_id=%s\n' "$EXPERIMENT_ID"
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'lab_commit=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
  printf 'minimind_source_commit=%s\n' "393e387e9ad99f0f04c296e4c5e7353f4444629f"
  printf 'dataset_path=%s\n' "$DATA_PATH"
  printf 'dataset_rows=%s\n' "$actual_data_rows"
  printf 'dataset_sha256=%s\n' "$actual_data_sha256"
  printf 'base_checkpoint=%s\n' "$BASE_CHECKPOINT"
  printf 'base_checkpoint_sha256=%s\n' "$actual_base_sha256"
  printf 'artifact_dir=%s\n' "$ARTIFACT_DIR"
  printf 'from_resume=%s\n' "$FROM_RESUME"
  printf 'swanlab_project=%s\n' "MiniMind-Lab"
  printf 'swanlab_run_name=%s\n' "S01-SFT-Mini-64M-Seq768"
} > "$RUNTIME_MANIFEST"

nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits \
  --loop=10 > "$HARDWARE_LOG" &
MONITOR_PID="$!"

GUARD_ARGS=()
if [[ "$DRY_RUN" == "1" ]]; then
  GUARD_ARGS+=(--dry-run)
fi

cd "$ROOT_DIR/minimind/trainer"
set +e
"$ROOT_DIR/scripts/launch/run_guarded.py" "${GUARD_ARGS[@]}" -- \
  "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node=8 \
  train_full_sft.py \
  --epochs 1 \
  --batch_size 2 \
  --accumulation_steps 1 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --dtype bfloat16 \
  --learning_rate 1e-5 \
  --grad_clip 1.0 \
  --num_workers 8 \
  --log_interval 10 \
  --save_interval 1000 \
  --data_path "$DATA_PATH" \
  --save_dir "$ARTIFACT_DIR/checkpoints" \
  --save_weight "$SAVE_WEIGHT" \
  --from_weight pretrain \
  --from_resume "$FROM_RESUME" \
  --use_wandb \
  --wandb_project MiniMind-Lab \
  --wandb_run_name S01-SFT-Mini-64M-Seq768 \
  2>&1 | tee "$TRAIN_LOG"
run_rc="${PIPESTATUS[0]}"
set -e
printf 'finished_at=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_rc" >> "$RUNTIME_MANIFEST"
exit "$run_rc"
