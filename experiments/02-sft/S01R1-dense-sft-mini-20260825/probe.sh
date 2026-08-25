#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
EXPERIMENT_ID="S01R1-dense-sft-mini-20260825"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/minimind-lab/bin/python}"
DATA_PATH="${DATA_PATH:-/data/datasets/minimind/312afb4f76391145c6902f765bb51691c09a12f5/sft_t2t_mini.jsonl}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/artifacts/minimind-lab/P01-dense-pretrain-mini-20260824/checkpoints/p01_pretrain_768.pth}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/data/artifacts/minimind-lab/$EXPERIMENT_ID}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/data/cache/huggingface}"
RESULTS="$ARTIFACT_DIR/probes/batch-probe.jsonl"

mkdir -p "$ARTIFACT_DIR/probes" "$ROOT_DIR/minimind/out"
ln -sfn "$BASE_CHECKPOINT" "$ROOT_DIR/minimind/out/pretrain_768.pth"
: > "$RESULTS"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME="$HF_CACHE_ROOT" HF_DATASETS_CACHE="$HF_CACHE_ROOT/datasets" HF_HUB_CACHE="$HF_CACHE_ROOT/hub"

cd "$ROOT_DIR/minimind/trainer"
for batch_size in 4 8 16 32; do
  log_path="$ARTIFACT_DIR/probes/batch-${batch_size}.log"
  set +e
  "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node=8 probe_sft_batch.py \
    --data_path "$DATA_PATH" \
    --from_weight pretrain \
    --batch_size "$batch_size" \
    --max_seq_len 768 \
    --warmup_steps 50 \
    --measure_steps 200 \
    --num_workers 8 2>&1 | tee "$log_path"
  rc="${PIPESTATUS[0]}"
  set -e
  if [[ "$rc" == "0" ]]; then
    grep '^PROBE_JSON=' "$log_path" | tail -1 | sed 's/^PROBE_JSON=//' >> "$RESULTS"
  else
    printf '{"status":"failed","per_gpu_batch_size":%s,"exit_code":%s}\n' "$batch_size" "$rc" >> "$RESULTS"
    pkill -f 'probe_sft_batch.py' 2>/dev/null || true
    sleep 5
  fi
done
printf 'results=%s\n' "$RESULTS"
