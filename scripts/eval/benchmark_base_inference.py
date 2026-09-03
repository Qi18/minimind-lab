#!/usr/bin/env python3
"""Benchmark HF Base TTFT and fixed-length generation with 5+20 runs each.

EOS stopping is disabled so MiniMind's custom generation cannot end a run early.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WARMUP_RUNS = 5
MEASURED_RUNS = 20
DEFAULT_PROMPT = "中国人工智能的发展趋势是"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def nearest_rank(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty list")
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": median(values),
        "p95": nearest_rank(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def atomic_write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def benchmark_scenario(
    model,
    encoded: dict[str, torch.Tensor],
    device: torch.device,
    output_tokens: int,
    pad_token_id: int | None,
) -> dict[str, Any]:
    input_tokens = int(encoded["input_ids"].shape[1])
    generate_args = {
        **encoded,
        "do_sample": False,
        "use_cache": True,
        "min_new_tokens": output_tokens,
        "max_new_tokens": output_tokens,
        "pad_token_id": pad_token_id,
        # MiniMind's custom generate may ignore min_new_tokens. Disabling EOS is
        # therefore the actual fixed-length guarantee; the length gate below verifies it.
        "eos_token_id": None,
    }
    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    throughput: list[float] = []
    records: list[dict[str, float | int]] = []
    with torch.inference_mode():
        for run_index in range(WARMUP_RUNS + MEASURED_RUNS):
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_wall = time.perf_counter()
            start_event.record()
            generated = model.generate(**generate_args)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_wall_ms = (time.perf_counter() - start_wall) * 1000.0
            elapsed_gpu_ms = float(start_event.elapsed_time(end_event))
            generated_tokens = int(generated.shape[1] - input_tokens)
            if generated_tokens != output_tokens:
                raise RuntimeError(
                    f"generation length changed: {generated_tokens} != {output_tokens}"
                )
            if run_index >= WARMUP_RUNS:
                tokens_per_second = generated_tokens / (elapsed_wall_ms / 1000.0)
                wall_ms.append(elapsed_wall_ms)
                gpu_ms.append(elapsed_gpu_ms)
                throughput.append(tokens_per_second)
                records.append({
                    "run": run_index - WARMUP_RUNS + 1,
                    "wall_latency_ms": elapsed_wall_ms,
                    "gpu_latency_ms": elapsed_gpu_ms,
                    "output_tokens": generated_tokens,
                    "output_tokens_per_second": tokens_per_second,
                })
    return {
        "output_tokens_per_run": output_tokens,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "wall_latency_ms": summarize(wall_ms),
        "gpu_latency_ms": summarize(gpu_ms),
        "output_tokens_per_second": {
            "median": median(throughput),
            "p05": nearest_rank(throughput, 0.05),
            "min": min(throughput),
            "max": max(throughput),
        },
        "runs": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF Base TTFT and generation latency; fixed 5 warmups + 20 runs."
    )
    parser.add_argument("--model", type=Path, required=True, help="Exported HF Base directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least 2")
    model_dir = args.model.resolve()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("latency benchmark requires an available CUDA device")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"exported model directory not found: {model_dir}")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    torch.manual_seed(42)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)
    encoded.pop("token_type_ids", None)
    input_tokens = int(encoded["input_ids"].shape[1])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    started_at = utc_now()
    torch.cuda.reset_peak_memory_stats(device)
    ttft = benchmark_scenario(
        model, encoded, device, 1, pad_token_id
    )
    generation = benchmark_scenario(
        model, encoded, device, args.max_new_tokens, pad_token_id
    )
    export_manifest_path = model_dir / "export_manifest.json"
    export_manifest = (
        json.loads(export_manifest_path.read_text(encoding="utf-8"))
        if export_manifest_path.is_file() else {}
    )
    payload = {
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "model": str(model_dir),
        "source_checkpoint": export_manifest.get("source_checkpoint"),
        "source_checkpoint_sha256": export_manifest.get("source_checkpoint_sha256"),
        "export_manifest_sha256": (
            sha256(export_manifest_path) if export_manifest_path.is_file() else None
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "dtype": args.dtype,
        "prompt": args.prompt,
        "input_tokens": input_tokens,
        "decoding": "greedy_fixed_length_eos_disabled",
        "fixed_length_eos_token_id": None,
        "fixed_length_policy": (
            "eos_token_id=None; max_new_tokens is authoritative; output length is asserted"
        ),
        "percentile_method": "median=statistics.median; p95=nearest-rank",
        "ttft": ttft,
        "generation": generation,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "torch_version": torch.__version__,
    }
    atomic_write_json(output, payload, args.overwrite)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
