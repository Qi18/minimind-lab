#!/usr/bin/env python3
"""Log existing evaluation JSON artifacts to one SwanLab evaluation run.

The script never reads API keys. Authentication is delegated to the existing
SwanLab client login. --dry-run validates and prints the exact payload without
importing SwanLab or performing a network write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT = "MiniMind-Lab"
EXPERIMENT_NAME = "P03-Eval-V1-1B28-64M-Seq768"
GROUP = "P03"
TASKS = (
    "ceval-valid", "cmmlu", "arc_easy", "piqa",
    "openbookqa", "hellaswag", "social_iqa",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def metric_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric is not numeric at {context}: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric is not finite at {context}: {number}")
    return number


def as_percent(value: Any, context: str) -> float:
    number = finite_number(value, context)
    return number * 100.0 if -1.0 <= number <= 1.0 else number


def pick_harness_metric(entry: dict[str, Any], task: str) -> float:
    for key in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if key in entry:
            return as_percent(entry[key], f"{task}.{key}")
    raise KeyError(f"no acc/acc_norm metric found for task {task}")


def extract_official(payload: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    aggregate = payload.get("results_percent")
    if isinstance(aggregate, dict):
        for task in TASKS:
            if task not in aggregate:
                raise KeyError(f"official aggregate missing task: {task}")
            values[task] = as_percent(aggregate[task], f"results_percent.{task}")
    else:
        raw_results = payload.get("results")
        if not isinstance(raw_results, dict):
            raise KeyError("official JSON has neither results_percent nor results")
        for task in TASKS:
            entry = raw_results.get(task)
            if not isinstance(entry, dict):
                raise KeyError(
                    f"raw harness JSON has no aggregate entry for {task}; "
                    "provide the experiment eval.json aggregate instead"
                )
            values[task] = pick_harness_metric(entry, task)
    values["macro_average"] = mean(values[task] for task in TASKS)
    return values


def extract_continuations(path: Path) -> dict[str, float]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"continuation line {line_number} is not an object")
            required = ("id", "prompt", "continuation", "generated_tokens")
            missing = [key for key in required if key not in record]
            if missing:
                raise KeyError(
                    f"continuation line {line_number} missing fields: {','.join(missing)}"
                )
            for key in ("id", "prompt", "continuation"):
                if not isinstance(record[key], str):
                    raise TypeError(
                        f"continuation line {line_number} field {key} is not a string"
                    )
            generated_tokens = finite_number(
                record["generated_tokens"],
                f"line {line_number}.generated_tokens",
            )
            if not generated_tokens.is_integer() or generated_tokens < 0:
                raise ValueError(
                    f"continuation line {line_number} generated_tokens must be "
                    "a non-negative integer"
                )
            records.append(record)
    if not records:
        raise ValueError(f"continuation file is empty: {path}")
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("continuation ids must be unique")
    nonempty = sum(bool(item["continuation"].strip()) for item in records)
    generated = [
        finite_number(item["generated_tokens"], f"{item['id']}.generated_tokens")
        for item in records
    ]
    immediate_empty = sum(
        not item["continuation"].strip()
        and finite_number(item["generated_tokens"], "generated_tokens") <= 1
        for item in records
    )
    return {
        "samples": float(len(records)),
        "nonempty_count": float(nonempty),
        "nonempty_rate": nonempty / len(records),
        "immediate_empty_count": float(immediate_empty),
        "immediate_empty_rate": immediate_empty / len(records),
        "generated_tokens_mean": mean(generated),
    }


def flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}/{metric_component(str(key))}", child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            output[prefix] = number


def build_metrics(args: argparse.Namespace) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    sources: list[str] = []
    # Keep elapsed_seconds for backward compatibility and add the explicit scope.
    if args.validation:
        payload = load_json(args.validation)
        for key in (
            "rows", "target_tokens", "nll", "ppl",
            "elapsed_seconds", "evaluation_elapsed_seconds",
            "total_wall_elapsed_seconds", "target_tokens_per_second",
        ):
            if key in payload and payload[key] is not None:
                metrics[f"eval/validation/{key}"] = finite_number(payload[key], key)
        sources.append(args.validation.name)
    if args.official:
        for task, value in extract_official(load_json(args.official)).items():
            metrics[f"eval/official/{metric_component(task)}_percent"] = value
        sources.append(args.official.name)
    if args.continuation:
        for key, value in extract_continuations(args.continuation).items():
            metrics[f"eval/continuation/{key}"] = value
        sources.append(args.continuation.name)
    if args.latency:
        flatten_numeric("eval/latency", load_json(args.latency), metrics)
        sources.append(args.latency.name)
    if args.system:
        flatten_numeric("eval/system", load_json(args.system), metrics)
        sources.append(args.system.name)
    if not metrics:
        raise ValueError("no metrics collected; provide at least one input artifact")
    return dict(sorted(metrics.items())), sources


def atomic_write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (use --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log existing MiniMind evaluation artifacts to SwanLab."
    )
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--official", type=Path)
    parser.add_argument("--continuation", type=Path)
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--system", type=Path)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--group", default=GROUP)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve() if args.output else None
    # This gate must happen before importing SwanLab or making any network write.
    if output is not None:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists (use --overwrite): {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    for argument in ("validation", "official", "continuation", "latency", "system"):
        path = getattr(args, argument)
        if path is not None:
            resolved = path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"input artifact not found: {resolved}")
            setattr(args, argument, resolved)

    source_artifacts = {
        argument: {
            "path": str(getattr(args, argument)),
            "size_bytes": getattr(args, argument).stat().st_size,
            "sha256": sha256(getattr(args, argument)),
        }
        for argument in ("validation", "official", "continuation", "latency", "system")
        if getattr(args, argument) is not None
    }

    metrics, sources = build_metrics(args)
    payload: dict[str, Any] = {
        "status": "dry_run" if args.dry_run else "logged",
        "project": args.project,
        "experiment_name": args.experiment_name,
        "group": args.group,
        "source_files": sources,
        "source_artifacts": source_artifacts,
        "metrics": metrics,
        "metric_count": len(metrics),
    }
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        import swanlab

        swanlab.init(
            project=args.project,
            experiment_name=args.experiment_name,
            group=args.group,
            job_type="evaluation",
            config={
                "group": args.group,
                "source_files": sources,
                "source_artifacts": source_artifacts,
            },
        )
        swanlab.log(metrics, step=0)
        swanlab.finish()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    if output:
        atomic_write_json(output, payload, args.overwrite)


if __name__ == "__main__":
    main()
