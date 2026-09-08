#!/usr/bin/env python3
"""Audit Phase5 R04 target-extraction data before training."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED = {"train": 4000, "validation": 500, "test": 1000}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    datasets = {}
    profiles = {}
    for split, expected in EXPECTED.items():
        path = args.data / f"{split}.jsonl"
        rows = read_jsonl(path)
        datasets[split] = rows
        answers = Counter(row["answer"] for row in rows)
        templates = Counter(str(row["template_id"]) for row in rows)
        if len(rows) != expected:
            failures.append(f"{split}: expected {expected} rows, got {len(rows)}")
        if answers != Counter({str(value): expected // 10 for value in range(10)}):
            failures.append(f"{split}: answer balance mismatch")
        if max(templates.values()) - min(templates.values()) > 1:
            failures.append(f"{split}: template imbalance exceeds one row")
        if len({row["task_key"] for row in rows}) != len(rows):
            failures.append(f"{split}: duplicate task_key")
        if any(row["kind"] != "target_extract" for row in rows):
            failures.append(f"{split}: unexpected kind")
        profiles[split] = {
            "rows": len(rows),
            "answers": answers,
            "templates": templates,
            "positions": Counter(str(row["position"]) for row in rows),
            "sha256": sha256(path),
        }
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = {row["task_key"] for row in datasets[left]} & {
            row["task_key"] for row in datasets[right]
        }
        overlaps[f"{left}_{right}"] = len(overlap)
        if overlap:
            failures.append(f"{left}/{right}: {len(overlap)} overlapping task keys")
    result = {
        "status": "passed" if not failures else "failed",
        "dataset": str(args.data),
        "profiles": profiles,
        "cross_split_task_overlap": overlaps,
        "failures": failures,
        "boundary": "Narrow marked-target extraction; does not measure general reasoning.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
