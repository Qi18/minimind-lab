#!/usr/bin/env python3
"""Fail-closed audit for the Phase5 verifiable arithmetic dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if c.isalnum())


def expected(row):
    if row["operation"] == "add":
        return row["a"] + row["b"]
    if row["operation"] == "sub":
        return row["a"] - row["b"]
    if row["operation"] == "mul":
        return row["a"] * row["b"]
    raise AssertionError(row["operation"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    all_rows = {}
    failures = []
    for split in ("train", "validation", "test"):
        path = args.data / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        all_rows[split] = rows
        entry = manifest["files"][split]
        if len(rows) != entry["rows"] or sha(path) != entry["sha256"]:
            failures.append(f"{split}: manifest mismatch")
        for row in rows:
            if row["split"] != split:
                failures.append(f"{split}: embedded split mismatch")
            if expected(row) != row["answer_value"]:
                failures.append(f"{split}: arithmetic mismatch {row['task_key']}")
            if set(row["options"]) != set("ABCD") or len(set(row["options"].values())) != 4:
                failures.append(f"{split}: invalid options {row['task_key']}")
            if row["options"][row["answer"]] != row["answer_value"]:
                failures.append(f"{split}: label mismatch {row['task_key']}")
    overlap = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field, transform in (("task_key", lambda x: x), ("prompt", normalized)):
            a = {transform(r[field]) for r in all_rows[left]}
            b = {transform(r[field]) for r in all_rows[right]}
            overlap[f"{left}_{right}_{field}"] = len(a & b)
            if a & b:
                failures.append(f"{left}/{right}: {field} overlap")
    profiles = {}
    for split, rows in all_rows.items():
        profiles[split] = {
            "rows": len(rows),
            "operation": Counter(r["operation"] for r in rows),
            "answer_position": Counter(r["answer"] for r in rows),
            "template": Counter(str(r["template_id"]) for r in rows),
            "unique_task_keys": len({r["task_key"] for r in rows}),
            "unique_normalized_prompts": len({normalized(r["prompt"]) for r in rows}),
            "answer_value_min": min(r["answer_value"] for r in rows),
            "answer_value_max": max(r["answer_value"] for r in rows),
        }
        if max(profiles[split]["answer_position"].values()) - min(profiles[split]["answer_position"].values()) > 2:
            failures.append(f"{split}: answer positions imbalanced")
    report = {
        "status": "passed" if not failures else "failed",
        "dataset": str(args.data),
        "manifest_sha256": sha(args.data / "manifest.json"),
        "profiles": profiles,
        "cross_split_overlap": overlap,
        "checks": [
            "manifest hashes and row counts",
            "programmatic arithmetic and option uniqueness",
            "answer label correctness",
            "task-key and normalized-prompt split isolation",
            "answer-position balance",
        ],
        "failures": failures,
        "boundary": "The audit proves internal Phase5 split isolation, not absence from all historical S10 pretraining/SFT corpora.",
    }
    output = args.output or args.data / "audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if failures:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    (args.data / "_SUCCESS").write_text("passed\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
