#!/usr/bin/env python3
"""Build a deterministic train/validation/test exact-verifier arithmetic benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

OPS = {"add": "+", "sub": "-", "mul": "*"}
LABELS = "ABCD"
TEMPLATES = [
    "{expr} 等于多少？从下面选项中选择正确答案，只回答选项字母。\n{options}",
    "计算 {expr}。请选择正确结果，并且只输出 A、B、C 或 D。\n{options}",
    "Which option is the exact value of {expr}? Reply with only A, B, C, or D.\n{options}",
    "请完成这道算术题：{expr}\n候选答案：\n{options}\n仅返回正确选项的字母。",
]
SPLIT_COUNTS = {
    "add": {"train": 400, "validation": 80, "test": 170},
    "sub": {"train": 400, "validation": 80, "test": 170},
    "mul": {"train": 200, "validation": 40, "test": 60},
}


def stable(seed: int, text: str) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode()).hexdigest(), 16)


def value(op: str, a: int, b: int) -> int:
    return a + b if op == "add" else a - b if op == "sub" else a * b


def pool(op: str):
    if op == "add":
        return [(a, b) for a in range(50) for b in range(a, 50)]
    if op == "sub":
        return [(a, b) for a in range(100) for b in range(a + 1)]
    return [(a, b) for a in range(2, 26) for b in range(a, 26)]


def distractors(op: str, a: int, b: int, correct: int, seed: int):
    candidates = [
        correct - 1, correct + 1, correct - 2, correct + 2,
        correct - 10, correct + 10, a + b, abs(a - b), a * b,
    ]
    rng = random.Random(stable(seed, f"distractors:{op}:{a}:{b}"))
    candidates += [correct + rng.randint(-20, 20) for _ in range(20)]
    result = []
    for item in candidates:
        if item != correct and item not in result:
            result.append(item)
        if len(result) == 3:
            return result
    raise RuntimeError("could not build distractors")


def render(op: str, a: int, b: int, split: str, index: int, seed: int):
    correct = value(op, a, b)
    correct_position = (index + stable(seed, f"position:{split}:{op}") % 4) % 4
    template_id = (index + stable(seed, f"template:{split}:{op}") % len(TEMPLATES)) % len(TEMPLATES)
    wrong = distractors(op, a, b, correct, seed)
    option_values = wrong[:]
    option_values.insert(correct_position, correct)
    options = "\n".join(f"{label}. {number}" for label, number in zip(LABELS, option_values))
    expr = f"{a}{OPS[op]}{b}"
    prompt = TEMPLATES[template_id].format(expr=expr, options=options)
    return {
        "task_id": hashlib.sha256(f"{op}:{a}:{b}".encode()).hexdigest()[:20],
        "task_key": f"{op}:{a}:{b}",
        "operation": op,
        "a": a,
        "b": b,
        "expression": expr,
        "prompt": prompt,
        "options": dict(zip(LABELS, option_values)),
        "answer": LABELS[correct_position],
        "answer_value": correct,
        "template_id": template_id,
        "split": split,
    }


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows_by_split = {s: [] for s in ("train", "validation", "test")}
    for op in OPS:
        need = sum(SPLIT_COUNTS[op].values())
        selected = sorted(pool(op), key=lambda x: stable(args.seed, f"select:{op}:{x[0]}:{x[1]}"))[:need]
        selected = sorted(selected, key=lambda x: stable(args.seed, f"split:{op}:{x[0]}:{x[1]}"))
        cursor = 0
        for split in ("train", "validation", "test"):
            count = SPLIT_COUNTS[op][split]
            part = selected[cursor:cursor + count]
            cursor += count
            part = sorted(part, key=lambda x: stable(args.seed, f"render:{split}:{op}:{x[0]}:{x[1]}"))
            rows_by_split[split].extend(render(op, a, b, split, i, args.seed) for i, (a, b) in enumerate(part))
    files = {}
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda r: stable(args.seed, f"order:{split}:{r['task_key']}"))
        path = args.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        files[split] = {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha(path)}
    manifest = {
        "name": "verifiable-math-v1",
        "seed": args.seed,
        "generator": str(Path(__file__).resolve()),
        "task": "four-way exact-verifier arithmetic multiple choice",
        "reward": "1 when one unambiguous leading/correct-answer option label matches the programmatic answer; otherwise 0; strict-letter format is tracked separately",
        "split_policy": "task_key disjoint, stratified by operation, deterministic SHA-256 ordering",
        "counts": {s: Counter(r["operation"] for r in rows) for s, rows in rows_by_split.items()},
        "files": files,
        "limitations": [
            "synthetic narrow-domain arithmetic is not GSM8K reasoning",
            "historical S10 training data cannot be proven semantically disjoint",
            "multiple-choice accuracy may not transfer to free-form arithmetic",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
