#!/usr/bin/env python3
"""Build answer-balanced, free-form modular arithmetic for Phase5 R03."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

OPS = ("add", "sub")
SPLIT_PER_BUCKET = {"train": 200, "validation": 25, "test": 50}
TEMPLATES = {
    "add": [
        "计算 ({a}+{b}) mod 10。只输出0到9之间的最终数字，不要解释。",
        "求 {a}+{b} 除以10的余数。只回答一个数字。",
        "What is ({a}+{b}) modulo 10? Reply with one digit only.",
        "取 {a} 与 {b} 之和的个位数。严格只输出该数字。",
    ],
    "sub": [
        "计算 ({a}-{b}) mod 10。只输出0到9之间的最终数字，不要解释。",
        "求 {a}-{b} 除以10的非负余数。只回答一个数字。",
        "What is ({a}-{b}) modulo 10 using a non-negative remainder? Reply with one digit only.",
        "取 {a} 减 {b} 后按模10得到的数字。严格只输出该数字。",
    ],
}


def stable(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), 16)


def answer_for(op: str, a: int, b: int) -> int:
    return (a + b) % 10 if op == "add" else (a - b) % 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_replay(path: Path, count: int, seed: int):
    candidates = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            turns = row.get("conversations", [])
            if len(turns) != 2 or turns[0].get("role") != "user" or turns[1].get("role") != "assistant":
                continue
            prompt, answer = turns[0].get("content", ""), turns[1].get("content", "")
            if not prompt or not answer or len(prompt) > 360 or len(answer) > 240:
                continue
            candidates.append({"task_id": "replay-" + hashlib.sha256((prompt + "\0" + answer).encode()).hexdigest()[:20], "prompt": prompt, "answer": answer, "kind": "replay", "source": str(path)})
    candidates.sort(key=lambda row: stable(seed, row["task_id"]))
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} replay rows available")
    return candidates[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--replay-data", type=Path, required=True)
    parser.add_argument("--replay-count", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    splits = {name: [] for name in SPLIT_PER_BUCKET}
    for op in OPS:
        for answer in range(10):
            pool = [(a, b) for a in range(100) for b in range(100) if answer_for(op, a, b) == answer]
            pool.sort(key=lambda pair: stable(args.seed, f"{op}:{answer}:{pair[0]}:{pair[1]}"))
            cursor = 0
            for split, count in SPLIT_PER_BUCKET.items():
                part = pool[cursor:cursor + count]
                cursor += count
                for index, (a, b) in enumerate(part):
                    phase = count % len(TEMPLATES[op]) if op == "sub" else 0
                    template_id = (index + phase) % len(TEMPLATES[op])
                    prompt = TEMPLATES[op][template_id].format(a=a, b=b)
                    key = f"{op}:{a}:{b}"
                    splits[split].append({
                        "task_id": hashlib.sha256(key.encode()).hexdigest()[:20], "task_key": key,
                        "operation": op, "a": a, "b": b, "prompt": prompt,
                        "answer": str(answer), "answer_value": answer, "template_id": template_id,
                        "kind": "math", "split": split,
                    })
    files = {}
    for split, rows in splits.items():
        rows.sort(key=lambda row: stable(args.seed, f"order:{split}:{row['task_key']}"))
        path = args.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[split] = {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256(path)}
    replay = extract_replay(args.replay_data, args.replay_count, args.seed)
    warmstart = splits["train"] + replay
    warmstart.sort(key=lambda row: stable(args.seed, f"warm:{row['task_id']}"))
    warm_path = args.output / "warmstart.jsonl"
    warm_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in warmstart))
    files["warmstart"] = {"path": str(warm_path), "rows": len(warmstart), "math_rows": len(splits["train"]), "replay_rows": len(replay), "bytes": warm_path.stat().st_size, "sha256": sha256(warm_path)}
    manifest = {
        "name": "verifiable-mod10-v1", "seed": args.seed,
        "task": "free-form add/sub modulo 10 with exact integer verifier",
        "reward": "1 only when the entire normalized completion parses to the programmatic integer answer; otherwise 0",
        "split_policy": "task_key disjoint; each split exactly balanced across operation and answers 0-9",
        "templates": TEMPLATES, "files": files,
        "answer_counts": {split: Counter(row["answer"] for row in rows) for split, rows in splits.items()},
        "operation_counts": {split: Counter(row["operation"] for row in rows) for split, rows in splits.items()},
        "warmstart_replay": {"source": str(args.replay_data), "rows": len(replay), "ratio": len(replay) / len(warmstart), "selection": "deterministic SHA-256; two-turn only; prompt<=360 chars; answer<=240 chars"},
        "limitations": ["synthetic modular arithmetic is a narrow verifier task", "task-key isolation does not prove semantic absence from all historical S10 corpora"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
