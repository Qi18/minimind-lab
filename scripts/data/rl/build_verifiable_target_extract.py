#!/usr/bin/env python3
"""Build answer-balanced target-digit extraction data for Phase5 R04."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SPLIT_ROWS = {"train": 4000, "validation": 500, "test": 1000}
TEMPLATES = (
    "数字序列：{left}[{answer}]{right}。只输出方括号内的一个数字，不要解释。",
    "读取记录：{left}<TARGET>{answer}</TARGET>{right}。严格只回答 TARGET 标签中的数字。",
    "候选项：{left}<{answer}>{right}。只返回尖括号标记的一个数字，不要解释。",
    "校验串：{left}#{answer}#{right}。严格只输出两个 # 之间的数字。",
)


def stable(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(template_id: int, digits: list[int], position: int) -> str:
    left = " ".join(map(str, digits[:position]))
    right = " ".join(map(str, digits[position + 1 :]))
    if left:
        left += " "
    if right:
        right = " " + right
    return TEMPLATES[template_id].format(left=left, answer=digits[position], right=right)


def build_split(split: str, count: int, seed: int, used_prompts: set[str]) -> list[dict]:
    if count % 10:
        raise ValueError("every split must be exactly answer-balanced")
    rows = []
    per_answer = count // 10
    for answer in range(10):
        accepted = 0
        attempt = 0
        while accepted < per_answer:
            template_id = (accepted + answer * per_answer) % len(TEMPLATES)
            rng = random.Random(stable(seed, f"{split}:{answer}:{attempt}"))
            position = rng.randrange(6)
            digits = [rng.randrange(10) for _ in range(6)]
            digits[position] = answer
            prompt = render(template_id, digits, position)
            attempt += 1
            if prompt in used_prompts:
                continue
            used_prompts.add(prompt)
            task_key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
            rows.append({
                "task_id": task_key,
                "task_key": task_key,
                "prompt": prompt,
                "answer": str(answer),
                "answer_value": answer,
                "operation": "target_extract",
                "position": position,
                "template_id": template_id,
                "kind": "target_extract",
                "split": split,
            })
            accepted += 1
    rows.sort(key=lambda row: stable(seed, f"order:{split}:{row['task_id']}"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    used_prompts: set[str] = set()
    files = {}
    profiles = {}
    for split, count in SPLIT_ROWS.items():
        rows = build_split(split, count, args.seed, used_prompts)
        path = args.output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        files[split] = {
            "path": str(path),
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        profiles[split] = {
            "answers": Counter(row["answer"] for row in rows),
            "templates": Counter(str(row["template_id"]) for row in rows),
            "positions": Counter(str(row["position"]) for row in rows),
        }
    manifest = {
        "name": "verifiable-target-extract-v1",
        "seed": args.seed,
        "task": "copy the uniquely marked target digit and emit exactly one digit",
        "reward": "1 only when the entire normalized completion parses to the marked integer; otherwise 0",
        "split_policy": "prompt and task_key disjoint; each split exactly balanced over answers 0-9",
        "templates": TEMPLATES,
        "files": files,
        "profiles": profiles,
        "limitations": [
            "This is a narrow controlled verifier task, not a general reasoning benchmark.",
            "Synthetic prompt isolation does not imply absence of semantically similar historical data.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
