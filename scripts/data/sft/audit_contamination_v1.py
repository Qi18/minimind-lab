#!/usr/bin/env python3
"""Audit SFT prompts against a frozen fixed-evaluation prompt reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")


class ContaminationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--eval-reference", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prompt_from_row(row: dict[str, Any]) -> str:
    messages = row.get("conversations")
    if not isinstance(messages, list):
        raise ContaminationError("conversations is not a list")
    prompt = normalize(
        "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        )
    )
    if not prompt:
        raise ContaminationError("normalized user prompt is empty")
    return prompt


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shingles(text: str, size: int = 5) -> set[int]:
    return {
        int.from_bytes(
            hashlib.blake2b(text[index : index + size].encode(), digest_size=8).digest(),
            "big",
        )
        for index in range(max(0, len(text) - size + 1))
    }


def simhash(values: set[int]) -> int:
    scores = [0] * 64
    for value in values:
        for bit in range(64):
            scores[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(scores):
        if score >= 0:
            result |= 1 << bit
    return result


def windows(value: str, size: int = 20) -> set[str]:
    if len(value) < size:
        return set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def load_eval(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            prompt = str(value["normalized_prompt"])
            if prompt != normalize(prompt):
                raise ContaminationError(f"eval prompt is not normalized at line {line_number}")
            value["_id"] = f"eval:{line_number}"
            result.append(value)
    if not result:
        raise ContaminationError("eval reference is empty")
    return result


def load_training(root: Path) -> list[dict[str, Any]]:
    result = []
    for split in SPLITS:
        with (root / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                result.append(
                    {
                        "_id": f"{split}:{line_number}",
                        "split": split,
                        "prompt": prompt_from_row(row),
                    }
                )
    if not result:
        raise ContaminationError("training payload is empty")
    return result


def exact_overlaps(
    training: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        by_digest[row["prompt_sha256"]].append(row)
    overlaps = []
    for row in training:
        digest = sha256_bytes(row["prompt"].encode())
        for matched in by_digest.get(digest, ()):
            overlaps.append(
                {
                    "kind": "exact",
                    "training_id": row["_id"],
                    "eval_id": matched["_id"],
                    "task": matched["task"],
                }
            )
    return overlaps


def containment_overlaps(
    training: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    minimum = 20
    eval_by_prefix: dict[str, list[int]] = defaultdict(list)
    training_anchors: set[str] = set()
    for index, row in enumerate(eval_rows):
        prompt = row["normalized_prompt"]
        if len(prompt) >= minimum:
            eval_by_prefix[prompt[:minimum]].append(index)
    for row in training:
        if len(row["prompt"]) >= minimum:
            training_anchors.add(row["prompt"][:minimum])
    eval_windows_for_training: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(eval_rows):
        prompt = row["normalized_prompt"]
        for window in windows(prompt, minimum):
            if window in training_anchors:
                eval_windows_for_training[window].append(index)
    overlaps = []
    seen: set[tuple[str, str]] = set()
    for row in training:
        prompt = row["prompt"]
        if len(prompt) < minimum:
            continue
        candidate_eval_ids: set[int] = set()
        for window in windows(prompt, minimum):
            candidate_eval_ids.update(eval_by_prefix.get(window, ()))
        for eval_index in candidate_eval_ids:
            eval_prompt = eval_rows[eval_index]["normalized_prompt"]
            if eval_prompt in prompt and eval_prompt != prompt:
                key = (row["_id"], eval_rows[eval_index]["_id"])
                if key not in seen:
                    seen.add(key)
                    overlaps.append(
                        {
                            "kind": "eval_prompt_contained_in_training",
                            "training_id": row["_id"],
                            "eval_id": eval_rows[eval_index]["_id"],
                            "task": eval_rows[eval_index]["task"],
                        }
                    )
        anchor = prompt[:minimum]
        for eval_index in eval_windows_for_training.get(anchor, ()):
            eval_prompt = eval_rows[eval_index]["normalized_prompt"]
            if prompt in eval_prompt and prompt != eval_prompt:
                key = (row["_id"], eval_rows[eval_index]["_id"])
                if key not in seen:
                    seen.add(key)
                    overlaps.append(
                        {
                            "kind": "training_prompt_contained_in_eval",
                            "training_id": row["_id"],
                            "eval_id": eval_rows[eval_index]["_id"],
                            "task": eval_rows[eval_index]["task"],
                        }
                    )
    return overlaps


def near_overlaps(
    training: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bands: dict[tuple[int, int], list[int]] = defaultdict(list)
    signatures: list[int] = []
    for index, row in enumerate(eval_rows):
        signature = simhash(shingles(row["normalized_prompt"]))
        signatures.append(signature)
        for band in range(8):
            bands[(band, (signature >> (band * 8)) & 0xFF)].append(index)
    eval_shingle_cache: dict[int, set[int]] = {}
    overlaps = []
    for row in training:
        train_values = shingles(row["prompt"])
        signature = simhash(train_values)
        candidates: set[int] = set()
        for band in range(8):
            candidates.update(bands[(band, (signature >> (band * 8)) & 0xFF)])
        for eval_index in sorted(candidates):
            hamming = (signature ^ signatures[eval_index]).bit_count()
            if hamming > 20:
                continue
            eval_values = eval_shingle_cache.get(eval_index)
            if eval_values is None:
                eval_values = shingles(eval_rows[eval_index]["normalized_prompt"])
                eval_shingle_cache[eval_index] = eval_values
            union = train_values | eval_values
            if not union:
                continue
            score = len(train_values & eval_values) / len(union)
            if 0.80 <= score < 1.0:
                overlaps.append(
                    {
                        "kind": "near",
                        "training_id": row["_id"],
                        "eval_id": eval_rows[eval_index]["_id"],
                        "task": eval_rows[eval_index]["task"],
                        "jaccard": round(score, 6),
                    }
                )
    return overlaps


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    eval_reference = args.eval_reference.resolve()
    eval_manifest_path = args.eval_manifest.resolve()
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(eval_reference) != eval_manifest["output_sha256"]:
        raise ContaminationError("eval reference sha256 differs from manifest")
    training = load_training(root)
    eval_rows = load_eval(eval_reference)
    exact = exact_overlaps(training, eval_rows)
    containment = containment_overlaps(training, eval_rows)
    near = near_overlaps(training, eval_rows)
    status = "accepted" if not exact and not containment and not near else "rejected"
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scan_scope": "full_final_train_validation_test",
        "training_rows": len(training),
        "eval_reference_rows": len(eval_rows),
        "eval_reference": str(args.eval_reference),
        "eval_reference_sha256": eval_manifest["output_sha256"],
        "eval_manifest": str(args.eval_manifest),
        "eval_manifest_sha256": file_sha256(eval_manifest_path),
        "prompt_builder_sha256": eval_manifest["prompt_builder_sha256"],
        "auditor": "scripts/data/sft/audit_contamination_v1.py",
        "auditor_sha256": file_sha256(Path(__file__)),
        "exact_overlap_count": len(exact),
        "containment_overlap_count": len(containment),
        "near_overlap_count": len(near),
        "overlap_examples": (exact + containment + near)[:100],
        "thresholds": {
            "containment_min_normalized_chars": 20,
            "near_char_ngram": 5,
            "near_jaccard_threshold": 0.80,
            "simhash_hamming_prefilter_max": 20,
        },
    }
    (root / "contamination_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "training_rows": len(training),
                "eval_reference_rows": len(eval_rows),
                "exact": len(exact),
                "containment": len(containment),
                "near": len(near),
            },
            ensure_ascii=False,
        )
    )
    return 0 if status == "accepted" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContaminationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
