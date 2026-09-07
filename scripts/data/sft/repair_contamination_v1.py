#!/usr/bin/env python3
"""Deterministically replace SFT rows rejected by the frozen eval audit."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_contamination_v1 import (
    containment_overlaps,
    exact_overlaps,
    load_eval,
    load_training,
    near_overlaps,
    shingles,
    simhash,
)
from build_sft_v1 import (
    BuildError,
    NearIndex,
    file_sha256,
    normalize_prompt,
    sha256_bytes,
    stable_json,
    utc_now,
    write_outputs,
)


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--candidate-db", type=Path)
    parser.add_argument("--eval-reference", type=Path)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


class EvalNearGuard:
    def __init__(self, eval_rows: list[dict[str, Any]]) -> None:
        self.eval_rows = eval_rows
        self.bands: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.signatures: list[int] = []
        self.shingle_cache: dict[int, set[int]] = {}
        for index, row in enumerate(eval_rows):
            signature = simhash(shingles(row["normalized_prompt"]))
            self.signatures.append(signature)
            for band in range(8):
                key = (band, (signature >> (band * 8)) & 0xFF)
                self.bands[key].append(index)

    def contains_near(self, prompt: str) -> bool:
        values = shingles(prompt)
        signature = simhash(values)
        candidates: set[int] = set()
        for band in range(8):
            key = (band, (signature >> (band * 8)) & 0xFF)
            candidates.update(self.bands.get(key, ()))
        for index in sorted(candidates):
            if (signature ^ self.signatures[index]).bit_count() > 20:
                continue
            other = self.shingle_cache.get(index)
            if other is None:
                other = shingles(self.eval_rows[index]["normalized_prompt"])
                self.shingle_cache[index] = other
            union = values | other
            if not union:
                continue
            score = len(values & other) / len(union)
            if 0.80 <= score < 1.0:
                return True
        return False


class EvalGuard:
    def __init__(self, eval_rows: list[dict[str, Any]]) -> None:
        self.eval_rows = eval_rows
        self.exact = {row["prompt_sha256"] for row in eval_rows}
        self.containment_prompts = [
            row["normalized_prompt"]
            for row in eval_rows
            if len(row["normalized_prompt"]) >= 20
        ]
        self.near = EvalNearGuard(eval_rows)

    def rejection_reason(self, prompt: str) -> str | None:
        if sha256_bytes(prompt.encode()) in self.exact:
            return "eval_exact"
        if len(prompt) >= 20:
            for other in self.containment_prompts:
                if prompt != other and (other in prompt or prompt in other):
                    return "eval_containment"
        if self.near.contains_near(prompt):
            return "eval_near"
        return None


def load_selected(root: Path) -> dict[str, list[dict[str, Any]]]:
    provenance: dict[tuple[str, int], dict[str, Any]] = {}
    with (root / "provenance.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["split"]), int(row["split_line_number"]))
            if key in provenance:
                raise BuildError(f"duplicate provenance key: {key}")
            provenance[key] = row
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        with (root / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                payload = json.loads(line)
                key = (split, line_number)
                if key not in provenance:
                    raise BuildError(f"missing provenance: {key}")
                prov = provenance[key]
                selected[split].append(
                    {
                        "candidate_id": str(prov["candidate_id"]),
                        "bucket": str(prov["bucket"]),
                        "selection_rank": "",
                        "payload_json": stable_json(payload),
                        "provenance_json": stable_json(prov),
                    }
                )
    if sum(map(len, selected.values())) != len(provenance):
        raise BuildError("orphan provenance rows exist")
    return selected


def overlap_ids(
    input_root: Path,
    eval_rows: list[dict[str, Any]],
) -> tuple[set[str], dict[str, list[str]]]:
    training = load_training(input_root)
    matches = (
        exact_overlaps(training, eval_rows)
        + containment_overlaps(training, eval_rows)
        + near_overlaps(training, eval_rows)
    )
    by_training: dict[str, list[str]] = defaultdict(list)
    for match in matches:
        by_training[str(match["training_id"])].append(str(match["kind"]))
    return set(by_training), dict(sorted(by_training.items()))


def selected_state(
    selected: dict[str, list[dict[str, Any]]],
) -> tuple[set[str], set[str], set[str], NearIndex, dict[str, Counter[str]], dict[str, Counter[str]]]:
    ids: set[str] = set()
    exact: set[str] = set()
    prompts: set[str] = set()
    near = NearIndex()
    rows = {split: Counter() for split in SPLITS}
    tokens = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        for item in selected[split]:
            payload = json.loads(item["payload_json"])
            prov = json.loads(item["provenance_json"])
            candidate_id = str(item["candidate_id"])
            prompt = normalize_prompt(payload["conversations"])
            exact_digest = sha256_bytes(item["payload_json"].encode())
            prompt_digest = sha256_bytes(prompt.encode())
            if candidate_id in ids or exact_digest in exact or prompt_digest in prompts:
                raise BuildError("remaining selected rows are not exactly unique")
            ids.add(candidate_id)
            exact.add(exact_digest)
            prompts.add(prompt_digest)
            near.add(candidate_id, prompt)
            bucket = str(item["bucket"])
            rows[split][bucket] += 1
            tokens[split][bucket] += int(prov["shifted_assistant_target_tokens"])
    return ids, exact, prompts, near, rows, tokens


def run_self_test() -> int:
    eval_rows = [
        {
            "_id": "eval:1",
            "task": "fixture",
            "normalized_prompt": "",
            "prompt_sha256": sha256_bytes(b""),
        },
        {
            "_id": "eval:2",
            "task": "fixture",
            "normalized_prompt": "a sufficiently long evaluation prompt",
            "prompt_sha256": sha256_bytes(b"a sufficiently long evaluation prompt"),
        },
    ]
    guard = EvalGuard(eval_rows)
    assert guard.rejection_reason("") == "eval_exact"
    assert guard.rejection_reason("prefix a sufficiently long evaluation prompt suffix") == "eval_containment"
    assert guard.rejection_reason("completely unrelated material") is None
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    required = {
        "--input-root": args.input_root,
        "--output-root": args.output_root,
        "--candidate-db": args.candidate_db,
        "--eval-reference": args.eval_reference,
        "--eval-manifest": args.eval_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise BuildError(f"missing required arguments: {missing}")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    candidate_db = args.candidate_db.resolve()
    eval_reference = args.eval_reference.resolve()
    eval_manifest_path = args.eval_manifest.resolve()
    if output_root.exists():
        raise BuildError(f"output already exists: {output_root}")
    eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if file_sha256(eval_reference) != eval_manifest["output_sha256"]:
        raise BuildError("eval reference sha256 differs from manifest")
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_rows = load_eval(eval_reference)
    bad_ids, overlap_kinds = overlap_ids(input_root, eval_rows)
    if not bad_ids:
        raise BuildError("input has no eval overlaps to repair")
    selected = load_selected(input_root)
    removed: list[dict[str, Any]] = []
    for split in SPLITS:
        kept = []
        for line_number, item in enumerate(selected[split], 1):
            training_id = f"{split}:{line_number}"
            if training_id in bad_ids:
                prov = json.loads(item["provenance_json"])
                removed.append(
                    {
                        "training_id": training_id,
                        "candidate_id": item["candidate_id"],
                        "bucket": item["bucket"],
                        "targets": int(prov["shifted_assistant_target_tokens"]),
                        "overlap_kinds": overlap_kinds[training_id],
                    }
                )
            else:
                kept.append(item)
        selected[split] = kept
    ids, exact, prompts, near, rows, tokens = selected_state(selected)
    removed_ids = {item["candidate_id"] for item in removed}
    guard = EvalGuard(eval_rows)
    repair_rejections: Counter[str] = Counter()
    replacements: list[dict[str, Any]] = []
    connection = sqlite3.connect(f"file:{candidate_db}?mode=ro", uri=True)
    try:
        quotas = manifest["selection"]["quotas"]
        for split in SPLITS:
            for bucket, quota in quotas[split].items():
                if tokens[split][bucket] >= int(quota):
                    continue
                query = """
                  SELECT candidate_id, selection_rank, valid_targets, prompt_norm,
                         prompt_digest, exact_digest, payload_json, provenance_json
                  FROM candidates
                  WHERE split = ? AND bucket = ?
                  ORDER BY selection_rank, candidate_id
                """
                for value in connection.execute(query, (split, bucket)):
                    (
                        candidate_id, rank, valid_targets, prompt, prompt_digest,
                        exact_digest, payload_json, provenance_json,
                    ) = value
                    if candidate_id in ids or candidate_id in removed_ids:
                        continue
                    if exact_digest in exact:
                        repair_rejections["exact_conversation"] += 1
                        continue
                    if prompt_digest in prompts:
                        repair_rejections["exact_prompt"] += 1
                        continue
                    if near.contains_near(prompt):
                        repair_rejections["near_prompt"] += 1
                        continue
                    reason = guard.rejection_reason(prompt)
                    if reason:
                        repair_rejections[reason] += 1
                        continue
                    item = {
                        "candidate_id": candidate_id,
                        "bucket": bucket,
                        "selection_rank": rank,
                        "payload_json": payload_json,
                        "provenance_json": provenance_json,
                    }
                    selected[split].append(item)
                    ids.add(candidate_id)
                    exact.add(exact_digest)
                    prompts.add(prompt_digest)
                    near.add(candidate_id, prompt)
                    rows[split][bucket] += 1
                    tokens[split][bucket] += int(valid_targets)
                    replacements.append(
                        {
                            "candidate_id": candidate_id,
                            "split": split,
                            "bucket": bucket,
                            "targets": int(valid_targets),
                            "selection_rank": rank,
                        }
                    )
                    if tokens[split][bucket] >= int(quota):
                        break
                if tokens[split][bucket] < int(quota):
                    raise BuildError(
                        f"repair capacity shortage: {split}/{bucket} "
                        f"{tokens[split][bucket]} < {quota}"
                    )
    finally:
        connection.close()
    manifest["created_at"] = utc_now()
    manifest["status"] = "pending_external_audit"
    manifest["trainable"] = False
    manifest["selection"]["selected_rows"] = {
        split: dict(rows[split]) for split in SPLITS
    }
    manifest["selection"]["selected_assistant_tokens"] = {
        split: dict(tokens[split]) for split in SPLITS
    }
    manifest["selection"]["dedup_rejections"].update(
        {f"repair_{key}": value for key, value in sorted(repair_rejections.items())}
    )
    manifest["bindings"]["contamination_repair"] = "scripts/data/sft/repair_contamination_v1.py"
    manifest["bindings"]["contamination_repair_sha256"] = file_sha256(Path(__file__))
    manifest["repair"] = {
        "policy": "remove_all_frozen_eval_overlaps_then_top_up_same_split_and_bucket",
        "input_root": str(input_root),
        "input_manifest_sha256": file_sha256(manifest_path),
        "candidate_db": str(candidate_db),
        "eval_reference": str(eval_reference),
        "eval_reference_sha256": eval_manifest["output_sha256"],
        "eval_manifest_sha256": file_sha256(eval_manifest_path),
        "removed": removed,
        "replacements": replacements,
    }
    manifest["audit"] = {
        "required": True,
        "status": "pending",
        "success_marker_may_only_be_written_by": "scripts/data/sft/audit_sft_v1.py",
    }
    write_outputs(output_root, selected, manifest)
    print(
        json.dumps(
            {
                "status": "pending_external_audit",
                "output": str(output_root),
                "removed": len(removed),
                "replacements": len(replacements),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
