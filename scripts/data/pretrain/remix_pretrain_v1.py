#!/usr/bin/env python3
"""Repair-mix MiniMind Pretrain v1 candidates with pinned eval filtering.

This script never materializes new source objects. It validates and reuses the
builder's immutable candidate pool, excludes benchmark-contaminated rows before
selection, and writes a fresh pending-audit corpus with deterministic evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

import audit_pretrain_v1 as auditor
import build_pretrain_v1 as builder


REPAIR_VERSION = "pretrain-v1-repair-mixer-1"
EXPECTED_CANDIDATE_OBJECTS = 54
EXPECTED_PINNED_SOURCES = 7
EXPECTED_PINNED_CONFIGS = 124
EXPECTED_PINNED_ROWS = 29638


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def semantic_sha256(value: Any) -> str:
    return sha256_text(builder.canonical_json(value))


def load_yaml_snapshot(
    path: Path,
    expected_schema_version: int,
    label: str,
) -> tuple[dict[str, Any], str]:
    payload = Path(path).read_bytes()
    value = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: YAML root must be a mapping")
    if int(value.get("schema_version", 0)) != expected_schema_version:
        raise ValueError(
            f"{path}: expected {label} schema_version="
            f"{expected_schema_version}, "
            f"got {value.get('schema_version')!r}"
        )
    return value, hashlib.sha256(payload).hexdigest()


def load_eval_config(path: Path) -> tuple[dict[str, Any], str]:
    value, fingerprint = load_yaml_snapshot(path, 2, "benchmark")
    if not isinstance(value.get("sources"), list):
        raise ValueError(f"{path}: sources must be a list")
    return value, fingerprint


def canonicalize_patterns(
    patterns: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove loader-order dependence and reject malformed pattern evidence."""
    canonical: list[dict[str, Any]] = []
    normalized_seen: set[str] = set()
    for original in patterns:
        normalized = original.get("normalized")
        entries_value = original.get("entries")
        if not isinstance(normalized, str) or not normalized:
            raise ValueError("benchmark pattern has empty normalized text")
        if auditor.normalize(normalized) != normalized:
            raise ValueError("benchmark pattern is not canonically normalized")
        if normalized in normalized_seen:
            raise ValueError("duplicate normalized benchmark pattern")
        normalized_seen.add(normalized)
        if not isinstance(entries_value, list) or not entries_value:
            raise ValueError("benchmark pattern has no source entries")
        entries: list[dict[str, str]] = []
        for entry in entries_value:
            if not isinstance(entry, dict):
                raise ValueError("benchmark pattern entry is not an object")
            source_id = entry.get("source_id")
            task = entry.get("task")
            raw = entry.get("raw")
            if not all(isinstance(item, str) and item for item in (source_id, task, raw)):
                raise ValueError("benchmark pattern entry fields are missing")
            entries.append(
                {
                    "raw": raw,
                    "source_id": source_id,
                    "task": task,
                }
            )
        entries.sort(key=builder.canonical_json)
        canonical.append({"entries": entries, "normalized": normalized})
    canonical.sort(key=lambda item: str(item["normalized"]))
    for identifier, pattern in enumerate(canonical):
        pattern["id"] = identifier
    if not canonical:
        raise ValueError("benchmark pattern set is empty")
    return canonical


def pattern_payload(patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entries": pattern["entries"],
            "normalized": pattern["normalized"],
        }
        for pattern in patterns
    ]


def sorted_config_names(values: Any) -> list[Any]:
    if not isinstance(values, list):
        raise ValueError("benchmark config_names must be a list")
    return sorted(values, key=lambda item: (item is not None, str(item)))


def strict_benchmark_snapshot(
    eval_config: dict[str, Any],
    patterns: list[dict[str, Any]],
    source_results: list[dict[str, Any]],
    load_errors: list[dict[str, Any]],
    matching: dict[str, Any],
) -> dict[str, Any]:
    pin_errors = auditor.validate_pinned_eval_sources(
        eval_config,
        source_results,
        load_errors,
        allow_test_config=False,
    )
    if pin_errors:
        raise RuntimeError("pinned benchmark validation failed: " + "; ".join(pin_errors))
    if load_errors:
        raise RuntimeError("pinned benchmark load errors are forbidden")
    if len(source_results) != EXPECTED_PINNED_SOURCES:
        raise RuntimeError(
            f"benchmark sources={len(source_results)} != {EXPECTED_PINNED_SOURCES}"
        )

    source_rows: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for result in source_results:
        source_id = str(result.get("id", ""))
        if not source_id or source_id in source_ids:
            raise RuntimeError("benchmark result source ids are empty or duplicated")
        source_ids.add(source_id)
        configs = int(result.get("configs", 0))
        succeeded = int(result.get("configs_succeeded", 0))
        failed = int(result.get("configs_failed", 0))
        rows = int(result.get("rows", 0))
        queries = int(result.get("queries", 0))
        empty_queries = int(result.get("empty_queries", 0))
        config_names = sorted_config_names(result.get("config_names", []))
        if len(config_names) != configs:
            raise RuntimeError(f"{source_id}: config_names count differs")
        if succeeded != configs or failed != 0:
            raise RuntimeError(f"{source_id}: not every benchmark config succeeded")
        if queries != rows or empty_queries != 0:
            raise RuntimeError(f"{source_id}: benchmark query/row count differs")
        source_rows.append(
            {
                "config_names": config_names,
                "configs": configs,
                "configs_failed": failed,
                "configs_succeeded": succeeded,
                "empty_queries": empty_queries,
                "id": source_id,
                "queries": queries,
                "repo_id": result.get("repo_id"),
                "revision": result.get("revision"),
                "rows": rows,
                "split": result.get("split"),
                "task": result.get("task"),
            }
        )
    source_rows.sort(key=lambda item: item["id"])
    configs_total = sum(int(item["configs"]) for item in source_rows)
    rows_total = sum(int(item["rows"]) for item in source_rows)
    entry_total = sum(len(pattern["entries"]) for pattern in patterns)
    if configs_total != EXPECTED_PINNED_CONFIGS:
        raise RuntimeError(
            f"benchmark configs={configs_total} != {EXPECTED_PINNED_CONFIGS}"
        )
    if rows_total != EXPECTED_PINNED_ROWS:
        raise RuntimeError(f"benchmark rows={rows_total} != {EXPECTED_PINNED_ROWS}")
    if entry_total != rows_total:
        raise RuntimeError(
            f"benchmark pattern entries={entry_total} != rows={rows_total}"
        )
    pattern_sha = semantic_sha256(pattern_payload(patterns))
    return {
        "configs": configs_total,
        "matching": matching,
        "pattern_entries": entry_total,
        "pattern_sha256": pattern_sha,
        "patterns_unique": len(patterns),
        "rows": rows_total,
        "sources": source_rows,
        "sources_count": len(source_rows),
        "status": "pinned_and_complete",
    }


class ContaminationFilter:
    def __init__(
        self,
        patterns: list[dict[str, Any]],
        matching: dict[str, Any],
    ) -> None:
        self.patterns = patterns
        self.exact_min_chars = int(matching["exact_min_normalized_chars"])
        self.containment_min_chars = int(
            matching["containment_min_normalized_chars"]
        )
        if self.exact_min_chars != 1:
            raise ValueError("repair filtering requires exact_min_normalized_chars=1")
        if self.containment_min_chars < 1:
            raise ValueError("containment minimum must be positive")
        self.exact = {
            str(pattern["normalized"]): pattern for pattern in self.patterns
        }
        self.automaton = auditor.AhoCorasick()
        self.long_pattern_ids: set[int] = set()
        for pattern in self.patterns:
            normalized = str(pattern["normalized"])
            if len(normalized) >= self.containment_min_chars:
                identifier = int(pattern["id"])
                self.long_pattern_ids.add(identifier)
                self.automaton.add(normalized, identifier)
        self.automaton.build()

    @property
    def policy(self) -> dict[str, Any]:
        return {
            "containment_min_normalized_chars": self.containment_min_chars,
            "empty_text": "reject_if_not_text.strip()",
            "exact_min_normalized_chars": self.exact_min_chars,
            "normalization": "NFKC_lower_keep_unicode_alnum",
            "piece_rechecks": ["original", "exact_prefix", "boundary_split"],
        }

    @staticmethod
    def descriptor(pattern: dict[str, Any]) -> dict[str, Any]:
        entries = list(pattern["entries"])
        normalized = str(pattern["normalized"])
        return {
            "eval_sources": sorted({str(entry["source_id"]) for entry in entries}),
            "normalized_chars": len(normalized),
            "query_normalized_sha256": sha256_text(normalized),
            "tasks": sorted({str(entry["task"]) for entry in entries}),
        }

    def classify(self, text: str) -> dict[str, Any] | None:
        if not isinstance(text, str):
            raise TypeError("candidate text must be a string")
        if not text.strip():
            return {
                "matches": [],
                "normalized_length": len(auditor.normalize(text)),
                "normalized_sha256": sha256_text(auditor.normalize(text)),
                "reason": "empty_text",
            }
        normalized = auditor.normalize(text)
        exact = self.exact.get(normalized)
        if exact is not None:
            return {
                "matches": [self.descriptor(exact)],
                "normalized_length": len(normalized),
                "normalized_sha256": sha256_text(normalized),
                "reason": "exact_eval_overlap",
            }
        match_ids = sorted(self.automaton.find(normalized))
        if match_ids:
            return {
                "matches": [self.descriptor(self.patterns[item]) for item in match_ids],
                "normalized_length": len(normalized),
                "normalized_sha256": sha256_text(normalized),
                "reason": "containment_eval_overlap",
            }
        return None


class ExclusionLedger:
    def __init__(self) -> None:
        self.events: dict[str, dict[str, Any]] = {}

    def record(
        self,
        record: dict[str, Any],
        split: str,
        source: str,
        phase: str,
        decision: dict[str, Any],
    ) -> None:
        event: dict[str, Any] = {
            "candidate_id": str(record.get("candidate_id", "")),
            "content_sha256": str(record.get("content_sha256", "")),
            "matches": decision["matches"],
            "normalized_length": int(decision["normalized_length"]),
            "normalized_sha256": str(decision["normalized_sha256"]),
            "phase": phase,
            "reason": str(decision["reason"]),
            "source": source,
            "split": split,
        }
        for key in ("object_id", "piece_index", "piece_of", "sample_key", "source_path"):
            if key in record:
                event[key] = record[key]
        event_id = semantic_sha256(event)
        event["event_id"] = event_id
        self.events.setdefault(event_id, event)

    def sorted_events(self) -> list[dict[str, Any]]:
        return [self.events[key] for key in sorted(self.events)]

    def snapshot(self) -> dict[str, Any]:
        by_phase: Counter[str] = Counter()
        by_reason: Counter[str] = Counter()
        by_source: Counter[str] = Counter()
        by_split: Counter[str] = Counter()
        by_split_source: Counter[str] = Counter()
        for event in self.sorted_events():
            by_phase[str(event["phase"])] += 1
            by_reason[str(event["reason"])] += 1
            by_source[str(event["source"])] += 1
            by_split[str(event["split"])] += 1
            by_split_source[f"{event['split']}:{event['source']}"] += 1
        return {
            "events": len(self.events),
            "events_by_phase": dict(sorted(by_phase.items())),
            "events_by_reason": dict(sorted(by_reason.items())),
            "events_by_source": dict(sorted(by_source.items())),
            "events_by_split": dict(sorted(by_split.items())),
            "events_by_split_source": dict(sorted(by_split_source.items())),
        }

    def write(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for event in self.sorted_events():
                handle.write(builder.canonical_json(event) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        return builder.sha256_file(path)


def reject_and_record(
    contamination: ContaminationFilter,
    ledger: ExclusionLedger,
    record: dict[str, Any],
    split: str,
    source: str,
    phase: str,
) -> dict[str, Any] | None:
    decision = contamination.classify(record["text"])
    if decision is not None:
        ledger.record(record, split, source, phase, decision)
    return decision


def replace_pending_record(
    seen: builder.SeenStore,
    old: dict[str, Any],
    new: dict[str, Any],
    split: str,
) -> bool:
    """Atomically replace the not-yet-written boundary reservation."""
    connection = seen.connection
    old_content = str(old["content_sha256"])
    old_candidate = str(old["candidate_id"])
    new_content = str(new["content_sha256"])
    new_candidate = str(new["candidate_id"])
    old_content_row = connection.execute(
        "SELECT split FROM content WHERE hash=?", (old_content,)
    ).fetchone()
    old_candidate_row = connection.execute(
        "SELECT split FROM candidate WHERE id=?", (old_candidate,)
    ).fetchone()
    if old_content_row != (split,) or old_candidate_row != (split,):
        raise RuntimeError("pending SeenStore reservation is missing")
    if new_content != old_content and connection.execute(
        "SELECT 1 FROM content WHERE hash=?", (new_content,)
    ).fetchone():
        return False
    if new_candidate != old_candidate and connection.execute(
        "SELECT 1 FROM candidate WHERE id=?", (new_candidate,)
    ).fetchone():
        return False
    connection.execute("SAVEPOINT repair_boundary_swap")
    try:
        connection.execute("DELETE FROM content WHERE hash=?", (old_content,))
        connection.execute("DELETE FROM candidate WHERE id=?", (old_candidate,))
        connection.execute(
            "INSERT INTO content(hash, split) VALUES (?, ?)",
            (new_content, split),
        )
        connection.execute(
            "INSERT INTO candidate(id, split) VALUES (?, ?)",
            (new_candidate, split),
        )
    except sqlite3.IntegrityError:
        connection.execute("ROLLBACK TO repair_boundary_swap")
        connection.execute("RELEASE repair_boundary_swap")
        return False
    connection.execute("RELEASE repair_boundary_swap")
    return True


def select_source_filtered(
    records: Iterator[dict[str, Any]],
    quota: int,
    split: str,
    source: str,
    output: Path,
    tokenizer: Any,
    sequence_length: int,
    seen: builder.SeenStore,
    contamination: ContaminationFilter,
    ledger: ExclusionLedger,
) -> dict[str, Any]:
    if quota == 1:
        raise ValueError(
            f"{split}/{source}: a one-token loss quota is not representable"
        )
    if output.exists():
        raise RuntimeError(f"selected output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    totals: dict[str, int] = defaultdict(int)
    pending: dict[str, Any] | None = None

    def write(handle: Any, record: dict[str, Any]) -> None:
        if contamination.classify(record["text"]) is not None:
            raise AssertionError("excluded text reached selected output")
        handle.write(builder.canonical_json(record) + "\n")
        totals["rows"] += 1
        for key in (
            "loss_target_tokens",
            "nonpad_input_tokens",
            "padded_compute_tokens",
            "raw_tokens",
            "text_tokens",
        ):
            totals[key] += int(record[key])

    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                totals["candidates_considered"] += 1
                if str(record.get("source")) != source:
                    raise ValueError(
                        f"{split}/{source}: candidate source differs: "
                        f"{record.get('source')!r}"
                    )
                original_decision = reject_and_record(
                    contamination, ledger, record, split, source, "original"
                )
                if original_decision is not None:
                    totals[f"filter_rejected_{original_decision['reason']}"] += 1
                    continue
                loss_tokens = int(record["loss_target_tokens"])
                if loss_tokens < 2:
                    raise ValueError(f"{split}/{source}: invalid loss token count")
                remaining = quota - totals["selected_budget"]
                if remaining <= 0:
                    break
                if loss_tokens <= remaining:
                    if not seen.reserve(record, split):
                        totals["dedup_rejected"] += 1
                        continue
                    if pending is not None:
                        write(handle, pending)
                    pending = record
                    totals["selected_budget"] += loss_tokens
                    if totals["selected_budget"] == quota:
                        break
                    continue
                if remaining >= 2:
                    piece = builder.exact_prefix_piece(
                        record,
                        remaining - 1,
                        tokenizer,
                        sequence_length,
                    )
                    if piece is None:
                        totals["unrepresentable_prefix"] += 1
                        continue
                    piece_decision = reject_and_record(
                        contamination,
                        ledger,
                        piece,
                        split,
                        source,
                        "exact_prefix",
                    )
                    if piece_decision is not None:
                        totals[f"filter_rejected_{piece_decision['reason']}"] += 1
                        totals["filter_rejected_prefix"] += 1
                        continue
                    if not seen.reserve(piece, split):
                        totals["dedup_rejected"] += 1
                        continue
                    if pending is not None:
                        write(handle, pending)
                    pending = piece
                    totals["selected_budget"] += int(piece["loss_target_tokens"])
                    break
                if remaining == 1 and pending is not None:
                    pieces = builder.split_for_one_extra_target(
                        pending,
                        tokenizer,
                        sequence_length,
                    )
                    if pieces is not None:
                        decisions = [
                            reject_and_record(
                                contamination,
                                ledger,
                                piece,
                                split,
                                source,
                                f"boundary_piece_{index}",
                            )
                            for index, piece in enumerate(pieces)
                        ]
                        if any(decision is not None for decision in decisions):
                            totals["filter_rejected_boundary_split"] += 1
                            for decision in decisions:
                                if decision is not None:
                                    totals[
                                        f"filter_rejected_{decision['reason']}"
                                    ] += 1
                        elif seen.replace_pending_content(pending, pieces, split):
                            write(handle, pieces[0])
                            pending = pieces[1]
                            totals["selected_budget"] += 1
                            break
                        else:
                            totals["dedup_rejected_boundary_split"] += 1
                    else:
                        totals["unrepresentable_boundary_split"] += 1

                    replacement = builder.exact_prefix_piece(
                        record,
                        int(pending["loss_target_tokens"]),
                        tokenizer,
                        sequence_length,
                    )
                    if replacement is None:
                        totals["unrepresentable_boundary_replacement"] += 1
                        continue
                    replacement_decision = reject_and_record(
                        contamination,
                        ledger,
                        replacement,
                        split,
                        source,
                        "boundary_replacement_prefix",
                    )
                    if replacement_decision is not None:
                        totals["filter_rejected_boundary_replacement"] += 1
                        totals[
                            f"filter_rejected_{replacement_decision['reason']}"
                        ] += 1
                        continue
                    if not replace_pending_record(
                        seen, pending, replacement, split
                    ):
                        totals["dedup_rejected_boundary_replacement"] += 1
                        continue
                    pending = replacement
                    totals["selected_budget"] += 1
                    totals["boundary_replacement_selected"] += 1
                    break

            if pending is not None:
                write(handle, pending)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    if (
        totals["selected_budget"] != quota
        or totals["loss_target_tokens"] != quota
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{split}/{source} quota short: {dict(totals)}; expected={quota}"
        )
    temporary.replace(output)
    seen.commit()
    result = dict(totals)
    result.update(
        {
            "output": str(output),
            "output_sha256": builder.sha256_file(output),
            "quota_loss_target_tokens": quota,
            "split": split,
            "status": "ok",
        }
    )
    return result


def expected_candidate_roster(
    sources: list[dict[str, Any]],
    required_sources: set[str],
    candidates_root: Path,
) -> list[dict[str, str]]:
    by_id = builder.validate_source_coverage(sources, required_sources)
    root = candidates_root.resolve()
    roster: list[dict[str, str]] = []
    for source_id in sorted(required_sources):
        source = by_id[source_id]
        for index, obj in enumerate(source["objects"]):
            object_id = builder.object_identity(source, obj, index)
            directory = root / builder.safe_name(source_id)
            roster.append(
                {
                    "done": str(
                        directory / f"{builder.safe_name(object_id)}.done.json"
                    ),
                    "object_id": object_id,
                    "output": str(
                        directory / f"{builder.safe_name(object_id)}.jsonl"
                    ),
                    "repo_id": str(source["repo_id"]),
                    "revision": str(source["revision"]),
                    "source": source_id,
                    "source_path": str(obj.get("path") or obj.get("url")),
                }
            )
    return sorted(roster, key=builder.canonical_json)


def manifest_candidate_roster(
    objects: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return sorted(
        [
            {
                "done": str(Path(str(item.get("done", ""))).resolve()),
                "object_id": str(item.get("object_id", "")),
                "output": str(Path(str(item.get("output", ""))).resolve()),
                "repo_id": str(item.get("repo_id", "")),
                "revision": str(item.get("revision", "")),
                "source": str(item.get("source", "")),
                "source_path": str(item.get("source_path", "")),
            }
            for item in objects
        ],
        key=builder.canonical_json,
    )


def validated_candidate_paths(
    manifest: dict[str, Any],
) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = defaultdict(list)
    identity_keys = (
        "builder_version",
        "candidate_target_loss_tokens",
        "fingerprint",
        "object_id",
        "output",
        "repo_id",
        "revision",
        "source",
        "source_path",
        "status",
    )
    for item in manifest["objects"]:
        path = Path(str(item["output"]))
        done_path = Path(str(item["done"]))
        if not path.is_file() or not done_path.is_file():
            raise FileNotFoundError(path if not path.is_file() else done_path)
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid candidate done evidence: {done_path}") from exc
        for key in identity_keys:
            if done.get(key) != item.get(key):
                raise ValueError(f"candidate done {key} differs: {done_path}")
        actual_sha = builder.sha256_file(path)
        if not (
            actual_sha
            == str(item.get("output_sha256", ""))
            == str(done.get("output_sha256", ""))
        ):
            raise ValueError(f"candidate output SHA evidence differs: {path}")
        if int(item.get("output_size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"candidate output size evidence differs: {path}")
        if int(done.get("output_size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"candidate done size evidence differs: {path}")
        output[str(item["source"])].append(path)
    for paths in output.values():
        paths.sort()
    return dict(output)


def validate_candidate_manifest(
    manifest: dict[str, Any],
    config_sha: str,
    domains: dict[str, str],
    tokenizer_fp: dict[str, Any],
    sharding: dict[str, Any],
    train_quotas: dict[str, int],
    validation_quotas: dict[str, int],
    full_budget: int,
    validation_budget: int,
    mix_key: str,
    sources_config_sha: str,
    expected_roster: list[dict[str, str]],
    expected_fingerprint: str,
    builder_scripts: dict[str, Any],
) -> None:
    required_equal = {
        "builder_version": builder.BUILDER_VERSION,
        "config_sha256": config_sha,
        "full_loss_target_tokens": full_budget,
        "hash_domains": domains,
        "mix_key": mix_key,
        "fingerprint": expected_fingerprint,
        "schema_version": builder.SCHEMA_VERSION,
        "sources_config_sha256": sources_config_sha,
        "sharding": sharding,
        "tokenizer": tokenizer_fp,
        "train_source_quotas": train_quotas,
        "validation_loss_target_tokens": validation_budget,
        "validation_source_quotas": validation_quotas,
    }
    if manifest.get("status") != "materialized":
        raise ValueError("candidate materialization is not complete")
    for key, expected in required_equal.items():
        if manifest.get(key) != expected:
            raise ValueError(f"candidate manifest {key} differs")
    if manifest.get("scripts", {}).get("builder") != builder_scripts["builder"]:
        raise ValueError("candidate builder fingerprint differs")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) != EXPECTED_CANDIDATE_OBJECTS:
        raise ValueError(
            f"candidate objects={len(objects) if isinstance(objects, list) else None} "
            f"!= {EXPECTED_CANDIDATE_OBJECTS}"
        )
    for key in ("object_id", "output", "done", "fingerprint"):
        values = [str(item.get(key, "")) for item in objects]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError(f"candidate object {key} values are empty or duplicated")
    actual_roster = manifest_candidate_roster(objects)
    if actual_roster != expected_roster:
        raise ValueError("candidate object roster differs from sources config")


def repair_script_fingerprint() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        "path": "scripts/data/pretrain/remix_pretrain_v1.py",
        "sha256": builder.sha256_file(path),
        "version": REPAIR_VERSION,
    }


def assert_inputs_unchanged(
    config_path: Path,
    config_sha: str,
    eval_config_path: Path,
    eval_config_sha: str,
    sources_config_path: Path,
    sources_config_sha: str,
    candidate_manifest_path: Path,
    candidate_manifest_sha: str,
    candidate_manifest: dict[str, Any],
    candidate_paths: dict[str, list[Path]],
    token_path: Path,
    tokenizer_fp: dict[str, Any],
    scripts: dict[str, Any],
    repair_script: dict[str, Any],
) -> None:
    for label, path, expected in (
        ("pretrain config", config_path, config_sha),
        ("eval config", eval_config_path, eval_config_sha),
        ("sources config", sources_config_path, sources_config_sha),
        ("candidate manifest", candidate_manifest_path, candidate_manifest_sha),
    ):
        if builder.sha256_file(path) != expected:
            raise RuntimeError(f"{label} changed during repair")
    current_scripts = builder.script_fingerprints()
    expected_scripts = {
        "auditor": scripts["auditor"],
        "builder": scripts["builder"],
    }
    if current_scripts != expected_scripts:
        raise RuntimeError("builder or auditor changed during repair")
    if repair_script_fingerprint() != repair_script:
        raise RuntimeError("repair mixer changed during repair")
    if builder.tokenizer_fingerprint(token_path) != tokenizer_fp:
        raise RuntimeError("tokenizer changed during repair")
    if validated_candidate_paths(candidate_manifest) != candidate_paths:
        raise RuntimeError("candidate object pool changed during repair")


def remix(args: argparse.Namespace) -> int:
    config, config_sha = load_yaml_snapshot(
        args.config, builder.SCHEMA_VERSION, "pretrain"
    )
    eval_config, eval_config_sha = load_eval_config(args.eval_config)
    sources_config, sources_config_sha = load_yaml_snapshot(
        args.sources_config, builder.SCHEMA_VERSION, "sources"
    )
    matching = auditor.matching_contract(eval_config)
    scripts = builder.script_fingerprints()
    repair_script = repair_script_fingerprint()
    scripts["repair_mixer"] = repair_script
    domains = builder.hash_domains(config)
    full_budget = builder.budget_value(config, "full_loss_target_tokens")
    validation_budget = builder.budget_value(
        config, "validation_loss_target_tokens"
    )
    train_quotas = builder.source_targets(config, full_budget, args.mix_key)
    validation_quotas = builder.source_targets(
        config, validation_budget, args.mix_key
    )
    sharding = builder.sharding_contract(config, full_budget, validation_budget)
    if (
        int(sharding["num_train_shards"]) != 40
        or int(sharding["num_validation_shards"]) != 1
    ):
        raise ValueError("repair requires the frozen 40 train + 1 validation contract")
    sequence_length = int(config["sequence_length"])
    seed = builder.sampling_seed(config)

    normalized_sources = builder.normalize_sources_config(sources_config)
    required_sources = set(train_quotas) | set(validation_quotas)
    candidate_root = args.work_root / "candidates"
    expected_roster = expected_candidate_roster(
        normalized_sources, required_sources, candidate_root
    )
    if len(expected_roster) != EXPECTED_CANDIDATE_OBJECTS:
        raise ValueError("sources config does not describe exactly 54 candidate objects")
    roster_sha = semantic_sha256(expected_roster)
    candidate_manifest_path = candidate_root / "manifest.json"
    if not candidate_manifest_path.is_file():
        raise FileNotFoundError(candidate_manifest_path)
    candidate_manifest_payload = candidate_manifest_path.read_bytes()
    candidate_manifest_sha = hashlib.sha256(
        candidate_manifest_payload
    ).hexdigest()
    candidate_manifest = json.loads(candidate_manifest_payload)
    token_path = builder.tokenizer_path(args.tokenizer, config)
    tokenizer = builder.load_tokenizer(token_path)
    tokenizer_fp = builder.tokenizer_fingerprint(token_path)
    candidate_multiplier = float(
        config.get("sampling", {}).get("candidate_multiplier", 1.15)
    )
    if candidate_multiplier <= 1.0:
        raise ValueError("candidate_multiplier must exceed 1.0")
    expected_candidate_fingerprint = builder.stable_sha256(
        builder.BUILDER_VERSION,
        config_sha,
        sources_config_sha,
        builder.canonical_json(tokenizer_fp),
        full_budget,
        validation_budget,
        candidate_multiplier,
        args.mix_key,
        builder.canonical_json(domains),
        builder.canonical_json(scripts["builder"]),
        builder.canonical_json(sharding),
    )
    validate_candidate_manifest(
        candidate_manifest,
        config_sha,
        domains,
        tokenizer_fp,
        sharding,
        train_quotas,
        validation_quotas,
        full_budget,
        validation_budget,
        args.mix_key,
        sources_config_sha,
        expected_roster,
        expected_candidate_fingerprint,
        scripts,
    )
    candidate_paths = validated_candidate_paths(candidate_manifest)
    if set(candidate_paths) != set(train_quotas):
        raise ValueError("candidate source set differs from requested mix")

    loaded_patterns, source_results, load_errors = auditor.load_eval_queries(
        eval_config, args.cache_dir
    )
    patterns = canonicalize_patterns(loaded_patterns)
    contamination = ContaminationFilter(patterns, matching)
    benchmark_snapshot = strict_benchmark_snapshot(
        eval_config,
        patterns,
        source_results,
        load_errors,
        matching,
    )
    benchmark_snapshot.update(
        {
            "eval_config": str(args.eval_config),
            "eval_config_sha256": eval_config_sha,
            "filter_policy": contamination.policy,
        }
    )
    benchmark_semantic_sha = semantic_sha256(benchmark_snapshot)
    base_fingerprint = builder.stable_sha256(
        REPAIR_VERSION,
        candidate_manifest_sha,
        candidate_manifest["fingerprint"],
        config_sha,
        builder.canonical_json(tokenizer_fp),
        sources_config_sha,
        roster_sha,
        eval_config_sha,
        builder.canonical_json(contamination.policy),
        repair_script["sha256"],
        builder.canonical_json(scripts["auditor"]),
        builder.canonical_json(scripts["builder"]),
        builder.canonical_json(sharding),
        benchmark_snapshot["pattern_sha256"],
        benchmark_semantic_sha,
        args.mix_key,
        builder.canonical_json(domains),
        seed,
    )

    final_manifest_path = args.output_root / "manifest.json"
    if final_manifest_path.exists() or args.output_root.exists():
        raise RuntimeError("output root already exists; repair reuse is forbidden")

    stage = args.output_root.with_name(
        f".{args.output_root.name}.repair-{base_fingerprint[:12]}"
    )
    if stage.exists():
        if not args.restart_incomplete:
            raise RuntimeError(f"incomplete repair stage exists: {stage}")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    filter_dir = stage / "contamination_filter"
    filter_dir.mkdir()
    builder.atomic_json(filter_dir / "benchmark_snapshot.json", benchmark_snapshot)
    benchmark_file_sha = builder.sha256_file(
        filter_dir / "benchmark_snapshot.json"
    )
    selected_dir = stage / ".selected"
    selected_dir.mkdir()
    ledger = ExclusionLedger()
    seen = builder.SeenStore(stage / ".seen.sqlite3")
    train_results: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    closed = False
    try:
        for source_id in sorted(train_quotas):
            train_results[source_id] = select_source_filtered(
                builder.merge_sorted_candidate_files(candidate_paths[source_id]),
                train_quotas[source_id],
                "train",
                source_id,
                selected_dir / f"train-{builder.safe_name(source_id)}.jsonl",
                tokenizer,
                sequence_length,
                seen,
                contamination,
                ledger,
            )
        for source_id in sorted(validation_quotas):
            quota = validation_quotas[source_id]
            if not quota:
                continue
            run_dir = stage / f".validation-{builder.safe_name(source_id)}-runs"

            def validation_key(
                record: dict[str, Any], sid: str = source_id
            ) -> str:
                return builder.stable_sha256(
                    domains["split"], seed, sid, record["candidate_id"]
                )

            runs, cleanup = builder.externally_sort_records(
                candidate_paths[source_id],
                validation_key,
                run_dir,
                f"validation-{builder.safe_name(source_id)}",
                args.sort_memory_mb * 1024 * 1024,
            )
            try:
                validation_results[source_id] = select_source_filtered(
                    builder.iter_records_from_runs(runs),
                    quota,
                    "validation",
                    source_id,
                    selected_dir
                    / f"validation-{builder.safe_name(source_id)}.jsonl",
                    tokenizer,
                    sequence_length,
                    seen,
                    contamination,
                    ledger,
                )
            finally:
                cleanup()
                shutil.rmtree(run_dir, ignore_errors=True)

        assert_inputs_unchanged(
            args.config,
            config_sha,
            args.eval_config,
            eval_config_sha,
            args.sources_config,
            sources_config_sha,
            candidate_manifest_path,
            candidate_manifest_sha,
            candidate_manifest,
            candidate_paths,
            token_path,
            tokenizer_fp,
            scripts,
            repair_script,
        )

        train_shards = builder.write_final_shards(
            [Path(result["output"]) for _, result in sorted(train_results.items())],
            "train",
            int(sharding["num_train_shards"]),
            stage,
            seed,
            sequence_length,
            args.sort_memory_mb * 1024 * 1024,
            domains["mix"],
            str(sharding["filename_format"]),
        )
        validation_shards = builder.write_final_shards(
            [
                Path(result["output"])
                for _, result in sorted(validation_results.items())
            ],
            "validation",
            int(sharding["num_validation_shards"]),
            stage,
            seed,
            sequence_length,
            args.sort_memory_mb * 1024 * 1024,
            domains["mix"],
            str(sharding["filename_format"]),
        )

        ledger_path = filter_dir / "exclusion_ledger.jsonl"
        ledger_sha = ledger.write(ledger_path)
        exclusion_snapshot = ledger.snapshot()
        exclusion_snapshot.update(
            {
                "ledger_file": "contamination_filter/exclusion_ledger.jsonl",
                "ledger_sha256": ledger_sha,
                "status": "complete",
            }
        )
        builder.atomic_json(
            filter_dir / "exclusion_snapshot.json", exclusion_snapshot
        )
        exclusion_snapshot_sha = builder.sha256_file(
            filter_dir / "exclusion_snapshot.json"
        )
        fingerprint = builder.stable_sha256(
            base_fingerprint, ledger_sha, exclusion_snapshot_sha
        )
        manifest = {
            "builder_version": builder.BUILDER_VERSION,
            "candidate_fingerprint": candidate_manifest["fingerprint"],
            "candidate_manifest": str(candidate_manifest_path),
            "candidate_manifest_sha256": candidate_manifest_sha,
            "config": str(args.config),
            "config_sha256": config_sha,
            "contamination_filter": {
                "benchmark_snapshot": "contamination_filter/benchmark_snapshot.json",
                "benchmark_snapshot_sha256": benchmark_file_sha,
                "eval_config": str(args.eval_config),
                "eval_config_sha256": eval_config_sha,
                "exclusion_ledger": "contamination_filter/exclusion_ledger.jsonl",
                "exclusion_ledger_sha256": ledger_sha,
                "exclusion_snapshot": "contamination_filter/exclusion_snapshot.json",
                "exclusion_snapshot_sha256": exclusion_snapshot_sha,
                "filter_policy": contamination.policy,
                "pattern_sha256": benchmark_snapshot["pattern_sha256"],
                "status": "applied",
            },
            "sources_config": str(args.sources_config),
            "sources_config_sha256": sources_config_sha,
            "created_at": builder.utc_now(),
            "external_audit": {
                "benchmark_contamination": "pending",
                "required_before_release": True,
            },
            "fingerprint": fingerprint,
            "hash_domains": domains,
            "loader_data_path": str(
                args.output_root / str(sharding["train_glob"])
            ),
            "loader_validation_path": str(
                args.output_root / str(sharding["validation_glob"])
            ),
            "mix_key": args.mix_key,
            "repair_mixer": {
                "base_fingerprint": base_fingerprint,
                "candidate_objects_reused": len(candidate_manifest["objects"]),
                "candidate_roster_sha256": roster_sha,
                "script": repair_script,
                "version": REPAIR_VERSION,
            },
            "schema_version": builder.SCHEMA_VERSION,
            "scripts": scripts,
            "sequence_length": sequence_length,
            "sharding": sharding,
            "status": "pending_external_audit",
            "tokenizer": tokenizer_fp,
            "train": {
                "budget_loss_target_tokens": full_budget,
                "selection": train_results,
                **train_shards,
            },
            "validation": {
                "budget_loss_target_tokens": validation_budget,
                "selection": validation_results,
                **validation_shards,
            },
        }
        builder.atomic_json(stage / "manifest.json", manifest)
        seen.close()
        closed = True
        shutil.rmtree(selected_dir)
        (stage / ".seen.sqlite3").unlink(missing_ok=True)
        if (stage / "_SUCCESS").exists():
            raise AssertionError("repair mixer created forbidden _SUCCESS")
        stage.replace(args.output_root)
    except Exception as exc:
        if not closed:
            try:
                seen.close()
            except sqlite3.ProgrammingError:
                pass
        try:
            ledger_sha = ledger.write(filter_dir / "exclusion_ledger.jsonl")
            failure_snapshot = ledger.snapshot()
            failure_snapshot.update(
                {
                    "ledger_sha256": ledger_sha,
                    "status": "incomplete_fail_closed",
                }
            )
            builder.atomic_json(
                filter_dir / "exclusion_snapshot.json", failure_snapshot
            )
            builder.atomic_json(
                stage / "failure.json",
                {
                    "base_fingerprint": base_fingerprint,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "failed_closed",
                },
            )
        except Exception:
            pass
        raise

    print(
        builder.canonical_json(
            {
                "manifest_sha256": builder.sha256_file(
                    args.output_root / "manifest.json"
                ),
                "output": str(args.output_root),
                "status": "repaired_pending_external_audit",
            }
        )
    )
    return 0


class _Encoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class CharTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> _Encoding:
        del add_special_tokens
        return _Encoding([ord(character) for character in text])

    def decode(self, values: Iterable[int], **_: Any) -> str:
        return "".join(chr(int(value)) for value in values)


def make_record(text: str, candidate_id: str, source: str = "synthetic") -> dict[str, Any]:
    token_ids = [ord(character) for character in text]
    return {
        "candidate_id": candidate_id,
        "content_sha256": builder.token_content_sha256(token_ids),
        "loss_target_tokens": len(token_ids) + 1,
        "nonpad_input_tokens": len(token_ids) + 2,
        "object_id": "synthetic-object",
        "padded_compute_tokens": 16,
        "raw_tokens": len(token_ids),
        "sample_key": sha256_text(candidate_id),
        "source": source,
        "source_path": "synthetic.jsonl",
        "text": text,
        "text_tokens": len(token_ids),
    }


def make_patterns(values: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    return canonicalize_patterns(
        [
            {
                "entries": [
                    {"raw": raw, "source_id": source_id, "task": source_id}
                ],
                "normalized": auditor.normalize(raw),
            }
            for raw, source_id in values
        ]
    )


def synthetic_pinned_fixture() -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]
]:
    tasks = {
        "ceval_valid": "ceval-valid",
        "cmmlu": "cmmlu",
        "arc_easy": "arc_easy",
        "piqa": "piqa",
        "openbookqa": "openbookqa",
        "hellaswag": "hellaswag",
        "social_iqa": "social_iqa",
    }
    sources = []
    results = []
    raw_patterns = []
    for source_id, expected in auditor.PINNED_EVAL_EXPECTATIONS.items():
        task = tasks[source_id]
        source = {
            "id": source_id,
            "query_fields": ["question"],
            "repo_id": expected["repo_id"],
            "revision": expected["revision"],
            "split": expected["split"],
            "task": task,
        }
        if int(expected["configs"]) > 1:
            source["all_configs"] = True
        else:
            source["config"] = "synthetic"
        sources.append(source)
        raw = f"query {source_id}"
        rows = int(expected["rows"])
        raw_patterns.append(
            {
                "entries": [
                    {"raw": raw, "source_id": source_id, "task": task}
                    for _ in range(rows)
                ],
                "normalized": auditor.normalize(raw),
            }
        )
        configs = int(expected["configs"])
        results.append(
            {
                "config_names": [f"config-{index}" for index in range(configs)],
                "configs": configs,
                "configs_failed": 0,
                "configs_succeeded": configs,
                "empty_queries": 0,
                "id": source_id,
                "queries": rows,
                "repo_id": expected["repo_id"],
                "revision": expected["revision"],
                "rows": rows,
                "split": expected["split"],
                "task": task,
            }
        )
    config = {
        "matching": {
            "containment_min_normalized_chars": 20,
            "exact_min_normalized_chars": 1,
        },
        "near_duplicate": {"min_normalized_chars": 20},
        "schema_version": 2,
        "sources": sources,
        "task_alignment": {"task_names": list(tasks.values())},
    }
    return config, canonicalize_patterns(raw_patterns), results


def run_selection_test(
    root: Path,
    records: list[dict[str, Any]],
    quota: int,
    contamination: ContaminationFilter,
    name: str,
) -> tuple[dict[str, Any], Path, ExclusionLedger]:
    ledger = ExclusionLedger()
    seen = builder.SeenStore(root / f"{name}.sqlite3")
    output = root / f"{name}.jsonl"
    try:
        result = select_source_filtered(
            iter(records),
            quota,
            "train",
            "synthetic",
            output,
            CharTokenizer(),
            16,
            seen,
            contamination,
            ledger,
        )
    finally:
        seen.close()
    return result, output, ledger


def self_test(_: argparse.Namespace) -> int:
    matching = {
        "configured_explicitly": True,
        "containment_min_normalized_chars": 20,
        "exact_min_normalized_chars": 1,
    }
    tests: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pretrain-v1-remix-selftest-") as raw:
        root = Path(raw)
        patterns = make_patterns(
            [
                ("pan", "short"),
                ("a" * 19, "len19"),
                ("b" * 20, "len20"),
            ]
        )
        contamination = ContaminationFilter(patterns, matching)
        assert contamination.classify("prefix pan suffix") is None
        assert contamination.classify("pan")["reason"] == "exact_eval_overlap"
        assert contamination.classify("x" + "a" * 19 + "y") is None
        assert (
            contamination.classify("x" + "b" * 20 + "y")["reason"]
            == "containment_eval_overlap"
        )
        assert contamination.classify(" \n\t\u00a0")["reason"] == "empty_text"
        assert contamination.classify("++--") is None
        tests.append("matching_exact_containment_empty_and_symbol_semantics")

        prefix_filter = ContaminationFilter(make_patterns([("abc", "prefix")]), matching)
        records = [make_record("abcZ", "prefix-bad"), make_record("wxy", "good")]
        result_a, output_a, ledger_a = run_selection_test(
            root, records, 4, prefix_filter, "prefix-a"
        )
        assert json.loads(output_a.read_text().strip())["text"] == "wxy"
        assert result_a["loss_target_tokens"] == 4
        assert any(event["phase"] == "exact_prefix" for event in ledger_a.sorted_events())
        tests.append("prefix_piece_rechecked_and_selection_continues")

        result_b, output_b, ledger_b = run_selection_test(
            root, records, 4, prefix_filter, "prefix-b"
        )
        ledger_a_path = root / "ledger-a.jsonl"
        ledger_b_path = root / "ledger-b.jsonl"
        assert ledger_a.write(ledger_a_path) == ledger_b.write(ledger_b_path)
        assert builder.sha256_file(output_a) == builder.sha256_file(output_b)
        assert result_a["output_sha256"] == result_b["output_sha256"]
        tests.append("selection_and_ledger_determinism")

        replacement_result, replacement_output, _ = run_selection_test(
            root,
            [make_record("ab", "replace-old"), make_record("wxyz", "replace-new")],
            4,
            ContaminationFilter(make_patterns([("a", "boundary")]), matching),
            "boundary-replacement",
        )
        assert json.loads(replacement_output.read_text().strip())["text"] == "wxy"
        assert replacement_result["boundary_replacement_selected"] == 1
        assert replacement_result["loss_target_tokens"] == 4
        tests.append("filtered_boundary_split_uses_later_candidate_replacement")

        boundary_filter = ContaminationFilter(make_patterns([("a", "boundary")]), matching)
        boundary_ledger = ExclusionLedger()
        boundary_seen = builder.SeenStore(root / "boundary.sqlite3")
        boundary_output = root / "boundary.jsonl"
        try:
            try:
                select_source_filtered(
                    iter([make_record("ab", "pending"), make_record("zz", "trigger")]),
                    4,
                    "train",
                    "synthetic",
                    boundary_output,
                    CharTokenizer(),
                    16,
                    boundary_seen,
                    boundary_filter,
                    boundary_ledger,
                )
                raise AssertionError("contaminated boundary split unexpectedly passed")
            except RuntimeError as exc:
                assert "quota short" in str(exc)
        finally:
            boundary_seen.close()
        assert not boundary_output.exists()
        assert any(
            event["phase"].startswith("boundary_piece_")
            for event in boundary_ledger.sorted_events()
        )
        tests.append("boundary_pieces_rechecked_and_shortage_fails_closed")

        shortage_filter = ContaminationFilter(make_patterns([("bad", "shortage")]), matching)
        shortage_ledger = ExclusionLedger()
        shortage_seen = builder.SeenStore(root / "shortage.sqlite3")
        shortage_output = root / "shortage.jsonl"
        try:
            try:
                select_source_filtered(
                    iter([make_record("bad", "bad"), make_record("   ", "empty")]),
                    3,
                    "train",
                    "synthetic",
                    shortage_output,
                    CharTokenizer(),
                    16,
                    shortage_seen,
                    shortage_filter,
                    shortage_ledger,
                )
                raise AssertionError("candidate shortage unexpectedly passed")
            except RuntimeError as exc:
                assert "quota short" in str(exc)
        finally:
            shortage_seen.close()
        assert not shortage_output.exists()
        tests.append("filtered_candidate_shortage_fails_closed")

        eval_config, pinned_patterns, source_results = synthetic_pinned_fixture()
        pinned_matching = auditor.matching_contract(eval_config)
        snapshot = strict_benchmark_snapshot(
            eval_config, pinned_patterns, source_results, [], pinned_matching
        )
        assert snapshot["sources_count"] == EXPECTED_PINNED_SOURCES
        assert snapshot["configs"] == EXPECTED_PINNED_CONFIGS
        assert snapshot["rows"] == EXPECTED_PINNED_ROWS
        reordered = canonicalize_patterns(
            [
                {
                    "entries": list(reversed(pattern["entries"])),
                    "normalized": pattern["normalized"],
                }
                for pattern in reversed(pinned_patterns)
            ]
        )
        assert semantic_sha256(pattern_payload(pinned_patterns)) == semantic_sha256(
            pattern_payload(reordered)
        )
        tests.append("pinned_7_124_29638_and_pattern_order_determinism")

        bad_results = json.loads(json.dumps(source_results))[:-1]
        for label, results, errors in (
            ("missing_source", bad_results, []),
            (
                "wrong_rows",
                [
                    {**item, "rows": item["rows"] - 1, "queries": item["queries"] - 1}
                    if index == 0
                    else item
                    for index, item in enumerate(json.loads(json.dumps(source_results)))
                ],
                [],
            ),
            ("load_error", source_results, [{"source": "ceval_valid", "error": "x"}]),
        ):
            try:
                strict_benchmark_snapshot(
                    eval_config, pinned_patterns, results, errors, pinned_matching
                )
                raise AssertionError(f"{label} unexpectedly passed")
            except RuntimeError:
                pass
        tests.append("pinned_source_row_and_load_errors_fail_closed")
        assert not list(root.rglob("_SUCCESS"))
        tests.append("no_success_marker")

    print(builder.canonical_json({"status": "self_test_ok", "tests": tests}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    remix_parser = subparsers.add_parser("remix")
    remix_parser.add_argument("--config", type=Path, required=True)
    remix_parser.add_argument("--eval-config", type=Path, required=True)
    remix_parser.add_argument("--sources-config", type=Path, required=True)
    remix_parser.add_argument("--work-root", type=Path, required=True)
    remix_parser.add_argument("--output-root", type=Path, required=True)
    remix_parser.add_argument("--tokenizer", type=Path)
    remix_parser.add_argument("--cache-dir", type=Path, default=Path("/data/cache/huggingface"))
    remix_parser.add_argument("--sort-memory-mb", type=int, default=256)
    remix_parser.add_argument("--mix-key", default="active_v1_mix")
    remix_parser.add_argument("--restart-incomplete", action="store_true")
    remix_parser.set_defaults(function=remix)
    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.set_defaults(function=self_test)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
