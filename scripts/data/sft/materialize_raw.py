#!/usr/bin/env python3
"""Resumable, source-level raw materialization for MiniMind Lab SFT v1.

Raw outputs intentionally retain upstream fields and are never valid training JSONL.
The later formal builder must enforce the canonical conversations-only schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCHEMA_STAGE = "raw_not_trainable"
TRAINABLE = False
PROTOCOL_VERSION = "2026-09-01-sft-raw-v1"


class MaterializationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resolved-sources", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/data/cache/huggingface"))
    parser.add_argument("--source-id", action="append")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return str(value)


def normalize_license(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()]
    return sorted(str(item).strip().lower() for item in value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if os.path.lexists(temp):
        raise MaterializationError(f"temporary path already exists: {temp}")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def scan_jsonl(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    fields: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if not raw.strip():
                raise MaterializationError(f"blank JSONL line: {path}:{line_number}")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MaterializationError(
                    f"invalid JSONL at {path}:{line_number}: {type(exc).__name__}"
                ) from exc
            if not isinstance(row, dict):
                raise MaterializationError(f"non-object JSONL row: {path}:{line_number}")
            fields.update(str(key) for key in row)
            rows += 1
    if rows == 0:
        raise MaterializationError(f"empty JSONL object: {path}")
    return {
        "rows": rows,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "fields": sorted(fields),
    }


def validate_required_fields(
    source_id: str,
    actual_fields: list[str],
    required_fields: list[str],
) -> None:
    missing = sorted(set(required_fields) - set(actual_fields))
    if missing:
        raise MaterializationError(
            f"{source_id} is missing required top-level fields: {missing}"
        )


def validate_config(config: dict[str, Any], allow_blockers: bool = False) -> list[dict[str, Any]]:
    if config.get("schema_version") != 1:
        raise MaterializationError("config schema_version must be 1")
    if config.get("dataset_stage") != SCHEMA_STAGE:
        raise MaterializationError(f"dataset_stage must be {SCHEMA_STAGE}")
    if int(config.get("target_assistant_tokens", 0)) <= 0:
        raise MaterializationError("target_assistant_tokens must be positive")

    policies = config.get("policies") or {}
    expected_top = ["conversations"]
    expected_message = [
        "role",
        "content",
        "reasoning_content",
        "tools",
        "tool_calls",
    ]
    if policies.get("raw_is_trainable") is not False:
        raise MaterializationError("raw_is_trainable must be false")
    if policies.get("final_training_top_level_keys") != expected_top:
        raise MaterializationError(
            "formal training top-level schema must be conversations-only"
        )
    if policies.get("final_message_keys") != expected_message:
        raise MaterializationError("formal message schema must contain the frozen five keys")
    if policies.get("provenance_storage") != "sidecar_only":
        raise MaterializationError("formal provenance must be sidecar_only")

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise MaterializationError("config sources must be a non-empty list")
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    for source in sources:
        source_id = str(source.get("id", ""))
        if not source_id or source_id in seen_ids:
            raise MaterializationError(f"invalid or duplicate source id: {source_id!r}")
        seen_ids.add(source_id)
        repo_id = str(source.get("repo_id", ""))
        revision = str(source.get("revision", ""))
        if "/" not in repo_id:
            raise MaterializationError(f"{source_id}: repo_id must be pinned")
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise MaterializationError(f"{source_id}: revision must be a 40-char lowercase SHA")
        if not normalize_license(source.get("license")):
            raise MaterializationError(f"{source_id}: declared license is required")
        materialization = source.get("materialization") or {}
        mode = materialization.get("mode")
        if mode not in {"hf_stream", "reuse_jsonl"}:
            raise MaterializationError(f"{source_id}: unsupported materialization mode {mode!r}")
        output = str(materialization.get("output", ""))
        output_path = Path(output)
        if (
            not output
            or output_path.is_absolute()
            or ".." in output_path.parts
            or output_path.suffix != ".jsonl"
            or output in seen_outputs
        ):
            raise MaterializationError(f"{source_id}: invalid or duplicate relative output")
        seen_outputs.add(output)
        required_fields = materialization.get("required_fields")
        if not isinstance(required_fields, list) or not required_fields:
            raise MaterializationError(f"{source_id}: required_fields must be non-empty")
        if mode == "reuse_jsonl":
            for key in (
                "reuse_path",
                "expected_rows",
                "expected_bytes",
                "expected_sha256",
            ):
                if not materialization.get(key):
                    raise MaterializationError(f"{source_id}: reuse mode requires {key}")

    raw_blockers = [
        item
        for item in (config.get("raw_materialization_blockers") or [])
        if item.get("severity") == "fatal"
    ]
    if raw_blockers and not allow_blockers:
        rendered = ", ".join(str(item.get("id")) for item in raw_blockers)
        raise MaterializationError(f"raw materialization blocked: {rendered}")
    return raw_blockers


def validate_resolved_sources(
    config: dict[str, Any],
    resolved_path: Path,
) -> dict[str, dict[str, Any]]:
    if not resolved_path.is_file():
        raise MaterializationError(f"resolved source report is missing: {resolved_path}")
    report = json.loads(resolved_path.read_text(encoding="utf-8"))
    if report.get("all_enabled_sources_ready") is not True:
        raise MaterializationError("resolved source report is not ready")
    resolved_by_id = {str(item.get("id")): item for item in report.get("sources", [])}
    for source in config["sources"]:
        source_id = source["id"]
        item = resolved_by_id.get(source_id)
        if item is None:
            raise MaterializationError(f"{source_id}: missing from resolved source report")
        if item.get("status") != "ok":
            raise MaterializationError(f"{source_id}: metadata resolution did not pass")
        if item.get("resolved_revision") != source["revision"]:
            raise MaterializationError(f"{source_id}: resolved revision drift")
        if normalize_license(item.get("resolved_license")) != normalize_license(
            source.get("license")
        ):
            raise MaterializationError(f"{source_id}: resolved license drift")
        if item.get("revision_match") is not True or item.get("license_match") is not True:
            raise MaterializationError(f"{source_id}: resolver match gates failed")
        if item.get("gated") not in (False, None):
            raise MaterializationError(f"{source_id}: gated source is not permitted")
        if item.get("private") or item.get("disabled"):
            raise MaterializationError(f"{source_id}: private or disabled source")
    return resolved_by_id


def source_fingerprint(materializer_sha256: str, source: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "protocol_version": PROTOCOL_VERSION,
            "schema_stage": SCHEMA_STAGE,
            "trainable": TRAINABLE,
            "materializer_sha256": materializer_sha256,
            "source": source,
        }
    )


def output_and_done(
    output_root: Path,
    source: dict[str, Any],
) -> tuple[Path, Path]:
    output = output_root / source["materialization"]["output"]
    done = output.with_name(f"{output.name}.done.json")
    return output, done


def completed_evidence(
    output: Path,
    done_path: Path,
    source: dict[str, Any],
    fingerprint: str,
) -> dict[str, Any] | None:
    output_exists = os.path.lexists(output)
    done_exists = done_path.is_file()
    if not output_exists and not done_exists:
        return None
    if output_exists != done_exists:
        raise MaterializationError(
            f"{source['id']}: output/done must either both exist or both be absent"
        )

    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("source_fingerprint") != fingerprint:
        raise MaterializationError(f"{source['id']}: completed object fingerprint drift")
    if done.get("schema_stage") != SCHEMA_STAGE or done.get("trainable") is not False:
        raise MaterializationError(f"{source['id']}: completed object schema boundary drift")
    actual = scan_jsonl(output)
    for key in ("rows", "bytes", "sha256", "fields"):
        if done.get(key) != actual[key]:
            raise MaterializationError(f"{source['id']}: completed object {key} mismatch")
    required_fields = source["materialization"]["required_fields"]
    validate_required_fields(source["id"], actual["fields"], required_fields)
    if source["materialization"]["mode"] == "reuse_jsonl":
        reuse = Path(source["materialization"]["reuse_path"]).resolve()
        if not output.is_symlink() or output.resolve() != reuse:
            raise MaterializationError(f"{source['id']}: reused output symlink drift")
        expected = source["materialization"]
        if (
            actual["rows"] != int(expected["expected_rows"])
            or actual["bytes"] != int(expected["expected_bytes"])
            or actual["sha256"] != expected["expected_sha256"]
        ):
            raise MaterializationError(f"{source['id']}: reused source evidence drift")

    return {
        **done,
        "status": "resumed_verified",
        "done_path": str(done_path),
    }


def prepare_temp(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f"{output.name}.tmp")
    if os.path.lexists(temp):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        abandoned = output.with_name(f"{output.name}.tmp.abandoned.{stamp}.{os.getpid()}")
        os.replace(temp, abandoned)
    return temp


def materialize_reuse(source: dict[str, Any], output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    materialization = source["materialization"]
    reuse = Path(materialization["reuse_path"])
    if not reuse.is_file():
        raise MaterializationError(f"{source['id']}: reuse source is missing: {reuse}")
    evidence = scan_jsonl(reuse)
    if (
        evidence["rows"] != int(materialization["expected_rows"])
        or evidence["bytes"] != int(materialization["expected_bytes"])
        or evidence["sha256"] != materialization["expected_sha256"]
    ):
        raise MaterializationError(f"{source['id']}: reuse source rows/bytes/SHA mismatch")
    validate_required_fields(
        source["id"],
        evidence["fields"],
        materialization["required_fields"],
    )
    temp = prepare_temp(output)
    os.symlink(str(reuse.resolve()), temp)
    os.replace(temp, output)
    return evidence, {
        "storage": "symlink_reuse",
        "reuse_source": str(reuse.resolve()),
    }


def materialize_hf(
    source: dict[str, Any],
    output: Path,
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from datasets import load_dataset

    materialization = source["materialization"]
    kwargs: dict[str, Any] = {
        "path": source["repo_id"],
        "split": materialization["split"],
        "revision": source["revision"],
        "streaming": True,
        "cache_dir": str(cache_dir),
    }
    if materialization.get("config"):
        kwargs["name"] = materialization["config"]

    temp = prepare_temp(output)
    rows = 0
    fields: set[str] = set()
    digest = hashlib.sha256()
    try:
        dataset = load_dataset(**kwargs)
        with temp.open("x", encoding="utf-8") as handle:
            for raw in dataset:
                row = jsonable(raw)
                if not isinstance(row, dict):
                    raise MaterializationError(f"{source['id']}: upstream row is not an object")
                missing = sorted(
                    set(materialization["required_fields"]) - set(row)
                )
                if missing:
                    raise MaterializationError(
                        f"{source['id']}: row {rows + 1} missing fields {missing}"
                    )
                fields.update(str(key) for key in row)
                rendered = (
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.write(rendered)
                digest.update(rendered.encode("utf-8"))
                rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        if rows == 0:
            raise MaterializationError(f"{source['id']}: upstream split produced zero rows")
        os.replace(temp, output)
    except Exception:
        # Keep a partial .tmp as failure evidence; the next attempt archives it.
        raise

    evidence = {
        "rows": rows,
        "bytes": output.stat().st_size,
        "sha256": digest.hexdigest(),
        "fields": sorted(fields),
    }
    return evidence, {
        "storage": "materialized_jsonl",
        "load_dataset": {
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "config": materialization.get("config"),
            "split": materialization["split"],
            "streaming": True,
        },
    }


def execute(
    config_path: Path,
    resolved_path: Path,
    output_root: Path,
    manifest_path: Path,
    cache_dir: Path,
    selected_ids: set[str] | None = None,
    allow_blockers: bool = False,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_blockers = validate_config(config, allow_blockers=allow_blockers)
    validate_resolved_sources(config, resolved_path)
    config_sha = sha256_file(config_path)
    materializer_sha = sha256_file(Path(__file__).resolve())
    resolved_sha = sha256_file(resolved_path)

    sources = config["sources"]
    if selected_ids:
        known = {source["id"] for source in sources}
        unknown = sorted(selected_ids - known)
        if unknown:
            raise MaterializationError(f"unknown source ids: {unknown}")
        sources = [source for source in sources if source["id"] in selected_ids]

    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for source in sources:
        output, done_path = output_and_done(output_root, source)
        fingerprint = source_fingerprint(materializer_sha, source)
        resumed = completed_evidence(output, done_path, source, fingerprint)
        if resumed is not None:
            results.append(resumed)
            print(
                json.dumps(
                    {"source": source["id"], "status": "resumed_verified"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        started = time.monotonic()
        mode = source["materialization"]["mode"]
        if mode == "reuse_jsonl":
            evidence, storage = materialize_reuse(source, output)
        else:
            evidence, storage = materialize_hf(source, output, cache_dir)
        done = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "created_at": utc_now(),
            "source_id": source["id"],
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "license": source["license"],
            "purpose": source.get("purpose"),
            "source_fingerprint": fingerprint,
            "config_sha256": config_sha,
            "materializer_sha256": materializer_sha,
            "schema_stage": SCHEMA_STAGE,
            "trainable": TRAINABLE,
            "not_trainable_reason": (
                "Raw upstream fields are preserved. Formal training JSONL must be "
                "conversations-only with exactly five normalized message keys."
            ),
            "output": str(output),
            **evidence,
            **storage,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        atomic_write_json(done_path, done)
        result = {**done, "status": "materialized", "done_path": str(done_path)}
        results.append(result)
        print(
            json.dumps(
                {
                    "source": source["id"],
                    "status": "materialized",
                    "rows": evidence["rows"],
                    "bytes": evidence["bytes"],
                    "sha256": evidence["sha256"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "created_at": utc_now(),
        "status": "raw_materialized_not_trainable",
        "schema_stage": SCHEMA_STAGE,
        "trainable": TRAINABLE,
        "config": str(config_path),
        "config_sha256": config_sha,
        "materializer_sha256": materializer_sha,
        "resolved_sources": str(resolved_path),
        "resolved_sources_sha256": resolved_sha,
        "output_root": str(output_root),
        "selected_source_ids": [source["id"] for source in sources],
        "sources": results,
        "derived_stages": config.get("derived_stages", []),
        "raw_materialization_blockers_bypassed_for_self_test": raw_blockers,
        "final_build_blockers": config.get("final_build_blockers", []),
        "final_build_allowed": not any(
            item.get("severity") == "fatal"
            for item in config.get("final_build_blockers", [])
        ),
        "training_schema_contract": {
            "top_level_keys": ["conversations"],
            "message_keys": [
                "role",
                "content",
                "reasoning_content",
                "tools",
                "tool_calls",
            ],
            "optional_values": "explicit null",
            "provenance": "sidecar only",
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "status": manifest["status"],
                "sources": len(results),
                "trainable": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return manifest


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="sft-v1-raw-self-test-") as temp_name:
        root = Path(temp_name)
        reuse = root / "fixture.jsonl"
        reuse.write_text(
            json.dumps({"id": 1, "payload": "alpha"}, sort_keys=True) + "\n"
            + json.dumps({"id": 2, "payload": "beta"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence = scan_jsonl(reuse)
        config = {
            "schema_version": 1,
            "name": "self-test",
            "dataset_stage": SCHEMA_STAGE,
            "target_assistant_tokens": 1,
            "policies": {
                "raw_is_trainable": False,
                "final_training_top_level_keys": ["conversations"],
                "final_message_keys": [
                    "role",
                    "content",
                    "reasoning_content",
                    "tools",
                    "tool_calls",
                ],
                "provenance_storage": "sidecar_only",
            },
            "raw_materialization_blockers": [],
            "final_build_blockers": [],
            "sources": [
                {
                    "id": "fixture",
                    "repo_id": "fixture/repo",
                    "revision": "a" * 40,
                    "license": "mit",
                    "purpose": "self_test",
                    "materialization": {
                        "mode": "reuse_jsonl",
                        "split": "train",
                        "output": "raw/fixture.jsonl",
                        "required_fields": ["id", "payload"],
                        "reuse_path": str(reuse),
                        "expected_rows": evidence["rows"],
                        "expected_bytes": evidence["bytes"],
                        "expected_sha256": evidence["sha256"],
                    },
                }
            ],
            "derived_stages": [],
        }
        config_path = root / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        resolved = {
            "schema_version": 1,
            "all_enabled_sources_ready": True,
            "sources": [
                {
                    "id": "fixture",
                    "status": "ok",
                    "resolved_revision": "a" * 40,
                    "resolved_license": "mit",
                    "revision_match": True,
                    "license_match": True,
                    "gated": False,
                    "private": False,
                    "disabled": False,
                }
            ],
        }
        resolved_path = root / "resolved.json"
        resolved_path.write_text(
            json.dumps(resolved, indent=2) + "\n",
            encoding="utf-8",
        )
        output_root = root / "output"
        manifest_path = root / "manifest.json"
        first = execute(
            config_path,
            resolved_path,
            output_root,
            manifest_path,
            root / "cache",
        )
        second = execute(
            config_path,
            resolved_path,
            output_root,
            manifest_path,
            root / "cache",
        )
        if first["sources"][0]["status"] != "materialized":
            raise MaterializationError("self-test first run did not materialize")
        if second["sources"][0]["status"] != "resumed_verified":
            raise MaterializationError("self-test second run did not verify resume")
        done_path = output_root / "raw/fixture.jsonl.done.json"
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("trainable") is not False or done.get("schema_stage") != SCHEMA_STAGE:
            raise MaterializationError("self-test raw schema boundary missing")

        blocked = dict(config)
        blocked["raw_materialization_blockers"] = [
            {"id": "fixture_blocker", "severity": "fatal"}
        ]
        try:
            validate_config(blocked)
        except MaterializationError:
            pass
        else:
            raise MaterializationError("self-test fatal blocker did not fail closed")

    print(json.dumps({"self_test": "passed", "network_used": False}))
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            return self_test()
        for name, value in (
            ("--config", args.config),
            ("--resolved-sources", args.resolved_sources),
            ("--output-root", args.output_root),
            ("--manifest", args.manifest),
        ):
            if value is None:
                raise MaterializationError(f"{name} is required")

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        raw_blockers = validate_config(config)
        if args.validate_only:
            validate_resolved_sources(config, args.resolved_sources)
            print(
                json.dumps(
                    {
                        "validation": "passed",
                        "raw_materialization_blockers": raw_blockers,
                        "final_build_blockers": config.get("final_build_blockers", []),
                        "final_build_allowed": not any(
                            item.get("severity") == "fatal"
                            for item in config.get("final_build_blockers", [])
                        ),
                        "network_used": False,
                    }
                )
            )
            return 0

        execute(
            config_path=args.config,
            resolved_path=args.resolved_sources,
            output_root=args.output_root,
            manifest_path=args.manifest,
            cache_dir=args.cache_dir,
            selected_ids=set(args.source_id or []) or None,
        )
        return 0
    except (MaterializationError, OSError, ValueError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked_or_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
