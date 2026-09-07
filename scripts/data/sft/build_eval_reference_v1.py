#!/usr/bin/env python3
"""Materialize the pinned fixed-evaluation prompts used by contamination audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class ReferenceError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/sft/contamination_v1.yaml"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/data/cache/huggingface/datasets"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_configs(cache_dir: Path, repo_id: str) -> list[str]:
    directory = cache_dir / repo_id.replace("/", "___")
    if not directory.is_dir():
        raise ReferenceError(f"all-config cache is missing: {directory}")
    values = sorted(
        item.name
        for item in directory.iterdir()
        if item.is_dir() and not item.name.startswith(".")
    )
    if not values:
        raise ReferenceError(f"no cached configs: {directory}")
    return values


def row_prompt(row: dict[str, Any], fields: list[str]) -> str:
    values = []
    for field in fields:
        value = row.get(field)
        if value is None:
            raise ReferenceError(f"query field is missing: {field}")
        text = str(value).strip()
        if not text:
            raise ReferenceError(f"query field is empty: {field}")
        values.append(text)
    prompt = normalize("\n".join(values))
    if not prompt:
        raise ReferenceError("normalized query is empty")
    return prompt


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    os.environ.setdefault("HF_ENDPOINT", config.get("metadata_endpoint", "https://hf-mirror.com"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from datasets import load_dataset

    if args.output.exists() or args.manifest.exists():
        raise ReferenceError("output or manifest already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counts = Counter()
    failures = Counter()
    seen: set[str] = set()
    for source in config["sources"]:
        source_id = str(source["id"])
        configs = (
            cached_configs(args.cache_dir, str(source["repo_id"]))
            if source.get("all_configs")
            else [source.get("config")]
        )
        for dataset_config in configs:
            kwargs: dict[str, Any] = {
                "path": source["repo_id"],
                "split": source["split"],
                "revision": source["revision"],
                "cache_dir": str(args.cache_dir),
                "trust_remote_code": True,
            }
            if dataset_config:
                kwargs["name"] = dataset_config
            try:
                dataset = load_dataset(**kwargs)
            except Exception as exc:
                failures[f"{source_id}:{dataset_config}:{type(exc).__name__}"] += 1
                continue
            for line_number, row in enumerate(dataset, 1):
                try:
                    prompt = row_prompt(dict(row), list(source["query_fields"]))
                except ReferenceError as exc:
                    failures[f"{source_id}:row:{str(exc)}"] += 1
                    continue
                digest = sha256_bytes(prompt.encode())
                if digest in seen:
                    counts[f"{source_id}:exact_duplicates"] += 1
                    continue
                seen.add(digest)
                rows.append(
                    {
                        "task": source["task"],
                        "source_id": source_id,
                        "repo_id": source["repo_id"],
                        "revision": source["revision"],
                        "config": dataset_config,
                        "split": source["split"],
                        "source_line_number": line_number,
                        "normalized_prompt": prompt,
                        "prompt_sha256": digest,
                    }
                )
                counts[f"{source_id}:rows"] += 1
    if failures:
        raise ReferenceError(f"fixed-eval prompt materialization failures: {dict(failures)}")
    rows.sort(
        key=lambda value: (
            value["task"],
            value["source_id"],
            str(value["config"]),
            value["prompt_sha256"],
        )
    )
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "config": str(args.config),
        "config_sha256": file_sha256(config_path),
        "prompt_builder": "scripts/data/sft/build_eval_reference_v1.py",
        "prompt_builder_sha256": file_sha256(Path(__file__)),
        "cache_dir": str(args.cache_dir),
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": file_sha256(args.output),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "rows": len(rows),
                "sha256": manifest["output_sha256"],
                "prompt_builder_sha256": manifest["prompt_builder_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReferenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
