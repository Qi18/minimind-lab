#!/usr/bin/env python3
"""Resolve pinned Hugging Face dataset metadata and fail on revision drift."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def normalize_license(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    return sorted(str(item).lower() for item in value)


def fetch_metadata(endpoint: str, repo_id: str, timeout: float) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/api/datasets/{repo_id}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "minimind-lab-data-v1/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    resolved: list[dict[str, Any]] = []
    failed = False

    for source in config["sources"]:
        row = {
            "id": source["id"],
            "repo_id": source["repo_id"],
            "declared_revision": source["revision"],
            "declared_license": source.get("license"),
            "enabled_for_proxy": bool(source.get("enabled_for_proxy")),
        }
        try:
            payload = fetch_metadata(args.endpoint, source["repo_id"], args.timeout)
            card = payload.get("cardData") or {}
            tags = payload.get("tags") or []
            live_license = card.get("license")
            if live_license is None:
                live_license = [tag[8:] for tag in tags if tag.startswith("license:")]
            row.update(
                {
                    "status": "ok",
                    "resolved_revision": payload.get("sha"),
                    "resolved_license": live_license,
                    "gated": payload.get("gated"),
                    "private": payload.get("private"),
                    "disabled": payload.get("disabled"),
                    "revision_match": payload.get("sha") == source["revision"],
                    "license_match": normalize_license(live_license)
                    == normalize_license(source.get("license")),
                }
            )
            if row["enabled_for_proxy"] and (
                not row["revision_match"]
                or not row["license_match"]
                or row["gated"] not in (False, None)
                or row["private"]
                or row["disabled"]
            ):
                failed = True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            if row["enabled_for_proxy"]:
                failed = True
        resolved.append(row)

    result = {
        "schema_version": 1,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "source_config": str(args.config),
        "all_enabled_sources_ready": not failed,
        "sources": resolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ready": not failed, "sources": len(resolved)}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
