#!/usr/bin/env python3
"""Download and verify the pinned MiniMind Stage1 dataset set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "jingyaogong/minimind_dataset"
REVISION = "312afb4f76391145c6902f765bb51691c09a12f5"
FILES = {
    "pretrain_t2t_mini.jsonl": {
        "size": 1241043656,
        "sha256": "6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c",
        "role": "mini_pretrain",
    },
    "sft_t2t_mini.jsonl": {
        "size": 1739201170,
        "sha256": "abb1e76b2056e14728beb78db96b7b3c491a0bef1ed3e34a9b381b28f29fa518",
        "role": "mini_sft",
    },
    "dpo.jsonl": {
        "size": 53653322,
        "sha256": "ee934a8a455ccc99d1334d63e1254dd1d64f497fd067cfcbb71e3043f5b46768",
        "role": "preference",
    },
    "rlaif.jsonl": {
        "size": 23754740,
        "sha256": "8c6634db971fa34b0217f7db4f7c30684f57d20bbb771eb717f1b5aeacb089ba",
        "role": "rlaif",
    },
    "agent_rl.jsonl": {
        "size": 82036930,
        "sha256": "cb96bcc8096aecc5eccaab858f75d5ace1dc22da2302c2230457e29744a761ab",
        "role": "agentic_rl",
    },
    "agent_rl_math.jsonl": {
        "size": 18372683,
        "sha256": "5f2f6f8e90d8d5c698561b7540ec8b4bc8cb0a568e329cd4e7fc4d378ccd60cb",
        "role": "agentic_rl_math",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("/data/datasets/minimind") / REVISION,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--files",
        nargs="+",
        choices=sorted(FILES),
        default=sorted(FILES),
    )
    args = parser.parse_args()

    requested = list(dict.fromkeys(args.files))
    args.target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=requested,
        local_dir=args.target_dir,
    )

    records = []
    for filename in requested:
        expected = FILES[filename]
        path = args.target_dir / filename
        if not path.is_file():
            raise SystemExit(f"download missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected["size"]:
            raise SystemExit(
                f"size mismatch for {filename}: {actual_size} != {expected['size']}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected["sha256"]:
            raise SystemExit(
                f"sha256 mismatch for {filename}: {actual_sha256} != {expected['sha256']}"
            )
        records.append(
            {
                "name": filename,
                "role": expected["role"],
                "path": str(path),
                "size_bytes": actual_size,
                "sha256": actual_sha256,
                "line_count": None,
                "invalid_json_lines": None,
            }
        )
        print(f"verified={filename} size={actual_size} sha256={actual_sha256}")

    manifest = {
        "schema_version": 1,
        "source": "huggingface",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
