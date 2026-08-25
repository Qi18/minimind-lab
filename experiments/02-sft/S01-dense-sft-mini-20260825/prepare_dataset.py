#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import random
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate the SFT dataset cache once before DDP launch.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root_dir / "minimind"))

    import torch
    from transformers import AutoTokenizer
    from dataset.lm_dataset import SFTDataset

    random.seed(42)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if len(dataset) != args.expected_rows:
        raise SystemExit(f"dataset_row_count_mismatch:{len(dataset)}")

    sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
    valid_counts = []
    for index in sample_indices:
        input_ids, labels = dataset[index]
        if input_ids.shape != labels.shape:
            raise SystemExit(f"shape_mismatch:index={index}")
        valid_count = int((labels != -100).sum().item())
        if valid_count <= 0:
            raise SystemExit(f"no_valid_labels:index={index}")
        valid_counts.append(valid_count)

    print(f"dataset_cache=ready rows={len(dataset)} max_seq_len={args.max_seq_len}")
    print("sample_valid_label_tokens=" + ",".join(str(value) for value in valid_counts))
    print(f"hf_home={os.environ.get('HF_HOME', '')}")


if __name__ == "__main__":
    main()
