#!/usr/bin/env python3
"""Run deterministic, no-chat-template continuation probes for a Base model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    {"id": "zh_capital", "prompt": "中国的首都是"},
    {"id": "en_capital", "prompt": "The capital of France is"},
    {"id": "arithmetic", "prompt": "1 + 1 ="},
    {"id": "science", "prompt": "水在标准大气压下的沸点是"},
    {"id": "code", "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True
    ).to(args.device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in PROMPTS:
            encoded = tokenizer(item["prompt"], return_tensors="pt").to(args.device)
            encoded.pop("token_type_ids", None)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            continuation_ids = generated[0, encoded.input_ids.shape[1]:]
            record = {
                **item,
                "continuation": tokenizer.decode(continuation_ids, skip_special_tokens=True),
                "generated_tokens": int(continuation_ids.numel()),
                "decoding": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "chat_template": False,
                "seed": 42,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
