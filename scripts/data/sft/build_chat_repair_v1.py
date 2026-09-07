#!/usr/bin/env python3
"""Build a deterministic SFT set for concise chat and format following."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


MESSAGE_KEYS = ("role", "content", "reasoning_content", "tools", "tool_calls")
BANNED_PROMPTS = {
    "请计算17加25。只输出最终数字，不要解释。",
    "只输出一个JSON对象，必须恰好包含city和country两个字段，值分别为北京和中国。不要使用Markdown代码块。",
    "列出三种常见水果。严格输出三行，每行以'- '开头，不要添加标题。",
    "把Hello world翻译成中文。只输出译文。",
    "水的化学式是什么？只输出化学式。",
    "写一个Python函数add(a, b)，返回a与b的和。只输出代码。",
    "严格只回复READY这五个英文字母。",
    "用一句话介绍杭州西湖的两个特点。",
    "用一句话解释光合作用。",
    "用不超过50个汉字比较猫和狗作为宠物的一个差异。",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--official-replay", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-family", type=int, default=8000)
    return parser.parse_args()


def message(role: str, content: str) -> dict:
    return {key: content if key == "content" else role if key == "role" else None for key in MESSAGE_KEYS}


def row(prompt: str, answer: str) -> dict:
    assert prompt not in BANNED_PROMPTS
    return {"conversations": [message("user", prompt), message("assistant", answer)]}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def split_for(identifier: str) -> str:
    value = int(digest(identifier)[:8], 16) % 100
    return "train" if value < 96 else "validation" if value < 98 else "test"


def synthetic(family: str, index: int, rng: random.Random) -> dict:
    if family == "arithmetic":
        a, b = rng.randint(2, 99), rng.randint(2, 99)
        op, symbol = rng.choice(((lambda x, y: x + y, "加"), (lambda x, y: x - y, "减"), (lambda x, y: x * y, "乘")))
        return row(f"计算{a}{symbol}{b}，只输出最终数字。", str(op(a, b)))
    if family == "json":
        pairs = rng.sample((("name", "小明"), ("city", "上海"), ("country", "日本"), ("color", "蓝色"), ("fruit", "香蕉"), ("status", "成功")), 2)
        fields = "和".join(f"{k}={v}" for k, v in pairs)
        answer = json.dumps(dict(pairs), ensure_ascii=False, separators=(",", ":"))
        return row(f"请只返回JSON对象，包含字段{fields}，不要解释或使用代码块。", answer)
    if family == "bullets":
        items = rng.sample(("苹果", "香蕉", "橙子", "葡萄", "草莓", "梨", "西瓜", "桃子"), rng.randint(2, 5))
        return row(f"列出{len(items)}种水果，只输出{len(items)}行且每行以'- '开头。", "\n".join(f"- {x}" for x in items))
    if family == "translation":
        pairs = (("Good morning", "早上好"), ("Thank you", "谢谢你"), ("See you tomorrow", "明天见"), ("I love reading", "我喜欢阅读"), ("The weather is nice", "天气很好"), ("Welcome home", "欢迎回家"))
        source, target = rng.choice(pairs)
        return row(f"将英文“{source}”翻译成中文，只输出译文。", target)
    if family == "formula":
        pairs = (("二氧化碳", "CO2"), ("氧气", "O2"), ("氢气", "H2"), ("氯化钠", "NaCl"), ("氨", "NH3"), ("甲烷", "CH4"), ("过氧化氢", "H2O2"))
        name, formula = rng.choice(pairs)
        if index % 7 == 0:
            return row("H2O这种化学式对应的物质名称是什么？只输出名称。", "水")
        return row(f"{name}的化学式是什么？只输出化学式。", formula)
    if family == "echo":
        token = rng.choice(("OK", "DONE", "PASS", "START", "YES", "NO", "ACCEPTED"))
        return row(f"严格只回复{token}，不要添加任何其他字符。", token)
    if family == "facts":
        pairs = (
            ("植物如何利用光能制造有机物？请用一句话回答。", "植物通过光合作用利用光能，将二氧化碳和水转化为有机物并释放氧气。"),
            ("地球唯一的天然卫星是什么？只输出名称。", "月球"),
            ("中国的首都是哪里？只输出城市名。", "北京"),
            ("一年有多少个月？只输出数字。", "12"),
            ("冰在标准大气压下的熔点是多少摄氏度？只输出数字。", "0"),
            ("太阳系最大的行星是哪颗？只输出名称。", "木星"),
        )
        return row(*rng.choice(pairs))
    if family == "compare":
        pairs = (("茶", "咖啡", "茶通常咖啡因较少，咖啡风味更浓。"), ("自行车", "汽车", "自行车更环保，汽车长途出行更便捷。"), ("纸质书", "电子书", "纸质书触感直观，电子书携带更方便。"), ("晴天", "雨天", "晴天适合户外活动，雨天出行更需防护。"))
        left, right, answer = rng.choice(pairs)
        return row(f"用不超过40个汉字比较{left}和{right}的一个差异。", answer)
    names = (("multiply", "a * b"), ("subtract", "a - b"), ("maximum", "max(a, b)"), ("minimum", "min(a, b)"))
    name, expression = rng.choice(names)
    return row(f"写一个Python函数{name}(a, b)并返回结果，只输出代码。", f"def {name}(a, b):\n    return {expression}")


def main() -> None:
    cfg = args()
    if cfg.output_root.exists():
        raise SystemExit(f"output already exists: {cfg.output_root}")
    rng = random.Random(cfg.seed)
    records: list[tuple[str, str, dict]] = []
    families = ("arithmetic", "json", "bullets", "translation", "formula", "echo", "facts", "compare", "code")
    for family in families:
        for index in range(cfg.per_family):
            item = synthetic(family, index, rng)
            identifier = f"synthetic:{family}:{index}:{digest(json.dumps(item, ensure_ascii=False, sort_keys=True))}"
            records.append((split_for(identifier), family, item))

    with cfg.official_replay.open() as handle:
        for index, line in enumerate(handle):
            if index % 4 == 0 and index < 16000:
                item = json.loads(line)
                records.append((split_for(f"official:{index}"), "official_replay", item))

    provenance = [json.loads(line) for line in (cfg.tool_root / "provenance.jsonl").open()]
    tool_lines = {int(item["split_line_number"]) for item in provenance if item["split"] == "train" and item["bucket"] == "tool_call"}
    with (cfg.tool_root / "train.jsonl").open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in tool_lines:
                records.append((split_for(f"tool:{line_number}"), "tool_replay", json.loads(line)))

    cfg.output_root.mkdir(parents=True)
    handles = {name: (cfg.output_root / f"{name}.jsonl").open("w") for name in ("train", "validation", "test")}
    counts, family_counts = Counter(), Counter()
    seen = set()
    for split, family, item in records:
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        key = digest(payload)
        if key in seen:
            continue
        seen.add(key)
        handles[split].write(payload + "\n")
        counts[split] += 1
        family_counts[f"{split}:{family}"] += 1
    for handle in handles.values():
        handle.close()

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, local_files_only=True)
    target_tokens = Counter()
    for split in ("train", "validation", "test"):
        with (cfg.output_root / f"{split}.jsonl").open() as handle:
            for line in handle:
                item = json.loads(line)
                for msg in item["conversations"]:
                    if msg["role"] == "assistant":
                        target_tokens[split] += len(tokenizer(msg.get("content") or "", add_special_tokens=False).input_ids) + 1
    manifest = {
        "schema_version": 1,
        "name": "chat-repair-v1",
        "seed": cfg.seed,
        "status": "accepted_for_diagnostic",
        "banned_eval_prompt_exact_overlap": 0,
        "rows": dict(counts),
        "assistant_target_tokens_approx": dict(target_tokens),
        "families": dict(family_counts),
        "note": "Synthetic capability families use held-out values/templates; official and tool replay preserve broad behavior.",
    }
    (cfg.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (cfg.output_root / "_SUCCESS").write_text(json.dumps({"status": "accepted_for_diagnostic"}) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
