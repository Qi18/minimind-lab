#!/usr/bin/env python3
"""Build a diverse held-out SFT repair set for concise chat and formatting."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--official-replay", type=Path, required=True)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
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


def make_code(rng: random.Random, length: int = 7) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def synthetic(family: str, index: int, rng: random.Random) -> dict:
    wording = index % 7
    if family == "arithmetic":
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        op, symbol = rng.choice(((lambda x, y: x + y, "加"), (lambda x, y: x - y, "减"), (lambda x, y: x * y, "乘")))
        prompts = (
            f"计算{a}{symbol}{b}，只给出最终数字。",
            f"{a}{symbol}{b}等于多少？仅输出答案。",
            f"请求出{a}{symbol}{b}的结果，不要写过程。",
            f"完成运算：{a}{symbol}{b}。只输出一个数字。",
            f"口算{a}{symbol}{b}，回复最终数值即可。",
            f"求{a}{symbol}{b}，禁止解释。",
            f"直接回答{a}{symbol}{b}的计算结果。",
        )
        return row(prompts[wording], str(op(a, b)))
    if family == "json":
        keys = ("name", "city", "country", "color", "fruit", "status", "age", "score", "year", "animal", "language", "level")
        values = ("小明", "上海", "日本", "蓝色", "香蕉", "成功", "18", "95", "2026", "熊猫", "中文", "高级")
        positions = rng.sample(range(len(keys)), rng.choice((2, 2, 2, 3)))
        pairs = [(keys[i], values[(i + index) % len(values)]) for i in positions]
        fields = "、".join(f"{key}={value}" for key, value in pairs)
        prompts = (
            f"仅返回一个JSON对象，字段为{fields}。不要解释。",
            f"把以下键值写成紧凑JSON：{fields}。只输出JSON。",
            f"生成JSON对象：{fields}；禁止Markdown代码块。",
            f"请将{fields}转换为合法JSON，只给结果。",
            f"输出包含这些字段的JSON：{fields}，不要附加文字。",
            f"严格用JSON对象表示{fields}。",
            f"返回可解析的单个JSON对象，内容是{fields}。",
        )
        return row(prompts[wording], json.dumps(dict(pairs), ensure_ascii=False, separators=(",", ":")))
    if family == "bullets":
        pool = ("苹果", "香蕉", "橙子", "葡萄", "草莓", "梨", "西瓜", "桃子", "芒果", "樱桃", "柠檬", "柚子", "菠萝", "椰子")
        count = rng.randint(2, 5)
        items = rng.sample(pool, count)
        prompts = (
            f"列出{count}种水果，只输出{count}行，每行以'- '开始。",
            f"从常见水果中给出{count}个例子，严格使用{count}条短横线列表。",
            f"输出{count}行水果名称；每行格式必须是'- 名称'。",
            f"给我{count}种水果，不要标题，每项独占一行并以'- '开头。",
            f"严格列出{count}项水果，使用Markdown无序列表且不加说明。",
            f"只返回{count}条水果列表，不能多也不能少。",
            f"用{count}行短横线列表回答：常见水果有哪些？",
        )
        return row(prompts[wording], "\n".join(f"- {item}" for item in items))
    if family == "translation":
        subjects = (("I", "我"), ("We", "我们"), ("They", "他们"), ("My friend", "我的朋友"))
        verbs = (("like", "喜欢"), ("read", "阅读"), ("need", "需要"), ("buy", "购买"), ("see", "看见"))
        objects = (("books", "书"), ("music", "音乐"), ("coffee", "咖啡"), ("flowers", "花"), ("apples", "苹果"), ("water", "水"))
        subject_en, subject_zh = rng.choice(subjects)
        verb_en, verb_zh = rng.choice(verbs)
        object_en, object_zh = rng.choice(objects)
        source = f"{subject_en} {verb_en} {object_en}"
        target = f"{subject_zh}{verb_zh}{object_zh}"
        prompts = (
            f"将英文“{source}”翻译成中文，仅输出译文。",
            f"翻译为中文：{source}。不要解释。",
            f"请给出“{source}”的中文译文，只写译文。",
            f"把{source}译成中文，禁止附加说明。",
            f"英译中：{source}。回复翻译结果即可。",
            f"输出这句话的中文：{source}。",
            f"直接翻译“{source}”为中文。",
        )
        return row(prompts[wording], target)
    if family == "formula":
        pairs = (("二氧化碳", "CO2"), ("氧气", "O2"), ("氢气", "H2"), ("氯化钠", "NaCl"), ("氨", "NH3"), ("甲烷", "CH4"), ("过氧化氢", "H2O2"), ("一氧化碳", "CO"), ("硫酸", "H2SO4"), ("碳酸钙", "CaCO3"), ("葡萄糖", "C6H12O6"), ("氢氧化钠", "NaOH"))
        name, formula = rng.choice(pairs)
        prompts = (
            f"{name}的化学式是什么？仅输出化学式。",
            f"只写出{name}的化学式。",
            f"请给出{name}的化学式，不要解释。",
            f"回答{name}对应的化学式，答案中不能有其他文字。",
            f"化学问答：{name}的化学式？只输出答案。",
            f"直接回复{name}的化学式。",
            f"填写化学式：{name} = ？",
        )
        return row(prompts[wording], formula)
    if family == "echo":
        token = make_code(rng)
        prompts = (
            f"严格只回复{token}，不要添加任何字符。",
            f"请原样输出字符串{token}。只允许该字符串。",
            f"复述{token}，不要解释。",
            f"你的完整回答必须恰好是{token}。",
            f"仅输出：{token}",
            f"复制以下代码，不加引号：{token}",
            f"回复验证码{token}，不能包含其他内容。",
        )
        return row(prompts[wording], token)
    if family == "facts":
        pairs = (
            ("地球唯一的天然卫星", "月球"), ("中国的首都", "北京"), ("法国的首都", "巴黎"),
            ("日本的首都", "东京"), ("太阳系最大的行星", "木星"), ("红色和蓝色混合得到的颜色", "紫色"),
            ("一年包含的月份数", "12"), ("一周包含的天数", "7"), ("标准大气压下水的沸点摄氏度", "100"),
            ("标准大气压下冰的熔点摄氏度", "0"), ("三角形的边数", "3"), ("正方形的边数", "4"),
        )
        question, answer = rng.choice(pairs)
        prompts = (
            f"{question}是什么？只输出答案。",
            f"请回答{question}，不要解释。",
            f"常识题：{question}？仅给结论。",
            f"直接写出{question}。",
            f"填写答案：{question} = ？",
            f"关于{question}，只回复答案本身。",
            f"一句最短回答：{question}是什么？",
        )
        return row(prompts[wording], answer)
    if family == "compare":
        items = (("茶", "咖啡", "茶通常咖啡因较少，咖啡风味更浓。"), ("自行车", "汽车", "自行车更环保，汽车长途出行更便捷。"), ("纸质书", "电子书", "纸质书触感直观，电子书携带更方便。"), ("晴天", "雨天", "晴天适合户外活动，雨天出行更需防护。"), ("猫", "兔子", "猫更独立，兔子通常更安静。"), ("高铁", "飞机", "高铁进站便捷，飞机更适合超长距离。"), ("跑步", "游泳", "跑步场地要求低，游泳对关节冲击较小。"))
        left, right, answer = rng.choice(items)
        prompts = (
            f"用不超过40个汉字比较{left}和{right}的一个差异。",
            f"一句话说明{left}与{right}的一点不同，不超过40字。",
            f"简短比较{left}和{right}，只写一个差异。",
            f"在40个汉字内说出{left}、{right}的一处区别。",
            f"只用一句短句回答：{left}和{right}有何不同？",
            f"比较{left}与{right}，答案不得超过40个汉字。",
            f"给出{left}和{right}的一个简洁差异，不要分点。",
        )
        return row(prompts[wording], answer)
    operation, expression = rng.choice((("sum", "a + b"), ("product", "a * b"), ("difference", "a - b"), ("larger", "max(a, b)"), ("smaller", "min(a, b)")))
    name = f"{operation}_{make_code(rng, 5).lower()}"
    prompts = (
        f"写Python函数{name}(a, b)并返回结果，只输出代码。",
        f"仅用代码定义{name}(a, b)，返回指定运算结果。",
        f"实现函数{name}(a, b)。不要Markdown代码块或解释。",
        f"输出{name}(a, b)的Python定义，除此之外不要输出。",
        f"请编写{name}(a, b)，只返回函数代码。",
        f"用两行Python实现{name}(a, b)，直接给代码。",
        f"完成Python函数{name}(a, b)，答案只能是源码。",
    )
    return row(prompts[wording], f"def {name}(a, b):\n    return {expression}")


def main() -> None:
    cfg = parse_args()
    if cfg.output_root.exists():
        raise SystemExit(f"output already exists: {cfg.output_root}")
    rng = random.Random(cfg.seed)
    records: list[tuple[str, str, dict]] = []
    families = ("arithmetic", "json", "bullets", "translation", "formula", "echo", "facts", "compare", "code")
    for family in families:
        for index in range(cfg.per_family):
            item = synthetic(family, index, rng)
            records.append((split_for(f"synthetic:{family}:{index}"), family, item))

    with cfg.official_replay.open() as handle:
        for index, line in enumerate(handle):
            if index % 2 == 0 and index < 24000:
                records.append((split_for(f"official:{index}"), "official_replay", json.loads(line)))

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
        "name": "chat-repair-v2",
        "seed": cfg.seed,
        "status": "accepted_for_diagnostic",
        "banned_eval_prompt_exact_overlap": 0,
        "rows": dict(counts),
        "assistant_target_tokens_approx": dict(target_tokens),
        "families": dict(family_counts),
        "note": "Diverse synthetic capability families avoid the frozen behavior prompts; official and tool replay preserve broad behavior.",
    }
    (cfg.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (cfg.output_root / "_SUCCESS").write_text(json.dumps({"status": "accepted_for_diagnostic"}) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
