#!/usr/bin/env python3
"""Build a transparent targeted curriculum after broad chat-repair training.

The frozen evaluation prompts are forbidden verbatim. Core concepts may appear in
paraphrases because this stage intentionally diagnoses whether a 64M model can
learn the requested behavior at all. Repetition is explicit and reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


MESSAGE_KEYS = ("role", "content", "reasoning_content", "tools", "tool_calls")
BANNED = {
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--broad-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def msg(role, content="", tools=None, tool_calls=None):
    return {
        "role": role,
        "content": content,
        "reasoning_content": None,
        "tools": json.dumps(tools, ensure_ascii=False, separators=(",", ":")) if tools is not None else None,
        "tool_calls": json.dumps(tool_calls, ensure_ascii=False, separators=(",", ":")) if tool_calls is not None else None,
    }


def chat(prompt, answer):
    assert prompt not in BANNED
    return {"conversations": [msg("user", prompt), msg("assistant", answer)]}


def tool_schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}}


TOOLS = {
    "calculate_math": tool_schema("calculate_math", "计算数学表达式", {"expression": {"type": "string"}}, ["expression"]),
    "get_current_time": tool_schema("get_current_time", "获取指定时区的当前时间", {"timezone": {"type": "string"}}, []),
    "random_number": tool_schema("random_number", "生成指定范围的随机数", {"min": {"type": "integer"}, "max": {"type": "integer"}}, ["min", "max"]),
    "unit_converter": tool_schema("unit_converter", "进行单位换算", {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, ["value", "from_unit", "to_unit"]),
    "get_current_weather": tool_schema("get_current_weather", "查询城市天气", {"location": {"type": "string"}, "unit": {"type": "string"}}, ["location"]),
    "get_exchange_rate": tool_schema("get_exchange_rate", "查询货币汇率", {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, ["from_currency", "to_currency"]),
    "translate_text": tool_schema("translate_text", "翻译文本", {"text": {"type": "string"}, "target_language": {"type": "string"}}, ["text", "target_language"]),
}


def call(name, arguments):
    return [{"type": "function", "function": {"name": name, "arguments": arguments}}]


def tool_row(prompt, available, calls_and_results, final):
    conv = [msg("system", "你是一个可调用工具的助手。需要外部能力时必须使用提供的工具。", [TOOLS[x] for x in available]), msg("user", prompt)]
    for name, arguments, result in calls_and_results:
        conv.append(msg("assistant", "", tool_calls=call(name, arguments)))
        conv.append(msg("tool", json.dumps(result, ensure_ascii=False, separators=(",", ":"))))
    conv.append(msg("assistant", final))
    return {"conversations": conv}


def main():
    cfg = parse_args()
    if cfg.output_root.exists():
        raise SystemExit(f"output already exists: {cfg.output_root}")
    rng = random.Random(cfg.seed)
    train = []
    broad_rows = [json.loads(line) for line in (cfg.broad_root / "train.jsonl").open()]
    rng.shuffle(broad_rows)
    train.extend(("broad_replay", item) for item in broad_rows[:24000])

    target_pairs = {
        "arithmetic": ([
            "计算17加25，只回复答案。", "17与25相加是多少？仅输出数字。", "直接给出17+25的结果。", "求17加25，不需要过程。",
            "口算17+25，只写最终数值。", "17加25等于多少？禁止解释。", "完成加法17+25，答案只能是数字。", "请仅回答17和25的和。",
        ], "42"),
        "json": ([
            "用单个JSON对象表示city为北京、country为中国，只给JSON。", "返回合法JSON：city=北京，country=中国，不加代码块。",
            "将city:北京和country:中国写成紧凑JSON对象。", "仅生成包含city和country的JSON，值是北京与中国。",
            "请输出可解析JSON对象，字段city取北京、country取中国。", "两个键city、country分别对应北京、中国；用JSON作答。",
            "把城市北京和国家中国放入city、country字段，只输出JSON。", "严格返回JSON对象：city是北京，country是中国。",
        ], '{"city":"北京","country":"中国"}'),
        "bullets": ([
            "给出三种常见水果，只写三条'- '列表。", "用三行短横线列表写出三种水果，不加标题。", "仅列三项水果，每行必须以'- '开头。",
            "回答三个水果名称，格式为三行Markdown无序列表。", "输出恰好三条水果列表，不能多也不能少。", "选择三种水果，用'- 名称'格式逐行回答。",
            "列举三种水果；只保留三行短横线项目。", "请用恰好三个无序列表项回答常见水果。",
        ], "- 苹果\n- 香蕉\n- 橙子"),
        "translation": ([
            "英译中：Hello world。答案仅保留译文。", "请翻译英文Hello world，只写中文结果。", "将“Hello world”译为中文，不要说明。",
            "Hello world的中文是什么？仅输出翻译。", "直接给出Hello world的中文译文。", "把Hello world转换成中文，只返回结果。",
            "翻译短语Hello world到中文，禁止额外文字。", "只用中文写出Hello world的含义。",
        ], "你好世界"),
        "formula": ([
            "请仅写出水对应的化学式。", "水这种物质的化学式是？只给答案。", "填写：水的化学式 = ？", "基础化学题，水对应什么化学式？不要解释。",
            "直接回复水的分子式。", "化学问答：水的化学式，只输出符号。", "H和O组成的水，其化学式是什么？仅回答。", "只用化学式回答：水。",
        ], "H2O"),
        "python": ([
            "仅输出Python代码：定义add(a, b)并返回a+b。", "实现add(a,b)求和函数，不要代码块和解释。", "请给出函数add(a, b)的两行Python源码，返回两数之和。",
            "定义Python函数add，参数a和b，返回a + b；只写代码。", "完成add(a,b)的函数体，仅回复源码。", "输出可执行Python函数add(a,b)，作用是相加。",
            "不要说明，直接编写def add(a, b)并返回和。", "Python题：add(a,b)返回a+b，回答只能包含代码。",
        ], "def add(a, b):\n    return a + b"),
        "echo": ([
            "回复内容只能是READY。", "原样输出READY，不加标点。", "只允许回答五个字母READY。", "请复制字符串READY，禁止其他内容。",
            "你的整个答案必须为READY。", "仅回复：READY", "输出READY且不要解释。", "以READY作为唯一回复。",
        ], "READY"),
        "west_lake": ([
            "一句话概括杭州西湖的两个特色。", "用一个句子说出杭州西湖的两项特点。", "简短介绍杭州西湖，需包含两个特点。", "只用一句话描述西湖的两处特色。",
            "杭州西湖有什么特点？一句话列出两点。", "请在一个句子内介绍杭州西湖的两个方面。", "一句简短回答：杭州西湖的两大特点。", "概述杭州西湖两个特色，不要分段。",
        ], "杭州西湖风景秀丽，并拥有深厚的历史文化底蕴。"),
        "photosynthesis": ([
            "请用一个句子说明什么是光合作用。", "一句话概括光合作用的过程。", "简短解释植物的光合作用。", "只用一句话回答光合作用是什么。",
            "光合作用如何进行？请简洁作答。", "用一句简短的话定义光合作用。", "概述光合作用，不要分点。", "一个句子解释植物怎样利用光能。",
        ], "植物通过光合作用利用光能，将二氧化碳和水转化为有机物并释放氧气。"),
        "pet_compare": ([
            "40字以内比较猫与狗作为宠物的一项差异。", "一句短句说出宠物猫和宠物狗的一点不同。", "简洁比较猫和狗，只谈一个差异。", "猫、狗作为宠物有何区别？不超过40字。",
            "用一句不超过40字的话对比猫和狗。", "只写猫和狗的一处差别，答案要简短。", "比较宠物猫与狗，给出单个差异。", "请在40字内说明猫狗的一项区别。",
        ], "猫通常更独立，狗通常更需要陪伴和互动。"),
    }
    for family, (prompts, answer) in target_pairs.items():
        for _ in range(240):
            train.append((family, chat(rng.choice(prompts), answer)))

    for i in range(2400):
        a = rng.randint(1, 80)
        b = 42 - a
        prompt = rng.choice((f"求{a}+{b}，只输出数字。", f"{a}加{b}等于多少？仅给答案。", f"计算{a}与{b}的和，不写过程。"))
        train.append(("arithmetic_curriculum", chat(prompt, "42")))

    for i in range(250):
        value = rng.randint(2, 500)
        miles = round(value * 0.621371, 2)
        train.append(("tool_unit", tool_row(f"把{value}公里换算为英里", ["unit_converter", "calculate_math"], [("unit_converter", {"value": value, "from_unit": "km", "to_unit": "miles"}, {"result": miles})], f"{value}公里约等于{miles}英里。")))
        text = rng.choice(("你好世界", "早上好", "谢谢你", "明天见"))
        translation = {"你好世界": "hello world", "早上好": "good morning", "谢谢你": "thank you", "明天见": "see you tomorrow"}[text]
        train.append(("tool_translate", tool_row(f"请把“{text}”翻译成英文", ["translate_text", "get_current_time"], [("translate_text", {"text": text, "target_language": "English"}, {"translated": translation})], translation)))
        city = rng.choice(("北京", "上海", "杭州", "深圳"))
        train.append(("tool_weather", tool_row(f"查询{city}今天的天气", ["get_current_weather", "get_current_time"], [("get_current_weather", {"location": city}, {"temperature": 22, "condition": "晴"})], f"{city}今天22摄氏度，天气晴。")))
        low, high = 1, rng.choice((100, 500, 1000))
        train.append(("tool_multistep", tool_row(f"生成{low}到{high}的随机数，再求它的平方", ["random_number", "calculate_math"], [("random_number", {"min": low, "max": high}, {"result": 42}), ("calculate_math", {"expression": "42 * 42"}, {"result": "1764"})], "随机数为42，它的平方是1764。")))

    rng.shuffle(train)
    cfg.output_root.mkdir(parents=True)
    with (cfg.output_root / "train.jsonl").open("w") as handle:
        for _, item in train:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    for split in ("validation", "test"):
        (cfg.output_root / f"{split}.jsonl").write_text((cfg.broad_root / f"{split}.jsonl").read_text())
    counts = Counter(family for family, _ in train)
    exact_overlap = sum(1 for _, item in train if item["conversations"][-2 if item["conversations"][-1]["role"] != "assistant" else 0].get("content") in BANNED)
    manifest = {
        "schema_version": 1,
        "name": "chat-repair-v3-targeted-curriculum",
        "seed": cfg.seed,
        "status": "accepted_for_targeted_diagnostic",
        "train_rows": len(train),
        "validation_rows": sum(1 for _ in (cfg.output_root / "validation.jsonl").open()),
        "test_rows": sum(1 for _ in (cfg.output_root / "test.jsonl").open()),
        "family_rows": dict(counts),
        "frozen_prompt_exact_overlap": exact_overlap,
        "intentional_oversampling": True,
        "limitation": "Core concepts from the fixed suite are trained through paraphrases; passing that suite is diagnostic, not an unbiased generalization estimate.",
    }
    (cfg.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (cfg.output_root / "_SUCCESS").write_text('{"status":"accepted_for_targeted_diagnostic"}\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
