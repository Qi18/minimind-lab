#!/usr/bin/env python3
"""Deterministic Chat/Tool and inference-system evaluation for MiniMind SFT."""

import argparse
import ast
import json
import math
import operator
import re
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


CHAT_CASES = [
    {"id": "chat_arithmetic_exact", "prompt": "请计算17加25。只输出最终数字，不要解释。", "score": "exact", "expected": "42", "format": True},
    {"id": "chat_json_format", "prompt": "只输出一个JSON对象，必须恰好包含city和country两个字段，值分别为北京和中国。不要使用Markdown代码块。", "score": "json_city", "format": True},
    {"id": "chat_three_bullets", "prompt": "列出三种常见水果。严格输出三行，每行以'- '开头，不要添加标题。", "score": "three_bullets", "format": True},
    {"id": "chat_translation", "prompt": "把Hello world翻译成中文。只输出译文。", "score": "contains_all", "expected": ["你好", "世界"], "format": False},
    {"id": "chat_formula", "prompt": "水的化学式是什么？只输出化学式。", "score": "formula", "format": True},
    {"id": "chat_python", "prompt": "写一个Python函数add(a, b)，返回a与b的和。只输出代码。", "score": "python_add", "format": False},
    {"id": "chat_exact_ready", "prompt": "严格只回复READY这五个英文字母。", "score": "exact", "expected": "READY", "format": True},
    {"id": "chat_benign_factual", "prompt": "用一句话介绍杭州西湖的两个特点。", "score": "contains_all", "expected": ["西湖"], "format": False},
    {"id": "chat_photosynthesis", "prompt": "用一句话解释光合作用。", "score": "contains_any", "expected": ["光", "二氧化碳", "氧气", "能量"], "format": False},
    {"id": "chat_pet_compare", "prompt": "用不超过50个汉字比较猫和狗作为宠物的一个差异。", "score": "pet_compare", "format": True},
]


TOOLS = {
    "calculate_math": {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    "get_current_time": {"type": "function", "function": {"name": "get_current_time", "description": "获取指定时区的当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []}}},
    "random_number": {"type": "function", "function": {"name": "random_number", "description": "生成指定范围的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    "unit_converter": {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    "get_current_weather": {"type": "function", "function": {"name": "get_current_weather", "description": "查询城市天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "unit": {"type": "string"}}, "required": ["location"]}}},
    "get_exchange_rate": {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询货币汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    "translate_text": {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
    "text_length": {"type": "function", "function": {"name": "text_length", "description": "统计文本长度", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
}


TOOL_CASES = [
    {"id": "tool_math", "prompt": "帮我算一下256乘以37等于多少", "available": ["calculate_math", "get_current_time"], "expected": ["calculate_math"], "answer": ["9472"]},
    {"id": "tool_time", "prompt": "现在几点了？", "available": ["get_current_time", "random_number"], "expected": ["get_current_time"], "answer": ["12:34"]},
    {"id": "tool_distance", "prompt": "帮我把100公里换算成英里", "available": ["unit_converter", "calculate_math"], "expected": ["unit_converter"], "answer": ["62.14", "62.1"]},
    {"id": "tool_random_square", "prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "available": ["random_number", "calculate_math", "text_length"], "expected": ["random_number", "calculate_math"], "answer": ["1764"]},
    {"id": "tool_weather", "prompt": "北京今天天气怎么样？", "available": ["get_current_weather", "get_current_time"], "expected": ["get_current_weather"], "answer": ["22", "晴"]},
    {"id": "tool_exchange", "prompt": "查一下美元兑人民币汇率", "available": ["get_exchange_rate", "get_current_time"], "expected": ["get_exchange_rate"], "answer": ["7.15"]},
    {"id": "tool_translate", "prompt": "把'你好世界'翻译成英文", "available": ["translate_text", "text_length"], "expected": ["translate_text"], "answer": ["hello world"]},
    {"id": "tool_multistep", "prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "available": ["get_current_weather", "unit_converter", "get_current_time"], "expected": ["get_current_weather", "unit_converter"], "answer": ["86"]},
]


REFUSAL_PATTERNS = ("抱歉", "无法回答", "不能回答", "不能帮助", "i can't", "i cannot")
ALLOWED_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}


def safe_arithmetic(expression):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_BINOPS:
            return ALLOWED_BINOPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("unsupported expression")
    return visit(ast.parse(str(expression), mode="eval"))


def strip_thinking(text):
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def repetition_ratio(text, n=4):
    tokens = list(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def generate(model, tokenizer, messages, tools, device, max_new_tokens):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.numel())


def score_chat(case, answer):
    clean = strip_thinking(answer)
    score_type = case["score"]
    format_pass = True
    if score_type == "exact":
        success = clean == case["expected"]
    elif score_type == "json_city":
        try:
            parsed = json.loads(clean)
            success = parsed == {"city": "北京", "country": "中国"}
        except Exception:
            success = False
    elif score_type == "three_bullets":
        lines = [line for line in clean.splitlines() if line.strip()]
        success = len(lines) == 3 and all(line.startswith("- ") for line in lines)
    elif score_type == "contains_all":
        success = all(value.lower() in clean.lower() for value in case["expected"])
    elif score_type == "contains_any":
        success = any(value.lower() in clean.lower() for value in case["expected"])
    elif score_type == "formula":
        normalized = clean.replace("₂", "2").replace(" ", "").lower()
        success = normalized == "h2o"
    elif score_type == "python_add":
        compact = re.sub(r"\s+", " ", clean)
        success = "def add" in compact and bool(re.search(r"return\s+a\s*\+\s*b", clean))
    elif score_type == "pet_compare":
        success = "猫" in clean and "狗" in clean and len(clean) <= 50
    else:
        raise ValueError(score_type)
    if case.get("format"):
        format_pass = success
    refusal = any(pattern in clean.lower() for pattern in REFUSAL_PATTERNS)
    repeat_ratio = repetition_ratio(clean)
    repeated = repeat_ratio > 0.25
    return success and not refusal and not repeated, format_pass, refusal, repeated, repeat_ratio, clean


def parse_tool_calls(text):
    tagged = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    calls = []
    for raw in tagged:
        try:
            value = json.loads(raw.strip())
            if isinstance(value, dict) and isinstance(value.get("name"), str) and isinstance(value.get("arguments", {}), dict):
                calls.append(value)
        except Exception:
            continue
    return tagged, calls


def validate_arguments(name, arguments):
    schema = TOOLS[name]["function"]["parameters"]
    if not isinstance(arguments, dict) or not all(key in arguments for key in schema.get("required", [])):
        return False
    if name == "calculate_math":
        return bool(str(arguments.get("expression", "")).strip())
    if name == "random_number":
        return arguments.get("min") == 1 and arguments.get("max") == 1000
    if name == "unit_converter":
        return isinstance(arguments.get("value"), (int, float))
    if name == "get_current_weather":
        return bool(arguments.get("location"))
    if name == "get_exchange_rate":
        return bool(arguments.get("from_currency")) and bool(arguments.get("to_currency"))
    if name == "translate_text":
        return bool(arguments.get("text")) and bool(arguments.get("target_language"))
    return True


def execute_tool(name, arguments):
    if name == "calculate_math":
        return {"result": str(safe_arithmetic(arguments["expression"]))}
    if name == "get_current_time":
        return {"datetime": "2026-08-27 12:34:56", "timezone": arguments.get("timezone", "Asia/Shanghai")}
    if name == "random_number":
        return {"result": 42}
    if name == "unit_converter":
        value = float(arguments["value"])
        source = str(arguments["from_unit"]).lower()
        target = str(arguments["to_unit"]).lower()
        if "km" in source or "公里" in source:
            result = round(value * 0.621371, 2)
        elif "celsius" in source and "fahrenheit" in target:
            result = round(value * 9 / 5 + 32, 2)
        else:
            return {"error": "unsupported conversion"}
        return {"result": result, "from_unit": source, "to_unit": target}
    if name == "get_current_weather":
        return {"city": arguments["location"], "temperature": 22, "condition": "晴"}
    if name == "get_exchange_rate":
        return {"from": arguments["from_currency"], "to": arguments["to_currency"], "rate": 7.15}
    if name == "translate_text":
        return {"translated": "hello world"}
    if name == "text_length":
        return {"characters": len(arguments["text"]), "words": len(arguments["text"].split())}
    return {"error": "unknown tool"}


def run_tool_case(model, tokenizer, case, device, max_new_tokens):
    messages = [{"role": "user", "content": case["prompt"]}]
    tool_specs = [TOOLS[name] for name in case["available"]]
    all_calls, turns, final_answer = [], [], ""
    invalid_tag_count = 0
    execution_error_count = 0
    for turn_index in range(3):
        response, generated_tokens = generate(model, tokenizer, messages, tool_specs, device, max_new_tokens)
        tagged, calls = parse_tool_calls(response)
        invalid_tag_count += len(tagged) - len(calls)
        turns.append({"turn": turn_index + 1, "response": response, "generated_tokens": generated_tokens, "calls": calls})
        if not calls:
            final_answer = strip_thinking(response)
            break
        messages.append({"role": "assistant", "content": response})
        for call in calls:
            name, arguments = call["name"], call["arguments"]
            all_calls.append(call)
            if name not in case["available"] or name not in TOOLS:
                result = {"error": "unavailable tool"}
                execution_error_count += 1
            else:
                try:
                    result = execute_tool(name, arguments)
                    execution_error_count += int("error" in result)
                except Exception as exc:
                    result = {"error": type(exc).__name__}
                    execution_error_count += 1
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})

    called_names = [call["name"] for call in all_calls]
    selection_pass = set(called_names) == set(case["expected"])
    argument_pass = invalid_tag_count == 0 and selection_pass and all(
        call["name"] in TOOLS and validate_arguments(call["name"], call["arguments"]) for call in all_calls
    )
    execution_pass = selection_pass and execution_error_count == 0
    answer_lower = final_answer.lower()
    final_answer_pass = selection_pass and any(value.lower() in answer_lower for value in case["answer"])
    end_to_end = selection_pass and argument_pass and execution_pass and final_answer_pass
    return {
        "id": case["id"],
        "type": "tool",
        "prompt": case["prompt"],
        "available_tools": case["available"],
        "expected_tools": case["expected"],
        "called_tools": called_names,
        "selection_pass": selection_pass,
        "argument_pass": argument_pass,
        "execution_pass": execution_pass,
        "final_answer_pass": final_answer_pass,
        "end_to_end_pass": end_to_end,
        "invalid_tag_count": invalid_tag_count,
        "execution_error_count": execution_error_count,
        "unfinished": not bool(final_answer),
        "final_answer": final_answer,
        "turns": turns,
    }


def timed_greedy(model, tokenizer, prompt, device, max_new_tokens=64):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, open_thinking=False)
    encoded = tokenizer(text, return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(**encoded, use_cache=True)
        next_token = output.logits[:, -1].argmax(dim=-1)
        past = output.past_key_values
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        first_token_seconds = time.perf_counter() - started
        generated = [int(next_token.item())]
        attention_mask = torch.cat([encoded["attention_mask"], torch.ones((1, 1), dtype=encoded["attention_mask"].dtype, device=device)], dim=1)
        for _ in range(max_new_tokens - 1):
            if generated[-1] == tokenizer.eos_token_id:
                break
            output = model(input_ids=next_token[:, None], attention_mask=attention_mask, past_key_values=past, use_cache=True)
            past = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1)
            generated.append(int(next_token.item()))
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    return first_token_seconds, total_seconds, len(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).eval().to(args.device)

    samples = []
    for case in CHAT_CASES:
        response, generated_tokens = generate(model, tokenizer, [{"role": "user", "content": case["prompt"]}], None, args.device, args.max_new_tokens)
        success, format_pass, refusal, repeated, repeat_ratio, clean = score_chat(case, response)
        samples.append({
            "id": case["id"], "type": "chat", "prompt": case["prompt"], "response": response,
            "scored_response": clean, "success": success, "format_pass": format_pass,
            "refusal_anomaly": refusal, "repetition_anomaly": repeated,
            "repetition_ratio_4gram": repeat_ratio, "generated_tokens": generated_tokens,
        })

    tool_samples = [run_tool_case(model, tokenizer, case, args.device, args.max_new_tokens) for case in TOOL_CASES]
    samples.extend(tool_samples)
    with (args.output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    chat_samples = [sample for sample in samples if sample["type"] == "chat"]
    task_eval = {
        "status": "completed",
        "seed": args.seed,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "open_thinking": False},
        "chat": {
            "cases": len(chat_samples),
            "success_count": sum(sample["success"] for sample in chat_samples),
            "success_rate": sum(sample["success"] for sample in chat_samples) / len(chat_samples),
            "format_cases": sum(case.get("format", False) for case in CHAT_CASES),
            "format_success_count": sum(sample["format_pass"] for sample, case in zip(chat_samples, CHAT_CASES) if case.get("format", False)),
            "format_success_rate": sum(sample["format_pass"] for sample, case in zip(chat_samples, CHAT_CASES) if case.get("format", False)) / sum(case.get("format", False) for case in CHAT_CASES),
            "refusal_anomaly_count": sum(sample["refusal_anomaly"] for sample in chat_samples),
            "repetition_anomaly_count": sum(sample["repetition_anomaly"] for sample in chat_samples),
        },
        "tool": {
            "cases": len(tool_samples),
            "tool_selection_accuracy": sum(sample["selection_pass"] for sample in tool_samples) / len(tool_samples),
            "argument_validity_rate": sum(sample["argument_pass"] for sample in tool_samples) / len(tool_samples),
            "execution_success_rate": sum(sample["execution_pass"] for sample in tool_samples) / len(tool_samples),
            "final_answer_accuracy": sum(sample["final_answer_pass"] for sample in tool_samples) / len(tool_samples),
            "end_to_end_success_rate": sum(sample["end_to_end_pass"] for sample in tool_samples) / len(tool_samples),
            "unfinished_rate": sum(sample["unfinished"] for sample in tool_samples) / len(tool_samples),
            "invalid_tool_call_tag_count": sum(sample["invalid_tag_count"] for sample in tool_samples),
        },
        "samples_path": "eval/samples.jsonl",
    }
    (args.output_dir / "task_eval.json").write_text(json.dumps(task_eval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    benchmark_prompt = "用一句话解释机器学习。"
    for _ in range(5):
        timed_greedy(model, tokenizer, benchmark_prompt, args.device)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    timings = [timed_greedy(model, tokenizer, benchmark_prompt, args.device) for _ in range(20)]
    ttft = [item[0] for item in timings]
    totals = [item[1] for item in timings]
    throughputs = [item[2] / item[1] for item in timings]
    system_metrics = {
        "status": "completed",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": "float16",
        "warmup_runs": 5,
        "measured_runs": 20,
        "prompt": benchmark_prompt,
        "max_new_tokens": 64,
        "first_token_latency_seconds": {"median": statistics.median(ttft), "p95": percentile(ttft, 0.95)},
        "total_latency_seconds": {"median": statistics.median(totals), "p95": percentile(totals, 0.95)},
        "decode_tokens_per_second": {"median": statistics.median(throughputs), "p95": percentile(throughputs, 0.95)},
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0,
        "peak_reserved_memory_mib": torch.cuda.max_memory_reserved() / 1024 ** 2 if torch.cuda.is_available() else 0,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (args.output_dir / "system_metrics.json").write_text(json.dumps(system_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"task_eval": task_eval, "system_metrics": system_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
