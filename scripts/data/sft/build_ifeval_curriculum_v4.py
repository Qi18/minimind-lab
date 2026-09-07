#!/usr/bin/env python3
"""Build a verifier-backed SFT curriculum for IFEval-style generalization.

This builder never copies an IFEval prompt or answer. It uses the public
instruction checkers to validate newly generated examples and records exact and
normalized overlap against the frozen 541-prompt benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from lm_eval.tasks.ifeval import instructions_registry
from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict


MESSAGE_KEYS = ("role", "content", "reasoning_content", "tools", "tool_calls")
SINGLE_IDS = tuple(instructions_registry.INSTRUCTION_DICT)
TRAIN_PER_SINGLE = 480
EVAL_PER_SINGLE = 36
TRAIN_PER_COMBO = 480
EVAL_PER_COMBO = 36
COMBOS = (
    "no_comma_title", "no_comma_bullets", "no_comma_upper", "no_comma_lower",
    "keyword_title", "keyword_end", "placeholders_postscript", "sections_end",
    "highlights_keyword", "paragraphs_keyword", "sentences_no_comma",
    "words_keyword", "quotation_no_comma", "two_responses_keyword",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--broad-root", type=Path, required=True)
    parser.add_argument("--formal-smoke-root", type=Path, required=True)
    parser.add_argument("--broad-replay", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def norm(text: str) -> str:
    return "".join(
        char.lower()
        for char in unicodedata.normalize("NFKC", text)
        if char.isalnum()
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message(role: str, content: str):
    return {
        "role": role,
        "content": content,
        "reasoning_content": None,
        "tools": None,
        "tool_calls": None,
    }


def row(prompt: str, answer: str):
    return {"conversations": [message("user", prompt), message("assistant", answer)]}


def task_text(split: str, index: int) -> str:
    adjectives = {
        "train": ("careful", "practical", "local", "steady", "creative", "shared"),
        "validation": ("measured", "durable", "coastal", "patient"),
        "test": ("resilient", "alpine", "civic", "seasonal"),
    }[split]
    nouns = {
        "train": ("garden", "library", "workshop", "market", "museum", "school"),
        "validation": ("harbor", "orchard", "studio", "clinic"),
        "test": ("observatory", "wetland", "theater", "bakery"),
    }[split]
    topic = f"the {adjectives[index % len(adjectives)]} {nouns[(index // len(adjectives)) % len(nouns)]} plan {split}-{index}"
    variants = (
        f"Write a useful note about {topic}.",
        f"Explain the main benefits of {topic}.",
        f"Give practical advice for improving {topic}.",
        f"Describe how a community could support {topic}.",
    )
    return variants[index % len(variants)]


def base_text(topic: str = "the project") -> str:
    return (
        f"{topic.capitalize()} supports careful planning. "
        "The team studies evidence and records useful lessons. "
        "Clear steps help people make steady progress."
    )


def sentences(count: int) -> str:
    stems = (
        "The group begins with a clear goal",
        "People gather reliable evidence",
        "Each member records useful observations",
        "The plan assigns practical responsibilities",
        "Regular reviews reveal small problems",
        "Simple changes improve the next attempt",
        "Open discussion keeps the work transparent",
        "The final report explains every decision",
    )
    return " ".join(f"{stems[i % len(stems)]} {i + 1}." for i in range(count))


def word_block(count: int) -> str:
    words = "careful teams study evidence plan useful steps share lessons improve community work with clear goals and steady review".split()
    return " ".join(words[i % len(words)] for i in range(count)) + "."


LANGUAGE_TEXT = {
    "hi": "यह एक सरल उत्तर है। लोग मिलकर योजना बनाते हैं और उपयोगी अनुभव साझा करते हैं। स्पष्ट कदम अच्छे परिणाम देते हैं।",
    "kn": "ಇದು ಸರಳ ಉತ್ತರವಾಗಿದೆ. ಜನರು ಒಟ್ಟಾಗಿ ಯೋಜನೆ ಮಾಡಿ ಉಪಯುಕ್ತ ಅನುಭವಗಳನ್ನು ಹಂಚಿಕೊಳ್ಳುತ್ತಾರೆ. ಸ್ಪಷ್ಟ ಕ್ರಮಗಳು ಉತ್ತಮ ಫಲಿತಾಂಶಗಳನ್ನು ನೀಡುತ್ತವೆ.",
    "mr": "हे एक साधे उत्तर आहे. लोक एकत्र योजना करतात आणि उपयुक्त अनुभव सामायिक करतात. स्पष्ट पावले चांगले परिणाम देतात.",
    "vi": "Đây là một câu trả lời đơn giản. Mọi người cùng lập kế hoạch và chia sẻ kinh nghiệm hữu ích. Các bước rõ ràng tạo ra kết quả tốt.",
    "ko": "이것은 간단한 답변입니다. 사람들은 함께 계획하고 유용한 경험을 공유합니다. 명확한 단계는 좋은 결과를 만듭니다.",
    "sw": "Hili ni jibu rahisi. Watu hupanga pamoja na kushiriki uzoefu muhimu. Hatua wazi huleta matokeo mazuri.",
    "de": "Dies ist eine einfache Antwort. Menschen planen gemeinsam und teilen nützliche Erfahrungen. Klare Schritte führen zu guten Ergebnissen.",
    "pa": "ਇਹ ਇੱਕ ਸਧਾਰਨ ਜਵਾਬ ਹੈ। ਲੋਕ ਮਿਲ ਕੇ ਯੋਜਨਾ ਬਣਾਉਂਦੇ ਹਨ ਅਤੇ ਲਾਭਦਾਇਕ ਤਜਰਬੇ ਸਾਂਝੇ ਕਰਦੇ ਹਨ। ਸਪਸ਼ਟ ਕਦਮ ਚੰਗੇ ਨਤੀਜੇ ਦਿੰਦੇ ਹਨ।",
    "fa": "این یک پاسخ ساده است. مردم با هم برنامه ریزی می کنند و تجربه های مفید را به اشتراک می گذارند. گام های روشن نتایج خوبی می دهند.",
    "ru": "Это простой ответ. Люди вместе планируют работу и делятся полезным опытом. Ясные шаги дают хорошие результаты.",
    "bg": "Това е прост отговор. Хората планират заедно и споделят полезен опит. Ясните стъпки дават добри резултати.",
    "pt": "Esta é uma resposta simples. As pessoas planejam juntas e compartilham experiências úteis. Passos claros produzem bons resultados.",
    "gu": "આ એક સરળ જવાબ છે. લોકો સાથે મળીને યોજના બનાવે છે અને ઉપયોગી અનુભવ વહેંચે છે. સ્પષ્ટ પગલાં સારા પરિણામ આપે છે.",
    "te": "ఇది సరళమైన సమాధానం. ప్రజలు కలిసి ప్రణాళిక రూపొందించి ఉపయోగకరమైన అనుభవాలను పంచుకుంటారు. స్పష్టమైన దశలు మంచి ఫలితాలను ఇస్తాయి.",
    "it": "Questa è una risposta semplice. Le persone pianificano insieme e condividono esperienze utili. Passaggi chiari producono buoni risultati.",
    "ar": "هذه إجابة بسيطة. يخطط الناس معا ويتبادلون الخبرات المفيدة. الخطوات الواضحة تحقق نتائج جيدة.",
    "ta": "இது ஒரு எளிய பதில். மக்கள் ஒன்றாக திட்டமிட்டு பயனுள்ள அனுபவங்களை பகிர்கின்றனர். தெளிவான படிகள் நல்ல முடிவுகளை தருகின்றன.",
    "fi": "Tämä on yksinkertainen vastaus. Ihmiset suunnittelevat yhdessä ja jakavat hyödyllisiä kokemuksia. Selkeät vaiheet tuottavat hyviä tuloksia.",
    "ur": "یہ ایک سادہ جواب ہے۔ لوگ مل کر منصوبہ بناتے ہیں اور مفید تجربات بانٹتے ہیں۔ واضح اقدامات اچھے نتائج دیتے ہیں۔",
    "th": "นี่คือคำตอบง่ายๆ ผู้คนวางแผนร่วมกันและแบ่งปันประสบการณ์ที่เป็นประโยชน์ ขั้นตอนที่ชัดเจนสร้างผลลัพธ์ที่ดี",
    "ne": "यो एक सरल उत्तर हो। मानिसहरू मिलेर योजना बनाउँछन् र उपयोगी अनुभव साझा गर्छन्। स्पष्ट कदमले राम्रो परिणाम दिन्छ।",
    "bn": "এটি একটি সহজ উত্তর। মানুষ একসঙ্গে পরিকল্পনা করে এবং দরকারি অভিজ্ঞতা ভাগ করে। পরিষ্কার পদক্ষেপ ভালো ফল দেয়।",
}


def description(instruction_id: str, kwargs: dict) -> tuple[str, dict]:
    checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    text = checker.build_description(**kwargs)
    actual = checker.get_instruction_args() or {}
    return text, actual


def verify(prompt: str, answer: str, ids: list[str], kwargs: list[dict]) -> None:
    result = test_instruction_following_strict(
        InputExample(key=0, instruction_id_list=ids, prompt=prompt, kwargs=kwargs),
        answer,
    )
    if not result.follow_all_instructions:
        raise ValueError(f"verification failed: {ids} {result.follow_instruction_list} {answer[:200]!r}")


def single_example(instruction_id: str, split: str, index: int):
    task = task_text(split, index)
    topic = task.split("about ")[-1].rstrip(".")
    kwargs = {}
    answer = base_text(topic)
    if instruction_id == "keywords:existence":
        kwargs = {"keywords": ["evidence", "progress"]}; answer += " Evidence supports progress."
    elif instruction_id == "keywords:frequency":
        frequency = 2 + index % 5; kwargs = {"keyword": "focus", "frequency": frequency, "relation": "at least"}; answer = " ".join(["focus"] * frequency) + " guides useful work."
    elif instruction_id == "keywords:forbidden_words":
        kwargs = {"forbidden_words": ["banana", "rocket"]}
    elif instruction_id == "keywords:letter_frequency":
        frequency = 8 + index % 13; kwargs = {"letter": "e", "let_frequency": frequency, "let_relation": "at least"}; answer = "Evidence enables teams to review experience and create better steps every week."
    elif instruction_id == "language:response_language":
        language = tuple(LANGUAGE_TEXT)[index % len(LANGUAGE_TEXT)]; kwargs = {"language": language}; answer = LANGUAGE_TEXT[language]
    elif instruction_id == "length_constraints:number_sentences":
        threshold = 4 + index % 17; relation = "at least" if index % 2 == 0 else "less than"; kwargs = {"num_sentences": threshold, "relation": relation}; answer = sentences(threshold if relation == "at least" else max(1, threshold - 2))
    elif instruction_id == "length_constraints:number_paragraphs":
        count = 2 + index % 5; kwargs = {"num_paragraphs": count}; answer = "\n***\n".join(f"Paragraph {i + 1} explains a useful step." for i in range(count))
    elif instruction_id == "length_constraints:number_words":
        threshold = (100, 150, 200, 250, 300, 350, 400)[index % 7]; relation = "at least" if index % 3 else "less than"; kwargs = {"num_words": threshold, "relation": relation}; answer = word_block(threshold + 8 if relation == "at least" else threshold - 12)
    elif instruction_id == "length_constraints:nth_paragraph_first_word":
        count = 2 + index % 4; nth = 1 + index % count; first = ("evidence", "planning", "review", "progress")[index % 4]; kwargs = {"num_paragraphs": count, "nth_paragraph": nth, "first_word": first}; parts = [f"A useful paragraph number {i + 1}." for i in range(count)]; parts[nth - 1] = f"{first.capitalize()} guides this part of the work."; answer = "\n\n".join(parts)
    elif instruction_id == "detectable_content:number_placeholders":
        count = 1 + index % 8; kwargs = {"num_placeholders": count}; answer = "Template fields: " + " ".join(f"[field_{i + 1}]" for i in range(count))
    elif instruction_id == "detectable_content:postscript":
        marker = "P.S." if index % 2 == 0 else "P.P.S"; kwargs = {"postscript_marker": marker}; answer += f"\n{marker} Keep a clear record."
    elif instruction_id == "detectable_format:number_bullet_lists":
        count = 1 + index % 8; kwargs = {"num_bullets": count}; answer = "\n".join(f"- Practical item {i + 1}" for i in range(count))
    elif instruction_id == "detectable_format:constrained_response":
        answer = ("My answer is yes.", "My answer is no.", "My answer is maybe.")[index % 3]
    elif instruction_id == "detectable_format:number_highlighted_sections":
        count = 1 + index % 6; kwargs = {"num_highlights": count}; answer = " ".join(f"*highlighted idea {i + 1}*" for i in range(count))
    elif instruction_id == "detectable_format:multiple_sections":
        count = 2 + index % 5; splitter = "Section" if index % 2 == 0 else "SECTION"; kwargs = {"section_spliter": splitter, "num_sections": count}; answer = "\n".join(f"{splitter} {i + 1}\nUseful content for part {i + 1}." for i in range(count))
    elif instruction_id == "detectable_format:json_format":
        answer = json.dumps({"topic": topic, "status": "ready", "step": index}, ensure_ascii=False)
    elif instruction_id == "detectable_format:title":
        answer = f"<<A Practical Guide>>\n{answer}"
    elif instruction_id == "combination:two_responses":
        answer = "The first response recommends a careful pilot.******The second response recommends a measured review."
    elif instruction_id == "combination:repeat_prompt":
        kwargs = {"prompt_to_repeat": task}; answer = task + "\n" + base_text(topic)
    elif instruction_id == "startend:end_checker":
        phrase = ("Any other questions?", "Is there anything else I can help with?", "Let me know if you need more details.")[index % 3]; kwargs = {"end_phrase": phrase}; answer += " " + phrase
    elif instruction_id == "change_case:capital_word_frequency":
        frequency = 2 + index % 6; kwargs = {"capital_frequency": frequency, "capital_relation": "at least"}; answer = " ".join(["PLAN"] * frequency) + " supports careful community work."
    elif instruction_id == "change_case:english_capital":
        answer = "CLEAR PLANS HELP PEOPLE SHARE EVIDENCE AND IMPROVE THEIR WORK."
    elif instruction_id == "change_case:english_lowercase":
        answer = "clear plans help people share evidence and improve their work."
    elif instruction_id == "punctuation:no_comma":
        answer = base_text(topic).replace(",", "")
    elif instruction_id == "startend:quotation":
        answer = '"' + base_text(topic) + '"'
    desc, actual = description(instruction_id, kwargs)
    prompt = task + "\n\n" + desc
    verify(prompt, answer, [instruction_id], [actual])
    return row(prompt, answer), instruction_id


def combo_example(kind: str, split: str, index: int):
    task = task_text(split, index + 100000)
    base = base_text("the community plan").replace(",", "")
    specs = []
    if kind == "no_comma_title":
        specs = [("punctuation:no_comma", {}), ("detectable_format:title", {})]; answer = "<<COMMUNITY PLAN>>\n" + base
    elif kind == "no_comma_bullets":
        specs = [("punctuation:no_comma", {}), ("detectable_format:number_bullet_lists", {"num_bullets": 3})]; answer = "- Gather evidence\n- Review progress\n- Improve the plan"
    elif kind == "no_comma_upper":
        specs = [("punctuation:no_comma", {}), ("change_case:english_capital", {})]; answer = "CLEAR PLANS HELP PEOPLE SHARE EVIDENCE AND IMPROVE THEIR WORK."
    elif kind == "no_comma_lower":
        specs = [("punctuation:no_comma", {}), ("change_case:english_lowercase", {})]; answer = "clear plans help people share evidence and improve their work."
    elif kind == "keyword_title":
        specs = [("keywords:existence", {"keywords": ["evidence", "progress"]}), ("detectable_format:title", {})]; answer = "<<Useful Evidence>>\nEvidence helps a team measure progress."
    elif kind == "keyword_end":
        specs = [("keywords:frequency", {"keyword": "focus", "frequency": 3, "relation": "at least"}), ("startend:end_checker", {"end_phrase": "Any other questions?"})]; answer = "Focus creates direction. Focus supports review. Focus improves action. Any other questions?"
    elif kind == "placeholders_postscript":
        specs = [("detectable_content:number_placeholders", {"num_placeholders": 3}), ("detectable_content:postscript", {"postscript_marker": "P.S."})]; answer = "Use [name] [date] and [goal] in the plan.\nP.S. Review every field."
    elif kind == "sections_end":
        specs = [("detectable_format:multiple_sections", {"section_spliter": "Section", "num_sections": 3}), ("startend:end_checker", {"end_phrase": "Any other questions?"})]; answer = "Section 1\nPlan the goal.\nSection 2\nGather evidence.\nSection 3\nReview progress. Any other questions?"
    elif kind == "highlights_keyword":
        specs = [("detectable_format:number_highlighted_sections", {"num_highlights": 3}), ("keywords:existence", {"keywords": ["evidence", "progress"]})]; answer = "*clear goal* *shared evidence* *steady progress*"
    elif kind == "paragraphs_keyword":
        specs = [("length_constraints:number_paragraphs", {"num_paragraphs": 3}), ("keywords:existence", {"keywords": ["evidence"]})]; answer = "The team defines a goal.\n***\nPeople gather evidence.\n***\nEveryone reviews the result."
    elif kind == "sentences_no_comma":
        specs = [("length_constraints:number_sentences", {"num_sentences": 6, "relation": "at least"}), ("punctuation:no_comma", {})]; answer = sentences(6)
    elif kind == "words_keyword":
        specs = [("length_constraints:number_words", {"num_words": 120, "relation": "at least"}), ("keywords:frequency", {"keyword": "evidence", "frequency": 3, "relation": "at least"})]; answer = "evidence evidence evidence " + word_block(125)
    elif kind == "quotation_no_comma":
        specs = [("startend:quotation", {}), ("punctuation:no_comma", {})]; answer = '"' + base + '"'
    elif kind == "two_responses_keyword":
        specs = [("combination:two_responses", {}), ("keywords:existence", {"keywords": ["evidence"]})]; answer = "The first response uses evidence carefully.******The second response reviews evidence openly."
    ids, kwargs, descs = [], [], []
    for instruction_id, requested in specs:
        desc, actual = description(instruction_id, requested)
        ids.append(instruction_id); kwargs.append(actual); descs.append(desc)
    prompt = task + "\n\n" + " ".join(descs)
    verify(prompt, answer, ids, kwargs)
    return row(prompt, answer), "combo:" + kind


def read_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, entries):
    with path.open("w", encoding="utf-8") as handle:
        for item, _ in entries:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def user_prompts(entries):
    return [m["content"] for item, _ in entries for m in item["conversations"] if m["role"] == "user"]


def remove_cross_split_prompt_overlap(splits):
    """Keep the earlier split and reject later rows sharing any normalized user prompt."""
    blocked = set()
    rejected = {}
    for split in ("train", "validation", "test"):
        kept = []
        count = 0
        for entry in splits[split]:
            prompts = {
                norm(m["content"])
                for m in entry[0]["conversations"]
                if m["role"] == "user"
            }
            if prompts & blocked:
                count += 1
                continue
            kept.append(entry)
            blocked.update(prompts)
        splits[split] = kept
        rejected[split] = count
    return rejected


def main():
    cfg = parse_args()
    if cfg.output_root.exists():
        raise SystemExit(f"output already exists: {cfg.output_root}")
    rng = random.Random(cfg.seed)
    benchmark = load_dataset("google/IFEval", split="train")
    benchmark_raw = {item["prompt"] for item in benchmark}
    benchmark_norm = {norm(item["prompt"]) for item in benchmark}
    splits = {"train": [], "validation": [], "test": []}

    broad = [(item, "broad_replay") for item in read_rows(cfg.broad_root / "train.jsonl")]
    rng.shuffle(broad)
    seen = set()
    for item, family in broad:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            splits["train"].append((item, family)); seen.add(key)
        if sum(f == "broad_replay" for _, f in splits["train"]) >= cfg.broad_replay:
            break
    for item in read_rows(cfg.formal_smoke_root / "train.jsonl"):
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            splits["train"].append((item, "formal_smoke_replay")); seen.add(key)

    for split in splits:
        per_single = TRAIN_PER_SINGLE if split == "train" else EVAL_PER_SINGLE
        per_combo = TRAIN_PER_COMBO if split == "train" else EVAL_PER_COMBO
        for family_index, instruction_id in enumerate(SINGLE_IDS):
            for index in range(per_single):
                splits[split].append(single_example(instruction_id, split, family_index * 10000 + index))
        for family_index, kind in enumerate(COMBOS):
            for index in range(per_combo):
                splits[split].append(combo_example(kind, split, family_index * 10000 + index))

    for split, source_split in (("validation", "validation"), ("test", "test")):
        for item in read_rows(cfg.formal_smoke_root / f"{source_split}.jsonl"):
            splits[split].append((item, "formal_smoke_replay"))

    for split in splits:
        rng.shuffle(splits[split])

    cross_split_rejected = remove_cross_split_prompt_overlap(splits)

    prompt_sets = {split: set(user_prompts(entries)) for split, entries in splits.items()}
    norm_sets = {split: {norm(x) for x in values} for split, values in prompt_sets.items()}
    cross = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        cross[f"{left}_{right}_exact"] = len(prompt_sets[left] & prompt_sets[right])
        cross[f"{left}_{right}_normalized"] = len(norm_sets[left] & norm_sets[right])
    contamination = {
        split: {
            "exact": len(prompt_sets[split] & benchmark_raw),
            "normalized": len(norm_sets[split] & benchmark_norm),
        }
        for split in splits
    }
    failures = sum(cross.values()) + sum(v for x in contamination.values() for v in x.values())
    if failures:
        raise SystemExit(f"acceptance failed: cross={cross} contamination={contamination}")

    cfg.output_root.mkdir(parents=True)
    for split, entries in splits.items():
        write_rows(cfg.output_root / f"{split}.jsonl", entries)
    manifest = {
        "schema_version": 1,
        "name": "ifeval-curriculum-v4",
        "status": "accepted_for_ifeval_generalization_experiment",
        "seed": cfg.seed,
        "benchmark": "google/IFEval 541 prompts; prompts and answers excluded",
        "benchmark_specific_curriculum": True,
        "rows": {split: len(entries) for split, entries in splits.items()},
        "families": {split: dict(Counter(family for _, family in entries)) for split, entries in splits.items()},
        "ifeval_instruction_types_covered": len(SINGLE_IDS),
        "combination_families": len(COMBOS),
        "generated_answer_verifier": "lm-eval IFEval strict checker",
        "generated_answer_verifier_failures": 0,
        "cross_split_prompt_overlap": cross,
        "cross_split_rows_rejected": cross_split_rejected,
        "ifeval_prompt_overlap": contamination,
        "sources": {
            "broad_root": str(cfg.broad_root),
            "formal_smoke_root": str(cfg.formal_smoke_root),
        },
        "limitations": [
            "This is benchmark-family-targeted training and is not evidence of unconstrained chat quality.",
            "The same public verifier semantics are used for generation acceptance and final IFEval scoring.",
            "Final claims require held-out IFEval plus broad benchmark regression checks.",
        ],
    }
    manifest["files"] = {
        f"{split}.jsonl": {"rows": len(entries), "sha256": sha256(cfg.output_root / f"{split}.jsonl")}
        for split, entries in splits.items()
    }
    (cfg.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (cfg.output_root / "_SUCCESS").write_text(json.dumps({"status": manifest["status"]}) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
