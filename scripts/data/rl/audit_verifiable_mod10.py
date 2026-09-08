#!/usr/bin/env python3
"""Audit R03 modular-arithmetic split isolation and shortcut controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,required=True); ap.add_argument("--output",type=Path); a=ap.parse_args()
    manifest=json.loads((a.data/"manifest.json").read_text()); data={s:rows(a.data/f"{s}.jsonl") for s in ("train","validation","test")}
    failures=[]; profiles={}
    for split,part in data.items():
        expected={"train":4000,"validation":500,"test":1000}[split]
        if len(part)!=expected: failures.append(f"{split}: rows {len(part)} != {expected}")
        answers=Counter(r["answer"] for r in part); ops=Counter(r["operation"] for r in part); joint=Counter((r["answer"],r["template_id"]) for r in part)
        if len(set(answers.values()))!=1 or set(answers)!={str(i) for i in range(10)}: failures.append(f"{split}: answer imbalance")
        if len(set(ops.values()))!=1: failures.append(f"{split}: operation imbalance")
        if set(joint)!={(str(i),t) for i in range(10) for t in range(4)} or max(joint.values())-min(joint.values())>1: failures.append(f"{split}: answer-template joint imbalance")
        for r in part:
            expected=((r["a"]+r["b"]) if r["operation"]=="add" else (r["a"]-r["b"]))%10
            if int(r["answer"])!=expected: failures.append(f"{split}: incorrect {r['task_id']}")
        path=a.data/f"{split}.jsonl"
        if sha256(path)!=manifest["files"][split]["sha256"]: failures.append(f"{split}: manifest hash mismatch")
        profiles[split]={"rows":len(part),"answers":answers,"operations":ops,"answer_template_joint_min":min(joint.values()),"answer_template_joint_max":max(joint.values()),"unique_task_keys":len({r["task_key"] for r in part}),"sha256":sha256(path)}
    for left,right in (("train","validation"),("train","test"),("validation","test")):
        overlap={r["task_key"] for r in data[left]} & {r["task_key"] for r in data[right]}
        if overlap: failures.append(f"{left}-{right}: task overlap {len(overlap)}")
    warm=rows(a.data/"warmstart.jsonl")
    if Counter(r["kind"] for r in warm)!=Counter({"math":4000,"replay":1000}): failures.append("warmstart composition mismatch")
    result={"status":"passed" if not failures else "failed","dataset":str(a.data),"profiles":profiles,"cross_split_task_overlap":{f"{x}_{y}":len({r['task_key'] for r in data[x]}&{r['task_key'] for r in data[y]}) for x,y in (("train","validation"),("train","test"),("validation","test"))},"warmstart":{"rows":len(warm),"composition":Counter(r["kind"] for r in warm),"sha256":sha256(a.data/"warmstart.jsonl")},"failures":failures,"boundary":"Internal task-key isolation only; synthetic mod-10 does not measure general mathematics."}
    if a.output: a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)


if __name__=="__main__": main()
