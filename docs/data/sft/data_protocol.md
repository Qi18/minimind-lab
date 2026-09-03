# SFT v1 数据设计与验收协议

更新时间：2026-09-02
当前状态：`design_pending_implementation`
训练资格：`trainable: false`

## 1. 为什么重新设计

旧方案直接把目标写成 160M assistant tokens，但当时只有 proxy 流水线、raw source 物化和容量分析，没有正式 builder、独立 auditor，也没有证据证明 160M 的混合比例和质量最优。因此不再把“凑满 160M”作为完成条件。

新方案先验证数据链路和训练收益，再逐级扩大：

| 阶段 | train assistant loss-target tokens | 目的 | 进入下一阶段的门 |
|---|---:|---|---|
| smoke | 1M | 验证 schema、loader、Tool 完整性和 auditor | 全部门禁通过，可重复构建 |
| pilot | 8M | 验证 mix 容量、训练信号和固定评测方向 | 相比 Pretrain 基线有目标能力增益且无异常退化 |
| formal v1 | 32M | 首个可复现、可对比的正式 SFT | 独立 audit accepted，完成固定评测 |
| optional scale | 64M | 只验证规模扩展是否继续有效 | 固定评测提升且 Base benchmark 无不可接受回归 |

64M 不是自动执行项；若 32M 已进入收益平台或出现遗忘，就停止扩量。160M 仅保留为旧目录名和历史方案，不再是当前预算。

## 2. 正式入口与缺口

- 来源配置：`configs/data/sft/sources_v1.yaml`
- 构建设计：`configs/data/sft/build_v1.yaml`
- 验收门禁：`configs/data/sft/acceptance_v1.yaml`
- 污染配置：`configs/data/sft/contamination_v1.yaml`
- raw resolver：`scripts/data/sft/resolve_sources.py`
- raw materializer：`scripts/data/sft/materialize_raw.py`
- formal builder：尚未实现
- independent auditor：尚未实现

只要后两项为空，就不得创建 `_SUCCESS`，也不得把 raw cache 或旧 proxy 数据用于正式 SFT。

## 3. 数据构建流水线

```text
freeze source revisions + licenses
  → materialize/reuse raw objects with .tmp/.done evidence
  → normalize into canonical conversations
  → reject incomplete turns / partial tool exchanges / zero targets
  → compute shifted assistant loss-target tokens
  → group exact + near duplicates by origin
  → deterministic train/validation/test split
  → fixed-eval contamination filtering
  → bucket quota allocation without origin reuse
  → payload + provenance sidecars + manifest
  → independent full audit
  → _SUCCESS.status=accepted
```

训练器只读取已经冻结的 split 文件，不能在运行时再次随机切分。

## 4. Raw cache 边界

当前已物化原始源保留在：

`/data/datasets/minimind-lab/data-v1/sft-v1-160m/`

目录名来自旧目标，现仅表示已验证的 raw cache，避免重复下载和移动大文件；它不表示已有 160M 可训练 tokens。新正式工作区和输出分别使用：

- work：`/data/datasets/minimind-lab/data-v1/sft-v1-32m/work/`
- final：`/data/datasets/minimind-lab/data-v1/sft-v1-32m/final/`

对象级 `.done` 只证明原始文件物化完成，不能替代 canonical build、污染检查或独立验收。

## 5. Canonical 训练 schema

每一行顶层只允许：

```json
{"conversations": [{"role": "user", "content": "...", "reasoning_content": null, "tools": null, "tool_calls": null}]}
```

- 顶层 key 必须精确为 `conversations`；
- message key 必须精确为 `role/content/reasoning_content/tools/tool_calls`；
- 可选值显式写 `null`，避免来源 schema 漂移；
- role 仅允许 `system/user/assistant/tool`；
- 来源、revision、license、origin、transform、bucket 和 split 放 sidecar，不写进训练 payload；
- Tool call 与 Tool response 必须作为完整交换保留；
- 截断只能保留完整 turn，不能截断在消息或 Tool 交换中间。

## 6. 核心验收门禁

1. invalid JSON/conversation、zero shifted target、非 assistant 监督 Token 均为 0；
2. incomplete assistant turn、partial Tool exchange、assistant EOS 漏监督均为 0；
3. exact/near duplicate 先组成 origin component，再做确定性 split；
4. 同一 origin 不得跨 bucket、split 或 variant family 重复使用；
5. train/validation/test 的 exact 与 near overlap 均为 0；
6. 固定七项、GSM8K、MBPP 的 exact/containment/near overlap 均为 0；
7. 来源 revision、license、行数、字节和 SHA-256 可追溯；
8. builder 产出的 manifest 只能是 pending，只有独立 auditor 可以签发 accepted marker。

完整阈值以 `configs/data/sft/acceptance_v1.yaml` 为准。`harness_commit` 与 `prompt_builder_sha256` 仍为空，是当前 fatal blocker。

## 7. 旧实验如何保留

- D01 的 5,907 条 proxy 数据只证明 schema、split、label mask 和污染流水线可运行，不是正式 SFT；
- D02 保存 raw source 解析与物化证据；
- D03-D05 保存容量、来源扩展、格式投影和 adapter 设计证据；
- 旧 profiler/生成脚本统一保存在 `experiments/00-preparation/D05-sft-capacity-v2-20260901/tools/`，不得作为正式入口；
- 历史 JSON、日志和命令保持原样，避免事后改写证据。

## 8. 下一步

1. 实现 `build_sft_v1.py`，先只产出 1M smoke 和 sidecars；
2. 独立实现 `audit_sft_v1.py`，不得复用 builder 的最终统计结果；
3. 冻结 harness commit 和 prompt builder SHA，跑全量污染门禁；
4. smoke accepted 后构建 8M pilot，完成最小 SFT 与固定评测；
5. 依据容量、Chat/Tool/格式/数学收益和七项回归冻结 32M mix；
6. 只有 32M 验收和训练评测完成后，才决定是否做 64M。
