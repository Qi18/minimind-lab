# 正式数据脚本

本目录只保留能够参与正式数据产出、复现或独立验收的脚本。一次性采样器、旧代理数据生成器、容量探针和历史修复脚本不放在这里。

## Pretrain

目录：`scripts/data/pretrain/`

| 脚本 | 作用 | 正式边界 |
|---|---|---|
| `build_pretrain_v1.py` | 按冻结配置读取上游语料，完成规范化、过滤、去重、切分和确定性选样 | 生成基础候选集；不单独签发验收结论 |
| `remix_pretrain_v1.py` | 按目标 Token 配额重混、写出 train/validation shards、manifest 与 provenance | 生成 1.28B 正式数据树 |
| `audit_pretrain_v1.py` | 独立回读全部产物，校验哈希、Token、切分、污染和来源绑定 | 仅全部门禁通过后签发 `_SUCCESS.status=accepted` |

正式配置位于 `configs/data/pretrain/`，协议位于 `docs/data/pretrain/data_protocol.md`。

## SFT

目录：`scripts/data/sft/`

目前只有原始数据阶段已经实现：

| 脚本 | 作用 | 当前状态 |
|---|---|---|
| `resolve_sources.py` | 解析冻结 source revision、许可证和物化计划 | 可用于原始数据准备 |
| `materialize_raw.py` | 顺序下载或复用原始源，写入对象级 `.tmp`/`.done` 证据 | 只产出 raw cache，不产出可训练 SFT |

以下正式组件尚未实现，因此当前 SFT 数据必须视为 `trainable: false`：

- `build_sft_v1.py`：canonical schema、完整轮次/Tool gate、Token 统计、去重分组、固定切分、污染过滤、配额选择及 sidecar 写出；
- `audit_sft_v1.py`：独立全量回读、复算和签发 accepted marker。

在这两个组件完成并通过 `configs/data/sft/acceptance_v1.yaml` 前，不得创建 `_SUCCESS`，也不得用 raw cache 启动正式 SFT。

## 历史实验工具

SFT 容量分析、格式投影和旧 proxy 构建脚本保存在：

`experiments/00-preparation/D05-sft-capacity-v2-20260901/tools/`

它们是方案设计和容量证据，不是正式数据入口。历史 D01-D05 报告可以保留其原始命令和输出，但新的正式流程只能引用本目录中的脚本。

## 共同约束

- 数据、缓存、权重和完整日志不进入 Git；
- 配置、脚本、manifest、聚合审计和报告进入 Git；
- Token 预算按 shifted assistant loss targets 或 Pretrain 有效 loss targets 计算，不按 `rows × max_seq_len` 估算；
- 来源 revision、许可证、原始文件行数/字节/SHA-256 必须可追溯；
- builder 负责产出，auditor 独立验收，二者不得混成一个自证步骤；
- 未通过阶段门的目录和 `.done` 文件不等于正式数据集。
