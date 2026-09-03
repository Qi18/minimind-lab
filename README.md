# MiniMind Lab

MiniMind Lab 是一个围绕 64M 级小语言模型展开的可复现实验项目，覆盖数据与 Tokenizer、模型结构、预训练、SFT/LoRA、DPO、GRPO/CISPO、Agentic RL、统一评测和推理服务。

本仓库保存实验配置、结果、源码阅读笔记和报告；模型源码通过 Git Subtree 导入 [`minimind/`](minimind/)，当前基线固定到上游 commit `393e387`。

## 项目目标

- 在 NVIDIA L20 环境完成 MiniMind 64M Dense 的端到端训练闭环。
- 用统一协议比较 Pretrain、SFT、LoRA、DPO、GRPO/CISPO 与 Agentic RL。
- 将每项结论绑定到源码 commit、实验配置、SwanLab run 和评测结果。
- 形成可用于博客与简历的工程化报告，而不是只保留训练日志。

## 仓库导航

- [统一实验与学习计划](docs/experiment_plan.md)
- [阶段报告索引](docs/phases/README.md)
- [仓库管理方式](docs/repository-management.md)
- [统一评测协议](docs/evaluation_protocol.md)
- [数据工程文档索引](docs/data/README.md)
  - [Pretrain v1 数据协议](docs/data/pretrain/data_protocol.md)
  - [SFT v1 数据协议](docs/data/sft/data_protocol.md)
- [最终报告模板](docs/final_report.md)
- [训练前准备实验](experiments/00-preparation/README.md)
- [实验登记规范](experiments/README.md)
- [源码阅读索引](docs/source_reading/README.md)
- [MiniMind 上游来源与同步](docs/upstream-minimind.md)

## 实验主线

```text
Preparation
   ├─ Environment / L20 Baseline
   ├─ Data / Tokenizer Audit
   └─ Model / DDP / Resume Probe
                  ↓
Pretrain ───────────────→ Base Evaluation
        ↓
Full SFT
   ├────┼───────────┬─────────────┐
   ↓    ↓           ↓             ↓
 LoRA  DPO     GRPO / CISPO   Agentic RL
   └────┴───────────┴─────────────┘
                  ↓
            Unified Evaluation
                  ↓
       Service / Report / Resume
```

DPO、GRPO/CISPO 和 Agentic RL 默认从同一个 Full SFT 基线分支，避免把多个阶段串联后无法归因。

## 结果总表

| 阶段 | 初始化权重 | 数据 | 训练成本 | 目标指标 | 通用能力回归 | SwanLab | 权重 |
|---|---|---|---:|---:|---:|---|---|
| Pretrain mini（P01） | Random | `pretrain_t2t_mini` 1,270,238 行 | 45.83 min / 6.11 GPU-hours | 七项宏平均 31.44 | 无自身基线；官方口径参考 +0.66 pp | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/nfax3tyg0j217j1cz8y0b) | L20 保留，SHA `71efd40d` |
| Pretrain full（P02） | Random | `pretrain_t2t` 8,468,827 行 | 285.78 min / 38.10 GPU-hours | 七项宏平均 30.91 | 相对 P01 -0.54 pp；官方口径参考 +0.12 pp | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/3i1muwq039fpfv89fq4ru) | L20 保留，SHA `7065a461` |
| Pretrain v1（P03） | Random | `pretrain-v1-1b28/final-remix-v1` 2,313,483 行 / 1.280B targets | 46.37 min / 6.18 GPU-hours | 七项宏平均 31.52；共享 validation NLL 2.60432 | 相对 P02 +0.62 pp，相对 P01 +0.08 pp | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/qdpjh47fjt98184oos4bl) | L20 保留，SHA `0cfb7fc8` |
| Full SFT mini（S01R1，基于 P01） | P01 64M | `sft_t2t_mini` 905,718 行 | 35.27 min / 4.70 GPU-hours | val loss 1.7024；Chat 10%；Tool E2E 37.5% | 七项宏平均 32.04，较 P01 +0.60 pp；发布验收失败 | [run](https://swanlab.cn/@richliu0153/MiniMind-Lab/runs/s2zj3jb9n8uh9v7raemx5) | L20 保留 best，SHA `239d48e4` |
| LoRA | Full SFT | 待填写 | 待填写 | 待评测 | 待评测 | 待填写 | 待填写 |
| DPO | Full SFT | 待填写 | 待填写 | 待评测 | 待评测 | 待填写 | 待填写 |
| GRPO / CISPO | Full SFT | 待填写 | 待填写 | 待评测 | 待评测 | 待填写 | 待填写 |
| Agentic RL | Full SFT | 待填写 | 待填写 | Tool Success 待评测 | 待评测 | 待填写 | 待填写 |

表格只填写已经完成并能追溯到实验目录的结果。

当前预训练结论：full 数据规模是 mini 的 6.67 倍，P02 末段平均 training loss 更低，但七项宏平均低于 P01，五个固定 Base 续写也全部立即 EOS。后续 SFT 将固定数据与超参数，分别从 P01/P02 初始化做对照，避免只凭 training loss 选择 Base checkpoint。

## 克隆

```bash
git clone git@github.com:Qi18/minimind-lab.git
cd minimind-lab
```

`minimind/` 已作为普通源码目录进入仓库，不需要初始化 Submodule。

## 产物边界

- GitHub：配置、命令、指标、评测结果、报告和外部链接。
- SwanLab：训练曲线、系统指标、样本输出与实验对比。
- Hugging Face：最终选中的少量模型权重与 Model Card。
- L20：数据集、活动 checkpoint、优化器状态、缓存和完整原始日志。

未经评测或无法复现的数字不进入 README 和简历。
