# MiniMind Lab

MiniMind Lab 是一个围绕 64M 级小语言模型展开的可复现实验项目，覆盖数据与 Tokenizer、模型结构、预训练、SFT/LoRA、DPO、GRPO/CISPO、Agentic RL、统一评测和推理服务。

本仓库保存实验配置、结果、源码阅读笔记和报告；模型源码通过 Git Submodule 固定到 [`minimind/`](minimind/) 的确定 commit。

## 项目目标

- 在 NVIDIA L20 环境完成 MiniMind 64M Dense 的端到端训练闭环。
- 用统一协议比较 Pretrain、SFT、LoRA、DPO、GRPO/CISPO 与 Agentic RL。
- 将每项结论绑定到源码 commit、实验配置、SwanLab run 和评测结果。
- 形成可用于博客与简历的工程化报告，而不是只保留训练日志。

## 仓库导航

- [训练与源码学习规划](docs/training_plan.md)
- [仓库管理方式](docs/repository-management.md)
- [统一评测协议](docs/evaluation_protocol.md)
- [最终报告模板](docs/final_report.md)
- [实验登记规范](experiments/README.md)
- [源码阅读索引](docs/source_reading/README.md)

## 实验主线

```text
Data / Tokenizer
        ↓
Model Architecture
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
| Pretrain mini | Random | 待填写 | 待填写 | 待评测 | - | 待填写 | 待填写 |
| Pretrain full | Random | 待填写 | 待填写 | 待评测 | - | 待填写 | 待填写 |
| Full SFT | Pretrain | 待填写 | 待填写 | 待评测 | 待评测 | 待填写 | 待填写 |
| LoRA | Full SFT | 待填写 | 待填写 | 待评测 | 待评测 | 待填写 | 待填写 |
| DPO | Full SFT | 待填写 | 待填写 | 待评测 | 待评测 | 待填写 | 待填写 |
| GRPO / CISPO | Full SFT | 待填写 | 待填写 | 待评测 | 待评测 | 待填写 | 待填写 |
| Agentic RL | Full SFT | 待填写 | 待填写 | Tool Success 待评测 | 待评测 | 待填写 | 待填写 |

表格只填写已经完成并能追溯到实验目录的结果。

## 克隆

```bash
git clone --recurse-submodules git@github.com:Qi18/minimind-lab.git
cd minimind-lab
```

已经克隆但没有初始化源码时：

```bash
git submodule update --init --recursive
```

## 产物边界

- GitHub：配置、命令、指标、评测结果、报告和外部链接。
- SwanLab：训练曲线、系统指标、样本输出与实验对比。
- Hugging Face：最终选中的少量模型权重与 Model Card。
- L20：数据集、活动 checkpoint、优化器状态、缓存和完整原始日志。

未经评测或无法复现的数字不进入 README 和简历。
