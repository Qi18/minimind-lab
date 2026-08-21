# 仓库管理方式

## 1. 仓库职责

`minimind-lab` 是实验与学习总仓库，负责保存：

- 训练和评测配置；
- L20 启动、同步、评测与清理脚本；
- 每次实验的元数据、指标和报告；
- 源码阅读笔记、博客草稿和最终项目报告。

`minimind/` 是通过 Git Subtree 导入的普通源码目录，负责保存：

- MiniMind 模型、Dataset 和 Trainer 源码；
- 通用 SwanLab 埋点；
- 对训练、奖励、Rollout 或推理逻辑的源码修改。

实验目录不复制 MiniMind 源码，而是同时记录 Lab commit 和 MiniMind 上游来源 commit。

## 2. 权威工作区

L20 的 `/data/projects/minimind-lab` 是训练期间的权威 checkout。代码修改、验证、commit 和 push 都在 L20 完成，不使用本地仓库中转。

开始正式实验前必须记录：

```text
lab_commit
minimind_source_commit
minimind_dirty
hardware
dtype
dataset_version
seed
command
swanlab_url
```

如果 Lab 工作区不干净，正式实验应停止；确有必要时保存 `source.patch` 并在报告中明确说明。

## 3. 分支策略

- `main`：只保存已经验证、可公开复现的实验资产。
- `feature/<name>`：训练逻辑、统一评测、奖励函数等代码能力改造。
- 同一份代码下仅超参数不同的运行，不创建新分支，用实验 ID 和 SwanLab run 区分。

建议实验 ID：

```text
<stage>-<model>-<dataset>-<variant>-<date>
```

示例：

```text
pretrain-dense64m-mini-baseline-20260821
agent-dense64m-agentrl-cispo-20260821
```

## 4. 实验目录契约

每个正式实验目录至少包含：

```text
config.json
command.sh
run.json
metrics.csv
eval.json
report.md
checkpoint-manifest.txt
swanlab-url.txt
```

`run.json` 应包含：

```json
{
  "experiment_id": "agent-dense64m-agentrl-cispo-20260821",
  "stage": "agentic_rl",
  "lab_commit": "<sha>",
  "minimind_source_commit": "393e387e9ad99f0f04c296e4c5e7353f4444629f",
  "entrypoint": "minimind/trainer/train_agent.py",
  "base_weight": "full_sft",
  "dataset": "agent_rl.jsonl",
  "hardware": "8x NVIDIA L20",
  "dtype": "bfloat16",
  "seed": 42,
  "status": "planned"
}
```

## 5. 结果存放位置

| 内容 | GitHub | SwanLab | Hugging Face | L20 |
|---|---:|---:|---:|---:|
| 配置与命令 | 是 | 可选 | 否 | 是 |
| 标量和曲线 | 摘要 | 是 | 否 | 是 |
| 评测 JSON/CSV | 是 | 可选 | 可选 | 是 |
| 完整日志 | 否 | 否 | 否 | 是 |
| 中间 checkpoint | 否 | 否 | 否 | 是 |
| 最终选中权重 | 链接 | 链接 | 是 | 是 |
| 数据集 | 仅 manifest | 否 | 外部来源 | 是 |

## 6. Checkpoint 策略

每个阶段最多长期保留：

- `last`：用于中断恢复；
- `best_target`：目标任务最优；
- `best_retention`：兼顾通用能力回归的候选；
- `release`：最终公开权重。

删除 checkpoint 前，先生成 `checkpoint-manifest.txt`，记录路径、step、指标、大小、保留或删除理由，以及是否已上传 Hugging Face。

## 7. 上游源码工作流

当前源码作为普通目录参与 Lab commit。来源、初始 commit 和审计结论记录在 [`upstream-minimind.md`](upstream-minimind.md)。

查看源码改动：

```bash
git status --short -- minimind
git diff -- minimind
```

更新官方源码必须在独立分支显式进行：

```bash
git switch -c feature/sync-minimind-upstream
git subtree pull \
  --prefix=minimind \
  https://github.com/jingyaogong/minimind.git \
  master --squash
```

同步后必须检查 `minimind/` 的 diff、运行 smoke test，并更新实验元数据中的 `minimind_source_commit`。正式实验开始后不得自动漂移源码版本。

## 8. 发布门槛

只有满足以下条件的实验才能进入 README：

1. 配置、命令、commit 和数据版本齐全；
2. SwanLab run 可访问；
3. 目标指标和通用能力回归都完成；
4. 报告解释收益、代价、失败与适用边界；
5. 结果可以由保存的命令重新运行。
