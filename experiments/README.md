# 实验目录规范

每个实验在对应阶段目录下建立独立目录，名称使用稳定实验 ID，不只使用日期。

```text
experiments/<stage>/<experiment-id>/
├── config.json
├── command.sh
├── run.json
├── metrics.csv
├── eval.json
├── report.md
├── checkpoint-manifest.txt
└── swanlab-url.txt
```

`registry.csv` 是全局索引。实验开始时登记为 `planned/running`，完成统一评测和报告后才能改为 `completed`。

使用模板创建实验，避免遗漏追溯文件：

```bash
scripts/launch/init_experiment.py 01-pretrain P01-dense-mini
scripts/sync/validate_experiment.py experiments/01-pretrain/P01-dense-mini
```

模板的 `command.sh` 默认拒绝执行。只有填写并 review 最终命令后才能移除占位退出逻辑。
正式训练必须通过 `scripts/launch/run_guarded.py`，并在 `run.json` 中记录直接基线、源码版本和 SwanLab run id。
权重和完整日志不进入 Git，只在 manifest 中保存 SHA-256、大小、路径和用途。
