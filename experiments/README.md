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
