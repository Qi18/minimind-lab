# 实验脚本

- `launch/`：调用 `minimind/trainer/` 的 L20 启动脚本；
- `sync/`：整理远端日志、SwanLab 链接和实验摘要；
- `eval/`：统一评测入口；
- `cleanup/`：按 checkpoint manifest 清理冗余产物。

脚本必须支持 dry-run 或输出最终命令，并禁止把凭据写进参数、日志和仓库。
