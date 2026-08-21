# MiniMind 上游来源与同步

## 来源

- 官方仓库：<https://github.com/jingyaogong/minimind>
- 初始导入 commit：`393e387e9ad99f0f04c296e4c5e7353f4444629f`
- 导入日期：2026-08-21
- 导入方式：`git subtree --squash`
- 目录：`minimind/`

初始导入使用 `Qi18/minimind` 的 `master` 读取该 commit。迁移审计确认 Fork 相对官方仓库没有独有 commit；当时 Fork 只比官方 `master` 落后 3 个 README commit。因此导入内容可以视为官方历史中的固定源码快照，而不是一个包含私有代码改动的分叉。

## 为什么使用 Subtree

- `minimind/` 是普通目录，克隆 Lab 后立即可用；
- 训练代码修改、SwanLab 埋点和实验材料可以在一个仓库评审；
- 每次实验只需记录一个 Lab commit，同时保留上游来源 commit；
- 上游更新必须显式进入 feature 分支，不会在训练中自动漂移。

## 同步官方更新

先保证工作区干净并完成当前实验，再执行：

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/sync-minimind-upstream
git subtree pull \
  --prefix=minimind \
  https://github.com/jingyaogong/minimind.git \
  master --squash
```

同步后必须：

1. 阅读 `minimind/` 目录的变更；
2. 运行数据、模型前向和训练 smoke test；
3. 检查已有配置和 checkpoint 兼容性；
4. 更新本文的上游 commit；
5. 通过评审后再进入 `main`。

不要在正式实验运行期间同步上游，也不要把未验证的官方最新 commit 直接作为实验基线。
