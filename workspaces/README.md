# 算子开发工作区

每个新任务使用独立目录，避免修改课程快照：

```bash
python3 scripts/new_operator_task.py fused-silu
# 或
make new-task NAME=fused-silu TITLE="Fused SiLU"
```

生成后先填写 `workspaces/fused-silu/TASK.md`，再让 AI Coding 助手阅读根目录 `AGENTS.md`、`docs/AI_CONTEXT.md` 和该任务合同。

工作区中的 `results/` 用于保存 JSON、CSV、日志和 profiler 证据。大体积 trace 不宜直接提交 Git；可以保存索引、摘要和受控存储位置。
