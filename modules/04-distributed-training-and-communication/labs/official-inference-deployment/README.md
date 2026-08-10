# 官方模块四推理部署实验（扩展材料）

上游：`https://gitee.com/nsddrezsd/lab4-1`

固定提交：`78288624113cfe6ae23025918673adda5db8f3a5`

`upstream/` 原样保存 Qwen3-4B、vLLM、FlagGems 和 `vllm-plugin-FL` 的部署手册及配图。它属于本期官方模块四 handout 中的 **inference** 分支，适合作为模型部署和算子库接入案例；它不替代 DP/TP/PP、AllReduce 或集合通信实验。

手册包含克隆仓库、安装包、下载模型和启动服务等会改变环境或产生大量数据的命令。执行前必须：

- 在独立可回收环境中核对固定提交、磁盘、显存和模型授权；
- 不覆盖平台预装的 CoreX 适配 PyTorch/Triton；
- 先确认 `/flagos/lab/models/Qwen3-4B` 是否存在，不把示例路径当作通用事实；
- 保存服务版本、启动参数、正确性请求、延迟和显存记录。

当前只完成资料归档和静态检查，没有下载模型、安装依赖或启动 vLLM。分布式训练 handout 指向的飞书 Wiki 在 2026-08-10 的未登录访问中进入登录流程，正文仍未取得。
