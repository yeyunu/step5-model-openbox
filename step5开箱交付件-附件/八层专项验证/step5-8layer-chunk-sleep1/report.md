# Step5 8 层补充功能验证 — 2026-10-16 14:33

**Chunked prefill 与 Sleep mode level 1 均已用现有 8 层 BF16 权重实机跑通。**
本报告为减层模型功能验证，不代表完整模型精度通过。

## 环境与范围

- A5 / yyy-cann92，8 张 Ascend950DT。
- checkpoint：/data/Step-5-Preview-BF16-8layer；8 层，BF16，真实权重，load-format=auto。
- TP4、DP2、EP8，FULL_DECODE_ONLY，图捕获大小 [1,2,4]。
- max_model_len=2048，max_num_seqs=4，gpu_memory_utilization=0.4。
- 每项包含 4 个短输入和 2 个长输入，两种生成接口，max_tokens=8、temperature=0、seed=0。
- 允许纯空格、重复或乱码；生成 token 必须非空、logprob 数量匹配且无 NaN/Inf。
- 本次新增验证脚本与 worker 观测扩展；复用前轮已验证的模型实现和权重。

## 结果

| 项目 | 结果 | 验证证据 |
| --- | --- | --- |
| Sleep mode level 1 | passed，退出码 0 | 3 轮 × 6 输入 × 2 接口，共 36 对 text/token/logprob 完全一致 |
| Chunked prefill | passed，退出码 0 | 串行 12 次 + 混合并发 12 次请求均成功，token/logprob 有效 |
| 实际分块 | passed | 全部 8 个 worker 观测到多步 prefill；总计 8 个不同长请求，32 条 rank/request 记录 |
| 显存恢复稳定性 | passed | 休眠每卡释放约 36.8 GiB；恢复后最大增长 0.4536%，低于 5% |
| 最终资源释放 | passed | 8 张卡均无运行进程，每卡约 91 GiB 可用 |
| Ruff | passed | verify_extra.py、schema_extension.py |

## Sleep level 1

使用 /sleep?level=1 → /wake_up，三轮均从 CPU offload 恢复，过程中不进行 Checkpoint 重载。
检查休眠/唤醒状态、全部 worker 的物理显存释放及增长，并逐字段严格比较两接口所有输出。

[Sleep 原始报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/reports/sleep_level1.json)；验证日志（原件未归档）。

## Chunked prefill

实际开启 enable_chunked_prefill，max_num_batched_tokens=512；每个 worker 的运行配置均通过 RPC 核对。
串行长输入为 967、977、907、917 tokens，均超过单步预算。
例如 967 tokens 实测分为 512 + 455，907 tokens 分为 512 + 395。
调度 trace 覆盖新请求与缓存请求的 computed token 进度，全部调度步总 token 数不超过 512。
混合并发使用不同的输入前缀，避免完全复用串行测试的 prefix cache。
只添加观测包装，不改写调度输出或执行参数。

与关闭 chunk 的同输入基线相比：全部文本/token 相同，长输入 logprob 存在差异，
最大绝对差 0.071979523；短输入三项完全一致。因此本项结论是功能通过，未宣称数值精度验收通过。

[Chunk 原始报告及调度记录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/reports/chunked_prefill.json)；验证日志（原件未归档）。

## 复现与证据

- [独立结果审计](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/reports/acceptance_audit.json)
- [实际执行命令](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/logs/launch_commands.txt)
- 最终显存快照（原件未归档）
- [环境记录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/environment.json)
- run-service.sh：sleep1 / chunk 两种服务配置。
- verify_extra.py：双接口、三轮 sleep、实际分块、串行/并发检查。
- schema_extension.py：各 worker 运行配置、物理显存和调度 trace。
- logs/*-raw.json：各接口原始响应。
