# 20261016-1327 — Step5 8 层功能验证

**这是减层模型功能验证，不代表完整模型精度通过。**

- 机器：A5，容器 yyy-cann92，8 × Ascend950DT。
- 权重：/data/Step-5-Preview-BF16-8layer；8 层，807 张量，16 shards，70.34 GiB，BF16。
- 配置及所有张量 shape/dtype 已对照源权重核查；未切换到 16 层。
- 允许空格、重复或乱码；仍要求 token 非空、logprob 数量匹配且无 NaN/Inf。
- 三种更新的原始结果已独立审计，24 对输出严格一致，三个验证退出码均为 0；见 reports/acceptance_audit.json。
- HCCL、IPC、Checkpoint 三种更新分别要求两个接口、全部 4 prompts 的 text/token/logprob 完全一致。
- 模型保持 FULL_DECODE_ONLY 图模式，max_tokens=8、temperature=0、seed=0。

## 特性结果

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 配置与权重 | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/checkpoint_schema.json) |
| 双接口基线 | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/baseline.json) |
| Checkpoint 重载 | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/checkpoint_reload.json) |
| 独立 level-2 sleep | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/sleep_mode.json) |
| HCCL 权重更新 | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/hccl_weight_update.json) |
| IPC 权重更新 | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/npu_ipc_weight_update.json) |
| Routing HTTP 双接口 | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_replay.json) |
| Routing 进程内 | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_inproc.json) |
| SP | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/sp.json) |
| PP | passed | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/pp.json) |
| DCP | blocked | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_dcp.json) |
| PCP | blocked | [JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_pcp.json) |

## 本次修复

1. 验证器去除纯空格和乱码的拒绝规则；保留 token/logprob 有效性及原始 text/token/logprob 完全一致检查。
2. Step5 流式更新按整轮累计参数，在 finish 校验全部参数和 QKV/indexer 融合分片；分块内不误触发完整 checkpoint 检查。
3. HCCL 接收端采用 CPU 暂存，避免整层延迟加载缓存挤占 NPU；IPC 同样采用 CPU 暂存。
4. IPC packed 接收原来未调用模型加载器，现已修复；Step5 IPC 使用分层重载恢复 checkpoint 布局，并在完成后恢复运行时布局。
5. HCCL 增加显式 communicator close；实机 v5 双接口更新前后完全一致，producer 退出码 0。
6. DP collective RPC 原接口只返回第一个 engine 的结果，验证扩展通过 CPU world group 收集全部 worker 的 schema/拓扑证据。

可审阅补丁：[step5-stream-update-fix.patch](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/step5-stream-update-fix.patch)。保留每次失败的日志和修改前文件，未覆盖为成功。

## 验证边界及差异

- 三种权重更新的严格一致要求没有放宽；服务健康或传输成功本身不算通过。
- Routing HTTP 使用 TP4/DP2/EP8；进程内验证如执行则使用 TP8/DP1，作为补充路径单独记录。
- SP 使用 TP4/DP2/EP8 + allgather_reducescatter + enable_flashcomm1；RPC 核查每个 worker 的 5 个 MoE 层实际启用 SP。
- SP 的文本/token 与 baseline 相同，但 logprob 有差异，最大绝对差约 0.05068445；只判功能通过，不作精度通过结论。
- PP 文本/token 与 baseline 相同，logprob 最大绝对差约 0.08298635；仅判功能通过。
- PP 使用 TP4/DP1/PP2（初次 TP2/DP2 因 DSA 要求每 rank 单 KV head 而失败，记录保留）；8 层分为 0–3 和 4–7，预检见 [pp-preflight.json](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/pp-preflight.json)。
- DCP：Step5 DSA 源码显式不支持 DCP>1；PCP：当前 ModelRunner V1 显式不支持 PCP>1，均未移除限制。
- Routing 进程内 4/4 组验证通过，退出码 0；HTTP 双接口 8/8 组通过。
- 最终任务进程全部退出，8 张卡均无运行进程，每卡约 91 GiB 可用；见 logs/npu-final.txt。
- IPC 成功退出时存在共享 NPU tensor 释放警告，见 ipc-verifier-v4.log；最终退出码以 ipc-verifier.exit 为证据。

## 测试与容量

- Step5 流式完整性/CPU 暂存：9 passed，覆盖遗漏参数、遗漏融合分片及下一轮状态重置。
- IPC 引擎回归：9 passed，包含 packed 权重实际加载和 Step5 分层重载合同。
- HCCL communicator close：1 passed。
- 8 个模型/引擎及回归测试文件 Ruff check：全部通过；pre-commit/hooks 已安装。未运行整个仓库测试或全量 pre-commit。
- 顶层 vLLM pytest conftest 缺少 tblib，因此 Step5 目标 CPU 测试使用 --confcutdir 隔离收集。
- [容量复核](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/capacity-resume.json)：每卡约 91 GiB 可用，主机约 2228 GiB 可用；CPU 暂存避免原 HCCL OOM 路径。
- IPC 实测传输时每卡总占用约 41 GiB；原 HCCL OOM 和后续修复日志均保留。

## 证据目录

- [README.md](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/README.md)：汇总及输出。
- reports/：逐特性结果、更新前后 text/token/logprob、schema 与生命周期状态。
- logs/launch_commands.txt：实际启动/验证命令。
- logs/：服务、producer、回归、显存快照及 routing 数组。
