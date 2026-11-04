# 225：SP 固定前缀 logprob 对照

同一首尾八层 BF16 权重，单机 TP4/DP2/EP8。4 条提示词分别走 completions 和 chat；用第一次 SP-off 返回的输入 ID（chat 含模板）及 8 个生成 ID 固定每条序列。各轮都用 completions 的 `prompt_logprobs=1` 对这 8 个 ID 打分，共 64 个位置。逐项核对响应回传的 prompt ID 和目标 ID；没有在生成时限制候选词或重新归一化概率。

| 对照 | 固定 token 平均绝对差 | 固定 token 最大绝对差 | 自由生成文本与 token 完全相同 |
|---|---:|---:|---:|
| SP-off 同服务重复 | 0.00000000 | 0.00000000 | 8/8 |
| SP-off 重启 | 0.00000000 | 0.00000000 | 8/8 |
| SP-off → SP-on | 0.07899429 | 0.50581288 | 3/8 |

SP-on 的 8 个实际 worker 均启用了 MoE 层 3–6 的序列并行。与 SP-off 相比，64 个固定位置中有 64 个分数不同。最大差出现在 `/v1/chat/completions` 的 `你好`，第 1 个目标 token（ID 48287）：SP-off 为 -3.51588941，SP-on 为 -4.02170229。

自由生成另外对照：只比较历史 token 和当前 token 均相同的位置，最大 logprob 绝对差为 0.41037846。整段固定序列的 prompt 打分和逐步 decode 使用不同的执行路径；两类指标各自对照，不直接相减。

结论限于本次样本：基线同服务及重启均未观测到漂移；切换到受测 SP 配置后，出现了可以在相同前缀、相同 token 下测出的差异。原表中的 0.62598705 是自由生成按位置比较的值，其中包含序列分叉后的不同 token，不能用作精度阈值。这次比较的是整套 SP 配置（含通信后端、FlashComm 和 SP pass）；未隔离单个算子、未验证完整模型任务精度，也未设业务验收容差。

原始材料：[固定 token 序列](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/fixed-inputs.json) · [SP-off 首轮](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/off-a.json) · [同服务重复](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/off-b.json) · [重启后 SP-off](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/off-cold.json) · [SP-on 响应与拓扑](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/on.json) · [逐组统计](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/comparison.json) · [实测探针](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/probe.py) · [统计脚本](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/analyze.py) · [各轮实际启动命令](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/launch.txt)。
