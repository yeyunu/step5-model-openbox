本报告汇总 Step5 的模型结构、部署环境、功能验证、性能、精度及报错修复结果，并提供对应配置、原始报告和日志。

**已完成模型部署、TP/DP/EP、SP、PP、图模式、chunk prefill、routing replay、sleep/wake 及三种权重更新机制验证；完成文本 2K→6K、多模态 32K→4K 性能与 0.8 无标杆分析，以及完整模型 GPQA Diamond、MMMU-Pro 精度评测。** CP 在受测路径受限；各项结论以表内配置和证据所覆盖的检查为准。

## 1. 验收依据与结论总览

按 Ascend 官方 MindSpeed-Agent 的 [model-rl-case-guard v0.23.0](https://gitcode.com/Ascend/MindSpeed-Agent/blob/40cf21ed89eb8ff707287d8dcd81eecb1405198c/skills/inference-model-skills/deployment/model-rl-case-guard/SKILL.md) 整理，核对版本为 `40cf21ed89eb8ff707287d8dcd81eecb1405198c`。

功能项记录受测拓扑、检查结果和原始证据；权重更新分别执行 HCCL、NPU IPC、Checkpoint reload。双接口指 `/v1/completions` 与 `/v1/chat/completions`。减层功能、减层性能与完整模型精度分别对应各自模型配置。

| 验收项                      | 实测结果                                              | 状态与范围                            | 证据入口        |
| ------------------------ | ------------------------------------------------- | -------------------------------- | ----------- |
| 模型结构、权重与算子               | 结构图、config、张量清单及减层构造校验齐备                          | 已交付；逐层前向 shape、算子支持矩阵及 JDC 闭环待完善 | 第 2 节       |
| 环境与基础配置                  | TP4/DP2/EP8；双接口共 16 条响应、320 个 token/logprob 对通过验证 | 已完成基础功能验证                        | 第 3 节       |
| BF16 反量化                 | 使用真实 BF16 权重                                      | 转换来源及脚本未归档，反量化项适用性待确认            | 第 2 节       |
| SP | 225 首尾八层，8 个 worker 的 4 个 MoE 层均启用 SP | 双接口 8/8 非空；精度未评估 | 第 4 节 |
| PP | 225 单机 TP4/DP1/PP2，stage 分别承载 0–3、4–7 层 | 双接口 8/8 非空；精度未评估 | 第 4 节 |
| 图模式                      | `FULL_DECODE_ONLY`，capture sizes `[1,2,4]`        | 已验证所列配置                          | 第 4 节       |
| CP（DCP/PCP）              | Step5 DSA / ModelRunner V1 路径拒绝所测配置               | `blocked`                        | 第 4 节       |
| chunk prefill            | 12 次串行 + 12 次混合并发成功，观测到多步分块                       | 功能验证完成                           | 第 4 节       |
| routing replay           | HTTP 8/8，进程内 4/4，专家 shape/range 有效                | 功能验证完成                           | 第 5.1 节     |
| sleep/wake               | Level 2、Level 1 各 5 轮，各 40 对输出精确相等                | 输出恢复及观测内存检查通过；metrics 证据范围见正文    | 第 5.2、5.3 节 |
| HCCL / Checkpoint reload | 各独立执行 1 轮，各 8 对输出精确相等                             | `passed`                         | 第 5.2 节     |
| NPU IPC                  | 功能复验 2 轮，16 对轮内及 8 对跨轮输出精确相等；import-once 实机复验 8/8 | 更新与客户端退出告警复验通过；正常停服待复验           | 第 5.2～5.4 节 |
| 文本 2K→6K                 | C1/C4/C8 各 3 轮，72/72 成功                           | 性能采集与分析完成                        | 第 6.1 节     |
| 多模态 32K→4K               | 实际 32768 输入、4096 输出，已取证图像编码                       | 性能采集与分析完成                        | 第 6.2 节     |
| 0.8 无标杆性能分析              | 文本、多模态各样本八卡计算侧评分均 ≥0.8                            | 按约定分析规则通过                        | 第 6.3 节     |
| GPQA / MMMU-Pro          | GPQA 完整三轮平均 93.77%；MMMU-Pro 两轮平均 76.19%           | 完整模型精度结果已交付                      | 第 7 节       |
| 报错修复与复现材料                | 修复记录、代码补丁、验证报告及证据索引                               | 已归档，剩余项单列                        | 第 8、9 节     |

## 2. 模型结构、权重与算子

### 2.1 模型结构与资产身份

| 项目 | 已核实信息 |
| --- | --- |
| 架构 | `Step5ForConditionalGeneration`，`model_type=step5_vl` |
| 文本主干 | 原始 92 层，hidden size=4096，vocab size=128896 |
| 注意力 | 64 Q heads，4 KV groups，head dim=192；sliding/full 混合，sliding window=512 |
| 稀疏注意力 | full-attention 层启用 sparse indexer；provider groups=4，Top-K=512，region block size=8 |
| MoE | 352 experts，每 token Top-8；原配置 `moe_layer_list=3..90` |
| 视觉编码器 | Perception Encoder：47 层，width=1536，16 heads，image size=728，patch size=14 |
| 完整精度模型 | 225/226 服务路径 `/data/Step-5-Preview-BF16`；同路径 config 的 10/27 观察为 92 层、`torch_dtype=bfloat16` |
| 首尾 8 层 | 原层 `[0,1,2,3,88,89,90,91]` 重编号 0–7；MoE 新索引 `[3,4,5,6]`，末层 Dense，MTP=0 |
| 首尾 8 层权重规模 | 802 tensors，13 shards，62,537,360,896 bytes（58.242456 GiB） |
| 最大张量 | `model.layers.3.moe.down_proj.weight`，shape `[352,4096,1536]`，4,429,185,024 bytes |

[![Step5 配置级模型结构图](%E5%9B%BE%E7%89%87/Step5%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.png)](https://github.com/yeyunu/step5-model-openbox/blob/main/%E6%A8%A1%E5%9E%8B%E5%BC%80%E7%AE%B1%E5%A4%8D%E7%9B%98/%E5%9B%BE%E7%89%87/Step5%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.png)

结构图为配置级示意。证据：[源 config](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/source-config.json)、[8 层 config](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/8layer-config.json)、[schema 校验](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/checkpoint_schema.json)、[资产审计](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/audit.md)、[完整模型 config 摘录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/full-model-config-observation.txt)。

### 2.2 权重构造与内容校验

首尾 8 层权重取自现存首尾 16 层父 checkpoint 的 `[0,1,2,3,12,13,14,15]` 层，重映射层号及对应配置，保留 embedding、norm、head 和视觉权重。复制时校验父模型归档的张量 hash，写入后再次读取目标张量核验。此流程没有重新读取完整 92 层源权重。

checkpoint-native 清单 SHA256 为 `ab948cd69e31c88de2c9bc22789f3d6d00ea556801d1616a561ccbf16f33de1e`；它是名称、shape、dtype、分片等清单摘要，逐张量内容 hash 另见构造清单。

证据：[构造清单与张量 hash](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/checkpoints/head-tail8/reduction_manifest.json) · [首尾 8 层 config](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/checkpoints/head-tail8/config.json) · [构造日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/build-head-tail8.log) · [代码身份](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/code-integration.json)

225 复用已复制的 13 个分片，逐一核对全部 802 个张量的名称、shape、dtype 与内容 SHA256，全部与已验证八层成品一致；没有重新减层或改写权重。config/tokenizer 从原成品复制，按现有分片头生成对应索引。分片打包顺序不同，因此 225 的 checkpoint manifest SHA256 为 `a4ad7b5df66ad7d5afeb2e6fa9b9596e9e41dd218a7717b3726e020b9b1db7bf`，不应与 A5 的清单摘要混用。

证据：[225 复用校验](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/reuse-checkpoint-validation.json) · [225 成品清单](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/checkpoint/reduction_manifest.json) · [225 实际配置](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/checkpoint/config.json) · [225 代码与运行摘要](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/resume-summary.json)

### 2.3 算子与分析材料

文本与视觉计算的 profiling、算子性能表和计算侧评分见第 6 节；BF16 摘要适配及验证见第 8 节与[报错修复](https://github.com/yeyunu/step5-model-openbox/blob/main/%E6%A8%A1%E5%9E%8B%E5%BC%80%E7%AE%B1%E5%A4%8D%E7%9B%98/%E6%8A%A5%E9%94%99%E4%BF%AE%E5%A4%8D.md)。逐层完整前向 shape、BF16 算子支持矩阵及 JDC 闭环尚待补齐。

## 3. 环境、版本与模型打通

### 3.1 部署环境与版本

| 项目 | 配置 |
| --- | --- |
| 架构 | `Step5ForConditionalGeneration`，真实 BF16 |
| 首尾八层成品 | A5 原成品：`/home/yyy/model_openbox/reduced-recheck-10.20/checkpoints/head-tail8`；225 校验复用目录：`/home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8` |
| 选层 | 原始 92 层中的 `[0,1,2,3,88,89,90,91]`，重新编号 0–7 |
| MoE / MTP | 新层索引 MoE `[3,4,5,6]`；末层 Dense；MTP=0 |
| checkpoint 格式 | `safetensors`；`load-format=auto`，真实 checkpoint 启动 |
| 最大单张量 | `model.layers.3.moe.down_proj.weight`，4,429,185,024 bytes |
| 张量规模 | 802 tensors、13 shards、62,537,360,896 bytes（58.242456 GiB） |
| 基础拓扑 | TP4 / DP2 / EP8，`FULL_DECODE_ONLY`，max-model-len=2048 |
| 运行环境 | 基础配置在 A5；验证在 225（177.2.25.1），CANN 9.2、Step5 自定义算子；容器 `step5-recheck-225-10.20` |
| 225 拓扑 | HCCL/reload/sleep：TP4/DP1/EP4，推理卡 0–3，HCCL producer 卡 4；IPC：TP4/DP2/EP8，卡 0–7 |
| vLLM | `c21cc405e9cbf0fcd6b958ec3c458360dfdcf9fe` |
| vLLM-Ascend 版本 | A5 基底 `94edc50f4a35911732fdd91781f889f436a1c4c7`；225 基底 `80d265f13e53b2c596da31388a1eedde2bb7f94d`，叠加已归档同步 diff 与 #20403 backport |
| 225 功能验证 diff SHA256 | `96ec5ccd8b3bacd4cfca80e94af28d172186112dac8ba5338180ec11b386d7f4` |
| MindSpeed-Agent | `d32b647e556a5670ffc2522b815f1a66e22df3d9` |

SP、PP、routing、chunk 专项使用的 vLLM commit 为 `5d7f9124db0304ad177f71969f26a70316ecc632`，vLLM-Ascend 基底为 `80d265f13e53b2c596da31388a1eedde2bb7f94d`，配套补丁与环境以 [manifest](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/manifest.json) 和[修复补丁](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/step5-stream-update-fix.patch)为准。受测权重为 `/data/Step-5-Preview-BF16-8layer`，8 层 BF16；各模式拓扑见第 4、5 节。

### 3.2 容量与运行拓扑

baseline 预估每卡需求 68.2349 GiB，IPC 为 74.7349 GiB，另保留 4 GiB 安全余量；当时每卡可用约 91.34 GiB，两个容量预检均为 `passed`。这是按保守分摊假设计算的预算，不是实测峰值。baseline 的 `gpu-memory-utilization=0.4`，IPC 降为 `0.3`，为 producer 和传输 buffer 留出空间。

| 配置 | 实际参数/观测 | 验证范围 |
| --- | --- | --- |
| TP | `--tensor-parallel-size 4`；worker 报告 `tp=4` | 实机基线与 IPC |
| DP | `--data-parallel-size 2`；8 worker 覆盖两个 DP rank | 实机基线与 IPC |
| EP | `--enable-expert-parallel`；worker 报告 `ep_enabled=true`，总 worker 数 8 | 实机基线与 IPC |
| 图模式 | 首尾八层基线，`FULL_DECODE_ONLY`，capture sizes=`[1,2,4]` | 在此配置下双接口返回非空文本；配置记录不证明每个 shape 均实际回放 | [同次服务日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/new8-baseline/server.log) · [8-token 逐条响应](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/new8-baseline8-strict.json) |

证据：[baseline 容量预算](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/new8-baseline/capacity.stdout.json) · [IPC 容量预算](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/new8-ipc/capacity.stdout.json) · [baseline 服务日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/new8-baseline/server.log) · [IPC worker 拓扑与清单](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/results/ipc-round1/report.json)

### 3.3 完整模型服务配置

完整模型评测服务使用 225/226、TP8/DP2/EP、`max-model-len=153300`、`FULL_DECODE_ONLY`。模型配置和启动/评测脚本见[服务脚本摘录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/full-model-service-scripts-observation.txt)。逐轮完整 commit/环境锁定清单待补；GPQA 部分覆盖批次的 TP4/DP4 配置在第 7.3 节单列。

### 3.4 基础配置与双接口功能验证

**验收要求：**按上游基础配置要求及公共双接口验证器，检查服务响应和返回结构；保存每条 text、生成 token 及对应 logprob。文本去除空白后必须非空，token 列表必须有效且非空，logprob 数量必须与 token 一致，每个 logprob 必须是有限数值。另需审阅返回文本有无明显编码乱码；这些条件不等价于模型精度达标。

**执行配置：**首尾八层 `[0,1,2,3,88,89,90,91]`；固定 4 条 prompt，temperature=0、top_p=1、seed=0。completions 请求 `logprobs=1`；chat 请求 `logprobs=true, top_logprobs=1`。执行入口为 `strict_probe.py`，直接调用未修改的上游 `validate_generation_output`。

| max_tokens | 接口 | HTTP/输出验证 | 每条 token 数 / logprob 数 | 有限数值检查 | 状态 |
| ---: | --- | --- | --- | --- | --- |
| 8 | `/v1/completions` | 4/4 | 8 / 8 | 32/32 | `passed` |
| 8 | `/v1/chat/completions` | 4/4 | 8 / 8 | 32/32 | `passed` |
| 32 | `/v1/completions` | 4/4 | 32 / 32 | 128/128 | `passed` |
| 32 | `/v1/chat/completions` | 4/4 | 32 / 32 | 128/128 | `passed` |

共 16 条响应、320 个生成 token/logprob 对，全部通过公共验证器；结束原因均为 `length`。这里的 token/logprob 数量一致属于单条响应的数据完整性检查；权重更新前后的逐项相等另见第 5.2 节。

**文本审阅与结论边界：**归档输出未见 Unicode 替换字符，包含可解码的词片段、重复词及中英文混合；例如 ` PropagationVirVir Nonlinear Nonlinear Zad Serious Genet`。这证明双接口探针返回结构和数据有效，不证明回答语义正常或完整模型精度通过。上游 `benchmark-aisbench.md` 的标准 curl 示例（`model=auto`、prompt=`Hello`、max_tokens=16）未见原样执行记录，记为 `not_run`；现有 Python HTTP 探针配置如上，不补写为已执行标准 curl。

验证器 SHA256：`17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037`，与归档的官方仓库文件逐字节一致。

证据：[8-token 基线与逐条响应](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/new8-baseline8-strict.json) · [32-token 基线与逐条响应](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/new8-baseline-strict.json) ·

### 3.5 已接入的上游 PR 与问题修复

本项目接入了 4 个已合入的 vLLM-Ascend PR，并参考 vLLM PR #51259 将导入缓存方案适配到 NPU packed IPC 接收路径。A5 集成提交为 `94edc50f4a35911732fdd91781f889f436a1c4c7`。

| 上游 PR                                                                        | 上游合入状态                                                                           | 本项目接入方式                                             | 解决的问题                                                                                                                                                                                                                              | 本项目日志证据                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [vLLM-Ascend #16959](https://github.com/vllm-project/vllm-ascend/pull/16959) | 已合入，merge `4004d10e9b82b6bb36d7093a8e8ae19bdb97a1f6`                             | backport `7c71fa4377e5362e78a0803863b68f0c3f712dda` | `start_weight_update` / `finish_weight_update` 原先为空，IPC 更新未进入 layerwise reload，checkpoint 状态与运行时表示无法同步恢复；packed 路径的 `load_weights` 缩进在分块循环内，导致逐块重载。修复后 IPC 与 HCCL 使用相同的 START → LOAD → FINISH 生命周期。                                | 未找到本项目修复前的直接报错。[集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)：61 passed，属于组合回归证据。                                                                                                                                                                                                                                                                                           |
| [vLLM-Ascend #17154](https://github.com/vllm-project/vllm-ascend/pull/17154) | 已合入，merge `cee16e8dbe44b417fee456c5fbc52490d4f5540e`                             | backport `285d40d925760f9577909767dd98287d06adfe7b` | HCCL 关闭时只丢弃 Python 通信器引用，没有销毁底层通信器；重新初始化还会覆盖旧通信器。新增幂等 `close()`，在 shutdown/重建前同步设备并调用 `hcclCommDestroy`，同时拒绝关闭后的 collective。                                                                                                       | 未找到 native 通信器未销毁的直接报错或泄漏测量。[集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)：61 passed。                                                                                                                                                                                                                                                                                       |
| [vLLM-Ascend #18848](https://github.com/vllm-project/vllm-ascend/pull/18848) | 已合入，merge `e992aa2e60bcd1dc99440958d5bd33bc4c6833e2`                             | backport `c8ac0609114d87bcd9fd6e7acc7a827202c74572` | NPU IPC 复制了上游设备无关逻辑，存在分块期间 MTP 完整性检查状态、按错误 NPU handle/UUID 选择设备和异常状态恢复问题。改为继承上游 IPC engine 与 trainer，仅保留 NPU 初始化、handle 收集、packed 收发和同步逻辑。                                                                                         | 未找到错误 UUID/handle 或分块 MTP 检查失败的直接报错。[集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)：61 passed。                                                                                                                                                                                                                                                                               |
| [vLLM-Ascend #18944](https://github.com/vllm-project/vllm-ascend/pull/18944) | 已合入，merge `ec26d2541fcbec6b4d719de1524c45d75b26c386`                             | backport `6fea64bf9b1d5e12a891ef3c34b5cd1cff141412` | HCCL 重复实现 NCCL trainer/worker 编排，旧静态 trainer 接口与上游状态机不一致，packed wire 配置可能由 sender/receiver 分别决定，异常和重初始化路径可能未关闭通信器。改为复用上游 NCCL engine/factory，统一初始化与更新配置，并补齐参数迭代、设备和失败清理检查。                                                         | 未找到 packed 配置冲突或旧 trainer 状态机异常的直接报错。[集成回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ascend-integration-cpu.log)：61 passed。                                                                                                                                                                                                                                                                              |
| [vLLM-Ascend #20740](https://github.com/vllm-project/vllm-ascend/pull/20740) | 未合入，底层方案来自 vLLM #51259。适配文件为 `npu_ipc_engine.py` / `packed_tensor.py`            | 直接接入                                                | packed producer 每次更新只导出一次 staging buffer，但旧 consumer 每个 chunk 都重建并释放同一映射，使跨进程引用计数变负；producer 丢弃 buffer 后无法由 IPC collect 回收，RL 每步同步权重会泄漏一个 `packed_buffer_size_bytes`。修复为同一导出只导入一次，并在下一次导出、`finish_weight_update` 或 shutdown 时释放一次。 | [225 客户端](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/225-ipc-before-round1.log) L217–224、[226 客户端](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/226-ipc-before.log) L226–233：`Producer process has been terminated before all shared NPU tensors released`。[修复前 CPU 回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/ipc-import-before-regression.log) L45、91：`assert 2 == 1`，证明同一导出重复导入；告警本身不证明泄漏量。 |
| [vLLM-Ascend #20403](https://github.com/vllm-project/vllm-ascend/pull/20403) | 未合入；固定生产补丁 `c4ed2abb7cdb8243731fdc4311c7bb3ad4370f33`，仅接入专家映射的两条 reload skip 规则。 | 选择性接入                                               | 把 `expert_ids_per_ep_rank` 同时加入 `meta.SKIP_LOAD_TENSORS` 和 `meta.SKIP_TENSORS`，保留专家映射及现有恢复逻辑，避免运行时生成项破坏映射和 checkpoint 加载完成计数。                                                                                                      | [失败回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/20403-regression-before.log) L14–16：专家映射被替换为 `device=meta`。[计数诊断](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/20403-count-diagnostic.log)：修复前 32/384、缓存 2 个张量；修复后 32/32、缓存 0 个。[修复后回归](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/20403-regression-after.log)：5 passed。                                                                         |

日志检索范围、摘录、原始路径及 SHA256 见[证据索引](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/pr-error-evidence/README.md)；未找到直接报错的项以回归证据标注。

本地验证覆盖：Ascend 权重传输、对照配置与运行时张量检查 61 项通过；#51259 NPU 适配的 CPU 源码测试 53 项通过；225 真实八层权重完成 TP4/DP2/EP8 packed IPC 更新，两接口 8/8 对更新前后输出精确相等，客户端退出阶段未复现共享 NPU 张量未释放告警。HCCL、IPC、reload 与 sleep/wake 的完整实机结果见第 5 节。

证据：[A5 代码与迁移记录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/code-migration/a5-migration-record.txt)、[上游 PR 状态查询记录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/code-migration/official-pr-status.json)、[早期四 PR 查询](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/code-migration/upstream-pr-status.json)、[#51259 适配与验证](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/README.md)。

证据：[上游固定补丁](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/upstream/c4ed2abb-production-commit.patch) · [实际 backport diff](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/pr20403-backport.diff) · [回归日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/pr20403-regression.log)

## 4. 基本功能验证

**双接口生成结果：**A5 首尾八层 BF16 基线复验中，两个接口各测 4 条 prompt，分别生成 8 和 32 个 token。共 16/16 条响应去除空白后仍有文本；每条响应的 token 与有限 logprob 数量一致。例如 8-token completions 对 `The future of AI is` 返回 ` PropagationVirVir Nonlinear Nonlinear Zad Serious Genet`。这些结果证明接口返回了生成文本，不评价回答质量。

**SP/PP 复验（225）：**使用同一份首尾八层 BF16 权重、4 条固定提示词、temperature=0 和 8 个生成 token，分别启动 TP/EP 基线、SP、PP。每种模式的 completions 和 chat 均各有 4 条非空响应，每条 8 个 token 对应 8 个有限 logprob。下表按同机基线逐请求对照；当生成 token 不同时，按位置比较 logprob 不能作为精度判据。证据：[225 双接口基线](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/baseline.json) · [三种模式的实际启动命令与版本](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/launch.txt)。

| 特性 | 实测配置/方法 | 结果及范围 | 原始证据 |
| --- | --- | --- | --- |
| TP/EP | 首尾八层，TP4 / DP2，开启 EP | 两接口各 4 条 prompt；8-token 和 32-token 两组均有非空生成文本 | [8-token 逐条响应](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/new8-baseline8-strict.json) · [32-token 逐条响应](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/new8-baseline-strict.json) · [同次服务日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/new8-baseline/server.log) |
| DCP | 受测 Step5 DSA 的 DCP>1 路径 | 显式 `NotImplementedError`，`blocked` | [DCP](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_dcp.json) |
| PCP | 受测 ModelRunner V1、PCP>1 | runner 拒绝，`blocked` | [PCP](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_pcp.json) |
| SP | TP4/DP2/EP；`allgather_reducescatter`、`enable_flashcomm1=true`、`pass_config.enable_sp=true`、`sp_min_token_num=1` | 8 worker 均在 MoE 层 3–6 实际启用 SP；双接口 8/8 非空，3/8 与基线文本和 token 全同；按位置最大 logprob 绝对差 0.62598705 | [SP 逐条响应与 worker 拓扑](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/sp.json) |
| PP | 单机 TP4/DP1/PP2，开启 EP | 8 worker 实测 stage 分层 0–3、4–7；双接口 8/8 非空，3/8 与基线文本和 token 全同；按位置最大 logprob 绝对差 0.63211966 | [PP 逐条响应与 worker 拓扑](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/pp.json) |
| 图模式 | 首尾八层基线，`FULL_DECODE_ONLY`，capture sizes=`[1,2,4]` | 在此配置下双接口返回非空文本；配置记录不证明每个 shape 均实际回放 | [同次服务日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/logs/new8-baseline/server.log) · [8-token 逐条响应](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/new8-baseline8-strict.json) |
| chunk prefill | TP4/DP2/EP8，开启 chunk，预算 512 tokens | 串行 12 次 + 混合并发 12 次全部成功；8 worker、8 长请求、32 条 rank/request 记录；967→512+455，907→512+395；每步预算≤512 | [chunk JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/reports/chunked_prefill.json) |

**SP 数值复核（225）：**固定 8 条输入的前缀及目标 token 后，SP-off 在同一服务重复和重启后的 64 个位置最大差均为 0；SP-off 与 SP-on 间平均绝对差 0.07899429，最大 0.50581288。表内 0.62598705 是自由生成按位置比较，包含分叉后的不同 token。非空功能验证通过，精度等价性未获证明。见[固定前缀对照与原始记录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/225-SP-PP/fixed-prefix/README.md)。

chunk 专项的串行 completions 中，`feature_outputs_by_endpoint` 数组下标 4、5 的文本去除空白后分别为 `1` 和 `-`；该组其余 4 条及串行 chat 的 6 条均为空白。chunk 对照文本/token 相同，长输入 logprob 最大绝对差约 0.071979523。CP 限制对应表内受测版本和 runner。

SP/PP 复现使用上表 225 的实际启动命令，三个服务依次独立运行；首次 SP 启动在自动阈值推导阶段因缺少 `_C.rotary_embedding` 退出，成功轮显式设置阈值为 1。旧版八层运行脚本（非本轮命令）：[主脚本](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/run-service.sh)、[chunk/sleep1 脚本](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/run-service.sh)。

### 4.1 CP 限制

Step5 DSA 的 DCP>1 和 ModelRunner V1 的 PCP>1 实测为 `blocked`。源码复核同时发现 Step5 稀疏注意力、MRV1/CSA 对相关 CP 配置的限制，见[静态复核](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/reports/current-cp-async-sp-code-review.json)与[源码摘录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/functional-api-audit.txt)。

## 5. RL 功能验证

### 5.1 Routing replay

8 层 `run-service.sh routing` 加 `--enable-return-routed-experts`：HTTP 双接口 **8/8**、进程内 **4/4**。进程内拓扑 TP8/DP1/EP，返回 `[tokens,8,8]`，样例 `[13,8,8]`，专家索引 0..351。两条路径均为 `enforce_eager=False`、`FULL_DECODE_ONLY`。

结论以专家返回、shape/range 和无报错为依据，不以减层文本可读性为依据。[HTTP JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_replay.json)、[进程内 JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_inproc.json)。原始专家 `.npy` 在 A5 任务目录，未全部下载到本地。

### 5.2 Sleep/wake 与权重更新

#### 验收条件与执行方式

服务先以真实八层 checkpoint 启动；采用同一 checkpoint 进行 HCCL/IPC 更新或 reload。两接口各固定 4 条 prompt，temperature=0、top_p=1、seed=0、max_tokens=8。每项独立采集基线，更新/恢复后逐项比较 text、tokens、token_logprobs，任一差异即不通过；同时由未修改的上游公共验证器检查响应有效性。比较的是同一接口、同一 prompt 的前后结果。

执行入口为 `strict_mechanism_runner.py`：HCCL、IPC、reload 分别加载上游 `test_hccl_weight_update.py`、`test_npu_ipc_weight_update.py`、`test_reload_weights.py`；Level 2 五轮循环使用 checkpoint reload 恢复真实权重，Level 1 使用生命周期 wrapper。五轮及 allocator 增长阈值是扩展检查，未放宽上游的输出精确相等条件。

#### 分项结果与双接口比较

| 项目 | 实际拓扑 | 轮数 | completions 前后精确相等 | chat/completions 前后精确相等 | 运行状态 / 客户端退出码 |
| --- | --- | ---: | --- | --- | --- |
| HCCL 权重更新 | TP4/DP1/EP4 + 独立 producer 卡 4 | 1 | 4/4 | 4/4 | `passed` / 0 |
| Checkpoint 重载 | TP4/DP1/EP4 | 1 | 4/4 | 4/4 | `passed` / 0 |
| sleep Level 2 + 分阶段 wake/reload | TP4/DP1/EP4 | 5 | 20/20 | 20/20 | `passed` / 0 |
| sleep Level 1 + wake | TP4/DP1/EP4 | 5 | 20/20 | 20/20 | `passed` / 0 |
| IPC 权重更新 | TP4/DP2/EP8，8 个同卡 producer | 连续 2 轮 | 8/8 | 8/8 | 两轮均 `passed` / 0 |

表中的“精确相等”同时包含完整 text、token 序列和逐 token 的 logprob 数值。合计 **112 对**；本地重放原验证器与逐项比较再次通过。IPC 第二轮更新后响应也与第一轮基线 **8/8 对完全一致**。各权重更新报告的 `endpoint_checks` 两项均为 true，每轮 `outputs_identical_by_endpoint` 两组均为 `[true,true,true,true]`。

HCCL 实际完成 `init/start/update/finish`，使用 4352 MiB packed buffer；IPC 使用同样大小的 buffer。HCCL/reload/sleep 的服务复用不合并各项结论，每个专项都有独立的报告、命令与前后输出。

证据：[HCCL 报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-hccl/report.json) · [独立 reload 报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-reload/report.json) · [Level 2 五轮汇总](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report.json) · [Level 1 五轮报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep1/report.json) · [IPC 第 1 轮](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round1/report.json) · [IPC 第 2 轮](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round2/report.json) · [独立输出核验](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/independent-output-audit.json)

### 5.3 状态、内存与证据覆盖

Level 2 的 sleep、wake(weights)、reload、wake(kv_cache) 和恢复生成按轮留有 HTTP/worker 事件；服务日志记录每 worker 释放约 35.87 GiB。四个 worker 的五轮末 allocated 与 reserved 均无增长。Level 1 五轮的 allocated 不变，reserved 首轮下降后保持不变。内存结论覆盖上述五轮及四个 worker；metrics=`discard_all` 尚缺独立验收证据。

IPC 的 checkpoint 清单取证覆盖 8 个 worker，但参数抽样和 allocator 取证仅覆盖 DP0 的 4 个 worker。内容检查采用每参数首/中/末元素抽样。观测到每个 DP0 worker：

| 时点 | allocated | reserved |
| --- | ---: | ---: |
| IPC 第 1 轮前 | 26,772,435,968 bytes | 29,481,762,816 bytes |
| IPC 第 1 轮后 | 26,772,440,064 bytes | 48,966,402,048 bytes |
| IPC 第 2 轮后 | 26,772,440,064 bytes | 48,966,402,048 bytes |

第二轮没有继续增长，但两轮不足以证明长期无泄漏。真实 all-gather 的清单/target schema 记录也不等价于独立的 source→target 映射兼容性证明。

### 5.4 资源释放与停服检查

两轮 IPC 客户端各出现 **8 条** `Producer process has been terminated before all shared NPU tensors released` 警告，客户端均退出 0。

已按 vLLM 已合入的 [#51259](https://github.com/vllm-project/vllm/pull/51259) 将导入缓存方案适配到 225 的 NPU 接收路径：同一导出的各分块复用一个映射，更新完成或引擎关闭时释放。原 NPU 路径未使用基类缓存，存在逐块重复导入、释放的代码缺口。适配后本机 CPU 源码测试 **53 项通过**；原代码的两个新增回归均在重复导入检查处失败。

225 NPU 实机复验使用真实八层 BF16 checkpoint，TP4/DP2/EP8，8 个同卡 producer，packed buffer 4352 MiB，`max_tokens=8`。结构化报告为 `passed`：8 个 worker 的 checkpoint manifest 一致，sleep level 2、wake weights、packed IPC checkpoint 权重流、wake KV cache、两接口更新后可用性、更新前后输出精确相等和参数采样一致均为 `true`。两接口各 4 条 prompt，完成 **8/8 对**更新前后精确相等。客户端退出阶段的控制台输出及服务日志未检出 `Producer process has been terminated before all shared NPU tensors released`。

本轮正常停服路径未通过验收：完成更新后，占用守卫检测到外来 NPU PID `2587278`、`2587282`，按共享机器安全策略立即停止了测试容器。因此，本轮结论覆盖 packed IPC 更新和客户端退出告警，不覆盖正常 `stop_service.py` 的 coordinator/shared-memory 清理路径。

证据：[适配说明与范围](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/README.md) · [增量补丁](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/npu-import-once.patch) · [CPU 测试日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/cpu-candidate.log) · [负向对照](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/cpu-before.log) · [版本与检查](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/validation.json) · [NPU 实机报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/225-npu-rerun/run/results/ipc-run-rerun/report.json) · [rank0 事件](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/225-npu-rerun/run/results/ipc-run-rerun/evidence-rank0.jsonl) · [服务日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/225-npu-rerun/run/logs/ipc-run-rerun/server.log) · [占用守卫状态](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/225-npu-rerun/monitor/state.json) · [最终资源状态](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/ipc-import-once/225-npu-rerun/npu-after.txt)

完成测试后，使用 `stop_service.py` 仅停止本任务记录的进程树。HCCL 与 IPC 停止脚本均退出 0、无存活任务子进程；IPC 停服阶段出现 DP coordinator 退出异常及 2 个 shared-memory 对象清理警告，服务最终退出 1。该异常发生于两轮验证完成后的停服阶段。最终 NPU 查询八卡均无运行进程，端口 8940 无监听；退出后的 zombie 记录不等于仍占用 NPU。

证据：[HCCL/生命周期服务日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/logs/resume-hccl/server.log) · [IPC 服务日志](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/logs/resume-ipc/server.log) · [IPC 第 1 轮客户端](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round1/client.log) · [IPC 第 2 轮客户端](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round2/client.log) · [最终资源状态](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/final-resource-state.txt) · [执行与收尾摘要](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/resume-summary.json)

## 6. 性能交付

### 6.1 文本 2K→6K：8 层受控复测

10/20，A5 8 层真实 BF16，TP8/DP1/EP，`FULL_DECODE_ONLY`，关闭 prefix cache；随机输入2048、实际输出6144 tokens，`ignore-eos`。C1/C4/C8 各3轮、每组8个计时请求，**72/72 成功**；每组另有2次预热。

| 并发 | 平均输出 tok/s | 平均 TPOT ms | 吞吐 CV |
| --- | ---: | ---: | ---: |
| C1 | 266.74 | 3.743 | 0.13% |
| C4 | 1042.08 | 3.823 | 0.38% |
| C8 | 1987.07 | 3.991 | 3.54% |

性能分析采用逐调用阈值及计算侧加权口径。证据：[受控复核](https://github.com/yeyunu/step5-model-openbox/blob/main/A5-8%E5%B1%82%E6%80%A7%E8%83%BD%E5%A4%8D%E6%A0%B8.md)、[逐调用分析](https://github.com/yeyunu/step5-model-openbox/blob/main/A5-8%E5%B1%82-0.8x%E6%97%A0%E6%A0%87%E6%9D%86%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md)、[prefill 取证](https://github.com/yeyunu/step5-model-openbox/blob/main/A5-8%E5%B1%82-prefill-trace.md)、[算子性能表](https://github.com/yeyunu/step5-model-openbox/blob/main/A5-8%E5%B1%82-%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E8%A1%A8-%E5%A4%8D%E6%A0%B8%E7%89%88.xlsx)。

### 6.2 多模态 32K→4K：已实测完成

10/22，A5 8 层真实 BF16，TP8/DP1/EP；`max-model-len=40960`、batch token预算32768、最大并发4、关闭 prefix cache、`FULL_DECODE_ONLY`。每请求 **1张728×728合成图**，`random-mm` 文本32254 tokens加图像/template开销后，服务端输入恰为 **32768 tokens**；实际输出 **4096 tokens**，T=0、`ignore-eos=true`。

| 负载 | 成功数 | 输出吞吐 tok/s | TTFT ms | TPOT ms |
| --- | --- | ---: | ---: | ---: |
| C1 单请求 | 1/1 | 277.98 | 554.12 | 3.463 |
| C1 三轮，每轮4请求 | 12/12 | 平均290.76，CV1.04% | 各轮549.50～553.18 | 各轮3.268～3.341 |
| C2，4请求 | 4/4 | 562.27 | 686.12 | 3.389 |
| C4，4请求 | 4/4 | 1030.18 | 1204.75 | 3.587 |

Prefill 每个 rank 的 trace 有4次视觉 `Conv2DV2`，证明图片编码实际执行。Decode profiling 分析覆盖连续40个调度步。输入是合成内容，性能结果不作视觉语义质量证明；完整模型的视觉精度另由 MMMU-Pro 结果提供。

归档：[完整报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E5%A4%9A%E6%A8%A1%E6%80%8132K-4K/report.md)、[验收汇总 JSON](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E5%A4%9A%E6%A8%A1%E6%80%8132K-4K/results/acceptance-summary.json)、[服务脚本](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E5%A4%9A%E6%A8%A1%E6%80%8132K-4K/run-service.sh)。

### 6.3 0.8 无标杆性能分析

该项采用遵循：逐次 kernel 调用，耗时<10 μs记1，否则取该调用最大有效 pipe ratio / 0.8，再按计算耗时加权；通信、纯搬运/元数据、整数索引及采样不入计算分母。多模态还计入视觉计算，八卡评分均≥0.8时判该项减层分析通过。

| 样本 | 八卡计算侧经验评分 |
| --- | --- |
| 文本 prefill C1 / C8 | 1.121393～1.130979 / 1.123419～1.132457 |
| 文本 decode | 0.951442～0.964741 |
| 多模态 prefill | 1.139681～1.146432 |
| 多模态 decode | 0.941439～0.955522 |

**上述八卡计算侧评分均达到约定的 0.8 阈值。** 指标为减层模型的计算侧经验评分，与 MFU、硬件峰值达成率及完整模型端到端吞吐比的定义不同。多模态原始评分：[prefill](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E5%A4%9A%E6%A8%A1%E6%80%8132K-4K/results/profile-prefill/score.json)、[decode](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E5%A4%9A%E6%A8%A1%E6%80%8132K-4K/results/profile-decode/score.json)。

## 7. 精度交付：GPQA Diamond / MMMU-Pro

完整 BF16 模型完成 GPQA Diamond、MMMU-Pro 评测，分轮结果与补充回归如下。T 检验或参考差异判定尚缺同条件参考数据。

### 7.1 服务与评测参数

完整模型精度使用真实 BF16 checkpoint 在 225/226 双节点上验证。表内服务参数是五份完整正式评测的归档配置；GPQA 与 MMMU-Pro 在此基础上分别设置数据格式和生成参数。

| 项目                       | 实际参数                                                                           |
| ------------------------ | ------------------------------------------------------------------------------ |
| Checkpoint               | `/data/Step-5-Preview-BF16`                                                    |
| 模型类 / 精度                 | `Step5ForConditionalGeneration` / BF16                                         |
| 模型结构                     | 92 层文本主干；47 层视觉编码器                                                             |
| 运行环境                     | `CANN_VERSION=9.2.0`                                                           |
| 服务节点                     | 225（head）与 226（headless）；API 入口为 `177.2.25.1:10086`，served model name 为 `step` |
| TP / DP / EP             | TP=8；DP=2（每节点 local DP=1，start rank 分别为 0、1）；启用 EP                             |
| 执行后端                     | `mp`                                                                           |
| 图模式                      | `FULL_DECODE_ONLY`                                                             |
| 长度与 batch                | `max-model-len=153300`；`max-num-seqs=32`；`max-num-batched-tokens=8192`         |
| 调度与同步                    | `--no-async-scheduling`；`--disable-nccl-for-dp-synchronization`                |
| 权重加载                     | `--safetensors-load-strategy prefetch`                                         |
| 服务选项                     | `--trust-remote-code`；`--api-server-count 1`；DP RPC 地址为 `177.2.25.1:13389`     |
| `gpu-memory-utilization` | 0.9                                                                            |

| 参数                   | GPQA Diamond 完整三轮                                                     | MMMU-Pro 完整两轮                                                 |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------- |
| 数据集与本地路径             | `gpqa_diamond`；`/home/yyy/evalscope-bf16/local_datasets/gpqa_diamond` | `mmmu_pro`；`/home/yyy/evalscope-bf16/local_datasets/mmmu_pro` |
| 每轮记录数                | 198                                                                   | 1730                                                          |
| `eval-batch-size`    | 32                                                                    | 128                                                           |
| `repeats`            | 1                                                                     | 1                                                             |
| 数据集格式                | 默认格式                                                                  | `standard (10 options)`                                       |
| `do_sample`          | `true`                                                                | `true`                                                        |
| `temperature`        | 1.0                                                                   | 1.0                                                           |
| `top_p` / `top_k`    | 0.95 / 0                                                              | 0.95 / 0                                                      |
| `logprobs`           | 未显式设置                                                                 | `true`                                                        |
| `timeout`            | 360000 s                                                              | 360000 s                                                      |
| 最大输出长度（`max_tokens`） | 149204                                                                | 149204                                                        |
| 任务级 `seed`           | 42                                                                    | 42                                                            |

`repeats=1` 是 evalscope 内部重复次数；GPQA 的三轮和 MMMU-Pro 的两轮分别由独立评测进程执行。评测工具为 `evalscope 1.13.0`；`GenerateConfig.max_tokens` 是可选字段，默认值为 `None`，`openai_api` 模式的默认生成配置也未设置该字段。五份正式轮次的命令行都没有配置 `max_tokens`，请求构造逻辑在值为 `None` 时省略该字段，因此正式请求没有固定最大输出长度。服务端 `max-model-len=153300` 是总上下文上限，不是只针对输出的固定上限。

证据：[服务与正式评测脚本摘录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/full-model-service-scripts-observation.txt) · [evalscope 最大输出参数源码核验](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/evalscope-max-tokens-source-observation.txt) · [完整模型 config](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/full-model-config-observation.txt) · [补充回归脚本](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/BF16%E6%91%98%E8%A6%81%E9%80%82%E9%85%8D/run_eval.sh)

### 7.2 完整轮次

| 数据集               | 轮次/输出时间目录                 |          记录数 |   Accuracy |
| ----------------- | ------------------------- | -----------: | ---------: |
| GPQA Diamond      | 第1轮 `20261025_024652`     |    198，188正确 |     94.95% |
| GPQA Diamond      | 第2轮 `20261025_033621`     |    198，184正确 |     92.93% |
| GPQA Diamond      | 第3轮 `20261025_043008`     |    198，185正确 |     93.43% |
| **GPQA 完整三轮平均**   | 同一组完整评测                   | 594条评分，557正确 | **93.77%** |
| MMMU-Pro          | 第1轮 `20261024_234407`     |         1730 |     76.65% |
| MMMU-Pro          | 第2轮 `20261025_052014`     |         1730 |     75.72% |
| **MMMU-Pro 两轮平均** | 按显示分数76.65%、75.72%计算并四舍五入 |      3460条评分 | **76.19%** |

GPQA 各完整轮次有198条预测/评审，MMMU-Pro 各1730条；第一轮 MMMU-Pro 另有 `valid=True`、退出0、1730唯一记录、1730次HTTP200 的审计。其余轮次已有完整预测/评审计数和正式评分，逐请求 HTTP 审计尚未完整归档。MMMU-Pro 使用 `standard (10 options)` 口径。

原始报告输出摘录：[五份完整评分](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/five-complete-reports.txt)、[轮次计数及MMMU审计](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/all-runs-counts-and-mmmu-audit.txt)。评分证据为工具读取摘录，保留终端换行/截断，并附会话路径、行号、时间及记录 SHA256。

### 7.3 GPQA 部分覆盖批次统计

按约定统计口径，任务 `20261019_125836` 的前198条 prediction，逐条匹配 review 后 **186/198=93.94%**，与前三轮合计的四份平均 **93.81%**。这198条只覆盖 **58个独立问题**，不是额外完整198题轮次；未回答或截断无答案按错误计入，不剔除。

该批次的[服务脚本摘录](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/partial-round-old-service-config.txt)为 TP4/DP4、EP开启、Ray、max-model-len=16384（EP16由TP×DP推得）。完整三轮均值为93.77%，计入该批次后的混合统计为93.81%；后者包含重复问题记录。

证据：[逐条匹配](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/partial-round-score-match.txt)、[覆盖范围](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/partial-round-coverage.txt)、[用户口径确认](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/%E7%B2%BE%E5%BA%A6%E4%BC%9A%E8%AF%9D%E8%AF%81%E6%8D%AE/user-partial-round-choice-1.txt)。

### 7.4 10/27 BF16 摘要适配后的补充回归

单列该软件栈结果，不混入上面的均值：GPQA **184/198=92.93%**，198个成功响应、0 API失败；197条自然结束，1条达到配置的 `max_tokens=149204` 并计错，全部样本保留。正式评测退出0，额外严格审计因这条截断退出1。另有7707输入tokens、38输出tokens的校验请求，checksum正确。

证据：[结果汇总](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/BF16%E6%91%98%E8%A6%81%E9%80%82%E9%85%8D/result-summary.json)、[审计](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/BF16%E6%91%98%E8%A6%81%E9%80%82%E9%85%8D/audit.json)、[原始评分](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/BF16%E6%91%98%E8%A6%81%E9%80%82%E9%85%8D/gpqa_diamond.json)。该条重复/超长输出的成因尚未通过对照实验区分采样、模型行为或适配回归，成因待定位。

## 8. 报错修复与 Agent 案例

详细记录见[报错修复](https://github.com/yeyunu/step5-model-openbox/blob/main/%E6%A8%A1%E5%9E%8B%E5%BC%80%E7%AE%B1%E5%A4%8D%E7%9B%98/%E6%8A%A5%E9%94%99%E4%BF%AE%E5%A4%8D.md)，代码关系见[本地修复 diff 与上游关系](https://github.com/yeyunu/step5-model-openbox/blob/main/%E6%A8%A1%E5%9E%8B%E5%BC%80%E7%AE%B1%E5%A4%8D%E7%9B%98/Step5%208%E5%B1%82%E5%8A%9F%E8%83%BD%E9%AA%8C%E8%AF%81%EF%BC%9A%E6%9C%AC%E5%9C%B0%E4%BF%AE%E5%A4%8D%20diff%20%E4%B8%8E%E4%B8%8A%E6%B8%B8%E5%85%B3%E7%B3%BB.md)。

| 案例 | 修复与验证结果 | 证据入口 |
| --- | --- | --- |
| 构建环境冲突 | 定位 FLA `.pth` 注入导致的算子签名冲突，配置 `FLA_NPU_DISABLE_PTH=1` 后构建运行 | [冒烟过程](https://github.com/yeyunu/step5-model-openbox/blob/main/vllm_ascend/A5%20Step5-preview%20%E5%86%92%E7%83%9F%E9%AA%8C%E8%AF%81%E5%85%A8%E8%BF%87%E7%A8%8B.md) |
| 真实权重流式更新 | 已迁移至上游已合入的 IPC/HCCL 实现并适配 Step5；HCCL、IPC、reload 分项验证完成 | 第 3.5、5 节 |
| 专家映射重载 | 接入 #20403 固定生产补丁的等价 backport；5 项回归通过 | 第 3.5 节 |
| 减层权重构造 | 首尾 8 层构造及 802 个张量内容校验完成；双接口 16 条响应通过公共验证器 | 第 2.2、3.4 节 |
| 生命周期与更新复验 | Level 1/2 各 5 轮、HCCL 1 轮、reload 1 轮、IPC 2 轮；112 对输出精确相等 | 第 5.2 节 |
| chunk 取证 | 24 次串行/混合并发请求成功，观测到实际多步分块及每步预算 | 第 4 节 |
| 性能分析 | 完成逐调用指标校正、受控复测和文本/视觉计算分析 | 第 6 节 |
| BF16 摘要适配 | 修复双字节缓存分配及入口，6 项回归和完整 GPQA 补充评测 | [修复记录](https://github.com/yeyunu/step5-model-openbox/blob/main/%E6%A8%A1%E5%9E%8B%E5%BC%80%E7%AE%B1%E5%A4%8D%E7%9B%98/%E6%8A%A5%E9%94%99%E4%BF%AE%E5%A4%8D.md)、第 7.4 节 |

IPC 共享张量释放路径已完成导入缓存适配、CPU 验证和一轮 NPU packed 更新复验；客户端退出阶段未复现原告警。正常停服清理路径仍待复验，详见第 5.4 节。Agent 案例已保存问题、修改和复验依据；未建立人工工时基准。

## 9. 交付目录与剩余事项

| 归档内容 | 证据入口 |
| --- | --- |
| 基本功能及 routing 专项 | [主报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/report.md)、[README](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/README.md)、[执行命令](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/logs/launch_commands.txt) |
| chunk prefill | [补充报告](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-8layer-chunk-sleep1/report.md) |
| 权重更新与生命周期 | [上游脚本生成的 README](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/README.md)、[执行与收尾摘要](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/resume-summary.json) |
| 复现命令与脚本 | [SERVICE/VERIFIER/STOP 命令](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/logs/launch_commands.txt)、[执行脚本](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/executed-scripts/run-resume-ipc-client.sh) |
| 证据完整性 | [SHA256 清单](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/evidence-sha256.json)、[输出独立核验](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/reports/independent-output-audit.json) |
| 性能与精度 | 第 6、7 节的分项报告、评分及脚本；[附件清单](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E9%AA%8C%E6%94%B6%E6%B1%87%E6%80%BB/manifest.json) |
| 报错修复与验证 | [报错修复](https://github.com/yeyunu/step5-model-openbox/blob/main/%E6%A8%A1%E5%9E%8B%E5%BC%80%E7%AE%B1%E5%A4%8D%E7%9B%98/%E6%8A%A5%E9%94%99%E4%BF%AE%E5%A4%8D.md) |

