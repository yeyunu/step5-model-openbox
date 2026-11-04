# A5 多模态 32768 输入 -> 4096 输出性能实测

日期：2026-10-22  
范围：Step5 Preview BF16 8 层减层模型，A5/yyy-cann92，TP8/DP1/EP，8 张 Ascend950DT

## 结论

按减层模型既有的“逐 kernel 调用、计算耗时加权、0.8 分母”经验规则，本次多模态长上下文测试在 Prefill 和 Decode 两侧都达到目标：

- Prefill 八卡评分范围 `1.139681～1.146432`，均值 `1.142720`。
- Decode 八卡评分范围 `0.941439～0.955522`，均值 `0.950103`。
- 正式请求 `1/1` 成功，实际输入 `32768` tokens、输出 `4096` tokens。
- 非 profile 的单请求耗时 `14.735 s`、输出吞吐 `277.98 tok/s`、TTFT `554.12 ms`、TPOT `3.463 ms`。

因此，与之前的判断相比可以更新为：**8 层减层模型上的多模态 32K->4K 已实测跑通且达到 0.8 规则；这仍不是 92 层完整模型、输出质量或正式产品性能的验收。**

## 之前无法判断 32K 的限制已排除

旧配置的 `max-model-len=12288` 无法接收 32K 输入，所以此前只能证明视觉通路功能，不能完成目标测试。本轮服务改为：

- `--max-model-len 40960`
- `--max-num-batched-tokens 32768`
- `--max-num-seqs 4`
- `--no-enable-prefix-caching`
- `FULL_DECODE_ONLY`，capture sizes `[1,2,4]`

请求确实携带一张原生分辨率的 `728x728` 合成图片。`random-mm` 的 `random-input-len=32254` 加上图片与 chat template 开销后，服务端统计的总输入为准确的 `32768` tokens；不是把 32K 文本长度直接当作可服务上限。

## 负载口径

- 数据源：`random-mm`，每请求一张 `728x728` 图片。
- 输入：服务端实测 `32768` tokens。
- 输出：`4096` tokens，`ignore-eos=true`，`temperature=0`。
- 测试使用随机合成内容，因此只评价性能/执行链路，不评价生成质量。
- Prefill profile 与 Decode profile 前均先跑了一次非 profile 的 `32514` 总输入预热，避免 profiler 落在首次重计算和脏设备状态上。

## 正常服务基准结果

|场景|成功/总数|时长 s|输出 tok/s|TTFT 均值 ms|TTFT P99 ms|TPOT 均值 ms|
|---|---:|---:|---:|---:|---:|---:|
|C1 单请求|1/1|14.735|277.98|554.12|554.12|3.463|
|C1 三轮各 4 请求|12/12|55.74～56.91/轮|290.76 平均|549.50～553.18/轮|559.42～561.01/轮|3.268～3.341/轮|
|C2 四请求|4/4|29.139|562.27|686.12|858.41|3.389|
|C4 四请求|4/4|15.904|1030.18|1204.75|1552.50|3.587|

C1 三轮输出吞吐为 `290.476 / 293.928 / 287.878 tok/s`，样本标准差 `3.035`，CV `1.04%`。这些是非 profiler 数字；带 profiler 的 Decode 请求为 `272.95 tok/s`、`15.006 s`，不能替代非 profile 吞吐基线。

## 评分方法

对每个 rank 的 `kernel_details.csv` 独立执行以下规则：

1. 范围包含所有浮点主干计算和视觉计算算子，视觉 `Conv2DV2`、`ResizeBilinearV2` 计入分母。
2. 对每一次 kernel 调用单独判断：耗时 `<10 us` 记 `1`；否则记 `max(本次调用有效 pipe ratio)/0.8`。
3. 用该次 kernel 的计算耗时加权，得到该 rank 的评分。
4. 通信、纯搬运/元数据、整数路由/索引、采样不进入计算分母。
5. 不允许拿同名算子跨调用取最大利用率，也不把 `N/A` 当 0。
6. 8 个 rank 均须不低于 `0.8` 才判减层模型通过。

评分是计算侧经验指标，不是 MFU、硬件峰值达成率或端到端加速比。评分可大于 1，原因是经验分母固定为 0.8。

本次两个 profile 的八个 rank 中，未分类计算耗时为 `0 us`，计算算子缺 pipe ratio 的耗时为 `0 us`。

## Prefill 评分

Profile 配置：`delay_iterations=0`、`max_iterations=1`。请求为 `32768` 输入、`1` 输出。每个 rank 的 trace 都包含 4 次视觉 `Conv2DV2`，确认图片编码执行经过 profiler 覆盖范围。

|Rank|经验评分|计算 kernel 求和 us|
|---:|---:|---:|
|0|1.145982|181544.474|
|1|1.142814|178147.907|
|2|1.146432|183592.696|
|3|1.141276|178210.748|
|4|1.140594|176183.999|
|5|1.144920|182573.931|
|6|1.140060|177861.498|
|7|1.139681|176474.233|

rank0 的主要计算项包括：`GenericBlockSparseAttention` 2 次/56.236 ms、`MatMulV3` 426 次/62.759 ms、`GroupedMatmul` 10 次/18.603 ms、`FusedInferAttentionScore` 100 次/7.018 ms。两次 GSA 的 cube utilization 分别约为 `99.63%` 和 `99.66%`，本次 32K Prefill trace 不支持“GSA 当前低利用率”的旧结论。

## Decode 评分

服务配置为 `delay_iterations=2048`、`max_iterations=20`。先跑 `32514` 总输入非 profile 预热，再跑准确的 `32768 -> 4096` profile 请求。服务进程内解析被 CANN 拒绝，改用 `torch_npu.profiler.profiler.analyse` 对 8 个 rank 离线解析，全部成功。

离线 CSV 每个 rank 均有 `6242` 行、40 次 `GenericBlockSparseAttention`、40 次 `_csa_decode_stage_flush_kernel`、120 次 `FusedInferAttentionScore`，即实际纳入统计的是连续 40 个 Decode 调度步；没有视觉卷积调用，符合 Decode 阶段特征。

|Rank|经验评分|计算 kernel 求和 us|
|---:|---:|---:|
|0|0.949002|22167.540|
|1|0.954206|21131.057|
|2|0.952163|22520.836|
|3|0.953358|21345.050|
|4|0.944052|22524.441|
|5|0.955522|21522.196|
|6|0.941439|22874.186|
|7|0.951080|21616.600|

rank0 的主要计算项为：`MatMulV3` 920 次/8.416 ms、`_csa_decode_stage_flush_kernel` 40 次/2.859 ms、`GroupedMatmul` 200 次/2.944 ms、`FusedInferAttentionScore` 120 次/1.672 ms、`GenericBlockSparseAttention` 40 次/1.175 ms。

## 限制与结论边界

- 这是 8 层真实 BF16 减层模型，不能外推为 92 层完整模型已达到同一评分或吞吐。
- 使用合成随机多模态输入和固定长度输出，不代表真实 OCR、图表、自然图片或动态输出分布。
- 长上下文正式性能只在单个请求上重复展开；C1/C2/C4 正常基准有多次请求，但没有做长时间稳定性或多机压测。
- Decode csv 覆盖连续 40 个调度步，而不是完整 4096 步；评分是这段稳态样本的计算侧结果。
- 所有评分与 trace 是当前 A5 软件栈、TP8 配置和模型版本的结果，不能直接迁移到其他 CANN、并行策略或模型。

## 证据文件

- 汇总：`/home/yyy/model_openbox/mm-32k-4k/results/acceptance-summary.json`
- Prefill 评分：`/home/yyy/model_openbox/mm-32k-4k/results/profile-prefill/score.json`
- Prefill benchmark：`/home/yyy/model_openbox/mm-32k-4k/results/profile-prefill/profile-prefill-full.json`
- Prefill trace：`/home/yyy/model_openbox/mm-32k-4k/prof-prefill/`
- Prefill 服务日志：`/home/yyy/model_openbox/mm-32k-4k/logs/serve-prefill-prof.log`
- Prefill 首次失败日志：`/home/yyy/model_openbox/mm-32k-4k/logs/serve-prefill-prof.failed-first.log`
- Decode 评分：`/home/yyy/model_openbox/mm-32k-4k/results/profile-decode/score.json`
- Decode benchmark：`/home/yyy/model_openbox/mm-32k-4k/results/profile-decode/profile-decode-full.json`
- Decode trace：`/home/yyy/model_openbox/mm-32k-4k/prof-decode/`
- Decode 离线解析日志：`/home/yyy/model_openbox/mm-32k-4k/results/profile-decode/analyse.log`
- 正常 C1：`/home/yyy/model_openbox/mm-32k-4k/results/c1-three-rounds/`
- 正常 C2/C4：`/home/yyy/model_openbox/mm-32k-4k/results/concurrency/`
- 评分脚本：`/home/yyy/model_openbox/mm-32k-4k/score_profile.py`
- 服务脚本：`run-service-prefill-prof.sh`、`run-service-decode-prof.sh`
