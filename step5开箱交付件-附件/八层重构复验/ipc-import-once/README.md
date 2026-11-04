# Packed NPU IPC 导入缓存适配

已把 vLLM 已合入 PR [#51259](https://github.com/vllm-project/vllm/pull/51259) 的按导出缓存映射方案适配到 225 的 Step5 测试代码，并在 225 上完成一轮 packed NPU IPC 实机验证。本轮更新、恢复和更新后输出比对通过；客户端退出阶段未检出原有的共享 NPU 张量未释放告警。由于占用守卫在随后停服时检测到外来 NPU 进程并立即停止了测试容器，正常停服路径仍需单独验证。

## 来源与适配范围

上游合入提交：`f37dff22fc77e21ac857f2023faa7d5b8f9c9591`。该 PR 修复 CUDA packed IPC 一次导出、逐块重复导入和释放造成的引用计数失衡。225 的 vLLM 基类已包含这项修复，但 Ascend 的接收覆盖方法直接逐块调用 `rebuild_npu_tensor`，没有使用基类缓存。

Ascend [#16770](https://github.com/vllm-project/vllm-ascend/pull/16770)、[#16960](https://github.com/vllm-project/vllm-ascend/pull/16960) 含相关适配思路，但查询结果为关闭且未合入。本补丁是基于已合入 vLLM 实现的 NPU 适配，不能表述成直接合入了 Ascend 已合并 PR。

`NPUPackedBufferImporter` 继承上游缓存状态和 `close()`，将重建函数替换为 `rebuild_npu_tensor`。NPU engine 持有一个 importer 并传入每次 packed 接收；继承基类的 `finish_weight_update()` / `shutdown()` 释放逻辑。完整重建参数作为缓存标识，以区分同一存储上的不同导出。分块解包仍 clone 到独立张量，不改变发送协议、unpacked 路径或模型权重。

代码目录：225 容器 `step5-recheck-225-10.20` 内 `/home/yyy/model_openbox/reduced-recheck-225-10.20/vllm-ascend`。只修改两个生产文件和一个测试文件，保留原有未提交改动；本次没有创建 Git 提交或推送。修改前文件位于同级 `ipc-import-once-10.20/before/`。

## 验证

- 本机 CPU 源码测试：53 项通过；使用真实 CPU 张量，硬件重建函数和配置依赖使用替身，真实 engine、importer、解包与生命周期源码保持原样。没有导入 `torch_npu` 或初始化 CUDA。
- 原代码负向对照：两个新增生命周期回归均在同一导出实际导入两次（`2 != 1`）处失败。
- 覆盖跨块数据更新、已加载权重独立性、完成/关闭释放、重复关闭、下一轮重新导入、同一存储不同导出替换，以及重建失败后的清理。
- Ruff lint/format 通过；远端三个文件 SHA256 与通过测试的本机文件一致；`git diff --check` 和本补丁反向 apply 检查通过。
- 仓库通用 CPU 单测启动受 Triton/Ascend 依赖初始化影响，未完成；采用独立 CPU 源码测试入口，不能称为整个仓库测试通过。
- 225 NPU 实机验证：真实八层 BF16 checkpoint，TP4/DP2/EP8，8 个同卡 producer，packed buffer 4352 MiB，`max_tokens=8`。结构化报告 `status=passed`；checkpoint manifest、sleep level 2、wake weights、packed IPC 权重流、wake KV cache、两个生成接口更新后可用性、输出精确相等和参数采样一致均为 `true`。
- 两接口各 4 条 prompt，即 completions 4/4、chat/completions 4/4 在更新前后精确相等；8 个 worker 的 checkpoint manifest 一致。客户端完成更新后退出阶段的控制台输出及服务日志均未检出 `Producer process has been terminated before all shared NPU tensors released`。
- 实机验证后的停服未完成：host-side 占用守卫检测到外来 NPU PID `2587278`、`2587282` 后按安全策略停止了测试容器。不能把该行为记作正常 `stop_service.py` 清理通过。

本轮组合证据覆盖真实 NPU packed 更新、输出与参数一致，以及更新客户端退出时未复现原告警。它不覆盖正常停服时的 DP coordinator/shared-memory 清理路径，也不证明长期、多轮场景不存在引用计数或内存增长问题。

## 证据

- `upstream.json`：上游状态与原始补丁。
- `npu-import-once.patch`：相对于本次修改前工作区的增量补丁。
- `validation.json`：修改前后 SHA256、CPU 测试结果与范围。
- `cpu-before.log` / `cpu-candidate.log`：负向对照和通过日志。
- `finalize.log`：225 文件一致性和 Git 检查。
- `before/` / `candidate/` / `dependencies/` / `cpu_check.py` / `validate.py`：源码快照、上游依赖和复现入口。
- `225-npu-rerun-10.20/`：225 实机运行容器内报告、8 rank 事件、服务日志、容量预算、占用守卫状态和最终资源快照。
