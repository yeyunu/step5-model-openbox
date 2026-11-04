---
tags:
  - vllm-ascend
  - Step5
  - 权重更新
date: 2026-10-16
---

# Step5 8 层功能验证：本地修复 diff 与上游关系

## 结论：不能认定这些修改在上游从未出现过

本次主会话针对 A5 上现有 Step5 8 层权重进行了本地修复和实机功能验证。不能由本地有 diff 推断这些改动都是上游没有的新修复。

| 问题方向                          | 上游工作                                                                            | 本次查询时状态        | 与本地修改的关系                                 |
| ----------------------------- | ------------------------------------------------------------------------------- | -------------- | ---------------------------------------- |
| Level-2 sleep 后 MoE 权重布局和加载属性 | [vllm-ascend PR #13012](https://github.com/vllm-project/vllm-ascend/pull/13012) | 已合入，2026-08-07 | 相关布局修复，不等于本次 Step5 IPC 补丁完全相同            |
| packed HCCL 加载顺序              | [vllm-ascend PR #18763](https://github.com/vllm-project/vllm-ascend/pull/18763) | open，未合入       | 针对跨流加载顺序，不能等同于本次 CPU 暂存或退出修复             |
| IPC 引擎及生命周期                   | [vllm-ascend PR #18848](https://github.com/vllm-project/vllm-ascend/pull/18848) | open，未合入       | 已有相关重构，尚未逐行比对本次补丁                        |
| 逐层重载缓存膨胀                      | [vLLM RFC #59502](https://github.com/vllm-project/vllm/issues/59502)            | open RFC       | 明确讨论未切分权重缓存导致额外显存占用，提出 modulewise reload |

以上状态在本次侧会话通过 GitHub API / 页面核对，后续可能变化。

**尚未完成的核查**：未将 Step5 完整性 guard、CPU 暂存、IPC packed 加载修正、通信器 close 逐一与最新上游提交及全部历史 PR 做等价性对照。因此不能保证都是“上游没有的改动”，也不能把上述相关 PR 当成本地补丁的直接来源。

`git diff HEAD` 只表示工作区与当前检出的 HEAD 有差异。目录名包含 `main`，不证明该 HEAD 就是上游最新 main。验证脚本修改属于 MindSpeed-Agent 和任务目录，不属于 vllm-ascend 主仓。

## 修改做了什么

1. Step5 分块加载期间累计已加载参数，整轮结束时检查缺失参数及融合 QKV/indexer 分片。
2. HCCL / IPC 接收权重暂存到 CPU，降低逐层重载期间 NPU 缓存峰值。
3. IPC packed 解包后实际调用模型 `load_weights`；Step5 在加载前后执行分层重载的布局恢复流程。
4. HCCL 增加显式 communicator close；修复后实机重跑正常退出。
5. 验证器按用户授权接受空格、重复或乱码，保留 token 非空、logprob 对齐且有限，以及三种更新前后 text/token/logprob 完全一致检查。
6. 下方同时收录相关回归测试及 routing 修改。

## 机器、范围与基线

- 机器：A5（177.8.21.1）；容器：`yyy-cann92`。
- 权重：`/data/Step-5-Preview-BF16-8layer`，8 层、807 张量、约 70.34 GiB。
- 任务目录：`/home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808`。
- 本文是远端现状的审阅快照；侧会话只读取远端，未修改远端实现。

下面 diff 按相关文件筛选，以各仓库当前 HEAD 为基线。它不是相对于最新上游 main 的补丁；相关文件可能混有历史修改，不能认定每一行都由本次修复新增。未收录依赖安装、环境文件和备份文件。

新增 `step5_weight_update.py` 尚未跟踪，单独使用 `git diff --no-index /dev/null ...` 导出，避免普通 diff 漏掉。

采集时间：2026-10-16 14:04:13 +08:00

## Diff：vllm-ascend 核心修复与测试

仓库：/home/yyy/vllm-ascend-main

HEAD：80d265f13e53b2c596da31388a1eedde2bb7f94d

~~~~diff
diff --git a/tests/ut/distributed/device_communicators/test_pyhccl.py b/tests/ut/distributed/device_communicators/test_pyhccl.py
index 09318c45e4..ac1520eab7 100644
--- a/tests/ut/distributed/device_communicators/test_pyhccl.py
+++ b/tests/ut/distributed/device_communicators/test_pyhccl.py
@@ -67,3 +67,19 @@ class TestPyHcclCommunicator(TestBase):
         self.assertEqual(comm.world_size, 2)
         self.assertFalse(comm.available)
         self.assertTrue(comm.disabled)
+
+
+def test_close_destroys_communicator_once_before_runtime_shutdown():
+    comm = PyHcclCommunicator.__new__(PyHcclCommunicator)
+    comm.available = True
+    comm.disabled = False
+    comm.device = "npu:0"
+    comm.comm = object()
+    comm.hccl = MagicMock()
+    with patch("torch.npu.device"), patch("torch.npu.synchronize") as sync:
+        comm.close()
+        comm.close()
+    sync.assert_called_once()
+    comm.hccl.hcclCommDestroy.assert_called_once_with(comm.comm)
+    assert not comm.available
+    assert comm.disabled
diff --git a/tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py b/tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py
index 5815327a17..3a80d2a6b7 100644
--- a/tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py
+++ b/tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py
@@ -35,6 +35,7 @@ import sys
 import types
 from unittest.mock import MagicMock, patch
 
+import pytest
 import torch
 
 from vllm_ascend.distributed.weight_transfer import npu_ipc_engine
@@ -162,22 +163,61 @@ def test_receive_weights_rebuilds_with_rebuild_npu_tensor():
     assert seen["args"][6] == device_index
 
 
-def test_start_weight_update():
+@pytest.mark.parametrize('model_type', ['step5', 'other'])
+def test_start_weight_update(model_type):
     engine = object.__new__(NPUIPCWeightTransferEngine)
     engine.model = MagicMock()
+    engine.model_config = MagicMock()
+    engine.model_config.hf_text_config.model_type = model_type
 
     with patch("vllm.model_executor.model_loader.reload.initialize_layerwise_reload") as mock_init:
         engine.start_weight_update()
 
-    mock_init.assert_not_called()
+    if model_type == 'step5':
+        mock_init.assert_called_once_with(engine.model)
+    else:
+        mock_init.assert_not_called()
 
 
-def test_finish_weight_update():
+@pytest.mark.parametrize('model_type', ['step5', 'other'])
+def test_finish_weight_update(model_type):
     engine = object.__new__(NPUIPCWeightTransferEngine)
     engine.model = MagicMock()
     engine.model_config = MagicMock()
+    engine.model_config.hf_text_config.model_type = model_type
 
     with patch("vllm.model_executor.model_loader.reload.finalize_layerwise_reload") as mock_finalize:
         engine.finish_weight_update()
 
-    mock_finalize.assert_not_called()
+    if model_type == 'step5':
+        mock_finalize.assert_called_once_with(engine.model, engine.model_config)
+    else:
+        mock_finalize.assert_not_called()
+
+
+def test_packed_receive_applies_checkpoint_values_to_model():
+    from types import SimpleNamespace
+
+    class Model:
+        def __init__(self):
+            self.weight = torch.zeros(2)
+
+        def load_weights(self, weights):
+            for name, value in weights:
+                assert name == "weight"
+                self.weight.copy_(value)
+
+    engine = NPUIPCWeightTransferEngine.__new__(NPUIPCWeightTransferEngine)
+    engine.model = Model()
+    engine.device = torch.device("npu:0")
+    engine.packed = True
+    info = SimpleNamespace(
+        ipc_handles={}, names=["weight"], shapes=[[2]],
+        dtype_names=["float32"], tensor_sizes=[8],
+    )
+    value = torch.tensor([2., 3.])
+    with patch(f"{_MODULE}.npu_generate_uuid", return_value="card-0"), patch(
+        f"{_MODULE}.packed_npu_ipc_consumer", return_value=[("weight", value)]
+    ):
+        engine.receive_weights(info)
+    torch.testing.assert_close(engine.model.weight, value)
diff --git a/vllm_ascend/distributed/device_communicators/pyhccl.py b/vllm_ascend/distributed/device_communicators/pyhccl.py
index 44d1b12fc3..8ae7f03acf 100644
--- a/vllm_ascend/distributed/device_communicators/pyhccl.py
+++ b/vllm_ascend/distributed/device_communicators/pyhccl.py
@@ -127,6 +127,16 @@ class PyHcclCommunicator:
             stream.synchronize()
             del data
 
+    def close(self) -> None:
+        """Release the communicator before the NPU runtime shuts down."""
+        if not self.available or self.disabled:
+            return
+        with torch.npu.device(self.device):
+            torch.npu.synchronize()
+            self.hccl.hcclCommDestroy(self.comm)
+        self.disabled = True
+        self.available = False
+
     def all_reduce(self, in_tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM, stream=None) -> torch.Tensor:
         if self.disabled:
             return None
diff --git a/vllm_ascend/distributed/weight_transfer/hccl_engine.py b/vllm_ascend/distributed/weight_transfer/hccl_engine.py
index 023ffd1ae2..15378cba9c 100644
--- a/vllm_ascend/distributed/weight_transfer/hccl_engine.py
+++ b/vllm_ascend/distributed/weight_transfer/hccl_engine.py
@@ -210,7 +210,7 @@ class HCCLWeightTransferEngine(WeightTransferEngine[HCCLWeightTransferInitInfo,
 
     def shutdown(self) -> None:
         if self.model_update_group is not None:
-            # Clean up the communicator by removing the reference
+            self.model_update_group.close()
             self.model_update_group = None
 
     @staticmethod
diff --git a/vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py b/vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py
index 7a63adf7b6..6e31cfd024 100644
--- a/vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py
+++ b/vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py
@@ -156,12 +156,22 @@ class NPUIPCWeightTransferEngine(  # type: ignore[no-redef]
         self.packed = init_info.packed
 
     def start_weight_update(self) -> None:
-        """No-op for NPU IPC engine (no layerwise reloading)."""
-        pass
+        """Restore Step5 checkpoint layouts before loading streamed weights."""
+        if self.model_config.hf_text_config.model_type == "step5":
+            from vllm.model_executor.model_loader.reload import (
+                initialize_layerwise_reload,
+            )
+
+            initialize_layerwise_reload(self.model)
 
     def finish_weight_update(self) -> None:
-        """No-op for NPU IPC engine (no layerwise reloading)."""
-        pass
+        """Restore Step5 runtime layouts after all checkpoint tensors arrive."""
+        if self.model_config.hf_text_config.model_type == "step5":
+            from vllm.model_executor.model_loader.reload import (
+                finalize_layerwise_reload,
+            )
+
+            finalize_layerwise_reload(self.model, self.model_config)
 
     def receive_weights(self, update_info: NPUIPCWeightTransferUpdateInfo) -> None:
         """Receive weights from the trainer via NPU IPC handles.
@@ -219,7 +229,7 @@ class NPUIPCWeightTransferEngine(  # type: ignore[no-redef]
                 weight = rebuild_npu_tensor(*list_args)
                 weights.append((name, weight))
 
-            self.model.load_weights(weights)
+        self.model.load_weights(weights)
 
     def shutdown(self) -> None:
         pass
diff --git a/vllm_ascend/worker/worker.py b/vllm_ascend/worker/worker.py
index a44ce0ce6c..c263b586c5 100644
--- a/vllm_ascend/worker/worker.py
+++ b/vllm_ascend/worker/worker.py
@@ -328,7 +328,22 @@ class NPUWorker(WorkerBase):
         self._check_nz_disabled()
 
         assert self.weight_transfer_engine is not None
-        self.weight_transfer_engine.start_weight_update()
+        self._step5_update_guard = None
+        if self.model_config.hf_text_config.model_type == "step5":
+            from vllm_ascend.patch.worker.step5_weight_update import Step5WeightUpdateGuard
+
+            self._step5_update_guard = Step5WeightUpdateGuard(
+                self.weight_transfer_engine.model,
+                stage_to_cpu=self.vllm_config.weight_transfer_config.backend
+                in ("hccl", "npu_ipc"),
+            )
+            logger.info("Step5 streamed weight completeness guard enabled.")
+        try:
+            self.weight_transfer_engine.start_weight_update()
+        except BaseException:
+            if self._step5_update_guard is not None:
+                self._step5_update_guard.abort()
+            raise
         self._weight_update_active = True
 
     def update_weights(self, update_info: dict) -> None:
@@ -343,6 +358,8 @@ class NPUWorker(WorkerBase):
         try:
             self.weight_transfer_engine.update_weights(update_info)
         except BaseException:
+            if self._step5_update_guard is not None:
+                self._step5_update_guard.abort()
             self._weight_update_active = False
             raise
 
@@ -354,8 +371,14 @@ class NPUWorker(WorkerBase):
             raise RuntimeError("start_weight_update must be called before finish_weight_update.")
 
         assert self.weight_transfer_engine is not None
-        self.weight_transfer_engine.finish_weight_update()
-        self._weight_update_active = False
+        try:
+            self.weight_transfer_engine.finish_weight_update()
+            if self._step5_update_guard is not None:
+                self._step5_update_guard.finish()
+        finally:
+            if self._step5_update_guard is not None:
+                self._step5_update_guard.abort()
+            self._weight_update_active = False
 
     def shutdown(self) -> None:
         if ensure_kv_transfer_shutdown is not None:
~~~~

## Diff：vLLM Step5 回归测试与 routing

仓库：/home/yyy/vllm-ascend-step4

HEAD：5d7f9124db0304ad177f71969f26a70316ecc632

~~~~diff
diff --git a/tests/model_executor/model_loader/test_step5_fused_weight_loading.py b/tests/model_executor/model_loader/test_step5_fused_weight_loading.py
index 444defc423..1705869f16 100644
--- a/tests/model_executor/model_loader/test_step5_fused_weight_loading.py
+++ b/tests/model_executor/model_loader/test_step5_fused_weight_loading.py
@@ -1223,3 +1223,49 @@ def test_router_bias_cpu_fallback_preserves_reference_semantics(
     assert ids.dtype == torch.int64
     assert ids[0].tolist() == expected_ids.tolist()
     assert ids[1].tolist() == [7, 7]
+
+
+@pytest.mark.parametrize("stage_to_cpu", [False, True])
+@pytest.mark.parametrize("omit", [None, "index_g", "dense"])
+def test_weight_update_transaction_checks_all_chunks_and_resets(omit, stage_to_cpu):
+    """Partial chunks are valid, but missing parameters or fused shards fail."""
+    from vllm_ascend.patch.worker.step5_weight_update import Step5WeightUpdateGuard
+
+    from vllm.model_executor.model_loader.mtp_validation import (
+        is_mtp_completeness_check_enabled,
+    )
+
+    class StreamModel(nn.Module):
+        def __init__(self):
+            super().__init__()
+            self.projection = _make_fused_projection()
+            self.projection.weight = nn.Parameter(torch.zeros(1))
+            self.dense = nn.Parameter(torch.zeros(1))
+
+        def load_weights(self, weights):
+            assert not is_mtp_completeness_check_enabled()
+            loaded = set()
+            for name, _ in weights:
+                if name == "dense":
+                    loaded.add("dense")
+                else:
+                    self.projection._loaded_shards.add(name)
+                    loaded.add("projection.weight")
+            return loaded
+
+    model = StreamModel()
+    original = model.load_weights
+    guard = Step5WeightUpdateGuard(model, stage_to_cpu=stage_to_cpu)
+    for shard in sorted(model.projection.required_shard_ids()) + ["dense"]:
+        if shard != omit:
+            model.load_weights([(shard, torch.zeros(1))])
+    if omit is None:
+        guard.finish()
+        second = Step5WeightUpdateGuard(model)
+        with pytest.raises(ValueError, match="Incomplete Step5"):
+            second.finish()
+    else:
+        with pytest.raises(ValueError, match="Incomplete"):
+            guard.finish()
+    assert model.load_weights == original
+    assert is_mtp_completeness_check_enabled()
diff --git a/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py b/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
index c71ee30fb0..dbd67e32ed 100644
--- a/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
+++ b/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
@@ -44,7 +44,7 @@ def get_num_experts(hf_config) -> int:
       - ``n_routed_experts``: DeepSeek-V2/V3
       - ``num_local_experts``: Mixtral (older exports)
     """
-    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
+    for key in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
         val = getattr(hf_config, key, None)
         if val is not None:
             return val
~~~~

## Diff：MindSpeed-Agent 验证脚本

仓库：/home/yyy/MindSpeed-Agent

HEAD：51db859b55aa2a615d1264133e2e61404d0e42bb

~~~~diff
diff --git a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/dual_api.py b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/dual_api.py
index 294299a..c5ac284 100644
--- a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/dual_api.py
+++ b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/dual_api.py
@@ -10,7 +10,7 @@ ENDPOINTS = ("/v1/completions", "/v1/chat/completions")
 
 def validate_generation_output(endpoint: str, output: dict[str, Any]) -> None:
     """Require generated text and aligned token/logprob evidence."""
-    if not isinstance(output.get("text"), str) or not output["text"].strip():
+    if not isinstance(output.get("text"), str) or not output["text"]:
         raise RuntimeError(f"{endpoint} 返回空输出")
     tokens = output.get("tokens")
     token_logprobs = output.get("token_logprobs")
diff --git a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/online_feature_common.py b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/online_feature_common.py
index da57294..2869030 100644
--- a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/online_feature_common.py
+++ b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/common/online_feature_common.py
@@ -138,6 +138,7 @@ def parse_baseline(command: str) -> dict:
         "tokens": tokens,
         "model_index": model_index,
         "model": tokens[model_index],
+        "served_model_name": _flag_value(tokens, {"--served-model-name", "--served_model_name"}, "auto", str),
         "tensor_parallel_size": _flag_value(tokens, {"--tensor-parallel-size", "-tp"}, 1, int),
         "data_parallel_size": _flag_value(tokens, {"--data-parallel-size", "-dp"}, 1, int),
         "pipeline_parallel_size": _flag_value(tokens, {"--pipeline-parallel-size", "-pp"}, 1, int),
@@ -306,8 +307,8 @@ def _get_json(url: str, *, timeout: float):
 
 
 def _post_completion(base_url: str, prompt: str, *, max_tokens: int, timeout: float,
-                     endpoint: str = "/v1/completions") -> str:
-    payload = {"model": "auto", "max_tokens": max_tokens, "temperature": 0}
+                     endpoint: str = "/v1/completions", model_name: str = "auto") -> str:
+    payload = {"model": model_name, "max_tokens": max_tokens, "temperature": 0}
     if endpoint == "/v1/chat/completions":
         payload["messages"] = [{"role": "user", "content": prompt}]
     else:
@@ -333,8 +334,8 @@ def _post_completion(base_url: str, prompt: str, *, max_tokens: int, timeout: fl
                    else choice.get("text", ""))
     except (KeyError, IndexError, AttributeError, TypeError) as exc:
         raise RuntimeError(f"completion response has no choices[].text: {data!r}") from exc
-    if not text.strip() or "\ufffd" in text:
-        raise RuntimeError("completion text is empty or contains replacement characters")
+    if not text:
+        raise RuntimeError("completion text is empty")
     return text
 
 
@@ -530,7 +531,8 @@ def run_online_pair(
             ) as service:
                 outputs_by_endpoint = {
                     endpoint: [_post_completion(service.base_url, prompt, max_tokens=args.max_tokens,
-                                                timeout=args.request_timeout, endpoint=endpoint)
+                                                timeout=args.request_timeout, endpoint=endpoint,
+                                                model_name=baseline.get("served_model_name") or "auto")
                                for prompt in prompts]
                     for endpoint in API_ENDPOINTS
                 }
@@ -564,7 +566,8 @@ def run_online_pair(
         ) as service:
             feature_outputs_by_endpoint = {
                 endpoint: [_post_completion(service.base_url, prompt, max_tokens=args.max_tokens,
-                                            timeout=args.request_timeout, endpoint=endpoint)
+                                            timeout=args.request_timeout, endpoint=endpoint,
+                                            model_name=baseline.get("served_model_name") or "auto")
                            for prompt in prompts]
                 for endpoint in API_ENDPOINTS
             }
@@ -584,7 +587,7 @@ def run_online_pair(
             mismatches_by_endpoint[endpoint] = [
                 index for index, (before, after) in enumerate(
                     zip(baseline_by_endpoint[endpoint], feature_outputs_by_endpoint[endpoint])
-                ) if compare_outputs and before.strip() != after.strip()
+                ) if compare_outputs and before != after
             ]
         mismatches = mismatches_by_endpoint["/v1/completions"]
         result.update({
diff --git a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/lifecycle/test_sleep_mode.py b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/lifecycle/test_sleep_mode.py
index 2e29510..f5a4a20 100644
--- a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/lifecycle/test_sleep_mode.py
+++ b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/lifecycle/test_sleep_mode.py
@@ -126,7 +126,7 @@ def run_sleep_cycle(
         arguments.server_url, "POST", "/sleep?level=1", timeout=arguments.request_timeout,
     )
     request_json(
-        arguments.server_url, "POST", "/wake_up?tags=weights", timeout=arguments.request_timeout,
+        arguments.server_url, "POST", "/wake_up", timeout=arguments.request_timeout,
     )
     wait_until_awake(
         arguments.server_url, timeout=arguments.request_timeout,
diff --git a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/routing/test_moe_routing_replay.py b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/routing/test_moe_routing_replay.py
index 0c68e09..7340d13 100644
--- a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/routing/test_moe_routing_replay.py
+++ b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/routing/test_moe_routing_replay.py
@@ -98,6 +98,8 @@ _VALUE_FLAGS = {
     "--pipeline-parallel-size",
     "-pp",
     "--max-model-len",
+    "--max-num-seqs",
+    "--gpu-memory-utilization",
     "--block-size",
     "--port",
     "-p",
@@ -140,6 +142,11 @@ def parse_baseline(command: str) -> dict:
         "pipeline_parallel_size": 1,
         "enable_expert_parallel": False,
         "graph_mode": None,
+        "max_model_len": 4096,
+        "block_size": 16,
+        "max_num_seqs": None,
+        "gpu_memory_utilization": None,
+        "enable_chunked_prefill": False,
         "port": 8000,
         "host": "127.0.0.1",
         "extra": [],
@@ -184,6 +191,26 @@ def parse_baseline(command: str) -> dict:
             parsed["extra"].extend([tok, tokens[i + 1]])
             i += 2
             continue
+        if tok == "--max-model-len":
+            parsed["max_model_len"] = int(tokens[i + 1])
+            i += 2
+            continue
+        if tok == "--block-size":
+            parsed["block_size"] = int(tokens[i + 1])
+            i += 2
+            continue
+        if tok == "--max-num-seqs":
+            parsed["max_num_seqs"] = int(tokens[i + 1])
+            i += 2
+            continue
+        if tok == "--gpu-memory-utilization":
+            parsed["gpu_memory_utilization"] = float(tokens[i + 1])
+            i += 2
+            continue
+        if tok == "--enable-chunked-prefill":
+            parsed["enable_chunked_prefill"] = True
+            i += 1
+            continue
         if tok in ("--port", "-p"):
             parsed["port"] = int(tokens[i + 1])
             i += 2
@@ -226,8 +253,12 @@ def infer_expert_shape(model: str) -> tuple[int|None, int|None, int|None]:
     if not config_path.exists():
         raise ValueError(f"model config not found: {config_path}")
     config = json.loads(config_path.read_text(encoding="utf-8"))
-    layers = config.get("num_hidden_layers") or config.get("num_layers")
-    experts_per_token = (config.get("num_experts_per_tok") or config.get("num_selected_experts")
+    # Step5 keeps text-side hyperparameters under text_config.
+    tc = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
+    layers = tc.get("num_hidden_layers") or tc.get("num_layers") or config.get("num_hidden_layers") or config.get("num_layers")
+    experts_per_token = (tc.get("num_experts_per_tok") or tc.get("num_selected_experts")
+                         or tc.get("top_k") or tc.get("moe_top_k")
+                         or config.get("num_experts_per_tok") or config.get("num_selected_experts")
                          or config.get("top_k") or config.get("moe_top_k"))
     if layers is None or experts_per_token is None:
         raise ValueError("config.json lacks num_hidden_layers/num_layers or num_experts_per_tok/top_k")
@@ -260,12 +291,9 @@ def validate_routed_experts_array(re_array: np.ndarray, expected_shape: tuple[in
 
 
 def validate_output_text(text: str) -> None:
-    """Validate the generated text has no garble/replacement chars (AI 判定)."""
-    assert text and len(text.strip()) > 0, "generated output text is empty"
-    bad = ("\ufffd",)  # U+FFFD replacement character => decode/garble issue
-    for ch in bad:
-        assert ch not in text, f"generated output contains replacement char {ch!r} (garble)"
-    print(f"[OK] output text ok (len={len(text)}): {text[:80]!r}")
+    """Validate text presence; readability is not a reduced-model quality gate."""
+    assert isinstance(text, str) and text, "generated output text is empty"
+    print(f"[OK] output text present (len={len(text)}): {text[:80]!r}")
 
 
 # ---------------------------------------------------------------------------
@@ -299,6 +327,16 @@ def build_http_command(baseline: dict, feature_args: list[str], port: int) -> li
         cmd += ["--pipeline-parallel-size", str(baseline["pipeline_parallel_size"])]
     if baseline["enable_expert_parallel"]:
         cmd += ["--enable-expert-parallel"]
+    if baseline.get("max_model_len") is not None:
+        cmd += ["--max-model-len", str(baseline["max_model_len"])]
+    if baseline.get("block_size") is not None:
+        cmd += ["--block-size", str(baseline["block_size"])]
+    if baseline.get("max_num_seqs") is not None:
+        cmd += ["--max-num-seqs", str(baseline["max_num_seqs"])]
+    if baseline.get("gpu_memory_utilization") is not None:
+        cmd += ["--gpu-memory-utilization", str(baseline["gpu_memory_utilization"])]
+    if baseline.get("enable_chunked_prefill"):
+        cmd += ["--enable-chunked-prefill"]
     cmd.extend(feature_args)
     # Ensure the routing replays switch is present.
     if "--enable-return-routed-experts" not in cmd:
@@ -483,6 +521,10 @@ def verify_inproc(baseline: dict) -> None:
     from vllm import LLM, SamplingParams
     from vllm.sampling_params import RequestOutputKind
 
+    # Match the baseline execution mode: if the baseline command uses eager
+    # mode, do not force CUDAGraph capture (some models fail device-side during
+    # graph capture, e.g. Ascend device error 507035).
+    enforce_eager = "--enforce-eager" in baseline.get("extra", [])
     llm_kwargs = dict(
         model=baseline["model"],
         tensor_parallel_size=eff_tp,
@@ -491,15 +533,19 @@ def verify_inproc(baseline: dict) -> None:
         enable_expert_parallel=baseline["enable_expert_parallel"],
         enable_return_routed_experts=True,
         distributed_executor_backend="mp",
-        # The baseline may carry a graph-mode >= eager setting; pass it through
-        # verbatim if provided, otherwise use the skill's default.
-        compilation_config=(
-            json.loads(baseline["graph_mode"]) if baseline["graph_mode"] else {"cudagraph_mode": "FULL_DECODE_ONLY"}
-        ),
-        max_model_len=1024,
-        block_size=16,
-        enable_chunked_prefill=True,
+        enforce_eager=enforce_eager,
+        max_model_len=baseline["max_model_len"],
+        block_size=baseline["block_size"],
+        enable_chunked_prefill=baseline["enable_chunked_prefill"],
     )
+    if baseline["max_num_seqs"] is not None:
+        llm_kwargs["max_num_seqs"] = baseline["max_num_seqs"]
+    if baseline["gpu_memory_utilization"] is not None:
+        llm_kwargs["gpu_memory_utilization"] = baseline["gpu_memory_utilization"]
+    if not enforce_eager:
+        llm_kwargs["compilation_config"] = (
+            json.loads(baseline["graph_mode"]) if baseline["graph_mode"] else {"cudagraph_mode": "FULL_DECODE_ONLY"}
+        )
     print("[STEP 2 / inproc] LLM kwargs:", json.dumps(llm_kwargs, default=str))
 
     sampling_params = SamplingParams(
diff --git a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py
index 627f310..d55005a 100644
--- a/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py
+++ b/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py
@@ -210,6 +210,7 @@ def run_validation(arguments: argparse.Namespace, report: dict[str, Any]) -> Non
         report["checks"]["checkpoint_stream_transfer"] = True
         report["checks"]["init_start_update_finish"] = True
 
+        transfer_group.close()
         transfer_group = None
         gc.collect()
         torch.npu.synchronize()
~~~~

## Diff：新增 Step5WeightUpdateGuard

~~~~diff
diff --git a/vllm_ascend/patch/worker/step5_weight_update.py b/vllm_ascend/patch/worker/step5_weight_update.py
new file mode 100644
index 0000000000..0e0f851101
--- /dev/null
+++ b/vllm_ascend/patch/worker/step5_weight_update.py
@@ -0,0 +1,44 @@
+# SPDX-License-Identifier: Apache-2.0
+"""Validate a complete Step5 update across streamed checkpoint chunks."""
+
+
+class Step5WeightUpdateGuard:
+    def __init__(self, model, *, stage_to_cpu=False):
+        from vllm.models.step5.model import _reset_fused_qkv_indexer_load_state
+
+        self.model = model
+        self.stage_to_cpu = stage_to_cpu
+        self.original_load = model.load_weights
+        self.expected = set(dict(model.named_parameters()))
+        self.loaded = set()
+        _reset_fused_qkv_indexer_load_state(model)
+        model.load_weights = self.load_weights
+
+    def load_weights(self, weights):
+        from vllm.model_executor.model_loader.mtp_validation import (
+            disable_mtp_completeness_check,
+        )
+
+        # A chunk is partial; completeness is enforced at transaction finish.
+        with disable_mtp_completeness_check():
+            if self.stage_to_cpu:
+                weights = ((name, tensor.cpu()) for name, tensor in weights)
+            loaded = self.original_load(weights)
+        if loaded is None:
+            raise RuntimeError("Step5 loader did not return loaded parameter names")
+        self.loaded.update(loaded)
+        return loaded
+
+    def finish(self):
+        from vllm.models.step5.model import _validate_fused_qkv_indexer_weights
+
+        try:
+            missing = self.expected - self.loaded
+            if missing:
+                raise ValueError(f"Incomplete Step5 weight update: {sorted(missing)}")
+            _validate_fused_qkv_indexer_weights(self.model)
+        finally:
+            self.abort()
+
+    def abort(self):
+        self.model.load_weights = self.original_load
~~~~

## 验证结果与边界

- HCCL、IPC、Checkpoint 重载分别完成双接口 × 4 prompts，共 24 对更新前后输出完全一致，三个验证退出码均为 0。
- baseline、routing、SP、PP 和独立 sleep/wake 有通过记录。
- DCP、PCP 仍为 blocked，不能表述为完整清单全部通过。
- SP / PP 与基线 logprob 有差异，只判功能通过，不代表数值精度等价。
- IPC 正常退出时仍有共享 NPU tensor 释放警告。
- **这是现有 8 层减层模型功能验证，不代表完整模型精度通过。**

### 验收审计原文

~~~~json
{
  "scope": "8-layer functional verification, not full-model accuracy",
  "status": "passed",
  "weight_update_exact_pairs": {
    "checkpoint_reload": 8,
    "hccl_weight_update": 8,
    "npu_ipc_weight_update": 8
  },
  "features": {
    "baseline": {
      "dual_api_cases": 8,
      "status": "passed"
    },
    "routing_replay": {
      "dual_api_cases": 8,
      "status": "passed"
    },
    "sp": {
      "dual_api_cases": 8,
      "status": "passed"
    },
    "pp": {
      "dual_api_cases": 8,
      "status": "passed"
    }
  },
  "producer_and_reload_exit_codes": {
    "hccl": 0,
    "ipc": 0,
    "reload": 0
  }
}
~~~~

## 如何再次查看远端修改

```bash
ssh A5
docker exec -it yyy-cann92 bash
cd /home/yyy/vllm-ascend-main
git status --short
git diff HEAD --stat
git diff HEAD -- vllm_ascend/distributed/device_communicators/pyhccl.py vllm_ascend/distributed/weight_transfer/hccl_engine.py vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py vllm_ascend/worker/worker.py tests/ut/distributed/device_communicators/test_pyhccl.py tests/ut/distributed/weight_transfer/test_npu_ipc_engine.py
git diff --no-index -- /dev/null vllm_ascend/patch/worker/step5_weight_update.py
```

`git diff --no-index` 发现差异返回 1 是正常行为。

```bash
git -C /home/yyy/vllm-ascend-step4 diff HEAD -- tests/model_executor/model_loader/test_step5_fused_weight_loading.py vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
git -C /home/yyy/MindSpeed-Agent diff HEAD -- skills/inference-model-skills/deployment/model-rl-case-guard/scripts/
```

补充验证文件位于独立任务目录，不在上述仓库 diff 中：

```bash
cd /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808
less feature_probe.py
less routing_inproc.py
less schema_extension.py
less observe_update.py
less logs/launch_commands.txt
cat reports/acceptance_audit.json
```

## 远端最终报告快照

以下保留报告文本。内部相对路径指向远端任务目录，未复制到 Obsidian，请在远端查阅。

~~~~text
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
~~~~
