# Step5 8-layer functional validation — NOT full-model accuracy

> 自动汇总的模型 RL 用例看护证据。原始日志、JSON 报告和启动命令应与本文件一起归档。

## Summary

- **Generated:** 2026-10-16T13:27:35+08:00
- **Model / checkpoint:** `/data/Step-5-Preview-BF16-8layer`
- **Result:** **11/13** report(s) passed

| Feature | Status | Evidence |
|---|---|---|
| `acceptance_audit` | ✅ **passed** | [`reports/acceptance_audit.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/acceptance_audit.json) |
| `baseline` | ✅ **passed** | [`reports/baseline.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/baseline.json) |
| `checkpoint_weight_update` | ✅ **passed** | [`reports/checkpoint_reload.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/checkpoint_reload.json) |
| `checkpoint_schema` | ✅ **passed** | [`reports/checkpoint_schema.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/checkpoint_schema.json) |
| `cp_dcp` | ⛔ **blocked** | [`reports/cp_dcp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_dcp.json) |
| `cp_pcp` | ⛔ **blocked** | [`reports/cp_pcp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_pcp.json) |
| `checkpoint_native_hccl_weight_update` | ✅ **passed** | [`reports/hccl_weight_update.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/hccl_weight_update.json) |
| `checkpoint_native_npu_ipc_weight_update` | ✅ **passed** | [`reports/npu_ipc_weight_update.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/npu_ipc_weight_update.json) |
| `pp` | ✅ **passed** | [`reports/pp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/pp.json) |
| `routing_inproc` | ✅ **passed** | [`reports/routing_inproc.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_inproc.json) |
| `routing_replay` | ✅ **passed** | [`reports/routing_replay.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_replay.json) |
| `sleep_mode` | ✅ **passed** | [`reports/sleep_mode.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/sleep_mode.json) |
| `sp` | ✅ **passed** | [`reports/sp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/sp.json) |

## Commands

The following commands were recorded in `logs/launch_commands.txt`:

```bash
2026-10-16T11:30:17+08:00 SERVICE baseline /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 
2026-10-16T11:32:10+08:00 SERVICE baseline /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 
2026-10-16T11:35:55+08:00 VERIFIER baseline /home/yyy/model_openbox/.venv/bin/python /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/baseline/run_baseline.py --server-url http://127.0.0.1:8410 --model step5-8layer --prompt The\ capital\ of\ France\ is --prompt Hello --max-tokens 8 --output /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/baseline.json 
2026-10-16T11:36:06+08:00 SERVICE reload /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --enable-sleep-mode 
2026-10-16T11:36:41+08:00 SERVICE reload /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --enable-sleep-mode 
2026-10-16T11:39:38+08:00 VERIFIER reload /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_reload_weights.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --updated-model /data/Step-5-Preview-BF16-8layer --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/checkpoint_reload.json 
2026-10-16T11:48:59+08:00 SERVICE reload /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --enable-sleep-mode 
2026-10-16T11:55:45+08:00 SERVICE reload /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --enable-sleep-mode 
2026-10-16T11:58:19+08:00 VERIFIER reload /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_reload_weights.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --updated-model /data/Step-5-Preview-BF16-8layer --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/checkpoint_reload.json 
2026-10-16T11:58:51+08:00 VERIFIER sleep /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_reload_weights.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --updated-model /data/Step-5-Preview-BF16-8layer --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/sleep_mode.json 
2026-10-16T11:59:17+08:00 SERVICE hccl /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"hccl\"\} 
2026-10-16T12:01:37+08:00 VERIFIER hccl /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --device 0 --master-address 127.0.0.1 --master-port 29591 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/hccl_weight_update.json 
2026-10-16T12:05:56+08:00 SERVICE hccl /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"hccl\"\} 
2026-10-16T12:09:36+08:00 VERIFIER hccl /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --device 0 --master-address 127.0.0.1 --master-port 29591 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/hccl_weight_update.json 
2026-10-16T12:11:16+08:00 SERVICE hccl /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"hccl\"\} 
2026-10-16T12:14:47+08:00 VERIFIER hccl /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --device 0 --master-address 127.0.0.1 --master-port 29591 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/hccl_weight_update.json 
2026-10-16T12:29:24+08:00 SERVICE hccl /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"hccl\"\} 
2026-10-16T12:31:44+08:00 VERIFIER hccl /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --device 0 --master-address 127.0.0.1 --master-port 29591 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/hccl_weight_update.json 
2026-10-16T12:34:33+08:00 SERVICE ipc /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"npu_ipc\"\} 
2026-10-16T12:37:03+08:00 VERIFIER ipc /home/yyy/model_openbox/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_npu_ipc_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --devices 0\,1\,2\,3\,4\,5\,6\,7 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/npu_ipc_weight_update.json 
2026-10-16T12:39:50+08:00 SERVICE ipc /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"npu_ipc\"\} 
2026-10-16T12:42:30+08:00 VERIFIER ipc /home/yyy/model_openbox/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_npu_ipc_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --devices 0\,1\,2\,3\,4\,5\,6\,7 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/npu_ipc_weight_update.json 
2026-10-16T12:47:11+08:00 SERVICE ipc /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"npu_ipc\"\} 
2026-10-16T12:51:11+08:00 VERIFIER ipc /home/yyy/model_openbox/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_npu_ipc_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --devices 0\,1\,2\,3\,4\,5\,6\,7 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/npu_ipc_weight_update.json 
2026-10-16T12:57:29+08:00 SERVICE ipc /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"npu_ipc\"\} 
2026-10-16T13:00:09+08:00 VERIFIER ipc /home/yyy/model_openbox/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_npu_ipc_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --devices 0\,1\,2\,3\,4\,5\,6\,7 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/npu_ipc_weight_update.json 
2026-10-16T13:04:25+08:00 SERVICE routing /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-return-routed-experts 
2026-10-16T13:08:35.273944 PROBE /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/feature_probe.py routing_replay
2026-10-16T13:08:49+08:00 SERVICE sp /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\,\"enable_flashcomm1\":true\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --all2all-backend allgather_reducescatter 
2026-10-16T13:11:29.914247 PROBE /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/feature_probe.py sp
2026-10-16T13:11:43+08:00 SERVICE pp /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --pipeline-parallel-size 2 
2026-10-16T13:15:26+08:00 SERVICE pp /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.4 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --pipeline-parallel-size 2 
2026-10-16T13:17:46.222695 PROBE /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/feature_probe.py pp
2026-10-16T13:17:59+08:00 SERVICE hccl /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 8 --no-enable-chunked-prefill --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --port 8410 --worker-extension-cls schema_extension.OpenboxWorkerExtension --enable-sleep-mode --weight-transfer-config \{\"backend\":\"hccl\"\} 
2026-10-16T13:20:09+08:00 VERIFIER hccl /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/observe_update.py /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --device 0 --master-address 127.0.0.1 --master-port 29591 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/hccl_weight_update.json 
2026-10-16T13:22:53+08:00 INPROC uv run --offline --no-project --python /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/routing_inproc.py
```

## Feature evidence

### 1. ✅ acceptance_audit — passed

Evidence file: [`reports/acceptance_audit.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/acceptance_audit.json)

### 2. ✅ baseline — passed

Evidence file: [`reports/baseline.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/baseline.json)

#### Checks

| Check | Result |
|---|---|
| `dual_api_tokens_and_finite_aligned_logprobs` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} |
| /v1/completions · prompt-2 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} |
| /v1/completions · prompt-3 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} |
| /v1/completions · prompt-4 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} |
| /v1/chat/completions · prompt-1 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} |
| /v1/chat/completions · prompt-2 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} |
| /v1/chat/completions · prompt-3 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} |
| /v1/chat/completions · prompt-4 | <missing> | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} |

### 3. ✅ checkpoint_weight_update — passed

Evidence file: [`reports/checkpoint_reload.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/checkpoint_reload.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `still_sleeping_after_weights_wake` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `awake_after_kv_wake` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_reload_weights.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --updated-model /data/Step-5-Preview-BF16-8layer --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/checkpoint_reload.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} |
| /v1/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} |
| /v1/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} |
| /v1/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} |
| /v1/chat/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} |
| /v1/chat/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} |
| /v1/chat/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} |
| /v1/chat/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} |

### 4. ✅ checkpoint_schema — passed

Evidence file: [`reports/checkpoint_schema.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/checkpoint_schema.json)

### 5. ⛔ cp_dcp — blocked

Evidence file: [`reports/cp_dcp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_dcp.json)

#### Checks

| Check | Result |
|---|---|
| `supported_by_model` | ❌ fail |

#### Endpoint checks

| Endpoint | Result |
|---|---|
- **Error:** Step5DSAAttention explicitly raises NotImplementedError for decode_context_parallel_size > 1.

### 6. ⛔ cp_pcp — blocked

Evidence file: [`reports/cp_pcp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/cp_pcp.json)
- **Error:** Current ModelRunner V1 rejects PCP > 1

### 7. ✅ checkpoint_native_hccl_weight_update — passed

Evidence file: [`reports/hccl_weight_update.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/hccl_weight_update.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_schema_preflight` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `still_sleeping_after_weights_wake` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `awake_after_kv_wake` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/data/Step-5-Preview-BF16-8layer","format":"safetensors","file_count":16,"total_file_bytes":75525918107,"tensor_count":807,"total_tensor_bytes":75525811072,"max_tensor_name":"model.layers.3.moe.down_proj.weight","max_tensor_bytes":4429185024,"manifest_sha256":"18a0d632eb4563a3f8c4d4b48ecb89aa8880efcb9fa6b7e1b5cbf675d9a6b008"}`

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_hccl_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --device 0 --master-address 127.0.0.1 --master-port 29591 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/hccl_weight_update.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.962145805358887,-6.499596118927002,-7.034921169281006,-6.874383926391602,-6.963172912597656,-6.761465549468994,-6.686456203460693,-6.696473598480225]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.962145805358887,-6.499596118927002,-7.034921169281006,-6.874383926391602,-6.963172912597656,-6.761465549468994,-6.686456203460693,-6.696473598480225]} |
| /v1/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} |
| /v1/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.1354660987854,-6.715044021606445,-7.00541353225708,-6.910356044769287,-6.871249198913574,-6.846577167510986,-6.698974609375,-6.7246623039245605]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.1354660987854,-6.715044021606445,-7.00541353225708,-6.910356044769287,-6.871249198913574,-6.846577167510986,-6.698974609375,-6.7246623039245605]} |
| /v1/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.306190013885498,-6.793038368225098,-6.795065402984619,-6.87112283706665,-6.820084095001221,-6.812725067138672,-6.738474369049072,-6.745595932006836]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.306190013885498,-6.793038368225098,-6.795065402984619,-6.87112283706665,-6.820084095001221,-6.812725067138672,-6.738474369049072,-6.745595932006836]} |
| /v1/chat/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.9774489402771,-6.9462809562683105,-6.888510704040527,-6.963003158569336,-7.037312030792236,-6.961303234100342,-6.925106525421143,-6.838779449462891]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.9774489402771,-6.9462809562683105,-6.888510704040527,-6.963003158569336,-7.037312030792236,-6.961303234100342,-6.925106525421143,-6.838779449462891]} |
| /v1/chat/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.064372539520264,-7.116147518157959,-6.935543537139893,-7.098240852355957,-7.080370903015137,-7.009914398193359,-6.957467079162598,-6.9208598136901855]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.064372539520264,-7.116147518157959,-6.935543537139893,-7.098240852355957,-7.080370903015137,-7.009914398193359,-6.957467079162598,-6.9208598136901855]} |
| /v1/chat/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079270839691162,-7.076261043548584,-6.986985683441162,-7.164831161499023,-7.099020957946777,-6.99400520324707,-7.009820461273193,-6.922610282897949]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079270839691162,-7.076261043548584,-6.986985683441162,-7.164831161499023,-7.099020957946777,-6.99400520324707,-7.009820461273193,-6.922610282897949]} |
| /v1/chat/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.0798726081848145,-7.063571929931641,-6.9931321144104,-6.97791051864624,-7.025484085083008,-7.024843692779541,-6.989664554595947,-6.922135353088379]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.0798726081848145,-7.063571929931641,-6.9931321144104,-6.97791051864624,-7.025484085083008,-7.024843692779541,-6.989664554595947,-6.922135353088379]} |

### 8. ✅ checkpoint_native_npu_ipc_weight_update — passed

Evidence file: [`reports/npu_ipc_weight_update.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/npu_ipc_weight_update.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_consistent` | ✅ pass |
| `checkpoint_schema_preflight` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `still_sleeping_after_weights_wake` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `awake_after_kv_wake` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/data/Step-5-Preview-BF16-8layer","format":"safetensors","file_count":16,"total_file_bytes":75525918107,"tensor_count":807,"total_tensor_bytes":75525811072,"max_tensor_name":"model.layers.3.moe.down_proj.weight","max_tensor_bytes":4429185024,"manifest_sha256":"18a0d632eb4563a3f8c4d4b48ecb89aa8880efcb9fa6b7e1b5cbf675d9a6b008"}`

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_npu_ipc_weight_update.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --checkpoint /data/Step-5-Preview-BF16-8layer --devices 0,1,2,3,4,5,6,7 --packed-buffer-mb 4352 --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/npu_ipc_weight_update.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} |
| /v1/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} |
| /v1/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} |
| /v1/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} |
| /v1/chat/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} |
| /v1/chat/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} |
| /v1/chat/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} |
| /v1/chat/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} |

### 9. ✅ pp — passed

Evidence file: [`reports/pp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/pp.json)

#### Checks

| Check | Result |
|---|---|
| `actual_topology` | ✅ pass |
| `dual_api_tokens_and_finite_aligned_logprobs` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.962266445159912,-6.499548435211182,-7.0343780517578125,-6.874410629272461,-6.9625372886657715,-6.745419502258301,-6.686190128326416,-6.6964826583862305]} |
| /v1/completions · prompt-2 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108839988708496,-6.663820266723633,-6.798262119293213,-6.886116027832031,-6.897974491119385,-6.699853897094727,-6.69162130355835,-6.654876232147217]} |
| /v1/completions · prompt-3 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135412693023682,-6.715174198150635,-7.003209590911865,-6.925683498382568,-6.843968868255615,-6.830397605895996,-6.6996331214904785,-6.7247443199157715]} |
| /v1/completions · prompt-4 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.305787563323975,-6.777224063873291,-6.794394493103027,-6.870743274688721,-6.902674198150635,-6.796250820159912,-6.737757205963135,-6.729531764984131]} |
| /v1/chat/completions · prompt-1 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977179050445557,-6.946038246154785,-6.888697624206543,-6.995782375335693,-7.037539005279541,-7.013794422149658,-6.944875240325928,-6.8387556076049805]} |
| /v1/chat/completions · prompt-2 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.0647196769714355,-7.098947048187256,-6.9357781410217285,-7.114171028137207,-7.045379638671875,-7.009909629821777,-6.957267761230469,-6.921023845672607]} |
| /v1/chat/completions · prompt-3 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079587936401367,-7.076168060302734,-6.986996173858643,-7.165093421936035,-7.098801612854004,-7.009552478790283,-7.01001501083374,-6.922550201416016]} |
| /v1/chat/completions · prompt-4 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079928398132324,-7.063552379608154,-6.975770473480225,-6.977787971496582,-7.025570392608643,-7.024796009063721,-6.989793300628662,-6.954407215118408]} |

### 10. ✅ routing_inproc — passed

Evidence file: [`reports/routing_inproc.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_inproc.json)
- **Note:** Offline LLM does not support single-process DP2; TP8 is supplementary to authoritative HTTP TP4/DP2.

#### Outputs

- {"prompt":"The future of AI is","text":"        ","tokens":[223,223,223,223,223,223,223,223],"logprobs":[-6.962179660797119,-6.499485969543457,-7.034923553466797,-6.875008583068848,-6.962724685668945,-6.745755195617676,-6.686246871948242,-6.696398735046387],"routing_shape":[13,8,8],"routing_min":0,"routing_max":344,"routing_path":"/home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/logs/routing-inproc-0.npy"}
- {"prompt":"Machine learning enables","text":"        ","tokens":[223,223,223,223,223,223,223,223],"logprobs":[-7.1088948249816895,-6.662750244140625,-6.799090385437012,-6.886401176452637,-6.8977131843566895,-6.69926118850708,-6.675463676452637,-6.638529300689697],"routing_shape":[11,8,8],"routing_min":0,"routing_max":350,"routing_path":"/home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/logs/routing-inproc-1.npy"}
- {"prompt":"Deep learning models can","text":"        ","tokens":[223,223,223,223,223,223,223,223],"logprobs":[-7.135501384735107,-6.7147297859191895,-7.005691051483154,-6.910974502563477,-6.871582984924316,-6.830171585083008,-6.698953628540039,-6.709001541137695],"routing_shape":[12,8,8],"routing_min":0,"routing_max":350,"routing_path":"/home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/logs/routing-inproc-2.npy"}
- {"prompt":"Step5 sparse attention is","text":"        ","tokens":[223,223,223,223,223,223,223,223],"logprobs":[-7.321918964385986,-6.792849540710449,-6.842600345611572,-6.8866286277771,-6.819355487823486,-6.796754837036133,-6.73858118057251,-6.745696544647217],"routing_shape":[13,8,8],"routing_min":0,"routing_max":351,"routing_path":"/home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/logs/routing-inproc-3.npy"}

### 11. ✅ routing_replay — passed

Evidence file: [`reports/routing_replay.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/routing_replay.json)

#### Checks

| Check | Result |
|---|---|
| `actual_topology` | ✅ pass |
| `dual_api_tokens_and_finite_aligned_logprobs` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} |
| /v1/completions · prompt-2 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} |
| /v1/completions · prompt-3 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} |
| /v1/completions · prompt-4 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} |
| /v1/chat/completions · prompt-1 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} |
| /v1/chat/completions · prompt-2 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} |
| /v1/chat/completions · prompt-3 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} |
| /v1/chat/completions · prompt-4 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} |

### 12. ✅ sleep_mode — passed

Evidence file: [`reports/sleep_mode.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/sleep_mode.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `still_sleeping_after_weights_wake` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `awake_after_kv_wake` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts/weight-update/test_reload_weights.py --server-url http://127.0.0.1:8410 --served-model-name step5-8layer --prompts-json /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/prompts.json --max-tokens 8 --timeout 600 --updated-model /data/Step-5-Preview-BF16-8layer --report /home/yyy/model_openbox/step5-preview-bf16-8layer-resume-20261016-112808/reports/sleep_mode.json
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} |
| /v1/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} |
| /v1/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} |
| /v1/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} |
| /v1/chat/completions · The future of AI is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} |
| /v1/chat/completions · Machine learning enables | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} |
| /v1/chat/completions · Deep learning models can | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} |
| /v1/chat/completions · Step5 sparse attention is | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} |

### 13. ✅ sp — passed

Evidence file: [`reports/sp.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E4%B8%93%E9%A1%B9%E9%AA%8C%E8%AF%81/step5-preview-bf16-8layer-resume/reports/sp.json)

#### Checks

| Check | Result |
|---|---|
| `actual_topology` | ✅ pass |
| `dual_api_tokens_and_finite_aligned_logprobs` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · prompt-1 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.97805118560791,-6.499545097351074,-7.034548282623291,-6.87445592880249,-6.962906360626221,-6.76118803024292,-6.685983657836914,-6.696591854095459]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.96214485168457,-6.499330520629883,-7.034553527832031,-6.874778747558594,-6.963194847106934,-6.761250019073486,-6.685910701751709,-6.69623327255249]} |
| /v1/completions · prompt-2 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.108782768249512,-6.6469926834106445,-6.799182891845703,-6.886170864105225,-6.897500514984131,-6.699454307556152,-6.6755690574646,-6.638632774353027]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.109100818634033,-6.6469645500183105,-6.799075603485107,-6.886317729949951,-6.897873401641846,-6.69941520690918,-6.675726890563965,-6.63857364654541]} |
| /v1/completions · prompt-3 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.135400772094727,-6.7148566246032715,-7.003444671630859,-6.926614284515381,-6.84407377243042,-6.830370903015137,-6.6991400718688965,-6.708800792694092]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.13532018661499,-6.714871406555176,-7.003162384033203,-6.926259517669678,-6.843847751617432,-6.83039665222168,-6.699094295501709,-6.724560737609863]} |
| /v1/completions · prompt-4 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322119235992432,-6.792964458465576,-6.794940948486328,-6.871264934539795,-6.819687843322754,-6.812644004821777,-6.738521099090576,-6.729896545410156]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.322019100189209,-6.793084144592285,-6.795295715332031,-6.871230125427246,-6.820083141326904,-6.8128862380981445,-6.738626003265381,-6.745397567749023]} |
| /v1/chat/completions · prompt-1 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977429389953613,-6.9459357261657715,-6.888690948486328,-6.962892532348633,-7.0215744972229,-6.961460590362549,-6.944464683532715,-6.8385701179504395]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-6.977322101593018,-6.946080684661865,-6.888639450073242,-6.962801456451416,-7.053666591644287,-6.91329288482666,-6.944261074066162,-6.8386125564575195]} |
| /v1/chat/completions · prompt-2 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.06466817855835,-7.116052150726318,-6.935471534729004,-7.098372936248779,-7.096277236938477,-7.0099406242370605,-6.957465171813965,-6.920825481414795]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.064902305603027,-7.099252223968506,-6.968242645263672,-7.098554611206055,-7.045592784881592,-7.010050296783447,-6.957354545593262,-6.920871257781982]} |
| /v1/chat/completions · prompt-3 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079222679138184,-7.076145172119141,-6.987241744995117,-7.165285587310791,-7.098609924316406,-7.009646415710449,-7.0100202560424805,-6.9228081703186035]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.079510688781738,-7.076260566711426,-6.986999988555908,-7.164897918701172,-7.09878396987915,-7.009642601013184,-7.0098724365234375,-6.922672271728516]} |
| /v1/chat/completions · prompt-4 | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.095804691314697,-7.063374042510986,-6.993111610412598,-6.977956295013428,-7.0254926681518555,-7.024902820587158,-6.989711284637451,-6.9220805168151855]} | {"text":"        ","tokens":[" "," "," "," "," "," "," "," "],"token_logprobs":[-7.080124378204346,-7.079212665557861,-6.993232727050781,-6.977945327758789,-7.025472640991211,-7.0249528884887695,-6.989897727966309,-6.922182083129883]} |

## Interpretation

- `passed` means the recorded checks completed; it is not a substitute for customer precision acceptance.
- `failed`, `blocked` and `not_run` must be reviewed together with the original logs and environment evidence.
- Text output is runtime evidence only; it does not prove numerical equivalence.

