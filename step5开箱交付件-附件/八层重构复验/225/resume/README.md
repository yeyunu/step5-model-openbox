# Step5 225 续测：权重更新与生命周期验证

> 自动汇总的模型 RL 用例看护证据。原始日志、JSON 报告和启动命令应与本文件一起归档。

## Summary

- **Generated:** 2026-11-02T23:42:13+08:00
- **Model / checkpoint:** `step5-head-tail8`
- **Result:** **11/11** report(s) passed

| Feature | Status | Evidence |
|---|---|---|
| `checkpoint_native_hccl_weight_update` | ✅ **passed** | [`results/resume-hccl-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-hccl/report.json) |
| `checkpoint_native_npu_ipc_weight_update` | ✅ **passed** | [`results/resume-ipc-round1-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round1/report.json) |
| `checkpoint_native_npu_ipc_weight_update` | ✅ **passed** | [`results/resume-ipc-round2-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round2/report.json) |
| `checkpoint_weight_update` | ✅ **passed** | [`results/resume-reload-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-reload/report.json) |
| `strict_sleep1` | ✅ **passed** | [`results/resume-sleep1-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep1/report.json) |
| `checkpoint_weight_update` | ✅ **passed** | [`results/resume-sleep2-10-20/report-cycle1.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle1.json) |
| `checkpoint_weight_update` | ✅ **passed** | [`results/resume-sleep2-10-20/report-cycle2.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle2.json) |
| `checkpoint_weight_update` | ✅ **passed** | [`results/resume-sleep2-10-20/report-cycle3.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle3.json) |
| `checkpoint_weight_update` | ✅ **passed** | [`results/resume-sleep2-10-20/report-cycle4.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle4.json) |
| `checkpoint_weight_update` | ✅ **passed** | [`results/resume-sleep2-10-20/report-cycle5.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle5.json) |
| `strict_sleep2_checkpoint_reload` | ✅ **passed** | [`results/resume-sleep2-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report.json) |

## Commands

The following commands were recorded in `logs/launch_commands.txt`:

```bash
# 本次 225 实际命令证据汇总。归档标签 10.20；执行日期以日志为准。
# Container: step5-recheck-225-10.20
# SERVICE 环境：source /home/yyy/model_openbox/reduced-recheck-225-10.20/env.sh

# SERVICE resume-hccl-10-20
/home/yyy/model_openbox/upstream-retest-20261030/uv run --offline --no-project --python /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --served-model-name step5-head-tail8 --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --enable-sleep-mode --max-model-len 2048 --max-num-seqs 4 --block-size 8 --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --host 127.0.0.1 --port 8940 --no-enable-chunked-prefill --max-num-batched-tokens 2048 --weight-transfer-config \{\"backend\":\"hccl\"\} --worker-extension-cls strict_mechanism_extension.StrictMechanismExtension

# SERVICE resume-ipc-10-20
/home/yyy/model_openbox/upstream-retest-20261030/uv run --offline --no-project --python /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/.venv/bin/python -m vllm.entrypoints.cli.main serve /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --served-model-name step5-head-tail8 --tensor-parallel-size 4 --data-parallel-size 2 --enable-expert-parallel --enable-sleep-mode --max-model-len 2048 --max-num-seqs 4 --block-size 8 --gpu-memory-utilization 0.3 --load-format auto --trust-remote-code --additional-config \{\"rl_config\":\{\"enabled\":true\}\} --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"cudagraph_capture_sizes\":\[1\,2\,4\]\} --host 127.0.0.1 --port 8940 --no-enable-chunked-prefill --max-num-batched-tokens 2048 --weight-transfer-config \{\"backend\":\"npu_ipc\"\} --worker-extension-cls strict_mechanism_extension.StrictMechanismExtension

# VERIFIER resume-hccl-10-20
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode hccl --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-hccl-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-hccl-10-20/events.jsonl --device 0 --master-address 127.0.0.1 --master-port 29681 --packed-buffer-mb 4352 --packed-num-buffers 1

# VERIFIER resume-ipc-round1-10-20
/home/yyy/model_openbox/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode ipc --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 8 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round1-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round1-10-20/events.jsonl --devices 0\,1\,2\,3\,4\,5\,6\,7 --packed-buffer-mb 4352

# VERIFIER resume-ipc-round2-10-20
/home/yyy/model_openbox/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode ipc --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 8 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round2-10-20/events.jsonl --devices 0\,1\,2\,3\,4\,5\,6\,7 --packed-buffer-mb 4352

# VERIFIER resume-reload-10-20
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode reload --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-reload-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-reload-10-20/events.jsonl

# VERIFIER resume-sleep1-10-20
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode sleep1 --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep1-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep1-10-20/events.jsonl

# VERIFIER resume-sleep2-10-20
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode sleep2 --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/events.jsonl

# STOP/CLEANUP resume-hccl-10-20
docker exec step5-recheck-225-10.20 /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/stop_service.py resume-hccl-10-20

# STOP/CLEANUP resume-ipc-10-20
docker exec step5-recheck-225-10.20 /home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/stop_service.py resume-ipc-10-20
```

## Feature evidence

### 1. ✅ checkpoint_native_hccl_weight_update — passed

Evidence file: [`results/resume-hccl-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-hccl/report.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `init_start_update_finish` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8","format":"safetensors","file_count":13,"total_file_bytes":62537461456,"tensor_count":802,"total_tensor_bytes":62537360896,"max_tensor_name":"model.layers.3.moe.down_proj.weight","max_tensor_bytes":4429185024,"manifest_sha256":"a4ad7b5df66ad7d5afeb2e6fa9b9596e9e41dd218a7717b3726e020b9b1db7bf"}`

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode hccl --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-hccl-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-hccl-10-20/events.jsonl --device 0 --master-address 127.0.0.1 --master-port 29681 --packed-buffer-mb 4352 --packed-num-buffers 1
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 2. ✅ checkpoint_native_npu_ipc_weight_update — passed

Evidence file: [`results/resume-ipc-round1-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round1/report.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `checkpoint_manifest_consistent` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8","format":"safetensors","file_count":13,"total_file_bytes":62537461456,"tensor_count":802,"total_tensor_bytes":62537360896,"max_tensor_name":"model.layers.3.moe.down_proj.weight","max_tensor_bytes":4429185024,"manifest_sha256":"a4ad7b5df66ad7d5afeb2e6fa9b9596e9e41dd218a7717b3726e020b9b1db7bf"}`

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode ipc --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 8 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round1-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round1-10-20/events.jsonl --devices 0,1,2,3,4,5,6,7 --packed-buffer-mb 4352
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-4.975212097167969,-5.163727760314941,-5.892099857330322,-6.164156913757324,-6.2205352783203125,-6.137774467468262,-5.317843914031982,-5.624701976776123]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-4.975212097167969,-5.163727760314941,-5.892099857330322,-6.164156913757324,-6.2205352783203125,-6.137774467468262,-5.317843914031982,-5.624701976776123]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人VirProp LennonProp Leone Leone Leone","tokens":["圣人","Vir","Prop"," Lennon","Prop"," Leone"," Leone"," Leone"],"token_logprobs":[-5.587474346160889,-5.904608726501465,-6.179454326629639,-6.07584285736084,-6.201801300048828,-5.685492515563965,-5.919314384460449,-5.6550679206848145]} | {"text":"圣人VirProp LennonProp Leone Leone Leone","tokens":["圣人","Vir","Prop"," Lennon","Prop"," Leone"," Leone"," Leone"],"token_logprobs":[-5.587474346160889,-5.904608726501465,-6.179454326629639,-6.07584285736084,-6.201801300048828,-5.685492515563965,-5.919314384460449,-5.6550679206848145]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Feldman Nonlinear","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Feldman"," Nonlinear"],"token_logprobs":[-4.74606466293335,-5.425720691680908,-5.9161529541015625,-5.658636569976807,-5.83663272857666,-5.919033050537109,-6.087036609649658,-5.695261478424072]} | {"text":"Vir Galile Genet Genet Genet Genet Feldman Nonlinear","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Feldman"," Nonlinear"],"token_logprobs":[-4.74606466293335,-5.425720691680908,-5.9161529541015625,-5.658636569976807,-5.83663272857666,-5.919033050537109,-6.087036609649658,-5.695261478424072]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.7112953662872314,-4.84763240814209,-4.6085100173950195,-4.752187252044678,-5.089812755584717,-4.850607872009277,-5.3611297607421875,-5.014275074005127]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.7112953662872314,-4.84763240814209,-4.6085100173950195,-4.752187252044678,-5.089812755584717,-4.850607872009277,-5.3611297607421875,-5.014275074005127]} |
| /v1/chat/completions · 你好 | {"text":" Argument Genet Genet Propagation Leoneynie Leone Wilder","tokens":[" Argument"," Genet"," Genet"," Propagation"," Leone","ynie"," Leone"," Wilder"],"token_logprobs":[-3.5158894062042236,-5.329631328582764,-5.776121616363525,-5.641989707946777,-5.3791584968566895,-5.13519811630249,-4.671998500823975,-4.860509872436523]} | {"text":" Argument Genet Genet Propagation Leoneynie Leone Wilder","tokens":[" Argument"," Genet"," Genet"," Propagation"," Leone","ynie"," Leone"," Wilder"],"token_logprobs":[-3.5158894062042236,-5.329631328582764,-5.776121616363525,-5.641989707946777,-5.3791584968566895,-5.13519811630249,-4.671998500823975,-4.860509872436523]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.755350112915039,-4.9692158699035645,-5.186671733856201,-4.921706199645996,-5.0431108474731445,-4.689030647277832,-5.189400672912598,-5.350473403930664]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.755350112915039,-4.9692158699035645,-5.186671733856201,-4.921706199645996,-5.0431108474731445,-4.689030647277832,-5.189400672912598,-5.350473403930664]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Reason Leone Leone Leone Leone Leone Moral面无","tokens":[" Reason"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-4.045271873474121,-5.140981674194336,-5.456576347351074,-5.561172962188721,-4.684103012084961,-5.20346736907959,-5.437243461608887,-5.3606438636779785]} | {"text":" Reason Leone Leone Leone Leone Leone Moral面无","tokens":[" Reason"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-4.045271873474121,-5.140981674194336,-5.456576347351074,-5.561172962188721,-4.684103012084961,-5.20346736907959,-5.437243461608887,-5.3606438636779785]} |

### 3. ✅ checkpoint_native_npu_ipc_weight_update — passed

Evidence file: [`results/resume-ipc-round2-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-ipc-round2/report.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `checkpoint_manifest_consistent` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_stream_transfer` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |
- **Checkpoint:** `{"path":"/home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8","format":"safetensors","file_count":13,"total_file_bytes":62537461456,"tensor_count":802,"total_tensor_bytes":62537360896,"max_tensor_name":"model.layers.3.moe.down_proj.weight","max_tensor_bytes":4429185024,"manifest_sha256":"a4ad7b5df66ad7d5afeb2e6fa9b9596e9e41dd218a7717b3726e020b9b1db7bf"}`

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode ipc --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 8 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-ipc-round2-10-20/events.jsonl --devices 0,1,2,3,4,5,6,7 --packed-buffer-mb 4352
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-4.975212097167969,-5.163727760314941,-5.892099857330322,-6.164156913757324,-6.2205352783203125,-6.137774467468262,-5.317843914031982,-5.624701976776123]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-4.975212097167969,-5.163727760314941,-5.892099857330322,-6.164156913757324,-6.2205352783203125,-6.137774467468262,-5.317843914031982,-5.624701976776123]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人VirProp LennonProp Leone Leone Leone","tokens":["圣人","Vir","Prop"," Lennon","Prop"," Leone"," Leone"," Leone"],"token_logprobs":[-5.587474346160889,-5.904608726501465,-6.179454326629639,-6.07584285736084,-6.201801300048828,-5.685492515563965,-5.919314384460449,-5.6550679206848145]} | {"text":"圣人VirProp LennonProp Leone Leone Leone","tokens":["圣人","Vir","Prop"," Lennon","Prop"," Leone"," Leone"," Leone"],"token_logprobs":[-5.587474346160889,-5.904608726501465,-6.179454326629639,-6.07584285736084,-6.201801300048828,-5.685492515563965,-5.919314384460449,-5.6550679206848145]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Feldman Nonlinear","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Feldman"," Nonlinear"],"token_logprobs":[-4.74606466293335,-5.425720691680908,-5.9161529541015625,-5.658636569976807,-5.83663272857666,-5.919033050537109,-6.087036609649658,-5.695261478424072]} | {"text":"Vir Galile Genet Genet Genet Genet Feldman Nonlinear","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Feldman"," Nonlinear"],"token_logprobs":[-4.74606466293335,-5.425720691680908,-5.9161529541015625,-5.658636569976807,-5.83663272857666,-5.919033050537109,-6.087036609649658,-5.695261478424072]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.7112953662872314,-4.84763240814209,-4.6085100173950195,-4.752187252044678,-5.089812755584717,-4.850607872009277,-5.3611297607421875,-5.014275074005127]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.7112953662872314,-4.84763240814209,-4.6085100173950195,-4.752187252044678,-5.089812755584717,-4.850607872009277,-5.3611297607421875,-5.014275074005127]} |
| /v1/chat/completions · 你好 | {"text":" Argument Genet Genet Propagation Leoneynie Leone Wilder","tokens":[" Argument"," Genet"," Genet"," Propagation"," Leone","ynie"," Leone"," Wilder"],"token_logprobs":[-3.5158894062042236,-5.329631328582764,-5.776121616363525,-5.641989707946777,-5.3791584968566895,-5.13519811630249,-4.671998500823975,-4.860509872436523]} | {"text":" Argument Genet Genet Propagation Leoneynie Leone Wilder","tokens":[" Argument"," Genet"," Genet"," Propagation"," Leone","ynie"," Leone"," Wilder"],"token_logprobs":[-3.5158894062042236,-5.329631328582764,-5.776121616363525,-5.641989707946777,-5.3791584968566895,-5.13519811630249,-4.671998500823975,-4.860509872436523]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.755350112915039,-4.9692158699035645,-5.186671733856201,-4.921706199645996,-5.0431108474731445,-4.689030647277832,-5.189400672912598,-5.350473403930664]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.755350112915039,-4.9692158699035645,-5.186671733856201,-4.921706199645996,-5.0431108474731445,-4.689030647277832,-5.189400672912598,-5.350473403930664]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Reason Leone Leone Leone Leone Leone Moral面无","tokens":[" Reason"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-4.045271873474121,-5.140981674194336,-5.456576347351074,-5.561172962188721,-4.684103012084961,-5.20346736907959,-5.437243461608887,-5.3606438636779785]} | {"text":" Reason Leone Leone Leone Leone Leone Moral面无","tokens":[" Reason"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-4.045271873474121,-5.140981674194336,-5.456576347351074,-5.561172962188721,-4.684103012084961,-5.20346736907959,-5.437243461608887,-5.3606438636779785]} |

### 4. ✅ checkpoint_weight_update — passed

Evidence file: [`results/resume-reload-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-reload/report.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode reload --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-reload-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-reload-10-20/events.jsonl
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 5. ✅ strict_sleep1 — passed

Evidence file: [`results/resume-sleep1-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep1/report.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `wake_weights` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |
| `observed_worker_memory_growth_within_5pct` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 6. ✅ checkpoint_weight_update — passed

Evidence file: [`results/resume-sleep2-10-20/report-cycle1.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle1.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode sleep2 --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/events.jsonl
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 7. ✅ checkpoint_weight_update — passed

Evidence file: [`results/resume-sleep2-10-20/report-cycle2.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle2.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode sleep2 --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/events.jsonl
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 8. ✅ checkpoint_weight_update — passed

Evidence file: [`results/resume-sleep2-10-20/report-cycle3.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle3.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode sleep2 --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/events.jsonl
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 9. ✅ checkpoint_weight_update — passed

Evidence file: [`results/resume-sleep2-10-20/report-cycle4.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle4.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode sleep2 --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/events.jsonl
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 10. ✅ checkpoint_weight_update — passed

Evidence file: [`results/resume-sleep2-10-20/report-cycle5.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report-cycle5.json)

#### Checks

| Check | Result |
|---|---|
| `both_generation_endpoints_baseline` | ✅ pass |
| `checkpoint_manifest_all_workers` | ✅ pass |
| `sleep_level2` | ✅ pass |
| `wake_weights` | ✅ pass |
| `checkpoint_reload` | ✅ pass |
| `wake_kv_cache` | ✅ pass |
| `both_generation_endpoints_post_update` | ✅ pass |
| `post_update_available` | ✅ pass |
| `outputs_identical` | ✅ pass |
| `sampled_parameters_identical` | ✅ pass |

#### Endpoint checks

| Endpoint | Result |
|---|---|
| `/v1/completions` | ✅ pass |
| `/v1/chat/completions` | ✅ pass |

<details>
<summary>Verification command</summary>

```text
/home/yyy/model_openbox/.venv/bin/python /home/yyy/model_openbox/reduced-recheck-225-10.20/strict_mechanism_runner.py --mode sleep2 --skill-scripts /home/yyy/MindSpeed-Agent/skills/inference-model-skills/deployment/model-rl-case-guard/scripts --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037 --server-url http://127.0.0.1:8940 --model step5-head-tail8 --checkpoint /home/yyy/model_openbox/reduced-recheck-225-10.20/checkpoints/head-tail8 --expected-workers 4 --prompts-json /home/yyy/model_openbox/reduced-recheck-225-10.20/prompts.json --max-tokens 8 --timeout 1200 --cycles 5 --report /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/report.json --evidence /home/yyy/model_openbox/reduced-recheck-225-10.20/results/resume-sleep2-10-20/events.jsonl
```

</details>

#### Output comparison

| Endpoint / prompt | Baseline | After update |
|---|---|---|
| /v1/completions · The future of AI is | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} | {"text":" PropagationVirVir Nonlinear Nonlinear Zad Serious Genet","tokens":[" Propagation","Vir","Vir"," Nonlinear"," Nonlinear"," Zad"," Serious"," Genet"],"token_logprobs":[-5.020313739776611,-5.217005729675293,-5.945528030395508,-6.23098611831665,-6.223852634429932,-6.204017639160156,-5.3594207763671875,-5.671105861663818]} |
| /v1/completions · 你好 | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} | {"text":" Syllabus Bryant Syllabus Bryant Zad Siegel Siegel Siegel","tokens":[" Syllabus"," Bryant"," Syllabus"," Bryant"," Zad"," Siegel"," Siegel"," Siegel"],"token_logprobs":[-6.335625648498535,-6.586969375610352,-6.072412014007568,-6.478002548217773,-5.507026195526123,-6.438643932342529,-5.553314208984375,-5.879095077514648]} |
| /v1/completions · 2 + 2 = | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} | {"text":"圣人Vir LeoneProp Leone Leone Leone Leone","tokens":["圣人","Vir"," Leone","Prop"," Leone"," Leone"," Leone"," Leone"],"token_logprobs":[-5.62786865234375,-5.868325710296631,-6.129631042480469,-6.250585079193115,-5.78883171081543,-5.7738165855407715,-5.565212249755859,-5.413992881774902]} |
| /v1/completions · What is the capital of France? | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} | {"text":"Vir Galile Genet Genet Genet Genet Nonlinear Genet","tokens":["Vir"," Galile"," Genet"," Genet"," Genet"," Genet"," Nonlinear"," Genet"],"token_logprobs":[-4.8449387550354,-5.39595365524292,-6.015028953552246,-5.632374286651611,-5.530783653259277,-5.872467041015625,-6.12302303314209,-5.78249979019165]} |
| /v1/chat/completions · The future of AI is | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} | {"text":" Moderate Serious Leone Leone Leone Leone Leone Serious","tokens":[" Moderate"," Serious"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.678833484649658,-4.813478469848633,-4.59259033203125,-4.781153202056885,-4.786850929260254,-4.856502056121826,-5.411288261413574,-5.050071716308594]} |
| /v1/chat/completions · 你好 | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} | {"text":" Qualitative Genet Genetynieynie Leoneynie Leone","tokens":[" Qualitative"," Genet"," Genet","ynie","ynie"," Leone","ynie"," Leone"],"token_logprobs":[-3.8748161792755127,-5.359394550323486,-5.810780048370361,-5.522716522216797,-5.38413143157959,-5.0877299308776855,-4.844022274017334,-4.75746488571167]} |
| /v1/chat/completions · 2 + 2 = | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} | {"text":" Intermediate Leone Leone Leone Leone Leone Leone Serious","tokens":[" Intermediate"," Leone"," Leone"," Leone"," Leone"," Leone"," Leone"," Serious"],"token_logprobs":[-3.75056791305542,-5.019367694854736,-5.218287467956543,-5.009299278259277,-4.983262538909912,-4.711666107177734,-5.21907377243042,-5.348895072937012]} |
| /v1/chat/completions · What is the capital of France? | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} | {"text":" Responsible Leone Leone Leone Leone Leone Moral面无","tokens":[" Responsible"," Leone"," Leone"," Leone"," Leone"," Leone"," Moral","面无"],"token_logprobs":[-3.469139337539673,-5.288978576660156,-4.678061485290527,-5.393932819366455,-4.833685398101807,-5.232358455657959,-5.411604404449463,-5.28941535949707]} |

### 11. ✅ strict_sleep2_checkpoint_reload — passed

Evidence file: [`results/resume-sleep2-10-20/report.json`](https://github.com/yeyunu/step5-model-openbox/blob/main/step5%E5%BC%80%E7%AE%B1%E4%BA%A4%E4%BB%98%E4%BB%B6-%E9%99%84%E4%BB%B6/%E5%85%AB%E5%B1%82%E9%87%8D%E6%9E%84%E5%A4%8D%E9%AA%8C/225/resume/results/resume-sleep2/report.json)

## Interpretation

- `passed` means the recorded checks completed; it is not a substitute for customer precision acceptance.
- `failed`, `blocked` and `not_run` must be reviewed together with the original logs and environment evidence.
- Text output is runtime evidence only; it does not prove numerical equivalence.

