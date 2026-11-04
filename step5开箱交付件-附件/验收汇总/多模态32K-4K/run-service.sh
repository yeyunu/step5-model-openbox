#!/bin/bash
set -e
mount -o remount,size=16G /dev/shm
source /usr/local/Ascend/cann-9.2.0/opp/vendors/step5_ops_transformer/bin/set_env.bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export PYTHONPATH=/home/yyy/vllm-ascend-step4:/home/yyy/vllm-ascend-main:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_STEP4_ENABLE_FA3=0
export FLA_NPU_DISABLE_PTH=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /home/yyy/model_openbox/mm-32k-4k/logs
exec vllm serve /data/Step-5-Preview-BF16-8layer \
  --served-model-name step5-8layer \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --max-model-len 40960 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --port 8451 \
  > serve.log 2>&1
