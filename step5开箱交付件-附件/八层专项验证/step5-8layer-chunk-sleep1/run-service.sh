#!/bin/bash
set -e
W=$(dirname "$(readlink -f "$0")")
MODE=${1:?mode}
source /usr/local/Ascend/cann-9.2.0/opp/vendors/step5_ops_transformer/bin/set_env.bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export PYTHONPATH="$W":/home/yyy/vllm-ascend-step4:/home/yyy/vllm-ascend-main:${PYTHONPATH:-}
export PATH=/home/yyy/model_openbox/.venv/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_STEP4_ENABLE_FA3=0 FLA_NPU_DISABLE_PTH=1
export VLLM_SERVER_DEV_MODE=1 VLLM_ASCEND_ENABLE_NZ=0
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
V=/home/yyy/model_openbox/.venv/bin/python
TP=4
DP=2
UTIL=0.4
BATCH=2048
CHUNK=--no-enable-chunked-prefill
PORT=8410
ADDITIONAL='{"rl_config":{"enabled":true}}'
EXTRA=(--worker-extension-cls schema_extension.OpenboxWorkerExtension)
case "$MODE" in
baseline) ;;
sleep1) EXTRA+=(--enable-sleep-mode);;
chunk) BATCH=512; CHUNK=--enable-chunked-prefill;;
routing) EXTRA+=(--enable-return-routed-experts);;
sp) ADDITIONAL='{"rl_config":{"enabled":true},"enable_flashcomm1":true}'; EXTRA+=(--all2all-backend allgather_reducescatter);;
pp) TP=4; DP=1; EXTRA+=(--pipeline-parallel-size 2);;
reload) EXTRA+=(--enable-sleep-mode);;
ipc) EXTRA+=(--enable-sleep-mode --weight-transfer-config '{"backend":"npu_ipc"}'); export VLLM_ALLOW_INSECURE_SERIALIZATION=1; UTIL=0.3;;
hccl) DP=1; export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3; UTIL=0.3; EXTRA+=(--enable-sleep-mode --weight-transfer-config '{"backend":"hccl"}');;
*) echo "Unsupported mode $MODE"; exit 2;;
esac
CMD=("$V" -m vllm.entrypoints.cli.main serve /data/Step-5-Preview-BF16-8layer --served-model-name step5-8layer --tensor-parallel-size "$TP" --data-parallel-size "$DP" --enable-expert-parallel --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens "$BATCH" --block-size 8 "$CHUNK" --gpu-memory-utilization "$UTIL" --load-format auto --trust-remote-code --additional-config "$ADDITIONAL" --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' --port "$PORT" "${EXTRA[@]}")
{
printf '%s SERVICE %s ' "$(date -Is)" "$MODE"
printf '%q ' "${CMD[@]}"
printf '\n'
} >> "$W/logs/launch_commands.txt"
printf '%q ' "${CMD[@]}"
printf '\n'
cd "$W"
exec uv run --offline --no-project --python "$V" "${CMD[@]}"
