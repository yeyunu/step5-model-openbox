#!/bin/bash
set -uo pipefail
P=/data/yyy/step5_bf16_gpqa_20261027_1600
phase=${1:?smoke or formal}
case "$phase" in
  smoke) batch=2; limit=(--limit 4);;
  formal) batch=32; limit=();;
  *) exit 2;;
esac
R=$P/$phase
mkdir -p "$R"
test ! -e "$R/started" || { echo 'Run already started; refusing duplicate'; exit 3; }
source /home/yyy/gpqa_eval_env.sh
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:10086/health > "$R/health-before.txt" || exit 4
date -Is > "$R/started"
echo $$ > "$R/supervisor.pid"
cmd=(/home/yyy/evalscope-bf16/.venv/bin/evalscope eval
  --model step --api-url http://177.2.25.1:10086/v1 --api-key EMPTY
  --eval-type openai_api --datasets gpqa_diamond
  --eval-batch-size "$batch" --repeats 1 "${limit[@]}"
  --dataset-args '{"gpqa_diamond":{"local_path":"/home/yyy/evalscope-bf16/local_datasets/gpqa_diamond"}}'
  --generation-config 'do_sample=true,temperature=1.0,top_p=0.95,top_k=0,max_tokens=149204,timeout=360000'
  --work-dir "$R/outputs")
printf '%q ' "${cmd[@]}" > "$R/command.sh"
printf '\n' >> "$R/command.sh"
"${cmd[@]}" > "$R/evalscope.log" 2>&1 &
child=$!
echo "$child" > "$R/pid"
trap 'echo "SIGTERM $(date -Is)" >> "$R/lifecycle.log"; kill -TERM "$child" 2>/dev/null || true' TERM
wait "$child"
rc=$?
echo "$rc" > "$R/exit"
date -Is > "$R/finished"
exit "$rc"
