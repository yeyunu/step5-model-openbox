#!/usr/bin/env bash
source /home/yyy/model_openbox/reduced-recheck-225-10.20/env.sh
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
cd "$RUN_ROOT"
for round in 1 2; do
  out="$RUN_ROOT/results/resume-ipc-round$round-10-20"
  mkdir -p "$out"
  test ! -e "$out/report.json"
  curl --noproxy '*' --fail --silent --show-error --max-time 5 http://127.0.0.1:8940/health > "$out/health-before.txt"
  cmd=(/home/yyy/model_openbox/.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8
    "$RUN_ROOT/strict_mechanism_runner.py" --mode ipc --skill-scripts "$STRICT_SKILL_SCRIPTS"
    --validator-sha256 17df09c58666e286f3f367a48b4699fcd9f0e77b187a4e63d3fc700987f4d037
    --server-url http://127.0.0.1:8940 --model step5-head-tail8
    --checkpoint "$RUN_ROOT/checkpoints/head-tail8" --expected-workers 8
    --prompts-json "$RUN_ROOT/prompts.json" --max-tokens 8 --timeout 1200
    --report "$out/report.json" --evidence "$out/events.jsonl"
    --devices 0,1,2,3,4,5,6,7 --packed-buffer-mb 4352)
  printf '%q ' "${cmd[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  date -Is > "$out/started-at.txt"
  echo $$ > "$out/supervisor.pid"
  set +e
  "${cmd[@]}" > "$out/client.log" 2>&1 &
  child=$!
  echo "$child" > "$out/client.pid"
  trap 'date -Is >> "$out/supervisor-signals.log"; kill -TERM "$child" 2>/dev/null || true' TERM INT
  wait "$child"
  result=$?
  echo "$result" > "$out/client.exit"
  echo "ipc_round=$round exit=$result"
  if [[ $result != 0 ]]; then exit "$result"; fi
  set -e
done
