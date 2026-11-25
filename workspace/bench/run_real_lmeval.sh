#!/bin/bash
# Inside omuh_k31_latestDspark: real-weight K3.1 TP4 + new DSpark (4 draft tokens) with REAL rejection sampling
# (standard: target-model outputs), then lm_eval gsm8k (full 1319, 5-shot chat, max_gen_toks 2048) and the real
# acceptance length from the spec-decode counters. Kills the server at the end.
set -u; ulimit -c 0; export HSA_ENABLE_COREDUMP=0
W=${W:-/workspace/k31}; PORT=8131; TAG=${TAG:-real_lmeval}; LOG=$W/logs/${TAG}_server.log
mkdir -p $W/lmeval
DCP=1 MAX_LEN=${MAX_LEN:-262144} MAX_SEQS=${MAX_SEQS:-128} GPU_UTIL=${GPU_UTIL:-0.94} SPEC_K=4 REJECT=real \
  bash $W/serve_k31_tp4.sh > $LOG 2>&1 &
trap 'bash $W/bench/kill_servers.sh >/dev/null 2>&1; echo SERVER_KILLED' EXIT
for i in $(seq 1 900); do
  curl -sf localhost:$PORT/health >/dev/null && break
  { pgrep -f "vllm serve" >/dev/null && ! grep -qa "EngineCore failed to start" $LOG; } || { echo SERVER_DIED; grep -aE "Error|Traceback" $LOG | tail -15; exit 1; }
  sleep 4
done
curl -sf localhost:$PORT/health >/dev/null || { echo NOT_READY; exit 1; }
echo "SERVER_READY $(date +%T)"
python3 $W/tools/k31_spec_metrics.py snap $W/lmeval/${TAG}_spec_before.json
$W/lmeval_venv/bin/lm_eval --model local-chat-completions \
  --model_args model=kimi-k31,base_url=http://localhost:$PORT/v1/chat/completions,num_concurrent=${LM_CONC:-64},max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs max_gen_toks=2048 --output_path $W/lmeval/gsm8k_$TAG > $W/logs/${TAG}_lmeval.log 2>&1
echo "lm_eval exit code: $? $(date +%T)"
grep -E "gsm8k|exact_match|flexible|strict" $W/logs/${TAG}_lmeval.log | tail -8
python3 $W/tools/k31_spec_metrics.py snap $W/lmeval/${TAG}_spec_after.json
python3 $W/tools/k31_spec_metrics.py diff $W/lmeval/${TAG}_spec_before.json $W/lmeval/${TAG}_spec_after.json | tail -12
grep -aE "SpecDecoding metrics" $LOG | tail -3 | cut -c1-230
echo LMEVAL_DONE
