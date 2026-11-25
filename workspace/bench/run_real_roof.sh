#!/bin/bash
# Inside omuh_k31_latestDspark: real-weight K3.1 TP4 DCP1 + new DSpark (4 draft tokens, synthetic AL 3.36), CUDA graphs on.
# One server for both roofline workloads. For each workload, in this order:
#   warmup pass (seed 101) -> clean pass (seed 202, the numbers) -> traced pass with the "start" window (seed 303,
#   profiler on as the pass begins: prefill-heavy steps) -> traced pass with the "decode" window (seed 404, profiler on
#   once no request is waiting). Each window records MAXIT iterations with shapes and moves its traces to
#   trace_<TAG>_<wl>_<window>/.  Workloads: 32k = 32768/32 c32 np32, 1k = 1024/1024 c64 np64.
#   TAG=real_roof bash bench/run_real_roof.sh
set -u; ulimit -c 0; export HSA_ENABLE_COREDUMP=0
W=${W:-/workspace/k31}; PORT=8131; TOK=${MODEL:-/models/Kimi-K3.1-Pro}; TAG=${TAG:-real_roof}; MAXIT=${MAXIT:-20}
TR=$W/trace_${TAG}_tmp; LOG=$W/logs/${TAG}_server.log
rm -rf $TR; mkdir -p $TR $W/bench
DCP=1 MAX_LEN=${MAX_LEN:-262144} MAX_SEQS=${MAX_SEQS:-128} GPU_UTIL=${GPU_UTIL:-0.94} SPEC_K=4 SYNTH_AL=3.36 \
  PROF_DIR=$TR PROF_SHAPES=1 PROF_MAXIT=$MAXIT bash $W/serve_k31_tp4.sh > $LOG 2>&1 &
trap 'bash $W/bench/kill_servers.sh >/dev/null 2>&1; echo SERVER_KILLED' EXIT
for i in $(seq 1 ${READY_STEPS:-900}); do
  curl -sf localhost:$PORT/health >/dev/null && break
  { pgrep -f "vllm serve" >/dev/null && ! grep -qa "EngineCore failed to start" $LOG; } || { echo SERVER_DIED; grep -aE "Error|Traceback" $LOG | grep -v NUMA | tail -15; exit 1; }
  sleep 4
done
curl -sf localhost:$PORT/health >/dev/null || { echo NOT_READY; exit 1; }
echo "SERVER_READY $(date +%T)"; grep -aE "GPU KV cache size|Maximum concurrency" $LOG | tail -2 | cut -c1-200
B(){ bash $W/bench/bench_chat.sh $TOK $CONC $2 $1 > $W/bench/$1.log 2>&1; echo "--- $1"; grep -E "Successful|Failed|duration|Output token|Total token|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Mean ITL|Acceptance length" $W/bench/$1.log; }
waittr(){ # wait until the 4 rank traces of the window are written and stable, then move them
  for i in $(seq 1 180); do n=$(ls $TR/*rank*.pt.trace.json.gz 2>/dev/null | wc -l); s1=$(du -sb $TR | cut -f1); sleep 10; s2=$(du -sb $TR | cut -f1); [ "$n" -ge 4 ] && [ "$s1" = "$s2" ] && break; done
  sleep 5; D=$W/trace_${TAG}_$1; rm -rf $D; mv $TR $D; mkdir -p $TR
  echo "window $1 saved $(date +%T): $(ls $D/*rank*.pt.trace.json.gz 2>/dev/null | wc -l) rank traces, $(du -sh $D | cut -f1)"; }
metric(){ curl -s localhost:$PORT/metrics | awk -v m="^vllm:$1" '$0 ~ m {s+=$2} END {print s+0}'; }
prof_on(){ curl -s -X POST localhost:$PORT/start_profile -H 'Content-Type: application/json' -d "{\"max_iterations\": $MAXIT}" >/dev/null; echo "profiler on ($1) $(date +%T)"; }
for WL in ${WORKLOADS:-32k 1k}; do
  if [ $WL = 32k ]; then export ISL=32768 OSL=32 NP=32; CONC=32; else export ISL=1024 OSL=1024 NP=64; CONC=64; fi
  echo "=== workload $WL: ISL=$ISL OSL=$OSL CONC=$CONC NP=$NP $(date +%T) ==="
  B ${TAG}_${WL}_warmup 101
  B ${TAG}_${WL}_clean 202
  B ${TAG}_${WL}_traced 303 > $W/bench/${TAG}_${WL}_traced.summary 2>&1 & bp=$!
  sleep 2; prof_on "$WL start"; wait $bp; curl -s -X POST localhost:$PORT/stop_profile >/dev/null; waittr ${WL}_start
  cat $W/bench/${TAG}_${WL}_traced.summary
  B ${TAG}_${WL}_decodepass 404 > $W/bench/${TAG}_${WL}_decodepass.summary 2>&1 & bp=$!
  for i in $(seq 1 1800); do
    [ "$(metric num_requests_waiting)" = 0 ] && [ "$(metric num_requests_running)" != 0 ] && break
    kill -0 $bp 2>/dev/null || break; sleep 1
  done
  prof_on "$WL decode"; wait $bp; curl -s -X POST localhost:$PORT/stop_profile >/dev/null; waittr ${WL}_decode
  cat $W/bench/${TAG}_${WL}_decodepass.summary
done
grep -aE "SpecDecoding metrics" $LOG | tail -2 | cut -c1-220
echo RUN_DONE
