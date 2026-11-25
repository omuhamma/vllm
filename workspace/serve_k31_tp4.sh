#!/bin/bash
# Kimi-K3.1 real weights, TP4 on all 4 GPUs, fp8 KV (-> fp8_k3_mla_640 on gfx1250),
# DSpark spec decode (new Inferact/Kimi-K3.1-Pro-DSpark, 4 draft tokens) with synthetic acceptance length 3.36 by default.
# NCCL_GRAPH_REGISTER=0: needed for DSpark on gfx1250; without it RCCL fails capturing the draft graph (HIP invalid argument). From /opt/k3/k3_455_tp8dcp8_server.sh in rocm/fw-bringup:gfx1250-atom-k3-20261113.
# Run inside the K3.1 container (rocm/vllm-dev:nightly_455_wip + k31_deploy overlay). Only when all 4 GPUs are free. Optional: PROF_DIR=<dir> enables the torch profiler.
# DCP=<n> (default 1): decode context parallelism. DCP=4 gives TP4 + DCP4 (KV sharded over the 4 ranks:
#   --decode-context-parallel-size 4 --dcp-comm-backend ${DCP_COMM:-ag_rs}, and the DSpark draft kept at
#   "draft_decode_context_parallel_size":1, i.e. a replicated draft cache; TRITON_MLA has no DCP draft path).
#   Validated at TP2 + DCP2 only (VLLM_NOTES.md "DCP"); TP4 + DCP4 has not been run.
DCP=${DCP:-1}
DCP_ARGS=""; DRAFT_DCP=""
if [ "$DCP" -gt 1 ]; then
  DCP_ARGS="--decode-context-parallel-size $DCP --dcp-comm-backend ${DCP_COMM:-ag_rs}"
  DRAFT_DCP=',"draft_decode_context_parallel_size":1'
fi
PROF_ARGS=""
[ -n "${PROF_DIR:-}" ] && PROF_ARGS="--profiler-config.profiler torch --profiler-config.torch_profiler_dir $PROF_DIR --profiler-config.torch_profiler_with_stack=false"
[ -n "${PROF_DIR:-}" ] && [ "${PROF_SHAPES:-0}" = 1 ] && PROF_ARGS="$PROF_ARGS --profiler-config.torch_profiler_record_shapes=true"
[ -n "${PROF_DIR:-}" ] && [ -n "${PROF_MAXIT:-}" ] && PROF_ARGS="$PROF_ARGS --profiler-config.max_iterations $PROF_MAXIT"
# RCCL settings from the ATOM K3 gfx1250 repro kit (rocm/fw-bringup:gfx1250-atom-k3-20261113,
# /opt/k3/k3-mlperf-repro/recipe.md), single-node subset. Always on (RCCL_ENV=0 exists only for A/B checks; never use it for real runs).
if [ "${RCCL_ENV:-1}" = 1 ]; then
  export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_CUMEM_ENABLE=1
  export RCCL_DDA_LL_ONESHOT_THRESHOLD=2097152 RCCL_DDA_LL_TWOSHOT_THRESHOLD=134217728
  # HSA settings of the same recipe (/opt/k3/k3_455_tp8dcp8_server.sh): SDMA copies, SVM, XNACK=0 only when amdgpu noretry=1.
  export HSA_ENABLE_SDMA=1 HSA_USE_SVM=1
  export HSA_XNACK=$([ "$(cat /sys/module/amdgpu/parameters/noretry 2>/dev/null)" = 1 ] && echo 0 || echo 1)
fi
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# REJECT=synthetic (default, acceptance ${SYNTH_AL:-3.36}: accepts draft tokens without the target check, throughput only)
#   | real (standard rejection sampling: target-model outputs, for accuracy).
# DRAFT=<dir> (default the new Inferact/Kimi-K3.1-Pro-DSpark), SPEC_K=<n> draft tokens (default 4),
# SPEC_EXTRA=',"key":value' extra speculative-config keys. The new drafter needs ',"attention_backend":"TRITON_ATTN"'
#   on K3.1 (ROCM_AITER_UNIFIED_ATTN has no BLHNC KV layout, which the DSA indexer cache needs); that is the default.
if [ "${REJECT:-synthetic}" = real ]; then
  REJ_ARGS=""
else
  REJ_ARGS=",\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":${SYNTH_AL:-3.36}"
fi
SPEC="{\"method\":\"dspark\",\"model\":\"${DRAFT:-/data2/moonshotai/Kimi-K3.1-Pro-DSpark}\",\"num_speculative_tokens\":${SPEC_K:-4},\"draft_sample_method\":\"probabilistic\"$REJ_ARGS$DRAFT_DCP${SPEC_EXTRA-,\"attention_backend\":\"TRITON_ATTN\"}}"
# VLLM_ROCM_DSA_GLUON_TOPK=1: aiter Gluon DSA prefill top-k on every prefill chunk (aiter omuhamma/gfx1250-dsa-prefill-topk).
#   vLLM's auto mode only uses it for rows <= 2048 keys; the C++ top_k_per_row_prefill slows 3-20x on tied logits
#   (32k chunk: 174 us distinct, 647 us at 50% exact zeros; Gluon 238 / 250 us; real traces measured 524 us).
export VLLM_ROCM_DSA_GLUON_TOPK=${VLLM_ROCM_DSA_GLUON_TOPK:-1}
# PTPC=1 (default): KDA in_proj (*in_proj_qkvgfab) quantized online to FP8 per-channel and run on aiter's Gluon PTPC GEMM
#   (aiter PR #8347; weight K tail pre-negated at load). PTPC=0 = bf16 in_proj (a16w16) as before.
if [ "${PTPC:-1}" = 1 ]; then export VLLM_ROCM_GLUON_PTPC=1; QUANT_CFG=${QUANT_CFG-'{"targets":{"*in_proj_qkvgfab":"fp8_per_channel"}}'}; else export VLLM_ROCM_GLUON_PTPC=0; fi
NCCL_GRAPH_REGISTER=0 \
HIP_VISIBLE_DEVICES=0,1,2,3 \
AITER_FORCE_A8W4=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
VLLM_ROCM_USE_SKINNY_GEMM=0 \
VLLM_ENGINE_READY_TIMEOUT_S=3600 \
exec vllm serve ${MODEL:-/models/Kimi-K3.1-Pro} \
   --served-model-name kimi-k31 --host localhost --port ${PORT:-8131} \
   --trust-remote-code \
   --tensor-parallel-size 4 \
   --load-format auto \
   --gpu-memory-utilization ${GPU_UTIL:-0.95} \
   --mm-encoder-tp-mode data \
   --max-num-seqs ${MAX_SEQS:-256} \
   --max-num-batched-tokens ${MAX_BT:-8192} \
   --max-model-len ${MAX_LEN:-524288} \
   --kv-cache-dtype fp8 \
   ${QUANT_CFG:+--quantization-config "$QUANT_CFG"} \
   --moe-backend aiter_flydsl_mxfp4_bf16 \
   \
   --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+fused_rms_norm_gated"]}' \
   --speculative-config "$SPEC" \
   --enable-auto-tool-choice \
   --tool-call-parser kimi_k3 \
   --reasoning-parser kimi_k3 \
   $DCP_ARGS $PROF_ARGS ${EXTRA_ARGS:-}
