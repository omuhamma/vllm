# Kimi-K3.1 on MI455X (gfx1250): reference deployment, 2026-11-25

This bundle reproduces the reference machine's K3.1 setup: container, aiter + vLLM state, server, benchmark and lm_eval.

## What's here

| Path | What it is |
|---|---|
| `run_container.sh` | Host: starts the container from the pinned image, with the same devices, mounts and options as the reference |
| `overlay/` + `overlay_files.txt` + `overlay.sha256` | The 129 aiter and vLLM files that differ from the stock image, laid out as site-packages paths |
| `install_overlay.sh` | In the container: checks the vLLM version, backs up the originals, copies the overlay into `/opt/venv/lib/python3.12/site-packages`, checks the sha256s |
| `verify_overlay.sh` | In the container (needs a GPU): checks the sha256s, then imports the PTPC / logits / top-k kernels |
| `install_workspace.sh` | In the container: copies the scripts to `$W` (default `/workspace/k31`) |
| `setup_lmeval_venv.sh` + `lmeval_requirements.txt` | In the container: a separate lm_eval 0.4.13 venv at `$W/lmeval_venv` |
| `workspace/serve_k31_tp4.sh` | The server: TP4, fp8 KV, FlyDSL a8w4 MoE, DSpark spec decode, CUDA graphs, PTPC in_proj |
| `workspace/bench/run_real_roof.sh` | 32k/32 c32 and 1k/1k c64: warmup → clean pass → 2 profiled passes |
| `workspace/bench/run_real_lmeval.sh` | Real rejection sampling + gsm8k (1319, 5-shot chat) + measured acceptance length |
| `workspace/bench/bench_chat.sh`, `kill_servers.sh`, `tools/k31_spec_metrics.py` | Helpers used by the two scripts above |

## Software state

- Image: `rocm/vllm-dev:nightly_455_wip`, pinned as `rocm/vllm-dev@sha256:7b7eb2b66a682285d1bd1a0ab89a700acbad3ccbf76f5a47942a8aaad0a6c045` (image id `sha256:9de00a24…`). vLLM 0.9.2rc2.dev18209+g17a4dcc5d.rocm103, amd-aiter 0.1.29, triton 3.9.0+git12225a8e, torch 2.13.0+rocm10.3.0a20261030, flydsl 0.4.2.
- vLLM and aiter run from the installed copies in `/opt/venv/lib/python3.12/site-packages`. Editing `/app/vllm-project` or `/app/aiter` has no effect.
- The overlay is the site-packages state of the reference container. It carries:
  - the reference vLLM `k31` branch (`4ff9fc4a62`: K3.1 model / KDA / DSA / MLA / DSpark fixes, #66280 bmm part);
  - the reference aiter `k31` branch (`1acc0108c`: tuned mqa_logits table, #7189 MoE rows, FlashKDA, the gfx1250 a16w16 configs);
  - aiter Gluon PTPC a8w8 GEMM, ROCm/aiter `omuhamma/a8w8-ptpc` @ `86e83790bf` (PR #8347);
  - aiter fp8_mqa_logits row-group, ROCm/aiter `omuhamma/gfx1250-fp8-mqa-logits-rowgroup` @ `bf74b62e76` (PR #8070);
  - aiter DSA prefill top-k v6 (PR #8069);
  - the vLLM hook `AiterGluonPtpcFp8ScaledMMLinearKernel`, opt-in via `VLLM_ROCM_GLUON_PTPC=1`.
- AITER JIT modules (`aiter/jit/*.so`) are not shipped. They build on first use, so the first server start is slower.

## Models

The server expects `/models/Kimi-K3.1-Pro` (target) and `/data2/moonshotai/Kimi-K3.1-Pro-DSpark` (drafter, 4.7 GB). `run_container.sh` mounts the host dir `$MODELS` (default `/data2/moonshotai`) at both places. Override with `MODEL=...` / `DRAFT=...`.

## Server knobs (`serve_k31_tp4.sh`, all env)

| Env | Default | Meaning |
|---|---|---|
| `PTPC` | `1` | KDA in_proj online FP8 per-channel on the Gluon PTPC GEMM (`--quantization-config '{"targets":{"*in_proj_qkvgfab":"fp8_per_channel"}}'`). `0` = bf16 in_proj |
| `REJECT` | `synthetic` | `synthetic` = acceptance `SYNTH_AL` (3.36), throughput only; `real` = real rejection sampling, for accuracy |
| `SPEC_K` | `4` | draft tokens |
| `MAX_LEN` / `MAX_SEQS` / `MAX_BT` / `GPU_UTIL` | 524288 / 256 / 8192 / 0.95 | the run scripts use 262144 / 128 / 8192 / 0.94 |
| `DCP` | `1` | decode context parallel (TP4 + DCP4 never validated) |
| `PROF_DIR`, `PROF_SHAPES`, `PROF_MAXIT` | off | torch profiler |
| `RCCL_ENV` | `1` | ATOM RCCL/HSA recipe env. Always leave on |
| `PORT` | `8131` | |

Always set: `NCCL_GRAPH_REGISTER=0` (without it RCCL fails capturing the DSpark draft graph), `AITER_FORCE_A8W4=1`, `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_DSA_GLUON_TOPK=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, `VLLM_ROCM_USE_SKINNY_GEMM=0`.

## Reference results (real weights, 2026-11-23)

These runs predate the 11-24 installs (PTPC #8347, logits cleanup, top-k v6) and ran with PTPC=0. Treat them as the baseline to compare against.

| Run | Result |
|---|---|
| 32k/32 c32 clean pass | 61.4 s duration, mean TTFT 32.0 s, mean TPOT 108.1 ms |
| 1k/1k c64 clean pass | 47.0 s duration, 1393.6 output tok/s, mean TPOT 5.7 ms |
| gsm8k (1319, 5-shot chat, max_gen_toks 2048, REJECT=real) | flexible-extract 0.4572 ± 0.0137, strict-match 0.3783 ± 0.0134 |
| real DSpark acceptance during gsm8k | mean acceptance length 2.80 (rate 0.45) |

Real-weight accuracy with PTPC=1 has not been run yet. Run lm_eval with PTPC=1, and with PTPC=0 as the control, before trusting it. lm_eval downloads gsm8k from the HF Hub, so it needs internet access or `HF_DATASETS_CACHE` pre-filled.
