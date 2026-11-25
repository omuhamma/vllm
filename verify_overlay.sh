#!/bin/bash
# Inside the container: check the overlay is in place and the K3.1 kernels import.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); SP=${SP:-/opt/venv/lib/python3.12/site-packages}
cd $SP && sha256sum -c --quiet $HERE/overlay.sha256 && echo "sha256: all $(wc -l < $HERE/overlay.sha256) files match"
python3 - <<'PY'
import vllm, aiter
from vllm.model_executor.kernels.linear.scaled_mm.aiter import AiterGluonPtpcFp8ScaledMMLinearKernel  # Gluon PTPC in_proj
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8, gemm_a8w8_negate_w_k_tail             # PTPC GEMM (#8347)
import aiter.ops.triton.attention.fp8_mqa_logits, aiter.ops.triton.topk_prefill                    # logits (#8070), top-k (#8069)
print("imports OK: vllm", vllm.__version__, "| aiter", aiter.__file__)
PY
