#!/bin/bash
# Host: start the K3.1 container exactly like the reference machine's omuh_k31_latestDspark.
# MODELS = host dir holding Kimi-K3.1-Pro and Kimi-K3.1-Pro-DSpark; WS = host dir mounted at /workspace.
set -eu
NAME=${NAME:-k31}; MODELS=${MODELS:-/data2/moonshotai}; WS=${WS:-$HOME}
IMAGE=${IMAGE:-rocm/vllm-dev@sha256:7b7eb2b66a682285d1bd1a0ab89a700acbad3ccbf76f5a47942a8aaad0a6c045}  # = rocm/vllm-dev:nightly_455_wip, image id sha256:9de00a2453932fa5474f10c61e9f589a71cf2dd98925f1ab48589d592395a20a
docker run -d --name $NAME --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v $MODELS:/data2/moonshotai -v $MODELS:/models -v $WS:/workspace -w /app \
  $IMAGE sleep infinity
docker exec $NAME python3 -c "import vllm; print('vllm', vllm.__version__)"
