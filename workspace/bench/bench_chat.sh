#!/bin/bash
# Benchmark in the user's standard form. Run inside omuh_k31 against a running server.
#   bash /workspace/k31/bench/bench_chat.sh <tokenizer_dir> <concurrency> [seed] [result_name]
#   ISL / OSL / NP (num prompts, default = concurrency) from the environment; defaults 1024 / 1024.
TOK=${1:-${MODEL:-/models/Kimi-K3.1-Pro}}; C=${2:-64}; SEED=${3:-0}; NAME=${4:-bench_c${C}}
vllm bench serve --backend openai-chat --model kimi-k31 \
    --tokenizer $TOK --trust-remote-code --host localhost --port ${PORT:-8131} \
    --endpoint /v1/chat/completions --dataset-name random --random-range-ratio 0 \
    --request-rate=inf --random-prefix-len 0 --ignore-eos --max-concurrency $C \
    --random-input-len ${ISL:-1024} --random-output-len ${OSL:-1024} --num-prompts ${NP:-$C} --seed $SEED \
    --save-result --result-dir ${W:-/workspace/k31}/bench --result-filename $NAME.json
