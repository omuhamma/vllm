#!/bin/bash
# Kill every K3.1 test server and run script in this container.
for p in $(pgrep -f "run_tp4_dummy.sh"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "bin/vllm serve /(workspace/k31|models|data2)"); do kill $p 2>/dev/null; done
sleep 10
for p in $(pgrep -f "bin/vllm serve /(workspace/k31|models|data2)"; pgrep -f "^VLLM::"); do kill -9 $p 2>/dev/null; done
sleep 4; rocm-smi --showpids 2>/dev/null | grep -E "No KFD|^[0-9]" | head -4
