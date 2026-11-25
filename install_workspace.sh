#!/bin/bash
# Inside the container: put the server / bench / lm_eval scripts at $W (default /workspace/k31).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); W=${W:-/workspace/k31}
mkdir -p $W/bench $W/tools $W/logs $W/lmeval
cp $HERE/workspace/serve_k31_tp4.sh $W/; cp $HERE/workspace/bench/*.sh $W/bench/; cp $HERE/workspace/tools/*.py $W/tools/
chmod +x $W/serve_k31_tp4.sh $W/bench/*.sh; echo "scripts installed in $W"
