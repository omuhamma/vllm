#!/bin/bash
# Inside the container: separate venv for lm_eval (same pins as the reference machine) at $W/lmeval_venv.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); W=${W:-/workspace/k31}
/usr/bin/python3 -m venv $W/lmeval_venv
$W/lmeval_venv/bin/python -m pip install -q --upgrade pip
$W/lmeval_venv/bin/python -m pip install -q -r $HERE/lmeval_requirements.txt
$W/lmeval_venv/bin/lm_eval --help >/dev/null && echo "lm_eval venv ready: $W/lmeval_venv"
