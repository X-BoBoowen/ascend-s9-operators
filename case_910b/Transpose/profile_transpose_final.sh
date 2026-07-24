#!/bin/bash
set -euo pipefail

export PATH=/home/ma-user/gcc/bin:$PATH
export LD_LIBRARY_PATH=/home/ma-user/gcc/lib64:${LD_LIBRARY_PATH:-}
export CC=/home/ma-user/gcc/bin/gcc
export CXX=/home/ma-user/gcc/bin/g++
source /home/ma-user/Ascend/cann-8.5.0/set_env.sh

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

profile_once() {
    local run_id="$1"
    echo "[Codex] TransposeFast final profile run ${run_id}"
    rm -rf -- PROF*
    timeout 180 msprof --application="python3 test_op.py 1"
    printf 'score_run_%s=' "$run_id"
    python3 get_time.py
}

profile_once 1
profile_once 2
profile_once 3
