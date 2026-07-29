#!/bin/bash
set -euo pipefail

case_id="${1:?Usage: bash run.sh <case_id>}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

export LD_LIBRARY_PATH="${ASCEND_OPP_PATH:?}/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH:-}"

rm -rf -- build dist custom_ops.egg-info
python3 setup.py build bdist_wheel

shopt -s nullglob
wheels=(dist/custom_ops*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "[ERROR] Expected exactly one wheel, found ${#wheels[@]}"
    exit 1
fi
python3 -m pip install --force-reinstall "${wheels[0]}"

rm -rf -- PROF*
test_log=".last_test.log"
set +e
timeout 180 msprof --application="python3 test_op.py $case_id" 2>&1 | tee "$test_log"
profile_status=${PIPESTATUS[0]}
set -e

if (( profile_status == 124 )); then
    echo "[ERROR] Test timed out"
    exit 1
fi
if (( profile_status != 0 )); then
    echo "[ERROR] Profiler failed with status $profile_status"
    exit "$profile_status"
fi
if ! grep -Fq "case${case_id} verify result pass!" "$test_log"; then
    echo "[ERROR] Accuracy verification did not pass"
    exit 1
fi

time_use="$(python3 get_time.py)"
time_base=9999999999999
echo "time_base = $time_base time_use = $time_use"
if [[ ! "$time_use" =~ ^[0-9]+$ ]] || (( time_use == 0 )); then
    echo "[ERROR] Performance not achieved"
    exit 1
fi
if (( time_use >= time_base )); then
    echo "[ERROR] Performance exceeds baseline"
    exit 1
fi

# Collect this run into artifact/<Op>/ for off-machine analysis. Runs after
# scoring so it can never change the pass/fail verdict; a collector failure is
# reported but not fatal.
python3 ../collect_artifact.py "SquareSumV1" "$case_id" \
    ${CASE_BYTES:+--bytes "$CASE_BYTES"} \
    --log "$test_log" || echo "[WARN] artifact collection failed"

echo "Operator performance and accuracy have passed"
