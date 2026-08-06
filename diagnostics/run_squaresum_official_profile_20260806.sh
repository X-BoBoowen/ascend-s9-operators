#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?usage: $0 CASE_NAME DTYPE [LABEL]}"
dtype_name="${2:?usage: $0 CASE_NAME DTYPE [LABEL]}"
label="${3:-candidate}"

if [[ ! "${case_name}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
   [[ ! "${dtype_name}" =~ ^(fp16|bf16|fp32)$ ]] ||
   [[ ! "${label}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "case, dtype, or label contains unsupported characters"
    exit 2
fi

rm -rf ./PROF*
set +e
timeout 180 msprof --application="python3 diagnostics/squaresum_official_profile_case_20260806.py --case ${case_name} --dtype ${dtype_name} --label ${label}"
status=$?
set -e
if [[ ${status} -eq 124 ]]; then
    echo "profile timed out"
    exit 1
fi
if [[ ${status} -ne 0 ]]; then
    echo "msprof failed with status ${status}"
    exit "${status}"
fi

python3 diagnostics/parse_squaresum_official_msprof_20260806.py \
    --profile-root . \
    --label "${label}" \
    --case "${case_name}" \
    --dtype "${dtype_name}" \
    --require-mul-count 30
