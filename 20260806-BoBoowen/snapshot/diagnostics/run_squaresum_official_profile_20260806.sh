#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?usage: $0 CASE_NAME DTYPE [LABEL] [RUN_ROOT]}"
dtype_name="${2:?usage: $0 CASE_NAME DTYPE [LABEL] [RUN_ROOT]}"
label="${3:-candidate}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_root="${repo_root}/artifact"
run_root="${4:-${artifact_root}/single_profile}"

cd "${repo_root}"

if [[ ! "${case_name}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
   [[ ! "${dtype_name}" =~ ^(fp16|bf16|fp32)$ ]] ||
   [[ ! "${label}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "case, dtype, or label contains unsupported characters"
    exit 2
fi

mkdir -p -- "${artifact_root}"
mkdir -p -- "$(dirname -- "${run_root}")"
run_parent="$(cd -- "$(dirname -- "${run_root}")" && pwd -P)"
run_root="${run_parent}/$(basename -- "${run_root}")"
case "${run_root}" in
    "${artifact_root}"/*) ;;
    *)
        echo "run root must be below ${artifact_root}"
        exit 2
        ;;
esac

rm -rf -- "${run_root}"
profile_root="${run_root}/msprof"
mkdir -p -- "${profile_root}"
set +e
timeout 180 msprof \
    --output="${profile_root}" \
    --application="python3 diagnostics/squaresum_official_profile_case_20260806.py --case ${case_name} --dtype ${dtype_name} --label ${label}"
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
    --profile-root "${profile_root}" \
    --label "${label}" \
    --case "${case_name}" \
    --dtype "${dtype_name}" \
    --require-mul-count 30 \
    --result-json "${run_root}/result.json"
