#!/usr/bin/env bash
set -euo pipefail

baseline_run="${1:?usage: $0 BASELINE_RUN S03E_RUN}"
candidate_run="${2:?usage: $0 BASELINE_RUN S03E_RUN}"
mode="${3:-fresh}"
if [[ "${mode}" != "fresh" && "${mode}" != "--compare-only" ]]; then
    echo "third argument must be --compare-only when provided" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
baseline_run="$(cd -- "${baseline_run}" && pwd -P)"
candidate_run="$(cd -- "${candidate_run}" && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1"
repeat_root="${detail_root}/s03e_singleton_fp32_repeat"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1

find_env_file() {
    local install_root="$1"
    local -a matches=()
    mapfile -t matches < <(
        find "${install_root}" -type f -path '*/vendors/*/bin/set_env.bash' -print
    )
    if (( ${#matches[@]} != 1 )); then
        echo "expected one set_env.bash below ${install_root}, found ${#matches[@]}" >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

run_with_opp() {
    local env_file="$1"
    shift
    bash -c 'set -eo pipefail; source "$1"; set -u; shift; exec "$@"' \
        _ "${env_file}" "$@"
}

cd -- "${repo_root}"
validation_dir="${repo_root}/validation/SquareSumV1"
export PYTHONPATH="${validation_dir}${PYTHONPATH:+:${PYTHONPATH}}"
baseline_env="$(find_env_file "${baseline_run}/install/s02f")"
candidate_env="$(find_env_file "${candidate_run}/install/s03e")"
case_dir="fp32_s03d_singleton_gap_fast2"
if [[ "${mode}" == "fresh" ]]; then
    rm -rf -- "${repeat_root}"
    mkdir -p -- "${repeat_root}"
    echo "S03E_SINGLETON_FP32_REPEAT_START"
    run_with_opp "${baseline_env}" bash \
        diagnostics/run_squaresum_official_profile_20260806.sh \
        s03d_singleton_gap_fast2 fp32 S02F_R1 \
        "${repeat_root}/baseline_a/${case_dir}"
    run_with_opp "${candidate_env}" bash \
        diagnostics/run_squaresum_official_profile_20260806.sh \
        s03d_singleton_gap_fast2 fp32 S03E_R \
        "${repeat_root}/candidate/${case_dir}"
    run_with_opp "${baseline_env}" bash \
        diagnostics/run_squaresum_official_profile_20260806.sh \
        s03d_singleton_gap_fast2 fp32 S02F_R2 \
        "${repeat_root}/baseline_b/${case_dir}"
else
    echo "S03E_SINGLETON_FP32_REPEAT_COMPARE_ONLY"
    for name in baseline_a candidate baseline_b; do
        direct_result="${repeat_root}/${name}/result.json"
        nested_dir="${repeat_root}/${name}/${case_dir}"
        if [[ -f "${direct_result}" && ! -f "${nested_dir}/result.json" ]]; then
            mkdir -p -- "${nested_dir}"
            cp -- "${direct_result}" "${nested_dir}/result.json"
        fi
    done
fi

python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${repeat_root}/baseline_a" \
    --candidate "${repeat_root}/candidate" \
    --baseline-b "${repeat_root}/baseline_b" \
    --minimum-improvement-percent 10 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${repeat_root}/comparison.json"

python3 - \
    "${artifact_dir}/result.json" \
    "${detail_root}/s03e_splitk_comparison.json" \
    "${detail_root}/s03e_singleton_comparison.json" \
    "${repeat_root}/comparison.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output, splitk_path, singleton_path, repeat_path = map(Path, sys.argv[1:])
splitk = json.loads(splitk_path.read_text(encoding="utf-8"))
singleton = json.loads(singleton_path.read_text(encoding="utf-8"))
repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
passed = (
    splitk["passed"]
    and singleton["regression_ok"]
    and singleton["improvement_ok"]
    and repeat["passed"]
)
value = {
    "schema_version": 1,
    "passed": passed,
    "baseline": "S02F",
    "candidate": "S03E",
    "components": {
        "splitk": {"passed": splitk["passed"], "details": splitk},
        "singleton_gap": {
            "passed": singleton["regression_ok"] and singleton["improvement_ok"] and repeat["passed"],
            "details": singleton,
            "baseline_drift_repeat": repeat,
        },
    },
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
if not passed:
    raise SystemExit(1)
PY
echo "S03E_GATE_PASS_WITH_REPEAT result=${artifact_dir}/result.json"
