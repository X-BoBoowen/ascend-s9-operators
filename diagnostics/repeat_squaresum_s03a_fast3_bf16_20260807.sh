#!/usr/bin/env bash
set -euo pipefail

s03_run="${1:?usage: $0 S03_RUN}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
s03_run="$(cd -- "${s03_run}" && pwd -P)"
artifact_dir="${repo_root}/artifact"
repeat_root="${artifact_dir}/SquareSumV1/s03a_fast3_bf16_repeat"
rm -rf -- "${artifact_dir}"
mkdir -p -- "${repeat_root}"
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
baseline_env="$(find_env_file "${s03_run}/install/s02f")"
candidate_env="$(find_env_file "${s03_run}/install/s03a")"
case_dir="bf16_fast3_noncontiguous_last"

echo "S03A_FAST3_BF16_REPEAT_START"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_profile_20260806.sh \
    fast3_noncontiguous_last bf16 S02F_R1 \
    "${repeat_root}/baseline_a/${case_dir}"
run_with_opp "${candidate_env}" bash \
    diagnostics/run_squaresum_official_profile_20260806.sh \
    fast3_noncontiguous_last bf16 S03A_R \
    "${repeat_root}/candidate/${case_dir}"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_profile_20260806.sh \
    fast3_noncontiguous_last bf16 S02F_R2 \
    "${repeat_root}/baseline_b/${case_dir}"

python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${repeat_root}/baseline_a" \
    --candidate "${repeat_root}/candidate" \
    --baseline-b "${repeat_root}/baseline_b" \
    --minimum-improvement-percent 2 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${repeat_root}/comparison.json"

python3 - "${artifact_dir}/result.json" "${repeat_root}/comparison.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
comparison = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
value = {
    "schema_version": 1,
    "passed": comparison["passed"],
    "baseline": "S02F",
    "candidate": "S03A",
    "purpose": "independent repeat of the only prior baseline-drift failure",
    "details": comparison,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
echo "S03A_FAST3_BF16_REPEAT_PASS result=${artifact_dir}/result.json"
