#!/usr/bin/env bash
set -euo pipefail

baseline_run="${1:?usage: $0 BASELINE_RUN}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
baseline_run="$(cd -- "${baseline_run}" && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1"
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

for required in \
    "${detail_root}/s03e_splitk_baseline_a" \
    "${detail_root}/s03e_splitk_candidate" \
    "${detail_root}/s03e_splitk_baseline_b" \
    "${detail_root}/s03e_singleton_baseline_a" \
    "${detail_root}/s03e_singleton_candidate"; do
    if [[ ! -d "${required}" ]]; then
        echo "required completed profile directory is missing: ${required}" >&2
        exit 2
    fi
done

cd -- "${repo_root}"
validation_dir="${repo_root}/validation/SquareSumV1"
export PYTHONPATH="${validation_dir}${PYTHONPATH:+:${PYTHONPATH}}"
baseline_env="$(find_env_file "${baseline_run}/install/s02f")"
echo "S03E_GATE_RESUME final baseline and comparison"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_D SquareSumV1/s03e_singleton_baseline_b singleton_gap_s03d fp16 bf16 fp32

set +e
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/s03e_splitk_baseline_a" \
    --candidate "${detail_root}/s03e_splitk_candidate" \
    --baseline-b "${detail_root}/s03e_splitk_baseline_b" \
    --minimum-improvement-percent 10 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${detail_root}/s03e_splitk_comparison.json"
splitk_status=$?
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/s03e_singleton_baseline_a" \
    --candidate "${detail_root}/s03e_singleton_candidate" \
    --baseline-b "${detail_root}/s03e_singleton_baseline_b" \
    --minimum-improvement-percent 10 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${detail_root}/s03e_singleton_comparison.json"
singleton_status=$?
set -e

python3 - "${artifact_dir}/result.json" \
    "${detail_root}/s03e_splitk_comparison.json" "${splitk_status}" \
    "${detail_root}/s03e_singleton_comparison.json" "${singleton_status}" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
items = {}
for name, result_path, status in (
    ("splitk", Path(sys.argv[2]), int(sys.argv[3])),
    ("singleton_gap", Path(sys.argv[4]), int(sys.argv[5])),
):
    items[name] = {
        "passed": status == 0,
        "exit_status": status,
        "details": json.loads(result_path.read_text(encoding="utf-8")),
    }
value = {
    "schema_version": 1,
    "passed": all(item["passed"] for item in items.values()),
    "baseline": "S02F",
    "candidate": "S03E",
    "components": items,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
PY

if (( splitk_status != 0 || singleton_status != 0 )); then
    echo "S03E_GATE_REJECT result=${artifact_dir}/result.json"
    exit 1
fi
echo "S03E_GATE_PASS result=${artifact_dir}/result.json log=${artifact_dir}/run.log"
