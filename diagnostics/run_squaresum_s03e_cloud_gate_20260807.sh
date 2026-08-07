#!/usr/bin/env bash
set -euo pipefail

template_project="${1:?usage: $0 TEMPLATE_PROJECT BASELINE_RUN WORK_ROOT}"
baseline_run="${2:?usage: $0 TEMPLATE_PROJECT BASELINE_RUN WORK_ROOT}"
work_root="${3:?usage: $0 TEMPLATE_PROJECT BASELINE_RUN WORK_ROOT}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
template_project="$(cd -- "${template_project}" && pwd -P)"
baseline_run="$(cd -- "${baseline_run}" && pwd -P)"

if [[ -e "${work_root}" ]]; then
    echo "work root already exists: ${work_root}" >&2
    exit 2
fi
work_parent="$(cd -- "$(dirname -- "${work_root}")" && pwd -P)"
work_root="${work_parent}/$(basename -- "${work_root}")"
if [[ "${work_root}" == "/" || "${work_root}" == "${repo_root}" ]]; then
    echo "refusing unsafe work root: ${work_root}" >&2
    exit 2
fi

candidate_source="${repo_root}/candidates/squaresum_s03e_s02f_splitk_singleton_20260807/SquareSumV1"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1"
baseline_install="${baseline_run}/install/s02f"
mkdir -p -- "${work_root}/projects" "${work_root}/install"
if [[ "${artifact_dir}" != "${repo_root}/artifact" || "${artifact_dir}" == "/" ]]; then
    echo "refusing unsafe artifact directory: ${artifact_dir}" >&2
    exit 2
fi
rm -rf -- "${artifact_dir}"
mkdir -p -- "${detail_root}"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1

gate_stage="initialization"
result_ready=0
write_failure_result() {
    local status="$1"
    if (( status == 0 || result_ready == 1 )); then
        return
    fi
    python3 - "${artifact_dir}/result.json" "${gate_stage}" "${status}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "schema_version": 1,
    "passed": False,
    "stage": sys.argv[2],
    "exit_status": int(sys.argv[3]),
    "message": "S03E combined cloud gate stopped before completion",
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}
trap 'write_failure_result "$?"' EXIT

echo "S03E_GATE_START repo=${repo_root} baseline=${baseline_run} work=${work_root}"
cd -- "${repo_root}"
validation_dir="${repo_root}/validation/SquareSumV1"
if ! compgen -G "${validation_dir}/square_sum_v1_validation_lib*.so" >/dev/null; then
    echo "SquareSumV1 validation extension is missing below ${validation_dir}" >&2
    exit 2
fi
export PYTHONPATH="${validation_dir}${PYTHONPATH:+:${PYTHONPATH}}"

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

find_package() {
    local project="$1"
    local -a matches=()
    mapfile -t matches < <(
        find "${project}/build_out" -maxdepth 1 -type f -name 'custom_opp*.run' -print
    )
    if (( ${#matches[@]} != 1 )); then
        echo "expected one package below ${project}, found ${#matches[@]}" >&2
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

gate_stage="local_preflight"
python3 diagnostics/validate_squaresum_profile_matrix_20260806.py
python3 diagnostics/squaresum_s03e_combined_static_20260807.py

gate_stage="build"
bash diagnostics/build_squaresum_source_20260806.sh \
    "${candidate_source}" "${template_project}" "${work_root}/projects/s03e"
package="$(find_package "${work_root}/projects/s03e")"
mkdir -- "${work_root}/install/s03e"
chmod u+x -- "${package}"
"${package}" --install-path="${work_root}/install/s03e"

baseline_env="$(find_env_file "${baseline_install}")"
candidate_env="$(find_env_file "${work_root}/install/s03e")"
gate_stage="isolated_import"
run_with_opp "${baseline_env}" python3 -c \
    'import torch, torch_npu, square_sum_v1_validation_lib; print("S02F_IMPORT_OK")'
run_with_opp "${candidate_env}" python3 -c \
    'import torch, torch_npu, square_sum_v1_validation_lib; print("S03E_IMPORT_OK")'

gate_stage="candidate_correctness"
run_with_opp "${candidate_env}" python3 \
    diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S03E --tier atlas --dtypes fp16 bf16 fp32 \
    | tee "${detail_root}/s03e_domain_atlas.log"
run_with_opp "${candidate_env}" python3 \
    diagnostics/squaresum_s03b_correctness_20260806.py S03E \
    | tee "${detail_root}/s03e_splitk_correctness.log"
run_with_opp "${candidate_env}" python3 \
    diagnostics/squaresum_singleton_gap_correctness_20260805.py \
    | tee "${detail_root}/s03e_singleton_correctness.log"

gate_stage="splitk_aba_profile"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_A SquareSumV1/s03e_splitk_baseline_a strided_splitk fp16 bf16 fp32
run_with_opp "${candidate_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S03E SquareSumV1/s03e_splitk_candidate strided_splitk fp16 bf16 fp32
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_B SquareSumV1/s03e_splitk_baseline_b strided_splitk fp16 bf16 fp32

gate_stage="singleton_aba_profile"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_C SquareSumV1/s03e_singleton_baseline_a singleton_gap_s03d fp16 bf16 fp32
run_with_opp "${candidate_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S03E SquareSumV1/s03e_singleton_candidate singleton_gap_s03d fp16 bf16 fp32
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_D SquareSumV1/s03e_singleton_baseline_b singleton_gap_s03d fp16 bf16 fp32

gate_stage="performance_comparison"
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
result_ready=1

if (( splitk_status != 0 || singleton_status != 0 )); then
    echo "S03E_GATE_REJECT result=${artifact_dir}/result.json"
    exit 1
fi
echo "S03E_GATE_PASS result=${artifact_dir}/result.json log=${artifact_dir}/run.log"
