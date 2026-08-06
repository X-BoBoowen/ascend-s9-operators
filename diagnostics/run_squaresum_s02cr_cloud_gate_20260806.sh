#!/usr/bin/env bash
set -euo pipefail

template_project="${1:?usage: $0 TEMPLATE_PROJECT WORK_ROOT}"
work_root="${2:?usage: $0 TEMPLATE_PROJECT WORK_ROOT}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
template_project="$(cd -- "${template_project}" && pwd -P)"

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

baseline_source="${repo_root}/baselines/squaresum_s02ca_formal_best_20260806/SquareSumV1"
candidate_source="${repo_root}/candidates/squaresum_s02cr_general_strided_splitk_20260806/SquareSumV1"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1"

mkdir -- "${work_root}"
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
    "message": "S02CR cloud gate stopped before performance comparison",
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

echo "S02CR_GATE_START repo=${repo_root} template=${template_project} work=${work_root}"
cd -- "${repo_root}"

gate_stage="local_preflight"
python3 diagnostics/validate_squaresum_profile_matrix_20260806.py
python3 diagnostics/squaresum_s02cr_general_strided_static_20260806.py
python3 diagnostics/squaresum_s02cr_splitk_semantics_model_20260806.py

gate_stage="build"
mkdir -p -- "${work_root}/projects" "${work_root}/install"
bash diagnostics/build_squaresum_source_20260806.sh \
    "${baseline_source}" \
    "${template_project}" \
    "${work_root}/projects/s02ca"
bash diagnostics/build_squaresum_source_20260806.sh \
    "${candidate_source}" \
    "${template_project}" \
    "${work_root}/projects/s02cr"

find_package() {
    local project="$1"
    local -a matches=()
    mapfile -t matches < <(
        find "${project}/build_out" -type f -name 'custom_opp*.run' -print
    )
    if (( ${#matches[@]} != 1 )); then
        echo "expected one package below ${project}, found ${#matches[@]}" >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

install_package() {
    local package="$1"
    local install_root="$2"
    mkdir -- "${install_root}"
    chmod u+x -- "${package}"
    "${package}" --install-path="${install_root}"
}

find_env_file() {
    local install_root="$1"
    local -a matches=()
    mapfile -t matches < <(
        find "${install_root}" -type f \
            -path '*/vendors/*/bin/set_env.bash' -print
    )
    if (( ${#matches[@]} != 1 )); then
        echo "expected one set_env.bash below ${install_root}, found ${#matches[@]}" >&2
        return 1
    fi
    printf '%s\n' "${matches[0]}"
}

baseline_package="$(find_package "${work_root}/projects/s02ca")"
candidate_package="$(find_package "${work_root}/projects/s02cr")"
gate_stage="isolated_install"
install_package "${baseline_package}" "${work_root}/install/s02ca"
install_package "${candidate_package}" "${work_root}/install/s02cr"
baseline_env="$(find_env_file "${work_root}/install/s02ca")"
candidate_env="$(find_env_file "${work_root}/install/s02cr")"

# Each command runs in a new child shell.  Neither package's set_env.bash can
# accumulate in the parent or leak into the other package's measurement.
run_with_opp() {
    local env_file="$1"
    shift
    bash -c 'set -euo pipefail; source "$1"; shift; exec "$@"' \
        _ "${env_file}" "$@"
}

run_with_opp "${baseline_env}" python3 -c \
    'import square_sum_v1_validation_lib; print("S02CA_IMPORT_OK")'
run_with_opp "${candidate_env}" python3 -c \
    'import square_sum_v1_validation_lib; print("S02CR_IMPORT_OK")'

# Baseline and candidate discovery use identical in-domain layouts and the
# official tolerance rule. Event values only locate the effect; the gate below
# still uses msprof's official-compatible task stream.
gate_stage="baseline_discovery"
run_with_opp "${baseline_env}" python3 \
    diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S02CA --tier strided_splitk --dtypes fp16 bf16 fp32 \
    | tee "${detail_root}/baseline_strided_splitk_atlas.log"

gate_stage="candidate_correctness_and_discovery"
run_with_opp "${candidate_env}" python3 \
    diagnostics/squaresum_s02cr_correctness_20260806.py S02CR \
    | tee "${detail_root}/candidate_correctness.log"
run_with_opp "${candidate_env}" python3 \
    diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S02CR --tier strided_splitk --dtypes fp16 bf16 fp32 \
    | tee "${detail_root}/candidate_strided_splitk_atlas.log"

# Official-compatible timing: 30 interleaved aclnnMul + SquareSumV1 calls,
# filter Mul rows, then take the median of task durations 10..30.  All three
# competition dtypes are collected into each A/B/A matrix.
gate_stage="official_aba_profile"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02CA_A SquareSumV1/baseline_a strided_splitk fp16 bf16 fp32
run_with_opp "${candidate_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02CR SquareSumV1/candidate strided_splitk fp16 bf16 fp32
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02CA_B SquareSumV1/baseline_b strided_splitk fp16 bf16 fp32

gate_stage="performance_comparison"
set +e
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/baseline_a" \
    --candidate "${detail_root}/candidate" \
    --baseline-b "${detail_root}/baseline_b" \
    --minimum-improvement-percent 10 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${artifact_dir}/result.json"
gate_status=$?
set -e
if [[ -f "${artifact_dir}/result.json" ]]; then
    result_ready=1
fi

if [[ ${gate_status} -ne 0 ]]; then
    echo "S02CR_GATE_REJECT status=${gate_status} result=${artifact_dir}/result.json"
    exit "${gate_status}"
fi

echo "S02CR_GATE_PASS result=${artifact_dir}/result.json log=${artifact_dir}/run.log"
