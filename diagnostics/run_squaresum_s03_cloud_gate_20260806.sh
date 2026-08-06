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

baseline_source="${repo_root}/baselines/squaresum_s02f_global_best_20260806/SquareSumV1"
s03a_source="${repo_root}/candidates/squaresum_s03a_s02f_bf16_fused_20260806/SquareSumV1"
s03b_source="${repo_root}/candidates/squaresum_s03b_s02f_fast4_splitk_20260806/SquareSumV1"
s03d_source="${repo_root}/candidates/squaresum_s03d_s02f_singleton_gap_20260806/SquareSumV1"
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
    "message": "S03 cloud gate stopped before both comparisons completed",
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

echo "S03_GATE_START repo=${repo_root} template=${template_project} work=${work_root}"
cd -- "${repo_root}"

gate_stage="local_preflight"
python3 diagnostics/validate_squaresum_profile_matrix_20260806.py
python3 diagnostics/squaresum_s03a_bf16_static_20260806.py
python3 diagnostics/squaresum_s02cq_bf16_semantics_model_20260806.py
python3 diagnostics/squaresum_s03b_fast4_splitk_static_20260806.py
python3 diagnostics/squaresum_s02cr_splitk_semantics_model_20260806.py
python3 diagnostics/squaresum_s03d_singleton_gap_static_20260806.py

gate_stage="build"
mkdir -p -- "${work_root}/projects" "${work_root}/install"
for name in s02f s03a s03b s03d; do
    case "${name}" in
        s02f) source_path="${baseline_source}" ;;
        s03a) source_path="${s03a_source}" ;;
        s03b) source_path="${s03b_source}" ;;
        s03d) source_path="${s03d_source}" ;;
    esac
    bash diagnostics/build_squaresum_source_20260806.sh \
        "${source_path}" \
        "${template_project}" \
        "${work_root}/projects/${name}"
done

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

for name in s02f s03a s03b s03d; do
    package="$(find_package "${work_root}/projects/${name}")"
    install_package "${package}" "${work_root}/install/${name}"
done

baseline_env="$(find_env_file "${work_root}/install/s02f")"
s03a_env="$(find_env_file "${work_root}/install/s03a")"
s03b_env="$(find_env_file "${work_root}/install/s03b")"
s03d_env="$(find_env_file "${work_root}/install/s03d")"

run_with_opp() {
    local env_file="$1"
    shift
    bash -c 'set -euo pipefail; source "$1"; shift; exec "$@"' \
        _ "${env_file}" "$@"
}

gate_stage="isolated_import"
run_with_opp "${baseline_env}" python3 -c \
    'import square_sum_v1_validation_lib; print("S02F_IMPORT_OK")'
run_with_opp "${s03a_env}" python3 -c \
    'import square_sum_v1_validation_lib; print("S03A_IMPORT_OK")'
run_with_opp "${s03b_env}" python3 -c \
    'import square_sum_v1_validation_lib; print("S03B_IMPORT_OK")'
run_with_opp "${s03d_env}" python3 -c \
    'import square_sum_v1_validation_lib; print("S03D_IMPORT_OK")'

gate_stage="candidate_correctness"
run_with_opp "${s03a_env}" python3 \
    diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S03A --tier atlas --dtypes fp16 bf16 fp32 \
    | tee "${detail_root}/s03a_domain_atlas.log"
run_with_opp "${s03b_env}" python3 \
    diagnostics/squaresum_s03b_correctness_20260806.py S03B \
    | tee "${detail_root}/s03b_correctness.log"
run_with_opp "${s03d_env}" python3 \
    diagnostics/squaresum_singleton_gap_correctness_20260805.py \
    | tee "${detail_root}/s03d_correctness.log"

gate_stage="s03a_official_aba_profile"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_matrix_20260806.sh \
    bf16 S02F_A SquareSumV1/s03a_baseline_a core
run_with_opp "${s03a_env}" bash \
    diagnostics/run_squaresum_official_matrix_20260806.sh \
    bf16 S03A SquareSumV1/s03a_candidate core
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_matrix_20260806.sh \
    bf16 S02F_B SquareSumV1/s03a_baseline_b core

gate_stage="s03b_official_aba_profile"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_C SquareSumV1/s03b_baseline_a strided_splitk fp16 bf16 fp32
run_with_opp "${s03b_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S03B SquareSumV1/s03b_candidate strided_splitk fp16 bf16 fp32
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_D SquareSumV1/s03b_baseline_b strided_splitk fp16 bf16 fp32

gate_stage="s03d_official_aba_profile"
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_E SquareSumV1/s03d_baseline_a singleton_gap_s03d fp16 bf16 fp32
run_with_opp "${s03d_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S03D SquareSumV1/s03d_candidate singleton_gap_s03d fp16 bf16 fp32
run_with_opp "${baseline_env}" bash \
    diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
    S02F_F SquareSumV1/s03d_baseline_b singleton_gap_s03d fp16 bf16 fp32

gate_stage="performance_comparison"
set +e
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/s03a_baseline_a" \
    --candidate "${detail_root}/s03a_candidate" \
    --baseline-b "${detail_root}/s03a_baseline_b" \
    --minimum-improvement-percent 2 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${detail_root}/s03a_comparison.json"
s03a_status=$?
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/s03b_baseline_a" \
    --candidate "${detail_root}/s03b_candidate" \
    --baseline-b "${detail_root}/s03b_baseline_b" \
    --minimum-improvement-percent 10 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${detail_root}/s03b_comparison.json"
s03b_status=$?
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/s03d_baseline_a" \
    --candidate "${detail_root}/s03d_candidate" \
    --baseline-b "${detail_root}/s03d_baseline_b" \
    --minimum-improvement-percent 10 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${detail_root}/s03d_comparison.json"
s03d_status=$?
set -e

python3 - \
    "${artifact_dir}/result.json" \
    "${detail_root}/s03a_comparison.json" "${s03a_status}" \
    "${detail_root}/s03b_comparison.json" "${s03b_status}" \
    "${detail_root}/s03d_comparison.json" "${s03d_status}" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
items = {}
for name, result_path, status in (
    ("S03A", Path(sys.argv[2]), int(sys.argv[3])),
    ("S03B", Path(sys.argv[4]), int(sys.argv[5])),
    ("S03D", Path(sys.argv[6]), int(sys.argv[7])),
):
    details = None
    if result_path.exists():
        details = json.loads(result_path.read_text(encoding="utf-8"))
    items[name] = {
        "passed": status == 0,
        "exit_status": status,
        "details": details,
    }
passing = [name for name, item in items.items() if item["passed"]]
value = {
    "schema_version": 1,
    "passed": bool(passing),
    "baseline": "S02F",
    "passing_candidates": passing,
    "candidates": items,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
PY
result_ready=1

if (( s03a_status != 0 && s03b_status != 0 && s03d_status != 0 )); then
    echo "S03_GATE_REJECT result=${artifact_dir}/result.json"
    exit 1
fi

echo "S03_GATE_PASS result=${artifact_dir}/result.json log=${artifact_dir}/run.log"
