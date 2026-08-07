#!/usr/bin/env bash
set -Eeuo pipefail

stage="${1:?usage: $0 STAGE TEMPLATE_PROJECT WORK_ROOT}"
template_project="${2:?usage: $0 STAGE TEMPLATE_PROJECT WORK_ROOT}"
work_root="${3:?usage: $0 STAGE TEMPLATE_PROJECT WORK_ROOT}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1/${stage}"
result_json="${artifact_dir}/result.json"
completed=0

case "${stage}" in
    s03m)
        baseline_source="${repo_root}/baselines/squaresum_s02f_global_best_20260806/SquareSumV1"
        candidate_source="${repo_root}/candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/SquareSumV1"
        correctness_stages=(s03m)
        ;;
    s03n)
        baseline_source="${repo_root}/candidates/squaresum_s03m_s02f_strided_grouped_clean_20260807/SquareSumV1"
        candidate_source="${repo_root}/candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807/SquareSumV1"
        correctness_stages=(s03m s03n)
        ;;
    s03o)
        baseline_source="${repo_root}/candidates/squaresum_s03n_arbitrary_grouped_reduce_20260807/SquareSumV1"
        candidate_source="${repo_root}/candidates/squaresum_s03o_unaligned_grouped_rows_20260807/SquareSumV1"
        correctness_stages=(s03m s03n s03o)
        ;;
    *)
        echo "unsupported stage: ${stage}" >&2
        exit 2
        ;;
esac

atomic_failure_result() {
    local status="$1"
    if (( completed == 1 )) || [[ ! -d "${artifact_dir}" ]]; then
        return
    fi
    RESULT_JSON="${result_json}" STAGE="${stage}" STATUS="${status}" \
        BASELINE_SOURCE="${baseline_source}" CANDIDATE_SOURCE="${candidate_source}" \
        python3 - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["RESULT_JSON"])


def relative_source(value):
    path = Path(value)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def load_optional(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


value = {
    "schema_version": 1,
    "stage": os.environ["STAGE"],
    "passed": False,
    "exit_status": int(os.environ["STATUS"]),
    "baseline_source": relative_source(os.environ["BASELINE_SOURCE"]),
    "candidate_source": relative_source(os.environ["CANDIDATE_SOURCE"]),
    "correctness": load_optional(output.parent / "SquareSumV1" / os.environ["STAGE"] / "correctness.json"),
    "target_comparison": load_optional(output.parent / "SquareSumV1" / os.environ["STAGE"] / "target_comparison.json"),
    "control_comparison": load_optional(output.parent / "SquareSumV1" / os.environ["STAGE"] / "control_comparison.json"),
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}
trap 'status=$?; atomic_failure_result "${status}"' EXIT

for variable in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH; do
    value="${!variable:-}"
    if [[ "${value}" != /home/ma-user/Ascend/cann-8.5.0* ]]; then
        echo "${variable} does not select CANN 8.5.0: ${value:-<unset>}" >&2
        exit 2
    fi
done

if [[ ! -d "${template_project}" ]] || [[ ! -d "${baseline_source}" ]] || [[ ! -d "${candidate_source}" ]]; then
    echo "template, baseline, or candidate source is missing" >&2
    exit 2
fi
if [[ -e "${work_root}" ]] || [[ -z "${work_root}" ]] || [[ "${work_root}" == "/" ]]; then
    echo "work root is unsafe or already exists: ${work_root}" >&2
    exit 2
fi
work_parent="$(dirname -- "${work_root}")"
mkdir -p -- "${work_parent}"
work_parent="$(cd -- "${work_parent}" && pwd -P)"
work_root="${work_parent}/$(basename -- "${work_root}")"
if [[ "${work_root}" == "${repo_root}" ]]; then
    echo "work root must not be the repository root" >&2
    exit 2
fi

rm -rf -- "${artifact_dir}"
mkdir -p -- "${detail_root}" "${work_root}/install"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1
cd -- "${repo_root}"
export PYTHONPATH="${repo_root}/validation/SquareSumV1${PYTHONPATH:+:${PYTHONPATH}}"

echo "SQUARESUM_STAGE_GATE_START stage=${stage} work=${work_root}"
python3 diagnostics/squaresum_s03m_s03o_static_20260807.py --stage "${stage}" --model-only
python3 diagnostics/validate_squaresum_profile_matrix_20260806.py >/dev/null

bash diagnostics/build_squaresum_source_20260806.sh \
    "${baseline_source}" "${template_project}" "${work_root}/baseline_project"
bash diagnostics/build_squaresum_source_20260806.sh \
    "${candidate_source}" "${template_project}" "${work_root}/candidate_project"

find_package() {
    local project="$1"
    mapfile -t packages < <(find "${project}/build_out" -maxdepth 1 -type f -name 'custom_opp*.run' -print)
    if (( ${#packages[@]} != 1 )); then
        echo "expected one package below ${project}, found ${#packages[@]}" >&2
        return 1
    fi
    printf '%s\n' "${packages[0]}"
}

find_env() {
    local install_root="$1"
    mapfile -t env_files < <(find "${install_root}" -type f -name set_env.bash -print)
    if (( ${#env_files[@]} != 1 )); then
        echo "expected one set_env.bash below ${install_root}, found ${#env_files[@]}" >&2
        return 1
    fi
    printf '%s\n' "${env_files[0]}"
}

run_with_opp() {
    local env_file="$1"
    shift
    bash -c 'set -eo pipefail; source "$1"; set -u; shift; exec "$@"' \
        _ "${env_file}" "$@"
}

baseline_package="$(find_package "${work_root}/baseline_project")"
candidate_package="$(find_package "${work_root}/candidate_project")"
chmod u+x -- "${baseline_package}" "${candidate_package}"
"${baseline_package}" --install-path="${work_root}/install/baseline" >/dev/null
"${candidate_package}" --install-path="${work_root}/install/candidate" >/dev/null
baseline_env="$(find_env "${work_root}/install/baseline")"
candidate_env="$(find_env "${work_root}/install/candidate")"

run_with_opp "${candidate_env}" python3 -c \
    'import torch, torch_npu, square_sum_v1_validation_lib; print("SQUARESUM_STAGE_IMPORT_OK")'

run_suite() {
    local name="$1"
    shift
    echo "CORRECTNESS_SUITE_START name=${name}"
    run_with_opp "${candidate_env}" "$@" | tee "${detail_root}/${name}.log"
    echo "CORRECTNESS_SUITE_PASS name=${name}"
}

run_suite directed python3 validation/SquareSumV1/test_matrix.py
run_suite bf16_semantics python3 validation/SquareSumV1/bf16_semantic_probe.py
run_suite random python3 validation/SquareSumV1/random_matrix.py
run_suite extended python3 validation/SquareSumV1/extended_matrix.py
run_suite s03j_boundary python3 diagnostics/squaresum_s03j_boundary_correctness_20260807.py
run_suite s03k_boundary python3 diagnostics/squaresum_s03k_middle_full_rows_correctness_20260807.py "${stage^^}"
for correctness_stage in "${correctness_stages[@]}"; do
    run_suite "${correctness_stage}_boundary" python3 \
        diagnostics/squaresum_s03m_s03o_correctness_20260807.py \
        --stage "${correctness_stage}" --label "${stage^^}"
done

STAGE="${stage}" OUTPUT="${detail_root}/correctness.json" python3 - <<'PY'
import json
import os
from pathlib import Path

stage = os.environ["STAGE"]
stage_counts = {"s03m": 105, "s03n": 45, "s03o": 45}
order = ("s03m", "s03n", "s03o")
included = order[: order.index(stage) + 1]
suites = {
    "directed": 46,
    "bf16_semantics": 4,
    "random": 150,
    "extended": 726,
    "s03j_boundary": 135,
    "s03k_boundary": 69,
}
suites.update({name: stage_counts[name] for name in included})
value = {"passed": True, "suites": suites, "total": sum(suites.values())}
output = Path(os.environ["OUTPUT"])
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

profile_matrix() {
    local env_file="$1"
    local label="$2"
    local output_name="$3"
    local tier="$4"
    run_with_opp "${env_file}" bash \
        diagnostics/run_squaresum_official_multidtype_matrix_20260806.sh \
        "${label}" "${output_name}" "${tier}" fp16 bf16 fp32
}

profile_root="SquareSumV1/${stage}/profiles"
profile_matrix "${baseline_env}" "${stage^^}_BASELINE_A_TARGET" "${profile_root}/baseline_a_target" "${stage}_target"
profile_matrix "${baseline_env}" "${stage^^}_BASELINE_A_CONTROL" "${profile_root}/baseline_a_control" "${stage}_control"
profile_matrix "${candidate_env}" "${stage^^}_CANDIDATE_TARGET" "${profile_root}/candidate_target" "${stage}_target"
profile_matrix "${candidate_env}" "${stage^^}_CANDIDATE_CONTROL" "${profile_root}/candidate_control" "${stage}_control"
profile_matrix "${baseline_env}" "${stage^^}_BASELINE_B_TARGET" "${profile_root}/baseline_b_target" "${stage}_target"
profile_matrix "${baseline_env}" "${stage^^}_BASELINE_B_CONTROL" "${profile_root}/baseline_b_control" "${stage}_control"

set +e
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/profiles/baseline_a_target" \
    --candidate "${detail_root}/profiles/candidate_target" \
    --baseline-b "${detail_root}/profiles/baseline_b_target" \
    --minimum-improvement-percent 50 \
    --minimum-point-improvement-percent 50 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${detail_root}/target_comparison.json"
target_status=$?
python3 diagnostics/compare_squaresum_official_matrices_20260806.py \
    --baseline-a "${detail_root}/profiles/baseline_a_control" \
    --candidate "${detail_root}/profiles/candidate_control" \
    --baseline-b "${detail_root}/profiles/baseline_b_control" \
    --minimum-improvement-percent -100 \
    --maximum-regression-percent 3 \
    --maximum-baseline-drift-percent 3 \
    --output "${detail_root}/control_comparison.json"
control_status=$?
set -e

mkdir -p -- "${detail_root}"
cp -- "${candidate_package}" "${detail_root}/custom_opp_euleros_aarch64.run"
sha256sum \
    "${candidate_source}/op_host/square_sum_v1.cpp" \
    "${candidate_source}/op_kernel/square_sum_v1.cpp" \
    "${detail_root}/custom_opp_euleros_aarch64.run" \
    > "${detail_root}/sha256.txt"

STAGE="${stage}" RESULT_JSON="${result_json}" DETAIL_ROOT="${detail_root}" \
BASELINE_SOURCE="${baseline_source}" CANDIDATE_SOURCE="${candidate_source}" \
TARGET_STATUS="${target_status}" CONTROL_STATUS="${control_status}" python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

repo = Path.cwd()
detail = Path(os.environ["DETAIL_ROOT"])
package = detail / "custom_opp_euleros_aarch64.run"


def load(name):
    return json.loads((detail / name).read_text(encoding="utf-8"))


def relative(value):
    return str(Path(value).relative_to(repo))


target = load("target_comparison.json")
control = load("control_comparison.json")
passed = (
    int(os.environ["TARGET_STATUS"]) == 0
    and int(os.environ["CONTROL_STATUS"]) == 0
    and target["passed"]
    and control["passed"]
)
value = {
    "schema_version": 1,
    "stage": os.environ["STAGE"],
    "passed": passed,
    "baseline_source": relative(os.environ["BASELINE_SOURCE"]),
    "candidate_source": relative(os.environ["CANDIDATE_SOURCE"]),
    "run_package": relative(package),
    "correctness": load("correctness.json"),
    "target_comparison": target,
    "control_comparison": control,
    "sha256": {
        "run_package": hashlib.sha256(package.read_bytes()).hexdigest(),
        "op_host": hashlib.sha256((Path(os.environ["CANDIDATE_SOURCE"]) / "op_host" / "square_sum_v1.cpp").read_bytes()).hexdigest(),
        "op_kernel": hashlib.sha256((Path(os.environ["CANDIDATE_SOURCE"]) / "op_kernel" / "square_sum_v1.cpp").read_bytes()).hexdigest(),
    },
}
output = Path(os.environ["RESULT_JSON"])
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(value, ensure_ascii=False, indent=2))
raise SystemExit(0 if passed else 2)
PY

completed=1
echo "SQUARESUM_STAGE_GATE_PASS stage=${stage} result=${result_json}"
