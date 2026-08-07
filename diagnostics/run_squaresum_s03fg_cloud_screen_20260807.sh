#!/usr/bin/env bash
set -euo pipefail

template_project="${1:?usage: $0 TEMPLATE_PROJECT S03E_RUN WORK_ROOT}"
baseline_run="${2:?usage: $0 TEMPLATE_PROJECT S03E_RUN WORK_ROOT}"
work_root="${3:?usage: $0 TEMPLATE_PROJECT S03E_RUN WORK_ROOT}"
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

s03f_source="${repo_root}/candidates/squaresum_s03f_s03e_short_tail_splitk_20260807/SquareSumV1"
s03g_source="${repo_root}/candidates/squaresum_s03g_s03e_last_output16_20260807/SquareSumV1"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1"
mkdir -p -- "${work_root}/projects" "${work_root}/install"
rm -rf -- "${artifact_dir}"
mkdir -p -- "${detail_root}"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1

stage="initialization"
result_ready=0
write_failure() {
    local status="$1"
    if (( status == 0 || result_ready == 1 )); then
        return
    fi
    python3 - "${artifact_dir}/result.json" "${stage}" "${status}" <<'PY'
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
    "message": "S03F/S03G cloud screen stopped before completion",
}
temporary = path.with_name(path.name + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}
trap 'write_failure "$?"' EXIT

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
echo "S03FG_SCREEN_START baseline=${baseline_run} work=${work_root}"

stage="preflight"
python3 diagnostics/validate_squaresum_profile_matrix_20260806.py
python3 diagnostics/squaresum_s03f_short_tail_static_20260807.py
python3 diagnostics/squaresum_s02cs_short_tail_semantics_model_20260806.py
python3 diagnostics/squaresum_s03g_last_output16_static_20260807.py
python3 diagnostics/squaresum_s02ct_last_output16_semantics_model_20260806.py

stage="build"
for name in s03f s03g; do
    if [[ "${name}" == "s03f" ]]; then
        source_path="${s03f_source}"
    else
        source_path="${s03g_source}"
    fi
    bash diagnostics/build_squaresum_source_20260806.sh \
        "${source_path}" "${template_project}" "${work_root}/projects/${name}"
    package="$(find_package "${work_root}/projects/${name}")"
    mkdir -- "${work_root}/install/${name}"
    chmod u+x -- "${package}"
    "${package}" --install-path="${work_root}/install/${name}"
done

baseline_env="$(find_env_file "${baseline_run}/install/s03e")"
s03f_env="$(find_env_file "${work_root}/install/s03f")"
s03g_env="$(find_env_file "${work_root}/install/s03g")"
stage="isolated_import"
for item in "${baseline_env}:S03E" "${s03f_env}:S03F" "${s03g_env}:S03G"; do
    env_file="${item%%:*}"
    label="${item##*:}"
    run_with_opp "${env_file}" python3 -c \
        "import torch, torch_npu, square_sum_v1_validation_lib; print('${label}_IMPORT_OK')"
done

stage="event_screen"
run_with_opp "${baseline_env}" python3 diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S03E_F --tier short_tail_splitk --dtypes fp16 bf16 fp32 \
    | tee "${detail_root}/s03f_baseline_event.log"
run_with_opp "${s03f_env}" python3 diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S03F --tier short_tail_splitk --dtypes fp16 bf16 fp32 \
    | tee "${detail_root}/s03f_candidate_event.log"
run_with_opp "${baseline_env}" python3 diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S03E_G --tier last_output_splitk --dtypes fp16 bf16 \
    | tee "${detail_root}/s03g_baseline_event.log"
run_with_opp "${s03g_env}" python3 diagnostics/squaresum_domain_event_atlas_20260806.py \
    --label S03G --tier last_output_splitk --dtypes fp16 bf16 \
    | tee "${detail_root}/s03g_candidate_event.log"

stage="comparison"
set +e
python3 diagnostics/compare_squaresum_event_atlas_20260806.py \
    --baseline "${detail_root}/s03f_baseline_event.log" \
    --candidate "${detail_root}/s03f_candidate_event.log" \
    --minimum-improvement-percent 10 --maximum-regression-percent 5 \
    --output "${detail_root}/s03f_event_comparison.json"
s03f_status=$?
python3 diagnostics/compare_squaresum_event_atlas_20260806.py \
    --baseline "${detail_root}/s03g_baseline_event.log" \
    --candidate "${detail_root}/s03g_candidate_event.log" \
    --minimum-improvement-percent 10 --maximum-regression-percent 5 \
    --output "${detail_root}/s03g_event_comparison.json"
s03g_status=$?
set -e

python3 - "${artifact_dir}/result.json" \
    "${detail_root}/s03f_event_comparison.json" "${s03f_status}" \
    "${detail_root}/s03g_event_comparison.json" "${s03g_status}" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
items = {}
for name, path, status in (
    ("S03F", Path(sys.argv[2]), int(sys.argv[3])),
    ("S03G", Path(sys.argv[4]), int(sys.argv[5])),
):
    items[name] = {
        "passed": status == 0,
        "exit_status": status,
        "details": json.loads(path.read_text(encoding="utf-8")),
    }
passing = [name for name, item in items.items() if item["passed"]]
value = {
    "schema_version": 1,
    "passed": bool(passing),
    "baseline": "S03E",
    "passing_candidates": passing,
    "candidates": items,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
result_ready=1

if (( s03f_status != 0 && s03g_status != 0 )); then
    echo "S03FG_SCREEN_REJECT result=${artifact_dir}/result.json"
    exit 1
fi
echo "S03FG_SCREEN_PASS result=${artifact_dir}/result.json"
