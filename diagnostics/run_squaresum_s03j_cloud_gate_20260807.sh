#!/usr/bin/env bash
set -euo pipefail

template_project="${1:?usage: $0 TEMPLATE_PROJECT BASELINE_INSTALL WORK_ROOT}"
baseline_install="${2:?usage: $0 TEMPLATE_PROJECT BASELINE_INSTALL WORK_ROOT}"
work_root="${3:?usage: $0 TEMPLATE_PROJECT BASELINE_INSTALL WORK_ROOT}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
candidate_source="${repo_root}/candidates/squaresum_s03j_s02f_fast2_small_inner_safe_20260807/SquareSumV1"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1/s03j_fast2_small_inner_safe"

if [[ -e "${work_root}" ]]; then
    echo "work root already exists: ${work_root}" >&2
    exit 2
fi
mkdir -p -- "${work_root}" "${work_root}/install"
rm -rf -- "${artifact_dir}"
mkdir -p -- "${detail_root}"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1

find_package() {
    find "$1/build_out" -maxdepth 1 -type f -name 'custom_opp*.run' -print -quit
}

find_env() {
    find "$1" -type f -name set_env.bash -print -quit
}

run_with_opp() {
    local env_file="$1"
    shift
    bash -c 'set -eo pipefail; source "$1"; set -u; shift; exec "$@"' \
        _ "${env_file}" "$@"
}

cd -- "${repo_root}"
export PYTHONPATH="${repo_root}/validation/SquareSumV1${PYTHONPATH:+:${PYTHONPATH}}"
echo "S03J_GATE_START work=${work_root}"
python3 diagnostics/squaresum_s03j_fast2_small_inner_safe_static_20260807.py

bash diagnostics/build_squaresum_source_20260806.sh \
    "${candidate_source}" "${template_project}" "${work_root}/project"
package="$(find_package "${work_root}/project")"
test -n "${package}"
chmod u+x -- "${package}"
"${package}" --install-path="${work_root}/install" >/dev/null

baseline_env="$(find_env "${baseline_install}")"
candidate_env="$(find_env "${work_root}/install")"
test -n "${baseline_env}"
test -n "${candidate_env}"
run_with_opp "${candidate_env}" python3 -c \
    'import torch, torch_npu, square_sum_v1_validation_lib; print("S03J_IMPORT_OK")'

for tier in core atlas; do
    run_with_opp "${baseline_env}" python3 \
        diagnostics/squaresum_domain_event_atlas_20260806.py \
        --label "S02F_B1_${tier}" --tier "${tier}" \
        --dtypes fp16 bf16 fp32 \
        | tee "${detail_root}/baseline_1_${tier}.log"
    run_with_opp "${candidate_env}" python3 \
        diagnostics/squaresum_domain_event_atlas_20260806.py \
        --label "S03J_${tier}" --tier "${tier}" \
        --dtypes fp16 bf16 fp32 \
        | tee "${detail_root}/candidate_${tier}.log"
    run_with_opp "${baseline_env}" python3 \
        diagnostics/squaresum_domain_event_atlas_20260806.py \
        --label "S02F_B2_${tier}" --tier "${tier}" \
        --dtypes fp16 bf16 fp32 \
        | tee "${detail_root}/baseline_2_${tier}.log"
done

python3 - "${artifact_dir}/result.json" "${detail_root}" <<'PY'
import json
import os
import re
import statistics
import sys
from pathlib import Path

output = Path(sys.argv[1])
root = Path(sys.argv[2])
pattern = re.compile(
    r"DOMAIN_EVENT_ATLAS .*?case=(\S+).*?dtype=(\S+).*?median_us=([0-9.]+).*? PASS$"
)


def read(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            values[(match.group(1), match.group(2))] = float(match.group(3))
    return values


points = []
for tier in ("core", "atlas"):
    first = read(root / f"baseline_1_{tier}.log")
    candidate = read(root / f"candidate_{tier}.log")
    second = read(root / f"baseline_2_{tier}.log")
    assert first.keys() == candidate.keys() == second.keys()
    for key in sorted(first):
        baseline = statistics.median((first[key], second[key]))
        improvement = (baseline - candidate[key]) / baseline * 100.0
        drift = abs(first[key] - second[key]) / baseline * 100.0
        points.append(
            {
                "tier": tier,
                "case": key[0],
                "dtype": key[1],
                "baseline_1_us": first[key],
                "candidate_us": candidate[key],
                "baseline_2_us": second[key],
                "baseline_median_us": baseline,
                "improvement_percent": improvement,
                "baseline_drift_percent": drift,
            }
        )

primary_names = {"atlas_fast2_inner2", "atlas_fast2_inner8"}
primary = [item for item in points if item["case"] in primary_names]
controls = [
    item
    for item in points
    if not (
        item["case"].startswith("atlas_fast2_inner")
        or item["case"].startswith("fast2_middle")
    )
]
baseline_total = sum(item["baseline_median_us"] for item in primary)
candidate_total = sum(item["candidate_us"] for item in primary)
aggregate = (baseline_total - candidate_total) / baseline_total * 100.0
minimum_primary = min(item["improvement_percent"] for item in primary)
worst_control = max(-item["improvement_percent"] for item in controls)
maximum_drift = max(item["baseline_drift_percent"] for item in points)
passed = (
    len(primary) == 6
    and aggregate >= 20.0
    and minimum_primary >= 15.0
    and worst_control <= 8.0
    and maximum_drift <= 12.0
)
value = {
    "schema_version": 1,
    "passed": passed,
    "baseline": "S02F",
    "candidate": "S03J",
    "hypothesis": "selectively port S02AY middle-tree handling, allocate the required FP32 finalizer buffer, and use 16K only for non-tree fastPath2 reduce>=2048 and inner<=64",
    "primary_aggregate_improvement_percent": aggregate,
    "minimum_primary_improvement_percent": minimum_primary,
    "worst_control_regression_percent": worst_control,
    "maximum_baseline_drift_percent": maximum_drift,
    "thresholds": {
        "minimum_primary_aggregate_percent": 20.0,
        "minimum_primary_point_percent": 15.0,
        "maximum_control_regression_percent": 8.0,
        "maximum_baseline_drift_percent": 12.0,
    },
    "points": points,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, output)
print(json.dumps(value, ensure_ascii=False, indent=2))
raise SystemExit(0 if passed else 1)
PY

echo "S03J_GATE_PASS result=${artifact_dir}/result.json"
