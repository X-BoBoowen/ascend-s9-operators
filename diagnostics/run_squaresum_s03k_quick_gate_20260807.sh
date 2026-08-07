#!/usr/bin/env bash
set -euo pipefail

candidate_source="${1:?usage: $0 SOURCE TEMPLATE BASELINE_INSTALL WORK_ROOT}"
template_project="${2:?usage: $0 SOURCE TEMPLATE BASELINE_INSTALL WORK_ROOT}"
baseline_install="${3:?usage: $0 SOURCE TEMPLATE BASELINE_INSTALL WORK_ROOT}"
work_root="${4:?usage: $0 SOURCE TEMPLATE BASELINE_INSTALL WORK_ROOT}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1/s03k_middle_full_rows"
matrix="${repo_root}/diagnostics/squaresum_s03k_middle_full_rows_matrix_20260807.json"

if [[ -e "${work_root}" ]]; then
    echo "work root already exists: ${work_root}" >&2
    exit 2
fi
mkdir -p -- "${work_root}/install"
rm -rf -- "${artifact_dir}"
mkdir -p -- "${detail_root}"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1

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
python3 diagnostics/validate_squaresum_profile_matrix_20260806.py \
    --matrix "${matrix}"
bash diagnostics/build_squaresum_source_20260806.sh \
    "${candidate_source}" "${template_project}" "${work_root}/project"
package="$(find "${work_root}/project/build_out" -maxdepth 1 \
    -type f -name 'custom_opp*.run' -print -quit)"
test -n "${package}"
chmod u+x -- "${package}"
"${package}" --install-path="${work_root}/install" >/dev/null

baseline_env="$(find_env "${baseline_install}")"
candidate_env="$(find_env "${work_root}/install")"
test -n "${baseline_env}"
test -n "${candidate_env}"

for phase in baseline_1 candidate baseline_2; do
    env_file="${baseline_env}"
    [[ "${phase}" == candidate ]] && env_file="${candidate_env}"
    run_with_opp "${env_file}" python3 \
        diagnostics/squaresum_domain_event_atlas_20260806.py \
        --label "S03K_${phase}" --tier s03k_target \
        --matrix "${matrix}" --dtypes fp16 bf16 fp32 \
        | tee "${detail_root}/${phase}.log"
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


first = read(root / "baseline_1.log")
candidate = read(root / "candidate.log")
second = read(root / "baseline_2.log")
assert first.keys() == candidate.keys() == second.keys()
points = []
for key in sorted(first):
    baseline = statistics.median((first[key], second[key]))
    improvement = (baseline - candidate[key]) / baseline * 100.0
    drift = abs(first[key] - second[key]) / baseline * 100.0
    points.append({
        "case": key[0],
        "dtype": key[1],
        "baseline_1_us": first[key],
        "candidate_us": candidate[key],
        "baseline_2_us": second[key],
        "baseline_median_us": baseline,
        "improvement_percent": improvement,
        "baseline_drift_percent": drift,
    })
baseline_total = sum(item["baseline_median_us"] for item in points)
candidate_total = sum(item["candidate_us"] for item in points)
aggregate = (baseline_total - candidate_total) / baseline_total * 100.0
minimum = min(item["improvement_percent"] for item in points)
maximum_drift = max(item["baseline_drift_percent"] for item in points)
passed = (
    len(points) == 15
    and aggregate >= 40.0
    and minimum >= 15.0
    and maximum_drift <= 12.0
)
value = {
    "schema_version": 1,
    "passed": passed,
    "baseline": "S03J",
    "candidate": "S03K",
    "target_aggregate_improvement_percent": aggregate,
    "minimum_target_improvement_percent": minimum,
    "maximum_baseline_drift_percent": maximum_drift,
    "points": points,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(value, indent=2))
raise SystemExit(0 if passed else 1)
PY

echo "S03K_QUICK_GATE_PASS result=${artifact_dir}/result.json"
