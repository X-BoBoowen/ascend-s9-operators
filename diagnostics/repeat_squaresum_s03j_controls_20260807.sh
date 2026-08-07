#!/usr/bin/env bash
set -euo pipefail

baseline_install="${1:?usage: $0 BASELINE_INSTALL CANDIDATE_INSTALL}"
candidate_install="${2:?usage: $0 BASELINE_INSTALL CANDIDATE_INSTALL}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1/s03j_control_repeat"
matrix="${repo_root}/diagnostics/squaresum_s03j_control_matrix_20260807.json"

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

baseline_env="$(find_env "${baseline_install}")"
candidate_env="$(find_env "${candidate_install}")"
cd -- "${repo_root}"
export PYTHONPATH="${repo_root}/validation/SquareSumV1${PYTHONPATH:+:${PYTHONPATH}}"

for item in "baseline_1:${baseline_env}" "candidate:${candidate_env}" "baseline_2:${baseline_env}"; do
    label="${item%%:*}"
    env_file="${item#*:}"
    run_with_opp "${env_file}" python3 \
        diagnostics/squaresum_domain_event_atlas_20260806.py \
        --label "${label}" --tier s03j_control --matrix "${matrix}" \
        --dtypes fp16 bf16 fp32 \
        | tee "${detail_root}/${label}.log"
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


def read(name):
    values = {}
    for line in (root / f"{name}.log").read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            values[(match.group(1), match.group(2))] = float(match.group(3))
    return values


first = read("baseline_1")
candidate = read("candidate")
second = read("baseline_2")
assert first.keys() == candidate.keys() == second.keys()
points = []
for key in sorted(first):
    baseline = statistics.median((first[key], second[key]))
    points.append(
        {
            "case": key[0],
            "dtype": key[1],
            "baseline_1_us": first[key],
            "candidate_us": candidate[key],
            "baseline_2_us": second[key],
            "regression_percent": (candidate[key] - baseline) / baseline * 100.0,
            "baseline_drift_percent": abs(first[key] - second[key]) / baseline * 100.0,
        }
    )
worst_regression = max(item["regression_percent"] for item in points)
maximum_drift = max(item["baseline_drift_percent"] for item in points)
passed = worst_regression <= 8.0 and maximum_drift <= 12.0
value = {
    "schema_version": 1,
    "passed": passed,
    "baseline": "S02F",
    "candidate": "S03J",
    "worst_control_regression_percent": worst_regression,
    "maximum_baseline_drift_percent": maximum_drift,
    "points": points,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(value, indent=2))
raise SystemExit(0 if passed else 1)
PY

echo "S03J_CONTROL_REPEAT_PASS result=${artifact_dir}/result.json"
