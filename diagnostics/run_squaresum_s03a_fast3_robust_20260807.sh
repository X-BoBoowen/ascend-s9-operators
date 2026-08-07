#!/usr/bin/env bash
set -euo pipefail

s03_run="${1:?usage: $0 S03_RUN}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
s03_run="$(cd -- "${s03_run}" && pwd -P)"
artifact_dir="${repo_root}/artifact"
run_root="${artifact_dir}/SquareSumV1/s03a_fast3_bf16_robust"
rm -rf -- "${artifact_dir}"
mkdir -p -- "${run_root}"
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

profile() {
    local env_file="$1"
    local label="$2"
    local output="$3"
    run_with_opp "${env_file}" bash \
        diagnostics/run_squaresum_official_profile_20260806.sh \
        fast3_noncontiguous_last bf16 "${label}" "${output}"
}

cd -- "${repo_root}"
validation_dir="${repo_root}/validation/SquareSumV1"
export PYTHONPATH="${validation_dir}${PYTHONPATH:+:${PYTHONPATH}}"
baseline_env="$(find_env_file "${s03_run}/install/s02f")"
candidate_env="$(find_env_file "${s03_run}/install/s03a")"

echo "S03A_FAST3_BF16_ROBUST_START order=B1,C1,B2,C2,B3"
profile "${baseline_env}" S02F_B1 "${run_root}/baseline_1"
profile "${candidate_env}" S03A_C1 "${run_root}/candidate_1"
profile "${baseline_env}" S02F_B2 "${run_root}/baseline_2"
profile "${candidate_env}" S03A_C2 "${run_root}/candidate_2"
profile "${baseline_env}" S02F_B3 "${run_root}/baseline_3"

python3 - "${artifact_dir}/result.json" "${run_root}" <<'PY'
import json
import os
import statistics
import sys
from pathlib import Path

output = Path(sys.argv[1])
root = Path(sys.argv[2])

def value(name):
    document = json.loads((root / name / "result.json").read_text(encoding="utf-8"))
    return float(document["official_compatible_time"])

baseline = [value(f"baseline_{index}") for index in range(1, 4)]
candidate = [value(f"candidate_{index}") for index in range(1, 3)]
baseline_median = statistics.median(baseline)
candidate_median = statistics.median(candidate)
improvement = (baseline_median - candidate_median) / baseline_median * 100.0
baseline_spread = (max(baseline) - min(baseline)) / baseline_median * 100.0
candidate_spread = (max(candidate) - min(candidate)) / candidate_median * 100.0
strict_dominance = max(candidate) < min(baseline)
passed = (
    improvement >= 5.0
    and baseline_spread <= 8.0
    and candidate_spread <= 5.0
    and strict_dominance
)
value = {
    "schema_version": 1,
    "passed": passed,
    "baseline": "S02F",
    "candidate": "S03A",
    "case": "fast3_noncontiguous_last",
    "dtype": "bf16",
    "order": ["baseline_1", "candidate_1", "baseline_2", "candidate_2", "baseline_3"],
    "baseline_samples": baseline,
    "candidate_samples": candidate,
    "baseline_median": baseline_median,
    "candidate_median": candidate_median,
    "improvement_percent": improvement,
    "baseline_spread_percent": baseline_spread,
    "candidate_spread_percent": candidate_spread,
    "strict_dominance": strict_dominance,
    "thresholds": {
        "minimum_improvement_percent": 5.0,
        "maximum_baseline_spread_percent": 8.0,
        "maximum_candidate_spread_percent": 5.0,
        "require_candidate_max_below_baseline_min": True,
    },
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(value, ensure_ascii=False, indent=2))
if not passed:
    raise SystemExit(1)
PY
echo "S03A_FAST3_BF16_ROBUST_PASS result=${artifact_dir}/result.json"
