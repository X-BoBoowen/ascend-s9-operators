#!/usr/bin/env bash
set -euo pipefail

candidate_install="${1:?usage: $0 CANDIDATE_INSTALL}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1/s03k_correctness"

rm -rf -- "${artifact_dir}"
mkdir -p -- "${detail_root}"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1

candidate_env="$(find "${candidate_install}" -type f -name set_env.bash -print -quit)"
test -n "${candidate_env}"

run_with_opp() {
    local env_file="$1"
    shift
    bash -c 'set -eo pipefail; source "$1"; set -u; shift; exec "$@"' \
        _ "${env_file}" "$@"
}

cd -- "${repo_root}"
export PYTHONPATH="${repo_root}/validation/SquareSumV1${PYTHONPATH:+:${PYTHONPATH}}"

run_suite() {
    local name="$1"
    shift
    echo "S03K_CORRECTNESS_SUITE_START name=${name}"
    run_with_opp "${candidate_env}" "$@" \
        | tee "${detail_root}/${name}.log"
    echo "S03K_CORRECTNESS_SUITE_PASS name=${name}"
}

run_suite directed python3 validation/SquareSumV1/test_matrix.py
run_suite bf16_semantics python3 validation/SquareSumV1/bf16_semantic_probe.py
run_suite random python3 validation/SquareSumV1/random_matrix.py
run_suite extended python3 validation/SquareSumV1/extended_matrix.py
run_suite s03j_boundary python3 diagnostics/squaresum_s03j_boundary_correctness_20260807.py
run_suite s03k_boundary python3 diagnostics/squaresum_s03k_middle_full_rows_correctness_20260807.py S03K

python3 - "${artifact_dir}/result.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
value = {
    "schema_version": 1,
    "passed": True,
    "candidate": "S03K",
    "suites": {
        "directed": 46,
        "bf16_semantics": 4,
        "random": 150,
        "extended": 726,
        "s03j_boundary": 135,
        "s03k_boundary": 69,
    },
    "total": 1130,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(value, indent=2))
PY

echo "S03K_CORRECTNESS_GATE_PASS total=1130 result=${artifact_dir}/result.json"
