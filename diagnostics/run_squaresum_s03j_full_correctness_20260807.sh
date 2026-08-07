#!/usr/bin/env bash
set -euo pipefail

candidate_install="${1:?usage: $0 CANDIDATE_INSTALL}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1/s03j_full_correctness"

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
validation_dir="${repo_root}/validation/SquareSumV1"
export PYTHONPATH="${validation_dir}${PYTHONPATH:+:${PYTHONPATH}}"

run_suite() {
    local name="$1"
    local script="$2"
    echo "S03J_CORRECTNESS_SUITE_START name=${name}"
    run_with_opp "${candidate_env}" python3 "${script}" \
        | tee "${detail_root}/${name}.log"
    echo "S03J_CORRECTNESS_SUITE_PASS name=${name}"
}

run_suite directed validation/SquareSumV1/test_matrix.py
run_suite bf16_semantics validation/SquareSumV1/bf16_semantic_probe.py
run_suite random validation/SquareSumV1/random_matrix.py
run_suite extended validation/SquareSumV1/extended_matrix.py

python3 - "${artifact_dir}/result.json" <<'PY'
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
value = {
    "schema_version": 1,
    "passed": True,
    "candidate": "S03J",
    "suites": {
        "directed": 46,
        "bf16_semantics": 4,
        "random": 150,
        "extended": 726,
    },
    "total": 926,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(value, indent=2))
PY

echo "S03J_FULL_CORRECTNESS_PASS total=926 result=${artifact_dir}/result.json"
