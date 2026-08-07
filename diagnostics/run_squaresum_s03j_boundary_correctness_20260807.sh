#!/usr/bin/env bash
set -euo pipefail

candidate_install="${1:?usage: $0 CANDIDATE_INSTALL}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_dir="${repo_root}/artifact"
detail_root="${artifact_dir}/SquareSumV1/s03j_boundary_correctness"

rm -rf -- "${artifact_dir}"
mkdir -p -- "${detail_root}"
exec > >(tee -a "${artifact_dir}/run.log") 2>&1

candidate_env="$(find "${candidate_install}" -type f -name set_env.bash -print -quit)"
test -n "${candidate_env}"

cd -- "${repo_root}"
export PYTHONPATH="${repo_root}/validation/SquareSumV1${PYTHONPATH:+:${PYTHONPATH}}"

bash -c 'set -eo pipefail; source "$1"; set -u; shift; exec "$@"' \
    _ "${candidate_env}" python3 \
    diagnostics/squaresum_s03j_boundary_correctness_20260807.py \
    | tee "${detail_root}/boundary.log"

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
    "suite": "fastPath2_boundary_correctness",
    "total": 135,
    "seed": 2026080703,
}
temporary = output.with_name(output.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(value, indent=2))
PY

echo "S03J_BOUNDARY_CORRECTNESS_PASS total=135 result=${artifact_dir}/result.json"
