#!/usr/bin/env bash
set -euo pipefail

dtype_name="${1:?usage: $0 DTYPE LABEL OUTPUT_NAME [TIER]}"
label="${2:?usage: $0 DTYPE LABEL OUTPUT_NAME [TIER]}"
output_name="${3:?usage: $0 DTYPE LABEL OUTPUT_NAME [TIER]}"
tier="${4:-core}"

if [[ ! "${dtype_name}" =~ ^(fp16|bf16|fp32)$ ]] ||
   [[ ! "${label}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
   [[ ! "${output_name}" =~ ^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$ ]] ||
   [[ ! "${tier}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "dtype, label, output name, or tier contains unsupported characters"
    exit 2
fi

IFS='/' read -r -a output_segments <<< "${output_name}"
for segment in "${output_segments[@]}"; do
    if [[ "${segment}" == "." || "${segment}" == ".." ]]; then
        echo "output name contains an unsafe path segment"
        exit 2
    fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
matrix="${script_dir}/squaresum_official_profile_matrix_20260806.json"
output_root="${repo_root}/artifact/${output_name}"

rm -rf -- "${output_root}"
mkdir -p -- "${output_root}"

mapfile -t cases < <(
    python3 - "${matrix}" "${tier}" <<'PY'
import json
import sys

path, tier = sys.argv[1:]
document = json.loads(open(path, encoding="utf-8").read())
for case in document["cases"]:
    if case.get("tier") == tier:
        print(case["name"])
PY
)

if (( ${#cases[@]} == 0 )); then
    echo "matrix tier ${tier} contains no cases"
    exit 2
fi

for case_name in "${cases[@]}"; do
    bash "${script_dir}/run_squaresum_official_profile_20260806.sh" \
        "${case_name}" \
        "${dtype_name}" \
        "${label}" \
        "${output_root}/${case_name}"
done

echo "OFFICIAL_MATRIX_SUMMARY label=${label} dtype=${dtype_name} tier=${tier} passed=${#cases[@]}/${#cases[@]} output=${output_root}"
