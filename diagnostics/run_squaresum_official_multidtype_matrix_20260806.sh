#!/usr/bin/env bash
set -euo pipefail

label="${1:?usage: $0 LABEL OUTPUT_NAME TIER [DTYPE ...]}"
output_name="${2:?usage: $0 LABEL OUTPUT_NAME TIER [DTYPE ...]}"
tier="${3:?usage: $0 LABEL OUTPUT_NAME TIER [DTYPE ...]}"
shift 3
dtypes=("$@")
if (( ${#dtypes[@]} == 0 )); then
    dtypes=(fp16 bf16 fp32)
fi

if [[ ! "${label}" =~ ^[A-Za-z0-9_.-]+$ ]] ||
   [[ ! "${output_name}" =~ ^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$ ]] ||
   [[ ! "${tier}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "label, output name, or tier contains unsupported characters" >&2
    exit 2
fi
IFS='/' read -r -a output_segments <<< "${output_name}"
for segment in "${output_segments[@]}"; do
    if [[ "${segment}" == "." || "${segment}" == ".." ]]; then
        echo "output name contains an unsafe path segment" >&2
        exit 2
    fi
done
for dtype_name in "${dtypes[@]}"; do
    if [[ ! "${dtype_name}" =~ ^(fp16|bf16|fp32)$ ]]; then
        echo "unsupported dtype: ${dtype_name}" >&2
        exit 2
    fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
artifact_root="${repo_root}/artifact"
matrix="${script_dir}/squaresum_official_profile_matrix_20260806.json"
output_root="${artifact_root}/${output_name}"

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
    echo "matrix tier ${tier} contains no cases" >&2
    exit 2
fi

passed=0
for dtype_name in "${dtypes[@]}"; do
    for case_name in "${cases[@]}"; do
        bash "${script_dir}/run_squaresum_official_profile_20260806.sh" \
            "${case_name}" \
            "${dtype_name}" \
            "${label}" \
            "${output_root}/${dtype_name}_${case_name}"
        passed=$((passed + 1))
    done
done

expected=$(( ${#dtypes[@]} * ${#cases[@]} ))
echo "OFFICIAL_MULTIDTYPE_MATRIX_SUMMARY label=${label} tier=${tier} passed=${passed}/${expected} output=${output_root}"
