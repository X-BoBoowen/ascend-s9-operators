#!/usr/bin/env bash
set -euo pipefail

repo="${1:-/home/ma-user/work/s9/repository/ascend-s9-operators}"
work="${2:-/home/ma-user/work/s9/runs/s02f_s02ay_atlas_20260807_1504}"
s02f_install="${3:-/home/ma-user/work/s9/runs/s03_gate_20260807_1309/install/s02f}"
s02ay_package="${4:-/home/ma-user/work/s9/experiments/squaresum_s02ay_cloud_20260805_1603/build_out/custom_opp_euleros_aarch64.run}"

mkdir -p -- "${work}/install/s02ay" "${work}/logs"
chmod u+x -- "${s02ay_package}"
"${s02ay_package}" --install-path="${work}/install/s02ay" >/dev/null

find_env() {
    local root="$1"
    find "${root}" -type f -name set_env.bash -print -quit
}

run_with_opp() {
    local env_file="$1"
    shift
    bash -c 'set -eo pipefail; source "$1"; set -u; shift; exec "$@"' \
        _ "${env_file}" "$@"
}

s02f_env="$(find_env "${s02f_install}")"
s02ay_env="$(find_env "${work}/install/s02ay")"
test -n "${s02f_env}"
test -n "${s02ay_env}"

cd -- "${repo}"
export PYTHONPATH="${repo}/validation/SquareSumV1${PYTHONPATH:+:${PYTHONPATH}}"

for tier in core atlas; do
    run_with_opp "${s02f_env}" python3 \
        diagnostics/squaresum_domain_event_atlas_20260806.py \
        --label "S02F_${tier}" --tier "${tier}" \
        --dtypes fp16 bf16 fp32 \
        | tee "${work}/logs/s02f_${tier}.log"
    run_with_opp "${s02ay_env}" python3 \
        diagnostics/squaresum_domain_event_atlas_20260806.py \
        --label "S02AY_${tier}" --tier "${tier}" \
        --dtypes fp16 bf16 fp32 \
        | tee "${work}/logs/s02ay_${tier}.log"
    set +e
    python3 diagnostics/compare_squaresum_event_atlas_20260806.py \
        --baseline "${work}/logs/s02f_${tier}.log" \
        --candidate "${work}/logs/s02ay_${tier}.log" \
        --minimum-improvement-percent 0 \
        --maximum-regression-percent 1000 \
        --output "${work}/logs/comparison_${tier}.json"
    set -e
    cat -- "${work}/logs/comparison_${tier}.json"
done

echo "S02F_S02AY_ATLAS_DONE work=${work}"
