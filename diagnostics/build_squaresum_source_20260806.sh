#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:?usage: $0 SOURCE_DIR TEMPLATE_PROJECT OUTPUT_PROJECT}"
template_project="${2:?usage: $0 SOURCE_DIR TEMPLATE_PROJECT OUTPUT_PROJECT}"
output_project="${3:?usage: $0 SOURCE_DIR TEMPLATE_PROJECT OUTPUT_PROJECT}"

source_dir="$(cd -- "${source_dir}" && pwd -P)"
template_project="$(cd -- "${template_project}" && pwd -P)"

for required in \
    "${source_dir}/op_host/square_sum_v1.cpp" \
    "${source_dir}/op_host/square_sum_v1_tiling.h" \
    "${source_dir}/op_kernel/square_sum_v1.cpp" \
    "${template_project}/build.sh" \
    "${template_project}/CMakeLists.txt"; do
    if [[ ! -f "${required}" ]]; then
        echo "required file is missing: ${required}" >&2
        exit 2
    fi
done

if [[ -e "${output_project}" ]]; then
    echo "output project already exists: ${output_project}" >&2
    exit 2
fi

output_parent="$(cd -- "$(dirname -- "${output_project}")" && pwd -P)"
output_project="${output_parent}/$(basename -- "${output_project}")"
if [[ "${output_project}" == "/" || "${output_project}" == "${template_project}" ]]; then
    echo "refusing unsafe output project: ${output_project}" >&2
    exit 2
fi

mkdir -- "${output_project}"
(
    cd -- "${template_project}"
    tar \
        --exclude='./build' \
        --exclude='./build_out' \
        --exclude='./dist' \
        --exclude='./host_native_tiling' \
        -cf - .
) | (
    cd -- "${output_project}"
    tar -xf -
)

# The output tree is newly created, so replacing its copied source cannot touch
# the template or any user-owned build directory.
rm -rf -- "${output_project}/op_host" "${output_project}/op_kernel"
mkdir -p -- "${output_project}/op_host" "${output_project}/op_kernel"
cp -a -- "${source_dir}/op_host/." "${output_project}/op_host/"
cp -a -- "${source_dir}/op_kernel/." "${output_project}/op_kernel/"

(
    cd -- "${output_project}"
    bash ./build.sh
)

mapfile -t packages < <(
    find "${output_project}/build_out" -type f -name 'custom_opp*.run' -print
)
if (( ${#packages[@]} != 1 )); then
    echo "expected exactly one custom_opp package, found ${#packages[@]}" >&2
    printf '%s\n' "${packages[@]}" >&2
    exit 1
fi

sha256sum \
    "${source_dir}/op_host/square_sum_v1.cpp" \
    "${source_dir}/op_kernel/square_sum_v1.cpp" \
    "${packages[0]}"
printf 'SQUARESUM_BUILD_OK project=%s package=%s\n' \
    "${output_project}" "${packages[0]}"
