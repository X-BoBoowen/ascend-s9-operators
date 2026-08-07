import argparse
import os
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


SOURCE_FILES = (
    "op_host/CMakeLists.txt",
    "op_host/square_sum_v1_tiling.h",
    "op_host/square_sum_v1.cpp",
    "op_kernel/CMakeLists.txt",
    "op_kernel/square_sum_v1.cpp",
)


def zip_info(name: str, source: Path | None, mode: int) -> ZipInfo:
    if source is None:
        timestamp = datetime.now()
    else:
        timestamp = datetime.fromtimestamp(source.stat().st_mtime)
    timestamp = timestamp.replace(microsecond=0)
    if timestamp.year < 1980:
        timestamp = timestamp.replace(year=1980)
    info = ZipInfo(name, timestamp.timetuple()[:6])
    info.create_system = 3
    info.external_attr = (mode & 0xFFFF) << 16
    info.compress_type = ZIP_DEFLATED
    info.flag_bits |= 0x800
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", default="SquareSumV1_zip")
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    run_file = args.run.resolve(strict=True)
    output = args.output.resolve()
    if not args.top or "/" in args.top or "\\" in args.top:
        raise ValueError("top directory must be one safe path component")
    for relative in SOURCE_FILES:
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(path)
    if not run_file.is_file() or run_file.stat().st_size == 0:
        raise ValueError("run package must be a non-empty file")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with ZipFile(
            temporary,
            "w",
            compression=ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for directory in (args.top, f"{args.top}/op_host", f"{args.top}/op_kernel"):
                archive.writestr(zip_info(directory + "/", None, 0o40755), b"")
            for relative in SOURCE_FILES:
                path = source / relative
                archive.writestr(
                    zip_info(f"{args.top}/{relative}", path, 0o100644),
                    path.read_bytes(),
                )
            archive.writestr(
                zip_info(
                    f"{args.top}/custom_opp_euleros_aarch64.run",
                    run_file,
                    0o100755,
                ),
                run_file.read_bytes(),
            )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(output)
    print(output.stat().st_size)


if __name__ == "__main__":
    main()
