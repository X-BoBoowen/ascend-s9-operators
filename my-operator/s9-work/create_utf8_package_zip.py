from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


parent = Path(r"D:\29722\Desktop\GCC\提交相关材料")
source = parent / "S9全量资料审计与网页仓库归档_20260725"
target = parent / "S9全量资料审计与网页仓库归档_20260725.zip"
temporary = parent / "S9全量资料审计与网页仓库归档_20260725.zip.tmp"

if temporary.exists():
    temporary.unlink()

with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
    for path in sorted(source.rglob("*"), key=lambda item: str(item).lower()):
        relative = Path(source.name) / path.relative_to(source)
        if path.is_dir():
            archive.writestr(relative.as_posix().rstrip("/") + "/", b"")
        else:
            long_path = "\\\\?\\" + str(path.resolve())
            archive.write(long_path, relative.as_posix())

temporary.replace(target)
print(target)
print(target.stat().st_size)
