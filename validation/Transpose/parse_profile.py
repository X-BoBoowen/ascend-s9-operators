import csv
import glob
import statistics
import sys


paths = sorted(glob.glob(sys.argv[1], recursive=True))
if not paths:
    raise SystemExit(f"no CSV matched: {sys.argv[1]}")

for path in paths:
    durations = []
    with open(path, newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if "transpose" not in row.get("OP Type", "").lower():
                continue
            value = row.get("Task Duration(us)", "").strip()
            if value:
                durations.append(float(value))
    if durations:
        stable = durations[-20:] if len(durations) >= 20 else durations
        print(
            f"{path}: count={len(durations)} "
            f"median_last={statistics.median(stable):.6f}us "
            f"min_last={min(stable):.6f}us "
            f"max_last={max(stable):.6f}us"
        )
