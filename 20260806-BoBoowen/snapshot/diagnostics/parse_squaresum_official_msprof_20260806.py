import argparse
import csv
import json
import os
import statistics
from collections import Counter
from pathlib import Path


def read_rows(profile_root):
    files = sorted(profile_root.rglob("op_summary*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no op_summary*.csv files found below {profile_root}"
        )

    kept = []
    skipped_mul = 0
    names = Counter()
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                name = row["Op Name"]
                if "aclnnMul" in name:
                    skipped_mul += 1
                    continue
                scaled_duration = int(float(row["Task Duration(us)"]) * 1_000_000)
                kept.append(scaled_duration)
                names[name] += 1
    return files, kept, skipped_mul, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, default=Path("."))
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--case", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--require-mul-count", type=int, default=None)
    parser.add_argument("--result-json", type=Path, default=None)
    args = parser.parse_args()

    files, durations, skipped_mul, names = read_rows(args.profile_root.resolve())
    if (
        args.require_mul_count is not None
        and skipped_mul != args.require_mul_count
    ):
        raise RuntimeError(
            "profile task stream does not match the competition wrapper: "
            f"expected {args.require_mul_count} aclnnMul rows, found {skipped_mul}"
        )
    selected = durations[10:30]
    if len(selected) != 20:
        raise RuntimeError(
            "official slice requires at least 30 non-Mul task durations; "
            f"found {len(durations)}"
        )

    result = {
        "schema_version": 1,
        "label": args.label,
        "case": args.case,
        "dtype": args.dtype,
        "official_compatible_time": int(statistics.median(selected)),
        "total_non_mul_tasks": len(durations),
        "skipped_mul_tasks": skipped_mul,
        "selected_indices": [10, 30],
        "selected_durations": selected,
        "non_mul_operator_counts": dict(sorted(names.items())),
        "source_files": [str(path) for path in files],
    }
    if args.result_json is not None:
        destination = args.result_json.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    print("OFFICIAL_PROFILE_RESULT " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
