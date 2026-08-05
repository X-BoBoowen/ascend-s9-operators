import argparse
import json
from pathlib import Path


def load_results(path):
    records = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("RESULT "):
            continue
        fields = {}
        for token in line.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        identity = (fields["case"], fields["dtype"])
        if identity in records:
            raise ValueError(f"duplicate record {identity} in {path}")
        records[identity] = float(fields["median_us"])
    if not records:
        raise ValueError(f"no RESULT records found in {path}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    candidate = load_results(args.candidate)
    if set(baseline) != set(candidate):
        raise ValueError("baseline/candidate record sets differ")

    rows = []
    for identity in sorted(baseline):
        before = baseline[identity]
        after = candidate[identity]
        rows.append(
            {
                "case": identity[0],
                "dtype": identity[1],
                "baseline_us": before,
                "candidate_us": after,
                "speedup": before / after,
                "change_pct": (after - before) / before * 100.0,
            }
        )
    result = {
        "records": len(rows),
        "baseline_sum_us": sum(row["baseline_us"] for row in rows),
        "candidate_sum_us": sum(row["candidate_us"] for row in rows),
        "best_speedup": max(row["speedup"] for row in rows),
        "worst_speedup": min(row["speedup"] for row in rows),
        "rows": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
