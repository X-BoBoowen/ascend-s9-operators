import argparse
import json
import os
import re
from pathlib import Path


LINE = re.compile(
    r"^DOMAIN_EVENT_ATLAS\s+.*?case=(?P<case>\S+)\s+.*?"
    r"dtype=(?P<dtype>\S+)\s+.*?median_us=(?P<median>[0-9.]+)\s+"
)


def parse(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LINE.search(line)
        if not match:
            continue
        key = (match.group("case"), match.group("dtype"))
        if key in values:
            raise ValueError(f"duplicate event point {key} in {path}")
        values[key] = float(match.group("median"))
    if not values:
        raise ValueError(f"no DOMAIN_EVENT_ATLAS rows in {path}")
    return values


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-improvement-percent", type=float, default=10)
    parser.add_argument("--maximum-regression-percent", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = parse(args.baseline)
    candidate = parse(args.candidate)
    if baseline.keys() != candidate.keys():
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"event key mismatch missing={missing} extra={extra}")

    points = []
    baseline_total = 0.0
    candidate_total = 0.0
    worst_regression = float("-inf")
    for key in sorted(baseline):
        before = baseline[key]
        after = candidate[key]
        if before <= 0.0 or after <= 0.0:
            raise ValueError(f"non-positive event median for {key}")
        improvement = (before - after) / before * 100.0
        regression = -improvement
        baseline_total += before
        candidate_total += after
        worst_regression = max(worst_regression, regression)
        points.append(
            {
                "case": key[0],
                "dtype": key[1],
                "baseline_us": before,
                "candidate_us": after,
                "improvement_percent": improvement,
            }
        )

    aggregate_improvement = (
        (baseline_total - candidate_total) / baseline_total * 100.0
    )
    passed = (
        aggregate_improvement >= args.minimum_improvement_percent
        and worst_regression <= args.maximum_regression_percent
    )
    result = {
        "schema_version": 1,
        "passed": passed,
        "baseline_total_us": baseline_total,
        "candidate_total_us": candidate_total,
        "aggregate_improvement_percent": aggregate_improvement,
        "worst_regression_percent": worst_regression,
        "minimum_improvement_percent": args.minimum_improvement_percent,
        "maximum_regression_percent": args.maximum_regression_percent,
        "points": points,
    }
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
