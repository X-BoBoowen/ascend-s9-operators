import argparse
import json
import os
import statistics
from pathlib import Path


def load_matrix(path):
    results = {}
    for result_path in sorted(path.glob("*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        key = (result["case"], result["dtype"])
        if key in results:
            raise ValueError(f"duplicate result for {key} below {path}")
        results[key] = result
    if not results:
        raise ValueError(f"no */result.json files found below {path}")
    return results


def atomic_write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-a", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-b", type=Path, required=True)
    parser.add_argument("--minimum-improvement-percent", type=float, default=5.0)
    parser.add_argument("--maximum-regression-percent", type=float, default=3.0)
    parser.add_argument("--maximum-baseline-drift-percent", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline_a = load_matrix(args.baseline_a.resolve())
    candidate = load_matrix(args.candidate.resolve())
    baseline_b = load_matrix(args.baseline_b.resolve())
    if not (baseline_a.keys() == candidate.keys() == baseline_b.keys()):
        raise ValueError("A/B/A matrices do not contain identical case/dtype keys")

    rows = []
    baseline_sum = 0.0
    candidate_sum = 0.0
    for key in sorted(baseline_a):
        a = float(baseline_a[key]["official_compatible_time"])
        c = float(candidate[key]["official_compatible_time"])
        b = float(baseline_b[key]["official_compatible_time"])
        baseline = statistics.median((a, b))
        drift = abs(a - b) / baseline * 100.0 if baseline else 0.0
        improvement = (baseline - c) / baseline * 100.0 if baseline else 0.0
        rows.append(
            {
                "case": key[0],
                "dtype": key[1],
                "baseline_a": a,
                "candidate": c,
                "baseline_b": b,
                "baseline_median": baseline,
                "baseline_drift_percent": drift,
                "improvement_percent": improvement,
            }
        )
        baseline_sum += baseline
        candidate_sum += c

    aggregate_improvement = (
        (baseline_sum - candidate_sum) / baseline_sum * 100.0
        if baseline_sum
        else 0.0
    )
    drift_ok = all(
        row["baseline_drift_percent"]
        <= args.maximum_baseline_drift_percent
        for row in rows
    )
    regression_ok = all(
        row["improvement_percent"] >= -args.maximum_regression_percent
        for row in rows
    )
    improvement_ok = aggregate_improvement >= args.minimum_improvement_percent
    passed = drift_ok and regression_ok and improvement_ok

    report = {
        "schema_version": 1,
        "passed": passed,
        "thresholds": {
            "minimum_improvement_percent": args.minimum_improvement_percent,
            "maximum_regression_percent": args.maximum_regression_percent,
            "maximum_baseline_drift_percent": args.maximum_baseline_drift_percent,
        },
        "baseline_sum": baseline_sum,
        "candidate_sum": candidate_sum,
        "aggregate_improvement_percent": aggregate_improvement,
        "drift_ok": drift_ok,
        "regression_ok": regression_ok,
        "improvement_ok": improvement_ok,
        "cases": rows,
    }
    if args.output is not None:
        atomic_write_json(args.output.resolve(), report)
    print("OFFICIAL_ABA_RESULT " + json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
