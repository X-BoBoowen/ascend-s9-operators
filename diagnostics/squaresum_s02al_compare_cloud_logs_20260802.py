import argparse
import json
from pathlib import Path


def load_performance(path):
    records = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "performance":
            continue
        key = (
            str(record["path"]),
            tuple(record["shape"]),
            tuple(record["axes"]),
            bool(record["keep_dims"]),
            str(record["dtype"]),
        )
        if key in records:
            raise ValueError(f"duplicate record {key} in {path}")
        records[key] = float(record["median_us"])
    if not records:
        raise ValueError(f"no performance records found in {path}")
    return records


def improvement(reference, candidate):
    return (reference - candidate) / reference * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--candidate-label", default="candidate")
    args = parser.parse_args()

    reference = load_performance(args.reference)
    candidate = load_performance(args.candidate)
    if set(reference) != set(candidate):
        raise ValueError(
            "record mismatch: "
            f"missing={sorted(set(reference) - set(candidate))}, "
            f"extra={sorted(set(candidate) - set(reference))}"
        )

    rows = []
    by_path = {}
    for key in sorted(reference):
        reference_us = reference[key]
        candidate_us = candidate[key]
        path = key[0]
        bucket = by_path.setdefault(
            path, {"reference_us": 0.0, "candidate_us": 0.0}
        )
        bucket["reference_us"] += reference_us
        bucket["candidate_us"] += candidate_us
        rows.append(
            {
                "path": path,
                "shape": key[1],
                "axes": key[2],
                "keep_dims": key[3],
                "dtype": key[4],
                "reference_us": reference_us,
                "candidate_us": candidate_us,
                "improvement_pct": improvement(
                    reference_us, candidate_us
                ),
            }
        )

    for values in by_path.values():
        values["improvement_pct"] = improvement(
            values["reference_us"], values["candidate_us"]
        )
    reference_sum = sum(reference.values())
    candidate_sum = sum(candidate.values())
    changes = [row["improvement_pct"] for row in rows]
    result = {
        "reference_label": args.reference_label,
        "candidate_label": args.candidate_label,
        "records": len(rows),
        "reference_sum_us": reference_sum,
        "candidate_sum_us": candidate_sum,
        "sum_improvement_pct": improvement(
            reference_sum, candidate_sum
        ),
        "best_improvement_pct": max(changes),
        "worst_improvement_pct": min(changes),
        "by_path": by_path,
        "rows": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
