"""Collect one operator run into artifact/ for off-machine analysis.

Overwrite-style: each run replaces the previous contents of the operator's
artifact subdirectory, so only the latest run is ever kept.

Reads the profiler output that run.sh already produces (PROF*/**/op_summary*.csv)
plus the test log, and emits:

  artifact/<Op>/summary.json   parsed timings, bytes moved, effective bandwidth
  artifact/<Op>/op_summary.csv the raw profiler rows, verbatim
  artifact/<Op>/run.log        the test/profile log

The bandwidth figure is the point of this script. A duration alone cannot say
whether a case is bandwidth-bound or launch-bound, and that distinction decides
where optimisation effort belongs.

Usage: python3 collect_artifact.py <Op> <case_id> [--bytes N] [--log FILE]
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

# 910B HBM class figure, used only to express a run as a fraction of peak.
HBM_PEAK_GBPS = 1600.0

SKIP_OPS = ("aclnnMul",)

# get_time.py takes the median of calls 10..30; mirror it so the artifact
# number and the scored number cannot drift apart.
WARMUP_SKIP = 10
WINDOW_END = 30


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_case_dir(op: str) -> Path:
    path = repo_root() / "case_910b" / op
    if not path.is_dir():
        raise SystemExit(f"[ERROR] no such operator directory: {path}")
    return path


def collect_rows(case_dir: Path) -> tuple[list[dict], Path | None]:
    """Return profiler rows for the operator under test, newest run only."""
    candidates = sorted(
        case_dir.rglob("op_summary*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return [], None
    newest = candidates[0]
    rows = []
    with open(newest, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Op Name", "")
            if any(skip in name for skip in SKIP_OPS):
                continue
            rows.append(row)
    return rows, newest


def durations_us(rows: list[dict]) -> list[float]:
    values = []
    for row in rows:
        raw = row.get("Task Duration(us)")
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round(fraction * (len(ordered) - 1)))
    return ordered[index]


def summarise(values: list[float]) -> dict:
    window = values[WARMUP_SKIP:WINDOW_END]
    scored = window if window else values
    return {
        "call_count": len(values),
        "scored_window": [WARMUP_SKIP, WINDOW_END],
        "scored_call_count": len(scored),
        "p50_us": percentile(scored, 0.50),
        "min_us": min(scored) if scored else None,
        "max_us": max(scored) if scored else None,
        "p90_us": percentile(scored, 0.90),
        "all_calls_us": values,
    }


def bandwidth(total_bytes: int | None, duration_us: float | None) -> dict:
    if not total_bytes or not duration_us:
        return {
            "total_bytes": total_bytes,
            "note": "pass --bytes to get effective bandwidth",
        }
    gbps = total_bytes / (duration_us * 1e-6) / 1e9
    return {
        "total_bytes": total_bytes,
        "effective_gbps": round(gbps, 2),
        "hbm_peak_gbps": HBM_PEAK_GBPS,
        "peak_fraction": round(gbps / HBM_PEAK_GBPS, 4),
        "bound": "bandwidth" if gbps > 0.5 * HBM_PEAK_GBPS else "launch/latency",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("op")
    parser.add_argument("case_id")
    parser.add_argument(
        "--bytes",
        type=int,
        default=None,
        help="bytes moved (read + write) for this case",
    )
    parser.add_argument("--log", default=".last_test.log")
    args = parser.parse_args()

    case_dir = find_case_dir(args.op)
    out_dir = repo_root() / "artifact" / args.op

    # Overwrite semantics: drop the whole previous directory first.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, source_csv = collect_rows(case_dir)
    values = durations_us(rows)

    summary = {
        "operator": args.op,
        "case_id": args.case_id,
        "profiler_csv": str(source_csv.relative_to(repo_root()))
        if source_csv
        else None,
        "timing": summarise(values),
        "throughput": bandwidth(args.bytes, summarise(values)["p50_us"]),
    }

    log_path = case_dir / args.log
    if log_path.is_file():
        shutil.copyfile(log_path, out_dir / "run.log")
        text = log_path.read_text(encoding="utf-8", errors="replace")
        summary["verify_passed"] = (
            f"case{args.case_id} verify result pass!" in text
        )
    else:
        summary["verify_passed"] = None

    if source_csv:
        shutil.copyfile(source_csv, out_dir / "op_summary.csv")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["timing"], indent=2))
    print(json.dumps(summary["throughput"], indent=2))
    print(f"[artifact] wrote {out_dir.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
