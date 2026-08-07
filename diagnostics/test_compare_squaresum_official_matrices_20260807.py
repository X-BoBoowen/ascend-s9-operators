import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name(
    "compare_squaresum_official_matrices_20260806.py"
)


def write_result(root, name, dtype, duration):
    destination = root / f"{dtype}_{name}" / "result.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "case": name,
                "dtype": dtype,
                "official_compatible_time": duration,
            }
        ),
        encoding="utf-8",
    )


class PointImprovementGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline_a = self.root / "baseline_a"
        self.candidate = self.root / "candidate"
        self.baseline_b = self.root / "baseline_b"
        for name in ("fast", "slow"):
            write_result(self.baseline_a, name, "fp16", 100)
            write_result(self.baseline_b, name, "fp16", 100)

    def tearDown(self):
        self.temporary.cleanup()

    def run_gate(self):
        output = self.root / "comparison.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--baseline-a",
                str(self.baseline_a),
                "--candidate",
                str(self.candidate),
                "--baseline-b",
                str(self.baseline_b),
                "--minimum-improvement-percent",
                "50",
                "--minimum-point-improvement-percent",
                "50",
                "--maximum-regression-percent",
                "3",
                "--maximum-baseline-drift-percent",
                "3",
                "--output",
                str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertTrue(
            output.is_file(),
            msg=f"comparison output missing: {completed.stderr}",
        )
        return completed, json.loads(output.read_text(encoding="utf-8"))

    def test_rejects_one_target_below_point_threshold(self):
        write_result(self.candidate, "fast", "fp16", 40)
        write_result(self.candidate, "slow", "fp16", 60)

        completed, report = self.run_gate()

        self.assertEqual(completed.returncode, 2)
        self.assertFalse(report["passed"])
        self.assertFalse(report["point_improvement_ok"])
        self.assertEqual(
            report["minimum_point_improvement_percent"], 40.0
        )

    def test_accepts_when_every_target_meets_point_threshold(self):
        write_result(self.candidate, "fast", "fp16", 40)
        write_result(self.candidate, "slow", "fp16", 40)

        completed, report = self.run_gate()

        self.assertEqual(completed.returncode, 0)
        self.assertTrue(report["passed"])
        self.assertTrue(report["point_improvement_ok"])
        self.assertEqual(
            report["minimum_point_improvement_percent"], 60.0
        )


if __name__ == "__main__":
    unittest.main()
