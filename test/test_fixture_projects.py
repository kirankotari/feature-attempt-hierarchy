import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import evaluate_offline
from harness import load_public_case


DATA = ROOT / "data"
NEW_CASES = (DATA / "case_old_constraint.json", DATA / "case_two_open_failures.json", DATA / "case_long_detour.json")


class FixtureProjectTests(unittest.TestCase):
    def test_new_cases_are_checkpoint_safe_and_expose_regressions(self):
        report = evaluate_offline(NEW_CASES)
        self.assertEqual(report["n_cases"], 3)
        by_case = {(row["case_id"], row["strategy"]): row for row in report["cases"]}
        for path in NEW_CASES:
            with self.subTest(path=path.name):
                case = load_public_case(path, "continuation")
                self.assertNotIn("oracle", case)
                self.assertNotIn("initial_files", case["project"])
                baseline = by_case[(case["case_id"], "chronological_suffix")]
                hierarchy = by_case[(case["case_id"], "feature_attempt_hierarchy")]
                self.assertEqual(baseline["context_gate"], path.name != "case_long_detour.json")
                self.assertTrue(hierarchy["context_gate"])
                self.assertGreater(hierarchy["discarded_history_proxy_units"], 0)
                self.assertTrue(report["feature_cost_attribution"][case["case_id"]]["matches_private_synthetic_reference"])

    def test_baselines_pass_and_checkpoint_projects_fail_for_intended_reasons(self):
        for path in NEW_CASES:
            with self.subTest(path=path.name):
                fixture = json.loads(path.read_text(encoding="utf-8"))
                for snapshot, expected_success in (("initial_files", True), ("current_files", False)):
                    with tempfile.TemporaryDirectory() as folder:
                        for name, content in fixture["project"][snapshot].items():
                            (Path(folder) / name).write_text(content, encoding="utf-8")
                        command = fixture["project"]["baseline_command" if expected_success else "continuation_command"]
                        result = subprocess.run([sys.executable, *command[1:]], cwd=folder, capture_output=True, text=True, timeout=10)
                        self.assertEqual(result.returncode == 0, expected_success, result.stderr)
                        if not expected_success:
                            self.assertIn("FAILED", result.stderr)
                            if path.name == "case_two_open_failures.json":
                                self.assertIn("KeyError", result.stderr)
                                self.assertIn("AssertionError", result.stderr)


if __name__ == "__main__":
    unittest.main()
