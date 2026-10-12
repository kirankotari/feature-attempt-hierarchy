import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import load_public_case, main
from hierarchy import build_hierarchy, feature_cost, render_hierarchy, select_feature_context


DATA = ROOT / "data"


class HierarchyTests(unittest.TestCase):
    def test_parent_child_structure_and_repair_status(self):
        case = load_public_case(DATA / "context-repair-benchmark.json", "continuation")
        graph = build_hierarchy(case)
        self.assertEqual(graph["parents"]["e04"], "attempt:e03")
        self.assertEqual(graph["parents"]["e09"], "attempt:e07")
        self.assertEqual(graph["parents"]["e10"], "thread:side")
        self.assertEqual(graph["parents"]["e12"], graph["root"])
        issues = {issue["event_id"]: issue for issue in graph["issues"]}
        self.assertEqual(issues["e04"]["status"], "resolved")
        self.assertEqual(issues["e04"]["resolved_by"], "e08")
        self.assertEqual(issues["e09"]["status"], "open")
        self.assertEqual(issues["e09"]["attempt_id"], "attempt:e07")
        self.assertIn("inferred", issues["e09"]["link_basis"])
        self.assertNotIn("oracle", json.dumps(graph))

    def test_selection_retains_open_error_without_old_error_or_side_turn(self):
        case = load_public_case(DATA / "context-repair-benchmark.json", "continuation")
        graph = build_hierarchy(case)
        selected = select_feature_context(case, graph)
        self.assertIn("e09", selected["event_ids"])
        self.assertIn("e12", selected["event_ids"])
        self.assertNotIn("e04", selected["event_ids"])
        self.assertNotIn("e10", selected["event_ids"])
        self.assertNotIn("e11", selected["event_ids"])
        self.assertLessEqual(selected["proxy_tokens"], case["history_budget_tokens"])
        outline = render_hierarchy(case, graph)
        self.assertIn("attempt:e07", outline)
        self.assertIn("e09 [open]", outline)
        self.assertIn("e04 [resolved]", outline)

    def test_expected_observed_mismatch_is_open_even_without_error_keyword(self):
        case = load_public_case(DATA / "case_refund_format.json", "continuation")
        graph = build_hierarchy(case)
        issues = {issue["event_id"]: issue for issue in graph["issues"]}
        self.assertEqual(issues["e04"]["status"], "resolved")
        self.assertEqual(issues["e07"]["status"], "open")
        selected = select_feature_context(case, graph)
        self.assertIn("e07", selected["event_ids"])
        self.assertIn("e10", selected["event_ids"])
        self.assertNotIn("e08", selected["event_ids"])
        self.assertNotIn("e09", selected["event_ids"])

    def test_cli_displays_hierarchy_without_private_oracle(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--strategy", "feature_attempt_hierarchy", "--view", "hierarchy"])
        text = output.getvalue()
        self.assertIn("feature:catalog-include-archived-v1", text)
        self.assertIn("e04 [resolved]", text)
        self.assertIn("e09 [open]", text)
        self.assertNotIn("private_cases", text)

    def test_open_failures_and_earlier_feature_instructions_stay_in_budget(self):
        shipping = load_public_case(DATA / "case_two_open_failures.json", "continuation")
        selected = select_feature_context(shipping)
        self.assertIn("e04", selected["event_ids"])
        self.assertIn("e07", selected["event_ids"])
        self.assertIn("e08", selected["event_ids"])
        compact = load_public_case(DATA / "case_old_constraint.json", "continuation")
        constrained = select_feature_context(compact)
        self.assertIn("e02", constrained["event_ids"])
        self.assertIn("e05", constrained["event_ids"])
        self.assertIn("e07", constrained["event_ids"])
        self.assertLessEqual(constrained["proxy_tokens"], compact["history_budget_tokens"])

    def test_feature_cost_is_not_session_cost_and_unknown_is_not_zero(self):
        case = load_public_case(DATA / "context-repair-benchmark.json", "continuation")
        cost = feature_cost(case, build_hierarchy(case))
        self.assertEqual(cost["feature_recorded_synthetic_usd"], 0.0062)
        self.assertEqual(cost["side_recorded_synthetic_usd"], 0.00026)
        self.assertEqual(cost["feature_rework_recorded_synthetic_usd"], 0.0046)
        second = load_public_case(DATA / "case_refund_format.json", "continuation")
        other_cost = feature_cost(second, build_hierarchy(second))
        self.assertEqual(other_cost["feature_recorded_synthetic_usd"], 0.00396)
        self.assertEqual(other_cost["side_recorded_synthetic_usd"], 0.00026)
        case["events"][2].pop("usage")
        self.assertIsNone(feature_cost(case, build_hierarchy(case))["feature_recorded_synthetic_usd"])


if __name__ == "__main__":
    unittest.main()
