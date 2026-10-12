import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import select_dependency_lexical_proxy, select_thread_path_proxy
from evaluation import evaluate_offline, load_submission, main, render_comparison
from harness import load_public_case, select_event_ids, select_last_error_latest_user


FIXTURE = ROOT / "data" / "context-repair-benchmark.json"
SECOND_FIXTURE = ROOT / "data" / "case_refund_format.json"


class EvaluationTests(unittest.TestCase):
    def test_both_builtin_methods_see_the_same_case_and_budget(self):
        report = evaluate_offline([FIXTURE])
        rows = {row["strategy"]: row for row in report["cases"]}
        self.assertEqual(set(rows), {"chronological_suffix", "last_error_latest_user", "feature_attempt_hierarchy",
                                     "thread_path_suffix_proxy", "dependency_lexical_proxy",
                                     "contextweaver_validation_proxy", "combined_feature_proxy"})
        self.assertEqual(rows["thread_path_suffix_proxy"]["implementation_status"], "memotree_exploratory_proxy")
        self.assertEqual(rows["contextweaver_validation_proxy"]["implementation_status"], "contextweaver_exploratory_proxy")
        self.assertEqual(rows["combined_feature_proxy"]["implementation_status"], "local_combination_proxy")
        self.assertEqual(report["n_cases"], 1)
        self.assertEqual(report["mode"], "offline_selection_only")
        self.assertFalse(report["research_claim_supported"])
        self.assertEqual(rows["chronological_suffix"]["selected_event_ids"], ["e09", "e10", "e11", "e12"])
        self.assertEqual(rows["last_error_latest_user"]["selected_event_ids"], ["e09", "e12"])
        self.assertEqual({row["history_budget_proxy_units"] for row in rows.values()}, {320})
        self.assertEqual({row["workspace_sha256"] for row in rows.values()}, {rows["chronological_suffix"]["workspace_sha256"]})
        self.assertEqual(rows["chronological_suffix"]["required_recall"], 1.0)
        self.assertEqual(rows["chronological_suffix"]["known_irrelevant_included"], 2)
        self.assertFalse(rows["chronological_suffix"]["context_gate"])
        self.assertTrue(rows["last_error_latest_user"]["context_gate"])
        self.assertLess(rows["last_error_latest_user"]["history_proxy_units"], rows["chronological_suffix"]["history_proxy_units"])
        self.assertEqual(rows["feature_attempt_hierarchy"]["implementation_status"], "project_prototype")
        self.assertEqual(report["feature_cost_attribution"]["catalog-include-archived-v1"]["feature_recorded_synthetic_usd"], 0.0062)

    def test_paper_inspired_proxies_are_budgeted_and_not_oracle_fed(self):
        for fixture in (FIXTURE, SECOND_FIXTURE):
            case = load_public_case(fixture, "continuation")
            threaded = select_thread_path_proxy(case)
            dependency = select_dependency_lexical_proxy(case)
            self.assertTrue(threaded["event_ids"])
            self.assertTrue(dependency["event_ids"])
            self.assertTrue(set(threaded["event_ids"]) <= {event["id"] for event in case["events"] if event.get("thread_id") == "feature"})
            self.assertLessEqual(threaded["proxy_tokens"], case["history_budget_tokens"])
            self.assertLessEqual(dependency["proxy_tokens"], case["history_budget_tokens"])
            self.assertNotIn("oracle", json.dumps(case))
        refund = load_public_case(SECOND_FIXTURE, "continuation")
        self.assertIn("e07", select_dependency_lexical_proxy(refund)["event_ids"])
        self.assertNotIn("e07", select_dependency_lexical_proxy(refund, skip_failed=True)["event_ids"])
        self.assertNotIn("e04", select_thread_path_proxy(refund)["event_ids"])

    def test_cost_is_labeled_proxy_and_model_outcome_is_unknown(self):
        report = evaluate_offline([FIXTURE])
        row = report["cases"][0]
        expected = row["history_proxy_units"] * 2 / 1_000_000 * 1000
        self.assertAlmostEqual(row["estimated_history_input_usd_per_1k"], expected)
        self.assertEqual(report["recorded_synthetic_model_cost_usd"], 0.00646)
        for row in report["cases"]:
            self.assertIsNone(row["actual_model_cost_usd"])
            self.assertIsNone(row["repair_pass"])
            self.assertIsNone(row["latency_s"])
        self.assertIn("proxy", report["cost_basis"])

    def test_candidate_submission_is_scored_without_oracle_access(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selection.json"
            path.write_text(json.dumps({"strategy": "proposed", "cases": {"catalog-include-archived-v1": {"event_ids": ["e09", "e12"]}}}), encoding="utf-8")
            name, selections = load_submission(path)
            report = evaluate_offline([FIXTURE], submissions={name: selections})
        row = next(item for item in report["cases"] if item["strategy"] == "proposed")
        self.assertTrue(row["context_gate"])
        self.assertEqual(row["implementation_status"], "external_selection_unverified")
        self.assertNotIn("private_cases", json.dumps(report))
        self.assertNotIn("reference_links", json.dumps(report))

    def test_future_or_duplicate_event_is_rejected_for_candidates(self):
        case = load_public_case(FIXTURE, "continuation")
        for ids in (["e13"], ["e09", "e09"], []):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_event_ids(case, ids)

    def test_last_error_strategy_does_not_require_oracle(self):
        case = load_public_case(FIXTURE, "continuation")
        selected = select_last_error_latest_user(case)
        self.assertEqual(selected["event_ids"], ["e09", "e12"])
        self.assertNotIn("oracle", json.dumps(case))

    def test_report_is_explicitly_not_an_agent_success_result(self):
        text = render_comparison(evaluate_offline([FIXTURE]))
        self.assertIn("OFFLINE SELECTION ONLY", text)
        self.assertIn("NOT a repair success or billed cost result", text)
        self.assertIn("chronological_suffix", text)
        self.assertIn("last_error_latest_user", text)
        self.assertIn("discarded", text)
        self.assertIn("n/a", text)
        self.assertNotIn("private_cases", text)

    def test_json_cli_report_can_be_replayed_without_live_calls(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--fixture", str(FIXTURE), "--json"])
        report = json.loads(output.getvalue())
        self.assertEqual(report["n_cases"], 1)
        self.assertFalse(report["research_claim_supported"])
        self.assertEqual(report["mode"], "offline_selection_only")

    def test_default_cli_includes_all_five_distinct_cases(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--json"])
        report = json.loads(output.getvalue())
        self.assertEqual(report["n_cases"], 5)
        self.assertEqual(len(report["cases"]), 35)
        self.assertFalse(report["research_claim_supported"])
        self.assertGreaterEqual(report["summary"]["feature_attempt_hierarchy"]["context_gate_count"],
                                report["summary"]["chronological_suffix"]["context_gate_count"])
        self.assertIn("local proxies", " ".join(report["limitations"]).lower() + render_comparison(report).lower())

    def test_cli_marks_external_candidate_as_selection_only(self):
        with tempfile.TemporaryDirectory() as folder:
            submission = Path(folder) / "candidate.json"
            submission.write_text(json.dumps({"strategy": "graph_trial", "cases": {"catalog-include-archived-v1": {"event_ids": ["e09", "e12"]}}}), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                main(["--fixture", str(FIXTURE), "--candidate", str(submission), "--json"])
        report = json.loads(output.getvalue())
        row = next(item for item in report["cases"] if item["strategy"] == "graph_trial")
        self.assertEqual(row["implementation_status"], "external_selection_unverified")
        self.assertIn("graph_trial", report["submission_artifact_sha256"])
        self.assertIsNone(row["repair_pass"])

    def test_missing_model_usage_is_unknown_not_free(self):
        with FIXTURE.open(encoding="utf-8") as source:
            fixture = json.load(source)
        del fixture["events"][2]["usage"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing-usage.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            report = evaluate_offline([path])
        self.assertIsNone(report["recorded_synthetic_model_cost_usd"])
        self.assertIn("n/a", render_comparison(report))

    def test_reports_do_not_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence = Path(folder) / "offline.json"
            with redirect_stdout(io.StringIO()):
                main(["--output", str(evidence)])
            before = evidence.read_text(encoding="utf-8")
            self.assertFalse(json.loads(before)["research_claim_supported"])
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                main(["--output", str(evidence)])
            self.assertEqual(before, evidence.read_text(encoding="utf-8"))

    def test_duplicate_cases_and_submission_gaps_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_offline([FIXTURE, FIXTURE])
        with self.assertRaises(ValueError):
            evaluate_offline([FIXTURE], submissions={"missing": {}})

    def test_malformed_private_checks_fail_closed(self):
        with FIXTURE.open(encoding="utf-8") as source:
            fixture = json.load(source)
        fixture["oracle"] = "not a mapping"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad-oracle.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_offline([path])

    def test_second_case_exposes_control_failure_under_same_protocol(self):
        report = evaluate_offline([FIXTURE, SECOND_FIXTURE])
        self.assertEqual(report["n_cases"], 2)
        rows = {row["strategy"]: row for row in report["cases"] if row["case_id"] == "refund-sign-position-v1"}
        self.assertIn("feature_attempt_hierarchy", rows)
        self.assertIn("contextweaver_validation_proxy", rows)
        self.assertIn("e07", rows["chronological_suffix"]["selected_event_ids"])
        self.assertNotIn("e07", rows["last_error_latest_user"]["selected_event_ids"])
        self.assertFalse(rows["last_error_latest_user"]["context_gate"])
        self.assertIn("refund-sign-position-v1", report["summary"]["last_error_latest_user"]["failed_case_ids"])


if __name__ == "__main__":
    unittest.main()
