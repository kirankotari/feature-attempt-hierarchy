import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from models import Event, EventKind, TokenUsage
from tokenizers import RegexProxyTokenizer, WhitespaceTokenizer
from context_selectors import FeatureAttemptSelector, SuffixWindowSelector, LatestErrorSelector


class TestComprehensiveHierarchy(unittest.TestCase):
    def setUp(self):
        self.tok = RegexProxyTokenizer()

    def test_suffix_window_fails_on_detours(self):
        """
        Tests that SuffixWindowSelector drops the critical original failure when
        a side detour fills the budget.
        """
        events = [
            Event("u1", 1, EventKind.USER, "Fix payment gateway timeout bug"),
            Event("e1", 2, EventKind.ASSISTANT_EDIT, "Patching client timeout parameter"),
            Event("t1", 3, EventKind.TOOL_RESULT, "pytest failed with ConnectionTimeoutError: timed out", exit_code=1),
            # Side detour events
            Event("s1", 4, EventKind.ASSISTANT_ANALYSIS, "Let me inspect the documentation and style guidelines", thread_id="side"),
            Event("s2", 5, EventKind.TOOL_RESULT, "documentation file read successfully", thread_id="side"),
            Event("s3", 6, EventKind.ASSISTANT_ANALYSIS, "Checking project changelog and version tags", thread_id="side"),
            Event("s4", 7, EventKind.TOOL_RESULT, "changelog entries retrieved", thread_id="side"),
        ]

        # Under small budget, sliding window fills with side events and drops t1
        suffix_res = SuffixWindowSelector().select(events, budget=25, tokenizer=self.tok)
        self.assertNotIn("t1", suffix_res.selected_ids)

        # Hierarchy selector skips side events and keeps t1 and u1
        hier_res = FeatureAttemptSelector().select(events, budget=25, tokenizer=self.tok)
        self.assertIn("t1", hier_res.selected_ids)
        self.assertNotIn("s1", hier_res.selected_ids)
        self.assertNotIn("s2", hier_res.selected_ids)

    def test_latest_error_fails_on_multiple_open_defects(self):
        """
        When two distinct errors exist, LatestError drops the first error,
        while FeatureAttemptSelector retains both up to budget.
        """
        events = [
            Event("u1", 1, EventKind.USER, "Implement multi-currency checkout"),
            Event("e1", 2, EventKind.ASSISTANT_EDIT, "Add EUR conversion"),
            Event("t1", 3, EventKind.TOOL_RESULT, "pytest tests/test_eur.py failed with AssertionError", exit_code=1),
            Event("e2", 4, EventKind.ASSISTANT_EDIT, "Add GBP conversion"),
            Event("t2", 5, EventKind.TOOL_RESULT, "pytest tests/test_gbp.py failed with KeyError: 'GBP'", exit_code=1),
        ]

        latest_res = LatestErrorSelector().select(events, budget=40, tokenizer=self.tok)
        self.assertIn("t2", latest_res.selected_ids)
        self.assertNotIn("t1", latest_res.selected_ids)  # Drops older open error

        hier_res = FeatureAttemptSelector().select(events, budget=40, tokenizer=self.tok)
        self.assertIn("t1", hier_res.selected_ids)
        self.assertIn("t2", hier_res.selected_ids)

    def test_fail_closed_on_empty_context(self):
        """
        Tests Algorithm 1 lines 9-11: if C is empty (no open issues and no user prompt fits),
        strict mode emits fail, whereas fallback mode provides recent events.
        """
        events_without_user = [
            Event("a1", 1, EventKind.ASSISTANT_ANALYSIS, "Autonomous background exploration step 1"),
            Event("t1", 2, EventKind.TOOL_RESULT, "Files indexed successfully with exit code 0", exit_code=0),
        ]

        res_strict = FeatureAttemptSelector(fail_closed=True).select(events_without_user, budget=50, tokenizer=self.tok)
        self.assertFalse(res_strict.passed_gate)
        self.assertEqual(res_strict.error_message, "fail: no active feature context")

        res_fallback = FeatureAttemptSelector(fail_closed=False).select(events_without_user, budget=50, tokenizer=self.tok)
        self.assertTrue(res_fallback.passed_gate)
        self.assertIn("a1", res_fallback.selected_ids)
        self.assertIn("t1", res_fallback.selected_ids)


if __name__ == "__main__":
    unittest.main()
