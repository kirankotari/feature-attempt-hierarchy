import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from models import Event, EventKind, TokenUsage
from tokenizers import RegexProxyTokenizer
from hierarchy import HierarchyBuilder
from selectors import FeatureAttemptSelector
from ledger import CostLedger


class TestWorkedCatalogTrace(unittest.TestCase):
    """
    Reproduces the exact catalog feature trace from Section 4 and Figure 2 of Kotari (2026).
    """
    def setUp(self):
        self.tokenizer = RegexProxyTokenizer()
        # Events corresponding exactly to Section 4 and Figure 2 in Kotari (2026)
        self.events = [
            Event("e01", 1, EventKind.USER, "implement catalog feature with caching and test harness"),
            Event("e02", 2, EventKind.TOOL_RESULT, "inspect codebase structure and files list"),
            Event("e03", 3, EventKind.ASSISTANT_EDIT, "write initial catalog implementation with memory cache",
                  usage=TokenUsage(200, 80)),
            Event("e04", 4, EventKind.TOOL_RESULT, "py_compile catalog.py failed with SyntaxError: unexpected EOF",
                  exit_code=1),
            Event("e05", 5, EventKind.ASSISTANT_ANALYSIS, "analyze syntax error in catalog.py",
                  usage=TokenUsage(150, 40)),
            Event("e06", 6, EventKind.TOOL_RESULT, "py_compile catalog.py succeeded ok", exit_code=0),
            Event("e07", 7, EventKind.ASSISTANT_EDIT, "add redis caching layer to catalog service",
                  usage=TokenUsage(240, 90)),
            Event("e08", 8, EventKind.TOOL_RESULT, "pytest tests/test_catalog.py passed ok 3 tests", exit_code=0),
            Event("e09", 9, EventKind.TOOL_RESULT, "redis connection error: KeyError: 'title' on row",
                  exit_code=1),
            Event("e10", 10, EventKind.USER, "what python version is running in this container?",
                  thread_id="side"),
            Event("e11", 11, EventKind.ASSISTANT_ANALYSIS, "python version is 3.12.2",
                  thread_id="side", usage=TokenUsage(180, 50)),
            Event("e12", 12, EventKind.USER, "make sure cache gracefully falls back to memory if redis fails"),
        ]

    def test_hierarchy_tree_structure(self):
        builder = HierarchyBuilder()
        tree = builder.build(self.events)

        self.assertEqual(len(tree.user_instructions), 2)  # e01, e12
        self.assertEqual(len(tree.attempts), 2)  # attempt e03, attempt e07
        self.assertTrue(tree.attempts[0].is_superseded)
        self.assertFalse(tree.attempts[1].is_superseded)
        self.assertEqual(len(tree.side_threads["side"]), 2)  # e10, e11

        # e04 was resolved by e06
        self.assertEqual(len(tree.attempts[0].open_issues), 0)
        self.assertEqual(len(tree.attempts[1].open_issues), 1)  # e09 is open

    def test_hierarchy_selection(self):
        """
        Under budget 320, selection emits [e01, e07, e09, e12].
        """
        selector = FeatureAttemptSelector()
        result = selector.select(self.events, budget=320, tokenizer=self.tokenizer)

        self.assertTrue(result.passed_gate)
        # Exactly [e01, e07, e09, e12] as stated in Section 4 and Figure 2
        self.assertEqual(result.selected_ids, ["e01", "e07", "e09", "e12"])

    def test_cost_ledger(self):
        ledger = CostLedger()
        report = ledger.compute(self.events, p_in=2.0, p_out=8.0)

        self.assertEqual(report.first_issue_turn, 4)
        self.assertGreater(report.feature_cost, 0.0)
        self.assertGreater(report.rework_cost, 0.0)
        self.assertGreater(report.side_cost, 0.0)
        # e11 is side thread: (180*2 + 50*8)/10^6 = (360 + 400)/10^6 = 0.000760
        self.assertAlmostEqual(report.side_cost, 0.000760, places=6)


if __name__ == "__main__":
    unittest.main()
