import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from swebench_adapter import (
    SWEBenchTrajectoryConverter,
    SWEBenchContextManager,
    SWEBenchCostTracker,
)
from models import EventKind
from tokenizers import RegexProxyTokenizer


class TestSWEBenchAdapter(unittest.TestCase):
    def setUp(self):
        self.converter = SWEBenchTrajectoryConverter()

    def test_trajectory_converter_classification(self):
        raw_messages = [
            {"id": "msg_1", "role": "user", "content": "Fix bug in Django models"},
            {"id": "msg_2", "role": "assistant", "action": "thought", "content": "I will examine models.py"},
            {"id": "msg_3", "role": "assistant", "action": "edit", "content": "patching models.py with changes"},
            {"id": "msg_4", "role": "tool", "action_cmd": "pytest tests/test_models.py", "exit_code": 1, "content": "FAIL: AssertionError in test_models"},
            {"id": "msg_5", "role": "assistant", "thread_id": "side", "content": "Checking local environment python version"},
        ]

        events = self.converter.history_to_events(raw_messages)
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0].kind, EventKind.USER)
        self.assertEqual(events[1].kind, EventKind.ASSISTANT_ANALYSIS)
        self.assertEqual(events[2].kind, EventKind.ASSISTANT_EDIT)
        self.assertEqual(events[3].kind, EventKind.TOOL_RESULT)
        self.assertEqual(events[4].thread_id, "side")

    def test_swebench_context_manager_pruning(self):
        """
        Simulates a SWE-bench trajectory where the agent makes an edit,
        encounters a test failure, runs a side detour, and then receives context for the next turn.
        """
        raw_messages = [
            {"id": "m1", "role": "user", "content": "Django Issue #1234: ForeignKey on_delete validation crash."},
            {"id": "m2", "role": "assistant", "action": "str_replace_editor", "content": "Updating models/fields.py to validate on_delete."},
            {"id": "m3", "role": "tool", "action_cmd": "pytest tests/model_fields/tests.py", "exit_code": 1, "content": "FAILED: KeyError: 'on_delete' in validate()"},
            {"id": "m4", "role": "assistant", "thread_id": "side", "content": "Let me check git status and current branch settings..."},
            {"id": "m5", "role": "tool", "thread_id": "side", "content": "On branch main, everything clean."},
            {"id": "m6", "role": "user", "content": "Remember to maintain backwards compatibility."},
        ]

        # Use RegexProxyTokenizer so test runs deterministically in offline environments
        tok = RegexProxyTokenizer()
        manager = SWEBenchContextManager(token_budget=100, tokenizer=tok, fail_closed=False)

        pruned = manager.select_prompt_messages(raw_messages, budget=40)
        pruned_ids = [m["id"] for m in pruned]

        # m1 (original issue) must be retained
        self.assertIn("m1", pruned_ids)
        # m3 (active open failure) must be retained
        self.assertIn("m3", pruned_ids)
        # m2 (causing edit) must be retained
        self.assertIn("m2", pruned_ids)
        # m4 & m5 (side detour) must be pruned
        self.assertNotIn("m4", pruned_ids)
        self.assertNotIn("m5", pruned_ids)

    def test_swebench_cost_tracker_metrics(self):
        """Tests Token Economics FinOps reporting on SWE-bench trajectories."""
        raw_messages = [
            {"id": "m1", "role": "user", "content": "Fix issue."},
            {
                "id": "m2", "role": "assistant", "content": "First attempt", "action": "edit",
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200}
            },
            {
                "id": "m3", "role": "tool", "action_cmd": "pytest", "exit_code": 1,
                "content": "FAIL: test failed with KeyError"
            },
            {
                "id": "m4", "role": "assistant", "content": "Second attempt (rework)", "action": "edit",
                "usage": {"prompt_tokens": 1200, "completion_tokens": 250}
            },
            {
                "id": "m5", "role": "assistant", "thread_id": "side", "content": "Detour",
                "usage": {"prompt_tokens": 300, "completion_tokens": 50}
            }
        ]

        tracker = SWEBenchCostTracker(p_in=2.0, p_out=8.0)
        report = tracker.analyze_trajectory(raw_messages)

        self.assertEqual(report.first_issue_turn, 3)
        self.assertEqual(report.assistant_calls_count, 3)
        self.assertGreater(report.feature_cost, 0.0)
        self.assertGreater(report.rework_cost, 0.0)
        self.assertGreater(report.side_cost, 0.0)
        # m4 occurred after turn 3 issue, so its cost must be part of rework
        m4_cost = (1200 * 2.0 + 250 * 8.0) / 1_000_000.0
        self.assertAlmostEqual(report.rework_cost, m4_cost, places=5)


if __name__ == "__main__":
    unittest.main()
