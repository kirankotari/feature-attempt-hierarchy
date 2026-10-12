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

from harness import count_proxy_tokens, load_public_case, main, materialize_workspace, select_chronological_history


FIXTURE = ROOT / "data" / "context-repair-benchmark.json"


class HarnessTests(unittest.TestCase):
    def test_checkpoint_view_excludes_oracle_and_initial_snapshot(self):
        view = load_public_case(FIXTURE, "continuation")
        self.assertEqual(view["case_id"], "catalog-include-archived-v1")
        self.assertIn("Get the build and tests passing", view["goal"])
        self.assertEqual(view["events"][-1]["id"], "e12")
        self.assertIn("catalog.py", view["project"]["current_files"])
        self.assertEqual(view["simulated_price_usd_per_million_tokens"], {"input": 2, "output": 8})
        self.assertNotIn("oracle", json.dumps(view))
        self.assertNotIn("initial_files", json.dumps(view))
        self.assertNotIn("private_cases", json.dumps(view))

    def test_future_events_and_unknown_event_fields_are_not_exposed(self):
        with FIXTURE.open(encoding="utf-8") as source:
            raw = json.load(source)
        raw["oracle"] = {"secret": "PRIVATE_SENTINEL"}
        raw["events"][0]["untrusted_metadata"] = "PRIVATE_SENTINEL"
        raw["events"].append({"id": "e13", "turn": 4, "kind": "user", "text": "FUTURE_SENTINEL"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            view = load_public_case(path, "continuation")
        serialized = json.dumps(view)
        self.assertNotIn("PRIVATE_SENTINEL", serialized)
        self.assertNotIn("FUTURE_SENTINEL", serialized)
        self.assertEqual(len(view["events"]), 12)

    def test_invalid_checkpoint_and_duplicate_events_are_rejected(self):
        with self.assertRaises(ValueError):
            load_public_case(FIXTURE, "unknown")
        with FIXTURE.open(encoding="utf-8") as source:
            raw = json.load(source)
        raw["events"].append(dict(raw["events"][0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_public_case(path, "continuation")

    def test_path_traversal_is_rejected_before_materialization(self):
        with FIXTURE.open(encoding="utf-8") as source:
            original = json.load(source)
        for name in ("../escape.py", "/absolute.py", "nested/../escape.py", "C:\\escape.py"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                raw = json.loads(json.dumps(original))
                raw["project"]["current_files"][name] = "unsafe"
                path = Path(folder) / "fixture.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_public_case(path, "continuation")

    def test_conflicting_project_paths_are_rejected_before_writing(self):
        view = load_public_case(FIXTURE, "continuation")
        view["project"]["current_files"] = {"nested": "file", "nested/child.py": "contents"}
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "workspace"
            target.mkdir()
            with self.assertRaises(ValueError):
                materialize_workspace(view, target)
            self.assertEqual(list(target.iterdir()), [])

    def test_materialization_requires_empty_directory_and_keeps_exact_contents(self):
        view = load_public_case(FIXTURE, "continuation")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "workspace"
            target.mkdir()
            materialize_workspace(view, target)
            for name, content in view["project"]["current_files"].items():
                self.assertEqual((target / name).read_text(encoding="utf-8"), content)
            with self.assertRaises(ValueError):
                materialize_workspace(view, target)
            self.assertEqual(sorted(path.name for path in target.iterdir()), ["catalog.py", "test_catalog.py"])

    def test_chronological_baseline_keeps_contiguous_recent_events(self):
        view = load_public_case(FIXTURE, "continuation")
        selection = select_chronological_history(view)
        ids = selection["event_ids"]
        self.assertEqual(ids, [event["id"] for event in view["events"][-len(ids):]])
        self.assertEqual(ids[-1], "e12")
        self.assertNotIn("e01", ids)
        self.assertNotIn("e04", ids)
        self.assertLessEqual(selection["proxy_tokens"], view["history_budget_tokens"])
        self.assertEqual(selection["proxy_tokens"], count_proxy_tokens(selection["text"]))

    def test_budget_too_small_for_last_event_is_rejected(self):
        view = load_public_case(FIXTURE, "continuation")
        with self.assertRaises(ValueError):
            select_chronological_history(view, budget=1)

    def test_baseline_does_not_skip_an_oversized_recent_event(self):
        view = load_public_case(FIXTURE, "continuation")
        last = view["events"][-1]
        size = count_proxy_tokens(f"[{last['id']} turn={last['turn']} {last['kind']}]\n{last['text']}")
        selection = select_chronological_history(view, budget=size)
        self.assertEqual(selection["event_ids"], ["e12"])

    def test_materialization_rejects_symlink_directory(self):
        view = load_public_case(FIXTURE, "continuation")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "workspace"
            target.mkdir()
            link = Path(folder) / "alias"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                materialize_workspace(view, link)
            self.assertEqual(list(target.iterdir()), [])

    def test_default_table_shows_full_trace_and_selected_context_separately(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main([])
        text = output.getvalue()
        self.assertIn("Full visible history", text)
        self.assertIn("Projected next model input", text)
        self.assertIn("e04", text)
        self.assertIn("e12", text)
        self.assertIn("OUT", text)
        self.assertIn("IN", text)
        self.assertIn("catalog.py", text)
        self.assertIn("$0.001600", text)
        self.assertIn("n/a", text)
        prompt = text.split("Projected next model input", 1)[1]
        self.assertIn("[e09", prompt)
        self.assertNotIn("[e04", prompt)
        self.assertNotIn("oracle", text.lower())
        self.assertNotIn("private_cases", text)

    def test_prompt_view_shows_goal_task_history_but_not_private_data(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--view", "prompt"])
        text = output.getvalue()
        self.assertIn("Goal: Add --include-archived", text)
        self.assertIn("Task: Continue from the failed feature attempt", text)
        self.assertIn("[e12", text)
        self.assertNotIn("Full visible history", text)
        self.assertNotIn("[e04", text)
        self.assertNotIn("private_cases", text)

    def test_json_view_preserves_machine_readable_baseline(self):
        output = io.StringIO()
        with redirect_stdout(output):
            main(["--view", "json"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["event_ids"], ["e09", "e10", "e11", "e12"])
        self.assertEqual(result["strategy"], "chronological_suffix")
        self.assertNotIn("oracle", result)


if __name__ == "__main__":
    unittest.main()
