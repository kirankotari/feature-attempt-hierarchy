import argparse
import hashlib
import json
from pathlib import Path

from baselines import select_combined_proxy, select_dependency_lexical_proxy, select_thread_path_proxy
from harness import DEFAULT_FIXTURE, load_public_case, select_chronological_history, select_event_ids, select_last_error_latest_user
from hierarchy import build_hierarchy, feature_cost, select_feature_context


DEFAULT_FIXTURES = (DEFAULT_FIXTURE, DEFAULT_FIXTURE.with_name("case_refund_format.json"),
                    DEFAULT_FIXTURE.with_name("case_old_constraint.json"),
                    DEFAULT_FIXTURE.with_name("case_two_open_failures.json"),
                    DEFAULT_FIXTURE.with_name("case_long_detour.json"))


def load_submission(path):
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Submission is too large")
    with path.open(encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict) or set(raw) != {"strategy", "cases"}:
        raise ValueError("Submission needs strategy and cases only")
    name, cases = raw["strategy"], raw["cases"]
    if not isinstance(name, str) or not name or not isinstance(cases, dict) or not cases:
        raise ValueError("Invalid submission")
    selections = {}
    for case_id, value in cases.items():
        if not isinstance(case_id, str) or not isinstance(value, dict) or set(value) != {"event_ids"}:
            raise ValueError("Each case needs only event_ids")
        selections[case_id] = value["event_ids"]
    return name, selections


def _checks(raw, visible):
    oracle = raw.get("oracle")
    if not isinstance(oracle, dict) or not isinstance(oracle.get("context_checks"), dict):
        raise ValueError("Missing private context checks")
    checks = oracle["context_checks"]
    required, irrelevant = checks.get("required_event_ids"), checks.get("known_irrelevant_event_ids")
    available = {event["id"] for event in visible["events"]}
    if (not isinstance(required, list) or not required or not isinstance(irrelevant, list)
            or any(not isinstance(event_id, str) for event_id in required + irrelevant)
            or len(set(required)) != len(required) or len(set(irrelevant)) != len(irrelevant)
            or set(required) & set(irrelevant) or not set(required + irrelevant) <= available):
        raise ValueError("Missing, conflicting, or unavailable private context checks")
    return set(required), set(irrelevant)


def _row(case, name, selection, required, irrelevant, full_units, workspace_hash, status):
    selected = set(selection["event_ids"])
    recall = len(selected & required) / len(required)
    stale = len(selected & irrelevant)
    history_units = selection["proxy_tokens"]
    rate = case["simulated_price_usd_per_million_tokens"]["input"]
    return {
        "case_id": case["case_id"], "strategy": name, "implementation_status": status,
        "selected_event_ids": selection["event_ids"], "history_budget_proxy_units": selection["budget"],
        "history_proxy_units": history_units, "full_history_proxy_units": full_units,
        "discarded_history_proxy_units": full_units - history_units,
        "workspace_sha256": workspace_hash, "required_recall": recall,
        "known_irrelevant_included": stale, "context_gate": recall == 1.0 and stale == 0,
        "estimated_history_input_usd_per_1k": round(history_units * rate / 1_000_000 * 1000, 6),
        "actual_model_cost_usd": None, "repair_pass": None, "latency_s": None,
    }


def evaluate_offline(fixture_paths, submissions=None):
    paths = [Path(path) for path in fixture_paths]
    if not paths:
        raise ValueError("At least one fixture is required")
    submissions = submissions or {}
    builtins = {"chronological_suffix", "last_error_latest_user", "feature_attempt_hierarchy",
                "thread_path_suffix_proxy", "dependency_lexical_proxy", "contextweaver_validation_proxy",
                "combined_feature_proxy"}
    statuses = {"chronological_suffix": "built_in_control", "last_error_latest_user": "built_in_control",
                "feature_attempt_hierarchy": "project_prototype",
                "thread_path_suffix_proxy": "memotree_exploratory_proxy",
                "dependency_lexical_proxy": "contextweaver_exploratory_proxy",
                "contextweaver_validation_proxy": "contextweaver_exploratory_proxy",
                "combined_feature_proxy": "local_combination_proxy"}
    if any(not isinstance(name, str) or not name or name in builtins for name in submissions):
        raise ValueError("Submission names must not reuse built-in strategies")
    rows, fingerprints, visible_ids, attribution = [], {}, set(), {}
    historical_cost = 0.0
    usage_incomplete = False
    for path in paths:
        with path.open(encoding="utf-8") as source:
            raw = json.load(source)
        checkpoints = raw.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 1:
            raise ValueError("This runner requires exactly one checkpoint per fixture")
        case = load_public_case(path, checkpoints[0]["id"])
        case_id = case["case_id"]
        if case_id in visible_ids:
            raise ValueError("Duplicate case ID")
        visible_ids.add(case_id)
        required, irrelevant = _checks(raw, case)
        fingerprints[case_id] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        workspace = json.dumps(case["project"]["current_files"], sort_keys=True).encode("utf-8")
        workspace_hash = hashlib.sha256(workspace).hexdigest()[:12]
        all_ids = [event["id"] for event in case["events"]]
        full_units = select_event_ids(case, all_ids, budget=1_000_000_000)["proxy_tokens"]
        prices = case["simulated_price_usd_per_million_tokens"]
        usage_incomplete |= any(event["kind"].startswith("assistant") and "usage" not in event for event in case["events"])
        historical_cost += sum((event["usage"]["input_tokens"] * prices["input"]
                                + event["usage"]["output_tokens"] * prices["output"]) / 1_000_000
                               for event in case["events"] if "usage" in event)
        graph = build_hierarchy(case)
        attribution[case_id] = feature_cost(case, graph)
        reference = raw["oracle"].get("simulated_cost_reference", {})
        checks = {"feature_recorded_synthetic_usd": "feature_past_micro_usd",
                  "side_recorded_synthetic_usd": "unrelated_past_micro_usd",
                  "feature_rework_recorded_synthetic_usd": "feature_rework_past_micro_usd"}
        expected = {key: reference.get(field) / 1_000_000 for key, field in checks.items() if type(reference.get(field)) is int}
        attribution[case_id]["matches_private_synthetic_reference"] = (all(attribution[case_id][key] == value for key, value in expected.items()) if len(expected) == len(checks) else None)
        selections = {
            "chronological_suffix": select_chronological_history(case),
            "last_error_latest_user": select_last_error_latest_user(case),
            "feature_attempt_hierarchy": select_feature_context(case, graph),
            "thread_path_suffix_proxy": select_thread_path_proxy(case),
            "dependency_lexical_proxy": select_dependency_lexical_proxy(case),
            "contextweaver_validation_proxy": select_dependency_lexical_proxy(case, skip_failed=True),
            "combined_feature_proxy": select_combined_proxy(case),
        }
        for name, per_case in submissions.items():
            if not isinstance(per_case, dict) or case_id not in per_case:
                raise ValueError(f"Submission {name!r} missing case {case_id!r}")
            selections[name] = select_event_ids(case, per_case[case_id])
        for name, selection in selections.items():
            status = statuses.get(name, "external_selection_unverified")
            rows.append(_row(case, name, selection, required, irrelevant, full_units, workspace_hash, status))
    for name, per_case in submissions.items():
        if set(per_case) != visible_ids:
            raise ValueError(f"Submission {name!r} has extra or missing case IDs")
    summaries = {}
    for name in sorted({row["strategy"] for row in rows}):
        group = [row for row in rows if row["strategy"] == name]
        summaries[name] = {
            "context_gate_count": sum(row["context_gate"] for row in group),
            "n_cases": len(group),
            "mean_history_proxy_units": round(sum(row["history_proxy_units"] for row in group) / len(group), 2),
            "mean_discarded_history_proxy_units": round(sum(row["discarded_history_proxy_units"] for row in group) / len(group), 2),
            "mean_estimated_history_input_usd_per_1k": round(sum(row["estimated_history_input_usd_per_1k"] for row in group) / len(group), 6),
            "failed_case_ids": sorted(row["case_id"] for row in group if not row["context_gate"]),
        }
    baseline_failed = set(summaries["chronological_suffix"]["failed_case_ids"])
    for summary in summaries.values():
        failed = set(summary["failed_case_ids"])
        summary["fixed_vs_chronological"] = sorted(baseline_failed - failed)
        summary["regressed_vs_chronological"] = sorted(failed - baseline_failed)
    return {
        "mode": "offline_selection_only", "n_cases": len(paths), "fixture_sha256": fingerprints,
        "research_claim_supported": False, "cost_basis": "regex proxy units and synthetic input price; not provider tokens or billing",
        "recorded_synthetic_model_cost_usd": None if usage_incomplete else round(historical_cost, 8),
        "feature_cost_attribution": attribution,
        "repair_success": None, "actual_model_cost_usd": None,
        "limitations": ["No model calls or measured repair outcomes", "Only context-event selection is evaluated, not summary fidelity", "Thread-based feature cost relies on provided metadata and synthetic rates, not real bills", "Fixture oracles share files with public inputs; real blinded studies need restricted held-out cases", "ContextWeaver rows are lexical proxies for arXiv:2604.23069, not its LLM parent ranker, summarizer, or SWE-bench runs", "The MemoTree row walks superseded attempts because the fixtures have no MemoTree parent ids; it is not the thapakrish/memotree app and not the MemTree paper", "One fixture does not support generalization" if len(paths) == 1 else "Broader independent cases and held-out outcomes remain necessary"],
        "summary": summaries, "cases": rows,
    }


def render_comparison(report):
    recorded = report["recorded_synthetic_model_cost_usd"]
    recorded_label = "n/a (missing model usage)" if recorded is None else f"${recorded:.6f}"
    lines = ["OFFLINE SELECTION ONLY — NOT a repair success or billed cost result",
             f"cases: {report['n_cases']} | recorded synthetic past model calls: {recorded_label}",
             "Cost below is an estimated history-only input $/1k cases from regex proxy units; outcome and actual cost are n/a.",
             "ContextWeaver and MemoTree rows are local proxies, not published system runs.",
             "", "Per-case checks (gate = required evidence retained and known irrelevant events excluded):",
             f"{'case':<29} {'strategy':<32} {'gate':<6} {'recall':>7} {'stale':>6} {'proxy':>7} {'discarded':>10} {'$/1k*':>9} {'repair':>8} {'actual $':>8}"]
    for row in report["cases"]:
        lines.append(f"{row['case_id'][:28]:<29} {row['strategy'][:31]:<32} "
                     f"{'PASS' if row['context_gate'] else 'FAIL':<6} {row['required_recall']:>7.0%} "
                     f"{row['known_irrelevant_included']:>6} {row['history_proxy_units']:>7} "
                     f"{row['discarded_history_proxy_units']:>10} "
                     f"{row['estimated_history_input_usd_per_1k']:>9.4f} {'n/a':>8} {'n/a':>8}")
    lines.extend(["", "Aggregate context checks (not code-repair success):",
                  f"{'strategy':<32} {'gate/cases':>11} {'mean proxy':>11} {'discarded':>10} {'mean $/1k*':>12}  vs chronological"])
    for name, summary in report["summary"].items():
        lines.append(f"{name[:31]:<32} {str(summary['context_gate_count']) + '/' + str(summary['n_cases']):>11} "
                     f"{summary['mean_history_proxy_units']:>11.1f} {summary['mean_discarded_history_proxy_units']:>10.1f} "
                     f"{summary['mean_estimated_history_input_usd_per_1k']:>12.4f}  "
                     f"fixed={summary['fixed_vs_chronological'] or '-'} regressed={summary['regressed_vs_chronological'] or '-'}")
    lines.extend(["", "Feature attribution from instrumented thread IDs (synthetic past model calls only):"])
    for case_id, cost in report["feature_cost_attribution"].items():
        amount = lambda value: "n/a" if value is None else f"${value:.6f}"
        lines.append(f"{case_id}: feature={amount(cost['feature_recorded_synthetic_usd'])} "
                     f"rework={amount(cost['feature_rework_recorded_synthetic_usd'])} "
                     f"side={amount(cost['side_recorded_synthetic_usd'])} "
                     f"reference_match={cost['matches_private_synthetic_reference']}")
    lines.append("* Proxy pricing excludes goal, task, workspace, tool costs, cache effects and continuation calls.")
    if report["n_cases"] == 1:
        lines.append("Only one synthetic fixture: no generalization or game-changer claim is supported.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline context-selection comparison; never calls a model or runs fixture code")
    parser.add_argument("--fixture", action="append", type=Path, help="Repeat for multiple independent fixture files")
    parser.add_argument("--candidate", action="append", type=Path, help="External selection JSON (strategy and case-to-event IDs)")
    parser.add_argument("--json", action="store_true", help="Print the evidence report as JSON")
    parser.add_argument("--output", type=Path, help="Save evidence report to a new JSON file; never overwrite")
    args = parser.parse_args(argv)
    submissions, hashes = {}, {}
    for path in args.candidate or []:
        name, selections = load_submission(path)
        if name in submissions:
            raise ValueError(f"Duplicate candidate name: {name}")
        submissions[name] = selections
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    report = evaluate_offline(args.fixture or DEFAULT_FIXTURES, submissions=submissions)
    report["submission_artifact_sha256"] = hashes
    display = json.dumps(report, indent=2) if args.json else render_comparison(report)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, indent=2, sort_keys=True)
            destination.write("\n")
    print(display)
    return 0


if __name__ == "__main__":
    main()
