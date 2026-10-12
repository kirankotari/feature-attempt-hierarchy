import re

from harness import select_event_ids


def _failure_type(event):
    if event["kind"] != "tool_result":
        return None
    text = event["text"].lower()
    if "syntaxerror" in text:
        return "syntax"
    if "traceback" in text or "keyerror" in text:
        return "exception"
    if "exit=1" in text or ("expected:" in text and "observed:" in text):
        return "mismatch"
    return None


def _is_successful_compile(event):
    return (event["kind"] == "tool_result" and "py_compile" in event.get("tool", "")
            and re.search(r"\bOK\b", event["text"]) is not None and _failure_type(event) is None)


def build_hierarchy(case):
    root = f"feature:{case['case_id']}"
    attempts, issues, parents, active_attempt = {}, [], {}, None
    thread_sources = {}
    for event in case["events"]:
        event_id = event["id"]
        thread_id = event.get("thread_id", "feature")
        thread_sources[event_id] = "instrumented" if "thread_id" in event else "fallback_feature"
        if thread_id != "feature":
            parents[event_id] = f"thread:{thread_id}"
            continue
        if event["kind"] == "assistant_edit":
            if active_attempt is not None:
                attempts[active_attempt]["status"] = "superseded"
            active_attempt = f"attempt:{event_id}"
            attempts[active_attempt] = {"parent": root, "edit_event_id": event_id, "events": [], "status": "active"}
        parent = active_attempt if active_attempt and event["kind"] != "user" else root
        parents[event_id] = parent
        if parent in attempts:
            attempts[parent]["events"].append(event_id)
        if _is_successful_compile(event):
            for issue in issues:
                if issue["type"] == "syntax" and issue["status"] == "open":
                    issue["status"] = "resolved"
                    issue["resolved_by"] = event_id
        failure = _failure_type(event)
        if failure is not None:
            issues.append({"event_id": event_id, "attempt_id": active_attempt, "type": failure,
                           "status": "open", "resolved_by": None, "link_basis": "inferred_from_sequence"})
    return {"root": root, "attempts": attempts, "parents": parents,
            "issues": issues, "thread_sources": thread_sources}


def select_feature_context(case, graph=None, budget=None):
    graph = graph if graph is not None else build_hierarchy(case)
    feature_events = [event for event in case["events"] if not graph["parents"][event["id"]].startswith("thread:")]
    latest_user = next((event for event in reversed(feature_events) if event["kind"] == "user"), None)
    open_issues = [issue for issue in graph["issues"] if issue["status"] == "open"]
    chosen = []

    def add(event_id):
        if not event_id or event_id in chosen:
            return
        try:
            select_event_ids(case, [*chosen, event_id], budget=budget)
        except ValueError as error:
            if "exceeds the history budget" not in str(error):
                raise
            return
        chosen.append(event_id)

    for issue in reversed(open_issues):
        add(issue["event_id"])
    if latest_user is not None:
        add(latest_user["id"])
    if not chosen:
        raise ValueError("No active feature context")
    for issue in reversed(open_issues):
        if issue["attempt_id"]:
            add(graph["attempts"][issue["attempt_id"]]["edit_event_id"])
    for event in reversed(feature_events):
        if event["kind"] == "user":
            add(event["id"])
    selection = select_event_ids(case, chosen, budget=budget)
    selection["strategy"] = "feature_attempt_hierarchy"
    return selection


def feature_cost(case, graph=None):
    graph = graph if graph is not None else build_hierarchy(case)
    prices = case["simulated_price_usd_per_million_tokens"]
    first_failure = next((index for index, event in enumerate(case["events"])
                          if any(issue["event_id"] == event["id"] for issue in graph["issues"])), None)
    totals = {"feature": 0.0, "side": 0.0, "rework": 0.0}
    unknown = {"feature": False, "side": False, "rework": False}
    for index, event in enumerate(case["events"]):
        if not event["kind"].startswith("assistant"):
            continue
        partition = "feature" if graph["parents"][event["id"]].startswith(("feature:", "attempt:")) else "side"
        partitions = [partition]
        if partition == "feature" and first_failure is not None and index > first_failure:
            partitions.append("rework")
        usage = event.get("usage")
        for key in partitions:
            if usage is None:
                unknown[key] = True
            else:
                totals[key] += (usage["input_tokens"] * prices["input"]
                                + usage["output_tokens"] * prices["output"]) / 1_000_000
    return {"feature_recorded_synthetic_usd": None if unknown["feature"] else round(totals["feature"], 8),
            "side_recorded_synthetic_usd": None if unknown["side"] else round(totals["side"], 8),
            "feature_rework_recorded_synthetic_usd": None if unknown["rework"] else round(totals["rework"], 8),
            "basis": "instrumented thread IDs and sequence-inferred repair boundary; synthetic usage, not billed cost"}


def render_hierarchy(case, graph=None):
    graph = graph if graph is not None else build_hierarchy(case)
    issues = {issue["event_id"]: issue for issue in graph["issues"]}
    lines = [f"{graph['root']} (goal)"]
    shown = set()
    for event in case["events"]:
        event_id = event["id"]
        parent = graph["parents"][event_id]
        if parent.startswith("thread:") and parent not in shown:
            lines.append(f"  {parent} (out-of-scope)")
            shown.add(parent)
        if parent.startswith("attempt:") and parent not in shown:
            lines.append(f"  {parent} ({graph['attempts'][parent]['status']})")
            shown.add(parent)
        status = f" [{issues[event_id]['status']}]" if event_id in issues else ""
        depth = "    " if parent.startswith(("attempt:", "thread:")) else "  "
        lines.append(f"{depth}{event_id}{status} {event['kind']}")
    return "\n".join(lines)
