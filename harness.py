import argparse
import json
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath


DEFAULT_FIXTURE = Path(__file__).resolve().parent / "data" / "context-repair-benchmark.json"


def _safe_relative_name(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or "\x00" in name
        or any(part in ("", ".", "..") for part in name.split("/"))
        or Path(name).is_absolute()
        or PureWindowsPath(name).drive
    ):
        raise ValueError(f"Unsafe project path: {name!r}")
    return Path(name)


def _valid_files(files):
    if not isinstance(files, dict) or not files or len(files) > 32:
        raise ValueError("Expected 1-32 project files")
    for name, content in files.items():
        _safe_relative_name(name)
        if not isinstance(content, str) or len(content.encode("utf-8")) > 128_000:
            raise ValueError(f"Invalid content for {name!r}")
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents if parent.as_posix() != "."):
            raise ValueError(f"Project path conflicts with a parent file: {name!r}")
    return dict(files)


def load_public_case(path, checkpoint_id):
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Fixture is too large")
    with path.open(encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ValueError("Unsupported fixture schema")
    checkpoints = raw.get("checkpoints")
    events = raw.get("events")
    project = raw.get("project")
    contract = raw.get("delivery_contract")
    if not isinstance(checkpoints, list) or not isinstance(events, list) or not isinstance(project, dict) or not isinstance(contract, dict):
        raise ValueError("Malformed fixture")
    if len(events) > 5_000 or not events:
        raise ValueError("Invalid event count")
    matches = [item for item in checkpoints if isinstance(item, dict) and item.get("id") == checkpoint_id]
    if len(matches) != 1 or matches[0].get("workspace") != "project.current_files":
        raise ValueError("Unknown or unsupported checkpoint")
    files = _valid_files(project.get("current_files"))
    budget = contract.get("history_prompt_budget_tokens")
    if type(budget) is not int or budget <= 0:
        raise ValueError("Invalid history budget")
    prices = contract.get("simulated_price_usd_per_million_tokens")
    if (
        not isinstance(prices, dict)
        or set(prices) != {"input", "output"}
        or any(type(price) not in (int, float) or not math.isfinite(price) or price < 0 for price in prices.values())
    ):
        raise ValueError("Invalid synthetic price")
    seen = set()
    public_events = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Malformed event")
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError("Missing or duplicate event ID")
        seen.add(event_id)
        if type(event.get("turn")) is not int or event["turn"] < 1:
            raise ValueError("Invalid event turn")
        if not isinstance(event.get("kind"), str) or not isinstance(event.get("text"), str) or len(event["text"]) > 50_000:
            raise ValueError("Invalid event content")
        visible = {key: event[key] for key in ("id", "turn", "kind", "text")}
        if "thread_id" in event:
            thread_id = event["thread_id"]
            if not isinstance(thread_id, str) or not thread_id or len(thread_id) > 64 or not thread_id.isidentifier():
                raise ValueError("Invalid thread ID")
            visible["thread_id"] = thread_id
        if "tool" in event:
            if not isinstance(event["tool"], str):
                raise ValueError("Invalid tool name")
            visible["tool"] = event["tool"]
        if "usage" in event:
            usage = event["usage"]
            if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens")):
                raise ValueError("Invalid usage")
            visible["usage"] = {key: usage[key] for key in ("input_tokens", "output_tokens")}
        public_events.append(visible)
    cutoff = next((index for index, event in enumerate(public_events) if event["id"] == matches[0].get("through_event")), None)
    if cutoff is None:
        raise ValueError("Checkpoint event not found")
    case_id, description, goal, task = raw.get("case_id"), raw.get("description"), raw.get("goal"), matches[0].get("task")
    if not all(isinstance(value, str) and value for value in (case_id, description, goal, task)):
        raise ValueError("Invalid case metadata")
    decision = raw.get("decision_task")
    if (not isinstance(decision, dict) or not isinstance(decision.get("question"), str)
            or not isinstance(decision.get("options"), dict) or not 2 <= len(decision["options"]) <= 4
            or any(not isinstance(key, str) or not key.isidentifier() or not isinstance(value, str) or not value
                   for key, value in decision["options"].items())):
        raise ValueError("Invalid public decision task")
    return {
        "case_id": case_id,
        "description": description,
        "goal": goal,
        "decision_task": {"question": decision["question"], "options": dict(decision["options"])}, 
        "checkpoint": {"id": checkpoint_id, "task": task},
        "project": {"current_files": files},
        "events": public_events[:cutoff + 1],
        "history_budget_tokens": budget,
        "simulated_price_usd_per_million_tokens": dict(prices),
    }


def materialize_workspace(case, destination):
    files = _valid_files(case["project"]["current_files"])
    destination = Path(destination)
    if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("Destination must be an existing, empty, non-symlink directory")
    for name, content in files.items():
        target = destination / _safe_relative_name(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def count_proxy_tokens(text):
    return len(re.findall(r"\w+|[^\w\s]", text))


def _render_event(event):
    header = f"[{event['id']} turn={event['turn']} {event['kind']}]"
    if "tool" in event:
        header += f" tool={event['tool']}"
    return f"{header}\n{event['text']}"


def select_chronological_history(case, budget=None):
    if budget is None:
        budget = case["history_budget_tokens"]
    if type(budget) is not int or budget <= 0:
        raise ValueError("History budget must be positive")
    selected = []
    event_ids = []
    for event in reversed(case["events"]):
        candidate = [_render_event(event), *selected]
        if count_proxy_tokens("\n\n".join(candidate)) > budget:
            break
        selected = candidate
        event_ids.insert(0, event["id"])
    if not selected:
        raise ValueError("Latest event does not fit the history budget")
    text = "\n\n".join(selected)
    return {
        "strategy": "chronological_suffix",
        "event_ids": event_ids,
        "text": text,
        "proxy_tokens": count_proxy_tokens(text),
        "budget": budget,
        "budget_metric": "regex_proxy_tokens_not_model_or_billing_tokens",
    }


def select_event_ids(case, event_ids, budget=None):
    if not isinstance(event_ids, list) or not event_ids or any(not isinstance(event_id, str) for event_id in event_ids):
        raise ValueError("Selection must contain event IDs")
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Duplicate event ID in selection")
    available = {event["id"] for event in case["events"]}
    if not set(event_ids) <= available:
        raise ValueError("Selection includes an unavailable event")
    if budget is None:
        budget = case["history_budget_tokens"]
    if type(budget) is not int or budget <= 0:
        raise ValueError("History budget must be positive")
    chosen = [event for event in case["events"] if event["id"] in event_ids]
    text = "\n\n".join(_render_event(event) for event in chosen)
    used = count_proxy_tokens(text)
    if used > budget:
        raise ValueError("Selection exceeds the history budget")
    return {"strategy": "selected_events", "event_ids": [event["id"] for event in chosen], "text": text,
            "proxy_tokens": used, "budget": budget, "budget_metric": "regex_proxy_tokens_not_model_or_billing_tokens"}


def select_last_error_latest_user(case, budget=None):
    latest_user = next((event for event in reversed(case["events"]) if event["kind"] == "user"), None)
    latest_error = next((event for event in reversed(case["events"]) if event["kind"] == "tool_result"
                         and re.search(r"traceback|error|failed|\bfail\b", event["text"], re.IGNORECASE)), None)
    event_ids = [event["id"] for event in (latest_error, latest_user) if event is not None]
    return select_event_ids(case, list(dict.fromkeys(event_ids)), budget=budget)


def projected_context(case, selection):
    files = ", ".join(sorted(case["project"]["current_files"]))
    return "\n".join((
        "Projected next model input (no LLM call has been made)",
        f"Goal: {case['goal']}",
        f"Task: {case['checkpoint']['task']}",
        f"Workspace files available separately, not inserted into history: {files}",
        f"Selected history: {', '.join(selection['event_ids'])} ({selection['proxy_tokens']}/{selection['budget']} regex proxy units)",
        "History:",
        selection["text"],
    ))


def _short(value, limit):
    visible = json.dumps(value, ensure_ascii=True)[1:-1]
    return visible if len(visible) <= limit else visible[:limit - 3] + "..."


def render_table(case, selection):
    selected = set(selection["event_ids"])
    prices = case["simulated_price_usd_per_million_tokens"]
    headers = ("ID", "Turn", "Prompt", "Kind", "Tool", "Cost", "Preview")
    rows = []
    recorded_cost = 0.0
    for event in case["events"]:
        cost = "n/a"
        if "usage" in event:
            usage = event["usage"]
            billed = (usage["input_tokens"] * prices["input"] + usage["output_tokens"] * prices["output"]) / 1_000_000
            recorded_cost += billed
            cost = f"${billed:.6f}"
        rows.append((event["id"], str(event["turn"]), "IN" if event["id"] in selected else "OUT",
                     event["kind"], _short(event.get("tool", ""), 22), cost, _short(event["text"], 54)))
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    def line(row):
        return " | ".join(value.ljust(width) for value, width in zip(row, widths))
    table = "\n".join((line(headers), "-+-".join("-" * width for width in widths), *(line(row) for row in rows)))
    files = ", ".join(sorted(case["project"]["current_files"]))
    return "\n\n".join((
        f"Case: {case['case_id']} | Checkpoint: {case['checkpoint']['id']}\nGoal: {case['goal']}\nCurrent workspace: {files}",
        "Full visible history (IN = selected for the next prompt; OUT = omitted, not deleted):\n" + table,
        f"Recorded simulated model cost across visible history: ${recorded_cost:.6f} (n/a means usage unknown; not a provider bill)",
        projected_context(case, selection),
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect a checkpoint-safe coding-agent fixture without executing project code")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--checkpoint", default="continuation")
    parser.add_argument("--budget", type=int, help="History budget in approximate regex units, not model tokens")
    parser.add_argument("--strategy", choices=("chronological_suffix", "feature_attempt_hierarchy", "last_error_latest_user"), default="chronological_suffix")
    parser.add_argument("--view", choices=("table", "prompt", "json", "hierarchy"), default="table", help="Display history, projected context, machine-readable selection, or inferred hierarchy")
    parser.add_argument("--materialize", type=Path, help="An existing, empty directory to receive the sample project")
    args = parser.parse_args(argv)
    case = load_public_case(args.fixture, args.checkpoint)
    if args.strategy == "feature_attempt_hierarchy":
        from hierarchy import select_feature_context
        selection = select_feature_context(case, budget=args.budget)
    elif args.strategy == "last_error_latest_user":
        selection = select_last_error_latest_user(case, budget=args.budget)
    else:
        selection = select_chronological_history(case, budget=args.budget)
    if args.materialize is not None:
        materialize_workspace(case, args.materialize)
    if args.view == "hierarchy":
        from hierarchy import build_hierarchy, render_hierarchy
        print(render_hierarchy(case, build_hierarchy(case)))
    elif args.view == "json":
        print(json.dumps({"case_id": case["case_id"], "goal": case["goal"], "checkpoint": case["checkpoint"]["id"], **selection}, indent=2))
    elif args.view == "prompt":
        print(projected_context(case, selection))
    else:
        print(render_table(case, selection))


if __name__ == "__main__":
    main()
