import re

from harness import select_chronological_history, select_event_ids
from hierarchy import build_hierarchy, select_feature_context


STOPWORDS = {"the", "this", "that", "with", "from", "into", "still", "must", "have", "does", "not", "for", "and", "its", "are", "but"}
FAILED = re.compile(r"traceback|syntaxerror|keyerror|exit=1|expected:|error|failed|\bfail\b", re.IGNORECASE)


def _feature_events(case):
    return [event for event in case["events"] if event.get("thread_id", "feature") == "feature"]


def _fit(case, event_ids, budget):
    try:
        return select_event_ids(case, event_ids, budget=budget)
    except ValueError as error:
        if "exceeds the history budget" not in str(error):
            raise
        return None


def select_thread_path_proxy(case, budget=None):
    """MemoTree-style proxy: keep the active attempt path, not abandoned branches.

    MemoTree walks an explicit parent chain stored by its UI. These fixtures have
    no parent ids, so a superseded attempt stands in for an inactive branch.
    """
    graph = build_hierarchy(case)
    abandoned = {attempt_id for attempt_id, attempt in graph["attempts"].items() if attempt["status"] == "superseded"}
    kept = [event for event in _feature_events(case) if graph["parents"].get(event["id"]) not in abandoned]
    selection = select_chronological_history(dict(case, events=kept), budget=budget)
    selection["strategy"] = "thread_path_suffix_proxy"
    return selection


def select_dependency_lexical_proxy(case, budget=None, skip_failed=False):
    """ContextWeaver-style proxy: link the latest turn to one earlier observation.

    Overlap stands in for the paper's LLM parent ranker. skip_failed applies the
    paper's rule that failed or superseded nodes are not parent candidates. Neither
    mode calls that model, summarizes dependencies, or runs SWE-bench.
    """
    events = _feature_events(case)
    latest_user = next((event for event in reversed(events) if event["kind"] == "user"), None)
    if latest_user is None:
        raise ValueError("No feature user turn")
    terms = set(re.findall(r"[a-z0-9_]+", latest_user["text"].lower())) - STOPWORDS
    candidates = []
    for index, event in enumerate(events):
        if event["kind"] != "tool_result" or event["id"] == latest_user["id"]:
            continue
        if skip_failed and FAILED.search(event["text"]):
            continue
        candidates.append((index, event))
    if not candidates:
        selection = select_event_ids(case, [latest_user["id"]], budget=budget)
    else:
        def rank(candidate):
            index, event = candidate
            words = set(re.findall(r"[a-z0-9_]+", (event["text"] + " " + event.get("tool", "")).lower()))
            return (len(terms & words), index)
        result = max(candidates, key=rank)[1]
        prior_edit = None
        for event in events:
            if event["id"] == result["id"]:
                break
            if event["kind"] == "assistant_edit":
                prior_edit = event["id"]
        event_ids = [result["id"], latest_user["id"]]
        if prior_edit is not None:
            event_ids.append(prior_edit)
        selection = _fit(case, event_ids, budget) or select_event_ids(case, [result["id"], latest_user["id"]], budget=budget)
    selection["strategy"] = "contextweaver_validation_proxy" if skip_failed else "dependency_lexical_proxy"
    return selection


def select_combined_proxy(case, budget=None):
    """Keep the hierarchy selection, then any lexical dependency that still fits."""
    chosen = list(select_feature_context(case, budget=budget)["event_ids"])
    for event_id in select_dependency_lexical_proxy(case, budget=budget)["event_ids"]:
        if event_id in chosen:
            continue
        if _fit(case, [*chosen, event_id], budget) is not None:
            chosen.append(event_id)
    selection = select_event_ids(case, chosen, budget=budget)
    selection["strategy"] = "combined_feature_proxy"
    return selection
