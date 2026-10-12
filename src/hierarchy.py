from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple, Any

try:
    from .models import Event, EventKind, AttemptNode, HierarchyTree
except (ImportError, ValueError):
    from models import Event, EventKind, AttemptNode, HierarchyTree


class HierarchyBuilder:
    """
    Implements BuildHierarchy(H) to transform a chronological event trace H
    into a structured feature-attempt tree G = (V, E).
    """
    ISSUE_PATTERNS = [
        re.compile(r"syntaxerror", re.IGNORECASE),
        re.compile(r"keyerror", re.IGNORECASE),
        re.compile(r"assertionerror", re.IGNORECASE),
        re.compile(r"traceback", re.IGNORECASE),
        re.compile(r"\bFAIL(?:ED|URES)?\b"),
        re.compile(r"\bError:", re.IGNORECASE),
    ]

    RESOLVED_PATTERNS = [
        re.compile(r"py_compile.*(?:ok|success|\b0\b)", re.IGNORECASE),
        re.compile(r"exit code 0", re.IGNORECASE),
        re.compile(r"\bpassed\b", re.IGNORECASE),
        re.compile(r"\bOK\b"),
    ]

    def __init__(self, feature_name: str = "root_feature"):
        self.feature_name = feature_name

    def is_issue(self, event: Event) -> bool:
        """Determines if a tool event represents an error/failure."""
        if event.exit_code is not None and event.exit_code != 0:
            return True
        return any(p.search(event.text) for p in self.ISSUE_PATTERNS)

    def is_resolution(self, event: Event) -> bool:
        """Determines if a tool event resolves previously active issues."""
        if event.exit_code == 0 and ("passed" in event.text.lower() or "ok" in event.text.lower()):
            return True
        return any(p.search(event.text) for p in self.RESOLVED_PATTERNS)

    def build(self, events: List[Event]) -> HierarchyTree:
        tree = HierarchyTree(feature_name=self.feature_name, all_events=events)
        attempts: List[AttemptNode] = []
        current_attempt: Optional[AttemptNode] = None

        def start_new_attempt(trigger_event_id: str) -> AttemptNode:
            nonlocal current_attempt
            if current_attempt is not None:
                current_attempt.is_superseded = True
            att = AttemptNode(attempt_id=f"attempt_{trigger_event_id}")
            attempts.append(att)
            current_attempt = att
            return att

        for ev in events:
            # Side thread routing
            if ev.thread_id is not None and ev.thread_id.lower() in ("side", "side_thread", "detour"):
                if ev.thread_id not in tree.side_threads:
                    tree.side_threads[ev.thread_id] = []
                tree.side_threads[ev.thread_id].append(ev)
                continue

            # User instructions form root anchors
            if ev.kind == EventKind.USER:
                tree.user_instructions.append(ev)
                continue

            # Assistant edits create/bound attempts
            if ev.kind == EventKind.ASSISTANT_EDIT:
                if current_attempt is None:
                    current_attempt = start_new_attempt(ev.id)
                elif current_attempt.open_issues or current_attempt.resolved_issues:
                    current_attempt = start_new_attempt(ev.id)
                current_attempt.edits.append(ev)

            elif ev.kind == EventKind.ASSISTANT_ANALYSIS:
                if current_attempt is None:
                    current_attempt = start_new_attempt(ev.id)
                current_attempt.analyses.append(ev)

            elif ev.kind == EventKind.TOOL_RESULT:
                if current_attempt is None:
                    current_attempt = start_new_attempt(ev.id)

                current_attempt.tool_events.append(ev)

                if self.is_resolution(ev):
                    # Move all currently open issues to resolved
                    current_attempt.resolved_issues.extend(current_attempt.open_issues)
                    current_attempt.open_issues.clear()
                    current_attempt.resolved_issues.append(ev)
                elif self.is_issue(ev):
                    current_attempt.open_issues.append(ev)

        tree.attempts = attempts
        return tree
