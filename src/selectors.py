from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set

try:
    from .models import Event, EventKind, AttemptNode, HierarchyTree
    from .tokenizers import BaseTokenizer, RegexProxyTokenizer
    from .hierarchy import HierarchyBuilder
except (ImportError, ValueError):
    from models import Event, EventKind, AttemptNode, HierarchyTree
    from tokenizers import BaseTokenizer, RegexProxyTokenizer
    from hierarchy import HierarchyBuilder


@dataclass
class SelectionResult:
    selected_events: List[Event]
    total_units: int
    budget: int
    discarded_units: int
    passed_gate: bool = True
    error_message: Optional[str] = None

    @property
    def selected_ids(self) -> List[str]:
        return [e.id for e in self.selected_events]


class BaseSelector(ABC):
    @abstractmethod
    def select(
        self,
        events: List[Event],
        budget: int,
        tokenizer: Optional[BaseTokenizer] = None
    ) -> SelectionResult:
        pass


class FeatureAttemptSelector(BaseSelector):
    """
    Implements Algorithm 1 (Feature-Attempt Selection) from Kotari (2026).
    """
    def __init__(self, builder: Optional[HierarchyBuilder] = None, fail_closed: bool = True):
        self.builder = builder or HierarchyBuilder()
        self.fail_closed = fail_closed

    def select(
        self,
        events: List[Event],
        budget: int,
        tokenizer: Optional[BaseTokenizer] = None
    ) -> SelectionResult:
        tok = tokenizer or RegexProxyTokenizer()
        tree = self.builder.build(events)

        unit_map = {e.id: tok.count(e.text) for e in events}
        total_trace_units = sum(unit_map.values())

        C: List[Event] = []
        c_ids: Set[str] = set()
        current_units = 0

        latest_user: Optional[Event] = tree.user_instructions[-1] if tree.user_instructions else None

        # 1. Append open issues (newest to oldest across attempts)
        kept_issue_attempts: List[AttemptNode] = []
        for issue_event, attempt in tree.all_open_issues:
            u_cost = unit_map.get(issue_event.id, 0)
            if current_units + u_cost <= budget:
                C.append(issue_event)
                c_ids.add(issue_event.id)
                current_units += u_cost
                kept_issue_attempts.append(attempt)

        # 2. Append latest user instruction
        if latest_user and latest_user.id not in c_ids:
            u_cost = unit_map.get(latest_user.id, 0)
            if current_units + u_cost <= budget:
                C.append(latest_user)
                c_ids.add(latest_user.id)
                current_units += u_cost

        # 3. Check for empty candidate set (Algorithm 1, lines 9-11)
        if not C:
            if self.fail_closed:
                return SelectionResult(
                    selected_events=[],
                    total_units=0,
                    budget=budget,
                    discarded_units=total_trace_units,
                    passed_gate=False,
                    error_message="fail: no active feature context"
                )
            else:
                # Online fallback: preserve root prompt and recent history
                for ev in reversed(events):
                    if ev.thread_id is None or ev.thread_id.lower() not in ("side", "side_thread"):
                        u_cost = unit_map.get(ev.id, 0)
                        if current_units + u_cost <= budget:
                            C.append(ev)
                            c_ids.add(ev.id)
                            current_units += u_cost

        # 4. Append edits that caused kept open issues
        for attempt in kept_issue_attempts:
            for edit in attempt.edits:
                if edit.id not in c_ids:
                    u_cost = unit_map.get(edit.id, 0)
                    if current_units + u_cost <= budget:
                        C.append(edit)
                        c_ids.add(edit.id)
                        current_units += u_cost

        # 5. Append earlier user instructions (newest to oldest)
        earlier_users = tree.user_instructions[:-1] if tree.user_instructions else []
        for u_ev in reversed(earlier_users):
            if u_ev.id not in c_ids:
                u_cost = unit_map.get(u_ev.id, 0)
                if current_units + u_cost <= budget:
                    C.append(u_ev)
                    c_ids.add(u_ev.id)
                    current_units += u_cost

        # 6. Restore original chronological trace order
        turn_map = {e.id: (e.turn, idx) for idx, e in enumerate(events)}
        C.sort(key=lambda x: turn_map.get(x.id, (0, 0)))

        return SelectionResult(
            selected_events=C,
            total_units=current_units,
            budget=budget,
            discarded_units=total_trace_units - current_units,
            passed_gate=True
        )


class SuffixWindowSelector(BaseSelector):
    """Chronological sliding window baseline: selects most recent events up to budget."""
    def select(
        self,
        events: List[Event],
        budget: int,
        tokenizer: Optional[BaseTokenizer] = None
    ) -> SelectionResult:
        tok = tokenizer or RegexProxyTokenizer()
        unit_map = {e.id: tok.count(e.text) for e in events}
        total_trace_units = sum(unit_map.values())

        selected: List[Event] = []
        current_units = 0

        for ev in reversed(events):
            cost = unit_map.get(ev.id, 0)
            if current_units + cost <= budget:
                selected.append(ev)
                current_units += cost
            else:
                break

        selected.reverse()
        return SelectionResult(
            selected_events=selected,
            total_units=current_units,
            budget=budget,
            discarded_units=total_trace_units - current_units
        )


class LatestErrorSelector(BaseSelector):
    """
    Heuristic baseline: picks the single most recent error and its immediate preceding edit,
    along with root user instructions up to budget.
    """
    def __init__(self, builder: Optional[HierarchyBuilder] = None):
        self.builder = builder or HierarchyBuilder()

    def select(
        self,
        events: List[Event],
        budget: int,
        tokenizer: Optional[BaseTokenizer] = None
    ) -> SelectionResult:
        tok = tokenizer or RegexProxyTokenizer()
        unit_map = {e.id: tok.count(e.text) for e in events}
        total_trace_units = sum(unit_map.values())

        latest_error: Optional[Event] = None
        for ev in reversed(events):
            if ev.kind == EventKind.TOOL_RESULT and self.builder.is_issue(ev):
                latest_error = ev
                break

        if not latest_error:
            return SuffixWindowSelector().select(events, budget, tokenizer)

        preceding_edit: Optional[Event] = None
        for ev in reversed(events):
            if ev.turn < latest_error.turn and ev.kind == EventKind.ASSISTANT_EDIT:
                preceding_edit = ev
                break

        selected: List[Event] = []
        current_units = 0

        for candidate in [latest_error, preceding_edit]:
            if candidate is not None:
                cost = unit_map.get(candidate.id, 0)
                if current_units + cost <= budget:
                    selected.append(candidate)
                    current_units += cost

        # Fill remaining budget with user instructions
        for ev in events:
            if ev.kind == EventKind.USER and ev not in selected:
                cost = unit_map.get(ev.id, 0)
                if current_units + cost <= budget:
                    selected.append(ev)
                    current_units += cost

        turn_map = {e.id: (e.turn, idx) for idx, e in enumerate(events)}
        selected.sort(key=lambda x: turn_map.get(x.id, (0, 0)))

        return SelectionResult(
            selected_events=selected,
            total_units=current_units,
            budget=budget,
            discarded_units=total_trace_units - current_units
        )
