from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

try:
    from .models import Event, EventKind, TokenUsage
    from .hierarchy import HierarchyBuilder
except (ImportError, ValueError):
    from models import Event, EventKind, TokenUsage
    from hierarchy import HierarchyBuilder


@dataclass
class CostLedgerReport:
    """Formal cost breakdown matching the Token Economics ledger in Kotari (2026)."""
    feature_cost: float
    rework_cost: float
    side_cost: float
    total_cost: float
    first_issue_turn: Optional[int]
    assistant_calls_count: int

    @property
    def rework_ratio(self) -> float:
        """Portion of feature spend that was rework (after the first failure)."""
        if self.feature_cost <= 0.0:
            return 0.0
        return self.rework_cost / self.feature_cost


class CostLedger:
    """
    Computes Token Economics cost attribution:
    - Feature Cost: Model calls advancing the core feature.
    - Rework Cost: Model calls occurring after the first detected error.
    - Side Cost: Model calls on detached side threads/detours.
    """
    def __init__(self, builder: Optional[HierarchyBuilder] = None, p_in: float = 2.0, p_out: float = 8.0):
        self.builder = builder or HierarchyBuilder()
        self.p_in = p_in
        self.p_out = p_out

    def compute(
        self,
        events: List[Event],
        p_in: Optional[float] = None,
        p_out: Optional[float] = None
    ) -> CostLedgerReport:
        pin = p_in if p_in is not None else self.p_in
        pout = p_out if p_out is not None else self.p_out

        # Identify first issue turn
        first_issue_turn: Optional[int] = None
        for ev in events:
            if ev.kind == EventKind.TOOL_RESULT and self.builder.is_issue(ev):
                first_issue_turn = ev.turn
                break

        feature_cost = 0.0
        rework_cost = 0.0
        side_cost = 0.0
        assistant_calls = 0

        for ev in events:
            if ev.kind in (EventKind.ASSISTANT_EDIT, EventKind.ASSISTANT_ANALYSIS):
                if ev.usage is not None:
                    call_cost = ev.usage.compute_cost(pin, pout)
                    assistant_calls += 1
                    is_side = ev.thread_id is not None and ev.thread_id.lower() in ("side", "side_thread", "detour")

                    if is_side:
                        side_cost += call_cost
                    else:
                        feature_cost += call_cost
                        if first_issue_turn is not None and ev.turn > first_issue_turn:
                            rework_cost += call_cost

        return CostLedgerReport(
            feature_cost=round(feature_cost, 6),
            rework_cost=round(rework_cost, 6),
            side_cost=round(side_cost, 6),
            total_cost=round(feature_cost + side_cost, 6),
            first_issue_turn=first_issue_turn,
            assistant_calls_count=assistant_calls
        )
