from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any

try:
    from .models import Event
    from .tokenizers import BaseTokenizer, RegexProxyTokenizer
    from .context_selectors import BaseSelector, SelectionResult
except (ImportError, ValueError):
    from models import Event
    from tokenizers import BaseTokenizer, RegexProxyTokenizer
    from context_selectors import BaseSelector, SelectionResult


@dataclass
class GateCheckResult:
    passed: bool
    missing_required: Set[str]
    included_forbidden: Set[str]
    total_units: int
    budget: int
    within_budget: bool


class ContextGateChecker:
    """Checks whether context selection passes the required and forbidden ID assertions."""
    @staticmethod
    def evaluate(
        selection: SelectionResult,
        required_ids: Set[str],
        forbidden_ids: Set[str],
    ) -> GateCheckResult:
        selected_ids = set(selection.selected_ids)
        missing = required_ids - selected_ids
        forbidden = forbidden_ids.intersection(selected_ids)
        within_budget = selection.total_units <= selection.budget

        passed = (len(missing) == 0) and (len(forbidden) == 0) and within_budget and selection.passed_gate

        return GateCheckResult(
            passed=passed,
            missing_required=missing,
            included_forbidden=forbidden,
            total_units=selection.total_units,
            budget=selection.budget,
            within_budget=within_budget,
        )


class BenchmarkHarness:
    """Evaluates multiple selectors across scenario traces and budgets."""
    def __init__(self, tokenizer: Optional[BaseTokenizer] = None):
        self.tokenizer = tokenizer or RegexProxyTokenizer()

    def run_scenario(
        self,
        name: str,
        events: List[Event],
        budget: int,
        required_ids: Set[str],
        forbidden_ids: Set[str],
        selectors: Dict[str, BaseSelector],
    ) -> Dict[str, GateCheckResult]:
        results = {}
        for s_name, selector in selectors.items():
            sel_res = selector.select(events, budget, self.tokenizer)
            gate_res = ContextGateChecker.evaluate(sel_res, required_ids, forbidden_ids)
            results[s_name] = gate_res
        return results
