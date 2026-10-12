"""
swebench_adapter.py
SWE-bench Runtime Adapter for Feature-Attempt Hierarchy (Kotari, 2026).

Provides seamless hooks for SWE-agent, OpenHands, and custom SWE-bench ReAct loops:
1. SWEBenchTrajectoryConverter: Maps raw message dictionaries into Event streams.
2. SWEBenchContextManager: Manages online context selection via Algorithm 1.
3. SWEBenchCostTracker: Computes real-world Token Economics FinOps metrics per task.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple

try:
    from .models import Event, EventKind, TokenUsage
    from .tokenizers import BaseTokenizer, RegexProxyTokenizer, TiktokenTokenizer
    from .hierarchy import HierarchyBuilder
    from .selectors import FeatureAttemptSelector, SelectionResult
    from .ledger import CostLedger, CostLedgerReport
except (ImportError, ValueError):
    from models import Event, EventKind, TokenUsage
    from tokenizers import BaseTokenizer, RegexProxyTokenizer, TiktokenTokenizer
    from hierarchy import HierarchyBuilder
    from selectors import FeatureAttemptSelector, SelectionResult
    from ledger import CostLedger, CostLedgerReport


class SWEBenchTrajectoryConverter:
    """
    Converts raw SWE-bench agent message histories (e.g., from SWE-agent or OpenHands)
    into strongly-typed Event objects for hierarchy construction.
    """
    TEST_CMD_PATTERNS = [
        re.compile(r"\bpytest\b", re.IGNORECASE),
        re.compile(r"\bpython\s+-m\s+unittest\b", re.IGNORECASE),
        re.compile(r"\btox\b", re.IGNORECASE),
        re.compile(r"\b(?:python3?|./runtests\.py|manage\.py\s+test)\b", re.IGNORECASE),
    ]

    EDIT_CMD_PATTERNS = [
        re.compile(r"\b(?:edit|str_replace_editor|create_file|patch)\b", re.IGNORECASE),
        re.compile(r"\bgit\s+(?:apply|checkout|commit)\b", re.IGNORECASE),
        re.compile(r">\s*[\w\.\-/]+\.(?:py|js|rs|cpp|h|go|java|txt|md)\b"),
        re.compile(r"\b(?:sed|awk|echo|cat)\b.*>", re.IGNORECASE),
    ]

    def __init__(self):
        pass

    def is_test_command(self, cmd: Optional[str]) -> bool:
        if not cmd:
            return False
        return any(p.search(cmd) for p in self.TEST_CMD_PATTERNS)

    def is_edit_action(self, action_name: Optional[str], cmd: Optional[str]) -> bool:
        if action_name and any(p.search(action_name) for p in self.EDIT_CMD_PATTERNS):
            return True
        if cmd and any(p.search(cmd) for p in self.EDIT_CMD_PATTERNS):
            return True
        return False

    def message_to_event(self, msg: Dict[str, Any], turn_idx: int) -> Event:
        """
        Converts a single message dict (e.g. {"role": "...", "content": "..."})
        to an Event instance.
        """
        role = msg.get("role", "system").lower()
        content = msg.get("content", "") or ""
        msg_id = msg.get("id", f"e_{turn_idx:03d}")
        thread_id = msg.get("thread_id", None)
        cmd = msg.get("command") or msg.get("action_cmd")
        action = msg.get("action")
        exit_code = msg.get("exit_code")

        # Extract token usage if provided by API response
        usage_data = msg.get("usage")
        usage = None
        if usage_data:
            usage = TokenUsage(
                n_in=usage_data.get("prompt_tokens", 0) or usage_data.get("n_in", 0),
                n_out=usage_data.get("completion_tokens", 0) or usage_data.get("n_out", 0),
            )

        # Classify EventKind
        if role == "user":
            kind = EventKind.USER
        elif role in ("assistant", "agent"):
            if self.is_edit_action(action, cmd) or "edit" in content.lower()[:30]:
                kind = EventKind.ASSISTANT_EDIT
            else:
                kind = EventKind.ASSISTANT_ANALYSIS
        elif role in ("tool", "observation", "environment"):
            kind = EventKind.TOOL_RESULT
        else:
            kind = EventKind.SYSTEM

        return Event(
            id=msg_id,
            turn=turn_idx,
            kind=kind,
            text=content,
            thread_id=thread_id,
            usage=usage,
            exit_code=exit_code,
            command=cmd,
            metadata={
                "role": role,
                "action": action,
                "raw_metadata": msg.get("metadata", {}),
            }
        )

    def history_to_events(self, messages: List[Dict[str, Any]]) -> List[Event]:
        """Converts an entire chronological message trajectory to Event list."""
        return [self.message_to_event(msg, idx + 1) for idx, msg in enumerate(messages)]


class SWEBenchContextManager:
    """
    Manages online context pruning for SWE-bench agents before LLM inference.
    """
    def __init__(
        self,
        token_budget: int = 16000,
        tokenizer: Optional[BaseTokenizer] = None,
        fail_closed: bool = False,  # Online agents default to resilient fallback
    ):
        self.token_budget = token_budget
        self.tokenizer = tokenizer or TiktokenTokenizer(model_name="gpt-4o")
        self.converter = SWEBenchTrajectoryConverter()
        self.selector = FeatureAttemptSelector(fail_closed=fail_closed)

    def select_prompt_messages(
        self,
        messages: List[Dict[str, Any]],
        budget: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Takes the raw agent history messages, runs Feature-Attempt Selection,
        and returns the pruned list of messages formatted for LLM completion API.
        """
        b = budget if budget is not None else self.token_budget
        events = self.converter.history_to_events(messages)
        res: SelectionResult = self.selector.select(events, budget=b, tokenizer=self.tokenizer)

        # Map selected events back to message format
        selected_ids = {e.id for e in res.selected_events}
        pruned_messages: List[Dict[str, Any]] = []

        for idx, msg in enumerate(messages):
            ev_id = msg.get("id", f"e_{idx + 1:03d}")
            if ev_id in selected_ids:
                pruned_messages.append(msg)

        # Fallback safeguard: ensure at least the initial user prompt is retained
        if not pruned_messages and messages:
            pruned_messages.append(messages[0])

        return pruned_messages


class SWEBenchCostTracker:
    """
    Computes enterprise Token Economics metrics for a SWE-bench task run.
    """
    def __init__(
        self,
        p_in: float = 2.0,   # Dollars per 1M input tokens
        p_out: float = 8.0,  # Dollars per 1M output tokens
    ):
        self.converter = SWEBenchTrajectoryConverter()
        self.ledger = CostLedger(p_in=p_in, p_out=p_out)

    def analyze_trajectory(
        self,
        messages: List[Dict[str, Any]],
        p_in: Optional[float] = None,
        p_out: Optional[float] = None
    ) -> CostLedgerReport:
        events = self.converter.history_to_events(messages)
        return self.ledger.compute(events, p_in=p_in, p_out=p_out)
