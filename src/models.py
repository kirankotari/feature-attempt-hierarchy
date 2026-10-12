from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any


class EventKind(str, Enum):
    USER = "user"
    ASSISTANT_EDIT = "assistant_edit"
    ASSISTANT_ANALYSIS = "assistant_analysis"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption metadata for LLM calls."""
    n_in: int
    n_out: int

    def compute_cost(self, p_in: float = 2.0, p_out: float = 8.0) -> float:
        """
        Computes cost in dollars based on prices per 1M tokens.
        Cost = (n_in * p_in + n_out * p_out) / 10^6
        """
        return (self.n_in * p_in + self.n_out * p_out) / 1_000_000.0


@dataclass
class Event:
    """A single interaction event in the agent execution trace H."""
    id: str
    turn: int
    kind: EventKind
    text: str
    thread_id: Optional[str] = None
    usage: Optional[TokenUsage] = None
    exit_code: Optional[int] = None
    command: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptNode:
    """Represents a development attempt cycle bounded by edits or test runs."""
    attempt_id: str
    edits: List[Event] = field(default_factory=list)
    open_issues: List[Event] = field(default_factory=list)
    resolved_issues: List[Event] = field(default_factory=list)
    analyses: List[Event] = field(default_factory=list)
    tool_events: List[Event] = field(default_factory=list)
    is_superseded: bool = False


@dataclass
class HierarchyTree:
    """
    Hierarchical context tree G = (V, E) built from trace H.
    Roots: User instructions and active feature node.
    Children: Attempts, edits, issues (open vs. resolved), and side threads.
    """
    feature_name: str
    user_instructions: List[Event] = field(default_factory=list)
    attempts: List[AttemptNode] = field(default_factory=list)
    side_threads: Dict[str, List[Event]] = field(default_factory=dict)
    all_events: List[Event] = field(default_factory=list)

    @property
    def all_open_issues(self) -> List[Tuple[Event, AttemptNode]]:
        """Returns all open issues ordered newest to oldest across attempts."""
        issues = []
        for att in reversed(self.attempts):
            for issue in reversed(att.open_issues):
                issues.append((issue, att))
        return issues
