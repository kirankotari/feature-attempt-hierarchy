"""
feature_attempt_hierarchy
A hierarchical context management and token economics library for coding agents.
"""

from .models import (
    Event,
    EventKind,
    AttemptNode,
    HierarchyTree,
    TokenUsage,
)
from .tokenizers import (
    BaseTokenizer,
    RegexProxyTokenizer,
    WhitespaceTokenizer,
    CharacterTokenizer,
    TiktokenTokenizer,
)
from .hierarchy import HierarchyBuilder
from .context_selectors import (
    BaseSelector,
    FeatureAttemptSelector,
    SuffixWindowSelector,
    LatestErrorSelector,
    SelectionResult,
)
from .ledger import (
    CostLedger,
    CostLedgerReport,
)
from .evaluator import (
    ContextGateChecker,
    BenchmarkHarness,
)
from .swebench_adapter import (
    SWEBenchTrajectoryConverter,
    SWEBenchContextManager,
    SWEBenchCostTracker,
)

__all__ = [
    "Event",
    "EventKind",
    "AttemptNode",
    "HierarchyTree",
    "TokenUsage",
    "BaseTokenizer",
    "RegexProxyTokenizer",
    "WhitespaceTokenizer",
    "CharacterTokenizer",
    "TiktokenTokenizer",
    "HierarchyBuilder",
    "BaseSelector",
    "FeatureAttemptSelector",
    "SuffixWindowSelector",
    "LatestErrorSelector",
    "SelectionResult",
    "CostLedger",
    "CostLedgerReport",
    "ContextGateChecker",
    "BenchmarkHarness",
    "SWEBenchTrajectoryConverter",
    "SWEBenchContextManager",
    "SWEBenchCostTracker",
]
