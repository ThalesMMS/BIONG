from __future__ import annotations

from .budget_profiles import resolve_budget
from .learning_evidence import resolve_learning_evidence_conditions
from . import learning_comparison as _learning_comparison
from .learning_comparison import *
from .learning_comparison import runner as _runner

try:
    del compare_learning_evidence
except NameError:
    pass


def __getattr__(name: str) -> object:
    if name == "compare_learning_evidence":
        return _runner.compare_learning_evidence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(getattr(_learning_comparison, "__all__", ()))
