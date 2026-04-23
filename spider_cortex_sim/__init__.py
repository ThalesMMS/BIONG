"""Lightweight package exports for the public simulator entrypoints."""

from importlib import import_module
from typing import Any

__all__ = ["ACTIONS", "SpiderBrain", "SpiderSimulation", "SpiderWorld"]

_EXPORTS = {
    "SpiderBrain": (".agent", "SpiderBrain"),
    "SpiderSimulation": (".simulation", "SpiderSimulation"),
    "SpiderWorld": (".world", "SpiderWorld"),
    "ACTIONS": (".world", "ACTIONS"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
