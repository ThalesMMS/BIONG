from __future__ import annotations

from . import distillation as _distillation
from . import evidence as _evidence
from . import runner as _runner

_EXPORT_MODULES = (_distillation, _evidence, _runner)


def _export(module: object) -> list[str]:
    return list(getattr(module, "__all__", ()))


__all__ = _export(_distillation) + _export(_evidence) + _export(_runner)


def __getattr__(name: str) -> object:
    for module in _EXPORT_MODULES:
        if name in getattr(module, "__all__", ()):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
