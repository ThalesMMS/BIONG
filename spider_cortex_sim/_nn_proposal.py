from __future__ import annotations

from .nn_internal import proposal as _target

globals().update(
    {
        _name: _value
        for _name, _value in vars(_target).items()
        if _name not in {"annotations"} and not _name.startswith("__")
    }
)
try:
    __all__ = list(_target.__all__)
except AttributeError:
    pass
