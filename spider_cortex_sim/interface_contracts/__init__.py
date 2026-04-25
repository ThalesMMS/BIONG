from __future__ import annotations

from . import actions as _actions
from . import module_specs as _module_specs
from . import observations as _observations
from . import registry as _registry
from . import variants as _variants


def _export(module: object) -> list[str]:
    """
    Re-export the public names listed by a submodule into this package's namespace.
    
    Parameters:
        module (object): Submodule whose `__all__` (if present) lists names to re-export from this package.
    
    Returns:
        list[str]: The list of names that were copied into this module's namespace.
    """
    names = list(getattr(module, "__all__", ()))
    globals().update({name: getattr(module, name) for name in names})
    return names


__all__ = (  # noqa: PLE0605
    _export(_actions)
    + _export(_observations)
    + _export(_module_specs)
    + _export(_variants)
    + _export(_registry)
)
