from __future__ import annotations

from . import conflicts as _conflicts
from . import predator as _predator
from . import rest_food as _rest_food


def _export(module: object) -> list[str]:
    """
    Re-export the symbols listed in a module's __all__ into this module's global namespace.
    
    Parameters:
        module (object): The module whose `__all__` will be read. If `__all__` is missing, no names are exported.
    
    Returns:
        list[str]: The list of names read from the module's `__all__` that were injected into this module's globals.
    """
    names = list(getattr(module, "__all__", ()))
    exported: list[str] = []
    for name in names:
        value = getattr(module, name)
        if name in globals():
            if globals()[name] is not value:
                raise ValueError(
                    f"_export({module.__name__}) would overwrite {name!r}."
                )
            continue
        globals()[name] = value
        exported.append(name)
    return exported


__all__ = _export(_rest_food) + _export(_predator) + _export(_conflicts)
