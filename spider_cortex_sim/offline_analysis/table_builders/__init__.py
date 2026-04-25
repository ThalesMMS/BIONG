from __future__ import annotations

from . import aggregate_benchmark as _aggregate_benchmark
from . import capacity as _capacity
from . import claims as _claims
from . import credit as _credit
from . import diagnostics as _diagnostics
from . import effect_sizes as _effect_sizes
from . import reward_profiles as _reward_profiles
from . import unified_ladder as _unified_ladder


def _export(module: object) -> list[str]:
    """
    Import the public names listed in a submodule's __all__ into this package's global namespace.
    
    Parameters:
        module (object): Submodule to re-export names from; its `__all__` attribute (if present) is used to determine which names to import.
    
    Returns:
        list[str]: The list of names that were exported from the given module.
    """
    names = list(getattr(module, "__all__", ()))
    collisions = [
        name
        for name in names
        if name in globals() and globals()[name] is not getattr(module, name)
    ]
    if collisions:
        raise ValueError(
            f"_export({module.__name__}) would overwrite: {sorted(collisions)}."
        )
    globals().update({name: getattr(module, name) for name in names})
    return names


exported_names = list(
    _export(_unified_ladder)
    + _export(_credit)
    + _export(_diagnostics)
    + _export(_capacity)
    + _export(_aggregate_benchmark)
    + _export(_claims)
    + _export(_reward_profiles)
    + _export(_effect_sizes)
)
__all__ = exported_names
