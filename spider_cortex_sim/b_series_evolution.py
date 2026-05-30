from __future__ import annotations

from . import _b_series_evolution_constants as __b_series_evolution_constants
from . import _b_series_evolution_checkpoint_paths as __b_series_evolution_checkpoint_paths
from . import _b_series_evolution_config_builders as __b_series_evolution_config_builders
from . import _b_series_evolution_gates_b1_b6 as __b_series_evolution_gates_b1_b6
from . import _b_series_evolution_gates_b7_b19 as __b_series_evolution_gates_b7_b19
from . import _b_series_evolution_gates_b20_b30 as __b_series_evolution_gates_b20_b30
from . import _b_series_evolution_gates_b31_b40 as __b_series_evolution_gates_b31_b40
from . import _b_series_evolution_gates_b41_b50 as __b_series_evolution_gates_b41_b50
from . import _b_series_evolution_gates_b51_b60 as __b_series_evolution_gates_b51_b60
from . import _b_series_evolution_gates_b61_requires as __b_series_evolution_gates_b61_requires
from . import _b_series_evolution_requires_sequences_b1_b5 as __b_series_evolution_requires_sequences_b1_b5
from . import _b_series_evolution_sequences_b5_b7 as __b_series_evolution_sequences_b5_b7
from . import _b_series_evolution_sequences_b7_b10 as __b_series_evolution_sequences_b7_b10
from . import _b_series_evolution_sequences_b10_b14 as __b_series_evolution_sequences_b10_b14
from . import _b_series_evolution_sequences_b14_b18 as __b_series_evolution_sequences_b14_b18
from . import _b_series_evolution_sequences_b18_b22 as __b_series_evolution_sequences_b18_b22
from . import _b_series_evolution_sequences_b22_b26 as __b_series_evolution_sequences_b22_b26
from . import _b_series_evolution_sequences_b26_b30 as __b_series_evolution_sequences_b26_b30
from . import _b_series_evolution_sequences_b30_b34 as __b_series_evolution_sequences_b30_b34
from . import _b_series_evolution_sequences_b34_b38 as __b_series_evolution_sequences_b34_b38
from . import _b_series_evolution_sequences_b38_b42 as __b_series_evolution_sequences_b38_b42
from . import _b_series_evolution_sequences_b42_b46 as __b_series_evolution_sequences_b42_b46
from . import _b_series_evolution_sequences_b46_b50 as __b_series_evolution_sequences_b46_b50
from . import _b_series_evolution_sequences_b50_b54 as __b_series_evolution_sequences_b50_b54
from . import _b_series_evolution_sequences_b54_b58 as __b_series_evolution_sequences_b54_b58
from . import _b_series_evolution_sequences_b58_b62 as __b_series_evolution_sequences_b58_b62
from . import _b_series_evolution_sequence_b62 as __b_series_evolution_sequence_b62
from . import _b_series_evolution_cli as __b_series_evolution_cli

_CANONICAL_MODULE = "spider_cortex_sim.b_series_evolution"
_PART_MODULES = (
    __b_series_evolution_constants,
    __b_series_evolution_checkpoint_paths,
    __b_series_evolution_config_builders,
    __b_series_evolution_gates_b1_b6,
    __b_series_evolution_gates_b7_b19,
    __b_series_evolution_gates_b20_b30,
    __b_series_evolution_gates_b31_b40,
    __b_series_evolution_gates_b41_b50,
    __b_series_evolution_gates_b51_b60,
    __b_series_evolution_gates_b61_requires,
    __b_series_evolution_requires_sequences_b1_b5,
    __b_series_evolution_sequences_b5_b7,
    __b_series_evolution_sequences_b7_b10,
    __b_series_evolution_sequences_b10_b14,
    __b_series_evolution_sequences_b14_b18,
    __b_series_evolution_sequences_b18_b22,
    __b_series_evolution_sequences_b22_b26,
    __b_series_evolution_sequences_b26_b30,
    __b_series_evolution_sequences_b30_b34,
    __b_series_evolution_sequences_b34_b38,
    __b_series_evolution_sequences_b38_b42,
    __b_series_evolution_sequences_b42_b46,
    __b_series_evolution_sequences_b46_b50,
    __b_series_evolution_sequences_b50_b54,
    __b_series_evolution_sequences_b54_b58,
    __b_series_evolution_sequences_b58_b62,
    __b_series_evolution_sequence_b62,
    __b_series_evolution_cli,
)

_namespace = {}
for _module in _PART_MODULES:
    _namespace.update(
        {
            _name: _value
            for _name, _value in vars(_module).items()
            if _name not in {"annotations"} and not _name.startswith("__")
        }
    )

for _name, _value in _namespace.items():
    if getattr(_value, "__module__", None) in {_module.__name__ for _module in _PART_MODULES}:
        try:
            _value.__module__ = _CANONICAL_MODULE
        except (AttributeError, TypeError):
            pass

globals().update(_namespace)
__all__ = [name for name in _namespace if not name.startswith("_")]

if __name__ == "__main__":
    raise SystemExit(main())
