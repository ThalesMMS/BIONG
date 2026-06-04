from __future__ import annotations

from .nn_internal import affordance_geometry as _affordance_geometry
from .nn_internal import affordance_position as _affordance_position
from .nn_internal import deep_arbitration as _deep_arbitration
from .nn_internal import motor as _motor
from .nn_internal import option_controller as _option_controller
from .nn_internal import proposal as _proposal
from .nn_internal import recurrent_monolithic as _recurrent_monolithic
from .nn_internal import shared as _shared

_EXPORT_MODULES = (
    _shared,
    _proposal,
    _motor,
    _recurrent_monolithic,
    _option_controller,
    _affordance_geometry,
    _affordance_position,
    _deep_arbitration,
)

_namespace = {}
for _module in _EXPORT_MODULES:
    _names = getattr(_module, "__all__", None)
    if _names is None:
        _names = [name for name in vars(_module) if not name.startswith("_")]
    _namespace.update({_name: getattr(_module, _name) for _name in _names})

globals().update(_namespace)
__all__ = [name for name in _namespace if not name.startswith("_")]
