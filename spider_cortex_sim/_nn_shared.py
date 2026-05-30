from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Optional

import numpy as np

from .direct_policy_affordances import (
    AFFORDANCE_GEOMETRY_TARGET_NAMES,
    AFFORDANCE_SHELTER_COLUMN_NAMES,
    AFFORDANCE_SHELTER_POSITION_NAMES,
    AFFORDANCE_SHELTER_ROLE_NAMES,
    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
    DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
    DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
)
from .direct_policy_events import EVENT_TYPE_NAMES, EVENT_TYPE_TO_INDEX
from .direct_policy_options import OPTION_NAMES, OPTION_TO_INDEX
from .interfaces import ACTION_TO_INDEX, MODULE_INTERFACES
from .nn_utils import (
    Array,
    _clip_grad_logits,
    _coerce_state_array,
    _parameter_norm_of,
    _sigmoid,
    _state_scalar,
    _validate_state_dict,
    _weight_scale,
    one_hot,
    softmax,
)


def _module_signal_flat_index(module_name: str, signal_name: str) -> int:
    offset = 0
    for spec in MODULE_INTERFACES:
        if spec.name == module_name:
            try:
                signal_offset = spec.signal_names.index(signal_name)
            except ValueError as exc:
                raise KeyError(
                    f"Unknown signal {signal_name!r} for module {module_name!r}."
                ) from exc
            return offset + signal_offset
        offset += spec.input_dim
    raise KeyError(f"Unknown module {module_name!r}.")


_SLEEP_FATIGUE_IDX = _module_signal_flat_index("sleep_center", "fatigue")
_SLEEP_HUNGER_IDX = _module_signal_flat_index("sleep_center", "hunger")
_SLEEP_ON_SHELTER_IDX = _module_signal_flat_index("sleep_center", "on_shelter")
_SLEEP_NIGHT_IDX = _module_signal_flat_index("sleep_center", "night")
_SLEEP_PHASE_LEVEL_IDX = _module_signal_flat_index(
    "sleep_center", "sleep_phase_level"
)
_SLEEP_REST_STREAK_IDX = _module_signal_flat_index(
    "sleep_center", "rest_streak_norm"
)
_SLEEP_DEBT_IDX = _module_signal_flat_index("sleep_center", "sleep_debt")
_SLEEP_SHELTER_ROLE_LEVEL_IDX = _module_signal_flat_index(
    "sleep_center", "shelter_role_level"
)
_SLEEP_SHELTER_MEMORY_AGE_IDX = _module_signal_flat_index(
    "sleep_center", "shelter_memory_age"
)
_SLEEP_SHELTER_MEMORY_DX_IDX = _module_signal_flat_index(
    "sleep_center", "shelter_memory_dx"
)
_SLEEP_SHELTER_MEMORY_DY_IDX = _module_signal_flat_index(
    "sleep_center", "shelter_memory_dy"
)
_HUNGER_ON_FOOD_IDX = _module_signal_flat_index("hunger_center", "on_food")
_HUNGER_FOOD_VISIBLE_IDX = _module_signal_flat_index("hunger_center", "food_visible")
_HUNGER_FOOD_CERTAINTY_IDX = _module_signal_flat_index("hunger_center", "food_certainty")
_HUNGER_FOOD_DX_IDX = _module_signal_flat_index("hunger_center", "food_dx")
_HUNGER_FOOD_DY_IDX = _module_signal_flat_index("hunger_center", "food_dy")
_HUNGER_FOOD_SMELL_STRENGTH_IDX = _module_signal_flat_index("hunger_center", "food_smell_strength")
_HUNGER_FOOD_SMELL_DX_IDX = _module_signal_flat_index("hunger_center", "food_smell_dx")
_HUNGER_FOOD_SMELL_DY_IDX = _module_signal_flat_index("hunger_center", "food_smell_dy")
_HUNGER_FOOD_MEMORY_DX_IDX = _module_signal_flat_index("hunger_center", "food_memory_dx")
_HUNGER_FOOD_MEMORY_DY_IDX = _module_signal_flat_index("hunger_center", "food_memory_dy")
_HUNGER_FOOD_MEMORY_AGE_IDX = _module_signal_flat_index("hunger_center", "food_memory_age")
_ALERT_PREDATOR_VISIBLE_IDX = _module_signal_flat_index(
    "alert_center", "predator_visible"
)
_ALERT_PREDATOR_CERTAINTY_IDX = _module_signal_flat_index(
    "alert_center", "predator_certainty"
)
_ALERT_PREDATOR_DX_IDX = _module_signal_flat_index("alert_center", "predator_dx")
_ALERT_PREDATOR_DY_IDX = _module_signal_flat_index("alert_center", "predator_dy")
_ALERT_PREDATOR_SMELL_STRENGTH_IDX = _module_signal_flat_index(
    "alert_center", "predator_smell_strength"
)
_ALERT_PREDATOR_MOTION_SALIENCE_IDX = _module_signal_flat_index(
    "alert_center", "predator_motion_salience"
)
_ALERT_VISUAL_PREDATOR_THREAT_IDX = _module_signal_flat_index(
    "alert_center", "visual_predator_threat"
)
_ALERT_OLFACTORY_PREDATOR_THREAT_IDX = _module_signal_flat_index(
    "alert_center", "olfactory_predator_threat"
)
_ALERT_RECENT_PAIN_IDX = _module_signal_flat_index("alert_center", "recent_pain")
_ALERT_RECENT_CONTACT_IDX = _module_signal_flat_index(
    "alert_center", "recent_contact"
)
_ALERT_PREDATOR_TRACE_STRENGTH_IDX = _module_signal_flat_index(
    "alert_center", "predator_trace_strength"
)
_LOCAL_ACTION_TO_POLICY_INDEX = {
    action_name: int(ACTION_TO_INDEX[action_name])
    for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES
}
_POLICY_ORIENTATION_ACTION_INDICES = tuple(
    int(ACTION_TO_INDEX[action_name])
    for action_name in ("ORIENT_UP", "ORIENT_DOWN", "ORIENT_LEFT", "ORIENT_RIGHT")
)
_DEEP_SHELTER_POSITION_INDICES = tuple(
    AFFORDANCE_SHELTER_POSITION_NAMES.index(name)
    for name in ("deep_left", "deep_center", "deep_right")
)
_INSIDE_SHELTER_POSITION_INDICES = tuple(
    AFFORDANCE_SHELTER_POSITION_NAMES.index(name)
    for name in ("inside_left", "inside_center", "inside_right")
)
_ENTRANCE_POSITION_INDICES = tuple(
    AFFORDANCE_SHELTER_POSITION_NAMES.index(name)
    for name in ("entrance_left", "entrance_center", "entrance_right")
)
_OUTSIDE_POSITION_INDEX = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
_GEOMETRY_DEEPEN_INDEX = AFFORDANCE_GEOMETRY_TARGET_NAMES.index("deepen_shelter")
_GEOMETRY_OUTSIDE_INDEX = AFFORDANCE_GEOMETRY_TARGET_NAMES.index("toward_outside")

__all__ = [name for name in globals() if not name.startswith("__")]
