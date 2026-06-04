from __future__ import annotations

from dataclasses import MISSING, Field, fields
from types import MappingProxyType
from typing import Any

from .types import BrainStep


_DIRECT_POLICY_FIRST_FIELD = "b_current_threat_pressure"
_DIRECT_POLICY_LAST_FIELD = "b62_genetic_candidate"

_RUNTIME_CONTEXT_FIELD_NAMES = (
    "b_level",
    "b_effective_level",
    "b_mode",
    "b_parent_level",
    "b_transfer_source_checkpoint",
    "b_transfer_coverage",
)

_RUNTIME_COMPUTED_FIELD_NAMES = (
    "semantic_action",
    "semantic_action_idx",
    "learned_semantic_action",
    "learned_semantic_action_idx",
    "semantic_action_source",
    "semantic_action_reason",
    "semantic_override_count",
    "semantic_logits",
    "semantic_policy",
    "bridge_primitive_action",
    "bridge_reason",
    "blocked_mask",
    "food_delta_used",
    "shelter_delta_used",
)

_EPISODE_TRACE_COMPUTED_FIELD_NAMES = (
    "semantic_action",
    "learned_semantic_action",
    "semantic_action_source",
    "semantic_action_reason",
    "semantic_override_count",
    "semantic_logits",
    "bridge_primitive_action",
    "bridge_reason",
    "blocked_mask",
    "food_delta_used",
    "shelter_delta_used",
    "external_override_count",
)


def _field_default(field: Field[Any]) -> object:
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    if field.default is not MISSING:
        return field.default
    return None


_BRAIN_STEP_FIELD_NAMES = tuple(field.name for field in fields(BrainStep))
_BRAIN_STEP_FIELD_DEFAULTS = MappingProxyType(
    {field.name: _field_default(field) for field in fields(BrainStep)}
)
_DIRECT_POLICY_START = _BRAIN_STEP_FIELD_NAMES.index(_DIRECT_POLICY_FIRST_FIELD)
_DIRECT_POLICY_END = _BRAIN_STEP_FIELD_NAMES.index(_DIRECT_POLICY_LAST_FIELD) + 1

B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES = _BRAIN_STEP_FIELD_NAMES[
    _DIRECT_POLICY_START:_DIRECT_POLICY_END
]
B_SERIES_RUNTIME_FIELD_NAMES = (
    _RUNTIME_CONTEXT_FIELD_NAMES
    + B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES
    + _RUNTIME_COMPUTED_FIELD_NAMES
)
B_SERIES_EPISODE_TRACE_FIELD_NAMES = (
    _RUNTIME_CONTEXT_FIELD_NAMES
    + B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES
    + _EPISODE_TRACE_COMPUTED_FIELD_NAMES
)
B_SERIES_TRACE_FIELD_DEFAULTS = MappingProxyType(
    {
        name: _BRAIN_STEP_FIELD_DEFAULTS[name]
        for name in B_SERIES_RUNTIME_FIELD_NAMES
        if name in _BRAIN_STEP_FIELD_DEFAULTS
    }
)


def b_series_trace_default(name: str) -> object:
    value = B_SERIES_TRACE_FIELD_DEFAULTS[name]
    if hasattr(value, "copy"):
        return value.copy()
    return value


def b_series_direct_policy_payload(source: dict[str, object]) -> dict[str, object]:
    return {
        name: source.get(name, b_series_trace_default(name))
        for name in B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES
    }


def b_series_inactive_runtime_payload() -> dict[str, object]:
    payload = {
        name: b_series_trace_default(name)
        for name in B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES
    }
    payload.update(
        {
            "b_level": -1,
            "b_effective_level": None,
            "b_mode": None,
            "b_parent_level": None,
            "b_transfer_source_checkpoint": None,
            "b_transfer_coverage": None,
            "semantic_action": None,
            "semantic_action_idx": -1,
            "learned_semantic_action": None,
            "learned_semantic_action_idx": -1,
            "semantic_action_source": None,
            "semantic_action_reason": None,
            "semantic_override_count": 0,
            "semantic_logits": b_series_trace_default("semantic_logits"),
            "semantic_policy": b_series_trace_default("semantic_policy"),
            "bridge_primitive_action": None,
            "bridge_reason": None,
            "blocked_mask": {},
            "food_delta_used": 0.0,
            "shelter_delta_used": 0.0,
        }
    )
    return payload


def b_series_episode_trace_value(decision: BrainStep, name: str) -> object:
    if name == "b_level":
        return int(decision.b_level)
    if name == "semantic_override_count":
        return int(decision.semantic_override_count)
    if name == "semantic_logits":
        return decision.semantic_logits.round(6).tolist()
    if name == "blocked_mask":
        return dict(decision.blocked_mask)
    if name == "food_delta_used":
        return round(float(decision.food_delta_used), 6)
    if name == "shelter_delta_used":
        return round(float(decision.shelter_delta_used), 6)
    if name == "external_override_count":
        return int(decision.external_override_count)
    return getattr(decision, name)


def append_registered_b_series_trace_fields(
    item: dict[str, object],
    decision: BrainStep,
) -> None:
    for name in B_SERIES_EPISODE_TRACE_FIELD_NAMES:
        item[name] = b_series_episode_trace_value(decision, name)
