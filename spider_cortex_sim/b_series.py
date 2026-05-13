from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


B_SERIES_POLICY_NAME = "b_series_policy"
B_CURRENT_BRIDGE_EFFECTIVE_LEVEL = "B0-current-simple"
B_CURRENT_BRIDGE_SELECTION_SOURCE = "legacy_direct_controller"
LOCOMOTION_ACTIONS: tuple[str, ...] = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "STAY",
    "ORIENT_UP",
    "ORIENT_DOWN",
    "ORIENT_LEFT",
    "ORIENT_RIGHT",
)
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(LOCOMOTION_ACTIONS)}
B_SEMANTIC_ACTIONS: tuple[str, ...] = (
    "MOVE_TO_FOOD",
    "MOVE_TO_SHELTER",
    "EXPLORE",
    "STAY",
    "EAT",
    "SLEEP",
)
B_SEMANTIC_ACTION_TO_INDEX = {
    name: idx for idx, name in enumerate(B_SEMANTIC_ACTIONS)
}
B_SERIES_MODES: tuple[str, ...] = ("legacy_semantic", "current_bridge")
BRIDGE_MOVE_ACTIONS: tuple[str, ...] = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
)
BRIDGE_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
}
BRIDGE_EXPLORE_ORDER: tuple[str, ...] = (
    "MOVE_RIGHT",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_UP",
)


@dataclass(frozen=True)
class BSeriesBridgeDecision:
    semantic_action: str
    primitive_action: str
    reason: str
    blocked_mask: dict[str, bool]
    food_delta_used: float
    shelter_delta_used: float
    external_override_count: int = 0

    @property
    def primitive_action_idx(self) -> int:
        return int(ACTION_TO_INDEX[self.primitive_action])


def _meta_from_observation(observation: Mapping[str, object]) -> Mapping[str, object]:
    meta = observation.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _submapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def b_series_blocked_mask(
    observation: Mapping[str, object],
) -> dict[str, bool]:
    meta = _meta_from_observation(observation)
    local_affordances = _submapping(meta.get("local_affordances"))
    blocked: dict[str, bool] = {}
    for action_name in LOCOMOTION_ACTIONS:
        if action_name in BRIDGE_MOVE_ACTIONS:
            affordance = _submapping(local_affordances.get(action_name))
            blocked[action_name] = bool(affordance.get("blocked", False))
        else:
            blocked[action_name] = False
    return blocked


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def _movement_candidates(blocked_mask: Mapping[str, bool]) -> list[str]:
    return [
        action_name
        for action_name in BRIDGE_MOVE_ACTIONS
        if not bool(blocked_mask.get(action_name, False))
    ]


def _transition_for(
    meta: Mapping[str, object],
    action_name: str,
) -> Mapping[str, object]:
    transitions = _submapping(meta.get("local_transition_consequences"))
    return _submapping(transitions.get(action_name))


def _geodesic_for(
    meta: Mapping[str, object],
    action_name: str,
) -> Mapping[str, object]:
    geodesics = _submapping(meta.get("local_geodesic_consequences"))
    return _submapping(geodesics.get(action_name))


def _affordance_for(
    meta: Mapping[str, object],
    action_name: str,
) -> Mapping[str, object]:
    affordances = _submapping(meta.get("local_affordances"))
    return _submapping(affordances.get(action_name))


def _target_vector(
    meta: Mapping[str, object],
    target_name: str,
) -> tuple[float, float, str]:
    vision = _submapping(_submapping(meta.get("vision")).get(target_name))
    if (
        _float_value(vision.get("visible"), 0.0) > 0.0
        and _float_value(vision.get("certainty"), 0.0) > 0.0
    ):
        dx = _float_value(vision.get("dx"), 0.0)
        dy = _float_value(vision.get("dy"), 0.0)
        if abs(dx) + abs(dy) >= 0.05:
            return dx, dy, f"{target_name}_vision_vector"

    memory = _submapping(_submapping(meta.get("memory_vectors")).get(target_name))
    if _float_value(memory.get("age"), 1.0) < 1.0:
        dx = _float_value(memory.get("dx"), 0.0)
        dy = _float_value(memory.get("dy"), 0.0)
        if abs(dx) + abs(dy) >= 0.05:
            return dx, dy, f"{target_name}_memory_vector"

    trace = _submapping(_submapping(meta.get("percept_traces")).get(target_name))
    dx = _float_value(trace.get("dx"), 0.0)
    dy = _float_value(trace.get("dy"), 0.0)
    strength = max(
        _float_value(trace.get("strength"), 0.0),
        _float_value(trace.get("freshness"), 0.0),
        _float_value(trace.get("confidence"), 0.0),
    )
    if strength > 0.0 and abs(dx) + abs(dy) >= 0.05:
        return dx, dy, f"{target_name}_trace_vector"

    return 0.0, 0.0, "no_target_vector"


def _predator_pressure(meta: Mapping[str, object]) -> float:
    return max(
        _float_value(meta.get("predator_smell_strength"), 0.0),
        _float_value(meta.get("visual_predator_threat"), 0.0),
        _float_value(meta.get("olfactory_predator_threat"), 0.0),
        _float_value(meta.get("predator_motion_salience"), 0.0),
        1.0 if bool(meta.get("predator_visible", False)) else 0.0,
    )


def _best_exit_action(
    meta: Mapping[str, object],
    candidates: Sequence[str],
) -> tuple[str, float, str]:
    best_exit_action = "STAY"
    best_exit_delta = 0.0
    best_exit_score = -1e9
    predator_pressure = _predator_pressure(meta)
    for action_name in candidates:
        transition = _transition_for(meta, action_name)
        geodesic = _geodesic_for(meta, action_name)
        exit_delta = _float_value(geodesic.get("exit_geodesic_delta"), 0.0)
        predator_delta = _float_value(transition.get("predator_dist_delta"), 0.0)
        score = exit_delta + (1.00 * predator_pressure * predator_delta)
        if predator_pressure >= 0.50 and predator_delta <= -0.5:
            score -= 4.0
        if bool(geodesic.get("next_on_exit_target", False)):
            score += 0.75
        if score > best_exit_score:
            best_exit_score = score
            best_exit_action = action_name
            best_exit_delta = _float_value(transition.get("food_dist_delta"), 0.0)
    if best_exit_score > 0.0:
        return best_exit_action, best_exit_delta, "food_exit_to_outside"
    return "STAY", 0.0, "no_exit_progress"


def _vector_guided_food_action(
    meta: Mapping[str, object],
    candidates: Sequence[str],
) -> tuple[str, float, str] | None:
    target_dx, target_dy, source = _target_vector(meta, "food")
    if source == "no_target_vector":
        return None

    best_action = "STAY"
    best_delta = 0.0
    best_score = -1e9
    current_role = str(meta.get("shelter_role", "outside"))
    predator_pressure = _predator_pressure(meta)
    for action_name in candidates:
        action_dx, action_dy = BRIDGE_ACTION_DELTAS[action_name]
        transition = _transition_for(meta, action_name)
        affordance = _affordance_for(meta, action_name)
        next_role = str(affordance.get("next_role", current_role))
        next_has_food = bool(transition.get("next_cell_has_food", False))
        food_delta = _float_value(transition.get("food_dist_delta"), 0.0)
        predator_delta = _float_value(transition.get("predator_dist_delta"), 0.0)
        score = (
            action_dx * target_dx
            + action_dy * target_dy
            + 0.20 * food_delta
            + (0.60 * predator_pressure * predator_delta)
        )
        if next_has_food:
            score += 4.0
        if predator_pressure >= 0.50 and predator_delta <= -0.5:
            score -= 5.0
        if not next_has_food and current_role == "outside" and next_role != "outside":
            score -= 1.25
        if (
            not next_has_food
            and current_role == "entrance"
            and next_role in {"inside", "deep"}
        ):
            score -= 1.75
        if score > best_score:
            best_score = score
            best_action = action_name
            best_delta = food_delta

    if best_action == "STAY" or best_score <= 0.0:
        return None
    return best_action, best_delta, source


def _best_food_action(
    meta: Mapping[str, object],
    candidates: Sequence[str],
) -> tuple[str, float, str]:
    if bool(meta.get("on_shelter", False)) and str(meta.get("shelter_role", "outside")) not in {
        "outside",
        "entrance",
    }:
        exit_action = _best_exit_action(meta, candidates)
        if exit_action[0] != "STAY":
            return exit_action
        if _predator_pressure(meta) >= 0.50:
            return exit_action

    vector_action = _vector_guided_food_action(meta, candidates)
    if vector_action is not None:
        return vector_action

    best_action = "STAY"
    best_delta = 0.0
    best_score = -1e9
    current_role = str(meta.get("shelter_role", "outside"))
    for action_name in candidates:
        transition = _transition_for(meta, action_name)
        affordance = _affordance_for(meta, action_name)
        delta = _float_value(transition.get("food_dist_delta"), 0.0)
        predator_delta = _float_value(transition.get("predator_dist_delta"), 0.0)
        next_has_food = bool(transition.get("next_cell_has_food", False))
        score = delta + 0.40 * predator_delta
        next_role = str(affordance.get("next_role", current_role))
        if not next_has_food and current_role == "outside" and next_role != "outside":
            score -= 1.50
        elif (
            not next_has_food
            and current_role == "entrance"
            and next_role in {"inside", "deep"}
        ):
            score -= 2.00
        if next_has_food:
            score += 2.0
        if score > best_score:
            best_score = score
            best_action = action_name
            best_delta = delta
    if best_score <= -1e8:
        return "STAY", 0.0, "no_food_candidate"
    return best_action, best_delta, "food_progress"


def _best_shelter_action(
    meta: Mapping[str, object],
    candidates: Sequence[str],
) -> tuple[str, float, str]:
    best_action = "STAY"
    best_delta = 0.0
    best_score = -1e9
    predator_pressure = _predator_pressure(meta)
    for action_name in candidates:
        transition = _transition_for(meta, action_name)
        geodesic = _geodesic_for(meta, action_name)
        shelter_delta = _float_value(transition.get("shelter_dist_delta"), 0.0)
        predator_delta = _float_value(transition.get("predator_dist_delta"), 0.0)
        exit_delta = _float_value(geodesic.get("exit_geodesic_delta"), 0.0)
        deep_delta = _float_value(geodesic.get("deep_geodesic_delta"), 0.0)
        score = (
            shelter_delta
            + 0.35 * exit_delta
            + 0.50 * deep_delta
            + (1.25 * predator_pressure * predator_delta)
        )
        if predator_pressure >= 0.50 and predator_delta <= -0.5:
            score -= 4.0
        if bool(geodesic.get("next_on_deep_target", False)):
            score += 1.0
        elif bool(geodesic.get("next_on_exit_target", False)):
            score += 0.5
        if score > best_score:
            best_score = score
            best_action = action_name
            best_delta = shelter_delta
    if best_score <= -1e8:
        return "STAY", 0.0, "no_shelter_candidate"
    return best_action, best_delta, "shelter_progress"


def _explore_action(
    candidates: Sequence[str],
    *,
    rng: np.random.Generator | None,
    sample: bool,
) -> tuple[str, str]:
    if not candidates:
        return "STAY", "explore_no_unblocked_move"
    if sample and rng is not None:
        order = list(candidates)
        rng.shuffle(order)
        return str(order[0]), "explore_seeded_shuffle"
    for action_name in BRIDGE_EXPLORE_ORDER:
        if action_name in candidates:
            return action_name, "explore_deterministic_order"
    return str(candidates[0]), "explore_first_unblocked"


def bridge_b_semantic_action(
    semantic_action: str,
    observation: Mapping[str, object],
    *,
    rng: np.random.Generator | None = None,
    sample: bool = False,
) -> BSeriesBridgeDecision:
    if semantic_action not in B_SEMANTIC_ACTION_TO_INDEX:
        raise ValueError(f"Unknown B-series semantic action: {semantic_action!r}.")
    meta = _meta_from_observation(observation)
    blocked_mask = b_series_blocked_mask(observation)
    candidates = _movement_candidates(blocked_mask)
    food_delta = 0.0
    shelter_delta = 0.0

    if semantic_action == "MOVE_TO_FOOD":
        if bool(meta.get("on_food", False)):
            primitive_action = "STAY"
            reason = "already_on_food"
        else:
            primitive_action, food_delta, reason = _best_food_action(meta, candidates)
    elif semantic_action == "MOVE_TO_SHELTER":
        shelter_role = str(meta.get("shelter_role", "outside"))
        shelter_role_level = _float_value(meta.get("shelter_role_level"), 0.0)
        if bool(meta.get("on_shelter", False)) and (
            shelter_role == "deep" or shelter_role_level >= 0.95
        ):
            primitive_action = "STAY"
            reason = "already_deep_shelter"
        elif bool(meta.get("on_shelter", False)) and not candidates:
            primitive_action = "STAY"
            reason = "shelter_hold_no_unblocked_move"
        else:
            primitive_action, shelter_delta, reason = _best_shelter_action(
                meta,
                candidates,
            )
    elif semantic_action == "EXPLORE":
        primitive_action, reason = _explore_action(
            candidates,
            rng=rng,
            sample=sample,
        )
    elif semantic_action in {"STAY", "EAT", "SLEEP"}:
        primitive_action = "STAY"
        reason = f"{semantic_action.lower()}_maps_to_stay"
    else:
        primitive_action = "STAY"
        reason = "fallback_stay"

    if bool(blocked_mask.get(primitive_action, False)):
        primitive_action = "STAY"
        reason = f"{reason}_blocked_to_stay"
    return BSeriesBridgeDecision(
        semantic_action=semantic_action,
        primitive_action=primitive_action,
        reason=reason,
        blocked_mask=dict(blocked_mask),
        food_delta_used=float(food_delta),
        shelter_delta_used=float(shelter_delta),
        external_override_count=0,
    )


__all__ = [
    "B_CURRENT_BRIDGE_EFFECTIVE_LEVEL",
    "B_CURRENT_BRIDGE_SELECTION_SOURCE",
    "B_SERIES_MODES",
    "B_SERIES_POLICY_NAME",
    "B_SEMANTIC_ACTIONS",
    "B_SEMANTIC_ACTION_TO_INDEX",
    "BSeriesBridgeDecision",
    "bridge_b_semantic_action",
    "b_series_blocked_mask",
]
