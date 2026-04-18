from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .ablations import BrainAblationConfig
from .interfaces import ACTION_TO_INDEX
from .modules import ModuleResult, ReflexDecision
from .nn_utils import one_hot, softmax
from .operational_profiles import OperationalProfile


def _action_dim(interface_registry: Mapping[str, object]) -> int:
    """
    Determine the dimensionality of the action space using the provided interface registry.

    If the registry contains an "actions" entry that is a non-string sequence, the function returns its length; otherwise it falls back to the length of the module's ACTION_TO_INDEX mapping.

    Parameters:
        interface_registry (Mapping[str, object]): Registry that may include an "actions" entry describing available actions.

    Returns:
        int: Number of possible actions (action vector length).
    """
    actions = interface_registry.get("actions")
    if isinstance(actions, Sequence) and not isinstance(actions, str):
        return len(actions)
    return len(ACTION_TO_INDEX)


def _action_target(action: str, *, interface_registry: Mapping[str, object]) -> np.ndarray:
    """
    Map a named action to a float one-hot probability vector.

    Parameters:
        action (str): Action name key looked up in the interface registry.
        interface_registry (Mapping[str, object]): Registry used to resolve action order and determine the action vector length.

    Returns:
        np.ndarray: Float one-hot vector with `1.0` at the action's index and `0.0` elsewhere.

    Raises:
        KeyError: If `action` is not present in the interface registry.
    """
    actions = interface_registry["actions"]
    if not isinstance(actions, Sequence) or isinstance(actions, str):
        raise TypeError("interface_registry['actions'] must be a sequence of action names")
    try:
        local_index = actions.index(action)
    except ValueError as exc:
        raise KeyError(action) from exc
    return one_hot(local_index, _action_dim(interface_registry))


def _direction_action(dx: float, dy: float, *, away: bool = False) -> str:
    """
    Map a displacement vector into a discrete movement action name.

    If `away` is True the displacement is negated before classification; if both x and y magnitudes are less than 0.05 the action is `"STAY"`. When one axis dominates, choose the corresponding cardinal move: horizontal selects `"MOVE_RIGHT"` for positive x or `"MOVE_LEFT"` for negative x; otherwise vertical selects `"MOVE_DOWN"` for positive y or `"MOVE_UP"` for negative y.

    Parameters:
        dx (float): Horizontal displacement (positive is right); will be negated when `away` is True.
        dy (float): Vertical displacement (positive is down); will be negated when `away` is True.
        away (bool): If True, interpret the action as moving away from the (dx, dy) vector.

    Returns:
        action (str): One of `"STAY"`, `"MOVE_RIGHT"`, `"MOVE_LEFT"`, `"MOVE_DOWN"`, or `"MOVE_UP"`.
    """
    dx = float(-dx if away else dx)
    dy = float(-dy if away else dy)
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return "STAY"
    if abs(dx) >= abs(dy):
        return "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT"
    return "MOVE_DOWN" if dy > 0 else "MOVE_UP"


def _signal_subset(observations: Mapping[str, float], *names: str) -> dict[str, float]:
    """
    Extract the specified keys from an observations mapping and return their values as floats.

    Parameters:
        observations (Mapping[str, float]): Mapping of observation names to numeric values (or values convertible to float).
        *names (str): One or more keys to extract from `observations`.

    Returns:
        dict[str, float]: A dictionary mapping each requested name to its value cast to `float`.

    Raises:
        KeyError: If a requested name is not present in `observations`.
        TypeError, ValueError: If a value cannot be converted to `float`.
    """
    return {
        name: float(observations[name])
        for name in names
    }


def _reflex_decision(
    module_name: str,
    *,
    action: str,
    reason: str,
    triggers: dict[str, float],
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision:
    """
    Create a ReflexDecision for the specified module, using the operational profile to set auxiliary weight and logit strength.

    Parameters:
        module_name (str): Name of the proposal module whose configuration is used to look up weights.
        action (str): Action name to convert into a one-hot target probability vector.
        reason (str): Short tag describing why the reflex was created.
        triggers (dict[str, float]): Mapping of observation signal names to the values that triggered this reflex.
        operational_profile (OperationalProfile): Source of `brain_aux_weights` and `brain_reflex_logit_strengths` for the module.
        interface_registry (Mapping[str, object]): Registry used to determine the action space when building `target_probs`.

    Returns:
        ReflexDecision: A decision containing `action`, `target_probs` (one-hot), `reason`, `triggers`, and the module's configured `auxiliary_weight` and `logit_strength` (defaults to 0.0 if not present).
    """
    return ReflexDecision(
        action=action,
        target_probs=_action_target(action, interface_registry=interface_registry),
        reason=reason,
        triggers=triggers,
        auxiliary_weight=operational_profile.brain_aux_weights.get(module_name, 0.0),
        logit_strength=operational_profile.brain_reflex_logit_strengths.get(module_name, 0.0),
    )


def _stay_reflex(
    module_name: str,
    *,
    reason: str,
    triggers: dict[str, float],
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision:
    """
    Create a ReflexDecision that selects the 'STAY' action for the given module.

    Parameters:
        reason (str): Short identifier or description explaining why the reflex was generated.
        triggers (dict[str, float]): Mapping of trigger names to their magnitudes that caused this reflex.

    Returns:
        ReflexDecision: A decision targeting the `STAY` action with the provided reason and trigger magnitudes.
    """
    return _reflex_decision(
        module_name,
        action="STAY",
        reason=reason,
        triggers=triggers,
        operational_profile=operational_profile,
        interface_registry=interface_registry,
    )


def _direction_reflex(
    module_name: str,
    *,
    dx: float,
    dy: float,
    away: bool = False,
    reason: str,
    triggers: dict[str, float],
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision:
    """
    Builds a ReflexDecision that directs movement based on a displacement vector.

    The returned decision's action is chosen from the displacement (dx, dy); if
    away is True the opposite direction is chosen. The decision is populated
    with the provided reason, triggers, and scaling metadata from the
    operational_profile and interface_registry.

    Parameters:
        dx (float): X-component of the displacement used to choose direction.
        dy (float): Y-component of the displacement used to choose direction.
        away (bool): If True, select the direction opposite to (dx, dy).
        reason (str): Short identifier describing why the reflex was created.
        triggers (dict[str, float]): Observational signals that triggered this reflex.
        operational_profile (OperationalProfile): Profile supplying per-module reflex weights and strengths.
        interface_registry (Mapping[str, object]): Registry used to determine action dimensionality and mapping.

    Returns:
        ReflexDecision: A decision targeting a directional action derived from the inputs.
    """
    return _reflex_decision(
        module_name,
        action=_direction_action(dx, dy, away=away),
        reason=reason,
        triggers=triggers,
        operational_profile=operational_profile,
        interface_registry=interface_registry,
    )


def _visual_reflex_decision(
    observations: Mapping[str, float],
    *,
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision | None:
    """
    Selects a visual reflex decision based on visual observations (predator, shelter at night, or visible food).

    Checks observation signals against the visual cortex thresholds in the provided OperationalProfile and, for the first matching condition, returns a ReflexDecision describing the action and observed triggers. Returns None when no visual reflex thresholds are met.

    Returns:
        ReflexDecision | None: A ReflexDecision for one of the visual cases (`retreat_from_visible_predator`, `return_to_shelter_at_night`, or `approach_visible_food`) or `None` if no visual reflex applies.
    """
    thresholds = operational_profile.brain_reflex_thresholds["visual_cortex"]
    if (
        observations["predator_visible"] > thresholds["visible"]
        and observations["predator_certainty"] >= thresholds["predator_certainty"]
    ):
        return _direction_reflex(
            "visual_cortex",
            dx=observations["predator_dx"],
            dy=observations["predator_dy"],
            away=True,
            reason="retreat_from_visible_predator",
            triggers=_signal_subset(
                observations,
                "predator_visible",
                "predator_certainty",
                "predator_dx",
                "predator_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["night"] > thresholds["visible"]
        and observations["shelter_visible"] > thresholds["visible"]
        and observations["shelter_certainty"] >= thresholds["shelter_certainty"]
    ):
        return _direction_reflex(
            "visual_cortex",
            dx=observations["shelter_dx"],
            dy=observations["shelter_dy"],
            reason="return_to_shelter_at_night",
            triggers=_signal_subset(
                observations,
                "night",
                "shelter_visible",
                "shelter_certainty",
                "shelter_dx",
                "shelter_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["food_visible"] > thresholds["visible"]
        and observations["food_certainty"] >= thresholds["food_certainty"]
    ):
        return _direction_reflex(
            "visual_cortex",
            dx=observations["food_dx"],
            dy=observations["food_dy"],
            reason="approach_visible_food",
            triggers=_signal_subset(
                observations,
                "food_visible",
                "food_certainty",
                "food_dx",
                "food_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    return None


def _sensory_reflex_decision(
    observations: Mapping[str, float],
    *,
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision | None:
    """
    Decide whether sensory observations trigger an immediate retreat or a hunger-driven approach to a smelled food source.

    Parameters:
        observations: Mapping of sensory signals used by the decision logic. Expected keys include:
            - recent_contact, recent_pain
            - predator_smell_strength, predator_smell_dx, predator_smell_dy
            - hunger, food_smell_strength, food_smell_dx, food_smell_dy
        operational_profile: OperationalProfile providing threshold values under "sensory_cortex".
        interface_registry: Registry used to determine action dimensionality and targets.

    Returns:
        ReflexDecision if a reflex is triggered (either an away-directed retreat from an immediate threat
        or a directed approach toward a food smell when hungry), `None` if no sensory thresholds are exceeded.
    """
    thresholds = operational_profile.brain_reflex_thresholds["sensory_cortex"]
    if (
        observations["recent_contact"] > thresholds["contact"]
        or observations["recent_pain"] > thresholds["pain"]
        or observations["predator_smell_strength"] > thresholds["predator_smell"]
    ):
        return _direction_reflex(
            "sensory_cortex",
            dx=observations["predator_smell_dx"],
            dy=observations["predator_smell_dy"],
            away=True,
            reason="retreat_from_immediate_threat",
            triggers=_signal_subset(
                observations,
                "recent_contact",
                "recent_pain",
                "predator_smell_strength",
                "predator_smell_dx",
                "predator_smell_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["hunger"] > thresholds["hunger"]
        and observations["food_smell_strength"] > thresholds["food_smell"]
    ):
        return _direction_reflex(
            "sensory_cortex",
            dx=observations["food_smell_dx"],
            dy=observations["food_smell_dy"],
            reason="follow_food_smell_when_hungry",
            triggers=_signal_subset(
                observations,
                "hunger",
                "food_smell_strength",
                "food_smell_dx",
                "food_smell_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    return None


def _hunger_reflex_decision(
    observations: Mapping[str, float],
    *,
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision | None:
    """
    Decide a hunger-driven reflex to stay or move toward food based on observation thresholds.

    Parameters:
        observations (Mapping[str, float]): Current sensory and internal signals used to evaluate hunger reflex conditions (e.g., `hunger`, `on_food`, `food_visible`, smell/memory vectors and strengths).
        operational_profile (OperationalProfile): Profile containing per-center reflex thresholds, auxiliary weights, and logit strengths.
        interface_registry (Mapping[str, object]): Registry describing available actions and interfaces (used to build action targets).

    Returns:
        ReflexDecision | None: A configured `ReflexDecision` that directs staying or moving toward/away from food when a threshold condition is met, or `None` if no hunger reflex applies.
    """
    thresholds = operational_profile.brain_reflex_thresholds["hunger_center"]
    if (
        observations["on_food"] > thresholds["on_food"]
        and observations["hunger"] > thresholds["stay_hunger"]
    ):
        return _stay_reflex(
            "hunger_center",
            reason="stay_on_food",
            triggers=_signal_subset(observations, "on_food", "hunger"),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["hunger"] > thresholds["visible_hunger"]
        and observations["food_visible"] > thresholds["visible_food"]
        and observations["food_certainty"] >= thresholds["visible_food_certainty"]
    ):
        return _direction_reflex(
            "hunger_center",
            dx=observations["food_dx"],
            dy=observations["food_dy"],
            reason="approach_visible_food",
            triggers=_signal_subset(
                observations,
                "hunger",
                "food_visible",
                "food_certainty",
                "food_dx",
                "food_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["hunger"] > thresholds["occluded_hunger"]
        and observations["food_occluded"] > thresholds["occluded_food"]
        and observations["food_smell_strength"] > thresholds["occluded_food_smell"]
    ):
        return _direction_reflex(
            "hunger_center",
            dx=observations["food_smell_dx"],
            dy=observations["food_smell_dy"],
            reason="follow_occluded_food_smell",
            triggers=_signal_subset(
                observations,
                "hunger",
                "food_occluded",
                "food_smell_strength",
                "food_smell_dx",
                "food_smell_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["hunger"] > thresholds["smell_hunger"]
        and observations["food_smell_strength"] > thresholds["food_smell"]
    ):
        return _direction_reflex(
            "hunger_center",
            dx=observations["food_smell_dx"],
            dy=observations["food_smell_dy"],
            reason="follow_food_smell",
            triggers=_signal_subset(
                observations,
                "hunger",
                "food_smell_strength",
                "food_smell_dx",
                "food_smell_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["hunger"] > thresholds["memory_hunger"]
        and observations["food_memory_age"] < thresholds["memory_age"]
    ):
        return _direction_reflex(
            "hunger_center",
            dx=observations["food_memory_dx"],
            dy=observations["food_memory_dy"],
            reason="follow_food_memory",
            triggers=_signal_subset(
                observations,
                "hunger",
                "food_memory_dx",
                "food_memory_dy",
                "food_memory_age",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    return None


def _sleep_reflex_decision(
    observations: Mapping[str, float],
    *,
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision | None:
    """
    Decides whether sleep-related observations should trigger staying in place or moving toward shelter for rest.

    Evaluates signals against the `sleep_center` thresholds in `operational_profile.brain_reflex_thresholds` and returns a `ReflexDecision` representing either a `"STAY"` reflex or a directional reflex toward shelter/memory/trace when a matching threshold condition is met; returns `None` when no sleep-related reflex applies.

    Parameters:
        observations: Mapping of observation names to float values used for threshold comparisons (e.g., `on_shelter`, `fatigue`, `sleep_debt`, `hunger`, shelter memory/trace fields).
        operational_profile: OperationalProfile containing `brain_reflex_thresholds` and per-module reflex configuration used to construct the returned `ReflexDecision`.
        interface_registry: Registry used to resolve action encoding/targets for constructing reflex decisions.

    Returns:
        ReflexDecision if a sleep-related reflex condition is met, `None` otherwise.
    """
    thresholds = operational_profile.brain_reflex_thresholds["sleep_center"]
    if (
        observations["on_shelter"] > thresholds["on_shelter"]
        and observations["sleep_phase_level"] > thresholds["sleep_phase"]
        and observations["hunger"] < thresholds["rest_hunger"]
    ):
        return _stay_reflex(
            "sleep_center",
            reason="stay_while_sleeping",
            triggers=_signal_subset(
                observations,
                "on_shelter",
                "sleep_phase_level",
                "hunger",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["on_shelter"] > thresholds["on_shelter"]
        and observations["shelter_role_level"] > thresholds["deep_shelter_level"]
        and observations["rest_streak_norm"] > thresholds["rest_streak"]
        and observations["hunger"] < thresholds["rest_hunger"]
    ):
        return _stay_reflex(
            "sleep_center",
            reason="stay_in_deep_shelter",
            triggers=_signal_subset(
                observations,
                "on_shelter",
                "shelter_role_level",
                "rest_streak_norm",
                "hunger",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["on_shelter"] > thresholds["on_shelter"]
        and (
            observations["night"] > thresholds["on_shelter"]
            or observations["fatigue"] > thresholds["fatigue_to_hold"]
            or observations["sleep_debt"] > thresholds["sleep_debt_to_hold"]
        )
        and observations["hunger"] < thresholds["rest_hunger"]
    ):
        return _stay_reflex(
            "sleep_center",
            reason="stay_in_shelter_to_rest",
            triggers=_signal_subset(
                observations,
                "on_shelter",
                "night",
                "fatigue",
                "sleep_debt",
                "hunger",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["shelter_memory_age"] < thresholds["memory_age"]
        and (
            observations["fatigue"] > thresholds["memory_fatigue"]
            or observations["sleep_debt"] > thresholds["sleep_debt_to_seek"]
            or observations["night"] > thresholds["on_shelter"]
        )
        and observations["hunger"] < thresholds["memory_rest_hunger"]
    ):
        return _direction_reflex(
            "sleep_center",
            dx=observations["shelter_memory_dx"],
            dy=observations["shelter_memory_dy"],
            reason="return_to_safe_shelter_memory",
            triggers=_signal_subset(
                observations,
                "fatigue",
                "sleep_debt",
                "night",
                "hunger",
                "shelter_memory_dx",
                "shelter_memory_dy",
                "shelter_memory_age",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        (
            observations["fatigue"] > thresholds["fatigue_to_seek"]
            or observations["sleep_debt"] > thresholds["sleep_debt_to_seek"]
            or observations["night"] > thresholds["on_shelter"]
        )
        and observations["hunger"] < thresholds["memory_rest_hunger"]
        and observations["shelter_trace_strength"] > 0.0
    ):
        return _direction_reflex(
            "sleep_center",
            dx=observations["shelter_trace_dx"],
            dy=observations["shelter_trace_dy"],
            reason="follow_shelter_trace_to_rest",
            triggers=_signal_subset(
                observations,
                "fatigue",
                "sleep_debt",
                "night",
                "hunger",
                "shelter_trace_dx",
                "shelter_trace_dy",
                "shelter_trace_strength",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    return None


def _alert_reflex_decision(
    observations: Mapping[str, float],
    *,
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision | None:
    """
    Evaluate observations against the alert-center thresholds and produce an alert-related reflex decision when a threshold is exceeded.

    Parameters:
        observations (Mapping[str, float]): Observation signals keyed by name used to evaluate alert conditions (e.g., predator visibility, smell, contact, memory/trace vectors).
        operational_profile (OperationalProfile): Provides alert-center threshold values and reflex configuration used to build the returned decision.
        interface_registry (Mapping[str, object]): Registry consulted to size action/target vectors.

    Returns:
        ReflexDecision: A reflex describing the chosen alert action when a threshold is met.
        None: If no alert reflex conditions are met.
    """
    thresholds = operational_profile.brain_reflex_thresholds["alert_center"]
    if (
        observations["on_shelter"] > thresholds["on_shelter"]
        and (
            observations["predator_visible"] > thresholds["predator_visible"]
            or observations["predator_occluded"] > thresholds["predator_occluded"]
            or observations["predator_smell_strength"] > thresholds["predator_smell"]
            or observations["recent_contact"] > thresholds["contact_any"]
        )
    ):
        return _stay_reflex(
            "alert_center",
            reason="freeze_in_shelter_under_threat",
            triggers=_signal_subset(
                observations,
                "on_shelter",
                "predator_visible",
                "predator_occluded",
                "predator_smell_strength",
                "recent_contact",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["predator_visible"] > thresholds["predator_visible"]
        and observations["predator_certainty"] >= thresholds["predator_certainty"]
    ):
        return _direction_reflex(
            "alert_center",
            dx=observations["predator_dx"],
            dy=observations["predator_dy"],
            away=True,
            reason="retreat_from_visible_predator",
            triggers=_signal_subset(
                observations,
                "predator_visible",
                "predator_certainty",
                "predator_dx",
                "predator_dy",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["predator_memory_age"] < thresholds["predator_memory_age"]
        and observations["on_shelter"] < thresholds["on_shelter"]
    ):
        return _direction_reflex(
            "alert_center",
            dx=observations["predator_memory_dx"],
            dy=observations["predator_memory_dy"],
            away=True,
            reason="retreat_from_predator_memory",
            triggers=_signal_subset(
                observations,
                "on_shelter",
                "predator_memory_dx",
                "predator_memory_dy",
                "predator_memory_age",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        (
            observations["predator_occluded"] > thresholds["predator_occluded"]
            or observations["predator_smell_strength"] > thresholds["predator_smell"]
            or observations["recent_contact"] > thresholds["contact_threat"]
            or observations["recent_pain"] > thresholds["pain"]
        )
        and observations["predator_trace_strength"] > 0.0
    ):
        return _direction_reflex(
            "alert_center",
            dx=observations["predator_trace_dx"],
            dy=observations["predator_trace_dy"],
            away=True,
            reason="retreat_from_predator_trace",
            triggers=_signal_subset(
                observations,
                "predator_occluded",
                "predator_smell_strength",
                "recent_contact",
                "recent_pain",
                "predator_trace_dx",
                "predator_trace_dy",
                "predator_trace_strength",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if (
        observations["escape_memory_age"] < thresholds["escape_memory_age"]
        and observations["on_shelter"] < thresholds["on_shelter"]
    ):
        return _direction_reflex(
            "alert_center",
            dx=observations["escape_memory_dx"],
            dy=observations["escape_memory_dy"],
            reason="repeat_recent_escape_route",
            triggers=_signal_subset(
                observations,
                "on_shelter",
                "escape_memory_dx",
                "escape_memory_dy",
                "escape_memory_age",
            ),
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    return None


def _module_reflex_decision(
    module_name: str,
    observations: Mapping[str, float],
    *,
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
) -> ReflexDecision | None:
    """
    Selects and runs the reflex evaluator for the specified brain module using the given observations.

    Parameters:
        module_name (str): Name of the brain module to evaluate. Supported values: "visual_cortex", "sensory_cortex", "hunger_center", "sleep_center", "alert_center".
        observations (Mapping[str, float]): Current observation signals keyed by name (e.g., sensory and memory cues).
        operational_profile (OperationalProfile): Operational thresholds and weights used by reflex evaluators.
        interface_registry (Mapping[str, object]): Registry describing available actions/interfaces used to construct action targets.

    Returns:
        ReflexDecision | None: A ReflexDecision if a reflex condition is met for the module, `None` otherwise.
    """
    if module_name == "visual_cortex":
        return _visual_reflex_decision(
            observations,
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if module_name == "sensory_cortex":
        return _sensory_reflex_decision(
            observations,
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if module_name == "hunger_center":
        return _hunger_reflex_decision(
            observations,
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if module_name == "sleep_center":
        return _sleep_reflex_decision(
            observations,
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    if module_name == "alert_center":
        return _alert_reflex_decision(
            observations,
            operational_profile=operational_profile,
            interface_registry=interface_registry,
        )
    return None


def _effective_reflex_scale(
    module_name: str,
    *,
    ablation_config: BrainAblationConfig,
    current_reflex_scale: float,
) -> float:
    """
    Compute the effective reflex scale for a module given ablation settings and a base reflex scale.

    If reflexes are disabled or the system is not modular, this returns 0.0. Otherwise returns the product of `current_reflex_scale` and the module-specific multiplier from `ablation_config.module_reflex_scales` (defaults to 1.0), clamped to be greater than or equal to 0.0.

    Parameters:
        module_name (str): Name of the module whose reflex scale is being computed.
        ablation_config (BrainAblationConfig): Configuration controlling whether reflexes are enabled, modular, and per-module reflex multipliers.
        current_reflex_scale (float): Base reflex scale to adjust by the module-specific multiplier.

    Returns:
        float: Effective reflex scale (>= 0.0).
    """
    if not ablation_config.enable_reflexes or not ablation_config.is_modular:
        return 0.0
    return max(
        0.0,
        float(current_reflex_scale)
        * float(ablation_config.module_reflex_scales.get(module_name, 1.0)),
    )


def _apply_reflex_path(
    module_results: list[ModuleResult],
    *,
    ablation_config: BrainAblationConfig,
    operational_profile: OperationalProfile,
    interface_registry: Mapping[str, object],
    current_reflex_scale: float,
    module_valence_roles: Mapping[str, str],
) -> None:
    """
    Apply reflex decisions to each ModuleResult, updating logits, probabilities, and reflex metadata in place.

    Parameters:
        module_results (list[ModuleResult]): List of module result objects to update; each element is mutated.
        ablation_config (BrainAblationConfig): Configuration controlling whether reflexes are enabled and per-module reflex scaling.
        operational_profile (OperationalProfile): Profile containing reflex thresholds, auxiliary weights, and default logit strengths used to build reflex decisions.
        interface_registry (Mapping[str, object]): Registry used to determine action dimensionality and construct reflex targets.
        current_reflex_scale (float): Global reflex strength multiplier applied before per-module scaling.
        module_valence_roles (Mapping[str, str]): Mapping from module name to its valence role (e.g., "support" or other role); used to set each result's valence_role.

    Behavior:
        - Mutates each ModuleResult in module_results by initializing reflex-related fields and, when modular reflexes are enabled via ablation_config, computing a ReflexDecision for active modules, scaling its strength, and applying its logit delta to update logits and probs.
        - Records whether a reflex was applied, whether it changed the module's argmax (module_reflex_override), a reflex dominance metric, effective_reflex_scale, and other reflex metadata on each ModuleResult.
    """
    for result in module_results:
        result.neural_logits = result.logits.copy()
        result.reflex_delta_logits = np.zeros_like(result.logits)
        result.post_reflex_logits = result.logits.copy()
        result.reflex = None
        result.reflex_applied = False
        result.effective_reflex_scale = 0.0
        result.module_reflex_override = False
        result.module_reflex_dominance = 0.0
        result.valence_role = module_valence_roles.get(result.name, "support")
        result.gate_weight = 1.0
        result.contribution_share = 0.0
        result.gated_logits = result.logits.copy()
        result.intent_before_gating = None
        result.intent_after_gating = None

    should_compute_reflexes = ablation_config.is_modular and ablation_config.enable_reflexes
    if should_compute_reflexes:
        for result in module_results:
            if not result.active:
                continue
            result.reflex = _module_reflex_decision(
                result.name,
                result.named_observation(),
                operational_profile=operational_profile,
                interface_registry=interface_registry,
            )
            reflex = result.reflex
            if reflex is None:
                continue
            effective_scale = _effective_reflex_scale(
                result.name,
                ablation_config=ablation_config,
                current_reflex_scale=current_reflex_scale,
            )
            result.effective_reflex_scale = float(effective_scale)
            effective_logit_strength = float(reflex.logit_strength) * effective_scale
            reflex.auxiliary_weight = float(reflex.auxiliary_weight) * effective_scale
            reflex.logit_strength = effective_logit_strength
            if effective_logit_strength <= 0.0:
                continue
            result.reflex_delta_logits = effective_logit_strength * reflex.target_probs
            result.post_reflex_logits = result.neural_logits + result.reflex_delta_logits
            result.logits = result.post_reflex_logits.copy()
            result.probs = softmax(result.logits)
            result.reflex_applied = bool(np.any(np.abs(result.reflex_delta_logits) > 1e-12))
            neural_argmax = int(np.argmax(result.neural_logits))
            post_argmax = int(np.argmax(result.post_reflex_logits))
            result.module_reflex_override = neural_argmax != post_argmax
            denom = (
                float(np.sum(np.abs(result.neural_logits)))
                + float(np.sum(np.abs(result.reflex_delta_logits)))
                + 1e-8
            )
            result.module_reflex_dominance = float(
                np.sum(np.abs(result.reflex_delta_logits)) / denom
            )
            result.gated_logits = result.logits.copy()
