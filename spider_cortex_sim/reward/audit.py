"""Reward component audit catalog and profile audit summaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from ._utils import policy_float
from .profiles import REWARD_PROFILES
from .shaping import (
    SHAPING_DISPOSITIONS,
    SHAPING_REDUCTION_ROADMAP,
    validate_shaping_disposition,
)

REWARD_COMPONENT_AUDIT = {
    "action_cost": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_action_and_terrain_effects",
        "gates": ["always", "scaled by action_name == STAY vs locomotion"],
        "config_keys": ["action_cost_stay", "action_cost_move"],
        "configured_weight_keys": ["action_cost_stay", "action_cost_move"],
        "shaping_risk": "low",
        "shaping_disposition": "defended",
        "disposition_rationale": (
            "Retained in the austere profile as a small universal action and energy budget, "
            "not as task-directed progress guidance."
        ),
        "notes": "Penaliza cada passo; funciona mais como custo universal do que como trilha guiada.",
    },
    "terrain_cost": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_action_and_terrain_effects",
        "gates": ["terrain == CLUTTER or NARROW", "extra branch when NARROW and predator is close"],
        "config_keys": ["clutter_cost", "narrow_cost", "narrow_predator_risk"],
        "configured_weight_keys": ["clutter_cost", "narrow_cost", "narrow_predator_risk"],
        "shaping_risk": "medium",
        "shaping_disposition": "defended",
        "disposition_rationale": (
            "Retained in the austere profile because clutter and narrow-passage penalties "
            "model environmental hazards rather than direct waypoint progress."
        ),
        "notes": "Encodes explicit environmental cost and pushes the agent away from dangerous bottlenecks.",
    },
    "night_exposure": {
        "category": "event",
        "source": "spider_cortex_sim.world.SpiderWorld.step",
        "gates": ["night", "not on shelter"],
        "config_keys": ["night_exposure_reward"],
        "configured_weight_keys": ["night_exposure_reward"],
        "shaping_risk": "medium",
        "shaping_disposition": "defended",
        "disposition_rationale": (
            "Retained in the austere profile because nighttime exposure is an ecological "
            "consequence of failing to use shelter."
        ),
        "notes": "Dense penalty for nighttime exposure outside shelter.",
    },
    "hunger_pressure": {
        "category": "internal_pressure",
        "source": "spider_cortex_sim.reward.apply_pressure_penalties",
        "gates": ["always, scaled by state.hunger"],
        "config_keys": ["hunger_pressure"],
        "configured_weight_keys": ["hunger_pressure"],
        "shaping_risk": "medium",
        "shaping_disposition": "defended",
        "disposition_rationale": (
            "Retained and strengthened in the austere profile as internal homeostatic "
            "pressure; it defines need without pointing directly to food."
        ),
        "notes": "Continuous homeostatic pressure rather than a terminal event.",
    },
    "fatigue_pressure": {
        "category": "internal_pressure",
        "source": "spider_cortex_sim.reward.apply_pressure_penalties",
        "gates": ["always, scaled by state.fatigue"],
        "config_keys": ["fatigue_pressure"],
        "configured_weight_keys": ["fatigue_pressure"],
        "shaping_risk": "medium",
        "shaping_disposition": "defended",
        "disposition_rationale": (
            "Retained and strengthened in the austere profile as an internal exhaustion "
            "cost rather than destination guidance."
        ),
        "notes": "Continuous internal pressure to seek rest.",
    },
    "sleep_debt_pressure": {
        "category": "internal_pressure",
        "source": "spider_cortex_sim.reward.apply_pressure_penalties",
        "gates": ["always, scaled by state.sleep_debt"],
        "config_keys": ["sleep_debt_pressure"],
        "configured_weight_keys": ["sleep_debt_pressure"],
        "shaping_risk": "medium",
        "shaping_disposition": "defended",
        "disposition_rationale": (
            "Retained and strengthened in the austere profile to represent physiological "
            "sleep debt while leaving shelter-finding to behavior."
        ),
        "notes": "Pushes shelter-seeking behavior even before clear behavioral failures appear.",
    },
    "predator_contact": {
        "category": "event",
        "source": "spider_cortex_sim.physiology.apply_predator_contact",
        "gates": ["predator contact"],
        "config_keys": [],
        "configured_weight_keys": [],
        "hardcoded_weight": 2.4,
        "shaping_risk": "low",
        "shaping_disposition": "outcome_signal",
        "disposition_rationale": (
            "Acute external damage anchors a negative outcome signal tied to predator contact, "
            "not incremental escape guidance."
        ),
        "notes": "Acute event anchored in real predator-inflicted damage.",
    },
    "feeding": {
        "category": "event",
        "source": "spider_cortex_sim.physiology.resolve_autonomic_behaviors",
        "gates": ["world.on_food()"],
        "config_keys": [],
        "configured_weight_keys": [],
        "hardcoded_weight": 3.48,
        "shaping_risk": "low",
        "shaping_disposition": "outcome_signal",
        "disposition_rationale": (
            "Consummatory reward marks the positive outcome of reaching food rather than "
            "providing distance-based shaping."
        ),
        "notes": "Event-level reinforcement when the spider actually reaches food.",
    },
    "resting": {
        "category": "event",
        "source": "spider_cortex_sim.physiology.resolve_autonomic_behaviors",
        "gates": ["world.on_shelter()", "rest conditions satisfied"],
        "config_keys": [
            "sleep_debt_resting_night",
            "sleep_debt_resting_day",
            "sleep_debt_deep_night",
            "sleep_debt_deep_day",
        ],
        "configured_weight_keys": [
            "sleep_debt_resting_night",
            "sleep_debt_resting_day",
            "sleep_debt_deep_night",
            "sleep_debt_deep_day",
        ],
        "hardcoded_weight": 2.2,
        "shaping_risk": "medium",
        "shaping_disposition": "weakened",
        "disposition_rationale": (
            "The austere profile keeps the direct rest outcome but trims configurable rest "
            "bonuses relative to the ecological profile; dense intermediate rest shaping remains monitored."
        ),
        "notes": "Combines a legitimate ecological event with dense shaping for intermediate rest phases.",
    },
    "food_progress": {
        "category": "progress",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["hunger above threshold", "not already on food"],
        "config_keys": ["food_progress"],
        "configured_weight_keys": ["food_progress"],
        "shaping_risk": "high",
        "shaping_disposition": "removed",
        "disposition_rationale": (
            "Zeroed in the austere profile so food seeking must be supported by sparse "
            "feeding outcomes and learned memory or planning rather than distance gradients."
        ),
        "notes": "Approach reward for food; reduces the need for long-horizon temporal credit assignment.",
    },
    "shelter_progress": {
        "category": "progress",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["fatigue or sleep_debt high, or night", "not already in deep shelter", "extra threat branch"],
        "config_keys": ["shelter_progress", "threat_shelter_progress"],
        "configured_weight_keys": ["shelter_progress", "threat_shelter_progress"],
        "shaping_risk": "high",
        "shaping_disposition": "removed",
        "disposition_rationale": (
            "Zeroed in the austere profile, including the threat branch, to expose whether "
            "shelter behavior survives without distance-to-shelter guidance."
        ),
        "notes": "Provides an explicit direction toward shelter even without full perception or planning.",
    },
    "predator_escape": {
        "category": "progress",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["predator visible or recent contact", "distance gain and escape bonus branches"],
        "config_keys": [
            "predator_escape",
            "predator_escape_bonus",
            "predator_escape_stay_penalty",
        ],
        "configured_weight_keys": [
            "predator_escape",
            "predator_escape_bonus",
            "predator_escape_stay_penalty",
        ],
        "shaping_risk": "high",
        "shaping_disposition": "removed",
        "disposition_rationale": (
            "Zeroed in the austere profile, including escape bonus and stay penalty, so "
            "survival must come from contact and exposure outcomes plus predator dynamics."
        ),
        "notes": "Incremental reward for increasing distance from the predator and a penalty for remaining exposed and still.",
    },
    "day_exploration": {
        "category": "progress",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["moved", "not night", "predator branch inactive"],
        "config_keys": ["day_exploration_hungry", "day_exploration_calm"],
        "configured_weight_keys": ["day_exploration_hungry", "day_exploration_calm"],
        "shaping_risk": "medium",
        "shaping_disposition": "removed",
        "disposition_rationale": (
            "Zeroed in the austere profile because it is auxiliary locomotion guidance "
            "rather than an outcome signal."
        ),
        "notes": "Auxiliary bonus for daytime exploration; explicit and not ecological on its own.",
    },
    "shelter_entry": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["crossed into shelter this step"],
        "config_keys": ["shelter_entry"],
        "configured_weight_keys": ["shelter_entry"],
        "shaping_risk": "medium",
        "shaping_disposition": "removed",
        "disposition_rationale": (
            "Zeroed in the austere profile because crossing the shelter boundary still "
            "provides navigational shaping before sustained rest or safety outcomes."
        ),
        "notes": "Clear event, but it still guides navigation toward the shelter edge.",
    },
    "night_shelter_bonus": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["night", "on shelter"],
        "config_keys": ["night_shelter_bonus"],
        "configured_weight_keys": ["night_shelter_bonus"],
        "shaping_risk": "medium",
        "shaping_disposition": "removed",
        "disposition_rationale": (
            "Zeroed in the austere profile because recurrent shelter occupancy reward can "
            "mask whether the agent learned night-shelter behavior from consequences."
        ),
        "notes": "Recurring bonus for occupying shelter at night.",
    },
    "homeostasis_penalty": {
        "category": "internal_pressure",
        "source": "spider_cortex_sim.physiology.apply_homeostasis_penalties",
        "gates": ["hunger/fatigue/sleep_debt above thresholds"],
        "config_keys": ["sleep_debt_overdue_threshold", "sleep_debt_health_penalty"],
        "configured_weight_keys": ["sleep_debt_health_penalty"],
        "hardcoded_weight": 0.72,
        "shaping_risk": "medium",
        "shaping_disposition": "defended",
        "disposition_rationale": (
            "Retained and strengthened in the austere profile as physiological deterioration; "
            "unlike progress rewards, it penalizes bad internal state rather than instructing a destination."
        ),
        "notes": "Penalizes progressive physiological degradation; part of the weight is hardcoded.",
    },
    "death_penalty": {
        "category": "event",
        "source": "spider_cortex_sim.world.SpiderWorld.step",
        "gates": ["health <= 0"],
        "config_keys": ["death_penalty"],
        "configured_weight_keys": ["death_penalty"],
        "shaping_risk": "low",
        "shaping_disposition": "outcome_signal",
        "disposition_rationale": (
            "Terminal episode failure anchors a negative outcome signal for death rather than "
            "dense behavioral guidance."
        ),
        "notes": "Explicit and rare terminal event.",
    },
}


def reward_component_audit() -> dict[str, dict[str, object]]:
    """
    Provide a defensive copy of the reward-component audit catalog.
    
    The catalog maps each reward component name to metadata describing its category
    (`event`, `progress`, or `internal_pressure`), source code locations, activation
    gates, which reward-profile keys influence its weight, and any hardcoded weight.
    
    Returns:
        dict[str, dict[str, object]]: A deep copy of `REWARD_COMPONENT_AUDIT` where
        each key is a reward component name and each value is its audit metadata.
    """
    return deepcopy(REWARD_COMPONENT_AUDIT)


def shaping_disposition_summary() -> dict[str, object]:
    """
    Group reward components by shaping disposition.
    
    Returns a mapping that contains, for each disposition in SHAPING_DISPOSITIONS, a list of component names assigned that disposition, plus a "counts" entry mapping each disposition to its integer component count.
    
    Returns:
        dict[str, object]: Mapping of disposition -> list[str] and "counts" -> dict[str, int].
    
    Raises:
        ValueError: If any component's shaping disposition is invalid according to validate_shaping_disposition.
    """
    summary: dict[str, list[str]] = {
        disposition: []
        for disposition in SHAPING_DISPOSITIONS
    }
    for component_name, metadata in sorted(REWARD_COMPONENT_AUDIT.items()):
        disposition = validate_shaping_disposition(metadata, component_name)
        summary[disposition].append(component_name)
    return {
        **summary,
        "counts": {
            disposition: len(summary[disposition])
            for disposition in SHAPING_DISPOSITIONS
        },
    }


def _roadmap_status_for_profile(profile: Mapping[str, float]) -> dict[str, dict[str, object]]:
    """
    Summarizes configurable and hardcoded weight proxies and activity status for each shaping-reduction roadmap target.
    
    Parameters:
    	profile (Mapping[str, float]): Mapping of profile configuration keys to numeric values used as weight inputs.
    
    Returns:
    	status_map (dict[str, dict[str, object]]): Mapping from roadmap target component name to a status dictionary containing:
    		- current_disposition (str): the component's current shaping disposition from the roadmap.
    		- target_disposition (str): the roadmap's target disposition for the component.
    		- reduction_priority (int | float): roadmap-specified priority for reduction.
    		- configured_weight_keys (list[str]): configured weight keys present in `profile`.
    		- active_configured_weight_keys (list[str]): subset of configured keys whose proxy weight is greater than 0.0.
    		- has_configurable_weight (bool): whether any configured weight keys exist for the component.
    		- configured_weight_proxy (float): summed absolute proxy of configured weights (rounded to 6 decimals).
    		- hardcoded_weight (float): absolute hardcoded weight from component metadata (rounded to 6 decimals).
    		- total_weight_proxy (float): sum of configured_weight_proxy and hardcoded_weight (rounded to 6 decimals).
    		- weight_status (str): one of "active_configurable", "hardcoded_only", or "zeroed" indicating the component's current weight activation state.
    """
    status: dict[str, dict[str, object]] = {}
    for component_name, roadmap_entry in sorted(SHAPING_REDUCTION_ROADMAP.items()):
        metadata = REWARD_COMPONENT_AUDIT.get(component_name, {})
        configured_keys = [
            str(key)
            for key in metadata.get("configured_weight_keys", [])
            if str(key) in profile
        ]
        configured_weights = {
            key: abs(policy_float(profile.get(key)))
            for key in configured_keys
        }
        active_configured_keys = sorted(
            key
            for key, value in configured_weights.items()
            if value > 0.0
        )
        configured_weight_proxy = sum(configured_weights.values())
        hardcoded_weight = abs(policy_float(metadata.get("hardcoded_weight", 0.0)))
        if active_configured_keys:
            weight_status = "active_configurable"
        elif hardcoded_weight > 0.0:
            weight_status = "hardcoded_only"
        else:
            weight_status = "zeroed"
        status[component_name] = {
            "current_disposition": roadmap_entry["current_disposition"],
            "target_disposition": roadmap_entry["target_disposition"],
            "reduction_priority": roadmap_entry["reduction_priority"],
            "configured_weight_keys": configured_keys,
            "active_configured_weight_keys": active_configured_keys,
            "has_configurable_weight": bool(configured_keys),
            "configured_weight_proxy": round(float(configured_weight_proxy), 6),
            "hardcoded_weight": round(float(hardcoded_weight), 6),
            "total_weight_proxy": round(
                float(configured_weight_proxy + hardcoded_weight),
                6,
            ),
            "weight_status": weight_status,
        }
    return status


def reward_profile_audit(profile_name: str) -> dict[str, object]:
    """
    Produce a human-readable audit of the named reward profile summarizing per-component and per-category weight proxies and shaping dispositions.
    
    Parameters:
        profile_name (str): Name of a profile defined in REWARD_PROFILES.
    
    Returns:
        dict[str, object]: Audit mapping including:
            - "profile": the requested profile name.
            - "component_weight_proxy": mapping of component name -> proxy weight (float).
            - "category_weight_proxy": mapping of category -> total proxy mass (float).
            - "configurable_category_weight_proxy": mapping of category -> configurable-only proxy mass (float).
            - "disposition_summary": mapping of shaping disposition -> {"components": [str], "component_count": int, "total_weight_proxy": float}.
            - "reduction_roadmap_status": mapping of reduction target -> configurable weight status for this profile.
            - "dominant_category": category name chosen as dominant (str).
            - "hardcoded_mass": total hardcoded weight summed across components (float).
            - "non_configurable_components": sorted list of component names without configurable weight keys.
            - "notes": list of explanatory strings.
    
    Raises:
        KeyError: If `profile_name` is not present in REWARD_PROFILES.
        ValueError: If any reward component has an invalid `shaping_disposition`.
    """
    if profile_name not in REWARD_PROFILES:
        raise KeyError(f"Unknown reward profile: {profile_name!r}.")
    profile = REWARD_PROFILES[profile_name]
    category_weights = {
        "event": 0.0,
        "progress": 0.0,
        "internal_pressure": 0.0,
    }
    configurable_category_weights = {
        "event": 0.0,
        "progress": 0.0,
        "internal_pressure": 0.0,
    }
    component_weights: dict[str, float] = {}
    disposition_summary: dict[str, dict[str, object]] = {
        disposition: {
            "components": [],
            "component_count": 0,
            "total_weight_proxy": 0.0,
        }
        for disposition in SHAPING_DISPOSITIONS
    }
    non_configurable_components: list[str] = []
    hardcoded_mass = 0.0
    for component_name, metadata in REWARD_COMPONENT_AUDIT.items():
        configured_weight = sum(
            abs(policy_float(profile.get(key)))
            for key in metadata.get("configured_weight_keys", [])
        )
        hardcoded_weight = policy_float(metadata.get("hardcoded_weight", 0.0))
        proxy_weight = configured_weight + hardcoded_weight
        component_weights[component_name] = round(proxy_weight, 6)
        category_name = str(metadata["category"])
        category_weights[category_name] += proxy_weight
        configurable_category_weights[category_name] += configured_weight
        hardcoded_mass += hardcoded_weight
        disposition = validate_shaping_disposition(metadata, component_name)
        disposition_data = disposition_summary[disposition]
        disposition_data["components"].append(component_name)
        disposition_data["component_count"] += 1
        disposition_data["total_weight_proxy"] += proxy_weight
        if not metadata.get("configured_weight_keys"):
            non_configurable_components.append(component_name)
    dominant_weight_source = configurable_category_weights
    if not any(value > 0.0 for value in dominant_weight_source.values()):
        dominant_weight_source = category_weights
    dominant_category = max(
        dominant_weight_source.items(),
        key=lambda item: item[1],
    )[0]
    return {
        "profile": profile_name,
        "component_weight_proxy": component_weights,
        "category_weight_proxy": {
            name: round(float(value), 6)
            for name, value in sorted(category_weights.items())
        },
        "configurable_category_weight_proxy": {
            name: round(float(value), 6)
            for name, value in sorted(configurable_category_weights.items())
        },
        "disposition_summary": {
            disposition: {
                "components": sorted(data["components"]),
                "component_count": data["component_count"],
                "total_weight_proxy": round(float(data["total_weight_proxy"]), 6),
            }
            for disposition, data in disposition_summary.items()
        },
        "reduction_roadmap_status": _roadmap_status_for_profile(profile),
        "dominant_category": dominant_category,
        "hardcoded_mass": round(float(hardcoded_mass), 6),
        "non_configurable_components": sorted(non_configurable_components),
        "notes": [
            "The weight proxy sums configurable per-component coefficients plus a few stable hardcoded weights.",
            "dominant_category prioritizes configurable mass only; when that is zero, it falls back to total mass.",
            "It is meant to compare how much each profile still depends on explicit shaping, not to reproduce total reward per episode.",
        ],
    }
