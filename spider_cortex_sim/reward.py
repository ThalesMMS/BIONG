from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Sequence

from .maps import CLUTTER, NARROW

if TYPE_CHECKING:
    from .world import SpiderWorld
    from .world_types import TickContext


REWARD_COMPONENT_NAMES: Sequence[str] = (
    "action_cost",
    "terrain_cost",
    "night_exposure",
    "hunger_pressure",
    "fatigue_pressure",
    "sleep_debt_pressure",
    "predator_contact",
    "feeding",
    "resting",
    "food_progress",
    "shelter_progress",
    "predator_escape",
    "day_exploration",
    "shelter_entry",
    "night_shelter_bonus",
    "homeostasis_penalty",
    "death_penalty",
)

REWARD_PROFILES = {
    "classic": {
        "action_cost_stay": 0.008,
        "action_cost_move": 0.014,
        "base_hunger_cost": 0.010,
        "move_hunger_cost": 0.006,
        "idle_hunger_cost": 0.002,
        "base_fatigue_cost": 0.007,
        "move_fatigue_cost": 0.006,
        "idle_fatigue_cost": 0.002,
        "idle_open_penalty": 0.020,
        "clutter_fatigue": 0.004,
        "clutter_cost": 0.012,
        "narrow_fatigue": 0.003,
        "narrow_cost": 0.010,
        "narrow_predator_risk": 0.035,
        "night_exposure_fatigue": 0.012,
        "night_exposure_debt": 0.018,
        "night_exposure_reward": 0.060,
        "hunger_pressure": 0.13,
        "fatigue_pressure": 0.08,
        "sleep_debt_pressure": 0.05,
        "food_progress": 0.30,
        "shelter_progress": 0.16,
        "threat_shelter_progress": 0.08,
        "predator_escape": 0.20,
        "predator_escape_bonus": 0.18,
        "predator_escape_stay_penalty": 0.08,
        "day_exploration_hungry": 0.05,
        "day_exploration_calm": 0.01,
        "shelter_entry": 0.10,
        "night_shelter_bonus": 0.08,
        "sleep_debt_night_awake": 0.014,
        "sleep_debt_day_awake": 0.003,
        "sleep_debt_interrupt": 0.070,
        "sleep_debt_resting_night": 0.14,
        "sleep_debt_resting_day": 0.07,
        "sleep_debt_deep_night": 0.22,
        "sleep_debt_deep_day": 0.10,
        "sleep_debt_overdue_threshold": 0.72,
        "sleep_debt_health_penalty": 0.24,
    },
    "ecological": {
        "action_cost_stay": 0.010,
        "action_cost_move": 0.016,
        "base_hunger_cost": 0.010,
        "move_hunger_cost": 0.006,
        "idle_hunger_cost": 0.002,
        "base_fatigue_cost": 0.007,
        "move_fatigue_cost": 0.006,
        "idle_fatigue_cost": 0.002,
        "idle_open_penalty": 0.028,
        "clutter_fatigue": 0.006,
        "clutter_cost": 0.020,
        "narrow_fatigue": 0.005,
        "narrow_cost": 0.015,
        "narrow_predator_risk": 0.060,
        "night_exposure_fatigue": 0.018,
        "night_exposure_debt": 0.028,
        "night_exposure_reward": 0.100,
        "hunger_pressure": 0.16,
        "fatigue_pressure": 0.10,
        "sleep_debt_pressure": 0.10,
        "food_progress": 0.04,
        "shelter_progress": 0.03,
        "threat_shelter_progress": 0.03,
        "predator_escape": 0.12,
        "predator_escape_bonus": 0.08,
        "predator_escape_stay_penalty": 0.12,
        "day_exploration_hungry": 0.01,
        "day_exploration_calm": 0.0,
        "shelter_entry": 0.02,
        "night_shelter_bonus": 0.0,
        "sleep_debt_night_awake": 0.022,
        "sleep_debt_day_awake": 0.004,
        "sleep_debt_interrupt": 0.100,
        "sleep_debt_resting_night": 0.18,
        "sleep_debt_resting_day": 0.08,
        "sleep_debt_deep_night": 0.28,
        "sleep_debt_deep_day": 0.12,
        "sleep_debt_overdue_threshold": 0.65,
        "sleep_debt_health_penalty": 0.34,
    },
    "austere": {
        "action_cost_stay": 0.010,
        "action_cost_move": 0.016,
        "base_hunger_cost": 0.010,
        "move_hunger_cost": 0.006,
        "idle_hunger_cost": 0.002,
        "base_fatigue_cost": 0.007,
        "move_fatigue_cost": 0.006,
        "idle_fatigue_cost": 0.002,
        "idle_open_penalty": 0.030,
        "clutter_fatigue": 0.006,
        "clutter_cost": 0.020,
        "narrow_fatigue": 0.005,
        "narrow_cost": 0.016,
        "narrow_predator_risk": 0.065,
        "night_exposure_fatigue": 0.020,
        "night_exposure_debt": 0.030,
        "night_exposure_reward": 0.110,
        "hunger_pressure": 0.18,
        "fatigue_pressure": 0.11,
        "sleep_debt_pressure": 0.12,
        "food_progress": 0.0,
        "shelter_progress": 0.0,
        "threat_shelter_progress": 0.0,
        "predator_escape": 0.0,
        "predator_escape_bonus": 0.0,
        "predator_escape_stay_penalty": 0.0,
        "day_exploration_hungry": 0.0,
        "day_exploration_calm": 0.0,
        "shelter_entry": 0.0,
        "night_shelter_bonus": 0.0,
        "sleep_debt_night_awake": 0.024,
        "sleep_debt_day_awake": 0.005,
        "sleep_debt_interrupt": 0.120,
        "sleep_debt_resting_night": 0.16,
        "sleep_debt_resting_day": 0.07,
        "sleep_debt_deep_night": 0.24,
        "sleep_debt_deep_day": 0.10,
        "sleep_debt_overdue_threshold": 0.60,
        "sleep_debt_health_penalty": 0.38,
    },
}


REWARD_COMPONENT_AUDIT = {
    "action_cost": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_action_and_terrain_effects",
        "gates": ["always", "scaled by action_name == STAY vs locomotion"],
        "config_keys": ["action_cost_stay", "action_cost_move"],
        "configured_weight_keys": ["action_cost_stay", "action_cost_move"],
        "shaping_risk": "low",
        "notes": "Penaliza cada passo; funciona mais como custo universal do que como trilha guiada.",
    },
    "terrain_cost": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_action_and_terrain_effects",
        "gates": ["terrain == CLUTTER or NARROW", "extra branch when NARROW and predator is close"],
        "config_keys": ["clutter_cost", "narrow_cost", "narrow_predator_risk"],
        "configured_weight_keys": ["clutter_cost", "narrow_cost", "narrow_predator_risk"],
        "shaping_risk": "medium",
        "notes": "Encodes explicit environmental cost and pushes the agent away from dangerous bottlenecks.",
    },
    "night_exposure": {
        "category": "event",
        "source": "spider_cortex_sim.world.SpiderWorld.step",
        "gates": ["night", "not on shelter"],
        "config_keys": ["night_exposure_reward"],
        "configured_weight_keys": ["night_exposure_reward"],
        "shaping_risk": "medium",
        "notes": "Dense penalty for nighttime exposure outside shelter.",
    },
    "hunger_pressure": {
        "category": "internal_pressure",
        "source": "spider_cortex_sim.reward.apply_pressure_penalties",
        "gates": ["always, scaled by state.hunger"],
        "config_keys": ["hunger_pressure"],
        "configured_weight_keys": ["hunger_pressure"],
        "shaping_risk": "medium",
        "notes": "Continuous homeostatic pressure rather than a terminal event.",
    },
    "fatigue_pressure": {
        "category": "internal_pressure",
        "source": "spider_cortex_sim.reward.apply_pressure_penalties",
        "gates": ["always, scaled by state.fatigue"],
        "config_keys": ["fatigue_pressure"],
        "configured_weight_keys": ["fatigue_pressure"],
        "shaping_risk": "medium",
        "notes": "Continuous internal pressure to seek rest.",
    },
    "sleep_debt_pressure": {
        "category": "internal_pressure",
        "source": "spider_cortex_sim.reward.apply_pressure_penalties",
        "gates": ["always, scaled by state.sleep_debt"],
        "config_keys": ["sleep_debt_pressure"],
        "configured_weight_keys": ["sleep_debt_pressure"],
        "shaping_risk": "medium",
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
        "notes": "Combines a legitimate ecological event with dense shaping for intermediate rest phases.",
    },
    "food_progress": {
        "category": "progress",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["hunger above threshold", "not already on food"],
        "config_keys": ["food_progress"],
        "configured_weight_keys": ["food_progress"],
        "shaping_risk": "high",
        "notes": "Approach reward for food; reduces the need for long-horizon temporal credit assignment.",
    },
    "shelter_progress": {
        "category": "progress",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["fatigue or sleep_debt high, or night", "not already in deep shelter", "extra threat branch"],
        "config_keys": ["shelter_progress", "threat_shelter_progress"],
        "configured_weight_keys": ["shelter_progress", "threat_shelter_progress"],
        "shaping_risk": "high",
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
        "notes": "Incremental reward for increasing distance from the predator and a penalty for remaining exposed and still.",
    },
    "day_exploration": {
        "category": "progress",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["moved", "not night", "predator branch inactive"],
        "config_keys": ["day_exploration_hungry", "day_exploration_calm"],
        "configured_weight_keys": ["day_exploration_hungry", "day_exploration_calm"],
        "shaping_risk": "medium",
        "notes": "Auxiliary bonus for daytime exploration; explicit and not ecological on its own.",
    },
    "shelter_entry": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["crossed into shelter this step"],
        "config_keys": ["shelter_entry"],
        "configured_weight_keys": ["shelter_entry"],
        "shaping_risk": "medium",
        "notes": "Clear event, but it still guides navigation toward the shelter edge.",
    },
    "night_shelter_bonus": {
        "category": "event",
        "source": "spider_cortex_sim.reward.apply_progress_and_event_rewards",
        "gates": ["night", "on shelter"],
        "config_keys": ["night_shelter_bonus"],
        "configured_weight_keys": ["night_shelter_bonus"],
        "shaping_risk": "medium",
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
        "notes": "Penalizes progressive physiological degradation; part of the weight is hardcoded.",
    },
    "death_penalty": {
        "category": "event",
        "source": "spider_cortex_sim.world.SpiderWorld.step",
        "gates": ["health <= 0"],
        "config_keys": [],
        "configured_weight_keys": [],
        "hardcoded_weight": 5.0,
        "shaping_risk": "low",
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


def reward_profile_audit(profile_name: str) -> dict[str, object]:
    """
    Produce a human-readable audit summary for the named reward profile.
    
    The result reports a per-component "weight proxy" (sum of absolute configured weights from the profile plus any hardcoded weight), aggregated category proxies for both configurable-only and total mass, the chosen dominant category (preferring configurable mass when present), a list of components that have no configurable weight keys, and the total hardcoded mass. Intended for diagnostics and profile comparison rather than exact reward decomposition.
    
    Parameters:
        profile_name (str): Name of the reward profile to audit.
    
    Returns:
        dict[str, object]: An audit mapping containing:
            - "profile": the requested profile name.
            - "component_weight_proxy": mapping of component -> proxy weight (float).
            - "category_weight_proxy": mapping of category -> total proxy mass (float).
            - "configurable_category_weight_proxy": mapping of category -> configurable-only proxy mass (float).
            - "dominant_category": category with the largest chosen proxy mass.
            - "hardcoded_mass": total hardcoded weight summed across components (float).
            - "non_configurable_components": sorted list of component names without configurable weight keys.
            - "notes": list of explanatory strings.
    
    Raises:
        KeyError: If `profile_name` is not a key in REWARD_PROFILES.
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
    non_configurable_components: list[str] = []
    hardcoded_mass = 0.0
    for component_name, metadata in REWARD_COMPONENT_AUDIT.items():
        configured_weight = sum(
            abs(float(profile[key]))
            for key in metadata.get("configured_weight_keys", [])
        )
        hardcoded_weight = float(metadata.get("hardcoded_weight", 0.0))
        proxy_weight = configured_weight + hardcoded_weight
        component_weights[component_name] = round(proxy_weight, 6)
        category_name = str(metadata["category"])
        category_weights[category_name] += proxy_weight
        configurable_category_weights[category_name] += configured_weight
        hardcoded_mass += hardcoded_weight
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
        "dominant_category": dominant_category,
        "hardcoded_mass": round(float(hardcoded_mass), 6),
        "non_configurable_components": sorted(non_configurable_components),
        "notes": [
            "The weight proxy sums configurable per-component coefficients plus a few stable hardcoded weights.",
            "dominant_category prioritizes configurable mass only; when that is zero, it falls back to total mass.",
            "It is meant to compare how much each profile still depends on explicit shaping, not to reproduce total reward per episode.",
        ],
    }


def empty_reward_components() -> dict[str, float]:
    """
    Create a mapping of every defined reward component name to 0.0.
    
    Returns:
        dict[str, float]: A dictionary whose keys are the names from REWARD_COMPONENT_NAMES and whose values are all 0.0.
    """
    return {name: 0.0 for name in REWARD_COMPONENT_NAMES}


def reward_total(reward_components: dict[str, float]) -> float:
    """
    Compute the scalar total of all reward component values.
    
    Parameters:
        reward_components (dict[str, float]): Mapping from reward component names to their numeric values.
    
    Returns:
        float: Sum of all values in `reward_components`.
    """
    return float(sum(reward_components.values()))


def copy_reward_components(reward_components: dict[str, float]) -> dict[str, float]:
    """
    Create a new dictionary containing only the defined reward component keys with their values as floats.
    
    Parameters:
        reward_components (dict[str, float]): Mapping of reward component names to numeric values; keys not in REWARD_COMPONENT_NAMES will be ignored.
    
    Returns:
        dict[str, float]: A new dictionary mapping each name in REWARD_COMPONENT_NAMES to its value converted to `float`.
    """
    return {name: float(reward_components[name]) for name in REWARD_COMPONENT_NAMES}


def apply_action_and_terrain_effects(
    world: "SpiderWorld",
    *,
    action_name: str,
    moved: bool,
    reward_components: dict[str, float],
) -> str:
    """
    Apply action- and terrain-related updates to the world's state and mutate reward components accordingly.
    
    This updates reward_components (deducting the configured action cost and any terrain cost), increments world.state.hunger and world.state.fatigue by per-step amounts, and applies additional fatigue and terrain penalties when the current terrain is CLUTTER or NARROW.
    
    Parameters:
        world (SpiderWorld): The simulation world whose state and reward_config are used and mutated.
        action_name (str): The action performed this step; "STAY" applies the configured stay cost, other values apply the move cost.
        moved (bool): Whether the agent moved this step; influences the per-step physiology increments.
        reward_components (dict[str, float]): Mutable mapping of reward component names to update (e.g., "action_cost", "terrain_cost").
    
    Returns:
        str: The terrain at the spider's current position (e.g., CLUTTER, NARROW, or other terrain identifiers).
    """
    cfg = world.reward_config
    if action_name == "STAY":
        reward_components["action_cost"] -= cfg["action_cost_stay"]
    else:
        reward_components["action_cost"] -= cfg["action_cost_move"]

    terrain_now = world.terrain_at(world.spider_pos())
    if terrain_now == CLUTTER:
        world.state.fatigue += cfg["clutter_fatigue"]
        reward_components["terrain_cost"] -= cfg["clutter_cost"]
    elif terrain_now == NARROW:
        world.state.fatigue += cfg["narrow_fatigue"]
        reward_components["terrain_cost"] -= cfg["narrow_cost"]

    world.state.hunger += cfg["base_hunger_cost"] + (cfg["move_hunger_cost"] if moved else cfg["idle_hunger_cost"])
    world.state.fatigue += cfg["base_fatigue_cost"] + (cfg["move_fatigue_cost"] if moved else cfg["idle_fatigue_cost"])
    return terrain_now


def compute_predator_threat(
    world: "SpiderWorld",
    *,
    prev_predator_visible: bool,
    prev_predator_dist: int,
) -> bool:
    """
    Determine whether the predator currently poses a threat to the spider.
    
    Checks the spider's recent-contact and recent-pain state against the operational-profile threat thresholds, whether the predator was previously visible, whether the previous predator distance is within the profile distance threshold, and whether the current predator smell strength exceeds the profile smell threshold.
    
    Returns:
        `true` if any threat condition is met, `false` otherwise.
    """
    from .perception import smell_gradient

    cfg = world.operational_profile.reward
    predator_smell_strength_now, _, _, _ = smell_gradient(
        world,
        [world.lizard_pos()],
        radius=world.predator_smell_range,
        apply_noise=False,
    )
    return (
        world.state.recent_contact > cfg["predator_threat_contact_threshold"]
        or world.state.recent_pain > cfg["predator_threat_recent_pain_threshold"]
        or prev_predator_visible
        or prev_predator_dist <= cfg["predator_threat_distance_threshold"]
        or predator_smell_strength_now > cfg["predator_threat_smell_threshold"]
    )


def apply_pressure_penalties(world: "SpiderWorld", reward_components: dict[str, float]) -> None:
    """
    Apply physiological pressure penalties to the mutable reward components based on the world state.
    
    Subtracts hunger, fatigue, and sleep-debt penalties (each scaled by the corresponding config value) from the `hunger_pressure`, `fatigue_pressure`, and `sleep_debt_pressure` entries of `reward_components` in place.
    
    Parameters:
        world: The simulation world providing `reward_config` and `state` with `hunger`, `fatigue`, and `sleep_debt` attributes.
        reward_components (dict[str, float]): Mutable mapping of reward component names to values; `hunger_pressure`, `fatigue_pressure`, and `sleep_debt_pressure` are modified.
    """
    cfg = world.reward_config
    reward_components["hunger_pressure"] -= cfg["hunger_pressure"] * world.state.hunger
    reward_components["fatigue_pressure"] -= cfg["fatigue_pressure"] * world.state.fatigue
    reward_components["sleep_debt_pressure"] -= cfg["sleep_debt_pressure"] * world.state.sleep_debt


def apply_progress_and_event_rewards(
    world: "SpiderWorld",
    *,
    tick_context: "TickContext",
) -> None:
    """
    Update reward components and world event counters based on movement, distances to food/shelter/predator, terrain, and physiology.
    
    This function mutates `reward_components` and `world.state` to record progress rewards, penalties, exploration bonuses, shelter-entry and predator-related events; it also populates `info["distance_deltas"]` with integer deltas for `"food"`, `"shelter"`, and `"predator"`. It may call physiology reset logic when predator proximity or recent contact/pain thresholds are met.
    
    Parameters:
        world: The simulation world object whose state and configuration are read and updated.
        tick_context (TickContext): Tick-level context carrying the pre-tick snapshot, mutable reward
            accumulator, info payload, and event log to be updated in place.
    """
    from .perception import predator_visible_to_spider
    from .physiology import reset_sleep_state

    snapshot = tick_context.snapshot
    action_name = tick_context.executed_action
    moved = tick_context.moved
    night = snapshot.night
    terrain_now = tick_context.terrain_now
    was_on_shelter = snapshot.was_on_shelter
    prev_food_dist = snapshot.prev_food_dist
    prev_shelter_dist = snapshot.prev_shelter_dist
    prev_predator_dist = snapshot.prev_predator_dist
    prev_predator_visible = snapshot.prev_predator_visible
    reward_components = tick_context.reward_components
    info = tick_context.info
    cfg = world.reward_config
    profile = world.operational_profile.reward

    _, new_food_dist = world.nearest(world.food_positions)
    _, new_shelter_dist = world.nearest(world.shelter_deep_cells or world.shelter_cells)
    new_predator_dist = world.manhattan(world.spider_pos(), world.lizard_pos())
    food_progress = float(prev_food_dist - new_food_dist)
    shelter_progress = float(prev_shelter_dist - new_shelter_dist)
    predator_distance_gain = float(new_predator_dist - prev_predator_dist)

    info["distance_deltas"] = {
        "food": int(prev_food_dist - new_food_dist),
        "shelter": int(prev_shelter_dist - new_shelter_dist),
        "predator": int(new_predator_dist - prev_predator_dist),
    }
    tick_context.record_event(
        "reward",
        "distance_deltas",
        food=int(food_progress),
        shelter=int(shelter_progress),
        predator=int(predator_distance_gain),
    )

    if (
        terrain_now == NARROW
        and not world.on_shelter()
        and new_predator_dist <= profile["narrow_predator_risk_max_distance"]
    ):
        max_distance = profile["narrow_predator_risk_max_distance"]
        risk = ((max_distance + 1.0) - new_predator_dist) / max_distance
        reward_components["terrain_cost"] -= cfg["narrow_predator_risk"] * risk
        tick_context.record_event(
            "reward",
            "narrow_predator_risk",
            risk=round(float(risk), 6),
            predator_distance=int(new_predator_dist),
        )

    if world.state.hunger > profile["food_progress_hunger_threshold"] and not world.on_food():
        reward_components["food_progress"] += (
            cfg["food_progress"]
            * food_progress
            * (profile["food_progress_hunger_bias"] + world.state.hunger)
        )
    if (
        world.state.fatigue > profile["shelter_progress_fatigue_threshold"]
        or night
        or world.state.sleep_debt > profile["shelter_progress_sleep_debt_threshold"]
    ) and not world.deep_shelter():
        reward_components["shelter_progress"] += (
            cfg["shelter_progress"]
            * shelter_progress
            * (
                profile["shelter_progress_bias"]
                + world.state.fatigue
                + profile["shelter_progress_sleep_debt_weight"] * world.state.sleep_debt
            )
        )
    if prev_predator_visible or world.state.recent_contact > profile["predator_escape_contact_threshold"]:
        reward_components["predator_escape"] += cfg["predator_escape"] * predator_distance_gain
        reward_components["shelter_progress"] += cfg["threat_shelter_progress"] * shelter_progress
        if action_name == "STAY" and not world.on_shelter():
            reward_components["predator_escape"] -= cfg["predator_escape_stay_penalty"]
    elif moved and not night:
        day_bonus = (
            cfg["day_exploration_hungry"]
            if world.state.hunger > profile["day_exploration_hunger_threshold"]
            else cfg["day_exploration_calm"]
        )
        reward_components["day_exploration"] += day_bonus

    on_shelter_now = world.on_shelter()
    if on_shelter_now and not was_on_shelter:
        world.state.shelter_entries += 1
        reward_components["shelter_entry"] += cfg["shelter_entry"]
        tick_context.record_event(
            "reward",
            "shelter_entry",
            shelter_role=world.shelter_role_at(world.spider_pos()),
        )
    if night and on_shelter_now:
        reward_components["night_shelter_bonus"] += cfg["night_shelter_bonus"] * world.shelter_role_level()

    predator_visible_now = (
        predator_visible_to_spider(world).visible > profile["predator_visibility_threshold"]
    )
    tick_context.predator_visible_now = bool(predator_visible_now)
    predator_escape = False
    escape_context_active = bool(
        prev_predator_visible
        or world.state.recent_contact > profile["predator_escape_contact_threshold"]
    )
    threat_episode_active_now = bool(
        escape_context_active
        or predator_visible_now
        or world.state.recent_pain > profile["reset_sleep_recent_pain_threshold"]
    )
    if threat_episode_active_now and not world._predator_threat_episode_active:
        world._predator_escape_bonus_pending = True
    elif not threat_episode_active_now:
        world._predator_escape_bonus_pending = False
    world._predator_threat_episode_active = threat_episode_active_now
    if predator_visible_now:
        world.state.alert_events += 1
        world.state.predator_sightings += 1
        tick_context.record_event(
            "reward",
            "predator_sighting",
            predator_distance=int(new_predator_dist),
        )
    if world._predator_escape_bonus_pending and escape_context_active and (
        world.inside_shelter()
        or (new_predator_dist - prev_predator_dist) >= profile["predator_escape_distance_gain_threshold"]
    ):
        world.state.predator_escapes += 1
        reward_components["predator_escape"] += cfg["predator_escape_bonus"]
        predator_escape = True
        world._predator_escape_bonus_pending = False
        tick_context.record_event(
            "reward",
            "predator_escape",
            predator_distance_gain=round(float(predator_distance_gain), 6),
            inside_shelter=bool(world.inside_shelter()),
        )

    if (
        new_predator_dist <= profile["reset_sleep_predator_distance_threshold"]
        or world.state.recent_contact > profile["reset_sleep_contact_threshold"]
        or world.state.recent_pain > profile["reset_sleep_recent_pain_threshold"]
    ):
        reset_sleep_state(world)
        tick_context.record_event(
            "reward",
            "sleep_reset_due_to_threat",
            predator_distance=int(new_predator_dist),
            recent_contact=round(float(world.state.recent_contact), 6),
            recent_pain=round(float(world.state.recent_pain), 6),
        )

    tick_context.predator_escape = bool(predator_escape)
