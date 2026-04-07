from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .maps import CLUTTER, NARROW

if TYPE_CHECKING:
    from .world import SpiderWorld


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
}


def empty_reward_components() -> dict[str, float]:
    """
    Create a dictionary mapping every reward component name to 0.0.
    
    Returns:
        dict[str, float]: A dictionary with each key from REWARD_COMPONENT_NAMES initialized to 0.0.
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
    action_name: str,
    moved: bool,
    night: bool,
    terrain_now: str,
    was_on_shelter: bool,
    prev_food_dist: int,
    prev_shelter_dist: int,
    prev_predator_dist: int,
    prev_predator_visible: bool,
    reward_components: dict[str, float],
    info: dict[str, object],
) -> tuple[bool, bool]:
    """
    Update reward components and world event counters based on movement, distances to food/shelter/predator, terrain, and physiology.
    
    This function mutates `reward_components` and `world.state` to record progress rewards, penalties, exploration bonuses, shelter-entry and predator-related events; it also populates `info["distance_deltas"]` with integer deltas for `"food"`, `"shelter"`, and `"predator"`. It may call physiology reset logic when predator proximity or recent contact/pain thresholds are met.
    
    Parameters:
        world: The simulation world object whose state and configuration are read and updated.
        action_name (str): The action taken this step (e.g., "STAY", movement actions).
        moved (bool): Whether the agent changed position this step.
        night (bool): Whether it is currently night.
        terrain_now (str): Terrain label at the agent's current position.
        was_on_shelter (bool): Whether the agent was on shelter at the start of the step.
        prev_food_dist (int): Previous step's nearest food distance.
        prev_shelter_dist (int): Previous step's nearest shelter distance.
        prev_predator_dist (int): Previous step's predator (lizard) Manhattan distance.
        prev_predator_visible (bool): Whether the predator was visible in the previous step.
        reward_components (dict[str, float]): Mutable mapping of reward component names to values; updated in-place.
        info (dict[str, object]): Mutable info dictionary; `info["distance_deltas"]` will be set to the integer deltas for food, shelter, and predator.
    
    Returns:
        tuple[bool, bool]: `(predator_escape, predator_visible_now)` where `predator_escape` is `true` if an escape event was counted this step, and `predator_visible_now` is `true` if the predator is currently detected as visible.
    """
    from .perception import predator_visible_to_spider
    from .physiology import reset_sleep_state

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

    if (
        terrain_now == NARROW
        and not world.on_shelter()
        and new_predator_dist <= profile["narrow_predator_risk_max_distance"]
    ):
        max_distance = profile["narrow_predator_risk_max_distance"]
        risk = ((max_distance + 1.0) - new_predator_dist) / max_distance
        reward_components["terrain_cost"] -= cfg["narrow_predator_risk"] * risk

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
    if night and on_shelter_now:
        reward_components["night_shelter_bonus"] += cfg["night_shelter_bonus"] * world.shelter_role_level()

    predator_visible_now = (
        predator_visible_to_spider(world).visible > profile["predator_visibility_threshold"]
    )
    predator_escape = False
    if predator_visible_now:
        world.state.alert_events += 1
        world.state.predator_sightings += 1
    if (prev_predator_visible or world.state.recent_contact > profile["predator_escape_contact_threshold"]) and (
        world.inside_shelter()
        or (new_predator_dist - prev_predator_dist) >= profile["predator_escape_distance_gain_threshold"]
    ):
        world.state.predator_escapes += 1
        reward_components["predator_escape"] += cfg["predator_escape_bonus"]
        predator_escape = True

    if (
        new_predator_dist <= profile["reset_sleep_predator_distance_threshold"]
        or world.state.recent_contact > profile["reset_sleep_contact_threshold"]
        or world.state.recent_pain > profile["reset_sleep_recent_pain_threshold"]
    ):
        reset_sleep_state(world)

    return predator_escape, predator_visible_now
