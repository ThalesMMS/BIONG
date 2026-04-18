"""Runtime reward component computation for simulation ticks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..maps import CLUTTER, NARROW
from ._utils import policy_float
from .profiles import REWARD_COMPONENT_NAMES

if TYPE_CHECKING:
    from ..world import SpiderWorld
    from ..world_types import TickContext

# Backward-compatible alias for existing private imports from the old module.
_policy_float = policy_float


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
    Return a new dictionary with only the names from REWARD_COMPONENT_NAMES, each coerced to float.
    
    Only keys listed in REWARD_COMPONENT_NAMES are included; values are converted using float().
    Missing REWARD_COMPONENT_NAMES entries are filled with 0.0.
    """
    return {name: float(reward_components.get(name, 0.0)) for name in REWARD_COMPONENT_NAMES}


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
    Decides whether current predator conditions constitute a threat to the spider.
    
    Considers recent contact and recent pain levels, prior predator visibility and previous predator distance, and the current predator smell strength against the world's configured threat thresholds.
    
    Returns:
        `True` if the predator is considered a threat, `False` otherwise.
    """
    from ..perception import smell_gradient

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
    Apply physiological pressure penalties to the reward components based on current world state.
    
    Subtracts configured multipliers of the world's `hunger`, `fatigue`, and `sleep_debt` from the
    `hunger_pressure`, `fatigue_pressure`, and `sleep_debt_pressure` entries of `reward_components` in place.
    
    Parameters:
        reward_components (dict[str, float]): Mutable mapping of reward component names to values; the
            keys `hunger_pressure`, `fatigue_pressure`, and `sleep_debt_pressure` are decremented.
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
    Update per-tick reward components and world/tick-context event state based on movement, distances, terrain, and physiological indicators.
    
    Mutates tick_context.reward_components, world.state counters, and tick_context fields; writes integer distance deltas to info["distance_deltas"]; records reward-related events via tick_context.record_event; and may call reset_sleep_state(world) when threat proximity/contact/pain thresholds are exceeded.
    
    Parameters:
        world (SpiderWorld): Simulation world providing state, positions, configuration, and helper methods.
        tick_context (TickContext): Tick-scoped context containing the pre-tick snapshot, mutable reward_components, info, executed action, movement/terrain flags, and event-recording helpers.
    """
    from ..perception import predator_visible_to_spider
    from ..physiology import reset_sleep_state

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
        if max_distance > 0.0:
            risk = max(
                0.0,
                min(1.0, ((max_distance + 1.0) - new_predator_dist) / max_distance),
            )
        else:
            risk = 0.0
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
