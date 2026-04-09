from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .world import SpiderWorld
    from .world_types import TickContext


SLEEP_PHASES: tuple[str, ...] = ("AWAKE", "SETTLING", "RESTING", "DEEP_SLEEP")
SLEEP_PHASE_LEVELS: dict[str, float] = {
    "AWAKE": 0.0,
    "SETTLING": 1.0 / 3.0,
    "RESTING": 2.0 / 3.0,
    "DEEP_SLEEP": 1.0,
}


def sleep_phase_level(world: "SpiderWorld") -> float:
    """
    Map the world's current sleep phase to its numeric level.
    
    Returns:
        float: Numeric level for world.state.sleep_phase (AWAKE=0.0, SETTLING=1/3, RESTING=2/3, DEEP_SLEEP=1.0). Returns 0.0 if the phase is unrecognized.
    """
    return float(SLEEP_PHASE_LEVELS.get(world.state.sleep_phase, 0.0))


def rest_streak_norm(world: "SpiderWorld") -> float:
    """
    Normalize the world's rest streak to a value between 0.0 and 1.0 by clamping at 3 and dividing by 3.0.
    
    Returns:
        float: Normalized rest streak in the range [0.0, 1.0].
    """
    return float(min(world.state.rest_streak, 3) / 3.0)


def set_sleep_state(world: "SpiderWorld", phase: str, rest_streak: int) -> None:
    """
    Set the world's sleep phase and normalize its rest streak.
    
    Parameters:
        phase (str): Sleep phase name; must be a key in SLEEP_PHASE_LEVELS.
        rest_streak (int): Value to store as the rest streak; converted to int and clamped to a minimum of 0.
    
    Raises:
        ValueError: If `phase` is not a valid sleep phase.
    """
    if phase not in SLEEP_PHASE_LEVELS:
        raise ValueError(f"Invalid sleep phase: {phase}")
    world.state.sleep_phase = phase
    world.state.rest_streak = max(0, int(rest_streak))


def reset_sleep_state(world: "SpiderWorld") -> None:
    """
    Reset the world's sleep state to the "AWAKE" phase and set the rest streak to 0.
    """
    set_sleep_state(world, "AWAKE", 0)


def sleep_phase_from_streak(rest_streak: int, *, night: bool, shelter_role: str) -> str:
    """
    Selects the sleep phase ('AWAKE', 'SETTLING', 'RESTING', or 'DEEP_SLEEP') based on consecutive rest turns, whether it is night, and the shelter role.
    
    Parameters:
        rest_streak (int): Consecutive number of resting turns (may be zero or negative).
        night (bool): True if it is nighttime.
        shelter_role (str): Shelter role at the agent's position; e.g. "outside", "entrance", "inside", or "deep".
    
    Returns:
        phase (str): One of "AWAKE", "SETTLING", "RESTING", or "DEEP_SLEEP". Determination rules:
            - rest_streak <= 0 -> "AWAKE"
            - rest_streak == 1 -> "SETTLING"
            - shelter_role in {"entrance", "inside"} -> "RESTING"
            - shelter_role == "deep":
                - rest_streak == 2 -> "RESTING"
                - otherwise -> "DEEP_SLEEP" if night else "RESTING"
            - otherwise -> "AWAKE"
    """
    if rest_streak <= 0:
        return "AWAKE"
    if rest_streak == 1:
        return "SETTLING"
    if shelter_role in {"entrance", "inside"}:
        return "RESTING"
    if shelter_role == "deep":
        if rest_streak == 2:
            return "RESTING"
        return "DEEP_SLEEP" if night else "RESTING"
    return "AWAKE"


def apply_predator_contact(
    world: "SpiderWorld",
    reward_components: dict[str, float],
    info: dict[str, object],
    *,
    tick_context: "TickContext | None" = None,
) -> None:
    """
    Apply the effects of a predator contact to the world state, update reward components, and set info flags.
    
    Mutates world.state by reducing health (floored at 0.0), raising recent pain/contact, and incrementing alert and predator contact counters. Decreases the `"predator_contact"` entry in reward_components and sets info["pain"] and info["predator_contact"] to True. If provided, records a compact `"predator_contact"` event to the tick_context with sampled damage and the post-change health, recent_pain, and recent_contact values (rounded).
    
    Parameters:
        tick_context (TickContext | None): Optional context used to record an event when present; no event is recorded if None.
    """
    damage = float(world.rng.uniform(0.08, 0.16))
    world.state.health = max(0.0, world.state.health - damage)
    world.state.recent_pain = min(1.0, max(world.state.recent_pain, 0.65) + 0.25)
    world.state.recent_contact = 1.0
    world.state.alert_events += 1
    world.state.predator_contacts += 1
    info["pain"] = True
    info["predator_contact"] = True
    reward_components["predator_contact"] -= 0.80 + 1.6 * damage
    if tick_context is not None:
        tick_context.record_event(
            "predator_contact",
            "predator_contact",
            damage=round(damage, 6),
            health=round(float(world.state.health), 6),
            recent_pain=round(float(world.state.recent_pain), 6),
            recent_contact=round(float(world.state.recent_contact), 6),
        )


def apply_wakefulness(
    world: "SpiderWorld",
    *,
    night: bool,
    exposed: bool,
    interrupted_rest: bool,
) -> None:
    """
    Increase the world's sleep debt based on time of day and contextual flags.
    
    Parameters:
    	night (bool): If true, add `sleep_debt_night_awake` from the world's reward config; otherwise add `sleep_debt_day_awake`.
    	exposed (bool): If true, add `night_exposure_debt` from the reward config.
    	interrupted_rest (bool): If true, add `sleep_debt_interrupt` from the reward config.
    """
    cfg = world.reward_config
    world.state.sleep_debt += cfg["sleep_debt_night_awake"] if night else cfg["sleep_debt_day_awake"]
    if exposed:
        world.state.sleep_debt += cfg["night_exposure_debt"]
    if interrupted_rest:
        world.state.sleep_debt += cfg["sleep_debt_interrupt"]


def apply_restoration(
    world: "SpiderWorld",
    sleep_phase: str,
    *,
    night: bool,
    shelter_role: str,
) -> None:
    """
    Apply restorative effects to the world's physiological state based on the current sleep phase.
    
    Modifies world.state.fatigue, world.state.sleep_debt, and world.state.health according to `sleep_phase`, with different magnitudes depending on `night` and `shelter_role`. Intended phases: "SETTLING", "RESTING", and "DEEP_SLEEP".
    
    Parameters:
        world (SpiderWorld): Environment and mutable state that will be updated.
        sleep_phase (str): One of "SETTLING", "RESTING", or "DEEP_SLEEP" indicating the current sleep phase.
        night (bool): Whether it is nighttime (affects restore magnitudes).
        shelter_role (str): Role of the shelter at the spider's position (e.g., "entrance", "inside", "deep") which influences restoration during "RESTING".
    """
    cfg = world.reward_config
    if sleep_phase == "RESTING":
        if shelter_role == "entrance":
            fatigue_restore = 0.12 if night else 0.06
            debt_restore = cfg["sleep_debt_resting_night"] * 0.55 if night else cfg["sleep_debt_resting_day"] * 0.55
        else:
            fatigue_restore = 0.24 if night else 0.14
            debt_restore = cfg["sleep_debt_resting_night"] if night else cfg["sleep_debt_resting_day"]
        world.state.fatigue = max(0.0, world.state.fatigue - fatigue_restore)
        world.state.sleep_debt = max(0.0, world.state.sleep_debt - debt_restore)
        world.state.health = min(1.0, world.state.health + (0.04 if night else 0.01))
    elif sleep_phase == "DEEP_SLEEP":
        world.state.fatigue = max(0.0, world.state.fatigue - 0.34)
        debt_restore = cfg["sleep_debt_deep_night"] if night else cfg["sleep_debt_deep_day"]
        world.state.sleep_debt = max(0.0, world.state.sleep_debt - debt_restore)
        world.state.health = min(1.0, world.state.health + 0.05)
    elif sleep_phase == "SETTLING":
        settle_restore = 0.10 if night else 0.05
        world.state.fatigue = max(0.0, world.state.fatigue - settle_restore)
        world.state.health = min(1.0, world.state.health + (0.02 if night else 0.0))


def resolve_autonomic_behaviors(
    world: "SpiderWorld",
    *,
    action_name: str,
    predator_threat: bool,
    night: bool,
    reward_components: dict[str, float],
    info: dict[str, object],
    tick_context: "TickContext | None" = None,
) -> None:
    """
    Decide and apply feeding, sheltering, and sleep-related autonomous behaviors, mutating world.state and updating reward_components and info.
    
    Evaluates current context (on food, on shelter, action, hunger, predator threat, shelter role, fatigue, rest streak, and sleep debt) to:
    - apply feeding effects when on food,
    - reset sleep state when off shelter,
    - attempt and apply resting (advance rest streak, set sleep phase, apply restoration, adjust resting rewards) when conditions permit,
    - or reset sleep state and apply conditional resting penalties when rest is blocked.
    If provided, records compact event summaries to tick_context.
    
    Parameters:
        world (SpiderWorld): Environment and mutable state container to update.
        action_name (str): Current action name used to decide feeding/rest behavior.
        predator_threat (bool): Whether a predator threat is present; prevents rest attempts when True.
        night (bool): Whether it is night; affects sleep resolution, restoration, and reward calculations.
        reward_components (dict[str, float]): Mutable accumulator for reward components to be adjusted.
        info (dict[str, object]): Mutable dictionary for observable flags (e.g., "ate", "slept").
        tick_context (TickContext | None): Optional recorder for compact runtime events; no effect if None.
    """
    if world.on_food():
        hunger_before = world.state.hunger
        feed_amount = 0.56 if action_name == "STAY" else 0.32
        world.state.hunger = max(0.0, world.state.hunger - feed_amount)
        world.state.health = min(1.0, world.state.health + 0.05)
        reward_components["feeding"] += (0.48 if action_name == "STAY" else 0.22) + 3.00 * hunger_before
        world.state.food_eaten += 1
        info["ate"] = True
        world.respawn_food(world.spider_pos())
        reset_sleep_state(world)
        if tick_context is not None:
            tick_context.fed_this_tick = True
            tick_context.record_event(
                "autonomic",
                "feeding",
                hunger_before=round(float(hunger_before), 6),
                hunger_after=round(float(world.state.hunger), 6),
                action=action_name,
            )
        return

    if not world.on_shelter():
        reset_sleep_state(world)
        if tick_context is not None:
            tick_context.record_event(
                "autonomic",
                "sleep_reset_off_shelter",
                action=action_name,
            )
        return

    fatigue_before = world.state.fatigue
    shelter_role = world.shelter_role_at(world.spider_pos())
    can_attempt_rest = (
        action_name == "STAY"
        and world.state.hunger < (0.62 if world.reward_profile == "classic" else 0.55)
        and not predator_threat
        and shelter_role != "outside"
        and (night or fatigue_before > 0.25 or world.state.rest_streak > 0 or world.state.sleep_debt > 0.28)
    )
    sleep_drive = max(fatigue_before + 0.75 * world.state.sleep_debt - 0.35, 0.0)

    if can_attempt_rest:
        new_rest_streak = world.state.rest_streak + 1
        phase = sleep_phase_from_streak(new_rest_streak, night=night, shelter_role=shelter_role)
        set_sleep_state(world, phase, new_rest_streak)
        apply_restoration(world, phase, night=night, shelter_role=shelter_role)
        if tick_context is not None:
            tick_context.record_event(
                "autonomic",
                "rest_phase",
                phase=phase,
                rest_streak=int(world.state.rest_streak),
                sleep_drive=round(float(sleep_drive), 6),
                shelter_role=shelter_role,
            )

        if phase == "SETTLING":
            reward_components["resting"] += 0.10 + 0.90 * sleep_drive + (0.06 if night else 0.02)
            reward_components["resting"] -= 0.24 * world.state.hunger
        elif phase == "RESTING":
            reward_components["resting"] += 0.14 + 1.40 * sleep_drive + (0.10 if night else 0.03)
            reward_components["resting"] -= 0.34 * world.state.hunger
            world.state.sleep_events += 1
            info["slept"] = True
        elif phase == "DEEP_SLEEP":
            reward_components["resting"] += 0.20 + 2.00 * sleep_drive + 0.16
            reward_components["resting"] -= 0.42 * world.state.hunger
            world.state.sleep_events += 1
            info["slept"] = True
        return

    reset_sleep_state(world)
    if tick_context is not None:
        tick_context.record_event(
            "autonomic",
            "rest_blocked",
            action=action_name,
            predator_threat=bool(predator_threat),
            shelter_role=shelter_role,
            fatigue_before=round(float(fatigue_before), 6),
            sleep_debt=round(float(world.state.sleep_debt), 6),
            hunger=round(float(world.state.hunger), 6),
        )
    if night or fatigue_before > 0.45 or world.state.sleep_debt > 0.45:
        if action_name != "STAY":
            reward_components["resting"] -= 0.10 + (0.08 if night else 0.04) + 0.32 * world.state.hunger
        else:
            reward_components["resting"] -= 0.10 + 0.20 * world.state.hunger
    elif action_name == "STAY" and world.state.hunger > 0.45 and not night:
        reward_components["resting"] -= 0.08 + 0.16 * world.state.hunger


def apply_homeostasis_penalties(world: "SpiderWorld", reward_components: dict[str, float]) -> None:
    """
    Apply health reductions and reward penalties when hunger, fatigue, or sleep debt exceed configured thresholds.
    
    This function mutates world.state and reward_components in-place:
    - If hunger > 0.75, reduces health by up to 0.08 scaled by the normalized excess and subtracts 0.42 * excess from reward_components["homeostasis_penalty"].
    - If fatigue > 0.82, reduces health by up to 0.07 scaled by the normalized excess and subtracts 0.30 * excess from reward_components["homeostasis_penalty"].
    - If sleep_debt exceeds world.reward_config["sleep_debt_overdue_threshold"], reduces health by up to 0.05 scaled by the normalized excess and subtracts world.reward_config["sleep_debt_health_penalty"] * excess from reward_components["homeostasis_penalty"].
    Health is floored at 0.0 after each reduction.
    
    Parameters:
        world (SpiderWorld): Environment object whose state (hunger, fatigue, sleep_debt, health) is modified.
        reward_components (dict[str, float]): Mutable reward accumulator; this function decreases the "homeostasis_penalty" entry.
    """
    if world.state.hunger > 0.75:
        excess = (world.state.hunger - 0.75) / 0.25
        world.state.health = max(0.0, world.state.health - 0.08 * excess)
        reward_components["homeostasis_penalty"] -= 0.42 * excess
    if world.state.fatigue > 0.82:
        excess = (world.state.fatigue - 0.82) / 0.18
        world.state.health = max(0.0, world.state.health - 0.07 * excess)
        reward_components["homeostasis_penalty"] -= 0.30 * excess
    overdue_threshold = world.reward_config["sleep_debt_overdue_threshold"]
    if world.state.sleep_debt > overdue_threshold:
        excess = (world.state.sleep_debt - overdue_threshold) / max(0.05, 1.0 - overdue_threshold)
        world.state.health = max(0.0, world.state.health - 0.05 * excess)
        reward_components["homeostasis_penalty"] -= world.reward_config["sleep_debt_health_penalty"] * excess


def clip_state(world: "SpiderWorld") -> None:
    """
    Clamp key physiological state variables to the range [0.0, 1.0].
    
    Clamps and converts to float the following world.state attributes: `hunger`, `fatigue`,
    `sleep_debt`, `health`, `recent_pain`, and `recent_contact`.
    """
    world.state.hunger = max(0.0, min(1.0, world.state.hunger))
    world.state.fatigue = max(0.0, min(1.0, world.state.fatigue))
    world.state.sleep_debt = max(0.0, min(1.0, world.state.sleep_debt))
    world.state.health = max(0.0, min(1.0, world.state.health))
    world.state.recent_pain = max(0.0, min(1.0, world.state.recent_pain))
    world.state.recent_contact = max(0.0, min(1.0, world.state.recent_contact))
