"""Tick pipeline stage registry and execution helpers.

The tick pipeline is intentionally linear: movement resolves first, terrain and
wakefulness costs come next, predator contact is checked before and after the
predator update, then reward aggregation runs before postprocess finalizes the
tick and memory refresh snapshots the result. This keeps cross-stage mutation
points explicit. Long-lived simulation state lives on ``world`` and tick-scoped
diagnostics live on ``context``; stages communicate by mutating those two
objects and by emitting events through ``context.record_event()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .interfaces import LOCOMOTION_ACTIONS
from .memory import refresh_all_memory
from .noise import apply_motor_noise
from .physiology import (
    apply_homeostasis_penalties,
    apply_predator_contact,
    apply_turn_fatigue,
    apply_wakefulness,
    clip_state,
    resolve_autonomic_behaviors,
)
from .reward import (
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_total,
)
from .world_types import MemorySlot, StageDescriptor, TickContext

if TYPE_CHECKING:
    from .world import SpiderWorld


ACTIONS = tuple(LOCOMOTION_ACTIONS)
EXPECTED_EVENT_LOG_STAGES = (
    "pre_tick",
    "action",
    "terrain_and_wakefulness",
    "predator_contact",
    "autonomic",
    "predator_update",
    "reward",
    "postprocess",
    "memory",
    "finalize",
)


def build_tick_context(world: "SpiderWorld", action_idx: int) -> TickContext:
    """Build a per-tick context and record the pre-tick bootstrap events."""
    if isinstance(action_idx, bool) or not isinstance(action_idx, (int, np.integer)):
        raise TypeError(
            f"Invalid action_idx {action_idx!r}; expected an integer in the range "
            f"[0, {len(ACTIONS) - 1}]"
        )
    action_idx = int(action_idx)
    if not 0 <= action_idx < len(ACTIONS):
        raise ValueError(
            f"Invalid action_idx {action_idx!r}; expected an integer in the range "
            f"[0, {len(ACTIONS) - 1}]"
        )

    intended_action = ACTIONS[action_idx]
    snapshot = world._capture_tick_snapshot()
    context = TickContext(
        action_idx=int(action_idx),
        intended_action=intended_action,
        executed_action=intended_action,
        motor_noise_applied=False,
        snapshot=snapshot,
        reward_components=empty_reward_components(),
        info={
            "action": intended_action,
            "intended_action": intended_action,
            "executed_action": intended_action,
            "motor_noise_applied": False,
            "motor_slip_reason": "none",
            "motor_execution_difficulty": 0.0,
            "motor_execution_components": {},
            "motor_execution": {
                "difficulty": 0.0,
                "components": {},
                "slip": {
                    "occurred": False,
                    "reason": "none",
                    "original_action": intended_action,
                    "executed_action": intended_action,
                    "slip_probability": 0.0,
                    "execution_difficulty": 0.0,
                    "terrain": "unknown",
                    "components": {},
                },
            },
            "momentum": float(np.clip(world.state.momentum, 0.0, 1.0)),
            "momentum_before": float(np.clip(world.state.momentum, 0.0, 1.0)),
            "momentum_after": float(np.clip(world.state.momentum, 0.0, 1.0)),
            "turn_angle": 0.0,
            "turn_fatigue_applied": 0.0,
            "motor_slip": {
                "occurred": False,
                "reason": "none",
                "original_action": intended_action,
                "executed_action": intended_action,
                "slip_probability": 0.0,
                "execution_difficulty": 0.0,
                "terrain": "unknown",
                "components": {},
            },
            "ate": False,
            "slept": False,
            "pain": False,
            "predator_contact": False,
            "predator_transition": None,
            "distance_deltas": {},
        },
    )
    context.record_event("pre_tick", "snapshot", **snapshot.to_payload())
    return context


def run_action_stage(world: "SpiderWorld", context: TickContext) -> None:
    """Apply the resolved action and record the movement result."""
    momentum_before = float(np.clip(world.state.momentum, 0.0, 1.0))
    heading_before = (int(world.state.heading_dx), int(world.state.heading_dy))
    slip_info = apply_motor_noise(world, context.intended_action)
    context.executed_action = str(slip_info["executed_action"])
    context.motor_noise_applied = bool(slip_info["occurred"])
    context.execution_difficulty = float(slip_info["execution_difficulty"])
    components = slip_info.get("components", {})
    context.execution_components = dict(components) if isinstance(components, dict) else {}
    context.motor_slip_info = dict(slip_info)
    context.info.update(
        {
            "action": context.intended_action,
            "intended_action": context.intended_action,
            "executed_action": context.executed_action,
            "motor_noise_applied": bool(context.motor_noise_applied),
            "motor_slip_reason": str(slip_info["reason"]),
            "slip_reason": str(slip_info["reason"]),
            "motor_execution_difficulty": context.execution_difficulty,
            "motor_execution_components": dict(context.execution_components),
            "motor_slip": dict(context.motor_slip_info),
            "motor_execution": {
                "difficulty": context.execution_difficulty,
                "components": dict(context.execution_components),
                "slip": dict(context.motor_slip_info),
            },
            "momentum_before": momentum_before,
        }
    )
    context.moved = bool(world._move_spider_action(context.executed_action))
    world._update_momentum(
        context.executed_action,
        previous_heading=heading_before,
        moved=bool(context.moved),
    )
    turn_cost = apply_turn_fatigue(
        world,
        heading_before,
        (int(world.state.heading_dx), int(world.state.heading_dy)),
    )
    momentum_after = float(np.clip(world.state.momentum, 0.0, 1.0))
    context.info.update(
        {
            "momentum": momentum_after,
            "momentum_after": momentum_after,
            "turn_angle": float(turn_cost["turn_angle"]),
            "turn_fatigue_applied": float(turn_cost["turn_fatigue_applied"]),
        }
    )
    context.record_event(
        "action",
        "action_resolved",
        action_index=int(context.action_idx),
        intended_action=context.intended_action,
        executed_action=context.executed_action,
        motor_noise_applied=bool(context.motor_noise_applied),
        slip_reason=str(slip_info["reason"]),
        slip_probability=round(float(slip_info["slip_probability"]), 6),
        execution_difficulty=round(float(context.execution_difficulty), 6),
        momentum_before=round(float(momentum_before), 6),
        momentum_after=round(float(momentum_after), 6),
        turn_angle=round(float(turn_cost["turn_angle"]), 6),
        turn_fatigue_applied=round(float(turn_cost["turn_fatigue_applied"]), 6),
        orientation_alignment=round(
            float(context.execution_components.get("orientation_alignment", 1.0)),
            6,
        ),
        terrain_difficulty=round(
            float(context.execution_components.get("terrain_difficulty", 0.0)),
            6,
        ),
        fatigue_factor=round(
            float(context.execution_components.get("fatigue_factor", 0.0)),
            6,
        ),
    )
    context.record_event(
        "action",
        "movement_applied",
        moved=bool(context.moved),
        spider_pos=[int(world.state.x), int(world.state.y)],
        last_move_dx=int(world.state.last_move_dx),
        last_move_dy=int(world.state.last_move_dy),
        momentum=round(float(momentum_after), 6),
    )


def run_terrain_stage(world: "SpiderWorld", context: TickContext) -> None:
    """Apply terrain costs, wakefulness effects, and pressure penalties."""
    cfg = world.reward_config
    context.terrain_now = apply_action_and_terrain_effects(
        world,
        action_name=context.executed_action,
        moved=context.moved,
        reward_components=context.reward_components,
    )
    context.predator_threat = bool(
        compute_predator_threat(
            world,
            prev_predator_visible=context.snapshot.prev_predator_visible,
            prev_predator_dist=context.snapshot.prev_predator_dist,
        )
    )
    context.interrupted_rest = bool(
        context.snapshot.rest_streak > 0
        and (
            context.executed_action != "STAY"
            or context.predator_threat
            or context.snapshot.prev_shelter_role == "outside"
        )
    )
    context.exposed_at_night = bool(
        context.snapshot.night and world.shelter_role_at(world.spider_pos()) == "outside"
    )
    apply_wakefulness(
        world,
        night=context.snapshot.night,
        exposed=context.exposed_at_night,
        interrupted_rest=context.interrupted_rest,
    )
    if context.exposed_at_night:
        world.state.fatigue += cfg["night_exposure_fatigue"]
        context.reward_components["night_exposure"] -= cfg["night_exposure_reward"]
    apply_pressure_penalties(world, context.reward_components)
    context.record_event(
        "terrain_and_wakefulness",
        "effects_applied",
        terrain=context.terrain_now,
        predator_threat=bool(context.predator_threat),
        interrupted_rest=bool(context.interrupted_rest),
        exposed_at_night=bool(context.exposed_at_night),
        sleep_debt=round(float(world.state.sleep_debt), 6),
        fatigue=round(float(world.state.fatigue), 6),
    )


def _maybe_apply_predator_contact(world: "SpiderWorld", context: TickContext) -> bool:
    """
    Apply predator contact when the spider overlaps any predator and contact hasn't been applied yet.
    
    If contact is applied, calls apply_predator_contact and sets context.predator_contact_applied to True.
    
    Returns:
        True if contact was applied, False otherwise.
    """
    if (
        any(world.spider_pos() == predator_pos for predator_pos in world.predator_positions())
        and not context.predator_contact_applied
    ):
        apply_predator_contact(world, context.reward_components, context.info, tick_context=context)
        context.predator_contact_applied = True
        return True
    return False


def run_predator_contact_stage(world: "SpiderWorld", context: TickContext) -> None:
    """Apply immediate predator contact if the spider and lizard overlap."""
    if not _maybe_apply_predator_contact(world, context):
        context.record_event(
            "predator_contact",
            "contact_check",
            predator_contact=bool(context.predator_contact_applied),
        )


def run_autonomic_stage(world: "SpiderWorld", context: TickContext) -> None:
    """Resolve feeding and resting behavior, including the idle-open penalty."""
    resolve_autonomic_behaviors(
        world,
        action_name=context.executed_action,
        predator_threat=context.predator_threat,
        night=context.snapshot.night,
        reward_components=context.reward_components,
        info=context.info,
        tick_context=context,
    )

    if (
        not world.on_shelter()
        and not context.fed_this_tick
        and not world.on_food()
        and context.executed_action == "STAY"
    ):
        context.reward_components["action_cost"] -= world.reward_config["idle_open_penalty"]
        context.record_event(
            "autonomic",
            "idle_open_penalty",
            penalty=round(float(world.reward_config["idle_open_penalty"]), 6),
        )

    context.record_event(
        "autonomic",
        "autonomic_summary",
        ate=bool(context.info["ate"]),
        slept=bool(context.info["slept"]),
        sleep_phase=world.state.sleep_phase,
        rest_streak=int(world.state.rest_streak),
    )


def run_predator_update_stage(world: "SpiderWorld", context: TickContext) -> None:
    """
    Advance each predator controller, record per-predator movement and mode transitions, and perform a post-movement predator contact check.
    
    Updates TickContext fields and emitted event:
    - Sets `context.predator_moved` to True if any predator moved.
    - Sets `context.info["predator_moved_each"]` to a list of per-predator moved flags.
    - Records any per-predator mode transitions in `context.info["predator_transitions"]`.
    - If the global lizard mode changed, sets `context.info["predator_transition"]` with `from`/`to`.
    - Emits a "predator_update" event containing `moved`, `mode_before`, `mode_after`, `transition`, and `lizard_pos`.
    
    Parameters:
        world: The simulation world containing predator controllers and lizard state.
        context: The tick-scoped context mutated with predator diagnostics and events.
    """
    predator_mode_before = world.lizard.mode
    predator_moved_each: list[bool] = [False] * world.predator_count
    predator_transitions: list[dict[str, object]] = []
    for controller in world.predator_controllers:
        predator_index = int(controller.predator_index)
        predator = world.get_predator(predator_index)
        mode_before = predator.mode
        moved = bool(controller.update(world))
        while len(predator_moved_each) <= predator_index:
            predator_moved_each.append(False)
        predator_moved_each[predator_index] = moved
        mode_after = predator.mode
        if mode_before != mode_after:
            predator_transitions.append(
                {
                    "index": int(predator_index),
                    "from": mode_before,
                    "to": mode_after,
                }
            )
    context.predator_moved = any(predator_moved_each)
    predator_mode_after = world.lizard.mode
    if predator_mode_before != predator_mode_after:
        context.info["predator_transition"] = {
            "from": predator_mode_before,
            "to": predator_mode_after,
        }
    context.info["predator_transitions"] = predator_transitions
    context.info["predator_moved"] = bool(context.predator_moved)
    context.info["predator_moved_each"] = predator_moved_each
    context.record_event(
        "predator_update",
        "predator_update",
        moved=bool(context.predator_moved),
        mode_before=predator_mode_before,
        mode_after=predator_mode_after,
        transition=context.info["predator_transition"],
        moved_each=list(predator_moved_each),
        transitions=list(predator_transitions),
        lizard_pos=[int(world.lizard.x), int(world.lizard.y)],
    )
    _maybe_apply_predator_contact(world, context)


def run_reward_stage(world: "SpiderWorld", context: TickContext) -> None:
    """Apply progress- and event-based reward shaping."""
    apply_progress_and_event_rewards(world, tick_context=context)


def run_postprocess_stage(world: "SpiderWorld", context: TickContext) -> None:
    """Finalize reward totals, state clipping, and terminal bookkeeping."""
    contact_decay = float(world.reward_config.get("recent_contact_decay", 0.35))
    pain_decay = float(world.reward_config.get("recent_pain_decay", 0.78))
    death_penalty = float(world.reward_config.get("death_penalty", 5.0))

    if not context.info["predator_contact"]:
        world.state.recent_contact *= contact_decay
    world.state.recent_pain *= pain_decay

    apply_homeostasis_penalties(world, context.reward_components)
    clip_state(world)

    context.done = bool(world.state.health <= 0.0)
    if context.done:
        context.reward_components["death_penalty"] -= death_penalty

    context.reward = float(reward_total(context.reward_components))
    world.state.last_reward = context.reward
    world.state.total_reward += context.reward
    world.state.steps_alive += 1
    world.state.last_action = context.executed_action
    world._last_on_shelter = world.on_shelter()
    world._last_predator_visible = bool(context.predator_visible_now)
    world.tick += 1
    context.record_event(
        "reward",
        "reward_summary",
        predator_escape=bool(context.predator_escape),
        predator_visible_now=bool(context.predator_visible_now),
        reward=round(float(context.reward), 6),
        reward_components=copy_reward_components(context.reward_components),
    )
    context.record_event(
        "postprocess",
        "postprocess_complete",
        reward=round(float(context.reward), 6),
        done=bool(context.done),
        health=round(float(world.state.health), 6),
        tick=int(world.tick),
    )


def run_memory_stage(world: "SpiderWorld", context: TickContext) -> None:
    """Refresh perception-derived memory slots and emit the updated targets."""

    def _target(slot: MemorySlot) -> list[int] | None:
        """
        Get the memory slot's target as a list of indices, or None if the slot has no target.
        
        Parameters:
            slot (MemorySlot): Memory slot whose `target` (an iterable of ints) will be converted to a list.
        
        Returns:
            list[int] or None: A list of target indices when `slot.target` is present, `None` otherwise.
        """
        return list(slot.target) if slot.target is not None else None

    refresh_all_memory(world, predator_escape=context.predator_escape)
    context.record_event(
        "memory",
        "memory_refreshed",
        predator_escape=bool(context.predator_escape),
        food_memory_target=_target(world.state.food_memory),
        predator_memory_target=_target(world.state.predator_memory),
        shelter_memory_target=_target(world.state.shelter_memory),
        escape_memory_target=_target(world.state.escape_memory),
        tick=int(world.tick),
    )


def finalize_step(
    world: "SpiderWorld",
    context: TickContext,
) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
    """Build the step return tuple and append the final tick event."""
    next_obs = world.observe()
    context.record_event(
        "finalize",
        "tick_complete",
        reward=round(float(context.reward), 6),
        done=bool(context.done),
        tick=int(world.tick),
    )
    context.info["reward_components"] = copy_reward_components(context.reward_components)
    context.info["momentum"] = float(np.clip(world.state.momentum, 0.0, 1.0))
    context.info["state"] = world.state_dict()
    context.info["predator_escape"] = bool(context.predator_escape)
    context.info["event_log"] = context.serialized_event_log()
    return next_obs, float(context.reward), bool(context.done), context.info


TICK_STAGES: tuple[StageDescriptor, ...] = (
    StageDescriptor(
        name="action",
        run=run_action_stage,
        mutates=(
            "context.{executed_action,motor_noise_applied,execution_difficulty,execution_components,motor_slip_info,moved,info,event_log}",
            "world.state.{x,y,last_move_dx,last_move_dy,heading_dx,heading_dy,fatigue,momentum,last_scan_tick_up,last_scan_tick_down,last_scan_tick_left,last_scan_tick_right,last_scan_tick_up_left,last_scan_tick_up_right,last_scan_tick_down_left,last_scan_tick_down_right}",
            "world._perceptual_buffer",
        ),
    ),
    StageDescriptor(
        name="terrain_and_wakefulness",
        run=run_terrain_stage,
        mutates=(
            "context.{terrain_now,predator_threat,interrupted_rest,exposed_at_night,reward_components,event_log}",
            "world.state.{hunger,fatigue,sleep_debt}",
        ),
    ),
    StageDescriptor(
        name="predator_contact",
        run=run_predator_contact_stage,
        mutates=(
            "context.{predator_contact_applied,reward_components,info,event_log}",
            "world.state.{health,recent_pain,recent_contact,predator_contacts}",
        ),
    ),
    StageDescriptor(
        name="autonomic",
        run=run_autonomic_stage,
        mutates=(
            "context.{fed_this_tick,reward_components,info,event_log}",
            "world.state.{hunger,health,food_eaten,sleep_phase,rest_streak,fatigue,sleep_debt,sleep_events,momentum}",
        ),
    ),
    StageDescriptor(
        name="predator_update",
        run=run_predator_update_stage,
        mutates=(
            "context.{predator_moved,predator_contact_applied,reward_components,info,event_log}",
            "world.lizard.{x,y,mode,mode_ticks,patrol_target,last_known_spider,investigate_ticks,investigate_target}",
        ),
    ),
    StageDescriptor(
        name="reward",
        run=run_reward_stage,
        mutates=(
            "context.{predator_visible_now,predator_escape,reward_components,info,event_log}",
            "world.state.{shelter_entries,alert_events,predator_sightings,predator_escapes,sleep_phase,rest_streak}",
        ),
    ),
    StageDescriptor(
        name="postprocess",
        run=run_postprocess_stage,
        mutates=(
            "context.{done,reward,event_log,reward_components}",
            "world.state.{recent_contact,recent_pain,hunger,fatigue,sleep_debt,health,momentum,last_reward,total_reward,steps_alive,last_action}",
            "world.{_last_on_shelter,_last_predator_visible,tick}",
        ),
    ),
    # Mechanical maintenance of perception-grounded memory and percept traces.
    StageDescriptor(
        name="memory",
        run=run_memory_stage,
        mutates=(
            "context.event_log",
            "world.state.{food_memory,predator_memory,shelter_memory,escape_memory,food_trace,shelter_trace,predator_trace}",
        ),
    ),
)


def get_stage_order() -> list[str]:
    """Return the registered stage names in execution order."""
    return [descriptor.name for descriptor in TICK_STAGES]


def validate_pipeline() -> None:
    """Raise if the registry no longer matches the expected event-stage order."""
    stage_names = tuple(get_stage_order())
    if len(stage_names) != len(set(stage_names)):
        raise ValueError(f"Duplicate tick stage names are not allowed: {stage_names!r}")
    full_pipeline = ("pre_tick", *stage_names, "finalize")
    if full_pipeline != EXPECTED_EVENT_LOG_STAGES:
        raise ValueError(
            "Tick stage registry no longer matches the expected event log stages: "
            f"expected {EXPECTED_EVENT_LOG_STAGES!r}, got {full_pipeline!r}."
        )


validate_pipeline()
