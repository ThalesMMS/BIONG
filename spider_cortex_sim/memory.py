from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Iterable

from .perception import (
    advance_percept_trace,
    has_line_of_sight,
    predator_visible_to_spider,
    visible_object,
    visible_range,
)
from .world_types import MemorySlot

if TYPE_CHECKING:
    from .world import SpiderWorld


MEMORY_TTLS = {
    "food": 12,
    "predator": 10,
    "shelter": 20,
    "escape": 8,
}


MEMORY_AUDIT_CLASSIFICATIONS = {
    "plausible_memory": {
        "definition": (
            "Perception-grounded state maintained by the environment pipeline. "
            "Allowed sources are local visual perception, contact events, and movement "
            "history; entries are TTL-bounded and age through refresh cycles."
        ),
        "allowed_sources": (
            "local visual perception",
            "contact events",
            "movement history",
        ),
        "disallowed_sources": (
            "unfiltered global object positions",
            "predator intent or exact predator state without perception",
            "grid state that bypasses the perception layer",
        ),
    },
    "world_owned_memory": {
        "definition": (
            "Hypothetical memory state requiring direct world knowledge, such as "
            "global food positions or predator intent. There are currently zero "
            "explicit spider memory slots in this classification."
        ),
    },
    "self-knowledge": {
        "definition": (
            "Inspection or action-history signals derived from the spider's own "
            "heading, scans, and observation history. These signals describe what "
            "the spider has inspected, not privileged world state."
        ),
    },
    "leakage": {
        "definition": (
            "Direct access to simulation internals that bypasses perception. This "
            "classification is disallowed for network-facing explicit memory."
        ),
    },
}


# Living contract against future leakage:
# - Do not write from world.food_positions unless filtered through visibility.
# - Do not write from world.predator.position without a perception check.
# - Do not write from grid state that bypasses the perception layer.
# - Active-sensing signals may expose the spider's own inspection history, but
#   must not expose hidden target positions or world-owned planning state.
MEMORY_LEAKAGE_AUDIT = {
    "food_memory": {
        "classification": "plausible_memory",
        "risk": "low",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "nearest visible food",
        "allowed_sources": ("local visual perception",),
        "rationale": (
            "Food targets come from _visible_positions(), which filters candidate "
            "positions by visible_range() and has_line_of_sight() before nearest() "
            "selects a target."
        ),
        "notes": "This is the most ecological case: it depends on visible food and persists only for a short TTL.",
    },
    "predator_memory": {
        "classification": "plausible_memory",
        "risk": "low",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "predator_visible_to_spider().position when visible; spider position on recent contact",
        "allowed_sources": ("local visual perception", "contact events"),
        "rationale": (
            "Predator targets come from predator_visible_to_spider() when the predator "
            "is visible, or from the spider's own position during recent contact."
        ),
        "notes": "Stores predator location only through visual perception, with contact events recorded as local proximity to the spider.",
    },
    "shelter_memory": {
        "classification": "plausible_memory",
        "risk": "low",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "visible_object(..., world.shelter_cells).position for perceived shelter cells",
        "allowed_sources": ("local visual perception",),
        "rationale": (
            "Shelter targets come from visible_object(), which selects only shelter "
            "cells available through the local perception query."
        ),
        "notes": "Stores shelter cells selected by local perception rather than a world-selected best shelter target.",
    },
    "escape_memory": {
        "classification": "plausible_memory",
        "risk": "low",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "escape_memory_target(world) from last_move_dx/last_move_dy",
        "allowed_sources": ("movement history",),
        "rationale": (
            "Escape targets are derived from last_move_dx and last_move_dy, then "
            "clamped to world bounds without querying walkability, shelter policy, or "
            "hidden object locations."
        ),
        "notes": "The escape target is derived from movement history and world-boundary clamping without walkability or shelter-policy queries.",
    },
    "foveal_scan_age": {
        "classification": "self-knowledge",
        "risk": "low",
        "source": "spider_cortex_sim.world._scan_age_for_heading",
        "update_rule": "world tick minus the last tick when the current heading was actively scanned",
        "allowed_sources": ("action history", "inspection history"),
        "rationale": (
            "Scan age is computed from last_scan_tick_* fields written when the "
            "spider changes heading through locomotion or ORIENT_* actions."
        ),
        "notes": "Active-sensing recency reflects the spider's own inspection history, not privileged world state.",
    },
    "food_trace_heading_dx": {
        "classification": "self-knowledge",
        "risk": "low",
        "source": "spider_cortex_sim.perception.advance_percept_trace",
        "update_rule": "current heading_dx when a food trace is refreshed from local perception",
        "allowed_sources": ("observation history", "inspection history"),
        "rationale": (
            "The heading is copied from SpiderState only after a food target has "
            "passed local visibility, field-of-view, and line-of-sight checks."
        ),
        "notes": "Trace heading context records how the spider inspected the target; it is not a target direction or hidden location.",
    },
    "food_trace_heading_dy": {
        "classification": "self-knowledge",
        "risk": "low",
        "source": "spider_cortex_sim.perception.advance_percept_trace",
        "update_rule": "current heading_dy when a food trace is refreshed from local perception",
        "allowed_sources": ("observation history", "inspection history"),
        "rationale": (
            "The heading is copied from SpiderState only after a food target has "
            "passed local visibility, field-of-view, and line-of-sight checks."
        ),
        "notes": "Trace heading context records how the spider inspected the target; it is not a target direction or hidden location.",
    },
    "shelter_trace_heading_dx": {
        "classification": "self-knowledge",
        "risk": "low",
        "source": "spider_cortex_sim.perception.advance_percept_trace",
        "update_rule": "current heading_dx when a shelter trace is refreshed from local perception",
        "allowed_sources": ("observation history", "inspection history"),
        "rationale": (
            "The heading is copied from SpiderState only after a shelter target has "
            "passed local visibility, field-of-view, and line-of-sight checks."
        ),
        "notes": "Trace heading context records how the spider inspected the target; it is not a target direction or hidden location.",
    },
    "shelter_trace_heading_dy": {
        "classification": "self-knowledge",
        "risk": "low",
        "source": "spider_cortex_sim.perception.advance_percept_trace",
        "update_rule": "current heading_dy when a shelter trace is refreshed from local perception",
        "allowed_sources": ("observation history", "inspection history"),
        "rationale": (
            "The heading is copied from SpiderState only after a shelter target has "
            "passed local visibility, field-of-view, and line-of-sight checks."
        ),
        "notes": "Trace heading context records how the spider inspected the target; it is not a target direction or hidden location.",
    },
    "predator_trace_heading_dx": {
        "classification": "self-knowledge",
        "risk": "low",
        "source": "spider_cortex_sim.perception.advance_percept_trace",
        "update_rule": "current heading_dx when a predator trace is refreshed from local perception",
        "allowed_sources": ("observation history", "inspection history"),
        "rationale": (
            "The heading is copied from SpiderState only after a predator target has "
            "passed local visibility, field-of-view, and line-of-sight checks."
        ),
        "notes": "Trace heading context records how the spider inspected the target; it is not a target direction or hidden location.",
    },
    "predator_trace_heading_dy": {
        "classification": "self-knowledge",
        "risk": "low",
        "source": "spider_cortex_sim.perception.advance_percept_trace",
        "update_rule": "current heading_dy when a predator trace is refreshed from local perception",
        "allowed_sources": ("observation history", "inspection history"),
        "rationale": (
            "The heading is copied from SpiderState only after a predator target has "
            "passed local visibility, field-of-view, and line-of-sight checks."
        ),
        "notes": "Trace heading context records how the spider inspected the target; it is not a target direction or hidden location.",
    },
}


def memory_leakage_audit() -> dict[str, dict[str, object]]:
    """
    Provide the audit catalog for perception-grounded explicit memory slots.

    The catalog distinguishes TTL-bounded memory fed by perception, contact,
    and movement history from state that would require direct world knowledge.
    
    Returns:
        audit (dict[str, dict[str, object]]): A deep copy of MEMORY_LEAKAGE_AUDIT mapping each memory slot
        name to its metadata dictionary (classification, risk, source, update_rule,
        allowed_sources, rationale, notes).
    """
    return deepcopy(MEMORY_LEAKAGE_AUDIT)


def empty_memory_slot() -> MemorySlot:
    """
    Create a new, cleared memory slot for storing a target and its age.
    
    Returns:
        MemorySlot: A slot with `target=None` and `age=0`.
    """
    return MemorySlot(target=None, age=0)


def memory_vector(world: "SpiderWorld", slot: MemorySlot, *, ttl_name: str) -> tuple[float, float, float]:
    """
    Convert a memory slot into a 3-tuple encoding direction and normalized age.
    
    Parameters:
        world (SpiderWorld): World context used to compute the relative displacement from the spider to the stored target.
        slot (MemorySlot): Memory slot containing an optional target and an age counter.
        ttl_name (str): Key into MEMORY_TTLS to obtain the time-to-live used to normalize the slot's age.
    
    Returns:
        tuple[float, float, float]: `(dx, dy, age_norm)` where `dx` and `dy` are the relative displacement from the spider to `slot.target`, and `age_norm` is `slot.age / ttl`. If `slot.target` is None or the slot's age exceeds the TTL, returns `(0.0, 0.0, 1.0)`.
    """
    ttl = MEMORY_TTLS[ttl_name]
    if slot.target is None or slot.age > ttl:
        return 0.0, 0.0, 1.0
    dx, dy, _ = world._relative(slot.target)
    return dx, dy, float(slot.age / ttl)


def age_or_clear_memory(slot: MemorySlot, *, ttl_name: str) -> None:
    """
    Age a perception-grounded memory slot and clear it after its configured TTL.
    
    If the slot has no target, the function does nothing. Otherwise it increments the slot's
    age by 1 and, if the new age is greater than MEMORY_TTLS[ttl_name], clears the target
    and resets the age to 0.
    
    Parameters:
        ttl_name (str): Key into MEMORY_TTLS selecting the time-to-live threshold used to
            determine when the memory should be cleared.
    """
    if slot.target is None:
        return
    slot.age += 1
    if slot.age > MEMORY_TTLS[ttl_name]:
        slot.target = None
        slot.age = 0


def set_memory(slot: MemorySlot, target: tuple[int, int] | None) -> None:
    """
    Set a memory slot's target from an approved source and reset its age.
    
    If `target` is None the slot is left unchanged. Otherwise the slot's
    `target` is set to `target` and its `age` is reset to 0.
    
    Parameters:
        slot: The MemorySlot to update.
        target: Grid coordinates to store as the slot's target, or None to keep the slot unchanged.
    """
    if target is None:
        return
    slot.target = target
    slot.age = 0


def escape_memory_target(world: "SpiderWorld") -> tuple[int, int]:
    """
    Compute a one-step escape target in the direction of the spider's last move, clamped to world bounds.
    
    Returns:
        tuple[int, int]: A target grid position one step from the spider in the sign of its last movement (dx, dy).
    """
    dx = (world.state.last_move_dx > 0) - (world.state.last_move_dx < 0)
    dy = (world.state.last_move_dy > 0) - (world.state.last_move_dy < 0)
    unclamped_target = (world.state.x + dx, world.state.y + dy)
    return (
        max(0, min(world.width - 1, unclamped_target[0])),
        max(0, min(world.height - 1, unclamped_target[1])),
    )


def _visible_positions(
    world: "SpiderWorld",
    positions: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Filter a sequence of positions to those within the spider's visible range and line of sight.
    
    Parameters:
        world: The simulation world containing the spider and perception parameters.
        positions: An iterable of (x, y) positions to be tested.
    
    Returns:
        list[tuple[int, int]]: Positions whose manhattan distance from the spider is less than or equal to the spider's perception radius and for which the spider has line of sight.
    """
    radius = visible_range(world)
    return [
        pos
        for pos in positions
        if world.manhattan(world.spider_pos(), pos) <= radius
        and has_line_of_sight(world, world.spider_pos(), pos)
    ]


def refresh_memory(
    world: "SpiderWorld",
    *,
    predator_escape: bool = False,
    initial: bool = False,
) -> None:
    """
    Maintain perception-grounded memory using allowed update sources.

    This is the environment-pipeline mechanism for aging, TTL expiration, and
    writing targets derived from perception, contact, or movement history. It
    does not provide a channel for direct world knowledge.
    
    If `initial` is False, age existing memories and clear those whose TTLs are exceeded. Then:
    - Set food memory to the nearest visible food position when any visible food exists.
    - Set predator memory to a perceived predator position, or to spider position on recent contact.
    - Set shelter memory to a perceived shelter cell.
    - When `predator_escape` is True, set escape memory to the computed escape target.
    
    Parameters:
        world (SpiderWorld): The simulation world whose state and perception are used and mutated.
        predator_escape (bool): If True, compute and store an escape target based on the last move.
        initial (bool): If True, skip aging/clearing of existing memory slots (used for initialization).
    """
    if not initial:
        age_or_clear_memory(world.state.food_memory, ttl_name="food")
        age_or_clear_memory(world.state.predator_memory, ttl_name="predator")
        age_or_clear_memory(world.state.shelter_memory, ttl_name="shelter")
        age_or_clear_memory(world.state.escape_memory, ttl_name="escape")

    visible_food = _visible_positions(world, world.food_positions)
    if visible_food:
        set_memory(world.state.food_memory, world.nearest(visible_food)[0])

    predator_view = predator_visible_to_spider(world)
    if predator_view.visible > 0.5:
        set_memory(world.state.predator_memory, predator_view.position)
    elif world.state.recent_contact > 0.0:
        set_memory(world.state.predator_memory, world.spider_pos())

    shelter_view = visible_object(
        world,
        world.shelter_cells,
        radius=visible_range(world),
    )
    if shelter_view.position is not None:
        set_memory(world.state.shelter_memory, shelter_view.position)

    if predator_escape:
        set_memory(world.state.escape_memory, escape_memory_target(world))


def refresh_perceptual_state(world: "SpiderWorld") -> None:
    """
    Update short-lived percept traces from the current immediate perceptions.

    The trace refresh uses non-noisy visibility views so the memory stage owns
    one deterministic update of food, shelter, and predator traces per tick.
    """
    radius = visible_range(world)
    food_view = visible_object(world, world.food_positions, radius=radius, apply_noise=False)
    shelter_view = visible_object(world, world.shelter_cells, radius=radius, apply_noise=False)
    predator_view = predator_visible_to_spider(world, apply_noise=False)

    world.state.food_trace = advance_percept_trace(
        world,
        world.state.food_trace,
        food_view,
        world.food_positions,
    )
    world.state.shelter_trace = advance_percept_trace(
        world,
        world.state.shelter_trace,
        shelter_view,
        world.shelter_cells,
    )
    world.state.predator_trace = advance_percept_trace(
        world,
        world.state.predator_trace,
        predator_view,
        world.predator_positions() or [world.lizard_pos()],
    )


def refresh_all_memory(
    world: "SpiderWorld",
    *,
    predator_escape: bool = False,
    initial: bool = False,
) -> None:
    """
    Refresh explicit memory slots and short-lived percept traces together.
    """
    refresh_memory(world, predator_escape=predator_escape, initial=initial)
    refresh_perceptual_state(world)
