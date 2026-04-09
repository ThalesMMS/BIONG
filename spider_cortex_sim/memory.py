from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Iterable

from .world_types import MemorySlot

if TYPE_CHECKING:
    from .world import SpiderWorld


MEMORY_TTLS = {
    "food": 12,
    "predator": 10,
    "shelter": 20,
    "escape": 8,
}


MEMORY_LEAKAGE_AUDIT = {
    "food_memory": {
        "classification": "plausible_memory",
        "risk": "low",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "nearest visible food",
        "notes": "This is the most ecological case: it depends on visible food and persists only for a short TTL.",
    },
    "predator_memory": {
        "classification": "world_owned_memory",
        "risk": "medium",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "world.lizard_pos() when predator_visible_to_spider() or recent_contact",
        "notes": "The recorded position is the lizard's real one; plausible as explicit memory, but still world-owned.",
    },
    "shelter_memory": {
        "classification": "world_owned_memory",
        "risk": "high",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "world.safest_shelter_target() on shelter or when shelter cells are visible",
        "notes": "Stores the safe shelter computed by the world, not just a shelter perceived by the agent.",
    },
    "escape_memory": {
        "classification": "world_owned_memory",
        "risk": "medium",
        "source": "spider_cortex_sim.memory.refresh_memory",
        "update_rule": "escape_memory_target(world) from last_move_dx/last_move_dy",
        "notes": "The escape route is built by an external rule and enters the observation space already formed.",
    },
}


def memory_leakage_audit() -> dict[str, dict[str, object]]:
    """
    Provide the structured audit catalog for the module's explicit world-owned memory slots.
    
    Returns:
        audit (dict[str, dict[str, object]]): A deep copy of MEMORY_LEAKAGE_AUDIT mapping each memory slot
        name to its metadata dictionary (classification, risk, source, update_rule, notes).
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
    Increment the age of a memory slot and clear it when its age exceeds the configured TTL.
    
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
    Set the memory slot's target and reset its age to zero.
    
    If `target` is None, the slot is not modified.
    
    Parameters:
        slot (MemorySlot): The memory slot to update.
        target (tuple[int, int] | None): Grid coordinates to store as the slot's target.
    """
    if target is None:
        return
    slot.target = target
    slot.age = 0


def escape_memory_target(world: "SpiderWorld") -> tuple[int, int]:
    """
    Compute a one-step escape target in the direction of the spider's last move, clamped to world bounds and falling back to the spider's current position if the candidate cell is not walkable.
    
    Returns:
        tuple[int, int]: A target grid position one step from the spider in the sign of its last movement (dx, dy). If that candidate is not walkable, returns the spider's current position.
    """
    dx = 0 if world.state.last_move_dx == 0 else (1 if world.state.last_move_dx > 0 else -1)
    dy = 0 if world.state.last_move_dy == 0 else (1 if world.state.last_move_dy > 0 else -1)
    unclamped_target = (world.state.x + dx, world.state.y + dy)
    target = (
        max(0, min(world.width - 1, unclamped_target[0])),
        max(0, min(world.height - 1, unclamped_target[1])),
    )
    if not world.is_walkable(target):
        return world.spider_pos()
    return target


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
    from .perception import has_line_of_sight, visible_range

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
    Update the spider's memory slots using current perceptions and optional predator-escape behavior.
    
    If `initial` is False, age existing memories and clear those whose TTLs are exceeded. Then:
    - Set food memory to the nearest visible food position when any visible food exists.
    - Set predator memory to the lizard position if predator visibility > 0.5 or recent contact > 0.0.
    - Set shelter memory to the safest shelter target if the spider is on shelter, otherwise to a safest shelter target when any shelter cells are visible.
    - When `predator_escape` is True, set escape memory to the computed escape target.
    
    Parameters:
        world (SpiderWorld): The simulation world whose state and perception are used and mutated.
        predator_escape (bool): If True, compute and store an escape target based on the last move.
        initial (bool): If True, skip aging/clearing of existing memory slots (used for initialization).
    """
    from .perception import predator_visible_to_spider

    if not initial:
        age_or_clear_memory(world.state.food_memory, ttl_name="food")
        age_or_clear_memory(world.state.predator_memory, ttl_name="predator")
        age_or_clear_memory(world.state.shelter_memory, ttl_name="shelter")
        age_or_clear_memory(world.state.escape_memory, ttl_name="escape")

    visible_food = _visible_positions(world, world.food_positions)
    if visible_food:
        set_memory(world.state.food_memory, world.nearest(visible_food)[0])

    if predator_visible_to_spider(world).visible > 0.5 or world.state.recent_contact > 0.0:
        set_memory(world.state.predator_memory, world.lizard_pos())

    if world.on_shelter():
        set_memory(world.state.shelter_memory, world.safest_shelter_target())
    else:
        visible_shelter = _visible_positions(world, world.shelter_cells)
        if visible_shelter:
            set_memory(world.state.shelter_memory, world.safest_shelter_target())

    if predator_escape:
        set_memory(world.state.escape_memory, escape_memory_target(world))
