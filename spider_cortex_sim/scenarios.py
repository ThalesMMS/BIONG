from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

from .metrics import (
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    build_behavior_check,
    build_behavior_score,
)
from .predator import LizardState
from .world import SpiderWorld


BehaviorScoreFn = Callable[[EpisodeStats, Sequence[Dict[str, object]]], BehavioralEpisodeScore]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    objective: str
    behavior_checks: Sequence[BehaviorCheckSpec]
    diagnostic_focus: str
    success_interpretation: str
    failure_interpretation: str
    budget_note: str
    max_steps: int
    map_template: str
    setup: Callable[[SpiderWorld], None]
    score_episode: BehaviorScoreFn


def _first_cell(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """
    Selects the lexicographically smallest (x, y) coordinate from a collection of grid cells.
    
    Parameters:
        cells (set[tuple[int, int]] | tuple[tuple[int, int], ...]): Collection of (x, y) coordinates.
    
    Returns:
        tuple[int, int]: The coordinate with the smallest x, breaking ties by y.
    """
    return sorted(cells)[0]


def _safe_lizard_cell(world: SpiderWorld) -> tuple[int, int]:
    """
    Selects a walkable lizard spawn cell that is farthest from the spider.
    
    Returns:
        cell (tuple[int, int]): A walkable lizard spawn cell maximizing Manhattan distance to the spider; if none are walkable, returns the bottom-right map cell `(world.width - 1, world.height - 1)`.
    """
    candidates = sorted(
        (
            cell
            for cell in world.map_template.lizard_spawn_cells
            if world.is_lizard_walkable(cell)
        ),
        key=lambda cell: -world.manhattan(cell, world.spider_pos()),
    )
    return candidates[0] if candidates else (world.width - 1, world.height - 1)


def _entrance_ambush_cell(world: SpiderWorld, entrance: tuple[int, int]) -> tuple[int, int]:
    """
    Selects a walkable cell adjacent to a shelter entrance for placing the lizard, preferring cells closest to the entrance.
    
    Parameters:
        world (SpiderWorld): The game world used to evaluate bounds, walkability, and distances.
        entrance (tuple[int, int]): (x, y) coordinates of the shelter entrance.
    
    Returns:
        tuple[int, int]: Coordinates of a walkable adjacent cell ordered by increasing Manhattan distance to the entrance (ties broken by coordinates); if no adjacent cell is walkable, returns a safe fallback lizard cell.
    """
    candidates = sorted(
        (
            (entrance[0] + dx, entrance[1] + dy)
            for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1))
            if 0 <= entrance[0] + dx < world.width and 0 <= entrance[1] + dy < world.height
        ),
        key=lambda cell: (world.manhattan(cell, entrance), cell[0], cell[1]),
    )
    for cell in candidates:
        if world.is_lizard_walkable(cell):
            return cell
    return _safe_lizard_cell(world)


def _teleport_spider(
    world: SpiderWorld,
    position: tuple[int, int],
    *,
    reset_heading: bool = True,
) -> None:
    """
    Reposition the spider and optionally restore its default heading.
    """
    world.state.x, world.state.y = position
    if reset_heading:
        world._reset_heading_after_teleport()


def _night_rest(world: SpiderWorld) -> None:
    """
    Initialize the world for a night-rest scenario by placing the spider in a deep shelter, setting the simulation tick and the spider's hunger, fatigue, and sleep debt, placing a patrolling lizard at a safe spawn, and refreshing world memory.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = world.day_length + 1
    _teleport_spider(world, deep)
    world.state.hunger = 0.08
    world.state.fatigue = 0.82
    world.state.sleep_debt = NIGHT_REST_INITIAL_SLEEP_DEBT
    lx, ly = _safe_lizard_cell(world)
    world.lizard = LizardState(x=lx, y=ly, mode="PATROL")
    world.refresh_memory(initial=True)


def _predator_edge(world: SpiderWorld) -> None:
    """
    Prepare a SpiderWorld for the "predator edge" scenario.
    
    Sets the spider near the map edge with physiology tuned for the scenario, places a single food at the last food-spawn cell, spawns a patrolling lizard at (1, 3), sets the spider's heading toward the lizard, and refreshes world memory. The provided SpiderWorld is mutated in place.
    
    Parameters:
        world (SpiderWorld): World instance to configure.
    """
    world.tick = 1
    _teleport_spider(world, (1, 1), reset_heading=False)
    world.state.hunger = 0.18
    world.state.fatigue = 0.15
    world.state.sleep_debt = 0.18
    world.food_positions = [world.map_template.food_spawn_cells[-1]]
    world.lizard = LizardState(x=1, y=3, mode="PATROL")
    world.state.heading_dx, world.state.heading_dy = world._heading_toward(world.lizard_pos())
    world.refresh_memory(initial=True)


def _entrance_ambush(world: SpiderWorld) -> None:
    """
    Initialize the world for the "entrance ambush" scenario.
    
    Mutates the provided SpiderWorld in place to place the spider inside a shelter interior, position a lizard in a waiting ambush near the shelter entrance (with failed chases preset), advance the world tick to a late-day value, set the spider's hunger/fatigue/sleep debt, and refresh the world's memory.
    
    Parameters:
        world (SpiderWorld): World instance to configure; mutated in place.
    """
    interior = _first_cell(world.shelter_interior_cells or world.shelter_deep_cells)
    entrance = _first_cell(world.shelter_entrance_cells)
    world.tick = world.day_length + 2
    _teleport_spider(world, interior)
    world.state.hunger = 0.32
    world.state.fatigue = 0.26
    world.state.sleep_debt = 0.28
    lx, ly = _entrance_ambush_cell(world, entrance)
    world.lizard = LizardState(
        x=lx,
        y=ly,
        mode="WAIT",
        wait_target=entrance,
        failed_chases=2,
    )
    world.refresh_memory(initial=True)


def _open_field_foraging(world: SpiderWorld) -> None:
    """
    Set up the world state for the open-field foraging scenario.
    
    Places the spider in a deep shelter, sets tick to 2 and the spider's hunger (0.88), fatigue (0.22), and sleep debt (0.20); selects a single food spawn farthest from the spider; places the lizard at a safe patrol spawn; and refreshes world memory with initial=True.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = 2
    _teleport_spider(world, deep)
    world.state.hunger = 0.88
    world.state.fatigue = 0.22
    world.state.sleep_debt = 0.20
    far_food = sorted(
        world.map_template.food_spawn_cells,
        key=lambda cell: -world.manhattan(cell, deep),
    )[0]
    world.food_positions = [far_food]
    lx, ly = _safe_lizard_cell(world)
    world.lizard = LizardState(x=lx, y=ly, mode="PATROL")
    world.refresh_memory(initial=True)


def _shelter_blockade(world: SpiderWorld) -> None:
    """
    Configure the world for the "shelter blockade" scenario.
    
    Mutates the given SpiderWorld: places the spider in a deep (prefer deep, then interior) shelter cell, advances time to just after the day length, sets spider state (hunger=0.54, fatigue=0.42, sleep_debt=0.40), positions a lizard in `WAIT` mode at an ambush cell near the shelter entrance with `wait_target` set to that entrance and `failed_chases` set to 2, then refreshes world memory with `initial=True`.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells)
    entrance = _first_cell(world.shelter_entrance_cells)
    world.tick = world.day_length + 1
    _teleport_spider(world, deep)
    world.state.hunger = 0.54
    world.state.fatigue = 0.42
    world.state.sleep_debt = 0.40
    lx, ly = _entrance_ambush_cell(world, entrance)
    world.lizard = LizardState(
        x=lx,
        y=ly,
        mode="WAIT",
        wait_target=entrance,
        failed_chases=2,
    )
    world.refresh_memory(initial=True)


def _recover_after_failed_chase(world: SpiderWorld) -> None:
    """
    Set up the world for the "recover after failed chase" scenario: place the spider in a deep shelter with preset hunger/fatigue/sleep_debt, position the lizard near the shelter in a chase/recover configuration, and refresh world memory.
    
    Parameters:
        world (SpiderWorld): The world to modify; this function mutates its state for the scenario setup.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells)
    entrance = _first_cell(world.shelter_entrance_cells)
    world.tick = world.day_length + 2
    _teleport_spider(world, deep)
    world.state.hunger = 0.26
    world.state.fatigue = 0.34
    world.state.sleep_debt = 0.36
    chase_origin = (max(0, entrance[0] + 2), entrance[1])
    if not world.is_lizard_walkable(chase_origin):
        chase_origin = _safe_lizard_cell(world)
    world.lizard = LizardState(
        x=chase_origin[0],
        y=chase_origin[1],
        mode="CHASE",
        mode_ticks=4,
        last_known_spider=deep,
        investigate_target=chase_origin,
        recover_ticks=world.predator_controller.RECOVER_MOVES,
        wait_target=entrance,
        chase_streak=4,
        failed_chases=1,
    )
    world.refresh_memory(initial=True)


def _corridor_gauntlet(world: SpiderWorld) -> None:
    """
    Configure the world for the "corridor gauntlet" scenario: place the spider in a shelter corridor row, select a single food spawn farthest from that shelter, position a patrolling lizard on the corridor (or a safe fallback), set the world tick to 3, and refresh the world's memory.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    corridor_y = deep[1]
    world.tick = 3
    _teleport_spider(world, deep)
    world.state.hunger = 0.90
    world.state.fatigue = 0.24
    world.state.sleep_debt = 0.20
    far_food = sorted(
        world.map_template.food_spawn_cells,
        key=lambda cell: -world.manhattan(cell, deep),
    )[0]
    world.food_positions = [far_food]
    lx = max(deep[0] + 4, min(world.width - 2, far_food[0] - 2))
    ly = corridor_y
    if not world.is_lizard_walkable((lx, ly)):
        lx, ly = _safe_lizard_cell(world)
    world.lizard = LizardState(x=lx, y=ly, mode="PATROL")
    world.refresh_memory(initial=True)


def _two_shelter_tradeoff(world: SpiderWorld) -> None:
    """
    Configure the world for the "two shelter tradeoff" scenario.
    
    Places the spider at the leftmost deep shelter, sets the tick to day_length - 2 and spider physiology (hunger=0.68, fatigue=0.38, sleep_debt=0.42), selects a single food spawn nearest the rightmost deep shelter, places the lizard at the spawn cell closest to the map center in "PATROL" mode, and refreshes the world's memory.
    """
    left_deep = sorted(world.shelter_deep_cells)[0]
    right_deep = sorted(world.shelter_deep_cells)[-1]
    world.tick = world.day_length - 2
    _teleport_spider(world, left_deep)
    world.state.hunger = 0.68
    world.state.fatigue = 0.38
    world.state.sleep_debt = 0.42
    world.food_positions = [min(world.map_template.food_spawn_cells, key=lambda cell: world.manhattan(cell, right_deep))]
    center_spawn = min(
        world.map_template.lizard_spawn_cells,
        key=lambda cell: abs(cell[0] - world.width // 2) + abs(cell[1] - world.height // 2),
    )
    world.lizard = LizardState(x=center_spawn[0], y=center_spawn[1], mode="PATROL")
    world.refresh_memory(initial=True)


def _exposed_day_foraging(world: SpiderWorld) -> None:
    """
    Set up an exposed daytime foraging scenario by configuring the world state and entities.
    
    Mutates the given SpiderWorld in place: places the spider in a shelter cell (preferring deep, then interior, then entrance), sets the time to the start of day and scenario-specific hunger, fatigue, and sleep debt, spawns a single food at the food-spawn cell farthest from the spider, places a lizard at the lizard-spawn cell closest to that food in "PATROL" mode, and refreshes the world's memory.
    
    Parameters:
        world (SpiderWorld): The world instance to configure; modified in place.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = 1
    _teleport_spider(world, deep)
    world.state.hunger = 0.94
    world.state.fatigue = 0.16
    world.state.sleep_debt = 0.18
    world.food_positions = [max(world.map_template.food_spawn_cells, key=lambda cell: world.manhattan(cell, deep))]
    lizard_spawn = min(
        world.map_template.lizard_spawn_cells,
        key=lambda cell: world.manhattan(cell, world.food_positions[0]),
    )
    world.lizard = LizardState(x=lizard_spawn[0], y=lizard_spawn[1], mode="PATROL")
    world.refresh_memory(initial=True)


def _food_deprivation(world: SpiderWorld) -> None:
    """
    Configure the world for a food-deprivation scenario with the spider sheltered and acutely hungry.
    
    Mutates `world` in place: places the spider in a shelter and adjusts tick and physiological state to represent acute hunger, sets a single food spawn at the farthest food spawn from the spider, positions a lizard in "PATROL" at a safe spawn, and refreshes world memory with `initial=True`.
    
    Parameters:
        world (SpiderWorld): The world instance to mutate for the scenario.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = 4
    _teleport_spider(world, deep)
    world.state.hunger = FOOD_DEPRIVATION_INITIAL_HUNGER
    world.state.fatigue = 0.22
    world.state.sleep_debt = 0.18
    far_food = sorted(
        world.map_template.food_spawn_cells,
        key=lambda cell: -world.manhattan(cell, deep),
    )[0]
    world.food_positions = [far_food]
    lx, ly = _safe_lizard_cell(world)
    world.lizard = LizardState(x=lx, y=ly, mode="PATROL")
    world.refresh_memory(initial=True)


def _food_vs_predator_conflict(world: SpiderWorld) -> None:
    """
    Configure a daytime conflict where reachable food competes with a visible nearby predator.
    """
    world.tick = 2
    preferred_food = (5, 2)
    food = (
        preferred_food
        if preferred_food in world.map_template.food_spawn_cells
        else sorted(world.map_template.food_spawn_cells)[0]
    )
    preferred_spider = (2, 2)
    if (
        world.is_walkable(preferred_spider)
        and preferred_spider not in world.shelter_cells
        and preferred_spider != food
        and world.manhattan(preferred_spider, food) <= 3
    ):
        spider = preferred_spider
    else:
        spider_candidates = sorted(
            (
                cell
                for cell in world.map_template.traversable_cells
                if cell not in world.shelter_cells
                and cell != food
                and world.manhattan(cell, food) <= 3
            ),
            key=lambda cell: (
                world.manhattan(cell, food),
                cell[0],
                cell[1],
            ),
        )
        spider = spider_candidates[0] if spider_candidates else world.spider_pos()
    preferred_lizard = (2, 1)
    if (
        world.is_lizard_walkable(preferred_lizard)
        and preferred_lizard not in {spider, food}
        and 1 <= world.manhattan(preferred_lizard, spider) <= 2
    ):
        lizard = preferred_lizard
    else:
        lizard_candidates = sorted(
            (
                cell
                for cell in world.map_template.traversable_cells
                if world.is_lizard_walkable(cell)
                and cell not in {spider, food}
                and 1 <= world.manhattan(cell, spider) <= 2
            ),
            key=lambda cell: (
                world.manhattan(cell, spider),
                cell[0],
                cell[1],
            ),
        )
        lizard = lizard_candidates[0] if lizard_candidates else _safe_lizard_cell(world)
    _teleport_spider(world, spider)
    world.state.hunger = 0.92
    world.state.fatigue = 0.24
    world.state.sleep_debt = 0.18
    world.food_positions = [food]
    world.lizard = LizardState(x=lizard[0], y=lizard[1], mode="PATROL")
    world.state.heading_dx, world.state.heading_dy = world._heading_toward(world.lizard_pos())
    world.refresh_memory(initial=True)


SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT = 0.92


def _sleep_vs_exploration_conflict(world: SpiderWorld) -> None:
    """
    Set up a night-time scenario that prioritizes sleep over exploration.
    
    Places the spider at the shelter entrance, advances the clock to night, sets low hunger, high fatigue, and an elevated sleep debt, spawns a distant food source to discourage exploration, positions a non-threatening lizard on a safe patrol cell, and refreshes the world's memory.
    """
    entrance = _first_cell(world.shelter_entrance_cells or world.shelter_interior_cells)
    far_food = sorted(
        world.map_template.food_spawn_cells,
        key=lambda cell: -world.manhattan(cell, entrance),
    )[0]
    world.tick = world.day_length + 1
    _teleport_spider(world, entrance)
    world.state.hunger = 0.12
    world.state.fatigue = 0.94
    world.state.sleep_debt = SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT
    world.food_positions = [far_food]
    lx, ly = _safe_lizard_cell(world)
    world.lizard = LizardState(x=lx, y=ly, mode="PATROL")
    world.refresh_memory(initial=True)


def _trace_states(trace: Sequence[Dict[str, object]]) -> list[Dict[str, object]]:
    """
    Extract the `state` dictionaries from trace records that include a dictionary-valued `state` field.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Sequence of trace records (mappings) which may contain a `"state"` key.
    
    Returns:
        list[Dict[str, object]]: List of the `state` dictionaries extracted from trace items that had a dict in their `"state"` field.
    """
    return [
        item["state"]
        for item in trace
        if isinstance(item.get("state"), dict)
    ]


def _trace_any_mode(trace: Sequence[Dict[str, object]], mode: str) -> bool:
    """
    Checks whether any traced state records the lizard in the given mode.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Sequence of trace records (each may contain a `state` dict).
        mode (str): Lizard mode string to match (e.g., "PATROL", "CHASE").
    
    Returns:
        True if any traced state has `lizard_mode` equal to `mode`, False otherwise.
    """
    return any(state.get("lizard_mode") == mode for state in _trace_states(trace))


def _trace_any_sleep_phase(trace: Sequence[Dict[str, object]], phase: str) -> bool:
    """
    Check whether any recorded trace state has the specified sleep phase.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Execution trace sequence; each item may contain a `"state"` dict.
        phase (str): Sleep phase to look for (e.g., `"DEEP_SLEEP"`).
    
    Returns:
        bool: `true` if any traced state has `state.get("sleep_phase") == phase`, `false` otherwise.
    """
    return any(state.get("sleep_phase") == phase for state in _trace_states(trace))


def _trace_predator_memory_seen(trace: Sequence[Dict[str, object]]) -> bool:
    """
    Checks whether any state in the execution trace records a predator memory target.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Sequence of trace items (dictionaries) produced during an episode; each item may contain a `"state"` mapping.
    
    Returns:
        bool: `True` if any traced state's `predator_memory` is a dict with a non-`None` `"target"`, `False` otherwise.
    """
    for state in _trace_states(trace):
        predator_memory = state.get("predator_memory", {})
        if isinstance(predator_memory, dict) and predator_memory.get("target") is not None:
            return True
    return False


def _trace_escape_seen(trace: Sequence[Dict[str, object]]) -> bool:
    """
    Determine whether any trace item records a predator escape.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Sequence of trace items to inspect; each item may include a
            "predator_escape" key whose truthiness is evaluated.
    
    Returns:
        bool: `True` if any trace item has a truthy `"predator_escape"` value, `False` otherwise.
    """
    return any(bool(item.get("predator_escape")) for item in trace)


def _trace_action_selection_payloads(trace: Sequence[Dict[str, object]]) -> list[Dict[str, object]]:
    """
    Extract payload dictionaries from action-selection messages in a trace.
    
    Scans each trace item for a "messages" list and collects the "payload" dict from messages
    whose "sender" is "action_center" and whose "topic" is "action.selection".
    
    Parameters:
    	trace (Sequence[Dict[str, object]]): Sequence of trace item dictionaries as produced by the simulation.
    
    Returns:
    	list[Dict[str, object]]: A list of payload dictionaries extracted from matching messages.
    """
    payloads: list[Dict[str, object]] = []
    for item in trace:
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("sender") != "action_center":
                continue
            if message.get("topic") != "action.selection":
                continue
            payload = message.get("payload")
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads

def _payload_float(payload: Dict[str, Any], *path: str) -> float | None:
    """
    Extract a nested value from a payload by following the given key path and convert it to a float.

    Parameters:
        payload (Dict[str, Any]): Nested dictionary to traverse.
        *path (str): Sequence of keys describing the nested lookup path.

    Returns:
        float | None: The value at the end of the path converted to float, or `None` if any key
        is missing, an intermediate value is not a dict, or conversion fails.
    """
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None

def _payload_text(payload: Dict[str, Any], *path: str) -> str | None:
    """
    Extract a nested value from a payload dictionary by a sequence of keys and return it as text.

    Parameters:
        payload (Dict[str, Any]): The dictionary to traverse.
        *path (str): Sequence of keys describing the path to the desired nested value.

    Returns:
        str | None: The string representation of the located value, or `None` if any key is
        missing, an intermediate value is not a dictionary, or the final value is `None`.
    """
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return str(current) if current is not None else None


NIGHT_REST_INITIAL_SLEEP_DEBT = 0.60
FOOD_DEPRIVATION_INITIAL_HUNGER = 0.96


def _progress_band(*, food_distance_delta: float, food_eaten: int) -> str:
    if food_eaten > 0:
        return "consumed_food"
    if food_distance_delta > 0.0:
        return "advanced"
    if food_distance_delta < 0.0:
        return "regressed"
    return "stalled"


def _weak_scenario_diagnostics(
    *,
    food_distance_delta: float,
    food_eaten: int,
    hunger_reduction: float = 0.0,
    alive: bool,
    predator_contacts: int,
    full_success: bool,
) -> dict[str, object]:
    progress_band = _progress_band(
        food_distance_delta=float(food_distance_delta),
        food_eaten=int(food_eaten),
    )
    partial_progress = bool(
        food_eaten > 0
        or food_distance_delta > 0.0
        or hunger_reduction > 0.0
    )
    died_after_progress = bool((not alive) and partial_progress)
    died_without_contact = bool((not alive) and predator_contacts == 0)
    survived_without_progress = bool(alive and not partial_progress)
    if full_success:
        outcome_band = "full_success"
    elif alive:
        outcome_band = "survived_but_unfinished"
    elif progress_band == "regressed":
        outcome_band = "regressed_and_died"
    elif partial_progress:
        outcome_band = "partial_progress_died"
    else:
        outcome_band = "stalled_and_died"
    return {
        "progress_band": progress_band,
        "outcome_band": outcome_band,
        "partial_progress": partial_progress,
        "died_after_progress": died_after_progress,
        "died_without_contact": died_without_contact,
        "survived_without_progress": survived_without_progress,
    }


NIGHT_REST_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("deep_night_shelter", "Maintains deep shelter occupancy during the night.", ">= 0.95"),
    BehaviorCheckSpec("deep_sleep_reached", "Reaches deep sleep during the scenario.", "true"),
    BehaviorCheckSpec("sleep_debt_reduced", "Reduces sleep debt relative to the initial state.", ">= 0.45"),
)
PREDATOR_EDGE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("predator_detected", "Detects the predator at the edge of the visual field.", "true"),
    BehaviorCheckSpec("predator_memory_recorded", "Records explicit predator memory after the encounter.", "true"),
    BehaviorCheckSpec("predator_reacted", "Shows an observable behavioral response to the encounter.", "true"),
)
ENTRANCE_AMBUSH_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("survives_ambush", "Survives the entrance ambush.", "true"),
    BehaviorCheckSpec("avoids_contact", "Avoids direct contact with the predator at the entrance.", "true"),
    BehaviorCheckSpec("keeps_shelter_safety", "Preserves shelter safety or survival under entrance pressure.", ">= 0.75 occupancy or escape"),
)
OPEN_FIELD_FORAGING_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("made_food_progress", "Reduces distance to food in exposed terrain.", "> 0"),
    BehaviorCheckSpec("foraging_viable", "Eats or substantially reduces distance to food.", "eat or delta >= 2"),
    BehaviorCheckSpec("survives_exposure", "Survives the exposed foraging scenario.", "true"),
)
SHELTER_BLOCKADE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("survives_blockade", "Survives the nighttime entrance blockade.", "true"),
    BehaviorCheckSpec("avoids_blockade_contact", "Avoids direct contact during the blockade.", "true"),
    BehaviorCheckSpec("preserves_safety_or_escapes", "Preserves nighttime safety or records an escape from the predator.", "occupancy >= 0.75 or escape"),
)
RECOVER_AFTER_FAILED_CHASE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("predator_enters_recover", "The predator enters RECOVER after failing the chase.", "true"),
    BehaviorCheckSpec("predator_returns_to_wait", "The predator returns to WAIT after the chase.", "true"),
    BehaviorCheckSpec("spider_survives", "The spider survives the post-chase sequence.", "true"),
)
CORRIDOR_GAUNTLET_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("corridor_food_progress", "Reduces distance to food inside the corridor.", "> 0"),
    BehaviorCheckSpec("corridor_avoids_contact", "Avoids direct contact in the narrow corridor.", "true"),
    BehaviorCheckSpec("corridor_survives", "Survives the risk corridor.", "true"),
)
TWO_SHELTER_TRADEOFF_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("tradeoff_survives", "Survives the two-shelter choice scenario.", "true"),
    BehaviorCheckSpec("tradeoff_makes_progress", "Makes progress toward food or safety.", "food delta > 0 or occupancy >= 0.9"),
    BehaviorCheckSpec("tradeoff_night_shelter", "Maintains high nighttime shelter occupancy.", ">= 0.9"),
)
EXPOSED_DAY_FORAGING_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("day_food_progress", "Reduces distance to food during daytime.", "> 0"),
    BehaviorCheckSpec("day_avoids_contact", "Avoids direct contact with the predator while foraging.", "true"),
    BehaviorCheckSpec("day_survives", "Survives exposed daytime foraging.", "true"),
)
FOOD_DEPRIVATION_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("hunger_reduced", "Reduces hunger or manages to eat under deprivation.", "eat or hunger reduction >= 0.18"),
    BehaviorCheckSpec("approaches_food", "Makes progress toward food.", "> 0"),
    BehaviorCheckSpec("survives_deprivation", "Survives the deprivation episode.", "true"),
)
CONFLICT_PASS_RATE = 0.8
FOOD_VS_PREDATOR_CONFLICT_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("threat_priority", "Threat wins arbitration when the predator is visible and dangerous.", "winning_valence=threat"),
    BehaviorCheckSpec("foraging_suppressed_under_threat", "Foraging drive is suppressed while threat dominates.", "hunger gate < 0.5"),
    BehaviorCheckSpec("survives_without_contact", "Remains alive and avoids direct contact during the conflict.", "alive and predator_contacts=0"),
)
SLEEP_VS_EXPLORATION_CONFLICT_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("sleep_priority", "Sleep wins arbitration in a safe nighttime context with high sleep pressure.", "winning_valence=sleep"),
    BehaviorCheckSpec("exploration_suppressed_under_sleep_pressure", "Residual exploration is suppressed while sleep pressure remains high.", "visual/sensory gates reduced"),
    BehaviorCheckSpec("resting_behavior_emerges", "The spider stays in or returns to the shelter and enters useful rest.", "sleep event or sleep debt reduction"),
)


def _score_night_rest(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Assess night-rest performance for the "night rest" scenario.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute occupancy, stillness, sleep events, and final sleep debt.
        trace (Sequence[Dict[str, object]]): Execution trace entries used to detect sleep-phase events (e.g., "DEEP_SLEEP").
    
    Returns:
        BehavioralEpisodeScore: Score containing three checks (deep-shelter occupancy rate, presence of a DEEP_SLEEP phase, and sleep-debt reduction) and behavior_metrics with keys:
            - "deep_night_rate": proportion of nights spent in deep shelter
            - "night_stillness_rate": recorded stillness rate during night
            - "sleep_debt_reduction": reduction in sleep debt compared to the scenario baseline
            - "sleep_events": count of sleep events
    """
    deep_night_rate = float(stats.night_role_distribution.get("deep", 0.0))
    sleep_debt_reduction = float(max(0.0, NIGHT_REST_INITIAL_SLEEP_DEBT - stats.final_sleep_debt))
    deep_sleep_reached = _trace_any_sleep_phase(trace, "DEEP_SLEEP")
    checks = (
        build_behavior_check(NIGHT_REST_CHECKS[0], passed=deep_night_rate >= 0.95, value=deep_night_rate),
        build_behavior_check(NIGHT_REST_CHECKS[1], passed=deep_sleep_reached, value=deep_sleep_reached),
        build_behavior_check(NIGHT_REST_CHECKS[2], passed=sleep_debt_reduction >= 0.45, value=sleep_debt_reduction),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate safe, reproducible rest in deep shelter during the night.",
        checks=checks,
        behavior_metrics={
            "deep_night_rate": deep_night_rate,
            "night_stillness_rate": float(stats.night_stillness_rate),
            "sleep_debt_reduction": sleep_debt_reduction,
            "sleep_events": int(stats.sleep_events),
        },
    )


def _score_predator_edge(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate an episode according to the "predator edge" scenario and produce a behavioral score with per-check results and metrics.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute detections, alerts, responses, and mode transitions.
        trace (Sequence[Dict[str, object]]): Execution trace of state snapshots used to detect predator memory records and escape events.
    
    Returns:
        BehavioralEpisodeScore: Score object containing three behavior checks (predator detected, predator memory recorded, predator reacted) and a metrics dictionary with keys:
        - `predator_sightings`: number of predator sightings,
        - `alert_events`: number of alert events,
        - `predator_response_events`: number of predator response events,
        - `predator_mode_transitions`: number of predator mode transitions.
    """
    predator_detected = bool(stats.predator_sightings > 0 or stats.alert_events > 0)
    predator_memory_recorded = _trace_predator_memory_seen(trace)
    predator_reacted = bool(stats.predator_response_events > 0 or stats.predator_mode_transitions > 0 or _trace_escape_seen(trace))
    checks = (
        build_behavior_check(PREDATOR_EDGE_CHECKS[0], passed=predator_detected, value=predator_detected),
        build_behavior_check(PREDATOR_EDGE_CHECKS[1], passed=predator_memory_recorded, value=predator_memory_recorded),
        build_behavior_check(PREDATOR_EDGE_CHECKS[2], passed=predator_reacted, value=predator_reacted),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate predator detection and response during a controlled encounter at the visual edge.",
        checks=checks,
        behavior_metrics={
            "predator_sightings": int(stats.predator_sightings),
            "alert_events": int(stats.alert_events),
            "predator_response_events": int(stats.predator_response_events),
            "predator_mode_transitions": int(stats.predator_mode_transitions),
        },
    )


def _score_entrance_ambush(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate an episode for the "entrance ambush" scenario and produce per-check results and behavior metrics.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to derive checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace entries used to detect predator escape events.
    
    Returns:
        BehavioralEpisodeScore: Score containing three behavior checks (survival, zero predator contacts, shelter safety)
        and metrics: `survival`, `predator_contacts`, `night_shelter_occupancy_rate`, and `predator_escapes`.
    """
    shelter_safety = float(max(stats.night_shelter_occupancy_rate, 1.0 if _trace_escape_seen(trace) else 0.0))
    checks = (
        build_behavior_check(ENTRANCE_AMBUSH_CHECKS[0], passed=bool(stats.alive), value=bool(stats.alive)),
        build_behavior_check(ENTRANCE_AMBUSH_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(ENTRANCE_AMBUSH_CHECKS[2], passed=shelter_safety >= 0.75, value=shelter_safety),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate survival and preserved safety under an ambush at the shelter entrance.",
        checks=checks,
        behavior_metrics={
            "survival": bool(stats.alive),
            "predator_contacts": int(stats.predator_contacts),
            "night_shelter_occupancy_rate": float(stats.night_shelter_occupancy_rate),
            "predator_escapes": int(stats.predator_escapes),
        },
    )


def _score_open_field_foraging(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate open-field foraging performance and produce a BehavioralEpisodeScore containing progress, viability, and survival checks.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute food progress, food eaten, alive status, and predator contact metrics.
        trace (Sequence[Dict[str, object]]): Execution trace (ignored by this scorer).
    
    Returns:
        BehavioralEpisodeScore: A score composed of three checks (food distance progress, foraging viability, and alive) and the following behavior metrics:
            - `food_distance_delta`: numeric progress toward food,
            - `food_eaten`: number of food items consumed,
            - `alive`: whether the agent survived the episode,
            - `predator_contacts`: number of predator contact events.
    """
    del trace
    food_progress = float(stats.food_distance_delta)
    foraging_viable = bool(stats.food_eaten > 0 or food_progress >= 2.0)
    checks = (
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[0], passed=food_progress > 0.0, value=food_progress),
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[1], passed=foraging_viable, value=bool(foraging_viable)),
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=food_progress,
        food_eaten=int(stats.food_eaten),
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate reproducible open-field foraging with an explicit progress metric.",
        checks=checks,
        behavior_metrics={
            "food_distance_delta": food_progress,
            "food_eaten": int(stats.food_eaten),
            "alive": bool(stats.alive),
            "predator_contacts": int(stats.predator_contacts),
            **diagnostics,
        },
    )


def _score_shelter_blockade(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Assess the episode for the shelter-blockade scenario and build a BehavioralEpisodeScore.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to evaluate survival, contacts, and occupancy.
        trace (Sequence[Dict[str, object]]): Execution trace states inspected for predator escape events that influence shelter safety.
    
    Returns:
        BehavioralEpisodeScore: Score object containing three behavior checks (alive, zero predator contacts, shelter safety threshold) and metrics:
            - night_shelter_occupancy_rate (float)
            - predator_escapes (int)
            - predator_contacts (int)
            - alive (bool)
    """
    safety_score = float(max(stats.night_shelter_occupancy_rate, 1.0 if _trace_escape_seen(trace) else 0.0))
    checks = (
        build_behavior_check(SHELTER_BLOCKADE_CHECKS[0], passed=bool(stats.alive), value=bool(stats.alive)),
        build_behavior_check(SHELTER_BLOCKADE_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(SHELTER_BLOCKADE_CHECKS[2], passed=safety_score >= 0.75, value=safety_score),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate safe, reproducible behavior when the shelter entrance is blocked.",
        checks=checks,
        behavior_metrics={
            "night_shelter_occupancy_rate": float(stats.night_shelter_occupancy_rate),
            "predator_escapes": int(stats.predator_escapes),
            "predator_contacts": int(stats.predator_contacts),
            "alive": bool(stats.alive),
        },
    )


def _score_recover_after_failed_chase(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Assess whether the predator entered RECOVER and subsequently returned to WAIT after a failed chase.
    
    Builds three behavior checks (entered RECOVER, returned to WAIT, spider alive) and aggregates related behavior metrics.
    
    Returns:
        BehavioralEpisodeScore: Score object containing the three checks and a `behavior_metrics` mapping with keys:
            - "entered_recover" (bool)
            - "returned_to_wait" (bool)
            - "predator_mode_transitions" (int)
            - "alive" (bool)
    """
    entered_recover = _trace_any_mode(trace, "RECOVER")
    returned_to_wait = _trace_any_mode(trace, "WAIT")
    checks = (
        build_behavior_check(RECOVER_AFTER_FAILED_CHASE_CHECKS[0], passed=entered_recover, value=entered_recover),
        build_behavior_check(RECOVER_AFTER_FAILED_CHASE_CHECKS[1], passed=returned_to_wait, value=returned_to_wait),
        build_behavior_check(RECOVER_AFTER_FAILED_CHASE_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate the predator's deterministic transition into RECOVER and WAIT after a failed chase.",
        checks=checks,
        behavior_metrics={
            "entered_recover": entered_recover,
            "returned_to_wait": returned_to_wait,
            "predator_mode_transitions": int(stats.predator_mode_transitions),
            "alive": bool(stats.alive),
        },
    )


def _score_corridor_gauntlet(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate corridor gauntlet episode performance by measuring food approach progress, predator contacts, and survival.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace (ignored by this scorer).
    
    Returns:
        BehavioralEpisodeScore: Score object containing three behavior checks (food progress, zero predator contacts, alive)
        and metrics: `food_distance_delta`, `predator_contacts`, `alive`, and `predator_mode_transitions`.
    """
    del trace
    food_progress = float(stats.food_distance_delta)
    checks = (
        build_behavior_check(CORRIDOR_GAUNTLET_CHECKS[0], passed=food_progress > 0.0, value=food_progress),
        build_behavior_check(CORRIDOR_GAUNTLET_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(CORRIDOR_GAUNTLET_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=food_progress,
        food_eaten=int(stats.food_eaten),
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate narrow-corridor progress without relying only on aggregated reward.",
        checks=checks,
        behavior_metrics={
            "food_distance_delta": food_progress,
            "predator_contacts": int(stats.predator_contacts),
            "alive": bool(stats.alive),
            "predator_mode_transitions": int(stats.predator_mode_transitions),
            **diagnostics,
        },
    )


def _score_two_shelter_tradeoff(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Compute the behavioral score and check results for the two-shelter tradeoff scenario.
    
    Builds three checks: survival (alive), progress toward food or shelter occupancy (passes if food distance improved or occupancy >= 0.9), and high shelter occupancy (occupancy >= 0.9). Returns a BehavioralEpisodeScore containing those checks and metrics derived from EpisodeStats.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to evaluate checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace (not used by this scorer).
    
    Returns:
        BehavioralEpisodeScore: Score bundle including the three check results and the following behavior_metrics:
            - "food_distance_delta" (float): change in distance to food.
            - "night_shelter_occupancy_rate" (float): fraction of nights spent in the alternative shelter.
            - "alive" (bool): whether the agent finished the episode alive.
            - "final_health" (float): agent's final health value.
    """
    del trace
    progress_score = float(max(stats.food_distance_delta, stats.night_shelter_occupancy_rate))
    checks = (
        build_behavior_check(TWO_SHELTER_TRADEOFF_CHECKS[0], passed=bool(stats.alive), value=bool(stats.alive)),
        build_behavior_check(TWO_SHELTER_TRADEOFF_CHECKS[1], passed=(stats.food_distance_delta > 0.0 or stats.night_shelter_occupancy_rate >= 0.9), value=progress_score),
        build_behavior_check(TWO_SHELTER_TRADEOFF_CHECKS[2], passed=stats.night_shelter_occupancy_rate >= 0.9, value=float(stats.night_shelter_occupancy_rate)),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate the behavioral trade-off between progress toward food and safe occupancy of an alternative shelter.",
        checks=checks,
        behavior_metrics={
            "food_distance_delta": float(stats.food_distance_delta),
            "night_shelter_occupancy_rate": float(stats.night_shelter_occupancy_rate),
            "alive": bool(stats.alive),
            "final_health": float(stats.final_health),
        },
    )


def _score_exposed_day_foraging(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Assess exposed-day foraging performance based on food-distance progress and predator contacts.
    
    The execution trace is ignored by this scorer.
    
    Returns:
        BehavioralEpisodeScore: Aggregated score containing three checks (positive food-distance progress, zero predator contacts, and survival) and a behavior_metrics dictionary with keys:
            - `food_distance_delta`: float progress in food distance,
            - `predator_contacts`: int number of predator contacts,
            - `alive`: bool whether the agent survived,
            - `final_health`: float final health value,
          plus additional diagnostic fields produced by the scenario diagnostics helper.
    """
    del trace
    food_progress = float(stats.food_distance_delta)
    checks = (
        build_behavior_check(EXPOSED_DAY_FORAGING_CHECKS[0], passed=food_progress > 0.0, value=food_progress),
        build_behavior_check(EXPOSED_DAY_FORAGING_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(EXPOSED_DAY_FORAGING_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=food_progress,
        food_eaten=int(stats.food_eaten),
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate exposed daytime foraging with a reproducible progress metric.",
        checks=checks,
        behavior_metrics={
            "food_distance_delta": food_progress,
            "predator_contacts": int(stats.predator_contacts),
            "alive": bool(stats.alive),
            "final_health": float(stats.final_health),
            **diagnostics,
        },
    )


def _score_food_deprivation(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate agent recovery and foraging progress for the food deprivation scenario.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode metrics used to build checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace (ignored by this scorer).
    
    Returns:
        BehavioralEpisodeScore: Score containing the scenario objective, per-check results, and behavior metrics including:
            - food_eaten: number of food items consumed during the episode.
            - food_distance_delta: change in distance to food (positive indicates progress toward food).
            - hunger_reduction: amount hunger decreased from the scenario initial value (clamped at zero).
            - alive: whether the agent survived the episode.
            - plus diagnostic fields produced by the scenario diagnostic helper.
    """
    del trace
    hunger_reduction = float(max(0.0, FOOD_DEPRIVATION_INITIAL_HUNGER - stats.final_hunger))
    checks = (
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[0], passed=(stats.food_eaten > 0 or hunger_reduction >= 0.18), value=hunger_reduction),
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[1], passed=stats.food_distance_delta > 0.0, value=float(stats.food_distance_delta)),
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=float(stats.food_distance_delta),
        food_eaten=int(stats.food_eaten),
        hunger_reduction=hunger_reduction,
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate measurable recovery or food progress under homeostatic deprivation.",
        checks=checks,
        behavior_metrics={
            "food_eaten": int(stats.food_eaten),
            "food_distance_delta": float(stats.food_distance_delta),
            "hunger_reduction": hunger_reduction,
            "alive": bool(stats.alive),
            **diagnostics,
        },
    )


def _score_food_vs_predator_conflict(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate whether predator threat overrides food-seeking and whether the agent survives without predator contact.
    
    Identifies action-selection payloads that signify a dangerous predator (predator visibility ≥ 0.5 and either proximity ≥ 0.3 or certainty ≥ 0.35), then computes three checks:
    - whether the selected valence favors `threat`,
    - whether hunger gating is suppressed under threat,
    - whether the agent survives with zero predator contacts.
    
    The returned score includes these checks and behavior metrics: `danger_tick_count`, `threat_priority_rate`, `mean_hunger_gate_under_threat`, `predator_contacts`, and `alive`.
    
    Parameters:
        stats (EpisodeStats): Episode summary statistics used for survival and contact checks.
        trace (Sequence[Dict[str, object]]): Execution trace from which action-selection payloads are extracted.
    
    Returns:
        BehavioralEpisodeScore: A score object containing the scenario objective, the three checks, and the computed behavior metrics.
    """
    payloads = _trace_action_selection_payloads(trace)
    dangerous_payloads: list[Dict[str, object]] = []
    for payload in payloads:
        predator_visible = _payload_float(payload, "evidence", "threat", "predator_visible")
        predator_proximity = _payload_float(payload, "evidence", "threat", "predator_proximity")
        predator_certainty = _payload_float(payload, "evidence", "threat", "predator_certainty")
        if predator_visible is None or predator_proximity is None or predator_certainty is None:
            continue
        if predator_visible >= 0.5 and (
            predator_proximity >= 0.3 or predator_certainty >= 0.35
        ):
            dangerous_payloads.append(payload)
    threat_priority_rate = float(
        sum(
            1.0
            for payload in dangerous_payloads
            if _payload_text(payload, "winning_valence") == "threat"
        ) / max(1, len(dangerous_payloads))
    )
    dangerous_hunger_gates = [
        hunger_gate
        for payload in dangerous_payloads
        if (hunger_gate := _payload_float(payload, "module_gates", "hunger_center")) is not None
    ]
    threat_wins = sum(
        1
        for payload in dangerous_payloads
        if _payload_text(payload, "winning_valence") == "threat"
    )
    foraging_suppressed_rate = float(
        sum(
            1.0
            for payload in dangerous_payloads
            if _payload_text(payload, "winning_valence") == "threat"
            and (hunger_gate := _payload_float(payload, "module_gates", "hunger_center")) is not None
            and hunger_gate < 0.5
        ) / max(1, threat_wins)
    )
    threat_priority = bool(
        len(dangerous_payloads) >= 1 and threat_priority_rate >= CONFLICT_PASS_RATE
    )
    foraging_suppressed = bool(
        len(dangerous_hunger_gates) >= 1
        and foraging_suppressed_rate >= CONFLICT_PASS_RATE
    )
    survives_without_contact = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(FOOD_VS_PREDATOR_CONFLICT_CHECKS[0], passed=threat_priority, value=threat_priority),
        build_behavior_check(FOOD_VS_PREDATOR_CONFLICT_CHECKS[1], passed=foraging_suppressed, value=foraging_suppressed),
        build_behavior_check(
            FOOD_VS_PREDATOR_CONFLICT_CHECKS[2],
            passed=survives_without_contact,
            value={"alive": bool(stats.alive), "predator_contacts": int(stats.predator_contacts)},
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate that threat overrides food drive when both compete at the same moment.",
        checks=checks,
        behavior_metrics={
            "danger_tick_count": len(dangerous_payloads),
            "threat_priority_rate": threat_priority_rate,
            "foraging_suppressed_rate": foraging_suppressed_rate,
            "mean_hunger_gate_under_threat": float(
                sum(dangerous_hunger_gates) / max(1, len(dangerous_hunger_gates))
            ),
            "predator_contacts": int(stats.predator_contacts),
            "alive": bool(stats.alive),
        },
    )


def _score_sleep_vs_exploration_conflict(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate whether sleep motivation overrides exploration in a nocturnal, low-threat context.
    
    Analyzes action-selection trace payloads to detect ticks with high sleep evidence and low predator visibility, then builds three behavioral checks:
    - whether sleep valence was selected when sleep pressure was high,
    - whether exploration-related gates (visual and sensory) were suppressed under those sleep-priority ticks,
    - whether the agent exhibited resting behavior (sleep event or sufficient sleep-debt reduction).
    
    Parameters:
        stats (EpisodeStats): Aggregate statistics collected for the episode (e.g., sleep events, final sleep debt).
        trace (Sequence[Dict[str, object]]): Execution trace of messages and states produced during the episode.
    
    Returns:
        BehavioralEpisodeScore: Score object containing the scenario objective, the three checks above, and behavior_metrics including:
            - `sleep_pressure_tick_count`: count of ticks meeting sleep-pressure criteria,
            - `sleep_priority_rate`: fraction of those ticks where `sleep` was the winning valence,
            - `mean_visual_gate_under_sleep`: mean visual gate value during sleep-pressure ticks,
            - `sleep_events`: number of sleep events recorded,
            - `sleep_debt_reduction`: amount of sleep-debt reduced from the scenario initial value.
    """
    payloads = _trace_action_selection_payloads(trace)
    sleepy_payloads: list[Dict[str, object]] = []
    for payload in payloads:
        sleep_debt = _payload_float(payload, "evidence", "sleep", "sleep_debt")
        fatigue = _payload_float(payload, "evidence", "sleep", "fatigue")
        predator_visible = _payload_float(payload, "evidence", "threat", "predator_visible")
        if sleep_debt is None or fatigue is None or predator_visible is None:
            continue
        if sleep_debt >= 0.6 and fatigue >= 0.6 and predator_visible < 0.5:
            sleepy_payloads.append(payload)
    sleep_priority_rate = float(
        sum(
            1.0
            for payload in sleepy_payloads
            if _payload_text(payload, "winning_valence") == "sleep"
        ) / max(1, len(sleepy_payloads))
    )
    sleepy_gate_payloads = [
        payload
        for payload in sleepy_payloads
        if _payload_float(payload, "module_gates", "visual_cortex") is not None
        and _payload_float(payload, "module_gates", "sensory_cortex") is not None
    ]
    sleep_priority_gate_payloads = [
        payload
        for payload in sleepy_gate_payloads
        if _payload_text(payload, "winning_valence") == "sleep"
    ]
    exploration_suppressed_rate = float(
        sum(
            1.0
            for payload in sleep_priority_gate_payloads
            if (visual_gate := _payload_float(payload, "module_gates", "visual_cortex")) is not None
            and (sensory_gate := _payload_float(payload, "module_gates", "sensory_cortex")) is not None
            and visual_gate < 0.6
            and sensory_gate < 0.7
        ) / max(1, len(sleep_priority_gate_payloads))
    )
    visual_gates_under_sleep = [
        visual_gate
        for payload in sleep_priority_gate_payloads
        if (visual_gate := _payload_float(payload, "module_gates", "visual_cortex")) is not None
    ]
    sleep_priority = bool(
        len(sleepy_payloads) >= 1 and sleep_priority_rate >= CONFLICT_PASS_RATE
    )
    exploration_suppressed = bool(
        len(sleep_priority_gate_payloads) >= 1
        and exploration_suppressed_rate >= CONFLICT_PASS_RATE
    )
    resting_behavior = bool(
        stats.sleep_events > 0
        or stats.final_sleep_debt <= (SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - 0.12)
    )
    checks = (
        build_behavior_check(SLEEP_VS_EXPLORATION_CONFLICT_CHECKS[0], passed=sleep_priority, value=sleep_priority),
        build_behavior_check(
            SLEEP_VS_EXPLORATION_CONFLICT_CHECKS[1],
            passed=exploration_suppressed,
            value=exploration_suppressed,
        ),
        build_behavior_check(
            SLEEP_VS_EXPLORATION_CONFLICT_CHECKS[2],
            passed=resting_behavior,
            value={
                "sleep_events": int(stats.sleep_events),
                "sleep_debt_reduction": float(
                    max(0.0, SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - stats.final_sleep_debt)
                ),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate that sleep pressure overrides residual exploration in a safe nighttime context.",
        checks=checks,
        behavior_metrics={
            "sleep_pressure_tick_count": len(sleepy_payloads),
            "sleep_priority_rate": sleep_priority_rate,
            "exploration_suppressed_rate": exploration_suppressed_rate,
            "mean_visual_gate_under_sleep": float(
                sum(visual_gates_under_sleep) / max(1, len(visual_gates_under_sleep))
            ),
            "sleep_events": int(stats.sleep_events),
            "sleep_debt_reduction": float(
                max(0.0, SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - stats.final_sleep_debt)
            ),
        },
    )


SCENARIOS: Dict[str, ScenarioSpec] = {
    "night_rest": ScenarioSpec(
        name="night_rest",
        description="Tired spider resting safely in deep shelter at night.",
        objective="Confirm safe rest with deep occupancy and reduced sleep debt.",
        behavior_checks=NIGHT_REST_CHECKS,
        diagnostic_focus="Deep rest, nighttime stillness, and reduced sleep debt.",
        success_interpretation="Success requires consistent deep occupancy and clear sleep recovery.",
        failure_interpretation="Failure suggests unstable rest, persistent sleep debt, or loss of deep shelter occupancy.",
        budget_note="This scenario already tends to pass under the short budget and serves as a positive rest control.",
        max_steps=12,
        map_template="central_burrow",
        setup=_night_rest,
        score_episode=_score_night_rest,
    ),
    "predator_edge": ScenarioSpec(
        name="predator_edge",
        description="Predator appears at the edge of the visual field.",
        objective="Confirm predator detection and response with reproducible explicit memory.",
        behavior_checks=PREDATOR_EDGE_CHECKS,
        diagnostic_focus="Predator detection, explicit memory, and observable encounter response.",
        success_interpretation="Success requires a perceived encounter, recorded memory, and a behavioral response.",
        failure_interpretation="Failure indicates a problem with detection, explicit memory, or encounter response.",
        budget_note="Works as a short sanity-check scenario for perception and predator response.",
        max_steps=18,
        map_template="central_burrow",
        setup=_predator_edge,
        score_episode=_score_predator_edge,
    ),
    "entrance_ambush": ScenarioSpec(
        name="entrance_ambush",
        description="Lizard waiting near the shelter entrance.",
        objective="Measure survival and shelter safety under an entrance ambush.",
        behavior_checks=ENTRANCE_AMBUSH_CHECKS,
        diagnostic_focus="Survival, avoided contact, and preserved shelter safety.",
        success_interpretation="Success requires surviving without direct contact while preserving a safe shelter or escape route.",
        failure_interpretation="Failure suggests a breach in entrance safety or an inability to sustain refuge.",
        budget_note="This scenario already tends to pass under the short budget and acts as a shelter-safety control.",
        max_steps=18,
        map_template="entrance_funnel",
        setup=_entrance_ambush,
        score_episode=_score_entrance_ambush,
    ),
    "open_field_foraging": ScenarioSpec(
        name="open_field_foraging",
        description="Food placed far from shelter in exposed terrain.",
        objective="Measure progress or success in a controlled open-field foraging task.",
        behavior_checks=OPEN_FIELD_FORAGING_CHECKS,
        diagnostic_focus="Separate real food progress from spatial regression and death in open terrain.",
        success_interpretation="Success requires viable progress toward food while surviving exposure.",
        failure_interpretation="Failure may mean regression away from food, stalling, or dying before consolidating foraging.",
        budget_note="Under the short budget, this scenario often exposes spatial regression and contact-free death as distinct failure signals.",
        max_steps=20,
        map_template="exposed_feeding_ground",
        setup=_open_field_foraging,
        score_episode=_score_open_field_foraging,
    ),
    "shelter_blockade": ScenarioSpec(
        name="shelter_blockade",
        description="Lizard blocking the shelter entrance at night.",
        objective="Validate preserved safety or escape under a nighttime route blockade.",
        behavior_checks=SHELTER_BLOCKADE_CHECKS,
        diagnostic_focus="Nighttime survival, avoided contact, and preserved shelter safety.",
        success_interpretation="Success requires survival and either preserved nighttime safety or an observable escape.",
        failure_interpretation="Failure suggests contact under blockade or loss of shelter safety during the night.",
        budget_note="Intermediate nighttime-pressure scenario that is useful for comparing sheltering versus escape.",
        max_steps=16,
        map_template="entrance_funnel",
        setup=_shelter_blockade,
        score_episode=_score_shelter_blockade,
    ),
    "recover_after_failed_chase": ScenarioSpec(
        name="recover_after_failed_chase",
        description="Lizard must exit ambush and enter recovery after losing the spider.",
        objective="Measure the predator's behavioral transition after a failed chase.",
        behavior_checks=RECOVER_AFTER_FAILED_CHASE_CHECKS,
        diagnostic_focus="Predator RECOVER -> WAIT sequence and spider survival after the chase.",
        success_interpretation="Success requires the predator's deterministic transition and episode survival.",
        failure_interpretation="Failure indicates a predator FSM regression or a lethal outcome for the spider.",
        budget_note="Short regression scenario for the predator FSM, less sensitive to training budget.",
        max_steps=14,
        map_template="entrance_funnel",
        setup=_recover_after_failed_chase,
        score_episode=_score_recover_after_failed_chase,
    ),
    "corridor_gauntlet": ScenarioSpec(
        name="corridor_gauntlet",
        description="Hungry spider crosses a narrow corridor with the lizard on the escape axis.",
        objective="Measure food progress and safety in a controlled risk corridor.",
        behavior_checks=CORRIDOR_GAUNTLET_CHECKS,
        diagnostic_focus="Distinguish safe stalling, partial progress, and death in the narrow corridor.",
        success_interpretation="Success requires advancing through the corridor, avoiding contact, and surviving the risk.",
        failure_interpretation="Failure may mean stalling, dying after partial progress, or showing no response under pressure.",
        budget_note="Under the short budget, this scenario often shows avoided contact without enough progress for full success.",
        max_steps=20,
        map_template="corridor_escape",
        setup=_corridor_gauntlet,
        score_episode=_score_corridor_gauntlet,
    ),
    "two_shelter_tradeoff": ScenarioSpec(
        name="two_shelter_tradeoff",
        description="Map with two shelters forcing a choice between food and a refuge route.",
        objective="Measure the trade-off between food exploration and safety across two shelters.",
        behavior_checks=TWO_SHELTER_TRADEOFF_CHECKS,
        diagnostic_focus="Trade-off between food progress and maintaining an alternative shelter.",
        success_interpretation="Success requires survival, preserved nighttime shelter use, and useful progress.",
        failure_interpretation="Failure suggests a poor choice between exploration and safety or inability to sustain shelter.",
        budget_note="Useful scenario for comparing policies even when the short budget remains limited.",
        max_steps=24,
        map_template="two_shelters",
        setup=_two_shelter_tradeoff,
        score_episode=_score_two_shelter_tradeoff,
    ),
    "exposed_day_foraging": ScenarioSpec(
        name="exposed_day_foraging",
        description="Daytime foraging with food in open terrain and a nearby lizard patrol.",
        objective="Measure daytime foraging progress in open terrain under a nearby patrol.",
        behavior_checks=EXPOSED_DAY_FORAGING_CHECKS,
        diagnostic_focus="Separate retreat, stalling, and death during exposed daytime foraging.",
        success_interpretation="Success requires progress toward food without contact while staying alive under the nearby patrol.",
        failure_interpretation="Failure may mean retreating away from food, lack of useful progress, or dying before the goal.",
        budget_note="Under the short budget, this scenario often reveals retreat and death even without direct predator contact.",
        max_steps=20,
        map_template="exposed_feeding_ground",
        setup=_exposed_day_foraging,
        score_episode=_score_exposed_day_foraging,
    ),
    "food_deprivation": ScenarioSpec(
        name="food_deprivation",
        description="Spider starts with acute hunger and must recover homeostasis with distant food.",
        objective="Measure homeostatic recovery or explicit food progress under deprivation.",
        behavior_checks=FOOD_DEPRIVATION_CHECKS,
        diagnostic_focus="Separate food approach from effective homeostatic recovery under deprivation.",
        success_interpretation="Success requires measurable hunger reduction, progress toward food, and survival.",
        failure_interpretation="Failure can hide real partial progress when the spider approaches food but dies before reducing hunger.",
        budget_note="Under the short budget, this scenario often records useful approach without full homeostatic recovery.",
        max_steps=22,
        map_template="central_burrow",
        setup=_food_deprivation,
        score_episode=_score_food_deprivation,
    ),
    "food_vs_predator_conflict": ScenarioSpec(
        name="food_vs_predator_conflict",
        description="Nearby food competes with a visible predator in open terrain.",
        objective="Validate that threat overrides foraging when the conflict is explicit and immediate.",
        behavior_checks=FOOD_VS_PREDATOR_CONFLICT_CHECKS,
        diagnostic_focus="Threat-versus-hunger arbitration, down-weighted foraging, and contact-free survival.",
        success_interpretation="Success requires threat priority, suppressed food drive, and survival without contact.",
        failure_interpretation="Failure suggests opaque arbitration or persistent foraging under visible risk.",
        budget_note="Short deterministic scenario focused on checking priority gating in the food-versus-predator conflict.",
        max_steps=16,
        map_template="exposed_feeding_ground",
        setup=_food_vs_predator_conflict,
        score_episode=_score_food_vs_predator_conflict,
    ),
    "sleep_vs_exploration_conflict": ScenarioSpec(
        name="sleep_vs_exploration_conflict",
        description="Safe night with high fatigue and sleep debt, but residual exploration is still possible.",
        objective="Validate that sleep overrides residual exploration when the context is safe and homeostatic pressure is high.",
        behavior_checks=SLEEP_VS_EXPLORATION_CONFLICT_CHECKS,
        diagnostic_focus="Sleep-versus-exploration arbitration, reduced exploration gates, and useful shelter rest.",
        success_interpretation="Success requires sleep priority, suppressed residual exploration, and observable rest.",
        failure_interpretation="Failure suggests residual exploration still dominates despite strong sleep pressure.",
        budget_note="Short homeostatic-conflict scenario for validating the semantics of sleep gating.",
        max_steps=14,
        map_template="central_burrow",
        setup=_sleep_vs_exploration_conflict,
        score_episode=_score_sleep_vs_exploration_conflict,
    ),
}

SCENARIO_NAMES: Sequence[str] = tuple(SCENARIOS.keys())


def get_scenario(name: str) -> ScenarioSpec:
    """
    Retrieve a ScenarioSpec from the module registry by name.
    
    Parameters:
        name (str): The registry key identifying the desired scenario.
    
    Returns:
        ScenarioSpec: The scenario specification associated with `name`.
    
    Raises:
        ValueError: If `name` is not a known scenario key.
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    return SCENARIOS[name]
