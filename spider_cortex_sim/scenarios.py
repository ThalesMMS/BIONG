from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Sequence

from .interfaces import ACTION_DELTAS
from .maps import build_map_template
from .metrics import (
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    build_behavior_check,
    build_behavior_score,
)
from .predator import (
    LizardState,
    OLFACTORY_HUNTER_PROFILE,
    VISUAL_HUNTER_PROFILE,
)
from .world import SpiderWorld


BehaviorScoreFn = Callable[[EpisodeStats, Sequence[Dict[str, object]]], BehavioralEpisodeScore]

TRACE_MAP_WIDTH_FALLBACK = 12
TRACE_MAP_HEIGHT_FALLBACK = 12
FAST_VISUAL_HUNTER_PROFILE = replace(
    VISUAL_HUNTER_PROFILE,
    name="fast_visual_hunter",
    move_interval=1,
    detection_threshold=0.45,
)


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


def _food_visible_from_cell(
    world: SpiderWorld,
    source: tuple[int, int],
    food: tuple[int, int],
) -> bool:
    """Return whether `food` is visibly detectable from `source`."""
    radius = world._visible_range()
    distance = world.manhattan(source, food)
    if distance > radius or not world._has_line_of_sight(source, food):
        return False
    threshold = float(world.operational_profile.perception["visibility_binary_threshold"])
    confidence = world._visibility_confidence(
        source=source,
        target=food,
        dist=distance,
        radius=radius,
    )
    return bool(confidence >= threshold)


def _first_food_visible_frontier(
    world: SpiderWorld,
    spider: tuple[int, int],
    food: tuple[int, int],
) -> list[tuple[int, int]]:
    """Return the nearest walkable cells where the spider first sees `food`."""
    visited = {spider}
    frontier = [spider]
    while frontier:
        visible_frontier = [
            cell for cell in frontier if _food_visible_from_cell(world, cell, food)
        ]
        if visible_frontier:
            return visible_frontier
        next_frontier: list[tuple[int, int]] = []
        for cell in frontier:
            for dx, dy in world.move_deltas:
                candidate = (cell[0] + dx, cell[1] + dy)
                if candidate in visited or not world.is_walkable(candidate):
                    continue
                visited.add(candidate)
                next_frontier.append(candidate)
        frontier = next_frontier
    return []


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


def _open_field_foraging_food_cell(
    world: SpiderWorld,
    deep: tuple[int, int],
) -> tuple[int, int]:
    """
    Choose a food spawn intended for open-field foraging that is reachable from the given deep shelter and biased toward an exposed direction.
    
    Parameters:
        world (SpiderWorld): The world containing map template and distance utilities.
        deep (tuple[int, int]): Coordinates of the deep shelter interior used as the foraging origin.
    
    Returns:
        tuple[int, int]: Coordinates of the selected food spawn cell.
    """
    food_spawns = list(world.map_template.food_spawn_cells)
    candidates = [
        cell
        for cell in food_spawns
        if 4 <= world.manhattan(cell, deep) <= 6
        and cell[0] > deep[0]
        and abs(cell[1] - deep[1]) > abs(cell[0] - deep[0])
    ]
    if candidates:
        return min(
            candidates,
            key=lambda cell: (
                abs(world.manhattan(cell, deep) - 5),
                cell[1],
                cell[0],
            ),
        )
    return sorted(
        food_spawns,
        key=lambda cell: -world.manhattan(cell, deep),
    )[0]


def _teleport_spider(
    world: SpiderWorld,
    position: tuple[int, int],
    *,
    reset_heading: bool = True,
) -> None:
    """
    Set the spider's map coordinates to `position` and, if requested, reset its heading.
    
    Parameters:
        position (tuple[int, int]): Target (x, y) cell coordinates to place the spider.
        reset_heading (bool): If True, call the world's heading reset routine after teleporting; if False, preserve the current heading.
    """
    world.state.x, world.state.y = position
    if reset_heading:
        world._reset_heading_after_teleport()


def _set_predators(world: SpiderWorld, predators: Sequence[LizardState]) -> None:
    """Replace the world's predator roster and refresh controllers."""
    if not predators:
        raise ValueError("At least one predator is required.")
    world.predators = list(predators)
    world._sync_predator_controllers()


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
    Configure the world for the open-field foraging scenario.
    
    Places the spider in a deep shelter, advances the tick to the early-foraging phase, sets physiology (hunger, fatigue, sleep debt) to values that encourage foraging, places a single food spawn positioned to encourage exposed foraging near the deep shelter, spawns a distant patrolling predator, and refreshes the world's memory for the scenario start.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = 2
    _teleport_spider(world, deep)
    world.state.hunger = 0.88
    world.state.fatigue = 0.22
    world.state.sleep_debt = 0.20
    world.food_positions = [_open_field_foraging_food_cell(world, deep)]
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
    Configure the world for the exposed-day foraging scenario.
    
    Teleports the spider into a shelter, sets the tick to the start of day and scenario-specific hunger, fatigue, and sleep-debt values, places a single food at the food-spawn cell farthest from the spider, spawns a lizard in `PATROL` at the lizard-spawn cell nearest that food while preferring walkable cells separated from the spider's first food-visible frontier (with all lizard-walkable cells as a fallback), and refreshes the world's memory.
    """
    spider = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = 1
    _teleport_spider(world, spider)
    world.state.hunger = 0.94
    world.state.fatigue = 0.16
    world.state.sleep_debt = 0.18
    world.food_positions = [max(world.map_template.food_spawn_cells, key=lambda cell: world.manhattan(cell, spider))]
    food = world.food_positions[0]
    lizard_candidates = [
        cell
        for cell in world.map_template.lizard_spawn_cells
        if world.is_lizard_walkable(cell)
    ]
    if not lizard_candidates:
        lizard_candidates = [
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if world.is_lizard_walkable((x, y))
        ]
    if not lizard_candidates:
        raise ValueError("exposed_day_foraging requires a lizard-walkable cell")
    spider_frontier = _first_food_visible_frontier(world, spider, food)
    # Refine separated_candidates from lizard_candidates with world.manhattan
    # as a threat-radius heuristic around food and the first food-visible frontier.
    separated_candidates = [
        cell for cell in lizard_candidates if world.manhattan(cell, food) >= 3
    ]
    frontier_candidates = [
        cell
        for cell in separated_candidates
        if all(world.manhattan(cell, frontier_cell) >= 3 for frontier_cell in spider_frontier)
    ]
    lizard_spawn = min(
        frontier_candidates or separated_candidates or lizard_candidates,
        key=lambda cell: world.manhattan(cell, food),
    )
    world.lizard = LizardState(x=lizard_spawn[0], y=lizard_spawn[1], mode="PATROL")
    world.refresh_memory(initial=True)


def _food_deprivation(world: SpiderWorld) -> None:
    """
    Configure a SpiderWorld for the food-deprivation scenario.
    
    Places the spider in a shelter deep cell, sets the tick and physiological state to represent acute hunger, selects a single reachable-but-distant food spawn (preferentially choosing a food at Manhattan distance 4-6 from the spider, with fallbacks to distance >=4 or the farthest spawn), places a lizard in PATROL at a safe spawn, and refreshes world memory with initial=True.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = 4
    _teleport_spider(world, deep)
    world.state.hunger = FOOD_DEPRIVATION_INITIAL_HUNGER
    world.state.fatigue = 0.22
    world.state.sleep_debt = 0.18
    # Timing/geometry calibration: at hunger 0.96 the death timer is roughly
    # 12-15 ticks, so food must be reachable enough to test learned commitment.
    distance_ranked_food = sorted(
        world.map_template.food_spawn_cells,
        key=lambda cell: (world.manhattan(cell, deep), cell[0], cell[1]),
    )
    calibrated_food = [
        cell
        for cell in distance_ranked_food
        if 4 <= world.manhattan(cell, deep) <= 6
    ]
    if calibrated_food:
        food = min(calibrated_food, key=lambda cell: (-world.manhattan(cell, deep), cell[0], cell[1]))
    else:
        safe_default = distance_ranked_food[-1] if distance_ranked_food else None
        food = next(
            (cell for cell in distance_ranked_food if world.manhattan(cell, deep) >= 4),
            safe_default,
        )
    world.food_positions = [] if food is None else [food]
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
    Create a night-time scenario that biases behavior toward sleep rather than exploration.
    
    Teleports the spider to the shelter entrance, advances the world clock to night, sets low hunger, high fatigue, and an elevated sleep debt, places a distant food spawn to discourage foraging, spawns a non-threatening patrol predator at a safe patrol cell, and refreshes the world's memory.
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


def _visual_olfactory_pincer(world: SpiderWorld) -> None:
    """
    Place the spider between a visible visual hunter and a hidden olfactory hunter and configure the world for the visual/olfactory pincer scenario.
    
    Sets the spider position and heading, initializes hunger/fatigue/sleep levels, places a single food target biased by x-coordinate, spawns one visual-hunter and one olfactory-hunter LizardState at nearby candidate cells, and refreshes world memory.
    """
    open_cells = sorted(
        cell
        for cell in world.map_template.traversable_cells
        if cell not in world.shelter_cells
    )
    preferred_spider = (6, 6)
    spider = (
        preferred_spider
        if preferred_spider in open_cells
        else open_cells[len(open_cells) // 2]
    )
    world.tick = 2
    _teleport_spider(world, spider, reset_heading=False)
    world.state.heading_dx, world.state.heading_dy = 1, 0
    world.state.hunger = 0.52
    world.state.fatigue = 0.20
    world.state.sleep_debt = 0.18
    world.food_positions = [
        max(
            world.map_template.food_spawn_cells,
            key=lambda cell: (cell[0], -abs(cell[1] - spider[1])),
        )
    ]

    visual_candidates = [
        cell
        for cell in ((spider[0] + 2, spider[1]), (spider[0] + 3, spider[1]), (spider[0] + 2, spider[1] - 1))
        if 0 <= cell[0] < world.width
        and 0 <= cell[1] < world.height
        and world.is_lizard_walkable(cell)
    ]
    if not visual_candidates:
        visual_candidates = sorted(
            (
                cell
                for cell in open_cells
                if world.is_lizard_walkable(cell)
                and cell != spider
            ),
            key=lambda cell: (
                abs(world.manhattan(cell, spider) - 2),
                -cell[0],
                abs(cell[1] - spider[1]),
            ),
        )
    visual_pos = visual_candidates[0]

    olfactory_candidates = [
        cell
        for cell in (
            (spider[0] - 2, spider[1] + 1),
            (spider[0] - 2, spider[1]),
            (spider[0] - 3, spider[1] + 1),
            (spider[0] - 1, spider[1] + 2),
            (spider[0] - 1, spider[1] + 3),
        )
        if 0 <= cell[0] < world.width
        and 0 <= cell[1] < world.height
        and world.is_lizard_walkable(cell)
        and cell != visual_pos
    ]
    if not olfactory_candidates:
        olfactory_candidates = sorted(
            (
                cell
                for cell in open_cells
                if world.is_lizard_walkable(cell)
                and cell not in {spider, visual_pos}
            ),
            key=lambda cell: (
                abs(world.manhattan(cell, spider) - 2),
                cell[0],
                abs(cell[1] - spider[1]),
            ),
        )
    olfactory_pos = olfactory_candidates[0]

    _set_predators(
        world,
        [
            LizardState(
                x=visual_pos[0],
                y=visual_pos[1],
                mode="PATROL",
                profile=VISUAL_HUNTER_PROFILE,
            ),
            LizardState(
                x=olfactory_pos[0],
                y=olfactory_pos[1],
                mode="PATROL",
                profile=OLFACTORY_HUNTER_PROFILE,
            ),
        ],
    )
    world.refresh_memory(initial=True)


def _olfactory_ambush(world: SpiderWorld) -> None:
    """
    Position the spider facing inward in its shelter and place a single olfactory predator waiting outside the entrance.
    
    Sets the world tick and the spider's heading and physiological state (hunger, fatigue, sleep debt), teleports the spider to an interior shelter cell, sets a single food spawn at the map cell farthest from that interior cell, and spawns one olfactory-mode predator in WAIT mode outside the entrance (preferring the entrance-adjacent cell when walkable, otherwise using a fallback ambush cell). Refreshes the world's memory after initialization.
    """
    interior = _first_cell(world.shelter_interior_cells or world.shelter_deep_cells)
    entrance = _first_cell(world.shelter_entrance_cells)
    world.tick = 2
    _teleport_spider(world, interior, reset_heading=False)
    world.state.heading_dx, world.state.heading_dy = -1, 0
    world.state.hunger = 0.34
    world.state.fatigue = 0.28
    world.state.sleep_debt = 0.26
    world.food_positions = [
        max(world.map_template.food_spawn_cells, key=lambda cell: world.manhattan(cell, interior))
    ]
    preferred_predator = (entrance[0] + 1, entrance[1])
    predator_pos = (
        preferred_predator
        if world.is_lizard_walkable(preferred_predator)
        else _entrance_ambush_cell(world, entrance)
    )
    _set_predators(
        world,
        [
            LizardState(
                x=predator_pos[0],
                y=predator_pos[1],
                mode="WAIT",
                wait_target=entrance,
                failed_chases=1,
                profile=OLFACTORY_HUNTER_PROFILE,
            )
        ],
    )
    world.refresh_memory(initial=True)


def _visual_hunter_open_field(world: SpiderWorld) -> None:
    """
    Place the spider in open terrain with a fast visual hunter nearby for an open-field visual-hunter scenario.
    
    Mutates the world by teleporting the spider to an open cell, fixing its heading and physiological state (hunger, fatigue, sleep debt), placing a single biased food spawn, spawning a single fast visual predator in a nearby patrol position, and refreshing the world's memory (initial state).
    """
    open_cells = sorted(
        cell
        for cell in world.map_template.traversable_cells
        if cell not in world.shelter_cells
    )
    preferred_spider = (6, 6)
    spider = (
        preferred_spider
        if preferred_spider in open_cells
        else open_cells[len(open_cells) // 2]
    )
    world.tick = 2
    _teleport_spider(world, spider, reset_heading=False)
    world.state.heading_dx, world.state.heading_dy = 1, 0
    world.state.hunger = 0.62
    world.state.fatigue = 0.18
    world.state.sleep_debt = 0.16
    world.food_positions = [
        max(
            world.map_template.food_spawn_cells,
            key=lambda cell: (cell[0], -abs(cell[1] - spider[1])),
        )
    ]
    predator_candidates = [
        cell
        for cell in ((spider[0] + 2, spider[1]), (spider[0] + 3, spider[1]), (spider[0] + 2, spider[1] - 1))
        if 0 <= cell[0] < world.width
        and 0 <= cell[1] < world.height
        and world.is_lizard_walkable(cell)
    ]
    if not predator_candidates:
        predator_candidates = sorted(
            (
                cell
                for cell in open_cells
                if world.is_lizard_walkable(cell)
                and cell != spider
            ),
            key=lambda cell: (
                abs(world.manhattan(cell, spider) - 2),
                -cell[0],
                abs(cell[1] - spider[1]),
            ),
        )
    predator_pos = predator_candidates[0]
    _set_predators(
        world,
        [
            LizardState(
                x=predator_pos[0],
                y=predator_pos[1],
                mode="PATROL",
                profile=FAST_VISUAL_HUNTER_PROFILE,
            )
        ],
    )
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
    
    Scans each trace item for a "messages" list and returns the `payload` dict from messages
    where `sender == "action_center"` and `topic == "action.selection"`.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Sequence of trace item dictionaries produced by the simulation.
    
    Returns:
        list[Dict[str, object]]: List of payload dictionaries from matching action-selection messages.
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


def _trace_meta_mappings(trace: Sequence[Dict[str, object]]) -> list[Mapping[str, object]]:
    """
    Collect observation `meta` mappings found in trace items and environment observation messages.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Ordered sequence of trace records to scan for observation metadata.
    
    Returns:
        list[Mapping[str, object]]: List of `meta` mapping objects extracted from `observation` / `next_observation` entries and from environment messages with `topic == "observation"`, in the order they were encountered.
    """
    metas: list[Mapping[str, object]] = []
    for item in trace:
        for key in ("observation", "next_observation"):
            observation = item.get(key)
            if not isinstance(observation, Mapping):
                continue
            meta = observation.get("meta")
            if isinstance(meta, Mapping):
                metas.append(meta)
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("sender") != "environment" or message.get("topic") != "observation":
                continue
            payload = message.get("payload")
            if not isinstance(payload, Mapping):
                continue
            meta = payload.get("meta")
            if isinstance(meta, Mapping):
                metas.append(meta)
    return metas


def _trace_max_predator_threat(
    trace: Sequence[Dict[str, object]],
    predator_type: str,
) -> float:
    """
    Compute the maximum recorded predator threat score for a given predator type from trace metadata.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Trace items to search for observation/message meta mappings.
        predator_type (str): Predator label used as the key prefix (e.g., "visual" or "olfactory").
    
    Returns:
        float: The highest threat value found for "{predator_type}_predator_threat", or 0.0 if none present.
    """
    key = f"{predator_type}_predator_threat"
    values = [
        _float_or_none(meta.get(key)) or 0.0
        for meta in _trace_meta_mappings(trace)
    ]
    return float(max(values, default=0.0))


def _trace_dominant_predator_types(trace: Sequence[Dict[str, object]]) -> set[str]:
    """
    Determine which dominant predator type labels appear in the trace metadata.
    
    Searches observation/message meta mappings for a `dominant_predator_type_label`, normalizes it to lowercase,
    and returns any recognized labels.
    
    Returns:
        set[str]: A set containing zero or more of the strings "visual" and "olfactory".
    """
    labels: set[str] = set()
    for meta in _trace_meta_mappings(trace):
        label = str(meta.get("dominant_predator_type_label") or "").strip().lower()
        if label in {"visual", "olfactory"}:
            labels.add(label)
    return labels


def _float_or_none(value: object) -> float | None:
    """
    Attempt to convert the given value to a float; return None when conversion is not possible.
    
    Parameters:
        value (object): Value to convert to a float.
    
    Returns:
        The converted float if conversion succeeds, `None` otherwise.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    """
    Convert a value to an integer if possible.
    
    Attempts to coerce the given value to an `int`. If the value cannot be converted
    (e.g., is `None` or not a numeric/string representation), returns `None`.
    
    Parameters:
        value (object): The value to convert to an integer.
    
    Returns:
        int | None: The converted integer on success, `None` if conversion fails.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _state_position(state: Mapping[str, object]) -> tuple[int, int] | None:
    """
    Extract the spider's (x, y) tile coordinates from a state mapping.
    
    Parameters:
        state (Mapping[str, object]): A state dictionary that may contain numeric coordinates under keys `"x"` and `"y"`, or a sequence under `"spider_pos"`.
    
    Returns:
        A tuple `(x, y)` of integers when both coordinates are present and convertible to integers, `None` otherwise.
    """
    x = _int_or_none(state.get("x"))
    y = _int_or_none(state.get("y"))
    if x is not None and y is not None:
        return (x, y)
    pos = state.get("spider_pos")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        x = _int_or_none(pos[0])
        y = _int_or_none(pos[1])
        if x is not None and y is not None:
            return (x, y)
    return None


def _trace_tick(
    item: Mapping[str, object],
    state: Mapping[str, object],
    fallback: int,
) -> int:
    """
    Select the timestep tick for a trace item, falling back to state or a provided default.
    
    Parameters:
        item (Mapping[str, object]): A trace record; may include a numeric "tick" field.
        state (Mapping[str, object]): The extracted state dict; may include a numeric "tick" field.
        fallback (int): The value to return if neither `item` nor `state` provide a valid tick.
    
    Returns:
        int: The chosen tick: `item["tick"]` if present and integer-coercible, else `state["tick"]` if integer-coercible, else `fallback`.
    """
    tick = _int_or_none(item.get("tick"))
    if tick is not None:
        return tick
    state_tick = _int_or_none(state.get("tick"))
    if state_tick is not None:
        return state_tick
    return fallback


def _state_alive(state: Mapping[str, object]) -> bool | None:
    """
    Determine whether the entity described by a state mapping is alive.
    
    Checks the mapping for an explicit boolean "alive" field; if absent, looks for a numeric "health" field and treats values greater than 0 as alive.
    
    Parameters:
        state (Mapping[str, object]): A state dictionary or mapping that may contain "alive" or "health" keys.
    
    Returns:
        True if the state indicates the entity is alive, False if it indicates death, or None if the alive status cannot be determined.
    """
    alive = state.get("alive")
    if isinstance(alive, bool):
        return alive
    health = _float_or_none(state.get("health"))
    if health is None:
        return None
    return health > 0.0


def _state_food_distance(state: Mapping[str, object]) -> float | None:
    """
    Extract the spider's nearest-food distance from a state mapping.
    
    Checks the keys "food_dist", "food_distance", and "nearest_food_dist" in that order and returns the first value that can be coerced to a float.
    
    Parameters:
        state (Mapping[str, object]): A state dictionary potentially containing food-distance fields.
    
    Returns:
        float | None: The parsed distance if available, otherwise `None`.
    """
    for key in ("food_dist", "food_distance", "nearest_food_dist"):
        distance = _float_or_none(state.get(key))
        if distance is not None:
            return distance
    return None


def _observation_meta_food_distance(observation: object) -> float | None:
    """Return observation meta.food_dist as a float, or None when absent."""
    if not isinstance(observation, dict):
        return None
    meta = observation.get("meta")
    if not isinstance(meta, dict):
        return None
    return _float_or_none(meta.get("food_dist"))


def _environment_observation_food_distances(item: Mapping[str, object]) -> list[float]:
    """
    Return list[float] post-step distances from environment observation messages.

    Scans dict messages with sender "environment" and topic "observation".
    """
    distances: list[float] = []
    messages = item.get("messages", [])
    if not isinstance(messages, list):
        return distances
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("sender") != "environment":
            continue
        if message.get("topic") != "observation":
            continue
        payload = message.get("payload")
        if not isinstance(payload, dict):
            continue
        distance = _observation_meta_food_distance(payload)
        if distance is not None:
            distances.append(distance)
    return distances


def _trace_pre_observation_food_distances(item: Mapping[str, object]) -> list[float]:
    """Return list[float] pre-step food distances from item["observation"]."""
    distances: list[float] = []
    distance = _observation_meta_food_distance(item.get("observation"))
    if distance is not None:
        distances.append(distance)
    return distances


def _trace_post_observation_food_distances(item: Mapping[str, object]) -> list[float]:
    """Return list[float] post-step distances from next_observation and environment messages."""
    distances: list[float] = []
    distance = _observation_meta_food_distance(item.get("next_observation"))
    if distance is not None:
        distances.append(distance)
    distances.extend(_environment_observation_food_distances(item))
    return distances


def _observation_food_distances(item: Mapping[str, object]) -> list[float]:
    """
    Extracts food-distance samples from a single execution trace item.
    
    Scans pre-step observation metadata and post-step next/environment observation data, returning each numeric distance found in encounter order.
    
    Parameters:
        item (Mapping[str, object]): A trace record (usually a dict) representing a single step, which may contain "observation", "next_observation", and "messages".
    
    Returns:
        list[float]: Collected food-distance values found in the trace item, in the order they were discovered.
    """
    distances: list[float] = []
    distances.extend(_trace_pre_observation_food_distances(item))
    distances.extend(_trace_post_observation_food_distances(item))
    return distances


def _trace_distance_deltas(item: Mapping[str, object]) -> Mapping[str, object] | None:
    """
    Return Mapping[str, object] from item["distance_deltas"], or item["info"]["distance_deltas"].

    Example: {"distance_deltas": {"food": 2}} -> {"food": 2}; missing keys return None.
    """
    distance_deltas = item.get("distance_deltas")
    if isinstance(distance_deltas, Mapping):
        return distance_deltas
    info = item.get("info")
    if not isinstance(info, Mapping):
        return None
    distance_deltas = info.get("distance_deltas")
    if isinstance(distance_deltas, Mapping):
        return distance_deltas
    return None


def _trace_pre_tick_snapshot_payload(item: Mapping[str, object]) -> Mapping[str, object] | None:
    """
    Return Mapping[str, object] from the first pre_tick snapshot event, or None.

    Example: {"event_log": [{"stage": "pre_tick", "name": "snapshot", "payload": {...}}]} -> payload.
    """
    event_log = item.get("event_log")
    if not isinstance(event_log, list):
        return None
    for event in event_log:
        if not isinstance(event, Mapping):
            continue
        if event.get("stage") != "pre_tick" or event.get("name") != "snapshot":
            continue
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            return payload
    return None


def _trace_food_distances(trace: Sequence[Dict[str, object]]) -> list[float]:
    """
    Collects all reported spider-to-food distance samples from a trace.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Execution trace items containing observation-derived food distances, optional `state` fields with a food distance, and optional `distance_deltas` mapping which may include a `food` delta.
    
    Returns:
        list[float]: All extracted food-distance samples in the order encountered. Samples include distances from observation metadata, distances present in `state`, and at most one synthesized post-step distance from `distance_deltas["food"]` when no explicit post-step distance is present.
    """
    distances: list[float] = []
    for item in trace:
        observation_distances = _observation_food_distances(item)
        distances.extend(observation_distances)
        pre_distances = _trace_pre_observation_food_distances(item)
        post_distances = _trace_post_observation_food_distances(item)
        state_distance = None
        state = item.get("state")
        if isinstance(state, dict):
            state_distance = _state_food_distance(state)
            if state_distance is not None:
                distances.append(state_distance)
        distance_deltas = _trace_distance_deltas(item)
        food_delta = (
            _float_or_none(distance_deltas.get("food"))
            if isinstance(distance_deltas, Mapping)
            else None
        )
        has_explicit_post_distance = bool(post_distances) or state_distance is not None
        if food_delta is not None and pre_distances and not has_explicit_post_distance:
            distances.append(pre_distances[-1] - food_delta)
    return distances


def _trace_cell(value: object) -> tuple[int, int] | None:
    """
    Return a tuple[int, int] cell from a two-item sequence, or None.

    Example: [2, 3] -> (2, 3); malformed or non-numeric values return None.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = _int_or_none(value[0])
    y = _int_or_none(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def _trace_food_positions(*sources: Mapping[str, object]) -> list[tuple[int, int]]:
    """
    Return list[tuple[int, int]] food cells from the first source exposing food position keys.

    Example: {"food_positions": [[1, 2], [3, 4]]} -> [(1, 2), (3, 4)]; missing keys return [].
    """
    for source in sources:
        for key in ("food_positions", "food_position", "food_pos"):
            value = source.get(key)
            if key == "food_positions" and isinstance(value, (list, tuple)):
                cells = [_trace_cell(cell) for cell in value]
                return [cell for cell in cells if cell is not None]
            cell = _trace_cell(value)
            if cell is not None:
                return [cell]
    return []


def _trace_previous_position(item: Mapping[str, object]) -> tuple[int, int] | None:
    """
    Return tuple[int, int] pre-step spider position from prev_state, snapshot, or reversed movement.

    Example: {"state": {"spider_pos": [3, 2], "last_move_dx": 1, "last_move_dy": 0}} -> (2, 2).
    """
    prev_state = item.get("prev_state")
    if isinstance(prev_state, Mapping):
        pos = _state_position(prev_state)
        if pos is not None:
            return pos

    snapshot = _trace_pre_tick_snapshot_payload(item)
    if snapshot is not None:
        pos = _trace_cell(snapshot.get("spider_pos"))
        if pos is not None:
            return pos

    state = item.get("state")
    if not isinstance(state, Mapping):
        return None
    pos = _state_position(state)
    if pos is None:
        return None
    move_dx = _int_or_none(state.get("last_move_dx"))
    move_dy = _int_or_none(state.get("last_move_dy"))
    if move_dx is not None and move_dy is not None:
        return (pos[0] - move_dx, pos[1] - move_dy)
    action = item.get("executed_action") or item.get("action") or item.get("intended_action")
    if isinstance(action, str) and action in ACTION_DELTAS:
        dx, dy = ACTION_DELTAS[action]
        return (pos[0] - dx, pos[1] - dy)
    return None


def _trace_reconstructed_initial_food_distance(item: Mapping[str, object]) -> float | None:
    """
    Return a scalar pre-step food distance reconstructed from first-item fallbacks, or None.

    Example: post distance 6 with {"distance_deltas": {"food": 2}} -> 8.0; missing data returns None.
    """
    prev_state = item.get("prev_state")
    if isinstance(prev_state, Mapping):
        distance = _state_food_distance(prev_state)
        if distance is not None:
            return distance

    snapshot = _trace_pre_tick_snapshot_payload(item)
    if snapshot is not None:
        distance = _float_or_none(snapshot.get("prev_food_dist"))
        if distance is not None:
            return distance

    distance_deltas = _trace_distance_deltas(item)
    food_delta = (
        _float_or_none(distance_deltas.get("food"))
        if isinstance(distance_deltas, Mapping)
        else None
    )
    if food_delta is not None:
        post_distances = _trace_post_observation_food_distances(item)
        if post_distances:
            return post_distances[0] + food_delta
        state = item.get("state")
        if isinstance(state, Mapping):
            state_distance = _state_food_distance(state)
            if state_distance is not None:
                return state_distance + food_delta

    state = item.get("state")
    sources: list[Mapping[str, object]] = [item]
    if isinstance(state, Mapping):
        sources.append(state)
    if isinstance(prev_state, Mapping):
        sources.append(prev_state)
    previous_pos = _trace_previous_position(item)
    food_positions = _trace_food_positions(*sources)
    if previous_pos is None or not food_positions:
        return None
    return float(
        min(abs(previous_pos[0] - x) + abs(previous_pos[1] - y) for x, y in food_positions)
    )


def _trace_initial_food_distance(trace: Sequence[Dict[str, object]]) -> float | None:
    """
    Return scalar initial pre-step food distance from explicit observations, or None.

    Example: {"observation": {"meta": {"food_dist": 8}}} -> 8.0; post-only traces return None.
    """
    for item in trace:
        pre_distances = _trace_pre_observation_food_distances(item)
        if pre_distances:
            return pre_distances[0]
    return None


def _trace_shelter_cells(state: Mapping[str, object]) -> set[tuple[int, int]]:
    """
    Compute the set of shelter cell coordinates from a serialized world state.
    
    Parameters:
        state (Mapping[str, object]): Trace/state dictionary that may contain `map_template` (str),
            `width` (int), and `height` (int). Width/height fall back to trace defaults when absent
            or non-numeric.
    
    Returns:
        set[tuple[int, int]]: Set of (x, y) coordinates belonging to the template shelter cells.
        Returns an empty set when `map_template` is missing or not a string, or when the named map
        template cannot be built.
    """
    template_name = state.get("map_template")
    if not isinstance(template_name, str):
        return set()
    width = _int_or_none(state.get("width")) or TRACE_MAP_WIDTH_FALLBACK
    height = _int_or_none(state.get("height")) or TRACE_MAP_HEIGHT_FALLBACK
    try:
        template = build_map_template(template_name, width=width, height=height)
    except ValueError:
        return set()
    return set(template.shelter_cells)


def _trace_shelter_template_key(state: Mapping[str, object]) -> tuple[str, int, int] | None:
    """
    Return tuple[str, int, int] from state map_template, width, and height, or None.

    Example: {"map_template": "central_burrow"} -> ("central_burrow", 12, 12) using fallbacks.
    """
    template_name = state.get("map_template")
    if not isinstance(template_name, str):
        return None
    width = _int_or_none(state.get("width")) or TRACE_MAP_WIDTH_FALLBACK
    height = _int_or_none(state.get("height")) or TRACE_MAP_HEIGHT_FALLBACK
    return (template_name, width, height)


def _trace_shelter_exit(
    trace: Sequence[Dict[str, object]],
) -> tuple[bool, int | None]:
    """
    Determine whether the spider left any recognized shelter during the trace and, if so, when.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Ordered trace items containing optional `state` dictionaries and tick information.
    
    Returns:
        tuple[bool, int | None]: `(escaped, tick)` where `escaped` is `True` if the spider is observed outside the computed shelter cells or has `shelter_role == "outside"` at any trace item; `tick` is the trace tick of the first such observation (or `None` when no exit is detected).
    """
    shelter_cells_cache: set[tuple[int, int]] = set()
    shelter_template_key: tuple[str, int, int] | None = None
    for index, item in enumerate(trace):
        state = item.get("state")
        if not isinstance(state, dict):
            continue
        tick = _trace_tick(item, state, index)
        pos = _state_position(state)
        current_template_key = _trace_shelter_template_key(state)
        if current_template_key is None:
            shelter_cells = set()
        elif current_template_key != shelter_template_key:
            shelter_template_key = current_template_key
            shelter_cells_cache = _trace_shelter_cells(state)
            shelter_cells = shelter_cells_cache
        else:
            shelter_cells = shelter_cells_cache
        if pos is not None and shelter_cells:
            if pos not in shelter_cells:
                return True, tick
            continue
        if state.get("shelter_role") == "outside":
            return True, tick
    return False, None


def _trace_death_tick(trace: Sequence[Dict[str, object]]) -> int | None:
    """
    Finds the trace tick corresponding to the first transition from alive to not alive.
    
    Scans trace records for the earliest item whose `state` indicates the agent is not alive while the previous known alive value was True (or unknown). Returns the tick associated with that record when available.
    
    Returns:
        int | None: The tick of the first death transition, or `None` if no death is observed.
    """
    previous_alive: bool | None = None
    for index, item in enumerate(trace):
        state = item.get("state")
        if not isinstance(state, dict):
            continue
        alive = _state_alive(state)
        if alive is None:
            continue
        if not alive and previous_alive is not False:
            return _trace_tick(item, state, index)
        previous_alive = alive
    return None


def _hunger_valence_rate(trace: Sequence[Dict[str, object]]) -> float:
    """
    Compute the proportion of action-selection payloads whose `winning_valence` equals "hunger".
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Execution trace records from which action-selection payloads are extracted.
    
    Returns:
        float: Proportion in [0.0, 1.0] of action-selection payloads with `winning_valence == "hunger"`. Returns 0.0 when no action-selection payloads are present.
    """
    payloads = _trace_action_selection_payloads(trace)
    if not payloads:
        return 0.0
    hunger_ticks = sum(
        _payload_text(payload, "winning_valence") == "hunger"
        for payload in payloads
    )
    return float(hunger_ticks / len(payloads))


def _predator_visible_flag(value: object) -> bool | None:
    """Coerce bool, numeric, and string predator-visibility values to bool.

    Numeric values are converted with _float_or_none and are True when >= 0.5.
    True strings are "true", "yes", and "visible"; false strings are
    "false", "no", "hidden", and "none". Returns None when conversion fails.
    """
    if isinstance(value, bool):
        return value
    numeric = _float_or_none(value)
    if numeric is not None:
        return numeric >= 0.5
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "visible"}:
            return True
        if lowered in {"false", "no", "hidden", "none"}:
            return False
    return None


def _mapping_predator_visible(source: Mapping[str, object]) -> bool:
    """Return True when a nested mapping contains a visible predator flag.

    ``source`` is expected to be a Mapping[str, object]. The lookup checks
    "predator_visible", "prev_predator_visible", nested "meta",
    "vision.predator.{visible,certainty}", and "evidence.threat". Flag values
    are interpreted by _predator_visible_flag, including the >= 0.5 numeric
    threshold used for certainty values, and Mapping values are searched
    recursively. Returns False when no checked flag indicates visibility.
    """
    for key in ("predator_visible", "prev_predator_visible"):
        visible = _predator_visible_flag(source.get(key))
        if visible:
            return True

    meta = source.get("meta")
    if isinstance(meta, Mapping) and _mapping_predator_visible(meta):
        return True

    vision = source.get("vision")
    if isinstance(vision, Mapping):
        predator = vision.get("predator")
        if isinstance(predator, Mapping):
            for key in ("visible", "certainty"):
                visible = _predator_visible_flag(predator.get(key))
                if visible:
                    return True

    evidence = source.get("evidence")
    if isinstance(evidence, Mapping):
        threat = evidence.get("threat")
        if isinstance(threat, Mapping) and _mapping_predator_visible(threat):
            return True

    return False


def _trace_predator_visible(item: Mapping[str, object]) -> bool:
    """
    Determine whether a trace item reports predator visibility via any supported channel.
    
    Parameters:
        item (Mapping[str, object]): A trace item or composite event dictionary to inspect.
    
    Returns:
        `true` if any inspected field or embedded payload indicates predator visibility, `false` otherwise.
    """
    if _mapping_predator_visible(item):
        return True

    for key in ("state", "observation", "next_observation", "debug"):
        value = item.get(key)
        if isinstance(value, Mapping) and _mapping_predator_visible(value):
            return True

    snapshot = _trace_pre_tick_snapshot_payload(item)
    if snapshot is not None and _mapping_predator_visible(snapshot):
        return True

    messages = item.get("messages", [])
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            payload = message.get("payload")
            if isinstance(payload, Mapping) and _mapping_predator_visible(payload):
                return True

    return False


def _trace_peak_food_progress(trace: Sequence[Dict[str, object]]) -> float:
    """
    Return the maximum reduction in food distance observed across a trace (>= 0.0).
    """
    initial_food_distance = _trace_initial_food_distance(trace)
    if initial_food_distance is None and trace:
        initial_food_distance = _trace_reconstructed_initial_food_distance(trace[0])
    food_distances = _trace_food_distances(trace)
    if initial_food_distance is None and food_distances:
        initial_food_distance = food_distances[0]
    if initial_food_distance is None or not food_distances:
        return 0.0
    return float(
        max(0.0, *(initial_food_distance - distance for distance in food_distances))
    )


def _trace_corridor_metrics(
    trace: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """
    Compute trace-derived diagnostics for corridor-gauntlet failures.
    """
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    return {
        "left_shelter": bool(left_shelter),
        "shelter_exit_tick": shelter_exit_tick,
        "predator_visible_ticks": sum(
            1 for item in trace if _trace_predator_visible(item)
        ),
        "peak_food_progress": _trace_peak_food_progress(trace),
        "death_tick": _trace_death_tick(trace),
    }


def _extract_exposed_day_trace_metrics(
    trace: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """
    Compute trace-derived diagnostics useful for exposed-day foraging classification.

    Parameters:
        trace (Sequence[Dict[str, object]]): Ordered trace items (event/state dictionaries) for an episode.

    Returns:
        Dict[str, object]: Diagnostics with the following keys:
            - "shelter_exit_tick" (Optional[int]): Tick when the spider first left a recognized shelter, or None.
            - "left_shelter" (bool): True if the spider was observed outside shelter at any point.
            - "peak_food_progress" (float): Maximum reduction in food distance relative to the initial distance (>= 0.0).
            - "predator_visible_ticks" (int): Count of trace items where the predator is considered visible.
            - "final_distance_to_food" (Optional[float]): Last observed food distance, or None if unavailable.
    """
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    food_distances = _trace_food_distances(trace)
    return {
        "shelter_exit_tick": shelter_exit_tick,
        "left_shelter": bool(left_shelter),
        "peak_food_progress": _trace_peak_food_progress(trace),
        "predator_visible_ticks": sum(
            1 for item in trace if _trace_predator_visible(item)
        ),
        "final_distance_to_food": food_distances[-1] if food_distances else None,
    }


def _food_approached(metrics: Mapping[str, object]) -> bool:
    """
    Determine whether the agent made an approach toward the target food during the episode.
    
    Considers `min_food_distance_reached` relative to `initial_food_distance` when both are present; if either is missing, falls back to the sign of `food_distance_delta` to infer approach.
    
    Parameters:
        metrics (Mapping[str, object]): Episode-derived metrics. Expected keys (optional): 
            `min_food_distance_reached`, `initial_food_distance`, `food_distance_delta`.
    
    Returns:
        bool: `True` if the agent made a measurable approach toward the food, `False` otherwise.
    """
    min_food_distance = _float_or_none(metrics.get("min_food_distance_reached"))
    initial_food_distance = _float_or_none(metrics.get("initial_food_distance"))
    if min_food_distance is not None and initial_food_distance is not None:
        return min_food_distance < initial_food_distance
    food_distance_delta = _float_or_none(metrics.get("food_distance_delta"))
    return food_distance_delta is not None and food_distance_delta > 0.0


def _classify_food_deprivation_failure(metrics: Mapping[str, object]) -> str:
    """
    Assigns a failure or outcome label for a food-deprivation episode from derived diagnostic metrics.

    Parameters:
        metrics (Mapping[str, object]): Derived episode metrics used to classify outcome. Recognized keys:
            - checks_passed: truthy indicates the scenario checks were satisfied.
            - left_shelter: whether the agent left its shelter.
            - hunger_valence_rate: fraction of action-selection ticks favoring hunger (numeric).
            - min_food_distance_reached: minimum observed distance to the target food (numeric).
            - initial_food_distance: initial distance to the target food (numeric).
            - food_distance_delta: change in food distance; positive indicates progress toward food (numeric).
            - alive: whether the agent survived the episode.
            - food_eaten: number of food items consumed (int).

    Returns:
        str: One of the outcome labels:
            - "success": scenario checks passed.
            - "no_commitment": agent did not leave shelter or showed insufficient hunger-driven commitment.
            - "orientation_failure": agent left shelter but did not approach the target food.
            - "timing_failure": agent died before consuming food or died despite eating without meeting checks.
            - "scoring_mismatch": agent survived and approached/acted but failed the scenario checks.
    """
    if metrics.get("checks_passed", False):
        return "success"

    left_shelter = bool(metrics.get("left_shelter", False))
    hunger_valence_rate = _float_or_none(metrics.get("hunger_valence_rate")) or 0.0
    approached_food = _food_approached(metrics)

    if not left_shelter or hunger_valence_rate < 0.5:
        return "no_commitment"
    if not approached_food:
        return "orientation_failure"

    alive = bool(metrics.get("alive", False))
    food_eaten = _int_or_none(metrics.get("food_eaten")) or 0
    if not alive and food_eaten <= 0:
        return "timing_failure"
    if alive:
        return "scoring_mismatch"
    return "timing_failure"


def _clamp01(value: float) -> float:
    """
    Clamp a numeric value to the inclusive range 0.0-1.0.
    
    Parameters:
        value (float): Input value to be clamped.
    
    Returns:
        float: The input coerced to a float and constrained to be between 0.0 and 1.0 inclusive.
    """
    return float(max(0.0, min(1.0, value)))


def _memory_vector_freshness(value: object) -> float:
    """Return [0,1] freshness for a non-zero dx/dy Mapping, using normalized age or age/ttl via _float_or_none and _clamp01."""
    if not isinstance(value, Mapping):
        return 0.0
    dx = _float_or_none(value.get("dx")) or 0.0
    dy = _float_or_none(value.get("dy")) or 0.0
    if abs(dx) + abs(dy) <= 0.0:
        return 0.0
    age = _float_or_none(value.get("age"))
    if age is None:
        return 0.0
    if 0.0 <= age <= 1.0:
        return _clamp01(1.0 - age)
    ttl = _float_or_none(value.get("ttl"))
    if ttl is None or ttl <= 0.0:
        return 0.0
    return _clamp01(1.0 - (age / ttl))


def _food_signal_strength(source: Mapping[str, object]) -> float:
    """
    Get the strongest food-direction cue strength available in a nested payload or metadata mapping.
    
    Scans the provided mapping for many possible food-cue fields (e.g. visibility, certainty, smell/trace strengths, memory freshness, nested `evidence` sections, `meta`, `vision.food`, `percept_traces.food`, `memory_vectors.food`, `food_trace`, and `food_memory`) and returns the maximum normalized strength found. Handles nested mappings recursively and clamps individual values into the [0.0, 1.0] range before taking the maximum.
    
    Parameters:
        source (Mapping[str, object]): A mapping (payload/state/meta) that may contain one or more food-cue fields or nested mappings.
    
    Returns:
        float: The strongest detected food-direction cue strength in the range 0.0-1.0; 0.0 when no cue is present.
    """
    strengths = [
        _clamp01(v)
        for key in (
            "food_visible",
            "food_certainty",
            "food_smell_strength",
            "food_trace_strength",
            "food_memory_freshness",
        )
        if (v := _float_or_none(source.get(key))) is not None
    ]

    evidence = source.get("evidence")
    if isinstance(evidence, Mapping):
        for key in ("hunger", "visual", "sensory"):
            values = evidence.get(key)
            if isinstance(values, Mapping):
                strengths.append(_food_signal_strength(values))

    meta = source.get("meta")
    if isinstance(meta, Mapping):
        strengths.append(_food_signal_strength(meta))

    vision = source.get("vision")
    if isinstance(vision, Mapping):
        food_view = vision.get("food")
        if isinstance(food_view, Mapping):
            for key in ("visible", "certainty"):
                value = _float_or_none(food_view.get(key))
                if value is not None:
                    strengths.append(_clamp01(value))

    hunger = source.get("hunger")
    if isinstance(hunger, Mapping):
        strengths.append(_food_signal_strength(hunger))

    percept_traces = source.get("percept_traces")
    if isinstance(percept_traces, Mapping):
        food_trace = percept_traces.get("food")
        if isinstance(food_trace, Mapping):
            value = _float_or_none(food_trace.get("strength"))
            if value is not None:
                strengths.append(_clamp01(value))

    memory_vectors = source.get("memory_vectors")
    if isinstance(memory_vectors, Mapping):
        strengths.append(_memory_vector_freshness(memory_vectors.get("food")))

    if "food_memory_dx" in source or "food_memory_dy" in source:
        flat_memory = {
            "dx": source.get("food_memory_dx"),
            "dy": source.get("food_memory_dy"),
            "age": source.get("food_memory_age"),
            "ttl": source.get("food_memory_ttl"),
        }
        flat_memory_strength = _memory_vector_freshness(flat_memory)
        if flat_memory_strength > 0.0:
            strengths.append(flat_memory_strength)
        else:
            dx = _float_or_none(source.get("food_memory_dx")) or 0.0
            dy = _float_or_none(source.get("food_memory_dy")) or 0.0
            if abs(dx) + abs(dy) > 0.0:
                strengths.append(1.0)

    state_food_trace = source.get("food_trace")
    if isinstance(state_food_trace, Mapping):
        value = _float_or_none(state_food_trace.get("strength"))
        if value is not None:
            strengths.append(_clamp01(value))

    state_food_memory = source.get("food_memory")
    if isinstance(state_food_memory, Mapping):
        memory_strength = _memory_vector_freshness(state_food_memory)
        if memory_strength > 0.0:
            strengths.append(memory_strength)
        elif state_food_memory.get("target") is not None:
            age = _float_or_none(state_food_memory.get("age"))
            ttl = _float_or_none(state_food_memory.get("ttl"))
            if age is not None and ttl is not None and ttl > 0.0:
                strengths.append(_clamp01(1.0 - (age / ttl)))

    return max(strengths, default=0.0)


def _trace_food_signal_strengths(trace: Sequence[Dict[str, object]]) -> list[float]:
    """
    Compute a per-tick food-signal strength series from trace items.
    
    For each trace item, inspects `state`, `observation`, `next_observation`, and any message `payload` objects and computes the strongest food-direction cue present by calling `_food_signal_strength` on each available mapping; the per-item value is the maximum strength found (or 0.0 if none).
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Ordered trace items produced by the environment/agent.
    
    Returns:
        list[float]: Per-tick maximum food-signal strengths aligned to `trace` (0.0 if no signal).
    """
    strengths: list[float] = []
    for item in trace:
        item_strengths: list[float] = []
        state = item.get("state")
        if isinstance(state, Mapping):
            item_strengths.append(_food_signal_strength(state))
        for key in ("observation", "next_observation"):
            observation = item.get(key)
            if isinstance(observation, Mapping):
                item_strengths.append(_food_signal_strength(observation))
        messages = item.get("messages", [])
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                payload = message.get("payload")
                if isinstance(payload, Mapping):
                    item_strengths.append(_food_signal_strength(payload))
        strengths.append(max(item_strengths, default=0.0))
    return strengths


def _classify_open_field_foraging_failure(metrics: Mapping[str, object]) -> str:
    """
    Classify an open-field foraging episode into a single outcome label based on diagnostic metrics.
    
    Parameters:
        metrics (Mapping[str, object]): Diagnostic values computed for an episode. Recognized keys include:
            - checks_passed: truthy when scenario checks indicate full success.
            - left_shelter: whether the agent left shelter.
            - hunger_valence_rate: numeric measure of hunger-driven action preference.
            - initial_food_signal_strength, max_food_signal_strength: numeric food-cue strength estimates.
            - initial_food_distance, min_food_distance_reached, food_distance_delta: numeric distance/progress metrics.
            - food_eaten: count of food items consumed.
            - alive: final alive status.
        Missing or unknown keys are treated as absent/zero where appropriate.
    
    Returns:
        str: One of:
            - "success": checks indicate full success.
            - "never_left_shelter": agent did not leave shelter.
            - "no_hunger_commitment": left shelter but hunger-driven commitment was insufficient.
            - "left_without_food_signal": left shelter with no detectable food cue.
            - "orientation_failure": food cues were present but the agent did not approach food.
            - "progressed_then_died": the agent approached food but died before completion.
            - "stall": the agent left shelter, trace-confirmed no predator visibility, made no food progress, and survived.
            - "scoring_mismatch": outcome did not match any above category (ambiguous/unexpected).
    """
    if metrics.get("checks_passed", False):
        return "success"

    if not bool(metrics.get("left_shelter", False)):
        return "never_left_shelter"

    hunger_valence_rate = _float_or_none(metrics.get("hunger_valence_rate")) or 0.0
    if hunger_valence_rate < 0.5:
        return "no_hunger_commitment"

    initial_food_signal_strength = _float_or_none(
        metrics.get("initial_food_signal_strength")
    ) or 0.0
    if initial_food_signal_strength <= 0.0:
        return "left_without_food_signal"

    food_eaten = _int_or_none(metrics.get("food_eaten")) or 0
    approached_food = food_eaten > 0 or _food_approached(metrics)
    predator_visible_ticks = _int_or_none(metrics.get("predator_visible_ticks"))
    alive = bool(metrics.get("alive", False))

    if (
        not approached_food
        and "predator_visible_ticks" in metrics
        and predator_visible_ticks == 0
        and alive
    ):
        return "stall"
    if not approached_food:
        return "orientation_failure"
    if not alive:
        return "progressed_then_died"
    return "scoring_mismatch"


def _classify_corridor_gauntlet_failure(
    stats: EpisodeStats,
    trace_metrics: Mapping[str, object],
    full_success: bool,
) -> str:
    """
    Assign a corridor-gauntlet outcome label from stats and trace diagnostics.
    """
    if full_success:
        return "success"

    if not bool(trace_metrics.get("left_shelter", False)):
        return "frozen_in_shelter"

    food_distance_delta = float(stats.food_distance_delta)
    predator_contacts = int(stats.predator_contacts)
    alive = bool(stats.alive)

    if predator_contacts > 0 and not alive:
        return "contact_failure_died"
    if predator_contacts > 0 and alive:
        return "contact_failure_survived"
    if alive and predator_contacts == 0 and food_distance_delta <= 0.0:
        return "survived_no_progress"
    if food_distance_delta > 0.0 and not alive and predator_contacts == 0:
        return "progress_then_died"
    return "scoring_mismatch"


def _classify_exposed_day_foraging_failure(metrics: Mapping[str, object]) -> str:
    """
    Assign an exposed-day foraging outcome label from trace-backed diagnostics.
    """
    if metrics.get("checks_passed", False):
        return "success"

    if not bool(metrics.get("left_shelter", False)):
        return "cautious_inert"

    food_eaten = _int_or_none(metrics.get("food_eaten")) or 0
    food_distance_delta = _float_or_none(metrics.get("food_distance_delta")) or 0.0
    peak_food_progress = _float_or_none(metrics.get("peak_food_progress")) or 0.0
    made_food_progress = bool(
        food_eaten > 0
        or food_distance_delta > 0.0
        or peak_food_progress > 0.0
    )
    predator_visible_ticks = _int_or_none(metrics.get("predator_visible_ticks")) or 0
    alive = bool(metrics.get("alive", False))

    if predator_visible_ticks > 0 and not made_food_progress:
        return "threatened_retreat"
    if made_food_progress and not alive:
        return "foraging_and_died"
    if made_food_progress:
        return "partial_progress"
    if predator_visible_ticks == 0 and alive:
        return "stall"
    return "scoring_mismatch"


def _payload_float(payload: Dict[str, Any], *path: str) -> float | None:
    """
    Extract a nested value from a payload by key path and coerce it to float.
    
    Parameters:
        payload (Dict[str, Any]): Nested mapping to traverse.
        *path (str): Sequence of keys describing the lookup path inside `payload`.
    
    Returns:
        float | None: The numeric value at the end of the path converted to `float`, or `None` if any key is missing, an intermediate value is not a mapping, or conversion to float fails.
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
    """
    Builds a compact set of diagnostic metrics summarizing food-progress and survival outcome for a scenario.
    
    Parameters:
        food_distance_delta (float): Net change in distance to target food (positive means closer).
        food_eaten (int): Number of food items consumed during the episode.
        hunger_reduction (float): Reduction in hunger (positive means hunger decreased).
        alive (bool): Whether the agent was alive at the end of the episode.
        predator_contacts (int): Number of predator contact events recorded.
        full_success (bool): Whether the scenario was fully completed (highest success level).
    
    Returns:
        dict[str, object]: A mapping containing:
            - "progress_band" (str): coarse category of progress as returned by _progress_band.
            - "outcome_band" (str): overall outcome category (e.g., "full_success", "survived_but_unfinished", "regressed_and_died", "partial_progress_died", "stalled_and_died").
            - "partial_progress" (bool): true if any progress was made (food eaten, positive distance delta, or hunger reduced).
            - "died_after_progress" (bool): true if the agent died after making some progress.
            - "died_without_contact" (bool): true if the agent died without predator contacts.
            - "survived_without_progress" (bool): true if the agent survived but made no progress.
    """
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


def _module_response_for_type(
    stats: EpisodeStats,
    predator_type: str,
) -> Mapping[str, float]:
    """
    Retrieve module response shares for a predator type from episode statistics.
    
    Returns a mapping of module names to response-share floats for the specified predator_type; returns an empty mapping if the stats field is missing, malformed, or does not contain the requested predator_type.
    
    Parameters:
        stats (EpisodeStats): Episode-level aggregated statistics possibly containing `module_response_by_predator_type`.
        predator_type (str): Predator type key to look up (e.g., "visual", "olfactory").
    
    Returns:
        Mapping[str, float]: Mapping from module name to its response share value.
    """
    responses = getattr(stats, "module_response_by_predator_type", {})
    if not isinstance(responses, Mapping):
        return {}
    response = responses.get(predator_type, {})
    return response if isinstance(response, Mapping) else {}


def _module_share_for_type(
    stats: EpisodeStats,
    predator_type: str,
    module_name: str,
) -> float:
    """
    Fetches the module's response share for a given predator type and module name.
    
    Parameters:
        stats (EpisodeStats): Episode statistics holding module responses keyed by predator type.
        predator_type (str): Predator type label (e.g., "visual" or "olfactory").
        module_name (str): Module identifier whose share to retrieve (e.g., "visual_cortex").
    
    Returns:
        float: The module's share value (coerced to a float, typically in [0,1]); returns 0.0 if the value is missing or cannot be parsed as a number.
    """
    response = _module_response_for_type(stats, predator_type)
    return float(_float_or_none(response.get(module_name)) or 0.0)


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
    BehaviorCheckSpec(
        "commits_to_foraging",
        "Spider leaves shelter and prioritizes hunger.",
        "left_shelter and hunger_valence_rate >= 0.5",
    ),
    BehaviorCheckSpec("survives_deprivation", "Survives the deprivation episode.", "true"),
)
VISUAL_OLFACTORY_PINCER_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("dual_threat_detected", "Both visual and olfactory predator pressures are represented.", "visual and olfactory threats > 0"),
    BehaviorCheckSpec("type_specific_response", "Visual threat favors visual_cortex while olfactory threat favors sensory_cortex.", "module preference splits by predator type"),
    BehaviorCheckSpec("survives_pincer", "Survives the dual-threat pincer without direct contact.", "alive and predator_contacts=0"),
)
OLFACTORY_AMBUSH_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("olfactory_threat_detected", "Registers the olfactory predator despite no visual contact.", "olfactory threat > 0 and predator_visible=false"),
    BehaviorCheckSpec("sensory_cortex_engaged", "Sensory cortex dominates the response to the olfactory ambush.", "sensory_cortex >= visual_cortex"),
    BehaviorCheckSpec("survives_olfactory_ambush", "Survives the ambush without direct contact.", "alive and predator_contacts=0"),
)
OLFACTORY_AMBUSH_WINDOW_TICKS = 4
VISUAL_HUNTER_OPEN_FIELD_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("visual_threat_detected", "Registers the fast visual hunter in open terrain.", "visual threat > 0"),
    BehaviorCheckSpec("visual_cortex_engaged", "Visual cortex dominates the response to the open-field hunter.", "visual_cortex >= sensory_cortex"),
    BehaviorCheckSpec("survives_visual_hunter", "Survives the open-field visual-hunter encounter.", "alive and predator_contacts=0"),
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
    Score open-field foraging performance by producing progress, viability, and survival checks plus trace-derived diagnostics and a failure-mode classification.
    
    Computes three behavior checks (food distance progress, foraging viability, and alive), extracts trace-backed diagnostics (initial and min food distances, shelter exit tick, death tick, hunger valence rate, per-tick food-signal strengths), derives weak-scenario diagnostic bands, and classifies a `failure_mode` from the aggregated metrics.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute food progress, food eaten, survival, and predator contacts.
        trace (Sequence[Dict[str, object]]): Execution trace used to derive commitment/orientation diagnostics and food-cue signals.
    
    Returns:
        BehavioralEpisodeScore: A score whose checks cover food progress, foraging viability, and survival. The returned `behavior_metrics` includes at least:
            - `food_distance_delta`, `food_eaten`, `alive`, `predator_contacts`
            - `initial_food_distance`, `min_food_distance_reached`
            - `left_shelter`, `shelter_exit_tick`, `death_tick`
            - `hunger_valence_rate`
            - `initial_food_signal_strength`, `max_food_signal_strength`, `food_signal_tick_rate`
            - `predator_visible_ticks`
          plus weak-scenario diagnostic fields and a computed `failure_mode`.
    """
    food_progress = float(stats.food_distance_delta)
    foraging_viable = bool(stats.food_eaten > 0 or food_progress >= 2.0)
    checks = (
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[0], passed=food_progress > 0.0, value=food_progress),
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[1], passed=foraging_viable, value=bool(foraging_viable)),
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    initial_food_distance = _trace_initial_food_distance(trace)
    if initial_food_distance is None and trace:
        initial_food_distance = _trace_reconstructed_initial_food_distance(trace[0])
    food_distances = _trace_food_distances(trace)
    if initial_food_distance is None and food_distances:
        initial_food_distance = food_distances[0]
    min_food_distance_reached = min(food_distances) if food_distances else None
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    death_tick = _trace_death_tick(trace)
    hunger_valence_rate = _hunger_valence_rate(trace)
    predator_visible_ticks = (
        sum(1 for item in trace if _trace_predator_visible(item))
        if trace
        else None
    )
    food_signal_strengths = _trace_food_signal_strengths(trace)
    initial_food_signal_strength = (
        food_signal_strengths[0] if food_signal_strengths else 0.0
    )
    max_food_signal_strength = max(food_signal_strengths, default=0.0)
    food_signal_tick_rate = (
        sum(1 for value in food_signal_strengths if value > 0.0)
        / len(food_signal_strengths)
        if food_signal_strengths
        else 0.0
    )
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=food_progress,
        food_eaten=int(stats.food_eaten),
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    behavior_metrics = {
        "food_distance_delta": food_progress,
        "food_eaten": int(stats.food_eaten),
        "alive": bool(stats.alive),
        "predator_contacts": int(stats.predator_contacts),
        "initial_food_distance": initial_food_distance,
        "min_food_distance_reached": min_food_distance_reached,
        "left_shelter": bool(left_shelter),
        "shelter_exit_tick": shelter_exit_tick,
        "death_tick": death_tick,
        "hunger_valence_rate": hunger_valence_rate,
        "predator_visible_ticks": predator_visible_ticks,
        "initial_food_signal_strength": initial_food_signal_strength,
        "max_food_signal_strength": max_food_signal_strength,
        "food_signal_tick_rate": food_signal_tick_rate,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_open_field_foraging_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate reproducible open-field foraging with an explicit progress metric.",
        checks=checks,
        behavior_metrics=behavior_metrics,
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
    Evaluate corridor gauntlet episode performance with trace-backed diagnostics.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace used to derive shelter exit, predator visibility, progress, and death diagnostics.
    
    Returns:
        BehavioralEpisodeScore: Score object containing three behavior checks (food progress, zero predator contacts, alive)
        and metrics including `failure_mode`, `left_shelter`, `shelter_exit_tick`, `predator_visible_ticks`, `peak_food_progress`, and `death_tick`.
    """
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
    trace_metrics = _trace_corridor_metrics(trace)
    behavior_metrics = {
        "food_distance_delta": food_progress,
        "predator_contacts": int(stats.predator_contacts),
        "alive": bool(stats.alive),
        "predator_mode_transitions": int(stats.predator_mode_transitions),
        **trace_metrics,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_corridor_gauntlet_failure(
        stats,
        trace_metrics,
        full_success,
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate narrow-corridor progress without relying only on aggregated reward.",
        checks=checks,
        behavior_metrics=behavior_metrics,
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
    Score an exposed-day foraging episode and attach trace-derived diagnostics and a failure-mode label.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to evaluate objective checks and numeric metrics.
        trace (Sequence[Dict[str, object]]): Ordered trace items used to extract shelter/exit, food-distance, and predator-visibility diagnostics.
    
    Returns:
        BehavioralEpisodeScore: The scenario score including three behavioral checks (food-distance progress, zero predator contacts, survival) and a `behavior_metrics` mapping. `behavior_metrics` contains numeric and boolean summaries from `stats` (e.g., `food_distance_delta`, `food_eaten`, `predator_contacts`, `alive`, `final_health`), trace-derived diagnostics (e.g., `left_shelter`, `shelter_exit_tick`, `peak_food_progress`, `predator_visible_ticks`), weaker diagnostic bands, and a `failure_mode` string produced by the exposed-day failure classifier.
    """
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
    trace_metrics = _extract_exposed_day_trace_metrics(trace)
    behavior_metrics = {
        "food_distance_delta": food_progress,
        "food_eaten": int(stats.food_eaten),
        "predator_contacts": int(stats.predator_contacts),
        "alive": bool(stats.alive),
        "final_health": float(stats.final_health),
        **trace_metrics,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_exposed_day_foraging_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate exposed daytime foraging with a reproducible progress metric.",
        checks=checks,
        behavior_metrics=behavior_metrics,
    )


def _score_food_deprivation(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Compute the scenario score for the food deprivation scenario, producing pass/fail checks and behavior metrics describing recovery and foraging progress.
    
    Parameters:
        stats: Aggregated episode metrics used to evaluate checks and populate metrics.
        trace: Ordered execution trace used to derive per-tick diagnostics (distances, shelter exit, death tick, action-selection payloads).
    
    Returns:
        BehavioralEpisodeScore: Contains the scenario objective, per-check results, and a behavior_metrics mapping. behavior_metrics includes:
            - food_eaten: number of food items consumed.
            - food_distance_delta: net change in distance to food (positive indicates progress).
            - hunger_reduction: reduction in hunger from the scenario's initial value (clamped at zero).
            - alive: whether the agent survived the episode.
            - min_food_distance_reached: smallest trace-derived distance to food or None.
            - left_shelter: whether the spider exited the scenario shelter.
            - shelter_exit_tick: tick of first shelter exit, or None.
            - death_tick: tick where the spider first became not alive, or None.
            - hunger_valence_rate: fraction of action-selection ticks where hunger was the winning valence.
            - failure_mode: classifier label describing the dominant failure mode.
            - plus additional diagnostic fields produced by the scenario diagnostic helper.
    """
    hunger_reduction = float(max(0.0, FOOD_DEPRIVATION_INITIAL_HUNGER - stats.final_hunger))
    initial_food_distance = _trace_initial_food_distance(trace)
    if initial_food_distance is None and trace:
        initial_food_distance = _trace_reconstructed_initial_food_distance(trace[0])
    food_distances = _trace_food_distances(trace)
    if initial_food_distance is None and food_distances:
        initial_food_distance = food_distances[0]
    min_food_distance_reached = min(food_distances) if food_distances else None
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    death_tick = _trace_death_tick(trace)
    hunger_valence_rate = _hunger_valence_rate(trace)
    commits_to_foraging = bool(left_shelter and hunger_valence_rate >= 0.5)
    checks = (
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[0], passed=(stats.food_eaten > 0 or hunger_reduction >= 0.18), value=hunger_reduction),
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[1], passed=stats.food_distance_delta > 0.0, value=float(stats.food_distance_delta)),
        build_behavior_check(
            FOOD_DEPRIVATION_CHECKS[2],
            passed=commits_to_foraging,
            value={
                "left_shelter": bool(left_shelter),
                "hunger_valence_rate": hunger_valence_rate,
            },
        ),
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[3], passed=bool(stats.alive), value=bool(stats.alive)),
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
    behavior_metrics = {
        "food_eaten": int(stats.food_eaten),
        "food_distance_delta": float(stats.food_distance_delta),
        "hunger_reduction": hunger_reduction,
        "alive": bool(stats.alive),
        "min_food_distance_reached": min_food_distance_reached,
        "left_shelter": bool(left_shelter),
        "shelter_exit_tick": shelter_exit_tick,
        "death_tick": death_tick,
        "hunger_valence_rate": hunger_valence_rate,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_food_deprivation_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
            "initial_food_distance": initial_food_distance,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate measurable recovery or food progress under homeostatic deprivation.",
        checks=checks,
        behavior_metrics=behavior_metrics,
    )


def _score_visual_olfactory_pincer(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate the agent's response when visual and olfactory predator threats occur simultaneously.
    
    Builds three checks (dual detection, type-specific module engagement, and surviving without predator contact) and returns a BehaviorEpisodeScore populated with those checks and behavior_metrics that include visual/olfactory threat peaks, dominant predator types seen, per-type module share values, predator contact and response maps, and the final alive flag.
    
    Returns:
        BehavioralEpisodeScore: Score object containing the composed checks and a behavior_metrics mapping with keys such as
        "visual_predator_threat_peak", "olfactory_predator_threat_peak", "dominant_predator_types_seen",
        per-type cortex share values, predator contact/response maps, and "alive".
    """
    visual_peak = _trace_max_predator_threat(trace, "visual")
    olfactory_peak = _trace_max_predator_threat(trace, "olfactory")
    dominant_types_seen = sorted(_trace_dominant_predator_types(trace))
    visual_response = _module_response_for_type(stats, "visual")
    olfactory_response = _module_response_for_type(stats, "olfactory")
    visual_visual_share = _module_share_for_type(stats, "visual", "visual_cortex")
    visual_sensory_share = _module_share_for_type(stats, "visual", "sensory_cortex")
    olfactory_visual_share = _module_share_for_type(stats, "olfactory", "visual_cortex")
    olfactory_sensory_share = _module_share_for_type(stats, "olfactory", "sensory_cortex")
    dual_detected = bool(
        visual_peak > 0.0
        and olfactory_peak > 0.0
        and (
            (visual_peak > 0.0 and olfactory_peak > 0.0)
            or (
                sum(float(value) for value in visual_response.values()) > 0.0
                and sum(float(value) for value in olfactory_response.values()) > 0.0
            )
        )
    )
    type_specific_response = bool(
        dual_detected
        and visual_visual_share > 0.0
        and olfactory_sensory_share > 0.0
        and visual_visual_share > visual_sensory_share
        and olfactory_sensory_share > olfactory_visual_share
    )
    survives_cleanly = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(
            VISUAL_OLFACTORY_PINCER_CHECKS[0],
            passed=dual_detected,
            value={"visual_peak": visual_peak, "olfactory_peak": olfactory_peak},
        ),
        build_behavior_check(
            VISUAL_OLFACTORY_PINCER_CHECKS[1],
            passed=type_specific_response,
            value={
                "visual_visual_cortex": visual_visual_share,
                "visual_sensory_cortex": visual_sensory_share,
                "olfactory_visual_cortex": olfactory_visual_share,
                "olfactory_sensory_cortex": olfactory_sensory_share,
            },
        ),
        build_behavior_check(
            VISUAL_OLFACTORY_PINCER_CHECKS[2],
            passed=survives_cleanly,
            value={
                "alive": bool(stats.alive),
                "predator_contacts": int(stats.predator_contacts),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Measure whether the agent distinguishes simultaneous visual and olfactory predator pressure.",
        checks=checks,
        behavior_metrics={
            "visual_predator_threat_peak": visual_peak,
            "olfactory_predator_threat_peak": olfactory_peak,
            "dominant_predator_types_seen": dominant_types_seen,
            "visual_visual_cortex_share": visual_visual_share,
            "visual_sensory_cortex_share": visual_sensory_share,
            "olfactory_visual_cortex_share": olfactory_visual_share,
            "olfactory_sensory_cortex_share": olfactory_sensory_share,
            "module_response_by_predator_type": {
                "visual": dict(visual_response),
                "olfactory": dict(olfactory_response),
            },
            "predator_contacts_by_type": dict(stats.predator_contacts_by_type),
            "predator_response_latency_by_type": dict(stats.predator_response_latency_by_type),
            "alive": bool(stats.alive),
        },
    )


def _score_olfactory_ambush(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate an olfactory ambush scenario for whether olfactory threat drives sensory-cortex-led responses and safe survival.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics (module responses, contacts, survival, etc.).
        trace (Sequence[Dict[str, object]]): Ordered trace records from the episode used to extract time-series diagnostics.
    
    Returns:
        BehavioralEpisodeScore: Scoring object containing pass/fail checks, derived behavior metrics, and failure-mode diagnostics for the olfactory-ambush scenario.
    """
    initial_trace = trace[:OLFACTORY_AMBUSH_WINDOW_TICKS]
    visual_peak = _trace_max_predator_threat(trace, "visual")
    visual_peak_initial = _trace_max_predator_threat(initial_trace, "visual")
    olfactory_peak = _trace_max_predator_threat(trace, "olfactory")
    olfactory_peak_initial = _trace_max_predator_threat(initial_trace, "olfactory")
    sensory_share = _module_share_for_type(stats, "olfactory", "sensory_cortex")
    visual_share = _module_share_for_type(stats, "olfactory", "visual_cortex")
    predator_visible_ticks = sum(1 for item in trace if _trace_predator_visible(item))
    predator_visible_ticks_initial = sum(
        1 for item in initial_trace if _trace_predator_visible(item)
    )
    olfactory_detected = bool(
        olfactory_peak_initial > 0.0
        and (
            olfactory_peak_initial > 0.0
            or sum(
                float(value)
                for value in _module_response_for_type(stats, "olfactory").values()
            )
            > 0.0
        )
    )
    sensory_engaged = bool(
        olfactory_detected
        and sensory_share > 0.0
        and sensory_share > visual_share
        and (not initial_trace or visual_peak_initial <= 0.05)
    )
    survives_cleanly = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(
            OLFACTORY_AMBUSH_CHECKS[0],
            passed=olfactory_detected
            and (not initial_trace or predator_visible_ticks_initial == 0),
            value={
                "olfactory_peak": olfactory_peak,
                "olfactory_peak_initial": olfactory_peak_initial,
                "visual_peak": visual_peak,
                "visual_peak_initial": visual_peak_initial,
                "predator_visible_ticks": predator_visible_ticks,
                "predator_visible_ticks_initial": predator_visible_ticks_initial,
            },
        ),
        build_behavior_check(
            OLFACTORY_AMBUSH_CHECKS[1],
            passed=sensory_engaged,
            value={
                "sensory_cortex_share": sensory_share,
                "visual_cortex_share": visual_share,
            },
        ),
        build_behavior_check(
            OLFACTORY_AMBUSH_CHECKS[2],
            passed=survives_cleanly,
            value={
                "alive": bool(stats.alive),
                "predator_contacts": int(stats.predator_contacts),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Measure whether hidden olfactory threat recruits sensory-cortex-led response without visual contact.",
        checks=checks,
        behavior_metrics={
            "olfactory_predator_threat_peak": olfactory_peak,
            "olfactory_predator_threat_peak_initial": olfactory_peak_initial,
            "visual_predator_threat_peak": visual_peak,
            "visual_predator_threat_peak_initial": visual_peak_initial,
            "predator_visible_ticks": predator_visible_ticks,
            "predator_visible_ticks_initial": predator_visible_ticks_initial,
            "olfactory_sensory_cortex_share": sensory_share,
            "olfactory_visual_cortex_share": visual_share,
            "predator_contacts_by_type": dict(stats.predator_contacts_by_type),
            "alive": bool(stats.alive),
        },
    )


def _score_visual_hunter_open_field(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate episode behavior in an open-field scenario with a fast visual predator.
    
    Builds three checks that detect (1) whether a visual predator signal was present, (2) whether the visual-cortex module dominated the sensory module for the visual predator, and (3) whether the spider survived without predator contacts. Returns a BehaviorScore containing those checks and metrics including the visual threat peak, module share values, per-type contact and response-latency maps, and final alive status.
    
    Returns:
        BehavioralEpisodeScore: Aggregated score, checks, and behavior metrics for the visual-hunter open-field scenario.
    """
    visual_peak = _trace_max_predator_threat(trace, "visual")
    visual_share = _module_share_for_type(stats, "visual", "visual_cortex")
    sensory_share = _module_share_for_type(stats, "visual", "sensory_cortex")
    visual_detected = bool(
        visual_peak > 0.0
        and (
            visual_peak > 0.0
            or sum(
                float(value)
                for value in _module_response_for_type(stats, "visual").values()
            )
            > 0.0
        )
    )
    visual_engaged = bool(
        visual_detected
        and visual_share > 0.0
        and visual_share > sensory_share
    )
    survives_cleanly = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(
            VISUAL_HUNTER_OPEN_FIELD_CHECKS[0],
            passed=visual_detected,
            value=visual_peak,
        ),
        build_behavior_check(
            VISUAL_HUNTER_OPEN_FIELD_CHECKS[1],
            passed=visual_engaged,
            value={
                "visual_cortex_share": visual_share,
                "sensory_cortex_share": sensory_share,
            },
        ),
        build_behavior_check(
            VISUAL_HUNTER_OPEN_FIELD_CHECKS[2],
            passed=survives_cleanly,
            value={
                "alive": bool(stats.alive),
                "predator_contacts": int(stats.predator_contacts),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Measure whether open-field visual threat recruits visual-cortex-led predator response.",
        checks=checks,
        behavior_metrics={
            "visual_predator_threat_peak": visual_peak,
            "visual_visual_cortex_share": visual_share,
            "visual_sensory_cortex_share": sensory_share,
            "predator_contacts_by_type": dict(stats.predator_contacts_by_type),
            "predator_response_latency_by_type": dict(stats.predator_response_latency_by_type),
            "alive": bool(stats.alive),
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
        diagnostic_focus="Trace-backed failure_mode classification for corridor traversal, shelter exit, predator visibility, peak food progress, contact, and death timing.",
        success_interpretation="Full success requires positive food-distance progress through the corridor, zero predator contacts, and survival to episode end.",
        failure_interpretation=(
            'Use behavior_metrics.failure_mode: "success" means all corridor checks passed; '
            '"frozen_in_shelter" means no trace-confirmed shelter exit; '
            '"contact_failure_died" means predator contact occurred and the spider died; '
            '"contact_failure_survived" means predator contact occurred but the spider survived; '
            '"survived_no_progress" means the spider survived without contact but made no food-distance progress; '
            '"progress_then_died" means the spider made food-distance progress, avoided contact, then died; '
            '"scoring_mismatch" means stats and trace diagnostics reached an unexpected combination.'
        ),
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
        diagnostic_focus=(
            "Hypothesis: previous geometry placed lizard on food, forcing spider into "
            "threat override before food signal was available; separate retreat, "
            "stalling, partial progress, and death during exposed daytime foraging."
        ),
        success_interpretation="Success requires progress toward food without contact while staying alive under the nearby patrol.",
        failure_interpretation="Use behavior_metrics.failure_mode to distinguish cautious_inert, threatened_retreat, foraging_and_died, partial_progress, stall, and scoring_mismatch under the heuristic lizard-food/frontier spacing.",
        budget_note="Geometry fix applies a threat-radius heuristic around food and the first food-visible frontier; remaining failures indicate arbitration or training limits.",
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
        diagnostic_focus=(
            "Primary hypothesis: spider commits to foraging but homeostasis death timer "
            "(~12-15 ticks at hunger 0.96) must beat calibrated 4-6 Manhattan-distance food, "
            "with farther >=4 spawns only as fallback"
        ),
        success_interpretation="Success requires measurable hunger reduction, progress toward food, and survival.",
        failure_interpretation="Use behavior_metrics.failure_mode to distinguish no_commitment, orientation_failure, timing_failure, and scoring_mismatch during reproducible diagnosis.",
        budget_note="Tests whether learned behavior can beat the calibrated geometric race condition instead of asking it to survive an unreachable far-food timer.",
        max_steps=22,
        map_template="central_burrow",
        setup=_food_deprivation,
        score_episode=_score_food_deprivation,
    ),
    "visual_olfactory_pincer": ScenarioSpec(
        name="visual_olfactory_pincer",
        description="Spider is pinned between a visible hunter ahead and an olfactory hunter behind.",
        objective="Test whether predator-type-specific signals recruit distinct module responses under dual threat.",
        behavior_checks=VISUAL_OLFACTORY_PINCER_CHECKS,
        diagnostic_focus="Dual-threat detection, predator-type differentiation, and split visual-vs-sensory response.",
        success_interpretation="Success requires representing both predator types, showing type-specific module preference, and surviving without contact.",
        failure_interpretation="Failure suggests collapsed threat representation, weak specialization, or inability to escape the pincer safely.",
        budget_note="Short specialization probe for the new multi-predator ecology: both threat types are present on the opening frame.",
        max_steps=16,
        map_template="exposed_feeding_ground",
        setup=_visual_olfactory_pincer,
        score_episode=_score_visual_olfactory_pincer,
    ),
    "olfactory_ambush": ScenarioSpec(
        name="olfactory_ambush",
        description="An olfactory predator waits outside the shelter entrance while the spider faces inward.",
        objective="Test sensory-cortex-led response to hidden olfactory threat near refuge geometry.",
        behavior_checks=OLFACTORY_AMBUSH_CHECKS,
        diagnostic_focus="Olfactory-only threat perception, suppressed visual evidence, and sensory-cortex recruitment.",
        success_interpretation="Success requires detecting the hidden threat, preferring sensory-cortex response, and surviving the ambush cleanly.",
        failure_interpretation="Failure suggests the hidden predator is missed, treated as a generic threat, or still produces avoidable contact.",
        budget_note="Designed to isolate olfactory pressure without visible predator evidence on the opening frame.",
        max_steps=18,
        map_template="entrance_funnel",
        setup=_olfactory_ambush,
        score_episode=_score_olfactory_ambush,
    ),
    "visual_hunter_open_field": ScenarioSpec(
        name="visual_hunter_open_field",
        description="A fast visual hunter pressures the spider in exposed terrain.",
        objective="Test visual-cortex-led response to a fast visual predator in open terrain.",
        behavior_checks=VISUAL_HUNTER_OPEN_FIELD_CHECKS,
        diagnostic_focus="Visual threat detection, visual-cortex engagement, and survival in exposed terrain.",
        success_interpretation="Success requires explicit visual threat detection, stronger visual than sensory response, and survival without contact.",
        failure_interpretation="Failure suggests weak visual specialization or inability to stabilize behavior under open-field pursuit pressure.",
        budget_note="Pairs with olfactory_ambush to contrast visual-specialist and olfactory-specialist predator ecologies.",
        max_steps=16,
        map_template="exposed_feeding_ground",
        setup=_visual_hunter_open_field,
        score_episode=_score_visual_hunter_open_field,
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
