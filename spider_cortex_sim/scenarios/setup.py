from __future__ import annotations

from typing import Sequence

from ..perception import has_line_of_sight, visibility_confidence, visible_range
from ..predator import LizardState, OLFACTORY_HUNTER_PROFILE, VISUAL_HUNTER_PROFILE
from ..world import SpiderWorld
from .specs import (
    FAST_VISUAL_HUNTER_PROFILE,
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
)

def _first_cell(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """
    Selects the lexicographically smallest (x, y) coordinate from a collection of grid cells.

    Parameters:
        cells (set[tuple[int, int]] | tuple[tuple[int, int], ...]): Collection of (x, y) coordinates.

    Returns:
        tuple[int, int]: The coordinate with the smallest x, breaking ties by y.
    """
    if not cells:
        raise ValueError("no candidate cells provided to _first_cell")
    return sorted(cells)[0]


def _safe_lizard_cell(world: SpiderWorld) -> tuple[int, int]:
    """
    Selects a walkable lizard spawn cell that is farthest from the spider.

    Returns:
        cell (tuple[int, int]): A walkable lizard cell maximizing Manhattan distance to the spider.

    Raises:
        ValueError: If the map has no lizard-walkable cell.
    """
    spider = world.spider_pos()
    candidates = sorted(
        (
            cell
            for cell in world.map_template.lizard_spawn_cells
            if world.is_lizard_walkable(cell)
        ),
        key=lambda cell: -world.manhattan(cell, spider),
    )
    if candidates:
        return candidates[0]
    fallback_candidates = sorted(
        (
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if world.is_lizard_walkable((x, y))
        ),
        key=lambda cell: -world.manhattan(cell, spider),
    )
    if fallback_candidates:
        return fallback_candidates[0]
    raise ValueError("scenario setup requires at least one lizard-walkable cell")


def _safe_distinct_lizard_cell(
    world: SpiderWorld,
    excluded: set[tuple[int, int]],
) -> tuple[int, int]:
    candidate = _safe_lizard_cell(world)
    if candidate not in excluded:
        return candidate
    candidates = sorted(
        (
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if (x, y) not in excluded
            and world.is_lizard_walkable((x, y))
        ),
        key=lambda cell: -world.manhattan(cell, world.spider_pos()),
    )
    if candidates:
        return candidates[0]
    raise ValueError("scenario setup requires a distinct lizard-walkable cell")


def _food_visible_from_cell(
    world: SpiderWorld,
    source: tuple[int, int],
    food: tuple[int, int],
) -> bool:
    """Return whether `food` is visibly detectable from `source`."""
    radius = visible_range(world)
    distance = world.manhattan(source, food)
    if distance > radius or not has_line_of_sight(world, source, food):
        return False
    threshold = float(world.operational_profile.perception["visibility_binary_threshold"])
    confidence = visibility_confidence(
        world,
        source=source,
        target=food,
        dist=distance,
        radius=radius,
    )
    return bool(confidence >= threshold)


def _open_field_foraging_has_initial_food_signal(
    world: SpiderWorld,
    source: tuple[int, int],
    food: tuple[int, int],
) -> bool:
    """Return whether open-field setup gives the spider an initial food cue."""
    return (
        world.manhattan(source, food) < world.food_smell_range
        or _food_visible_from_cell(world, source, food)
    )


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
                if (
                    candidate in visited
                    or not (
                        0 <= candidate[0] < world.width
                        and 0 <= candidate[1] < world.height
                    )
                    or not world.is_walkable(candidate)
                ):
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
    Choose a food spawn intended for open-field foraging that is reachable from the given deep shelter, starts with a positive food cue, and is biased toward an exposed direction.

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
        and _open_field_foraging_has_initial_food_signal(world, deep, cell)
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
    signal_candidates = [
        cell
        for cell in food_spawns
        if _open_field_foraging_has_initial_food_signal(world, deep, cell)
    ]
    if signal_candidates:
        return min(
            signal_candidates,
            key=lambda cell: (
                abs(world.manhattan(cell, deep) - 5),
                cell[1],
                cell[0],
            ),
        )
    raise ValueError(
        "open_field_foraging requires at least one food spawn with a positive initial food signal"
    )


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
    deep_cells = sorted(world.shelter_deep_cells)
    food_spawns = world.map_template.food_spawn_cells
    lizard_spawns = world.map_template.lizard_spawn_cells
    if not deep_cells:
        raise ValueError("two_shelter_tradeoff requires at least one deep shelter cell")
    if not food_spawns:
        raise ValueError("two_shelter_tradeoff requires at least one food spawn cell")
    if not lizard_spawns:
        raise ValueError("two_shelter_tradeoff requires at least one lizard spawn cell")
    left_deep = deep_cells[0]
    right_deep = deep_cells[-1]
    world.tick = world.day_length - 2
    _teleport_spider(world, left_deep)
    world.state.hunger = 0.68
    world.state.fatigue = 0.38
    world.state.sleep_debt = 0.42
    world.food_positions = [min(food_spawns, key=lambda cell: world.manhattan(cell, right_deep))]
    center_spawn = min(
        lizard_spawns,
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
    if not world.map_template.food_spawn_cells:
        raise ValueError("food_deprivation has no food spawn cells in map_template")
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
        food = min(
            calibrated_food,
            key=lambda cell: (-world.manhattan(cell, deep), cell[0], cell[1]),
        )
    else:
        safe_default = distance_ranked_food[-1]
        food = next(
            (cell for cell in distance_ranked_food if world.manhattan(cell, deep) >= 4),
            safe_default,
        )
    world.food_positions = [food]
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
        if not spider_candidates:
            raise ValueError(
                "food_vs_predator_conflict requires an open traversable spider cell "
                f"within 3 steps of food {food}"
            )
        spider = spider_candidates[0]
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
        if not lizard_candidates:
            raise ValueError(
                "food_vs_predator_conflict requires a lizard-walkable cell within 1-2 steps of the spider"
            )
        lizard = lizard_candidates[0]
    _teleport_spider(world, spider)
    world.state.hunger = 0.92
    world.state.fatigue = 0.24
    world.state.sleep_debt = 0.18
    world.food_positions = [food]
    world.lizard = LizardState(x=lizard[0], y=lizard[1], mode="PATROL")
    world.state.heading_dx, world.state.heading_dy = world._heading_toward(world.lizard_pos())
    world.refresh_memory(initial=True)


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
    if not open_cells:
        raise ValueError("visual_olfactory_pincer requires at least one open traversable cell")
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
    visual_pos = (
        visual_candidates[0]
        if visual_candidates
        else _safe_distinct_lizard_cell(world, {spider})
    )

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
    olfactory_pos = (
        olfactory_candidates[0]
        if olfactory_candidates
        else _safe_distinct_lizard_cell(world, {spider, visual_pos})
    )

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
    spider_cells = (
        open_cells
        or sorted(world.map_template.traversable_cells)
        or sorted(world.shelter_cells)
    )
    if not spider_cells:
        raise ValueError("visual_hunter_open_field requires at least one spider placement cell")
    preferred_spider = (6, 6)
    spider = (
        preferred_spider
        if preferred_spider in spider_cells
        else spider_cells[len(spider_cells) // 2]
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
    if predator_candidates:
        predator_pos = predator_candidates[0]
    else:
        nearest_candidates = sorted(
            (
                cell
                for cell in world.map_template.traversable_cells
                if world.is_lizard_walkable(cell)
                and cell != spider
            ),
            key=lambda cell: (
                world.manhattan(cell, spider),
                cell[0],
                cell[1],
            ),
        )
        predator_pos = (
            nearest_candidates[0]
            if nearest_candidates
            else _safe_distinct_lizard_cell(world, {spider})
        )
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

__all__ = [
    '_corridor_gauntlet',
    '_entrance_ambush',
    '_entrance_ambush_cell',
    '_exposed_day_foraging',
    '_first_cell',
    '_first_food_visible_frontier',
    '_food_deprivation',
    '_food_visible_from_cell',
    '_food_vs_predator_conflict',
    '_night_rest',
    '_olfactory_ambush',
    '_open_field_foraging',
    '_open_field_foraging_food_cell',
    '_open_field_foraging_has_initial_food_signal',
    '_predator_edge',
    '_recover_after_failed_chase',
    '_safe_lizard_cell',
    '_set_predators',
    '_shelter_blockade',
    '_sleep_vs_exploration_conflict',
    '_teleport_spider',
    '_two_shelter_tradeoff',
    '_visual_hunter_open_field',
    '_visual_olfactory_pincer',
]
