from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple


OPEN = "OPEN"
CLUTTER = "CLUTTER"
NARROW = "NARROW"
BLOCKED = "BLOCKED"

MAP_TEMPLATE_NAMES: Sequence[str] = (
    "central_burrow",
    "side_burrow",
    "corridor_escape",
    "two_shelters",
    "exposed_feeding_ground",
    "entrance_funnel",
)


@dataclass(frozen=True)
class MapTemplate:
    name: str
    width: int
    height: int
    terrain: Dict[Tuple[int, int], str]
    shelter_entrance: tuple[Tuple[int, int], ...]
    shelter_interior: tuple[Tuple[int, int], ...]
    shelter_deep: tuple[Tuple[int, int], ...]
    blocked_cells: tuple[Tuple[int, int], ...]
    food_spawn_cells: tuple[Tuple[int, int], ...]
    lizard_spawn_cells: tuple[Tuple[int, int], ...]
    spider_start: Tuple[int, int]

    @property
    def shelter_cells(self) -> set[Tuple[int, int]]:
        return set(self.shelter_entrance) | set(self.shelter_interior) | set(self.shelter_deep)

    @property
    def traversable_cells(self) -> set[Tuple[int, int]]:
        blocked = set(self.blocked_cells)
        return {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in blocked
        }


def terrain_at(template: MapTemplate, pos: Tuple[int, int]) -> str:
    if pos in template.blocked_cells:
        return BLOCKED
    return template.terrain.get(pos, OPEN)


def build_map_template(name: str, *, width: int, height: int) -> MapTemplate:
    """
    Create a MapTemplate for the given template name and dimensions.
    
    Parameters:
        name (str): Template identifier (one of MAP_TEMPLATE_NAMES).
        width (int): Map width in cells.
        height (int): Map height in cells.
    
    Returns:
        MapTemplate: A map template configured for the specified name and size.
    
    Raises:
        ValueError: If `name` is not a supported template identifier.
    """
    builders = {
        "central_burrow": _build_central_burrow,
        "side_burrow": _build_side_burrow,
        "corridor_escape": _build_corridor_escape,
        "two_shelters": _build_two_shelters,
        "exposed_feeding_ground": _build_exposed_feeding_ground,
        "entrance_funnel": _build_entrance_funnel,
    }
    if name not in builders:
        raise ValueError(f"Mapa desconhecido: {name}")
    return builders[name](width=width, height=height)


def _clamp_cell(width: int, height: int, pos: Tuple[int, int]) -> Tuple[int, int]:
    return (
        max(0, min(width - 1, int(pos[0]))),
        max(0, min(height - 1, int(pos[1]))),
    )


def _valid_cells(width: int, height: int, cells: Iterable[Tuple[int, int]]) -> list[Tuple[int, int]]:
    valid = []
    seen: set[Tuple[int, int]] = set()
    for cell in cells:
        if not (0 <= cell[0] < width and 0 <= cell[1] < height):
            continue
        if cell in seen:
            continue
        seen.add(cell)
        valid.append(cell)
    return valid


def _base_spawn_cells(
    *,
    width: int,
    height: int,
    blocked: set[Tuple[int, int]],
    shelter: set[Tuple[int, int]],
    min_dist_from_shelter: int,
    x_range: Tuple[int, int] | None = None,
) -> list[Tuple[int, int]]:
    """
    Selects candidate spawn cells that are not blocked, not part of the shelter, and are at least a given Manhattan distance from the shelter center, optionally limited to an inclusive x-range.
    
    Parameters:
        width (int): Map width used to clamp the x_range.
        height (int): Map height used to enumerate y coordinates.
        blocked (set[Tuple[int,int]]): Cells to exclude from candidates.
        shelter (set[Tuple[int,int]]): Shelter cells to exclude; used to compute the shelter center.
        min_dist_from_shelter (int): Minimum Manhattan distance from the shelter center required for a cell to be included (cells with distance >= this value are kept).
        x_range (Tuple[int,int] | None): Optional inclusive (x0, x1) bounds to restrict candidate x coordinates; values are clamped into [0, width-1]. If None the full x-span is used.
    
    Returns:
        list[Tuple[int,int]]: List of (x, y) positions meeting the criteria, in iteration order.
    """
    sx = [pos[0] for pos in shelter]
    sy = [pos[1] for pos in shelter]
    shelter_center = (sum(sx) / max(1, len(sx)), sum(sy) / max(1, len(sy)))
    cells = []
    x0, x1 = x_range if x_range is not None else (0, width - 1)
    for x in range(max(0, x0), min(width - 1, x1) + 1):
        for y in range(height):
            pos = (x, y)
            if pos in blocked or pos in shelter:
                continue
            dist = abs(pos[0] - shelter_center[0]) + abs(pos[1] - shelter_center[1])
            if dist < min_dist_from_shelter:
                continue
            cells.append(pos)
    return cells


def _available_cells(
    *,
    width: int,
    height: int,
    blocked: set[Tuple[int, int]],
    shelter: set[Tuple[int, int]],
    x_range: Tuple[int, int] | None = None,
    y_range: Tuple[int, int] | None = None,
) -> list[Tuple[int, int]]:
    """
    Enumerates grid cells within optional x/y ranges that are not in `blocked` or `shelter`.
    
    Parameters:
        width (int): Map width in cells.
        height (int): Map height in cells.
        blocked (set[Tuple[int, int]]): Positions to exclude.
        shelter (set[Tuple[int, int]]): Shelter positions to exclude.
        x_range (Tuple[int, int] | None): Inclusive (min_x, max_x) bounds to restrict x; if `None` uses [0, width-1]. Out-of-bounds endpoints are clamped to the map.
        y_range (Tuple[int, int] | None): Inclusive (min_y, max_y) bounds to restrict y; if `None` uses [0, height-1]. Out-of-bounds endpoints are clamped to the map.
    
    Returns:
        list[Tuple[int, int]]: Sorted (by iteration order) list of (x, y) positions inside the specified ranges and map bounds, excluding any positions present in `blocked` or `shelter`.
    """
    x0, x1 = x_range if x_range is not None else (0, width - 1)
    y0, y1 = y_range if y_range is not None else (0, height - 1)
    cells = []
    for x in range(max(0, x0), min(width - 1, x1) + 1):
        for y in range(max(0, y0), min(height - 1, y1) + 1):
            pos = (x, y)
            if pos in blocked or pos in shelter:
                continue
            cells.append(pos)
    return cells


def _distance_to_region(pos: Tuple[int, int], region: Iterable[Tuple[int, int]]) -> int:
    """
    Compute the minimum Manhattan distance from `pos` to any cell in `region`.
    
    Parameters:
        pos (tuple[int, int]): The (x, y) position to measure from.
        region (Iterable[tuple[int, int]]): An iterable of (x, y) positions representing the region.
    
    Returns:
        distance (int): The smallest Manhattan distance from `pos` to a cell in `region`, or 10**9 if `region` is empty.
    """
    return min((abs(pos[0] - cx) + abs(pos[1] - cy) for cx, cy in region), default=10**9)


def _build_central_burrow(*, width: int, height: int) -> MapTemplate:
    """
    Constructs a centered "central_burrow" MapTemplate with a three-row shelter and surrounding terrain/features.
    
    The returned template places the shelter centered at the map midpoint (entrance, interior, deep rows), marks a small ring of nearby blocked cells, leaves terrain empty, and computes food and lizard spawn lists and the spider start position consistent with the central burrow layout.
    
    Returns:
        MapTemplate: A MapTemplate named "central_burrow" containing shelter cell groups, blocked_cells, terrain, food_spawn_cells, lizard_spawn_cells, and spider_start.
    """
    cx = width // 2
    cy = height // 2
    entrance = tuple(
        _valid_cells(width, height, [(cx - 1, cy - 1), (cx, cy - 1), (cx + 1, cy - 1)])
    )
    interior = tuple(
        _valid_cells(width, height, [(cx - 1, cy), (cx, cy), (cx + 1, cy)])
    )
    deep = tuple(
        _valid_cells(width, height, [(cx - 1, cy + 1), (cx, cy + 1), (cx + 1, cy + 1)])
    )

    blocked = set(
        _valid_cells(
            width,
            height,
            [
                (cx - 2, cy),
                (cx + 2, cy),
                (cx - 1, cy + 2),
                (cx, cy + 2),
                (cx + 1, cy + 2),
            ],
        )
    )
    shelter = set(entrance) | set(interior) | set(deep)
    food_spawn = _base_spawn_cells(
        width=width,
        height=height,
        blocked=blocked,
        shelter=shelter,
        min_dist_from_shelter=4,
    )
    lizard_spawn = [
        cell
        for cell in food_spawn
        if cell[0] in {0, width - 1} or cell[1] in {0, height - 1}
    ]
    return MapTemplate(
        name="central_burrow",
        width=width,
        height=height,
        terrain={},
        shelter_entrance=entrance,
        shelter_interior=interior,
        shelter_deep=deep,
        blocked_cells=tuple(sorted(blocked)),
        food_spawn_cells=tuple(food_spawn),
        lizard_spawn_cells=tuple(lizard_spawn),
        spider_start=deep[0],
    )


def _build_side_burrow(*, width: int, height: int) -> MapTemplate:
    cy = height // 2
    entrance = tuple(
        _valid_cells(width, height, [(2, cy - 1), (2, cy), (2, cy + 1)])
    )
    interior = tuple(
        _valid_cells(width, height, [(3, cy - 1), (3, cy), (3, cy + 1)])
    )
    deep = tuple(
        _valid_cells(width, height, [(4, cy - 1), (4, cy), (4, cy + 1)])
    )

    blocked = set(
        _valid_cells(
            width,
            height,
            [
                (5, cy - 1),
                (5, cy),
                (5, cy + 1),
            ],
        )
    )
    terrain = {
        cell: CLUTTER
        for cell in _valid_cells(
            width,
            height,
            [(width // 2, y) for y in range(1, height - 1)],
        )
    }
    shelter = set(entrance) | set(interior) | set(deep)
    food_spawn = _base_spawn_cells(
        width=width,
        height=height,
        blocked=blocked,
        shelter=shelter,
        min_dist_from_shelter=5,
        x_range=(width // 2, width - 1),
    )
    lizard_spawn = [
        cell
        for cell in food_spawn
        if cell[0] >= max(0, width - 3) or cell[1] in {0, height - 1}
    ]
    return MapTemplate(
        name="side_burrow",
        width=width,
        height=height,
        terrain=terrain,
        shelter_entrance=entrance,
        shelter_interior=interior,
        shelter_deep=deep,
        blocked_cells=tuple(sorted(blocked)),
        food_spawn_cells=tuple(food_spawn),
        lizard_spawn_cells=tuple(lizard_spawn),
        spider_start=deep[0],
    )


def _build_corridor_escape(*, width: int, height: int) -> MapTemplate:
    """
    Constructs the "corridor_escape" MapTemplate: a shelter with a narrow corridor and surrounding blocked cells.
    
    The template places a three-cell shelter aligned horizontally near the left edge and creates a corridor of `NARROW` terrain extending rightwards. Cells adjacent to the corridor are blocked to form the escape passage. Food spawn candidates are selected on the far right (x starting at max(5, width-4)) and must be at least 6 Manhattan distance from the shelter; lizard spawns are the subset of those food spawn cells with x >= width-3. The returned MapTemplate sets `spider_start` to the shelter deep cell.
    
    Parameters:
        width (int): map width in cells.
        height (int): map height in cells.
    
    Returns:
        MapTemplate: a populated MapTemplate named "corridor_escape" with computed `terrain`, `blocked_cells`, `shelter_entrance`, `shelter_interior`, `shelter_deep`, `food_spawn_cells`, `lizard_spawn_cells`, and `spider_start`.
    """
    cy = height // 2
    entrance = (_clamp_cell(width, height, (3, cy)),)
    interior = (_clamp_cell(width, height, (2, cy)),)
    deep = (_clamp_cell(width, height, (1, cy)),)

    blocked = set()
    for x in range(4, max(4, width - 3)):
        for y in range(height):
            if y == cy:
                continue
            if y in {cy - 1, cy + 1} and x < width - 4:
                blocked.add((x, y))
    blocked.update(
        _valid_cells(
            width,
            height,
            [
                (1, cy - 1),
                (1, cy + 1),
                (2, cy - 1),
                (2, cy + 1),
                (3, cy - 1),
                (3, cy + 1),
            ],
        )
    )
    terrain = {
        (x, cy): NARROW
        for x in range(4, max(4, width - 2))
    }
    shelter = set(entrance) | set(interior) | set(deep)
    food_spawn = _base_spawn_cells(
        width=width,
        height=height,
        blocked=blocked,
        shelter=shelter,
        min_dist_from_shelter=6,
        x_range=(max(5, width - 4), width - 1),
    )
    lizard_spawn = [
        cell
        for cell in food_spawn
        if cell[0] >= max(0, width - 3)
    ]
    return MapTemplate(
        name="corridor_escape",
        width=width,
        height=height,
        terrain=terrain,
        shelter_entrance=entrance,
        shelter_interior=interior,
        shelter_deep=deep,
        blocked_cells=tuple(sorted(blocked)),
        food_spawn_cells=tuple(food_spawn),
        lizard_spawn_cells=tuple(lizard_spawn),
        spider_start=deep[0],
    )


def _build_two_shelters(*, width: int, height: int) -> MapTemplate:
    """
    Constructs a map template with two symmetric shelter clusters left and right of center, a cluttered central lane, and computed spawn/blocked cells.
    
    The returned MapTemplate places paired entrance/interior/deep shelter regions, computes blocked cells near each shelter and at the top/bottom central column, marks CLUTTER terrain in a narrow central vertical band (excluding the shelter rows), selects food spawn cells at Manhattan distance >= 3 from any shelter (and extends the list until at least four cells exist), selects lizard spawn cells on the outer columns, the center column, or outer rows, and sets the spider start position to the first deep shelter cell.
    
    Returns:
        MapTemplate: A fully populated map template named "two_shelters" for the given width and height.
    """
    if width < 8:
        raise ValueError("two_shelters requires width >= 8")

    cy = height // 2
    left_x = 2
    right_x = max(left_x + 5, width - 3)

    entrance = tuple(
        _valid_cells(
            width,
            height,
            [
                (left_x, cy - 1),
                (left_x, cy),
                (right_x, cy - 1),
                (right_x, cy),
            ],
        )
    )
    interior = tuple(
        _valid_cells(
            width,
            height,
            [
                (left_x + 1, cy - 1),
                (left_x + 1, cy),
                (right_x - 1, cy - 1),
                (right_x - 1, cy),
            ],
        )
    )
    deep = tuple(
        _valid_cells(
            width,
            height,
            [
                (left_x + 2, cy - 1),
                (left_x + 2, cy),
                (right_x - 2, cy - 1),
                (right_x - 2, cy),
            ],
        )
    )

    blocked = set(
        _valid_cells(
            width,
            height,
            [
                (left_x + 3, cy - 1),
                (left_x + 3, cy),
                (right_x - 3, cy - 1),
                (right_x - 3, cy),
                (width // 2, 0),
                (width // 2, height - 1),
            ],
        )
    )
    shelter = set(entrance) | set(interior) | set(deep)
    blocked -= shelter
    terrain = {
        cell: CLUTTER
        for cell in _available_cells(
            width=width,
            height=height,
            blocked=blocked,
            shelter=shelter,
            x_range=(max(0, width // 2 - 1), min(width - 1, width // 2 + 1)),
            y_range=(1, max(1, height - 2)),
        )
        if cell[1] not in {cy - 1, cy, cy + 1}
    }
    central_lane = _available_cells(
        width=width,
        height=height,
        blocked=blocked,
        shelter=shelter,
        x_range=(max(0, width // 2 - 1), min(width - 1, width // 2 + 1)),
        y_range=(1, max(1, height - 2)),
    )
    food_spawn = [
        cell
        for cell in central_lane
        if _distance_to_region(cell, shelter) >= 3 and terrain.get(cell) != CLUTTER
    ]
    if len(food_spawn) < 4:
        food_spawn.extend(
            cell
            for cell in _available_cells(
                width=width,
                height=height,
                blocked=blocked,
                shelter=shelter,
            )
            if _distance_to_region(cell, shelter) >= 3 and cell not in food_spawn
        )
    lizard_spawn = [
        cell
        for cell in _available_cells(
            width=width,
            height=height,
            blocked=blocked,
            shelter=shelter,
        )
        if cell[0] in {0, width - 1, width // 2} or cell[1] in {0, height - 1}
    ]
    if not deep:
        raise ValueError("cannot determine spider_start: deep is empty")
    spider_start = sorted(deep)[0]
    return MapTemplate(
        name="two_shelters",
        width=width,
        height=height,
        terrain=terrain,
        shelter_entrance=entrance,
        shelter_interior=interior,
        shelter_deep=deep,
        blocked_cells=tuple(sorted(blocked)),
        food_spawn_cells=tuple(food_spawn),
        lizard_spawn_cells=tuple(lizard_spawn),
        spider_start=spider_start,
    )


def _build_exposed_feeding_ground(*, width: int, height: int) -> MapTemplate:
    """
    Create a MapTemplate named "exposed_feeding_ground" with a three-cell vertical shelter on the left side and a designated feeding zone to the right.
    
    The template defines:
    - shelter entrance, interior, and deep regions positioned at x=2, x=3, and x=4 (centered on the map's vertical midpoint and clamped to bounds);
    - a small set of blocked cells near the shelter mouth and two near the top/bottom on the right;
    - a feeding zone computed as available (non-blocked, non-shelter) cells within a right-side rectangle; these cells are used as food spawn locations;
    - `CLUTTER` terrain for available cells that are not in the feeding zone and have x < width - 2;
    - lizard spawn cells chosen from available cells that are near the right boundary (x >= width-3) or on the top/bottom rows;
    - `spider_start` set to the first cell of the deep shelter region.
    
    @returns:
        A MapTemplate with fields populated as described: `name="exposed_feeding_ground"`, `width`, `height`, `terrain` (mapping cells to `CLUTTER`), `shelter_entrance`, `shelter_interior`, `shelter_deep`, `blocked_cells` (sorted tuple), `food_spawn_cells` (sorted tuple from the feeding zone), `lizard_spawn_cells` (tuple), and `spider_start` (first deep cell).
    """
    cy = height // 2
    entrance = tuple(
        _valid_cells(width, height, [(2, cy - 1), (2, cy), (2, cy + 1)])
    )
    interior = tuple(
        _valid_cells(width, height, [(3, cy - 1), (3, cy), (3, cy + 1)])
    )
    deep = tuple(
        _valid_cells(width, height, [(4, cy - 1), (4, cy), (4, cy + 1)])
    )
    blocked = set(
        _valid_cells(
            width,
            height,
            [
                (5, cy - 1),
                (5, cy),
                (5, cy + 1),
                (width - 2, 1),
                (width - 2, height - 2),
            ],
        )
    )
    shelter = set(entrance) | set(interior) | set(deep)

    feeding_zone = set(
        _available_cells(
            width=width,
            height=height,
            blocked=blocked,
            shelter=shelter,
            x_range=(max(5, width // 2 - 1), max(5, width - 2)),
            y_range=(2, max(2, height - 3)),
        )
    )
    if not deep:
        raise ValueError(
            f"exposed_feeding_ground: no deep shelter cells for width={width} height={height}"
        )
    if not feeding_zone:
        raise ValueError(
            f"exposed_feeding_ground: no food spawn cells for width={width} height={height}"
        )
    terrain = {
        cell: CLUTTER
        for cell in _available_cells(
            width=width,
            height=height,
            blocked=blocked,
            shelter=shelter,
        )
        if cell not in feeding_zone and cell[0] < width - 2
    }
    food_spawn = sorted(feeding_zone)
    lizard_spawn = [
        cell
        for cell in _available_cells(
            width=width,
            height=height,
            blocked=blocked,
            shelter=shelter,
        )
        if cell[0] >= max(0, width - 3) or cell[1] in {0, height - 1}
    ]
    return MapTemplate(
        name="exposed_feeding_ground",
        width=width,
        height=height,
        terrain=terrain,
        shelter_entrance=entrance,
        shelter_interior=interior,
        shelter_deep=deep,
        blocked_cells=tuple(sorted(blocked)),
        food_spawn_cells=tuple(food_spawn),
        lizard_spawn_cells=tuple(lizard_spawn),
        spider_start=deep[0],
    )


def _build_entrance_funnel(*, width: int, height: int) -> MapTemplate:
    """
    Constructs the "entrance_funnel" MapTemplate with a narrow funnel approach to the shelter.
    
    The shelter is placed near x=2..4 around the vertical center; funnel blockers are placed above and below the approach, and `NARROW` terrain is set on the funnel line at x in {5,6,7} when not blocked or part of the shelter. `food_spawn_cells` are available cells to the right (x >= max(7, width//2)) that are at least Manhattan distance 5 from the shelter. `lizard_spawn_cells` are available cells on the right side (x >= max(5, width//2)) or on the top/bottom rows. `spider_start` is the first deep shelter cell.
    
    Returns:
        MapTemplate: A MapTemplate named "entrance_funnel" populated with computed shelter regions, blocked cells, terrain, food and lizard spawn lists, and the spider start cell.
    """
    cy = height // 2
    entrance = (_clamp_cell(width, height, (4, cy)),)
    interior = tuple(
        _valid_cells(width, height, [(3, cy - 1), (3, cy), (3, cy + 1)])
    )
    deep = tuple(
        _valid_cells(width, height, [(2, cy - 1), (2, cy), (2, cy + 1)])
    )

    blocked = set(
        _valid_cells(
            width,
            height,
            [
                (1, cy - 2),
                (1, cy + 2),
                (2, cy - 2),
                (2, cy + 2),
                (3, cy - 2),
                (3, cy + 2),
                (4, cy - 2),
                (4, cy + 2),
                (5, cy - 1),
                (5, cy + 1),
                (6, cy - 1),
                (6, cy + 1),
            ],
        )
    )
    shelter = set(entrance) | set(interior) | set(deep)
    terrain = {
        cell: NARROW
        for cell in _valid_cells(
            width,
            height,
            [
                (5, cy),
                (6, cy),
                (7, cy),
            ],
        )
        if cell not in blocked and cell not in shelter
    }
    food_spawn = [
        cell
        for cell in _available_cells(
            width=width,
            height=height,
            blocked=blocked,
            shelter=shelter,
            x_range=(max(7, width // 2), width - 1),
        )
        if _distance_to_region(cell, shelter) >= 5
    ]
    lizard_spawn = [
        cell
        for cell in _available_cells(
            width=width,
            height=height,
            blocked=blocked,
            shelter=shelter,
        )
        if cell[0] >= max(5, width // 2) or cell[1] in {0, height - 1}
    ]
    if not deep:
        raise ValueError(
            f"entrance_funnel: no deep shelter cells for width={width} height={height}"
        )
    if not food_spawn:
        raise ValueError(
            f"entrance_funnel: no food spawn cells for width={width} height={height}"
        )
    if not lizard_spawn:
        raise ValueError(
            f"entrance_funnel: no lizard spawn cells for width={width} height={height}"
        )
    return MapTemplate(
        name="entrance_funnel",
        width=width,
        height=height,
        terrain=terrain,
        shelter_entrance=entrance,
        shelter_interior=interior,
        shelter_deep=deep,
        blocked_cells=tuple(sorted(blocked)),
        food_spawn_cells=tuple(food_spawn),
        lizard_spawn_cells=tuple(lizard_spawn),
        spider_start=deep[0],
    )
