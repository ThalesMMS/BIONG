from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

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


def _night_rest(world: SpiderWorld) -> None:
    """
    Initialize the world for a night-rest scenario by placing the spider in a deep shelter, setting the simulation tick and the spider's hunger, fatigue, and sleep debt, placing a patrolling lizard at a safe spawn, and refreshing world memory.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = world.day_length + 1
    world.state.x, world.state.y = deep
    world.state.hunger = 0.08
    world.state.fatigue = 0.82
    world.state.sleep_debt = NIGHT_REST_INITIAL_SLEEP_DEBT
    lx, ly = _safe_lizard_cell(world)
    world.lizard = LizardState(x=lx, y=ly, mode="PATROL")
    world.refresh_memory(initial=True)


def _predator_edge(world: SpiderWorld) -> None:
    """
    Configure the world for the "predator edge" scenario.
    
    Sets the world state for a scenario where the spider starts near the map edge with a patrolling predator:
    - places the spider at (1, 1) and sets tick, hunger, fatigue, and sleep_debt;
    - places a single food item at the last food spawn cell;
    - places a lizard at (1, 3) in "PATROL" mode;
    - refreshes world memory.
    
    The function mutates the provided SpiderWorld in place.
    """
    world.tick = 1
    world.state.x, world.state.y = 1, 1
    world.state.hunger = 0.18
    world.state.fatigue = 0.15
    world.state.sleep_debt = 0.18
    world.food_positions = [world.map_template.food_spawn_cells[-1]]
    world.lizard = LizardState(x=1, y=3, mode="PATROL")
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
    world.state.x, world.state.y = interior
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
    world.state.x, world.state.y = deep
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
    world.state.x, world.state.y = deep
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
    world.state.x, world.state.y = deep
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
    world.state.x, world.state.y = deep
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
    world.state.x, world.state.y = left_deep
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
    world.state.x, world.state.y = deep
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
    Configure `world` for a deterministic food-deprivation scenario where the spider starts sheltered and acutely hungry.
    
    Sets the world tick and spider state (position, hunger, fatigue, sleep debt), places a single food spawn farthest from the spider, positions the lizard in `"PATROL"` at a safe spawn, and refreshes world memory with `initial=True`.
    
    Parameters:
        world (SpiderWorld): The world instance to mutate for the scenario.
    """
    deep = _first_cell(world.shelter_deep_cells or world.shelter_interior_cells or world.shelter_entrance_cells)
    world.tick = 4
    world.state.x, world.state.y = deep
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
    Checks whether any trace entry has a truthy "predator_escape" flag.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Sequence of trace items (dictionaries) to inspect. Each item may contain a "predator_escape" key whose truthiness is tested.
    
    Returns:
        bool: `True` if any trace item has a truthy `predator_escape` value, `False` otherwise.
    """
    return any(bool(item.get("predator_escape")) for item in trace)


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
    BehaviorCheckSpec("deep_night_shelter", "Mantem ocupacao profunda do abrigo durante a noite.", ">= 0.95"),
    BehaviorCheckSpec("deep_sleep_reached", "Atinge fase de sono profundo durante o cenário.", "true"),
    BehaviorCheckSpec("sleep_debt_reduced", "Reduz a dívida de sono em relação ao estado inicial.", ">= 0.45"),
)
PREDATOR_EDGE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("predator_detected", "Detecta a presença do predador no bordo do campo visual.", "true"),
    BehaviorCheckSpec("predator_memory_recorded", "Registra memória explícita do predador após o encontro.", "true"),
    BehaviorCheckSpec("predator_reacted", "Gera reação comportamental observável ao encontro.", "true"),
)
ENTRANCE_AMBUSH_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("survives_ambush", "Sobrevive à emboscada de entrada.", "true"),
    BehaviorCheckSpec("avoids_contact", "Evita contato direto com o predador na entrada.", "true"),
    BehaviorCheckSpec("keeps_shelter_safety", "Mantem abrigo/sobrevivência mesmo sob pressão de entrada.", ">= 0.75 occupancy or escape"),
)
OPEN_FIELD_FORAGING_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("made_food_progress", "Reduz a distância até a comida em área exposta.", "> 0"),
    BehaviorCheckSpec("foraging_viable", "Come ou reduz substancialmente a distância até a comida.", "eat or delta >= 2"),
    BehaviorCheckSpec("survives_exposure", "Sobrevive ao cenário de forrageio exposto.", "true"),
)
SHELTER_BLOCKADE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("survives_blockade", "Sobrevive ao bloqueio noturno da entrada.", "true"),
    BehaviorCheckSpec("avoids_blockade_contact", "Evita contato direto durante o bloqueio.", "true"),
    BehaviorCheckSpec("preserves_safety_or_escapes", "Mantem segurança noturna ou registra fuga do predador.", "occupancy >= 0.75 or escape"),
)
RECOVER_AFTER_FAILED_CHASE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("predator_enters_recover", "O predador entra em RECOVER após falhar a perseguição.", "true"),
    BehaviorCheckSpec("predator_returns_to_wait", "O predador retorna a WAIT após a perseguição.", "true"),
    BehaviorCheckSpec("spider_survives", "A aranha sobrevive ao pós-perseguição.", "true"),
)
CORRIDOR_GAUNTLET_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("corridor_food_progress", "Reduz a distância até a comida no corredor.", "> 0"),
    BehaviorCheckSpec("corridor_avoids_contact", "Evita contato direto no corredor estreito.", "true"),
    BehaviorCheckSpec("corridor_survives", "Sobrevive ao corredor de risco.", "true"),
)
TWO_SHELTER_TRADEOFF_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("tradeoff_survives", "Sobrevive ao cenário de escolha entre abrigos.", "true"),
    BehaviorCheckSpec("tradeoff_makes_progress", "Faz progresso em direção a comida ou segurança.", "food delta > 0 or occupancy >= 0.9"),
    BehaviorCheckSpec("tradeoff_night_shelter", "Mantem ocupação noturna alta em abrigo.", ">= 0.9"),
)
EXPOSED_DAY_FORAGING_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("day_food_progress", "Reduz a distância até a comida no período diurno.", "> 0"),
    BehaviorCheckSpec("day_avoids_contact", "Evita contato direto com o predador durante o forrageio.", "true"),
    BehaviorCheckSpec("day_survives", "Sobrevive ao forrageio diurno exposto.", "true"),
)
FOOD_DEPRIVATION_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("hunger_reduced", "Reduz a fome ou consegue comer sob privação.", "eat or hunger reduction >= 0.18"),
    BehaviorCheckSpec("approaches_food", "Faz progresso em direção à comida.", "> 0"),
    BehaviorCheckSpec("survives_deprivation", "Sobrevive ao episódio de privação.", "true"),
)


def _score_night_rest(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate night-rest behavior for the "night rest" scenario and produce a BehavioralEpisodeScore.
    
    Computes the proportion of nights spent in deep shelter and the reduction in sleep debt, verifies presence of a DEEP_SLEEP phase in the execution trace, builds the three scenario checks, and returns a behavior score containing those checks plus numeric metrics.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics (e.g., night_role_distribution, final_sleep_debt, night_stillness_rate, sleep_events).
        trace (Sequence[Dict[str, object]]): Execution trace entries; used to detect sleep-phase events such as "DEEP_SLEEP".
    
    Returns:
        BehavioralEpisodeScore: Score object summarizing check outcomes and behavior_metrics with keys:
            - "deep_night_rate": proportion of nights in deep shelter
            - "night_stillness_rate": recorded stillness rate during night
            - "sleep_debt_reduction": computed reduction in sleep debt
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
        objective="Validar repouso seguro e reproduzível em abrigo profundo durante a noite.",
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
        objective="Validar detecção e reação ao predador em um encontro controlado na borda visual.",
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
        objective="Validar sobrevivência e manutenção de segurança sob emboscada na entrada do abrigo.",
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
        objective="Validar forrageio reproduzível em campo aberto com métrica explícita de progresso.",
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
        objective="Validar comportamento seguro e reproduzível quando a entrada do abrigo está bloqueada.",
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
    Evaluate whether the predator transitioned into `RECOVER` and then returned to `WAIT` after a failed chase, and assemble the scenario behavior score.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics.
        trace (Sequence[Dict[str, object]]): Execution trace frames; each frame may contain a `state` dict with `lizard_mode`.
    
    Returns:
        BehavioralEpisodeScore: Score object containing the three behavior checks (entered `RECOVER`, returned to `WAIT`, spider alive) and a `behavior_metrics` map with keys:
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
        objective="Validar a transição determinística do predador para RECOVER/WAIT após uma perseguição fracassada.",
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
        objective="Validar progresso por corredor estreito sem depender apenas de reward agregado.",
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
        objective="Validar escolha comportamental entre progresso para comida e permanência segura em abrigo alternativo.",
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
    Evaluate exposed-day foraging outcome using food-distance progress and predator contact metrics.
    
    Parameters:
        stats (EpisodeStats): Episode-level metrics used to compute checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace (ignored by this scorer).
    
    Returns:
        BehavioralEpisodeScore: Aggregated score containing three checks (positive food-distance progress, zero predator contacts, and survival) and behavior_metrics with keys:
            - `food_distance_delta`: float progress in food distance,
            - `predator_contacts`: int number of predator contacts,
            - `alive`: bool whether the agent survived,
            - `final_health`: float final health value.
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
        objective="Validar forrageio diurno exposto com métrica de progresso reproduzível.",
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
    Compute the behavior score for the "food deprivation" scenario by evaluating recovery and foraging progress from episode statistics.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode metrics used to build checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace (ignored by this scorer).
    
    Returns:
        BehavioralEpisodeScore: Score object containing the scenario objective, a sequence of per-check results, and metrics:
            - food_eaten (int): number of food items consumed.
            - food_distance_delta (float): change in distance to food during the episode.
            - hunger_reduction (float): amount by which hunger decreased, clamped at zero.
            - alive (bool): whether the agent survived the episode.
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
        objective="Validar recuperação ou progresso alimentar mensurável sob privação homeostática.",
        checks=checks,
        behavior_metrics={
            "food_eaten": int(stats.food_eaten),
            "food_distance_delta": float(stats.food_distance_delta),
            "hunger_reduction": hunger_reduction,
            "alive": bool(stats.alive),
            **diagnostics,
        },
    )


SCENARIOS: Dict[str, ScenarioSpec] = {
    "night_rest": ScenarioSpec(
        name="night_rest",
        description="Aranha cansada e segura em abrigo profundo à noite.",
        objective="Confirmar repouso seguro com ocupação profunda e redução de dívida de sono.",
        behavior_checks=NIGHT_REST_CHECKS,
        diagnostic_focus="Repouso profundo, imobilidade noturna e redução de dívida de sono.",
        success_interpretation="Sucesso exige ocupação profunda consistente e recuperação clara de sono.",
        failure_interpretation="Falha sugere repouso instável, dívida de sono persistente ou perda do abrigo profundo.",
        budget_note="Este cenário já tende a passar no orçamento curto e serve como controle positivo de repouso.",
        max_steps=12,
        map_template="central_burrow",
        setup=_night_rest,
        score_episode=_score_night_rest,
    ),
    "predator_edge": ScenarioSpec(
        name="predator_edge",
        description="Predador aparece na borda do campo visual.",
        objective="Confirmar detecção e reação ao predador com memória explícita reproduzível.",
        behavior_checks=PREDATOR_EDGE_CHECKS,
        diagnostic_focus="Detecção do predador, memória explícita e reação observável ao encontro.",
        success_interpretation="Sucesso exige encontro percebido, memória registrada e resposta comportamental.",
        failure_interpretation="Falha indica problema de detecção, de memória explícita ou de reação ao encontro.",
        budget_note="Funciona como cenário curto de sanity check para percepção e resposta ao predador.",
        max_steps=18,
        map_template="central_burrow",
        setup=_predator_edge,
        score_episode=_score_predator_edge,
    ),
    "entrance_ambush": ScenarioSpec(
        name="entrance_ambush",
        description="Lagarto esperando perto da entrada do abrigo.",
        objective="Medir sobrevivência e segurança do abrigo sob emboscada na entrada.",
        behavior_checks=ENTRANCE_AMBUSH_CHECKS,
        diagnostic_focus="Sobrevivência, contato evitado e preservação da segurança do abrigo.",
        success_interpretation="Sucesso exige sobreviver sem contato direto e manter abrigo seguro ou rota de fuga.",
        failure_interpretation="Falha sugere quebra de segurança na entrada ou incapacidade de sustentar refúgio.",
        budget_note="Este cenário já tende a passar no orçamento curto e atua como controle de segurança do abrigo.",
        max_steps=18,
        map_template="entrance_funnel",
        setup=_entrance_ambush,
        score_episode=_score_entrance_ambush,
    ),
    "open_field_foraging": ScenarioSpec(
        name="open_field_foraging",
        description="Comida em área exposta e distante do abrigo.",
        objective="Medir progresso ou sucesso de forrageio em área aberta e controlada.",
        behavior_checks=OPEN_FIELD_FORAGING_CHECKS,
        diagnostic_focus="Separar progresso real para comida, regressão espacial e morte em campo aberto.",
        success_interpretation="Sucesso exige avançar para a comida de forma viável e sobreviver à exposição.",
        failure_interpretation="Falha pode significar regressão para longe da comida, estagnação ou morte antes de consolidar o forrageio.",
        budget_note="No orçamento curto, este cenário costuma expor regressão espacial e morte sem contato como sinais distintos de falha.",
        max_steps=20,
        map_template="exposed_feeding_ground",
        setup=_open_field_foraging,
        score_episode=_score_open_field_foraging,
    ),
    "shelter_blockade": ScenarioSpec(
        name="shelter_blockade",
        description="Lagarto bloqueando a entrada do abrigo durante a noite.",
        objective="Validar manutenção de segurança ou fuga sob bloqueio noturno de rota.",
        behavior_checks=SHELTER_BLOCKADE_CHECKS,
        diagnostic_focus="Sobrevivência noturna, contato evitado e manutenção da segurança do abrigo.",
        success_interpretation="Sucesso exige sobreviver e preservar segurança noturna ou fuga observável.",
        failure_interpretation="Falha sugere contato sob bloqueio ou perda da segurança do abrigo durante a noite.",
        budget_note="Cenário intermediário de pressão noturna; útil para comparar abrigo versus fuga.",
        max_steps=16,
        map_template="entrance_funnel",
        setup=_shelter_blockade,
        score_episode=_score_shelter_blockade,
    ),
    "recover_after_failed_chase": ScenarioSpec(
        name="recover_after_failed_chase",
        description="Lagarto precisa sair do ambush e entrar em recuperação após perder a aranha.",
        objective="Medir a transição comportamental do predador após uma perseguição fracassada.",
        behavior_checks=RECOVER_AFTER_FAILED_CHASE_CHECKS,
        diagnostic_focus="Sequência RECOVER -> WAIT do predador e sobrevivência da aranha após a perseguição.",
        success_interpretation="Sucesso exige a transição determinística do predador e sobrevivência do episódio.",
        failure_interpretation="Falha indica quebra da FSM do predador ou desfecho letal para a aranha.",
        budget_note="Cenário curto de regressão para a FSM do predador, menos sensível ao orçamento de treino.",
        max_steps=14,
        map_template="entrance_funnel",
        setup=_recover_after_failed_chase,
        score_episode=_score_recover_after_failed_chase,
    ),
    "corridor_gauntlet": ScenarioSpec(
        name="corridor_gauntlet",
        description="Aranha faminta cruza um corredor estreito com o lagarto no eixo de fuga.",
        objective="Medir progresso alimentar e segurança em um corredor de risco controlado.",
        behavior_checks=CORRIDOR_GAUNTLET_CHECKS,
        diagnostic_focus="Distinguir estagnação segura, progresso parcial e morte no corredor estreito.",
        success_interpretation="Sucesso exige avançar no corredor, evitar contato e sobreviver ao risco.",
        failure_interpretation="Falha pode significar estagnação, morte após progresso parcial ou ausência de resposta sob pressão.",
        budget_note="No orçamento curto, este cenário frequentemente mostra contato evitado sem progresso suficiente para sucesso completo.",
        max_steps=20,
        map_template="corridor_escape",
        setup=_corridor_gauntlet,
        score_episode=_score_corridor_gauntlet,
    ),
    "two_shelter_tradeoff": ScenarioSpec(
        name="two_shelter_tradeoff",
        description="Mapa com dois abrigos forçando escolha entre comida e rota de refúgio.",
        objective="Medir trade-off entre exploração alimentar e segurança entre dois abrigos.",
        behavior_checks=TWO_SHELTER_TRADEOFF_CHECKS,
        diagnostic_focus="Trade-off entre progresso alimentar e manutenção de abrigo alternativo.",
        success_interpretation="Sucesso exige sobreviver, preservar abrigo noturno e registrar progresso útil.",
        failure_interpretation="Falha sugere escolha ruim entre exploração e segurança ou incapacidade de sustentar abrigo.",
        budget_note="Cenário útil para comparar políticas mesmo quando o orçamento curto ainda é limitado.",
        max_steps=24,
        map_template="two_shelters",
        setup=_two_shelter_tradeoff,
        score_episode=_score_two_shelter_tradeoff,
    ),
    "exposed_day_foraging": ScenarioSpec(
        name="exposed_day_foraging",
        description="Forrageio diurno com comida em terreno aberto e patrulha próxima do lagarto.",
        objective="Medir progresso de forrageio diurno em terreno aberto sob patrulha próxima.",
        behavior_checks=EXPOSED_DAY_FORAGING_CHECKS,
        diagnostic_focus="Separar recuo, estagnação e morte em forrageio diurno exposto.",
        success_interpretation="Sucesso exige avançar para a comida sem contato e permanecer vivo sob patrulha próxima.",
        failure_interpretation="Falha pode significar recuo para longe da comida, ausência de progresso útil ou morte antes do objetivo.",
        budget_note="No orçamento curto, o cenário costuma revelar recuo e morte mesmo sem contato direto com o predador.",
        max_steps=20,
        map_template="exposed_feeding_ground",
        setup=_exposed_day_foraging,
        score_episode=_score_exposed_day_foraging,
    ),
    "food_deprivation": ScenarioSpec(
        name="food_deprivation",
        description="Aranha inicia com fome aguda e precisa recuperar homeostase com comida distante.",
        objective="Medir recuperação homeostática ou progresso alimentar explícito sob privação.",
        behavior_checks=FOOD_DEPRIVATION_CHECKS,
        diagnostic_focus="Separar aproximação da comida de recuperação homeostática efetiva sob privação.",
        success_interpretation="Sucesso exige reduzir fome de forma mensurável, progredir para a comida e sobreviver.",
        failure_interpretation="Falha pode esconder progresso parcial real quando a aranha se aproxima da comida mas morre antes de reduzir a fome.",
        budget_note="No orçamento curto, este cenário frequentemente registra aproximação útil sem recuperação homeostática completa.",
        max_steps=22,
        map_template="central_burrow",
        setup=_food_deprivation,
        score_episode=_score_food_deprivation,
    ),
}

SCENARIO_NAMES: Sequence[str] = tuple(SCENARIOS.keys())


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIOS:
        raise ValueError(f"Cenário desconhecido: {name}")
    return SCENARIOS[name]
