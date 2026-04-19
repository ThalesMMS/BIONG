from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ..interfaces import ACTION_DELTAS
from ..maps import build_map_template
from .specs import TRACE_MAP_HEIGHT_FALLBACK, TRACE_MAP_WIDTH_FALLBACK

from .trace_core import _float_or_none, _int_or_none, _payload_text, _state_alive, _state_position, _trace_action_selection_payloads, _trace_tick
from .trace_food import _trace_item_food_distances, _trace_peak_food_progress, _trace_pre_tick_snapshot_payload

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
        if state.get("shelter_role") == "outside":
            return True, tick
        if pos is not None and shelter_cells:
            if pos not in shelter_cells:
                return True, tick
            continue
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
    final_distance_to_food = None
    for item in reversed(trace):
        item_distances = _trace_item_food_distances(item)
        if item_distances:
            final_distance_to_food = item_distances[-1]
            break
    return {
        "shelter_exit_tick": shelter_exit_tick,
        "left_shelter": bool(left_shelter),
        "peak_food_progress": _trace_peak_food_progress(trace),
        "predator_visible_ticks": sum(
            1 for item in trace if _trace_predator_visible(item)
        ),
        "final_distance_to_food": final_distance_to_food,
    }
