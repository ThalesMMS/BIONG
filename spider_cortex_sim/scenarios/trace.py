from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ..interfaces import ACTION_DELTAS
from ..maps import build_map_template
from .specs import TRACE_MAP_HEIGHT_FALLBACK, TRACE_MAP_WIDTH_FALLBACK

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


def _trace_item_food_distances(item: Mapping[str, object]) -> list[float]:
    distances: list[float] = []
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


def _trace_food_distances(trace: Sequence[Dict[str, object]]) -> list[float]:
    """
    Collects all reported spider-to-food distance samples from a trace.

    Parameters:
        trace (Sequence[Dict[str, object]]): Execution trace items containing observation-derived food distances, optional `state` fields with a food distance, and optional `distance_deltas` mapping which may include a `food` delta.

    Returns:
        list[float]: Unique extracted food-distance samples in first-seen order. Samples include distances from observation metadata, distances present in `state`, and at most one synthesized post-step distance from `distance_deltas["food"]` when no explicit post-step distance is present.
    """
    distances: list[float] = []
    for item in trace:
        distances.extend(_trace_item_food_distances(item))
    return list(dict.fromkeys(distances))


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
        if state.get("shelter_role") == "outside":
            return True, tick
        if pos is not None and shelter_cells:
            if pos not in shelter_cells:
                return True, tick
            continue
    return False, None


def _resolve_initial_food_distance(trace: Sequence[Dict[str, object]]) -> float | None:
    """
    Resolve the best available initial food distance from a trace using a fallback chain.

    Tries in order: explicit pre-step observation, reconstructed from first item deltas/positions,
    and first distance in the collected food distances. Returns None when nothing is available.
    """
    initial = _trace_initial_food_distance(trace)
    if initial is None and trace:
        initial = _trace_reconstructed_initial_food_distance(trace[0])
    if initial is None:
        food_distances = _trace_food_distances(trace)
        if food_distances:
            initial = food_distances[0]
    return initial


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

__all__ = [
    '_clamp01',
    '_environment_observation_food_distances',
    '_extract_exposed_day_trace_metrics',
    '_float_or_none',
    '_food_signal_strength',
    '_hunger_valence_rate',
    '_int_or_none',
    '_mapping_predator_visible',
    '_memory_vector_freshness',
    '_observation_food_distances',
    '_observation_meta_food_distance',
    '_payload_float',
    '_payload_text',
    '_predator_visible_flag',
    '_state_alive',
    '_state_food_distance',
    '_state_position',
    '_trace_action_selection_payloads',
    '_trace_any_mode',
    '_trace_any_sleep_phase',
    '_trace_cell',
    '_trace_corridor_metrics',
    '_trace_death_tick',
    '_trace_distance_deltas',
    '_trace_dominant_predator_types',
    '_trace_escape_seen',
    '_trace_food_distances',
    '_trace_food_positions',
    '_trace_food_signal_strengths',
    '_trace_initial_food_distance',
    '_trace_max_predator_threat',
    '_trace_meta_mappings',
    '_trace_peak_food_progress',
    '_trace_post_observation_food_distances',
    '_trace_pre_observation_food_distances',
    '_trace_pre_tick_snapshot_payload',
    '_trace_predator_memory_seen',
    '_trace_predator_visible',
    '_trace_previous_position',
    '_trace_reconstructed_initial_food_distance',
    '_trace_shelter_cells',
    '_trace_shelter_exit',
    '_trace_shelter_template_key',
    '_trace_states',
    '_trace_tick',
]
