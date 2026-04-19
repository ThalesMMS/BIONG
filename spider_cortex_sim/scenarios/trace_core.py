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

def _clamp01(value: float) -> float:
    """
    Clamp a numeric value to the inclusive range 0.0-1.0.

    Parameters:
        value (float): Input value to be clamped.

    Returns:
        float: The input coerced to a float and constrained to be between 0.0 and 1.0 inclusive.
    """
    return float(max(0.0, min(1.0, value)))

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
