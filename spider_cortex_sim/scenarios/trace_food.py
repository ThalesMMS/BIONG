from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ..interfaces import ACTION_DELTAS
from ..maps import build_map_template
from .specs import TRACE_MAP_HEIGHT_FALLBACK, TRACE_MAP_WIDTH_FALLBACK

from .trace_core import _clamp01, _float_or_none, _int_or_none, _state_food_distance, _state_position

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
