"""World state, geometry, and query helpers for the spider simulation.

`SpiderWorld` owns the durable simulation state, map geometry, and helper
queries used throughout the environment. The per-tick transformation logic
itself lives in `spider_cortex_sim.stages`, where the pipeline stages mutate
the world and tick context in a fixed order.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import stages as tick_stages
from .interfaces import ACTION_DELTAS, ACTION_TO_INDEX, LOCOMOTION_ACTIONS, OBSERVATION_VIEW_BY_KEY, ORIENT_HEADINGS
from .maps import BLOCKED, CLUTTER, MAP_TEMPLATE_NAMES, NARROW, build_map_template, terrain_at
from .memory import MEMORY_TTLS, empty_memory_slot, refresh_all_memory
from .noise import NoiseConfig, resolve_noise_profile
from .perception import (
    empty_percept_trace,
    has_line_of_sight,
    observe_world,
    predator_detects_spider,
    predator_motion_salience,
    predator_visible_to_spider,
    smell_gradient,
    trace_view,
    visible_range,
)
from .physiology import (
    apply_predator_contact,
    reset_sleep_state,
    rest_streak_norm,
    sleep_phase_level,
)
from .predator import (
    DEFAULT_LIZARD_PROFILE,
    LizardState,
    PredatorController,
    PredatorProfile,
)
from .operational_profiles import OperationalProfile, runtime_operational_profile
from .reward import REWARD_COMPONENT_NAMES, REWARD_PROFILES
from .world_types import SpiderState, TickContext, TickSnapshot

"""World state, geometry, and query helpers for the spider simulation.

`SpiderWorld` owns the durable simulation state, map geometry, and helper
queries used throughout the environment. The per-tick transformation logic
itself lives in `spider_cortex_sim.stages`, where the pipeline stages mutate
the world and tick context in a fixed order.
"""

ACTIONS: Sequence[str] = tuple(LOCOMOTION_ACTIONS)

MOVE_DELTAS = tuple(delta for delta in ACTION_DELTAS.values() if delta != (0, 0))

MOMENTUM_DECAY_ON_STOP = 0.8

MOMENTUM_BOOST_ON_SAME_DIR = 0.15

MOMENTUM_FRICTION_ON_TURN = 0.5

SHELTER_ROLE_LEVELS = {
    "outside": 0.0,
    "entrance": 1.0 / 3.0,
    "inside": 2.0 / 3.0,
    "deep": 1.0,
}

SCAN_AGE_NEVER = 100

SCAN_TICK_FIELDS = {
    (0, -1): "last_scan_tick_up",
    (0, 1): "last_scan_tick_down",
    (-1, 0): "last_scan_tick_left",
    (1, 0): "last_scan_tick_right",
    (-1, -1): "last_scan_tick_up_left",
    (1, -1): "last_scan_tick_up_right",
    (-1, 1): "last_scan_tick_down_left",
    (1, 1): "last_scan_tick_down_right",
}

def _scan_tick_field_for_heading(heading_dx: int, heading_dy: int) -> str | None:
    """Return the SpiderState scan field for a tracked heading."""
    return SCAN_TICK_FIELDS.get((int(heading_dx), int(heading_dy)))

def _scan_age_never_for_world(world: "SpiderWorld") -> int:
    """Return a never-scanned age that is stale for this world's profile."""
    max_scan_age = float(world.operational_profile.perception.get("max_scan_age", SCAN_AGE_NEVER))
    if not np.isfinite(max_scan_age):
        return SCAN_AGE_NEVER
    return max(SCAN_AGE_NEVER, int(np.ceil(max_scan_age)))

def _scan_age_for_heading(world: "SpiderWorld", heading_dx: int, heading_dy: int) -> int:
    """Return ticks since a tracked heading was actively scanned."""
    never_scanned_age = _scan_age_never_for_world(world)
    field_name = _scan_tick_field_for_heading(heading_dx, heading_dy)
    if field_name is None:
        return never_scanned_age
    last_scan_tick = int(getattr(world.state, field_name))
    if last_scan_tick < 0:
        return never_scanned_age
    return max(0, int(world.tick) - last_scan_tick)

def _refresh_perception_for_active_scan(world: "SpiderWorld") -> dict[str, object]:
    """
    Refresh the current tick's raw perception after an active heading scan.

    This updates only the perceptual buffer entry for ``world.tick``. It does
    not advance the tick counter or run memory, reward, predator, or physiology
    stages.
    """
    delay_ticks = world._perceptual_delay_ticks()
    raw_observation = world._raw_observation()
    world._ensure_perceptual_buffer(delay_ticks)
    world._perceptual_buffer.push(int(world.tick), raw_observation)
    return raw_observation

def _copy_observation_payload(observation: dict[str, object]) -> dict[str, object]:
    """
    Create a deep copy of an observation payload, copying NumPy arrays as arrays and deep-copying all other values.
    
    Parameters:
        observation (dict[str, object]): Observation mapping of field names to values (may include numpy.ndarray).
    
    Returns:
        dict[str, object]: A new observation dict where each numpy.ndarray value is a shallow copy of the array and all other values are deep-copied.
    """
    copied: dict[str, object] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = deepcopy(value)
    return copied

class PerceptualBuffer:
    """Ring buffer that stores raw observations by simulation tick."""

    def __init__(self, max_delay: int) -> None:
        """
        Initialize the perceptual buffer with a configured maximum delay.
        
        Parameters:
            max_delay (int): Maximum perceptual delay in ticks; negative values are treated as 0, and the buffer capacity will be `max_delay + 1`.
        """
        self.max_delay = max(0, int(max_delay))
        self._entries: list[tuple[int, dict[str, object]]] = []

    @property
    def capacity(self) -> int:
        """
        Number of entries the perceptual buffer can hold.
        
        Returns:
            capacity (int): The maximum number of stored payloads, equal to max_delay + 1.
        """
        return self.max_delay + 1

    def clear(self) -> None:
        """
        Clear all stored perceptual entries from the buffer.
        
        After calling this, the buffer is empty and subsequent retrievals will behave as if no observations have been pushed.
        """
        self._entries.clear()

    def push(self, tick: int, observation: dict[str, object]) -> None:
        """
        Store a timestamped observation payload in the ring buffer, replacing colliding ticks and pruning old entries to maintain capacity.
        
        The provided `observation` is copied before storage to prevent external mutation. If an entry for the same `tick` already exists as the newest entry, it is replaced; otherwise the entry is appended. If the buffer exceeds its capacity after insertion, the oldest entries are removed so the buffer length equals `capacity`.
        
        Parameters:
            tick (int): Simulation tick associated with the observation.
            observation (dict[str, object]): Observation payload to store; will be deep-copied internally.
        """
        entry = (int(tick), _copy_observation_payload(observation))
        if self._entries and self._entries[-1][0] == int(tick):
            self._entries[-1] = entry
        else:
            self._entries.append(entry)
        if len(self._entries) > self.capacity:
            del self._entries[: len(self._entries) - self.capacity]

    def get(self, delay_ticks: int) -> tuple[dict[str, object], int]:
        """
        Retrieve a delayed observation payload from the perceptual buffer.
        
        Parameters:
            delay_ticks (int): Desired delay in ticks (clipped to zero if negative).
        
        Returns:
            payload (dict[str, object]): A deep-copied observation payload corresponding to the requested delay or the oldest available entry if the buffer is shorter than requested.
            effective_delay (int): The actual delay (in ticks) applied, which may be smaller than `delay_ticks` when the buffer does not contain that many past entries.
        
        Raises:
            ValueError: If the perceptual buffer is empty.
        """
        if not self._entries:
            raise ValueError("PerceptualBuffer is empty.")
        requested = max(0, int(delay_ticks))
        index = max(0, len(self._entries) - 1 - requested)
        effective_delay = len(self._entries) - 1 - index
        return _copy_observation_payload(self._entries[index][1]), effective_delay

def _is_temporal_direction_field(field_name: str) -> bool:
    """
    Determine whether an observation field name represents a temporal direction component.
    
    A field is considered temporal direction if it ends with `_dx` or `_dy` but is not one of `heading_dx`, `heading_dy`, `last_move_dx`, `last_move_dy`, does not contain `_memory_`, and is not trace-heading context.
    
    Returns:
        `true` if the name denotes a temporal direction component, `false` otherwise.
    """
    if not (field_name.endswith("_dx") or field_name.endswith("_dy")):
        return False
    if field_name in {"heading_dx", "heading_dy", "last_move_dx", "last_move_dy"}:
        return False
    if "_trace_heading_" in field_name:
        return False
    if "_memory_" in field_name:
        return False
    return True
