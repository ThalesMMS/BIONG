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


class SpiderWorld:
    """Grid world with shelter geometry, perception-grounded memory, and predator FSMs.

    Explicit memory data is sourced from local perception, contact events, and
    movement history. The world pipeline maintains mechanical aging, TTL
    expiration, traces, and GUI/export views; it does not grant the brain direct
    access to hidden world state.
    """

    move_deltas = MOVE_DELTAS

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 36,
        night_length: int = 24,
        seed: int = 0,
        vision_range: int = 4,
        food_smell_range: int = 10,
        predator_smell_range: int = 7,
        lizard_vision_range: int = 3,
        lizard_move_interval: int = 2,
        reward_profile: str = "classic",
        map_template: str = "central_burrow",
        operational_profile: str | OperationalProfile | None = None,
        noise_profile: str | NoiseConfig | None = None,
    ) -> None:
        """
        Create and initialize a SpiderWorld simulation with map, agent state, predator, and RNG.

        Parameters:
            width (int): Grid width in cells.
            height (int): Grid height in cells.
            food_count (int): Number of food items maintained on the map.
            day_length (int): Number of ticks per daytime.
            night_length (int): Number of ticks per nighttime.
            seed (int): RNG seed for reproducible sampling.
            vision_range (int): Spider visual radius (Manhattan distance).
            food_smell_range (int): Radius used for the food smell gradient.
            predator_smell_range (int): Radius used for the predator smell gradient.
            lizard_vision_range (int): Base visual radius for the lizard predator.
            lizard_move_interval (int): Minimum ticks between lizard moves (clipped to at least 1).
            reward_profile (str): Key of the reward parameter set to use (must be in REWARD_PROFILES).
            map_template (str): Map template name to build the environment (must be in MAP_TEMPLATE_NAMES).
            operational_profile (str | OperationalProfile | None): Registered operational profile name,
                explicit `OperationalProfile` instance, or `None` to use the default profile. String values are
                resolved via `resolve_operational_profile`, and unrecognized names raise `ValueError`.
            noise_profile (str | NoiseConfig | None): Registered noise profile name, explicit `NoiseConfig`
                instance, or `None` to use the default noise profile. String values are resolved via
                `resolve_noise_profile`, and unrecognized names raise `ValueError`.
        
        Raises:
            ValueError: If `reward_profile`, `map_template`, or a string `operational_profile` name is not recognized.
        """
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f"Invalid reward profile: {reward_profile}")
        if map_template not in MAP_TEMPLATE_NAMES:
            raise ValueError(f"Invalid map template: {map_template}")

        self.width = width
        self.height = height
        self.food_count = food_count
        self.day_length = day_length
        self.night_length = night_length
        self.cycle_length = day_length + night_length
        self.seed = seed
        self.vision_range = int(vision_range)
        self.food_smell_range = int(food_smell_range)
        self.predator_smell_range = int(predator_smell_range)
        self.lizard_vision_range = int(lizard_vision_range)
        self.lizard_move_interval = max(1, int(lizard_move_interval))
        self.reward_profile = reward_profile
        self.reward_config = REWARD_PROFILES[reward_profile]
        self.operational_profile = runtime_operational_profile(operational_profile)
        self.noise_profile = resolve_noise_profile(noise_profile)
        self.map_template_name = map_template
        self.map_template = build_map_template(map_template, width=width, height=height)
        self.shelter_entrance_cells = set()
        self.shelter_interior_cells = set()
        self.shelter_deep_cells = set()
        self.shelter_cells: set[Tuple[int, int]] = set()
        self.blocked_cells: set[Tuple[int, int]] = set()
        self.configure_map_template(map_template)
        self.episode_seed = int(seed)
        self._reset_rngs(self.episode_seed)
        self._perceptual_buffer = PerceptualBuffer(self._perceptual_delay_ticks())
        self.tick = 0
        self.food_positions: List[Tuple[int, int]] = []
        self.predators: List[LizardState] = []
        self.predator_controllers: List[PredatorController] = []
        self._predator_profiles: List[PredatorProfile] = [DEFAULT_LIZARD_PROFILE]
        self.state = self._initial_spider_state(*self.map_template.spider_start)
        self.predators = self._spawn_predators(list(self._predator_profiles))
        self._sync_predator_controllers()
        self._last_on_shelter = False
        self._last_predator_visible = False
        self._predator_threat_episode_active = False
        self._predator_escape_bonus_pending = False

    def _reset_rngs(self, resolved_seed: int) -> None:
        """
        Reinitialize per-episode random number generators and record the episode seed.
        
        Sets `self.episode_seed` from `resolved_seed` and derives six independent NumPy RNGs (spawn, predator, visual, olfactory, motor, delay) from a single SeedSequence seeded by the instance seed and the episode seed. Also sets `self.rng` as a backward-compatible alias for `self.predator_rng`.
        
        Parameters:
            resolved_seed (int): Episode-specific seed used to derive the RNG channels.
        
        """
        self.episode_seed = int(resolved_seed)
        seed_sequence = np.random.SeedSequence([int(self.seed), self.episode_seed])
        channel_sequences = seed_sequence.spawn(6)
        self.spawn_rng = np.random.default_rng(channel_sequences[0])
        self.predator_rng = np.random.default_rng(channel_sequences[1])
        self.visual_rng = np.random.default_rng(channel_sequences[2])
        self.olfactory_rng = np.random.default_rng(channel_sequences[3])
        self.motor_rng = np.random.default_rng(channel_sequences[4])
        self.delay_rng = np.random.default_rng(channel_sequences[5])
        # Backward-compatible alias for legacy world-level random events.
        self.rng = self.predator_rng

    def configure_map_template(self, name: str) -> None:
        """
        Set the world's map template by name and refresh derived shelter and blocked-cell sets.
        
        Validates that `name` is a known map template, rebuilds `self.map_template` for the current grid size, stores `self.map_template_name`, and updates `shelter_entrance_cells`, `shelter_interior_cells`, `shelter_deep_cells`, `shelter_cells`, and `blocked_cells` from the new template.
        
        Parameters:
            name (str): The key name of the map template to load; must be one of the recognized MAP_TEMPLATE_NAMES.
        
        Raises:
            ValueError: If `name` is not a recognized map template.
        """
        if name not in MAP_TEMPLATE_NAMES:
            raise ValueError(f"Invalid map template: {name}")
        self.map_template_name = name
        self.map_template = build_map_template(name, width=self.width, height=self.height)
        self.shelter_entrance_cells = set(self.map_template.shelter_entrance)
        self.shelter_interior_cells = set(self.map_template.shelter_interior)
        self.shelter_deep_cells = set(self.map_template.shelter_deep)
        self.shelter_cells = self.map_template.shelter_cells
        self.blocked_cells = set(self.map_template.blocked_cells)

    @staticmethod
    def _heading_components_from_delta(dx: int, dy: int) -> tuple[int, int]:
        """
        Convert an arbitrary delta into a compact signed heading vector.
        """
        return int(np.sign(dx)), int(np.sign(dy))

    def _record_scan_for_heading(self, heading_dx: int, heading_dy: int) -> None:
        """
        Mark a tracked heading as actively scanned on the current world tick.
        """
        field_name = _scan_tick_field_for_heading(heading_dx, heading_dy)
        if field_name is not None:
            setattr(self.state, field_name, int(self.tick))

    def _scan_age_for_heading(self, heading_dx: int, heading_dy: int) -> int:
        """
        Return ticks since the given tracked heading was actively scanned.
        """
        return _scan_age_for_heading(self, heading_dx, heading_dy)

    def _refresh_perception_for_active_scan(self) -> dict[str, object]:
        """
        Refresh the current tick's perceptual buffer after an ORIENT action.
        """
        return _refresh_perception_for_active_scan(self)

    def _heading_toward(
        self,
        target: Tuple[int, int] | None,
        *,
        origin: Tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """
        Compute a signed heading vector pointing from origin to target.
        """
        if target is None:
            return 0, 0
        ox, oy = origin if origin is not None else self.spider_pos()
        return self._heading_components_from_delta(target[0] - ox, target[1] - oy)

    def _reset_heading_after_teleport(self) -> None:
        """
        Reinitialize heading after externally repositioning the spider.
        """
        heading_dx, heading_dy = self._heading_toward(
            self.nearest_shelter_entrance(origin=self.spider_pos()),
            origin=self.spider_pos(),
        )
        self.state.heading_dx = heading_dx
        self.state.heading_dy = heading_dy
        self._record_scan_for_heading(heading_dx, heading_dy)

    def _initial_spider_state(self, start_x: int, start_y: int) -> SpiderState:
        """
        Create a SpiderState positioned at the given coordinates with randomized initial physiology and defaulted memories, traces, and bookkeeping fields.
        
        Physiology fields (hunger, fatigue, sleep_debt) are sampled from the spawn RNG; heading is initialized toward the nearest shelter entrance from the spawn location. Memory slots and percept traces are set to empty defaults and counters/event trackers are initialized to baseline values.
        
        Parameters:
            start_x (int): Starting x-coordinate for the spider.
            start_y (int): Starting y-coordinate for the spider.
        
        Returns:
            SpiderState: Initialized spider state with sampled physiology, heading, empty memory/trace slots, and default counters.
        """
        initial_heading_dx, initial_heading_dy = self._heading_toward(
            self.nearest_shelter_entrance(origin=(start_x, start_y)),
            origin=(start_x, start_y),
        )
        state = SpiderState(
            x=start_x,
            y=start_y,
            hunger=float(self.spawn_rng.uniform(0.40, 0.60)),
            fatigue=float(self.spawn_rng.uniform(0.08, 0.20)),
            sleep_debt=float(self.spawn_rng.uniform(0.12, 0.24)),
            health=1.0,
            recent_pain=0.0,
            recent_contact=0.0,
            sleep_phase="AWAKE",
            rest_streak=0,
            last_reward=0.0,
            total_reward=0.0,
            food_eaten=0,
            sleep_events=0,
            shelter_entries=0,
            alert_events=0,
            predator_contacts=0,
            predator_sightings=0,
            predator_escapes=0,
            steps_alive=0,
            last_action="STAY",
            last_move_dx=0,
            last_move_dy=0,
            heading_dx=initial_heading_dx,
            heading_dy=initial_heading_dy,
            food_memory=empty_memory_slot(),
            predator_memory=empty_memory_slot(),
            shelter_memory=empty_memory_slot(),
            escape_memory=empty_memory_slot(),
            food_trace=empty_percept_trace(),
            shelter_trace=empty_percept_trace(),
            predator_trace=empty_percept_trace(),
        )
        field_name = _scan_tick_field_for_heading(initial_heading_dx, initial_heading_dy)
        if field_name is not None:
            setattr(state, field_name, int(getattr(self, "tick", 0)))
        return state

    def reset(
        self,
        seed: int | None = None,
        predator_profiles: Sequence[PredatorProfile] | None = None,
    ) -> Dict[str, object]:
        """
        Reset the environment for a new episode and produce the initial observation.
        
        Reinitializes episode RNG channels and the perceptual buffer, resets the simulation tick and spider state, repopulates food, spawns predator entities according to the provided profiles (at least one profile is required), refreshes memory/traces, and returns the observation for tick 0.
        
        Parameters:
        	seed (int | None): Optional episode seed; if None the instance's configured seed is used.
        	predator_profiles (Sequence[PredatorProfile] | None): Sequence of predator profiles to spawn for the episode. If None, the previously selected roster is reused; new worlds start with a default single lizard profile. The sequence must contain at least one profile.
        
        Returns:
        	initial_observation (Dict[str, object]): Observation dictionary representing the world's state at tick 0.
        
        Raises:
        	ValueError: If `predator_profiles` is provided but is empty.
        """
        resolved_seed = int(seed) if seed is not None else int(self.seed)
        self._reset_rngs(resolved_seed)
        self._perceptual_buffer = PerceptualBuffer(self._perceptual_delay_ticks())
        self.tick = 0
        self.state = self._initial_spider_state(*self.map_template.spider_start)
        reset_sleep_state(self)
        self._last_on_shelter = True
        self._last_predator_visible = False
        self._predator_threat_episode_active = False
        self._predator_escape_bonus_pending = False
        self.food_positions = []
        for _ in range(self.food_count):
            self.food_positions.append(self._random_food_cell())
        if predator_profiles is None:
            profiles = list(self._predator_profiles)
        else:
            profiles = list(predator_profiles)
            if not profiles:
                raise ValueError("predator_profiles must contain at least one predator profile.")
            self._predator_profiles = list(profiles)
        if not profiles:
            raise ValueError("predator_profiles must contain at least one predator profile.")
        self.predators = self._spawn_predators(profiles)
        self._sync_predator_controllers()
        self.refresh_memory(initial=True)
        return self.observe()

    def _sync_predator_controllers(self) -> None:
        """
        Initialize the predator_controllers list so there is one PredatorController for each predator.
        
        Replaces any existing controllers with a new list of PredatorController instances whose predator_index values match the current indices of self.predators.
        """
        self.predator_controllers = [
            PredatorController(predator_index=index)
            for index in range(len(self.predators))
        ]

    @property
    def predator_count(self) -> int:
        """
        Number of predator entities currently present in the world.
        
        Returns:
            int: The count of predator entities managed by this SpiderWorld.
        """
        return len(self.predators)

    def get_predator(self, index: int) -> LizardState:
        """
        Retrieve the predator state for the given zero-based index.
        
        Parameters:
            index (int): Zero-based index of the predator to retrieve.
        
        Returns:
            LizardState: The predator state at the specified index.
        """
        return self.predators[int(index)]

    def predator_positions(self) -> List[Tuple[int, int]]:
        """
        Get integer grid positions of all predators in the world.
        
        Returns:
            list[tuple[int, int]]: A list of (x, y) tuples representing each predator's grid cell coordinates as integers; returns an empty list if there are no predators.
        """
        return [
            (int(predator.x), int(predator.y))
            for predator in getattr(self, "predators", [])
        ]

    @property
    def lizard(self) -> LizardState:
        """
        Access the first predator's state (legacy "lizard" accessor).
        
        Returns:
            LizardState: The predator state at index 0.
        """
        return self.get_predator(0)

    @lizard.setter
    def lizard(self, value: LizardState) -> None:
        """
        Set the primary predator (predator at index 0) to the provided LizardState.
        
        If the world has existing predators, replace the predator at index 0 with `value`.
        If no predators exist, create a new predator list containing `value`. After assignment,
        synchronize predator controllers to reflect the updated predator list.
        
        Parameters:
            value (LizardState): The predator state to assign as the primary predator.
        """
        if self.predators:
            self.predators[0] = value
        else:
            self.predators = [value]
        self._sync_predator_controllers()

    @property
    def predator_controller(self) -> PredatorController:
        """
        Access the controller for the primary predator (predator index 0).
        
        Returns:
            PredatorController: The controller instance assigned to predator index 0.
        """
        return self.predator_controllers[0]

    @predator_controller.setter
    def predator_controller(self, value: PredatorController) -> None:
        """
        Set the primary predator controller and ensure it is assigned to predator index 0.
        
        Parameters:
            value (PredatorController): Controller to assign as the primary predator controller (stored at index 0). The controller's `predator_index` will be set to 0.
        """
        if self.predator_controllers:
            self.predator_controllers[0] = value
        else:
            self.predator_controllers = [value]
        self.predator_controllers[0].predator_index = 0

    def spider_pos(self) -> Tuple[int, int]:
        """
        Get the spider's current grid coordinates.
        
        Returns:
            pos (Tuple[int, int]): The spider's `(x, y)` position on the grid.
        """
        return self.state.x, self.state.y

    def lizard_pos(self) -> Tuple[int, int]:
        return self.lizard.x, self.lizard.y

    def terrain_at(self, pos: Tuple[int, int]) -> str:
        return terrain_at(self.map_template, pos)

    def is_walkable(self, pos: Tuple[int, int]) -> bool:
        return self.terrain_at(pos) != BLOCKED

    def is_lizard_walkable(self, pos: Tuple[int, int]) -> bool:
        return self.is_walkable(pos) and pos not in self.shelter_cells

    def shelter_role_at(self, pos: Tuple[int, int]) -> str:
        if pos in self.shelter_deep_cells:
            return "deep"
        if pos in self.shelter_interior_cells:
            return "inside"
        if pos in self.shelter_entrance_cells:
            return "entrance"
        return "outside"

    def shelter_role_level(self, pos: Tuple[int, int] | None = None) -> float:
        role = self.shelter_role_at(pos if pos is not None else self.spider_pos())
        return float(SHELTER_ROLE_LEVELS[role])

    def on_shelter(self) -> bool:
        return self.spider_pos() in self.shelter_cells

    def on_shelter_entrance(self) -> bool:
        return self.spider_pos() in self.shelter_entrance_cells

    def inside_shelter(self) -> bool:
        return self.spider_pos() in (self.shelter_interior_cells | self.shelter_deep_cells)

    def deep_shelter(self) -> bool:
        return self.spider_pos() in self.shelter_deep_cells

    def on_food(self) -> bool:
        return self.spider_pos() in self.food_positions

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def nearest(
        self,
        positions: Iterable[Tuple[int, int]],
        *,
        origin: Tuple[int, int] | None = None,
    ) -> tuple[Tuple[int, int], int]:
        ox, oy = origin if origin is not None else self.spider_pos()
        best = None
        best_dist = 10**9
        for x, y in positions:
            dist = abs(x - ox) + abs(y - oy)
            if dist < best_dist:
                best = (x, y)
                best_dist = dist
        if best is None:
            return (ox, oy), 0
        return best, int(best_dist)

    def nearest_shelter_entrance(self, *, origin: Tuple[int, int] | None = None) -> Tuple[int, int] | None:
        if not self.shelter_entrance_cells:
            return None
        return self.nearest(self.shelter_entrance_cells, origin=origin)[0]

    def safest_shelter_target(self) -> Tuple[int, int]:
        if self.shelter_deep_cells:
            return self.nearest(self.shelter_deep_cells)[0]
        if self.shelter_interior_cells:
            return self.nearest(self.shelter_interior_cells)[0]
        if self.shelter_entrance_cells:
            return self.nearest(self.shelter_entrance_cells)[0]
        return self.spider_pos()

    def _relative(
        self,
        target: Tuple[int, int],
        *,
        origin: Tuple[int, int] | None = None,
    ) -> tuple[float, float, float]:
        ox, oy = origin if origin is not None else self.spider_pos()
        max_dx = max(1, self.width - 1)
        max_dy = max(1, self.height - 1)
        dx = (target[0] - ox) / max_dx
        dy = (target[1] - oy) / max_dy
        dist = self.manhattan(target, (ox, oy)) / float(self.width + self.height)
        return float(dx), float(dy), float(dist)

    def is_night(self, tick: int | None = None) -> bool:
        if tick is None:
            tick = self.tick
        phase = tick % self.cycle_length
        return phase >= self.day_length

    def light_level(self) -> float:
        return 0.15 if self.is_night() else 1.0

    def phase_features(self) -> tuple[float, float]:
        """
        Compute sine and cosine features for the current day/night cycle phase.
        
        Returns:
            (sin_phase, cos_phase) (tuple[float, float]): Sine and cosine of 2π times the normalized phase (tick % cycle_length / cycle_length).
        """
        phase = (self.tick % self.cycle_length) / float(self.cycle_length)
        angle = 2.0 * np.pi * phase
        return float(np.sin(angle)), float(np.cos(angle))

    def sleep_phase_level(self) -> float:
        """
        Compute the spider's current sleep-phase level.
        
        Returns:
            float: Normalized sleep-phase level between 0.0 and 1.0, where larger values indicate a deeper sleep phase.
        """
        return sleep_phase_level(self)

    def rest_streak_norm(self) -> float:
        """
        Return the spider's current rest-streak level normalized to the range [0, 1].
        
        Returns:
            float: Normalized rest-streak value (0.0 means no recent rest, 1.0 means fully rested).
        """
        return rest_streak_norm(self)

    def _line_cells(self, origin: Tuple[int, int], target: Tuple[int, int]) -> list[Tuple[int, int]]:
        """
        Return the ordered list of grid cells that form a straight line from origin to target, including both endpoints.
        
        Parameters:
            origin (Tuple[int, int]): (x, y) coordinates of the line start cell.
            target (Tuple[int, int]): (x, y) coordinates of the line end cell.
        
        Returns:
            List[Tuple[int, int]]: Ordered list of (x, y) integer cell coordinates along the line from origin to target.
        """
        from .perception import line_cells

        return line_cells(origin, target)

    def lizard_detects_spider(self) -> bool:
        """
        Indicates whether the primary predator (lizard) currently detects the spider.
        
        Returns:
            True if the lizard detects the spider, False otherwise.
        """
        return predator_detects_spider(self, self.lizard)

    def refresh_memory(self, *, predator_escape: bool = False, initial: bool = False) -> None:
        """
        Reset/setup convenience for perception-grounded memory maintenance.
        
        This ages and clears explicit memory slots by TTL, writes only targets
        derived from allowed perception/contact/movement sources, and advances
        short percept traces through `spider_cortex_sim.memory`. The tick
        pipeline calls the memory subsystem directly; this method remains for
        reset and scenario setup that also need to clear the perceptual buffer
        baseline.
        
        Parameters:
            predator_escape (bool): If True, treat this refresh as occurring after a predator escape and update escape-related memory accordingly.
            initial (bool): If True, perform an initial/baseline refresh (used on reset) that avoids normal decay effects and clears the perceptual buffer.
        """
        refresh_all_memory(self, predator_escape=predator_escape, initial=initial)
        if initial and hasattr(self, "_perceptual_buffer"):
            self._perceptual_buffer.clear()

    def _perceptual_delay_ticks(self) -> int:
        """
        Compute the configured perceptual delay as an integer tick count.
        
        Reads the `perceptual_delay_ticks` value from `self.operational_profile.perception`, rounds it to the nearest integer, and clamps the result to be at least 0.
        
        Returns:
            int: The perceptual delay in ticks (>= 0).
        """
        value = self.operational_profile.perception.get("perceptual_delay_ticks", 1.0)
        return max(0, round(float(value)))

    def _perceptual_delay_noise_scale(self) -> float:
        """
        Get the configured perceptual delay noise scale, coerced to a float and clipped to a minimum of 0.0.
        
        Returns:
            float: Perceptual delay noise scale (>= 0.0).
        """
        value = self.operational_profile.perception.get("perceptual_delay_noise", 0.5)
        return max(0.0, float(value))

    def _ensure_perceptual_buffer(self, delay_ticks: int) -> None:
        """
        Ensure a PerceptualBuffer exists and matches the requested delay capacity.
        
        If no buffer exists, creates one sized for `delay_ticks`. If an existing
        buffer has a different `max_delay`, it is replaced with a new buffer
        sized to `delay_ticks`. The provided `delay_ticks` is converted to an int.
        """
        if not hasattr(self, "_perceptual_buffer"):
            self._perceptual_buffer = PerceptualBuffer(delay_ticks)
            return
        if self._perceptual_buffer.max_delay != int(delay_ticks):
            self._perceptual_buffer = PerceptualBuffer(delay_ticks)

    def _raw_observation(self) -> dict[str, object]:
        """
        Get the raw instantaneous observation of the world state without perceptual delay or temporal noise.
        
        The returned payload contains sensor readings, grid-state arrays, and a `meta` diagnostics section representing the current world snapshot before any buffering, delay, or temporal perturbation is applied.
        
        Returns:
            dict[str, object]: Mapping of observation field names to values (arrays or scalars), including a `meta` entry with diagnostic information.
        """
        return observe_world(self)

    def _jitter_delayed_direction(self, value: object, amplitude: float) -> float:
        """
        Apply uniform temporal jitter to a single directional component and clip the result to [-1, 1].
        
        If `amplitude` is greater than 0, a uniform offset in [-amplitude, amplitude] is sampled (using the instance's delay RNG) and added to `value`; otherwise `value` is returned clipped. The final result is always constrained to the range [-1.0, 1.0].
        
        Parameters:
            value (object): Numeric input representing a directional component (typically in [-1, 1]).
            amplitude (float): Maximum magnitude of the uniform jitter to apply. If <= 0, no jitter is applied.
        
        Returns:
            float: The jittered directional component, clipped to [-1.0, 1.0].
        """
        base = float(value)
        if amplitude <= 0.0:
            return float(np.clip(base, -1.0, 1.0))
        jitter = float(self.delay_rng.uniform(-amplitude, amplitude))
        return float(np.clip(base + jitter, -1.0, 1.0))

    def _apply_temporal_noise_to_vector(
        self,
        key: str,
        vector: np.ndarray,
        *,
        decay_factor: float,
        direction_jitter: float,
    ) -> np.ndarray:
        """
        Apply temporal decay and directional jitter to an observation vector according to its view definition.
        
        Given a view identified by `key` (or `motor_extra` mapped to `motor_context`), this multiplies any fields whose names end with `_certainty` by `decay_factor` (clipped to the range [0.0, 1.0]) and applies `direction_jitter` via `_jitter_delayed_direction` to fields classified as temporal direction components (e.g., suffix `_dx`/`_dy`). If `key` does not correspond to a known observation view, the input `vector` is returned unchanged.
        
        Parameters:
            key (str): Observation view key used to look up field names.
            vector (np.ndarray): 1-D observation vector whose elements correspond to the view's fields.
            decay_factor (float): Multiplicative factor applied to `_certainty` fields (values clipped to [0.0, 1.0]).
            direction_jitter (float): Jitter amplitude passed to `_jitter_delayed_direction` for temporal direction fields.
        
        Returns:
            np.ndarray: A (possibly) modified copy of `vector` with decayed certainties and jittered temporal directions.
        """
        view_key = "motor_context" if key == "motor_extra" else key
        view_cls = OBSERVATION_VIEW_BY_KEY.get(view_key)
        if view_cls is None:
            return vector
        delayed = vector.copy()
        field_names = view_cls.field_names()
        field_index = {name: idx for idx, name in enumerate(field_names)}
        visibility_threshold = float(
            self.operational_profile.perception["visibility_binary_threshold"]
        )
        for idx, field_name in enumerate(field_names):
            if idx >= delayed.shape[0]:
                continue
            if field_name.endswith("_certainty"):
                delayed[idx] = float(np.clip(float(delayed[idx]) * decay_factor, 0.0, 1.0))
        for idx, field_name in enumerate(field_names):
            if idx >= delayed.shape[0] or not field_name.endswith("_certainty"):
                continue
            entity = field_name[: -len("_certainty")]
            visible_idx = field_index.get(f"{entity}_visible")
            if visible_idx is None or visible_idx >= delayed.shape[0]:
                continue
            was_visible = float(delayed[visible_idx]) > 0.0
            is_visible = was_visible and float(delayed[idx]) >= visibility_threshold
            delayed[visible_idx] = 1.0 if is_visible else 0.0
            if is_visible:
                continue
            for direction_name in (f"{entity}_dx", f"{entity}_dy"):
                direction_idx = field_index.get(direction_name)
                if (
                    direction_idx is not None
                    and direction_idx < delayed.shape[0]
                    and _is_temporal_direction_field(direction_name)
                ):
                    delayed[direction_idx] = 0.0
        for idx, field_name in enumerate(field_names):
            if idx >= delayed.shape[0] or not _is_temporal_direction_field(field_name):
                continue
            entity = field_name.rsplit("_", 1)[0]
            sibling_idx = field_index.get(f"{entity}_visible")
            if sibling_idx is None:
                sibling_idx = field_index.get(f"{entity}_strength")
            sibling_active = (
                sibling_idx is not None
                and sibling_idx < delayed.shape[0]
                and float(delayed[sibling_idx]) > 0.0
            )
            if not sibling_active:
                delayed[idx] = 0.0
                continue
            delayed[idx] = self._jitter_delayed_direction(delayed[idx], direction_jitter)
        return delayed

    def _apply_temporal_noise_to_meta(
        self,
        meta: dict[str, object],
        *,
        configured_delay: int,
        effective_delay: int,
        decay_factor: float,
        direction_jitter: float,
    ) -> None:
        """
        Apply temporal decay and directional jitter to the observation meta sections and record delay diagnostics.
        
        This mutates the provided `meta` dictionary in-place by:
        - scaling any `certainty` entries inside `meta["vision"]` and `meta["percept_traces"]` by `decay_factor` (clipped to the range [0.0, 1.0]),
        - applying directional jitter to any `dx`/`dy` fields within those sections using `direction_jitter` and the instance's jitter routine,
        - and writing a `meta["perceptual_delay"]` dictionary containing `configured_ticks`, `effective_ticks`, `certainty_decay_factor`, and `direction_jitter`.
        
        Parameters:
            meta (dict[str, object]): Observation meta dictionary to modify.
            configured_delay (int): Configured perceptual delay in ticks.
            effective_delay (int): Actual delay applied (may be less than configured due to buffer length).
            decay_factor (float): Multiplicative factor applied to certainty values (expected in [0.0, 1.0]).
            direction_jitter (float): Amplitude used to jitter temporal direction components (`dx`/`dy`).
        
        """
        visibility_threshold = float(
            self.operational_profile.perception["visibility_binary_threshold"]
        )
        for section_name in ("vision", "percept_traces"):
            section = meta.get(section_name)
            if not isinstance(section, dict):
                continue
            for target_name, target in section.items():
                if not isinstance(target, dict):
                    continue
                visible_key = None
                if "certainty" in target:
                    target["certainty"] = float(
                        np.clip(float(target["certainty"]) * decay_factor, 0.0, 1.0)
                    )
                    if "visible" in target:
                        visible_key = "visible"
                    elif f"{target_name}_visible" in target:
                        visible_key = f"{target_name}_visible"
                    if visible_key is not None:
                        was_visible = float(target[visible_key]) > 0.0
                        is_visible = was_visible and float(target["certainty"]) >= visibility_threshold
                        target[visible_key] = 1.0 if is_visible else 0.0
                        if not is_visible:
                            for axis in ("dx", "dy"):
                                if axis in target:
                                    target[axis] = 0.0
                for axis in ("dx", "dy"):
                    if axis in target:
                        if visible_key is not None:
                            sibling_active = float(target[visible_key]) > 0.0
                        elif "strength" in target:
                            sibling_active = float(target["strength"]) > 0.0
                        elif f"{target_name}_strength" in target:
                            sibling_active = float(target[f"{target_name}_strength"]) > 0.0
                        else:
                            sibling_active = False
                        if not sibling_active:
                            target[axis] = 0.0
                            continue
                        target[axis] = self._jitter_delayed_direction(
                            target[axis],
                            direction_jitter,
                        )
        vision = meta.get("vision")
        if isinstance(vision, dict):
            predator = vision.get("predator")
            if isinstance(predator, dict):
                meta["predator_visible"] = bool(float(predator.get("visible", 0.0)) > 0.5)

        meta["perceptual_delay"] = {
            "configured_ticks": int(configured_delay),
            "effective_ticks": int(effective_delay),
            "certainty_decay_factor": float(decay_factor),
            "direction_jitter": float(direction_jitter),
        }

    def _apply_perceptual_delay_noise(
        self,
        observation: dict[str, object],
        *,
        configured_delay: int,
        effective_delay: int,
    ) -> dict[str, object]:
        """
        Apply configured temporal noise to a delayed observation payload, updating temporal certainty and direction fields.
        
        If `effective_delay` is 0 or less, the observation is returned unchanged except that `meta["perceptual_delay"]` is set to reflect the configured delay and zero effective delay. Otherwise, certainty values are multiplied by a decay factor (clipped to [0.0, 1.0]) and temporal direction components receive bounded jitter; these mutations are applied to any numpy-array fields in the observation and to the structures inside `meta`.
        
        Parameters:
            observation (dict[str, object]): Observation payload to mutate; expected to contain numpy array fields and an optional `meta` dict.
            configured_delay (int): The configured perceptual delay in ticks from the operational profile.
            effective_delay (int): The delay in ticks actually applied (may be smaller than configured due to buffer length).
        
        Returns:
            dict[str, object]: The (possibly mutated) observation payload. When present, `meta["perceptual_delay"]` is updated with keys `configured_ticks`, `effective_ticks`, `certainty_decay_factor`, and `direction_jitter`.
        """
        if effective_delay <= 0:
            meta = observation.get("meta")
            if isinstance(meta, dict):
                meta["perceptual_delay"] = {
                    "configured_ticks": int(configured_delay),
                    "effective_ticks": 0,
                    "certainty_decay_factor": 1.0,
                    "direction_jitter": 0.0,
                }
            return observation

        noise_scale = self._perceptual_delay_noise_scale()
        delay_cfg = self.noise_profile.delay
        decay_per_tick = max(0.0, float(delay_cfg["certainty_decay_per_tick"])) * noise_scale
        direction_jitter = (
            max(0.0, float(delay_cfg["direction_jitter_per_tick"]))
            * noise_scale
            * float(effective_delay)
        )
        decay_factor = float(np.clip(1.0 - decay_per_tick * float(effective_delay), 0.0, 1.0))

        for key, value in list(observation.items()):
            if isinstance(value, np.ndarray):
                observation[key] = self._apply_temporal_noise_to_vector(
                    key,
                    value,
                    decay_factor=decay_factor,
                    direction_jitter=direction_jitter,
                )
        meta = observation.get("meta")
        if isinstance(meta, dict):
            self._apply_temporal_noise_to_meta(
                meta,
                configured_delay=configured_delay,
                effective_delay=effective_delay,
                decay_factor=decay_factor,
                direction_jitter=direction_jitter,
            )
        return observation

    def observe(self) -> Dict[str, object]:
        """
        Constructs the spider's current observation and, if configured, returns a delayed, temporally noised view.
        
        When a perceptual delay is configured, the function stores the immediate (raw) observation in the internal perceptual buffer and returns a payload reflecting the effective delayed observation with temporal decay/jitter applied and diagnostic metadata about the perceptual delay. When no delay is configured, the raw observation is returned unchanged.
        
        Returns:
            Dict[str, object]: Mapping of observation keys (e.g., visual maps, smell fields, scalar features, and diagnostics) to arrays/objects representing the agent's perception and related metadata. If delay is active, the payload includes noise-modified values and a `meta["perceptual_delay"]` entry describing configured and effective delay.
        """
        delay_ticks = self._perceptual_delay_ticks()
        raw_observation = self._raw_observation()
        self._ensure_perceptual_buffer(delay_ticks)
        self._perceptual_buffer.push(int(self.tick), raw_observation)
        if delay_ticks <= 0:
            return raw_observation
        delayed_observation, effective_delay = self._perceptual_buffer.get(delay_ticks)
        return self._apply_perceptual_delay_noise(
            delayed_observation,
            configured_delay=delay_ticks,
            effective_delay=effective_delay,
        )

    def visibility_overlay(self, *, origin: Tuple[int, int] | None = None) -> Dict[str, object]:
        """
        Return which grid cells within the current visible range are visible or occluded from a given origin.
        
        Parameters:
            origin (Tuple[int, int] | None): Source cell for visibility checks; when None uses the spider's current position.
        
        Returns:
            info (dict): Mapping with:
                - "origin": source cell used (Tuple[int, int])
                - "radius": visibility radius (int)
                - "visible": list of cells (List[Tuple[int, int]]) within radius that have line of sight
                - "occluded": list of cells (List[Tuple[int, int]]) within radius that do not have line of sight
        """
        source = origin if origin is not None else self.spider_pos()
        radius = visible_range(self)
        visible: List[Tuple[int, int]] = []
        occluded: List[Tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                dist = self.manhattan(source, pos)
                if pos == source or dist > radius:
                    continue
                if has_line_of_sight(self, source, pos):
                    visible.append(pos)
                else:
                    occluded.append(pos)
        return {
            "origin": source,
            "radius": radius,
            "visible": visible,
            "occluded": occluded,
        }

    def smell_field(self, kind: str) -> List[List[float]]:
        """
        Compute a 2D grid of smell strengths for the specified scent source.
        
        Parameters:
            kind (str): Scent type to compute; must be either "food" or "predator".
        
        Returns:
            List[List[float]]: A height-by-width nested list where each element is the smell strength
            at that grid cell (row-major: outer list indexed by y, inner by x).
        
        Raises:
            ValueError: If `kind` is not "food" or "predator".
        """
        if kind == "food":
            positions = list(self.food_positions)
            radius = self.food_smell_range
        elif kind == "predator":
            positions = self.predator_positions()
            radius = self.predator_smell_range
        else:
            raise ValueError(f"Unknown scent field: {kind}")
        field: List[List[float]] = []
        for y in range(self.height):
            row: List[float] = []
            for x in range(self.width):
                strength, _, _, _ = smell_gradient(
                    self,
                    positions,
                    radius=radius,
                    origin=(x, y),
                    apply_noise=False,
                )
                row.append(float(strength))
            field.append(row)
        return field

    def _random_spawn_cell(
        self,
        candidates: Sequence[Tuple[int, int]],
        *,
        min_spider_distance: int = 0,
        avoid_lizard: bool = True,
        excluded_positions: Sequence[Tuple[int, int]] | None = None,
        min_predator_distance: int = 0,
    ) -> Tuple[int, int]:
        """
        Choose a spawn cell from candidate positions, applying distance and occupancy constraints and falling back to walkable non-shelter cells.
        
        Filters out the spider's current cell, existing food, any explicitly provided excluded positions, and (if requested) all current predator positions. Enforces minimum Manhattan distance from the spider (`min_spider_distance`) and from any occupied/excluded positions (`min_predator_distance`). If no candidate remains, falls back to uniform sampling from all walkable, non-shelter cells that satisfy the same exclusion and spacing constraints. Raises ValueError if no valid cell can be found after fallback.
        
        Parameters:
            candidates (Sequence[Tuple[int, int]]): Candidate (x, y) cells to consider.
            min_spider_distance (int): Minimum required Manhattan distance from the spider. Defaults to 0.
            avoid_lizard (bool): If True, exclude all current predator positions from candidates. Defaults to True.
            excluded_positions (Sequence[Tuple[int, int]] | None): Additional positions to exclude. Defaults to None.
            min_predator_distance (int): Minimum required Manhattan distance from any excluded/occupied position. Defaults to 0.
        
        Returns:
            Tuple[int, int]: The chosen spawn cell coordinates (x, y).
        """
        spider_pos = self.spider_pos()
        occupied_positions = list(excluded_positions or [])
        if avoid_lizard:
            occupied_positions.extend(self.predator_positions())
        occupied_positions = list(dict.fromkeys(occupied_positions))
        filtered = [
            cell
            for cell in candidates
            if cell != spider_pos
            and cell not in self.food_positions
            and cell not in occupied_positions
            and self.manhattan(cell, spider_pos) >= min_spider_distance
            and all(
                self.manhattan(cell, other) >= min_predator_distance
                for other in occupied_positions
            )
        ]
        if not filtered:
            filtered = [
                (x, y)
                for x in range(self.width)
                for y in range(self.height)
                if self.is_walkable((x, y))
                and (x, y) not in self.shelter_cells
                and (x, y) != spider_pos
                and (x, y) not in occupied_positions
                and self.manhattan((x, y), spider_pos) >= min_spider_distance
                and all(
                    self.manhattan((x, y), other) >= min_predator_distance
                    for other in occupied_positions
                )
            ]
        if not filtered:
            raise ValueError("Could not find a spawn cell satisfying predator spacing constraints.")
        idx = int(self.spawn_rng.integers(0, len(filtered)))
        return filtered[idx]

    def _random_food_cell(self) -> Tuple[int, int]:
        """
        Choose a grid cell for spawning a new food item.
        
        Prefers unused candidate cells from the map template, weighting them by proximity to the nearest shelter entrance and mixing with a uniform distribution according to self.noise_profile.spawn["uniform_mix"]. If no template candidates are available (or none remain after excluding the spider and existing food), falls back to _random_spawn_cell with the map template's food spawn set.
        
        Returns:
            tuple[int, int]: Coordinates (x, y) of the selected food spawn cell.
        """
        candidates = [
            cell
            for cell in self.map_template.food_spawn_cells
            if cell not in self.food_positions and cell != self.spider_pos()
        ]
        if not candidates:
            return self._random_spawn_cell(self.map_template.food_spawn_cells)
        shelter_origin = self.nearest(self.shelter_entrance_cells or self.shelter_cells)[0]
        distances = np.array([self.manhattan(cell, shelter_origin) for cell in candidates], dtype=float)
        weights = np.exp(-distances / max(2.5, self.width / 3.0))
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            weights = np.full(len(candidates), 1.0 / float(len(candidates)), dtype=float)
        else:
            weights = weights / total_weight
        uniform_mix = float(np.clip(self.noise_profile.spawn["uniform_mix"], 0.0, 1.0))
        uniform = np.full(len(candidates), 1.0 / float(len(candidates)), dtype=float)
        if uniform_mix > 0.0:
            weights = (1.0 - uniform_mix) * weights + uniform_mix * uniform
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0 or not np.isfinite(total_weight):
            weights = uniform
        else:
            weights = weights / total_weight
        idx = int(self.spawn_rng.choice(len(candidates), p=weights))
        return candidates[idx]

    def _spawn_predators(self, profiles: List[PredatorProfile]) -> List[LizardState]:
        """
        Select spawn cells for each predator profile and return their initialized states.
        
        Each predator is placed in a distinct lizard spawn cell while maintaining a minimum
        Manhattan distance from the spider and previously placed predators; spawn order
        corresponds to the order of the provided profiles.
        
        Raises:
            ValueError: If `profiles` is empty.
        
        Returns:
            list[LizardState]: Predator states initialized at their spawn positions, in the same order as `profiles`.
        """
        if not profiles:
            raise ValueError("At least one predator profile is required.")
        min_dist = max(3, min(self.width, self.height) // 3)
        predators: List[LizardState] = []
        occupied_positions: List[Tuple[int, int]] = []
        for profile in profiles:
            pos = self._random_spawn_cell(
                self.map_template.lizard_spawn_cells,
                min_spider_distance=min_dist,
                avoid_lizard=False,
                excluded_positions=occupied_positions,
                min_predator_distance=min_dist,
            )
            predators.append(
                LizardState(
                    x=pos[0],
                    y=pos[1],
                    mode="PATROL",
                    profile=profile,
                )
            )
            occupied_positions.append(pos)
        return predators

    def respawn_food(self, eaten_position: Tuple[int, int]) -> None:
        """
        Replace a consumed food item with a newly spawned food cell.
        
        Removes the food at the given grid cell from the world's food positions (if present) and appends a new food location sampled by the world's spawn policy.
        
        Parameters:
            eaten_position (Tuple[int, int]): Grid coordinates (x, y) of the consumed food item.
        """
        self.food_positions = [p for p in self.food_positions if p != eaten_position]
        self.food_positions.append(self._random_food_cell())

    def _move(self, dx: int, dy: int) -> bool:
        """
        Minimal durable position mutation interface used by the action stage.
        """
        target = (
            int(np.clip(self.state.x + dx, 0, self.width - 1)),
            int(np.clip(self.state.y + dy, 0, self.height - 1)),
        )
        if not self.is_walkable(target):
            self.state.last_move_dx = 0
            self.state.last_move_dy = 0
            return False
        moved = target != (self.state.x, self.state.y)
        self.state.x, self.state.y = target
        self.state.last_move_dx = dx if moved else 0
        self.state.last_move_dy = dy if moved else 0
        if moved and (dx != 0 or dy != 0):
            self.state.heading_dx, self.state.heading_dy = self._heading_components_from_delta(dx, dy)
            self._record_scan_for_heading(self.state.heading_dx, self.state.heading_dy)
        return moved

    def _move_spider_action(self, action_name: str) -> bool:
        """
        Set the spider's heading or execute a locomotion action.

        This is the stage-facing movement mutation boundary: motor execution
        decisions live outside `SpiderWorld`, while this method applies the
        selected action to durable position, heading, and last-move state.
        
        For orientation actions (members of ORIENT_HEADINGS) this updates the spider's heading and clears the last-move deltas without changing position. For locomotion actions this attempts to move the spider by the corresponding ACTION_DELTAS.
        
        Parameters:
            action_name (str): Action identifier indicating a locomotion action, "STAY", or an orientation change.
        
        Returns:
            bool: True if the spider moved to a different cell, False otherwise.
        """
        if action_name in ORIENT_HEADINGS:
            heading_dx, heading_dy = ORIENT_HEADINGS[action_name]
            self.state.heading_dx = int(heading_dx)
            self.state.heading_dy = int(heading_dy)
            self._record_scan_for_heading(self.state.heading_dx, self.state.heading_dy)
            self._refresh_perception_for_active_scan()
            self.state.last_move_dx = 0
            self.state.last_move_dy = 0
            return False
        dx, dy = ACTION_DELTAS[action_name]
        return self._move(dx, dy)

    def _update_momentum(
        self,
        action_name: str,
        *,
        previous_heading: tuple[int, int],
        moved: bool,
    ) -> None:
        """
        Update bounded execution momentum after the resolved action is applied.

        Momentum builds only through successful aligned movement. Failed
        movement attempts decay like interrupted locomotion instead of creating
        movement continuity.
        """
        momentum = float(np.clip(self.state.momentum, 0.0, 1.0))
        if action_name in ORIENT_HEADINGS:
            self.state.momentum = 0.0
            return
        if action_name == "STAY" or not moved:
            self.state.momentum = float(
                np.clip(momentum * MOMENTUM_DECAY_ON_STOP, 0.0, 1.0)
            )
            return

        move_dx, move_dy = ACTION_DELTAS[action_name]
        heading_dx, heading_dy = previous_heading
        if (heading_dx, heading_dy) == (move_dx, move_dy):
            momentum += MOMENTUM_BOOST_ON_SAME_DIR
        elif (heading_dx, heading_dy) == (-move_dx, -move_dy):
            momentum = 0.0
        else:
            momentum *= MOMENTUM_FRICTION_ON_TURN
        self.state.momentum = float(np.clip(momentum, 0.0, 1.0))

    def _apply_predator_contact(
        self,
        reward_components: Dict[str, float],
        info: Dict[str, object],
        *,
        tick_context: TickContext | None = None,
    ) -> None:
        """
        Apply the effects of an immediate predator contact to the spider, updating rewards and diagnostics.
        
        Updates `reward_components` with contact-related reward/penalty adjustments and mutates `info` to include diagnostic flags and details about the predator contact event. Also updates the spider's internal state counters and trackers related to predator contact.
         
        Parameters:
            reward_components (Dict[str, float]): Mapping of reward component names to their current values; this function will modify entries related to predator contact.
            info (Dict[str, object]): Diagnostic information dictionary that will be populated or updated with predator-contact flags and metadata.
        """
        apply_predator_contact(self, reward_components, info, tick_context=tick_context)
        if tick_context is not None:
            tick_context.predator_contact_applied = True

    def _capture_tick_snapshot(self) -> TickSnapshot:
        """
        Capture a lightweight snapshot of the current tick state for constructing a TickContext.
        
        Returns:
            TickSnapshot: snapshot with these fields populated:
                - tick: current tick as an int
                - spider_pos: (x, y) spider position
                - lizard_pos: (x, y) lizard position
                - was_on_shelter: `True` if the spider is on any shelter cell, `False` otherwise
                - prev_shelter_role: shelter role string at the spider position
                - prev_food_dist: Manhattan distance to the nearest food as an int
                - prev_shelter_dist: Manhattan distance to the nearest deep (preferred) or any shelter as an int
                - prev_predator_dist: Manhattan distance to the lizard as an int
                - prev_predator_visible: `True` if predator visibility confidence exceeds the configured threshold, `False` otherwise
                - night: `True` if the world is currently in night phase, `False` otherwise
                - rest_streak: current rest streak as an int
        """
        _, prev_food_dist = self.nearest(self.food_positions)
        _, prev_shelter_dist = self.nearest(self.shelter_deep_cells or self.shelter_cells)
        _, prev_predator_dist = self.nearest(
            self.predator_positions() or [self.lizard_pos()],
            origin=self.spider_pos(),
        )
        visibility_threshold = self.operational_profile.reward["predator_visibility_threshold"]
        prev_predator_visible = predator_visible_to_spider(self).visible > visibility_threshold
        return TickSnapshot(
            tick=int(self.tick),
            spider_pos=self.spider_pos(),
            lizard_pos=self.lizard_pos(),
            was_on_shelter=bool(self.on_shelter()),
            prev_shelter_role=self.shelter_role_at(self.spider_pos()),
            prev_food_dist=int(prev_food_dist),
            prev_shelter_dist=int(prev_shelter_dist),
            prev_predator_dist=int(prev_predator_dist),
            prev_predator_visible=bool(prev_predator_visible),
            night=bool(self.is_night()),
            rest_streak=int(self.state.rest_streak),
            momentum=float(np.clip(self.state.momentum, 0.0, 1.0)),
        )

    def step(self, action_idx: int) -> tuple[Dict[str, object], float, bool, Dict[str, object]]:
        """
        Advance the environment by one tick using the action at the given index.
        
        Parameters:
            action_idx (int): Index into ACTIONS selecting the spider's action for this step.
        
        Returns:
            next_obs (Dict[str, object]): Observation payload for the new timestep (arrays and metadata).
            reward (float): Scalar reward accumulated during this tick.
            done (bool): `True` if the episode has terminated (e.g., spider death), `False` otherwise.
            info (Dict[str, object]): Diagnostic information including, at minimum, the selected action name and flags/fields such as
                "ate", "slept", "pain", "predator_contact", "predator_transition", "predator_moved", "distance_deltas",
                "predator_escape", "reward_components", and a serialized "state" snapshot.
        """
        context = tick_stages.build_tick_context(self, action_idx)
        for descriptor in tick_stages.TICK_STAGES:
            try:
                descriptor.run(self, context)
            except Exception as exc:
                stage_name = getattr(descriptor, "name", None)
                if not stage_name:
                    stage_name = getattr(descriptor, "stage_name", repr(descriptor))
                exc.add_note(f"Tick stage {stage_name!r} failed.")
                raise
        return tick_stages.finalize_step(self, context)

    def state_dict(self) -> Dict[str, object]:
        """
        Serialize the world and agent state into a flat dictionary.
        
        The returned mapping contains the spider dataclass fields plus runtime diagnostics and derived views. Memory slot entries ("food_memory", "predator_memory", "shelter_memory", "escape_memory") are augmented with a "ttl" key. Percept trace fields ("food_trace", "shelter_trace", "predator_trace") are converted to their serialized trace views. Predator information includes a list of serialized predator dataclasses with an "index" for each entry and a separate "predator_positions" list of [x, y] coordinates. The mapping also provides tick-level diagnostics (tick, is_night, sleep_phase_level), shelter role and level at the spider, terrain under the spider, active map/reward/noise profile names, episode_seed, and a computed predator_motion_salience entry.
        
        Returns:
            dict: A flat dictionary of state and environment keys to their current values suitable for logging, debugging, or serialization. Memory slots contain a "ttl" entry and trace fields are serialized views; predators is a list of predator dicts with indices and predator_positions is a list of [x, y].
        """
        state = asdict(self.state)
        for key in ("food_memory", "predator_memory", "shelter_memory", "escape_memory"):
            ttl_name = key.replace("_memory", "")
            state[key]["ttl"] = MEMORY_TTLS[ttl_name]
        for key in ("food_trace", "shelter_trace", "predator_trace"):
            state[key] = trace_view(self, getattr(self.state, key))
        predator_view = predator_visible_to_spider(self, apply_noise=False)
        state.update(
            {
                "predator_count": self.predator_count,
                "predator_positions": [list(pos) for pos in self.predator_positions()],
                "predators": [
                    {
                        "index": idx,
                        **asdict(predator),
                    }
                    for idx, predator in enumerate(self.predators)
                ],
                "lizard_x": self.lizard.x,
                "lizard_y": self.lizard.y,
                "lizard_mode": self.lizard.mode,
                "lizard_mode_ticks": self.lizard.mode_ticks,
                "lizard_patrol_target": self.lizard.patrol_target,
                "lizard_last_known_spider": self.lizard.last_known_spider,
                "lizard_investigate_ticks": self.lizard.investigate_ticks,
                "lizard_investigate_target": self.lizard.investigate_target,
                "lizard_recover_ticks": self.lizard.recover_ticks,
                "lizard_wait_target": self.lizard.wait_target,
                "lizard_ambush_ticks": self.lizard.ambush_ticks,
                "lizard_chase_streak": self.lizard.chase_streak,
                "lizard_failed_chases": self.lizard.failed_chases,
                "tick": self.tick,
                "is_night": self.is_night(),
                "sleep_phase_level": self.sleep_phase_level(),
                "shelter_role": self.shelter_role_at(self.spider_pos()),
                "shelter_role_level": self.shelter_role_level(),
                "terrain": self.terrain_at(self.spider_pos()),
                "map_template": self.map_template_name,
                "reward_profile": self.reward_profile,
                "noise_profile": self.noise_profile.name,
                "episode_seed": self.episode_seed,
                "predator_motion_salience": predator_motion_salience(self, predator_view=predator_view),
            }
        )
        return state

    def render(self) -> str:
        """
        Render a compact textual snapshot of the current world state.
        
        The returned string begins with a single-line header containing tick, day/night phase, map and reward profile names, key physiology (hunger, fatigue, sleep debt, health), recent pain and last reward, shelter role and sleep encoding, last action, and the primary predator's position and mode. The header is followed by a grid where each character represents a cell: '#' blocked, ':' clutter, '=' narrow, 'E' shelter entrance, 'I' shelter interior, 'D' shelter deep, 'F' food, 'L' the primary predator, 'P' other predators, and 'A' the spider (or 'X' if the spider shares a cell with any predator).
        
        Returns:
            A multi-line string with the header on the first line and the ASCII grid on subsequent lines.
        """
        chars = [["." for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.blocked_cells:
            chars[y][x] = "#"
        for x in range(self.width):
            for y in range(self.height):
                if chars[y][x] != ".":
                    continue
                if self.terrain_at((x, y)) == CLUTTER:
                    chars[y][x] = ":"
                elif self.terrain_at((x, y)) == NARROW:
                    chars[y][x] = "="
        for x, y in self.shelter_entrance_cells:
            chars[y][x] = "E"
        for x, y in self.shelter_interior_cells:
            chars[y][x] = "I"
        for x, y in self.shelter_deep_cells:
            chars[y][x] = "D"
        for x, y in self.food_positions:
            chars[y][x] = "F"
        for index, (px, py) in enumerate(self.predator_positions()):
            chars[py][px] = "L" if index == 0 else "P"
        sx, sy = self.spider_pos()
        chars[sy][sx] = "X" if (sx, sy) in set(self.predator_positions()) else "A"
        rows = ["".join(row) for row in chars]
        phase = "NIGHT" if self.is_night() else "DAY"
        header = (
            f"tick={self.tick} phase={phase} map={self.map_template_name} reward_profile={self.reward_profile} "
            f"hunger={self.state.hunger:.2f} fatigue={self.state.fatigue:.2f} sleep_debt={self.state.sleep_debt:.2f} "
            f"health={self.state.health:.2f} pain={self.state.recent_pain:.2f} reward={self.state.last_reward:.2f} "
            f"shelter={self.shelter_role_at(self.spider_pos())} sleep={self.state.sleep_phase}:{self.state.rest_streak} "
            f"last_action={self.state.last_action} lizard=({self.lizard.x},{self.lizard.y})/{self.lizard.mode}"
        )
        return header + "\n" + "\n".join(rows)
