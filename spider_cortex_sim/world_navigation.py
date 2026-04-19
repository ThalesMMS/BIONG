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

from .world_support import ACTIONS, MOVE_DELTAS, MOMENTUM_DECAY_ON_STOP, MOMENTUM_BOOST_ON_SAME_DIR, MOMENTUM_FRICTION_ON_TURN, SHELTER_ROLE_LEVELS, SCAN_AGE_NEVER, SCAN_TICK_FIELDS, _scan_tick_field_for_heading, _scan_age_never_for_world, _scan_age_for_heading, _refresh_perception_for_active_scan, _copy_observation_payload, PerceptualBuffer, _is_temporal_direction_field


class WorldNavigationMixin:
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
