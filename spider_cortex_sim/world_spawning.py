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


class WorldSpawningMixin:
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
