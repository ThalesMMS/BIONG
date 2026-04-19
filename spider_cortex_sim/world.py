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

from .world_dynamics import WorldDynamicsMixin
from .world_export import WorldExportMixin
from .world_navigation import WorldNavigationMixin
from .world_perception import WorldPerceptionMixin
from .world_spawning import WorldSpawningMixin
from .world_support import ACTIONS, MOVE_DELTAS, MOMENTUM_DECAY_ON_STOP, MOMENTUM_BOOST_ON_SAME_DIR, MOMENTUM_FRICTION_ON_TURN, SHELTER_ROLE_LEVELS, SCAN_AGE_NEVER, SCAN_TICK_FIELDS, _scan_tick_field_for_heading, _scan_age_never_for_world, _scan_age_for_heading, _refresh_perception_for_active_scan, _copy_observation_payload, PerceptualBuffer, _is_temporal_direction_field


class SpiderWorld(WorldNavigationMixin, WorldPerceptionMixin, WorldSpawningMixin, WorldDynamicsMixin, WorldExportMixin):
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
