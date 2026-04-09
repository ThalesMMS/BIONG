from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .interfaces import ACTION_DELTAS, ACTION_TO_INDEX, LOCOMOTION_ACTIONS
from .maps import BLOCKED, CLUTTER, MAP_TEMPLATE_NAMES, NARROW, build_map_template, terrain_at
from .memory import MEMORY_TTLS, age_or_clear_memory, empty_memory_slot, escape_memory_target, memory_vector, refresh_memory, set_memory
from .noise import NoiseConfig, resolve_noise_profile
from .perception import (
    PerceivedTarget,
    has_line_of_sight,
    lizard_detects_spider,
    observe_world,
    predator_motion_salience,
    predator_visible_to_spider,
    smell_gradient,
    visible_object,
    visibility_confidence,
    visible_range,
)
from .physiology import (
    SLEEP_PHASE_LEVELS,
    apply_homeostasis_penalties,
    apply_predator_contact,
    apply_restoration,
    apply_wakefulness,
    clip_state,
    reset_sleep_state,
    resolve_autonomic_behaviors,
    rest_streak_norm,
    set_sleep_state,
    sleep_phase_from_streak,
    sleep_phase_level,
)
from .predator import LizardState, PredatorController
from .operational_profiles import OperationalProfile, resolve_operational_profile
from .reward import (
    REWARD_COMPONENT_NAMES,
    REWARD_PROFILES,
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_total,
)
from .world_types import MemorySlot, PerceptTrace, SpiderState, TickContext, TickSnapshot


ACTIONS: Sequence[str] = tuple(LOCOMOTION_ACTIONS)
MOVE_DELTAS = tuple(ACTION_DELTAS[action] for action in ACTIONS if action != "STAY")
SHELTER_ROLE_LEVELS = {
    "outside": 0.0,
    "entrance": 1.0 / 3.0,
    "inside": 2.0 / 3.0,
    "deep": 1.0,
}


class SpiderWorld:
    """Grid world with explicit shelter geometry, memories, and a predator FSM.

    Explicit memory is maintained by the world and is observable in traces and the GUI.
    The brain only consumes those signals; there is no second latent memory system outside this module.
    """

    move_deltas = MOVE_DELTAS

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        food_count: int = 4,
        day_length: int = 18,
        night_length: int = 12,
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
        self.operational_profile = resolve_operational_profile(operational_profile)
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
        self.tick = 0
        self.food_positions: List[Tuple[int, int]] = []
        self.predator_controller = PredatorController()
        self.state = self._initial_spider_state(*self.map_template.spider_start)
        self.lizard = self._spawn_lizard()
        self._last_on_shelter = False
        self._last_predator_visible = False
        self._predator_threat_episode_active = False
        self._predator_escape_bonus_pending = False

    def _reset_rngs(self, resolved_seed: int) -> None:
        """
        Reinitialize per-episode random number generators and record the episode seed.
        
        Sets `self.episode_seed` from `resolved_seed` and derives five independent NumPy RNGs (spawn, predator, visual, olfactory, motor) from a single SeedSequence seeded by the instance seed and the episode seed. Also sets `self.rng` as a backward-compatible alias for `self.predator_rng`.
        
        Parameters:
            resolved_seed (int): Episode-specific seed used to derive the RNG channels.
        
        """
        self.episode_seed = int(resolved_seed)
        seed_sequence = np.random.SeedSequence([int(self.seed), self.episode_seed])
        channel_sequences = seed_sequence.spawn(5)
        self.spawn_rng = np.random.default_rng(channel_sequences[0])
        self.predator_rng = np.random.default_rng(channel_sequences[1])
        self.visual_rng = np.random.default_rng(channel_sequences[2])
        self.olfactory_rng = np.random.default_rng(channel_sequences[3])
        self.motor_rng = np.random.default_rng(channel_sequences[4])
        # Backward-compatible alias for legacy world-owned random events.
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

    def _empty_memory_slot(self) -> MemorySlot:
        """
        Create a new empty memory slot for the spider.
        
        Returns:
            MemorySlot: A freshly initialized memory slot with default (empty) contents.
        """
        return empty_memory_slot()

    @staticmethod
    def _empty_percept_trace() -> PerceptTrace:
        """
        Create a new empty short-lived percept trace slot.
        """
        return PerceptTrace(target=None, age=0, certainty=0.0)

    @staticmethod
    def _heading_components_from_delta(dx: int, dy: int) -> tuple[int, int]:
        """
        Convert an arbitrary delta into a compact signed heading vector.
        """
        return int(np.sign(dx)), int(np.sign(dy))

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

    def _percept_trace_ttl(self) -> int:
        """
        Return the configured TTL for short percept traces.
        """
        return max(1, round(self.operational_profile.perception["percept_trace_ttl"]))

    def _percept_trace_decay(self) -> float:
        """
        Return the configured multiplicative decay for short percept traces.
        """
        return float(np.clip(self.operational_profile.perception["percept_trace_decay"], 0.0, 1.0))

    def _trace_strength(self, trace: PerceptTrace) -> float:
        """
        Compute the current decayed strength of a short percept trace.
        """
        if trace.target is None or trace.age >= self._percept_trace_ttl():
            return 0.0
        return float(np.clip(trace.certainty * (self._percept_trace_decay() ** trace.age), 0.0, 1.0))

    def _trace_view(self, trace: PerceptTrace) -> dict[str, object]:
        """
        Serialize a percept trace with derived direction and strength metadata.
        """
        strength = self._trace_strength(trace)
        if trace.target is None or strength <= 0.0:
            dx = 0.0
            dy = 0.0
        else:
            dx, dy, _ = self._relative(trace.target)
        return {
            "target": None if trace.target is None else [int(trace.target[0]), int(trace.target[1])],
            "age": int(trace.age),
            "certainty": float(trace.certainty),
            "strength": float(strength),
            "dx": float(dx),
            "dy": float(dy),
            "ttl": self._percept_trace_ttl(),
            "decay": self._percept_trace_decay(),
        }

    def _advance_percept_trace(
        self,
        trace: PerceptTrace,
        percept: PerceivedTarget,
        positions: Iterable[Tuple[int, int]],
    ) -> PerceptTrace:
        """
        Refresh or decay a short-lived `PerceptTrace` based on the latest raw percept.

        A trace is refreshed only when a visible percept provides an explicit
        `percept.position` that matches the candidate set and remains consistent
        with the reported distance, current heading, and line of sight. On
        refresh the returned trace resets to `age=0` and stores certainty
        clipped to the unit interval; otherwise the existing trace ages or clears
        through the normal decay path.
        """
        if percept.visible > 0.0 and percept.occluded <= 0.0 and percept.position is not None:
            source = self.spider_pos()
            candidate_set = {tuple(pos) for pos in positions}
            percept_dist = int(percept.dist)
            target = tuple(percept.position)
            if target in candidate_set:
                rel_x = int(target[0] - source[0])
                rel_y = int(target[1] - source[1])
                heading = (int(self.state.heading_dx), int(self.state.heading_dy))
                heading_allows_target = heading == (0, 0) or (
                    rel_x == 0
                    and rel_y == 0
                ) or (heading[0] * rel_x + heading[1] * rel_y) >= 0
                if (
                    self.manhattan(source, target) == percept_dist
                    and heading_allows_target
                    and has_line_of_sight(self, source, target)
                ):
                    return PerceptTrace(
                        target=(int(target[0]), int(target[1])),
                        age=0,
                        certainty=float(np.clip(percept.certainty, 0.0, 1.0)),
                    )

        if trace.target is None:
            return self._empty_percept_trace()
        aged = PerceptTrace(
            target=trace.target,
            age=int(trace.age) + 1,
            certainty=float(np.clip(trace.certainty, 0.0, 1.0)),
        )
        if self._trace_strength(aged) <= 0.0:
            return self._empty_percept_trace()
        return aged

    def _refresh_perceptual_state(self) -> None:
        """
        Update world-owned perceptual traces after reset and each completed tick.
        """
        radius = visible_range(self)
        food_view = visible_object(self, self.food_positions, radius=radius, apply_noise=False)
        shelter_view = visible_object(self, self.shelter_cells, radius=radius, apply_noise=False)
        predator_view = predator_visible_to_spider(self, apply_noise=False)

        self.state.food_trace = self._advance_percept_trace(
            self.state.food_trace,
            food_view,
            self.food_positions,
        )
        self.state.shelter_trace = self._advance_percept_trace(
            self.state.shelter_trace,
            shelter_view,
            self.shelter_cells,
        )
        self.state.predator_trace = self._advance_percept_trace(
            self.state.predator_trace,
            predator_view,
            [self.lizard_pos()],
        )

    def _initial_spider_state(self, start_x: int, start_y: int) -> SpiderState:
        """
        Create a new SpiderState for a spider spawned or reset at the given coordinates.
        
        The state has position (start_x, start_y); physiology fields (`hunger`, `fatigue`, `sleep_debt`) initialized by sampling from the configured initial ranges; `health`, recent-event trackers, counters, reward accumulators, and action bookkeeping set to their default baselines; and all memory slots initialized to empty entries.
        
        Parameters:
            start_x (int): Starting x-coordinate for the spider.
            start_y (int): Starting y-coordinate for the spider.
        
        Returns:
            SpiderState: Initialized spider state with randomized initial physiology and defaulted counters, memory, and bookkeeping fields.
        """
        initial_heading_dx, initial_heading_dy = self._heading_toward(
            self.nearest_shelter_entrance(origin=(start_x, start_y)),
            origin=(start_x, start_y),
        )
        return SpiderState(
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
            food_memory=self._empty_memory_slot(),
            predator_memory=self._empty_memory_slot(),
            shelter_memory=self._empty_memory_slot(),
            escape_memory=self._empty_memory_slot(),
            food_trace=self._empty_percept_trace(),
            shelter_trace=self._empty_percept_trace(),
            predator_trace=self._empty_percept_trace(),
        )

    def reset(self, seed: int | None = None) -> Dict[str, np.ndarray]:
        """
        Reset the world to its initial state.
        
        Reinitializes RNGs, simulation tick, spider state, food spawns, lizard predator, and observable memory, then returns the initial observation.
        
        Parameters:
            seed (int | None): Optional seed to initialize the world's RNG channels; if None the instance's configured seed is used.
        
        Returns:
            Dict[str, np.ndarray]: The initial observation dictionary as produced by observe().
        """
        resolved_seed = int(seed) if seed is not None else int(self.seed)
        self._reset_rngs(resolved_seed)
        self.tick = 0
        self.state = self._initial_spider_state(*self.map_template.spider_start)
        self._last_on_shelter = True
        self._last_predator_visible = False
        self._predator_threat_episode_active = False
        self._predator_escape_bonus_pending = False
        self.food_positions = []
        for _ in range(self.food_count):
            self.food_positions.append(self._random_food_cell())
        self.lizard = self._spawn_lizard()
        self.refresh_memory(initial=True)
        return self.observe()

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

    def _visible_range(self) -> int:
        """
        Get the visibility radius used for this world's perception.
        
        Returns:
            visibility_radius (int): Maximum number of grid cells around an origin considered when computing visibility.
        """
        return visible_range(self)

    def _visibility_confidence(
        self,
        *,
        source: Tuple[int, int],
        target: Tuple[int, int],
        dist: int,
        radius: int,
        motion_bonus: float = 0.0,
    ) -> float:
        """
        Compute the visibility confidence that a target cell is visible from a source cell.
        
        Parameters:
            source (Tuple[int, int]): (x, y) coordinates of the observation origin.
            target (Tuple[int, int]): (x, y) coordinates of the target cell.
            dist (int): Manhattan distance between source and target.
            radius (int): Maximum visibility radius to consider.
            motion_bonus (float): Extra confidence added for observed motion; defaults to 0.0.
        
        Returns:
            float: Confidence score between 0.0 and 1.0 indicating the likelihood the target is visible from the source.
        """
        return visibility_confidence(
            self,
            source=source,
            target=target,
            dist=dist,
            radius=radius,
            motion_bonus=motion_bonus,
        )

    @staticmethod
    def _empty_reward_components() -> Dict[str, float]:
        """
        Create a new mapping of reward component names initialized to their default float values.
        
        Returns:
            Dict[str, float]: A dictionary mapping each reward component key to its initial numeric value.
        """
        return empty_reward_components()

    @staticmethod
    def _reward_total(reward_components: Dict[str, float]) -> float:
        """
        Aggregate individual reward component values into a single total reward.
        
        Parameters:
            reward_components (Dict[str, float]): Mapping from reward component names to their numeric values.
        
        Returns:
            total (float): Total reward computed from the provided components.
        """
        return reward_total(reward_components)

    @staticmethod
    def _copy_reward_components(reward_components: Dict[str, float]) -> Dict[str, float]:
        """
        Return a shallow copy of the reward components mapping.
        
        Parameters:
            reward_components (Dict[str, float]): Mapping from reward component names to their numeric values.
        
        Returns:
            Dict[str, float]: A new dictionary containing the same keys and numeric values as `reward_components`.
        """
        return copy_reward_components(reward_components)

    def _set_sleep_state(self, phase: str, rest_streak: int) -> None:
        """
        Update the spider's sleep phase and rest-streak counters in its state.
        
        Parameters:
            phase (str): Label identifying the sleep phase to apply to the spider (for example "AWAKE" or "ASLEEP").
            rest_streak (int): Number of consecutive ticks the spider has been resting; used to track/restoration and sleep-phase progression.
        """
        set_sleep_state(self, phase, rest_streak)

    def _reset_sleep_state(self) -> None:
        """
        Reset the spider's sleep-related state to default values.
        
        Resets internal sleep bookkeeping (sleep phase, rest streak, and related flags) on the spider state so it begins from a neutral/default sleep configuration.
        """
        reset_sleep_state(self)

    def _sleep_phase_from_streak(self, rest_streak: int, *, night: bool, shelter_role: str) -> str:
        """
        Map a rest streak and context to the corresponding sleep phase.
        
        Parameters:
            rest_streak (int): Consecutive ticks the spider has been resting.
            night (bool): Whether the current time is night.
            shelter_role (str): One of the shelter role names (e.g., "outside", "entrance", "inside", "deep").
        
        Returns:
            str: The sleep phase name for the given rest streak and context (e.g., "AWAKE", "LIGHT", "DEEP").
        """
        return sleep_phase_from_streak(rest_streak, night=night, shelter_role=shelter_role)

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

    def _has_line_of_sight(
        self,
        origin: Tuple[int, int],
        target: Tuple[int, int],
        *,
        block_clutter: bool = True,
    ) -> bool:
        """
        Determine whether there is an unobstructed line of sight between two grid cells.
        
        Parameters:
            origin (Tuple[int, int]): (x, y) coordinates of the source cell.
            target (Tuple[int, int]): (x, y) coordinates of the destination cell.
            block_clutter (bool): If True, treat clutter terrain as blocking sight; if False, clutter does not block sight.
        
        Returns:
            bool: `True` if the target cell is visible from the origin (no blocking terrain or clutter per `block_clutter`), `False` otherwise.
        """
        return has_line_of_sight(self, origin, target, block_clutter=block_clutter)

    def _visible_object(
        self,
        positions: Iterable[Tuple[int, int]],
        *,
        radius: int,
        origin: Tuple[int, int] | None = None,
        motion_bonus: float = 0.0,
    ) -> PerceivedTarget:
        """
        Determine the perceived target among candidate positions from a source cell.
        
        Parameters:
            positions (Iterable[Tuple[int, int]]): Candidate (x, y) positions to evaluate.
            radius (int): Maximum sensing radius to consider.
            origin (Tuple[int, int] | None): Source cell from which visibility is evaluated; if None, the spider's current position is used.
            motion_bonus (float): Bonus added to visibility confidence for recently moving targets.
        
        Returns:
            PerceivedTarget: Perception result for the supplied positions from the given origin (contains the selected target and associated visibility metrics).
        """
        return visible_object(
            self,
            positions,
            radius=radius,
            origin=origin,
            motion_bonus=motion_bonus,
        )

    def _smell_gradient(
        self,
        positions: Iterable[Tuple[int, int]],
        *,
        radius: int,
        origin: Tuple[int, int] | None = None,
        apply_noise: bool = True,
    ) -> tuple[float, float, float, int]:
        """
        Compute the smell gradient from the given source positions at a query origin within a search radius.
        
        Parameters:
            positions (Iterable[Tuple[int, int]]): Source cell coordinates that emit the smell.
            radius (int): Maximum Manhattan radius to consider when computing the gradient.
            origin (tuple[int, int] | None): Query cell coordinates; if None, the spider's position is used.
        
        Returns:
            tuple[float, float, float, int]: A 4-tuple containing (strength, grad_x, grad_y, distance) where
                - strength (float): aggregated smell intensity at the origin,
                - grad_x, grad_y (float): gradient components indicating smell direction,
                - distance (int): Manhattan distance from the origin to the nearest source considered.
        """
        return smell_gradient(
            self,
            positions,
            radius=radius,
            origin=origin,
            apply_noise=apply_noise,
        )

    def lizard_detects_spider(self) -> bool:
        """
        Determines whether the lizard currently detects the spider.
        
        Returns:
            True if the lizard detects the spider, False otherwise.
        """
        return lizard_detects_spider(self)

    def _predator_visible_to_spider(self) -> PerceivedTarget:
        """
        Determine whether the predator is visible from the spider's perspective and produce a perception record.
        
        Returns:
            PerceivedTarget: A perception record describing the predator's visibility and associated observation metrics as seen by the spider.
        """
        return predator_visible_to_spider(self)

    def _memory_vector(self, slot: MemorySlot, *, ttl_name: str) -> tuple[float, float, float]:
        """
        Compute a fixed-length vector representation for a memory slot.
        
        Parameters:
        	slot (MemorySlot): Memory slot to encode.
        	ttl_name (str): Name of the TTL category used to normalize the memory's remaining lifetime.
        
        Returns:
        	(dx_norm, dy_norm, ttl_frac) (tuple[float, float, float]): Normalized x and y components of the memory vector and the remaining time-to-live as a fraction between 0.0 and 1.0.
        """
        return memory_vector(self, slot, ttl_name=ttl_name)

    def _age_or_clear_memory(self, slot: MemorySlot, *, ttl_name: str) -> None:
        """
        Age or clear a memory slot based on the named TTL policy.
        
        Parameters:
            slot (MemorySlot): The memory slot to update; its internal age/TTL will be advanced and the slot cleared if it has expired.
            ttl_name (str): The TTL category name used to look up expiration rules for the slot.
        """
        age_or_clear_memory(slot, ttl_name=ttl_name)

    def _set_memory(self, slot: MemorySlot, target: Tuple[int, int] | None) -> None:
        """
        Set a memory slot to reference a target grid position or clear that memory.
        
        Parameters:
            slot (MemorySlot): The memory slot to update.
            target (Tuple[int, int] | None): Coordinates (x, y) to store in the slot, or `None` to clear it.
        """
        set_memory(slot, target)

    def _escape_memory_target(self) -> Tuple[int, int]:
        """
        Get the target cell coordinates stored in the spider's escape memory.
        
        Returns:
            (x, y) tuple of int: Coordinates of the escape target cell.
        """
        return escape_memory_target(self)

    def refresh_memory(self, *, predator_escape: bool = False, initial: bool = False) -> None:
        """
        Refresh the spider's episodic memories by aging, clearing expired entries, and updating memory traces.
        
        This updates the spider's memory slots (food, predator, shelter, escape) in place according to their time-to-live rules and current perceptions. If predator_escape is True, memory processing will record or strengthen an escape-related memory; if initial is True, perform an initial refresh that may bypass normal decay to establish baseline memory state.
        Parameters:
            predator_escape (bool): If True, treat this refresh as occurring after a predator escape event, causing escape-related memory updates.
            initial (bool): If True, perform an initial/baseline refresh (used on reset) that establishes memory state without normal decay.
        """
        refresh_memory(self, predator_escape=predator_escape, initial=initial)
        self._refresh_perceptual_state()

    def observe(self) -> Dict[str, np.ndarray]:
        """
        Compose the spider agent's current observation tensors from the world state.
        
        Returns:
            observations (Dict[str, np.ndarray]): Mapping from observation keys (e.g., visual map, smell fields, scalar features) to NumPy arrays representing the agent's current perception and state-derived inputs.
        """
        return observe_world(self)

    def visibility_overlay(self, *, origin: Tuple[int, int] | None = None) -> Dict[str, object]:
        """
        Compute which grid cells are visible from a given origin within the current visible range.
        
        Parameters:
            origin (Tuple[int, int] | None): Grid cell (x, y) to use as the visibility source. If None, uses the spider's current position.
        
        Returns:
            Dict[str, object]: A dictionary with keys:
                - "origin": the source cell used (Tuple[int, int])
                - "radius": the visibility radius (int)
                - "visible": list of cells (List[Tuple[int, int]]) within radius that have line of sight from the origin
                - "occluded": list of cells (List[Tuple[int, int]]) within radius that are not in line of sight
        """
        source = origin if origin is not None else self.spider_pos()
        radius = self._visible_range()
        visible: List[Tuple[int, int]] = []
        occluded: List[Tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                dist = self.manhattan(source, pos)
                if pos == source or dist > radius:
                    continue
                if self._has_line_of_sight(source, pos):
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
        if kind == "food":
            positions = list(self.food_positions)
            radius = self.food_smell_range
        elif kind == "predator":
            positions = [self.lizard_pos()]
            radius = self.predator_smell_range
        else:
            raise ValueError(f"Unknown scent field: {kind}")
        field: List[List[float]] = []
        for y in range(self.height):
            row: List[float] = []
            for x in range(self.width):
                strength, _, _, _ = self._smell_gradient(
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
    ) -> Tuple[int, int]:
        """
        Selects a spawn cell from the given candidate cells, applying filters and fallbacks.
        
        Filters out the current spider cell, existing food positions, (optionally) the lizard cell, and any cell closer than `min_spider_distance` by Manhattan distance. If no candidates remain after filtering, selects uniformly from all walkable, non-shelter cells excluding the spider cell. Selection is sampled uniformly using the spawn RNG.
        
        Parameters:
            candidates (Sequence[Tuple[int, int]]): Candidate cells to consider for spawning.
            min_spider_distance (int, optional): Minimum required Manhattan distance from the spider. Defaults to 0.
            avoid_lizard (bool, optional): If True, exclude the current lizard position from candidates. Defaults to True.
        
        Returns:
            Tuple[int, int]: The chosen spawn cell coordinates (x, y).
        """
        spider_pos = self.spider_pos()
        lizard_pos = self.lizard_pos() if avoid_lizard else None
        filtered = [
            cell
            for cell in candidates
            if cell != spider_pos
            and cell not in self.food_positions
            and (lizard_pos is None or cell != lizard_pos)
            and self.manhattan(cell, spider_pos) >= min_spider_distance
        ]
        if not filtered:
            filtered = [
                (x, y)
                for x in range(self.width)
                for y in range(self.height)
                if self.is_walkable((x, y))
                and (x, y) not in self.shelter_cells
                and (x, y) != spider_pos
            ]
        idx = int(self.spawn_rng.integers(0, len(filtered)))
        return filtered[idx]

    def _random_food_cell(self) -> Tuple[int, int]:
        """
        Selects a cell to spawn a new food item, preferring candidate cells near a shelter entrance while avoiding the spider and existing food.
        
        If the configured map template provides food spawn candidates, returns one of those candidates; otherwise falls back to a general random spawn selection. Candidate selection weights favor proximity to the nearest shelter entrance but are mixed with a uniform distribution according to self.noise_profile.spawn["uniform_mix"] (clipped to [0,1]) to introduce randomness.
        
        Returns:
            tuple[int, int]: Coordinates (x, y) of the chosen food spawn cell.
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

    def _spawn_lizard(self) -> LizardState:
        """
        Select a spawn cell for the lizard (ensuring a minimum distance from the spider) and return its initial state.
        
        The spawn distance is at least 3 cells or one third of the smaller map dimension, whichever is larger. The selection is drawn from the map template's lizard spawn cells and may place the lizard anywhere those candidates allow.
        
        Returns:
            LizardState: A newly initialized lizard state with `x`, `y` set to the chosen cell and `mode` set to `"PATROL"`.
        """
        min_dist = max(3, min(self.width, self.height) // 3)
        pos = self._random_spawn_cell(
            self.map_template.lizard_spawn_cells,
            min_spider_distance=min_dist,
            avoid_lizard=False,
        )
        return LizardState(x=pos[0], y=pos[1], mode="PATROL")

    def respawn_food(self, eaten_position: Tuple[int, int]) -> None:
        self.food_positions = [p for p in self.food_positions if p != eaten_position]
        self.food_positions.append(self._random_food_cell())

    def _move(self, dx: int, dy: int) -> bool:
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
        return moved

    def _move_spider_action(self, action_name: str) -> bool:
        """
        Move the spider according to the given action name.
        
        Parameters:
            action_name (str): Action identifier (one of the entries in `ACTIONS`) indicating the direction or `"STAY"`.
        
        Returns:
            `true` if the spider moved to a different cell, `false` otherwise.
        """
        dx, dy = ACTION_DELTAS[action_name]
        return self._move(dx, dy)

    def _apply_motor_noise(self, action_name: str) -> tuple[str, bool]:
        """
        Selects the action to execute, optionally replacing the intended action with a randomly chosen alternative according to the configured motor-noise probability.
        
        Parameters:
            action_name (str): The intended action name.
        
        Returns:
            tuple[str, bool]: (executed_action, motor_noise_applied) where `executed_action` is the action to perform and `motor_noise_applied` is `True` if the action was replaced due to motor noise, `False` otherwise.
        """
        flip_prob = float(np.clip(self.noise_profile.motor["action_flip_prob"], 0.0, 1.0))
        if flip_prob <= 0.0 or len(ACTIONS) <= 1:
            return action_name, False
        if float(self.motor_rng.random()) >= flip_prob:
            return action_name, False
        alternatives = [candidate for candidate in ACTIONS if candidate != action_name]
        index = int(self.motor_rng.integers(0, len(alternatives)))
        return alternatives[index], True

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

    def _apply_wakefulness(self, *, night: bool, exposed: bool, interrupted_rest: bool) -> None:
        """
        Update the spider's wakefulness and related sleep-state fields based on current conditions.
        
        Parameters:
            night (bool): True if the world is currently in the night phase.
            exposed (bool): True if the spider is exposed to the environment (not sheltered) at this tick.
            interrupted_rest (bool): True if the spider's ongoing rest was interrupted by action, predator threat, or movement.
        """
        apply_wakefulness(self, night=night, exposed=exposed, interrupted_rest=interrupted_rest)

    def _apply_restoration(self, sleep_phase: str, *, night: bool, shelter_role: str) -> None:
        """
        Update the spider's physiological restoration state according to the current sleep phase, time of day, and shelter role.
        
        Parameters:
            sleep_phase (str): Label of the spider's current sleep phase (e.g., "AWAKE", "SLEEP_PHASE_*").
            night (bool): True if the world is currently in the night portion of the cycle.
            shelter_role (str): Shelter role at the spider's position ("outside", "entrance", "inside", or "deep").
        """
        apply_restoration(self, sleep_phase, night=night, shelter_role=shelter_role)

    def _capture_tick_snapshot(self) -> TickSnapshot:
        """
        Capture a lightweight snapshot of the current tick state for constructing a TickContext.
        
        The snapshot records current tick and positions, recent-distance metrics to food/shelter/predator,
        whether the predator was visible, the spider's shelter membership and role, night flag, and the
        current rest streak.
        
        Returns:
            TickSnapshot: Fields populated are
                - tick: current tick as int
                - spider_pos: (x, y) spider position
                - lizard_pos: (x, y) lizard position
                - was_on_shelter: True if spider was on any shelter cell
                - prev_shelter_role: shelter role string at the spider position
                - prev_food_dist: Manhattan distance to nearest food as int
                - prev_shelter_dist: Manhattan distance to nearest (deep or any) shelter as int
                - prev_predator_dist: Manhattan distance to the lizard as int
                - prev_predator_visible: True if predator visibility confidence > 0.5
                - night: True if it is currently night
                - rest_streak: current rest streak as int
        """
        _, prev_food_dist = self.nearest(self.food_positions)
        _, prev_shelter_dist = self.nearest(self.shelter_deep_cells or self.shelter_cells)
        prev_predator_dist = self.manhattan(self.spider_pos(), self.lizard_pos())
        visibility_threshold = self.operational_profile.reward["predator_visibility_threshold"]
        prev_predator_visible = self._predator_visible_to_spider().visible > visibility_threshold
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
        )

    def _new_tick_context(self, action_idx: int) -> TickContext:
        """
        Create and initialize a TickContext for a new simulation tick.
        
        The context contains the intended and executed actions (after motor-noise resolution), a pre-tick snapshot, an empty reward-components container, an initial info dictionary (booleans and placeholders for events), and records the initial `"pre_tick"` and `"action_resolved"` events on the context.
        
        Parameters:
            action_idx (int): Index of the chosen action in ACTIONS.
        
        Returns:
            TickContext: A fully initialized tick context ready for the per-stage pipeline.
        """
        if isinstance(action_idx, bool) or not isinstance(action_idx, (int, np.integer)):
            raise TypeError(
                f"Invalid action_idx {action_idx!r}; expected an integer in the range "
                f"[0, {len(ACTIONS) - 1}]"
            )
        action_idx = int(action_idx)
        if not 0 <= action_idx < len(ACTIONS):
            raise ValueError(
                f"Invalid action_idx {action_idx!r}; expected an integer in the range "
                f"[0, {len(ACTIONS) - 1}]"
            )
        intended_action = ACTIONS[action_idx]
        executed_action, motor_noise_applied = self._apply_motor_noise(intended_action)
        snapshot = self._capture_tick_snapshot()
        context = TickContext(
            action_idx=int(action_idx),
            intended_action=intended_action,
            executed_action=executed_action,
            motor_noise_applied=bool(motor_noise_applied),
            snapshot=snapshot,
            reward_components=self._empty_reward_components(),
            info={
                "action": executed_action,
                "intended_action": intended_action,
                "executed_action": executed_action,
                "motor_noise_applied": bool(motor_noise_applied),
                "ate": False,
                "slept": False,
                "pain": False,
                "predator_contact": False,
                "predator_transition": None,
                "distance_deltas": {},
            },
        )
        context.record_event("pre_tick", "snapshot", **snapshot.to_payload())
        context.record_event(
            "action",
            "action_resolved",
            action_index=int(action_idx),
            intended_action=intended_action,
            executed_action=executed_action,
            motor_noise_applied=bool(motor_noise_applied),
        )
        return context

    def _run_action_stage(self, context: TickContext) -> None:
        """
        Apply the current tick's resolved movement to the spider and record the resulting movement event.
        
        Updates context.moved to indicate whether the spider changed cells, moves the spider according to context.executed_action, and records an "action"/"movement_applied" event containing the spider's new position and last move deltas.
        
        Parameters:
            context (TickContext): Tick context to update with the movement outcome and event entry.
        """
        context.moved = bool(self._move_spider_action(context.executed_action))
        context.record_event(
            "action",
            "movement_applied",
            moved=bool(context.moved),
            spider_pos=[int(self.state.x), int(self.state.y)],
            last_move_dx=int(self.state.last_move_dx),
            last_move_dy=int(self.state.last_move_dy),
        )

    def _run_terrain_and_wakefulness_stage(self, context: TickContext) -> None:
        """
        Apply action and terrain effects, compute predator threat and exposure, and update wakefulness-related physiology and rewards for the current tick.
        
        Updates the provided TickContext with: `terrain_now`, `predator_threat`, `interrupted_rest`, and `exposed_at_night`. Calls wakefulness/restoration logic, increments `state.fatigue` and adjusts `context.reward_components` when exposed at night, applies pressure penalties, and records a terrain/wakefulness event in the context's event log.
        
        Parameters:
            context (TickContext): Per-tick context to be mutated with terrain, threat, rest/exposure flags, and reward updates.
        """
        cfg = self.reward_config
        context.terrain_now = apply_action_and_terrain_effects(
            self,
            action_name=context.executed_action,
            moved=context.moved,
            reward_components=context.reward_components,
        )
        context.predator_threat = bool(
            compute_predator_threat(
                self,
                prev_predator_visible=context.snapshot.prev_predator_visible,
                prev_predator_dist=context.snapshot.prev_predator_dist,
            )
        )
        context.interrupted_rest = bool(
            context.snapshot.rest_streak > 0
            and (
                context.executed_action != "STAY"
                or context.predator_threat
                or context.snapshot.prev_shelter_role == "outside"
            )
        )
        context.exposed_at_night = bool(
            context.snapshot.night and self.shelter_role_at(self.spider_pos()) == "outside"
        )
        self._apply_wakefulness(
            night=context.snapshot.night,
            exposed=context.exposed_at_night,
            interrupted_rest=context.interrupted_rest,
        )
        if context.exposed_at_night:
            self.state.fatigue += cfg["night_exposure_fatigue"]
            context.reward_components["night_exposure"] -= cfg["night_exposure_reward"]
        apply_pressure_penalties(self, context.reward_components)
        context.record_event(
            "terrain_and_wakefulness",
            "effects_applied",
            terrain=context.terrain_now,
            predator_threat=bool(context.predator_threat),
            interrupted_rest=bool(context.interrupted_rest),
            exposed_at_night=bool(context.exposed_at_night),
            sleep_debt=round(float(self.state.sleep_debt), 6),
            fatigue=round(float(self.state.fatigue), 6),
        )

    def _run_immediate_predator_contact_stage(self, context: TickContext) -> None:
        """
        Check for immediate predator contact and handle it if present.
        
        If the spider occupies the same cell as the lizard, applies predator-contact effects by calling
        _internal handler with the tick context's reward components and info. If no contact occurs,
        records a `"predator_contact"` `"contact_check"` event on the provided TickContext.
        
        Parameters:
        	context (TickContext): Per-tick context containing `reward_components`, `info`, and event logging.
        """
        if self.spider_pos() == self.lizard_pos() and not context.predator_contact_applied:
            self._apply_predator_contact(
                context.reward_components,
                context.info,
                tick_context=context,
            )
            context.predator_contact_applied = True
            return
        context.record_event(
            "predator_contact",
            "contact_check",
            predator_contact=bool(context.predator_contact_applied),
        )

    def _run_autonomic_stage(self, context: TickContext) -> None:
        """
        Resolve and apply autonomic behaviors for the current tick, update rewards and info, and record an autonomic summary event.
        
        This evaluates feeding, sleeping, and other autonomic responses using the tick's executed action, predator threat, and day/night status; it updates context.reward_components and context.info accordingly. If the spider remains in place ("STAY") while not on shelter and not on food, an idle-open penalty is applied to reward_components and logged. Finally, an event summarizing whether the agent ate, slept, the current sleep phase, and rest streak is recorded on the context.
        
        Parameters:
            context (TickContext): Per-tick mutable context containing executed_action, predator_threat, snapshot (including night flag), reward_components, info, and event recording utilities.
        """
        resolve_autonomic_behaviors(
            self,
            action_name=context.executed_action,
            predator_threat=context.predator_threat,
            night=context.snapshot.night,
            reward_components=context.reward_components,
            info=context.info,
            tick_context=context,
        )

        if (
            not self.on_shelter()
            and not context.fed_this_tick
            and not self.on_food()
            and context.executed_action == "STAY"
        ):
            context.reward_components["action_cost"] -= self.reward_config["idle_open_penalty"]
            context.record_event(
                "autonomic",
                "idle_open_penalty",
                penalty=round(float(self.reward_config["idle_open_penalty"]), 6),
            )

        context.record_event(
            "autonomic",
            "autonomic_summary",
            ate=bool(context.info["ate"]),
            slept=bool(context.info["slept"]),
            sleep_phase=self.state.sleep_phase,
            rest_streak=int(self.state.rest_streak),
        )

    def _run_predator_update_stage(self, context: TickContext) -> None:
        """
        Advance the predator one update step, record its movement/mode transition in the tick context, and apply contact effects if the predator and spider collide.
        
        Parameters:
            context (TickContext): Per-tick context that will be updated with:
                - `predator_moved` (bool): whether the predator moved this tick.
                - `info["predator_transition"]` (optional): dict with `"from"` and `"to"` modes when a mode change occurred.
                - an event entry named `"predator_update"` added via `context.record_event`.
                - `reward_components` and `info` may be mutated if predator contact occurs.
        """
        predator_mode_before = self.lizard.mode
        context.predator_moved = bool(self.predator_controller.update(self))
        predator_mode_after = self.lizard.mode
        if predator_mode_before != predator_mode_after:
            context.info["predator_transition"] = {
                "from": predator_mode_before,
                "to": predator_mode_after,
            }
        context.info["predator_moved"] = bool(context.predator_moved)
        context.record_event(
            "predator_update",
            "predator_update",
            moved=bool(context.predator_moved),
            mode_before=predator_mode_before,
            mode_after=predator_mode_after,
            transition=context.info["predator_transition"],
            lizard_pos=[int(self.lizard.x), int(self.lizard.y)],
        )
        if self.spider_pos() == self.lizard_pos() and not context.predator_contact_applied:
            self._apply_predator_contact(
                context.reward_components,
                context.info,
                tick_context=context,
            )
            context.predator_contact_applied = True

    def _run_reward_stage(self, context: TickContext) -> None:
        """
        Apply progress- and event-based rewards to the current tick context.
        
        This updates the provided TickContext's reward components and any reward-derived flags via apply_progress_and_event_rewards. The final reward-summary event is emitted later, after postprocess adds any remaining reward components.
        
        Parameters:
            context (TickContext): Per-tick context object whose reward_components and event log will be updated.
        """
        apply_progress_and_event_rewards(self, tick_context=context)

    def _run_postprocess_stage(self, context: TickContext) -> None:
        """
        Finalize per-tick bookkeeping: apply state decays and homeostasis penalties, enforce state bounds, compute and store rewards, determine episode termination, and record postprocess info.
        
        This mutates the world and tick context: it decays recent contact and pain, applies homeostasis penalties to `context.reward_components`, clips physiological state values, sets `context.done` when health is <= 0, applies a death penalty, computes `context.reward` from reward components, updates cumulative state fields (`last_reward`, `total_reward`, `steps_alive`, `last_action`), updates internal visibility/shelter flags, advances `self.tick`, and records a `"postprocess_complete"` event on the context.
        
        Parameters:
            context (TickContext): The per-tick context containing reward components, info flags, and event logging that will be updated with postprocess results.
        """
        if not context.info["predator_contact"]:
            self.state.recent_contact *= 0.35
        self.state.recent_pain *= 0.78

        apply_homeostasis_penalties(self, context.reward_components)
        clip_state(self)

        context.done = bool(self.state.health <= 0.0)
        if context.done:
            context.reward_components["death_penalty"] -= 5.0

        context.reward = float(self._reward_total(context.reward_components))
        self.state.last_reward = context.reward
        self.state.total_reward += context.reward
        self.state.steps_alive += 1
        self.state.last_action = context.executed_action
        on_shelter_now = self.on_shelter()
        self._last_on_shelter = on_shelter_now
        self._last_predator_visible = bool(context.predator_visible_now)
        self.tick += 1
        context.record_event(
            "reward",
            "reward_summary",
            predator_escape=bool(context.predator_escape),
            predator_visible_now=bool(context.predator_visible_now),
            reward=round(float(context.reward), 6),
            reward_components=self._copy_reward_components(context.reward_components),
        )
        context.record_event(
            "postprocess",
            "postprocess_complete",
            reward=round(float(context.reward), 6),
            done=bool(context.done),
            health=round(float(self.state.health), 6),
            tick=int(self.tick),
        )

    def _run_memory_stage(self, context: TickContext) -> None:
        """
        Update episodic memory slots and record a memory refresh event on the tick context.
        
        Refreshes the world's food, predator, shelter, and escape memory slots, passing the
        tick context's `predator_escape` flag to the memory update logic. Records a
        `"memory_refreshed"` event on `context` that includes whether a predator escape
        occurred and the current memory targets (each as a two-element [x, y] list or
        `None` if absent).
        
        Parameters:
            context (TickContext): Per-tick context used to read `predator_escape` and to
                record the memory refresh event.
        """
        def _target(slot: MemorySlot) -> list[int] | None:
            return list(slot.target) if slot.target is not None else None

        self.refresh_memory(predator_escape=context.predator_escape)
        context.record_event(
            "memory",
            "memory_refreshed",
            predator_escape=bool(context.predator_escape),
            food_memory_target=_target(self.state.food_memory),
            predator_memory_target=_target(self.state.predator_memory),
            shelter_memory_target=_target(self.state.shelter_memory),
            escape_memory_target=_target(self.state.escape_memory),
            tick=int(self.tick),
        )

    def _finalize_step_result(
        self,
        context: TickContext,
    ) -> tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        """
        Finalize the current tick by producing the next observation, recording a final event, and assembling the step result.
        
        Parameters:
            context (TickContext): Per-tick context containing accumulated reward, done flag, event log, and the info dictionary to populate.
        
        Returns:
            tuple:
                - next_obs (Dict[str, np.ndarray]): Observation tensors for the next state.
                - reward (float): Scalar reward for the tick.
                - done (bool): Whether the episode has terminated.
                - info (Dict[str, object]): Diagnostic dictionary including at least:
                    - "reward_components": copied components breakdown,
                    - "state": serialized spider/world state,
                    - "predator_escape": boolean flag,
                    - "event_log": serialized event log.
        """
        next_obs = self.observe()
        context.record_event(
            "finalize",
            "tick_complete",
            reward=round(float(context.reward), 6),
            done=bool(context.done),
            tick=int(self.tick),
        )
        context.info["reward_components"] = self._copy_reward_components(context.reward_components)
        context.info["state"] = self.state_dict()
        context.info["predator_escape"] = bool(context.predator_escape)
        context.info["event_log"] = context.serialized_event_log()
        return next_obs, float(context.reward), bool(context.done), context.info

    def step(self, action_idx: int) -> tuple[Dict[str, np.ndarray], float, bool, Dict[str, object]]:
        """
        Advance the simulation by one tick executing the selected action and updating world, spider, and predator state.
        
        Parameters:
            action_idx (int): Index into ACTIONS selecting the spider's action for this step.
        
        Returns:
            next_obs (Dict[str, np.ndarray]): Observation arrays for the new timestep.
            reward (float): Total scalar reward accumulated this step.
            done (bool): `True` if the episode terminated because the spider's health reached zero, `False` otherwise.
            info (Dict[str, object]): Diagnostic information including at minimum:
                - "action": selected action name
                - "ate", "slept", "pain", "predator_contact": booleans for immediate events this step
                - "predator_transition": predator mode change (or `None`)
                - "predator_moved": whether the predator moved
                - "distance_deltas": distance change diagnostics
                - "predator_escape": whether an escape event was recorded
                - "reward_components": per-component reward breakdown used to compute `reward`
                - "state": serialized snapshot of the current spider/world state
        """
        context = self._new_tick_context(action_idx)
        self._run_action_stage(context)
        self._run_terrain_and_wakefulness_stage(context)
        self._run_immediate_predator_contact_stage(context)
        self._run_autonomic_stage(context)
        self._run_predator_update_stage(context)
        self._run_reward_stage(context)
        self._run_postprocess_stage(context)
        self._run_memory_stage(context)
        return self._finalize_step_result(context)

    def state_dict(self) -> Dict[str, object]:
        """
        Serialize the full spider and environment state to a flat dictionary.
        
        The returned mapping contains the dataclass fields from the agent state plus additional runtime diagnostics and lizard/internal world context. Each memory slot entry ("food_memory", "predator_memory", "shelter_memory", "escape_memory") is augmented with a "ttl" key reflecting its time-to-live. The dictionary also includes detailed lizard debug fields (position, mode, timers, targets, and counters), the current tick and day/night flag, sleep/shelter encodings, terrain under the spider, and the active map template and reward profile.
        
        Returns:
            dict: A mapping of state and environment keys to their current values, suitable for logging, debugging, or serialization.
        """
        state = asdict(self.state)
        for key in ("food_memory", "predator_memory", "shelter_memory", "escape_memory"):
            ttl_name = key.replace("_memory", "")
            state[key]["ttl"] = MEMORY_TTLS[ttl_name]
        for key in ("food_trace", "shelter_trace", "predator_trace"):
            state[key] = self._trace_view(getattr(self.state, key))
        predator_view = predator_visible_to_spider(self, apply_noise=False)
        state.update(
            {
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
        lx, ly = self.lizard_pos()
        chars[ly][lx] = "L"
        sx, sy = self.spider_pos()
        chars[sy][sx] = "X" if (sx, sy) == (lx, ly) else "A"
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
