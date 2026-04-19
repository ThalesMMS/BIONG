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


class WorldPerceptionMixin:
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
