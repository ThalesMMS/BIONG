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


class WorldDynamicsMixin:
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
