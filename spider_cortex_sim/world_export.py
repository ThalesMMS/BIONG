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


class WorldExportMixin:
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
