import unittest
from collections import deque
from collections.abc import Mapping, Sequence
from unittest.mock import patch

import numpy as np

from spider_cortex_sim import stages as tick_stages
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_DELTAS,
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    ORIENT_HEADINGS,
    ActionContextObservation,
    MotorContextObservation,
)
from spider_cortex_sim.maps import MAP_TEMPLATE_NAMES, build_map_template
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.predator import (
    OLFACTORY_HUNTER_PROFILE,
    VISUAL_HUNTER_PROFILE,
    LizardState,
)
from spider_cortex_sim.world_types import PerceptTrace, TickContext
from spider_cortex_sim.world import (
    ACTION_TO_INDEX,
    MOVE_DELTAS,
    REWARD_COMPONENT_NAMES,
    PerceptualBuffer,
    SpiderWorld,
    _copy_observation_payload,
    _is_temporal_direction_field,
    _refresh_perception_for_active_scan,
    _scan_age_for_heading,
)

def _profile_with_perception(**perception_overrides: float) -> OperationalProfile:
    """
    Return the default OperationalProfile with validated perception overrides.
    Raises ValueError for unknown perception keys.
    """
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    perception = summary["perception"]
    allowed_keys = set(perception.keys())
    unknown_keys = sorted(set(perception_overrides) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown perception override keys: {unknown_keys}")
    perception.update(perception_overrides)
    return OperationalProfile.from_summary(summary)

class SpiderWorldTestBase(unittest.TestCase):
    def _deep_cell(self, world: SpiderWorld) -> tuple[int, int]:
        return sorted(world.shelter_deep_cells)[len(world.shelter_deep_cells) // 2]

    def _interior_cell(self, world: SpiderWorld) -> tuple[int, int]:
        return sorted(world.shelter_interior_cells)[len(world.shelter_interior_cells) // 2]

    def _entrance_cell(self, world: SpiderWorld) -> tuple[int, int]:
        """
        Return the median cell (by sorted order) from the world's shelter entrance cells.
        
        Returns:
            (int, int): The (x, y) coordinate tuple of the median entrance cell.
        """
        return sorted(world.shelter_entrance_cells)[len(world.shelter_entrance_cells) // 2]

    def _outside_entrance_cell(self, world: SpiderWorld) -> tuple[int, int]:
        """
        Return a lizard-walkable cell next to the entrance.
        Falls back to the first lizard spawn cell if none is adjacent.
        """
        entrance = self._entrance_cell(world)
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            candidate = (entrance[0] + dx, entrance[1] + dy)
            if not (0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height):
                continue
            if world.is_lizard_walkable(candidate):
                return candidate
        return sorted(world.map_template.lizard_spawn_cells)[0]

    def _move_lizard_to_safe_corner(self, world: SpiderWorld) -> None:
        """
        Move the lizard to the spawn cell farthest from the spider.
        """
        candidates = sorted(
            world.map_template.lizard_spawn_cells,
            key=lambda cell: -world.manhattan(cell, world.spider_pos()),
        )
        world.lizard.x, world.lizard.y = candidates[0]

    def _assert_reward_components(self, reward: float, info: dict[str, object]) -> None:
        reward_components = info["reward_components"]
        self.assertEqual(set(reward_components.keys()), set(REWARD_COMPONENT_NAMES))
        self.assertAlmostEqual(sum(reward_components.values()), reward)

    def _reflex_brain(self, seed: int = 0) -> SpiderBrain:
        """
        Return a deterministic reflex brain with learned weights neutralized.
        Uses the supplied seed for reproducible initialization.
        """
        brain = SpiderBrain(seed=seed, module_dropout=0.0)
        brain.action_center.W1.fill(0.0)
        brain.action_center.b1.fill(0.0)
        brain.action_center.W2_policy.fill(0.0)
        brain.action_center.b2_policy.fill(0.0)
        brain.action_center.W2_value.fill(0.0)
        brain.action_center.b2_value.fill(0.0)
        brain.motor_cortex.W1.fill(0.0)
        brain.motor_cortex.b1.fill(0.0)
        brain.motor_cortex.W2.fill(0.0)
        brain.motor_cortex.b2.fill(0.0)
        if brain.module_bank is not None:
            for network in brain.module_bank.modules.values():
                network.W1.fill(0.0)
                network.b1.fill(0.0)
                network.W2.fill(0.0)
                network.b2.fill(0.0)
        return brain

    def _reachable(self, world: SpiderWorld, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        """
        Determine whether the goal cell can be reached from the start cell within the world's walkable area.
        
        Parameters:
        	world (SpiderWorld): The world providing dimensions, walkability checks, and movement deltas.
        	start (tuple[int, int]): Starting (x, y) coordinates.
        	goal (tuple[int, int]): Target (x, y) coordinates.
        
        Returns:
        	`true` if the goal cell can be reached from start by traversing walkable cells, `false` otherwise.
        """
        queue = deque([start])
        seen = {start}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                return True
            for dx, dy in world.move_deltas:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nxt[0] < world.width and 0 <= nxt[1] < world.height):
                    continue
                if nxt in seen or not world.is_walkable(nxt):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False
