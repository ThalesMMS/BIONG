"""Focused reward tests grouped by reward subpackage responsibility."""

from __future__ import annotations

import math
import unittest

from spider_cortex_sim.claim_tests import canonical_claim_tests
from spider_cortex_sim.maps import NARROW
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.reward.audit import (
    REWARD_COMPONENT_AUDIT,
    _roadmap_status_for_profile,
    reward_component_audit,
    reward_profile_audit,
    shaping_disposition_summary,
)
from spider_cortex_sim.reward.computation import (
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_total,
)
from spider_cortex_sim.reward.profiles import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_COMPONENT_NAMES,
    REWARD_PROFILES,
)
from spider_cortex_sim.reward.shaping import (
    DISPOSITION_EVIDENCE_CRITERIA,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_DISPOSITIONS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    shaping_reduction_roadmap,
    validate_gap_policy,
)
from spider_cortex_sim.scenarios import CAPABILITY_PROBE_SCENARIOS, SCENARIOS
from spider_cortex_sim.world import SpiderWorld
from spider_cortex_sim.world_types import TickContext, TickSnapshot

class RewardComputationModuleTestBase(unittest.TestCase):
    def _profile_with_reward_updates(self, **updates: float) -> OperationalProfile:
        """
        Create an OperationalProfile derived from DEFAULT_OPERATIONAL_PROFILE with specified reward fields overridden.
        
        Parameters:
            **updates: float
                Reward field names and their new numeric values to apply to the profile's `reward` section. Keys must be valid reward keys present in the default profile; values will be cast to `float`.
        
        Returns:
            OperationalProfile: An OperationalProfile constructed from the modified summary with the provided reward updates applied.
        
        Raises:
            ValueError: If any provided update key is not a valid reward key; the exception message lists the unknown keys and the allowed keys.
        """
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "reward_test_profile"
        summary["version"] = 7
        allowed_keys = set(summary["reward"].keys())
        invalid_keys = sorted(set(updates) - allowed_keys)
        if invalid_keys:
            raise ValueError(
                f"Unknown reward profile field(s): {', '.join(invalid_keys)}. "
                f"Allowed keys: {', '.join(sorted(allowed_keys))}."
            )
        summary["reward"].update({name: float(value) for name, value in updates.items()})
        return OperationalProfile.from_summary(summary)

    def _tick_context(
        self,
        world: SpiderWorld,
        *,
        action_name: str,
        moved: bool,
        night: bool,
        terrain_now: str | None = None,
        was_on_shelter: bool,
        prev_food_dist: int,
        prev_shelter_dist: int,
        prev_predator_dist: int,
        prev_predator_visible: bool,
        prev_spider_pos: tuple[int, int],
        prev_lizard_pos: tuple[int, int],
        prev_rest_streak: int | None = None,
    ) -> TickContext:
        """
        Builds a TickContext for unit tests using supplied previous-tick state values and sensible defaults.
        
        Parameters:
            world (SpiderWorld): World instance used to read derived values (e.g., shelter role, tick, and state).
            action_name (str): Name of the intended and executed action for the tick.
            moved (bool): Whether the action resulted in movement.
            night (bool): Whether the tick occurs during night.
            terrain_now (str | None): Current terrain type at the spider's position; when None, derived from the world at prev_spider_pos.
            was_on_shelter (bool): Whether the spider was on shelter in the previous tick.
            prev_food_dist (int): Previous tick's Manhattan distance to the nearest food.
            prev_shelter_dist (int): Previous tick's Manhattan distance to the nearest shelter.
            prev_predator_dist (int): Previous tick's estimated distance to the predator.
            prev_predator_visible (bool): Whether the predator was visible in the previous tick.
            prev_spider_pos (tuple[int, int]): Spider position in the previous tick.
            prev_lizard_pos (tuple[int, int]): Lizard position in the previous tick.
            prev_rest_streak (int | None): Optional previous rest streak; if None, uses world.state.rest_streak.
        
        Returns:
            TickContext: A TickContext populated with the provided snapshot, zeroed reward components, an empty distance_deltas info entry, and the given movement/terrain flags.
        """
        terrain_now = (
            terrain_now
            if terrain_now is not None
            else world.terrain_at(prev_spider_pos)
        )
        return TickContext(
            action_idx=0,
            intended_action=action_name,
            executed_action=action_name,
            motor_noise_applied=False,
            snapshot=TickSnapshot(
                tick=int(world.tick),
                spider_pos=prev_spider_pos,
                lizard_pos=prev_lizard_pos,
                was_on_shelter=bool(was_on_shelter),
                prev_shelter_role=world.shelter_role_at(prev_spider_pos),
                prev_food_dist=int(prev_food_dist),
                prev_shelter_dist=int(prev_shelter_dist),
                prev_predator_dist=int(prev_predator_dist),
                prev_predator_visible=bool(prev_predator_visible),
                night=bool(night),
                rest_streak=int(world.state.rest_streak if prev_rest_streak is None else prev_rest_streak),
            ),
            reward_components=empty_reward_components(),
            info={"distance_deltas": {}},
            moved=bool(moved),
            terrain_now=terrain_now,
        )
