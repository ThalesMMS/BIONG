"""Compatibility tests for reward package-level exports."""

from __future__ import annotations

import unittest

import spider_cortex_sim.reward as reward


EXPECTED_REWARD_EXPORTS = [
    "DISPOSITION_EVIDENCE_CRITERIA",
    "DispositionEvidenceEntry",
    "DispositionName",
    "MINIMAL_SHAPING_SURVIVAL_THRESHOLD",
    "REWARD_COMPONENT_AUDIT",
    "REWARD_COMPONENT_NAMES",
    "REWARD_PROFILES",
    "ReductionPriority",
    "SCENARIO_AUSTERE_REQUIREMENTS",
    "SHAPING_DISPOSITIONS",
    "SHAPING_GAP_POLICY",
    "SHAPING_REDUCTION_ROADMAP",
    "ScenarioAustereRequirement",
    "ScenarioRequirementLevel",
    "ShapingGapPolicy",
    "ShapingRoadmapEntry",
    "apply_action_and_terrain_effects",
    "apply_pressure_penalties",
    "apply_progress_and_event_rewards",
    "compute_predator_threat",
    "copy_reward_components",
    "empty_reward_components",
    "reward_component_audit",
    "reward_profile_audit",
    "reward_total",
    "shaping_disposition_summary",
    "shaping_reduction_roadmap",
    "validate_gap_policy",
    "validate_shaping_disposition",
]


class RewardPackageExportsTest(unittest.TestCase):
    def test_reward_package_exports_public_api(self) -> None:
        self.assertEqual(list(reward.__all__), EXPECTED_REWARD_EXPORTS)
        for name in EXPECTED_REWARD_EXPORTS:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(reward, name))


if __name__ == "__main__":
    unittest.main()
