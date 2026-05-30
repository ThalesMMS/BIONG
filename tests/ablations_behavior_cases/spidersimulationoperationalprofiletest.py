from __future__ import annotations

from .shared import *


class SpiderSimulationOperationalProfileTest(unittest.TestCase):
    """Tests for operational_profile propagation through SpiderSimulation."""

    def test_simulation_stores_resolved_operational_profile(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        self.assertEqual(sim.operational_profile.name, "default_v1")

    def test_simulation_accepts_profile_by_name(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile="default_v1")
        self.assertEqual(sim.operational_profile.name, "default_v1")

    def test_simulation_accepts_profile_instance(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.operational_profile, profile)

    def test_simulation_propagates_profile_to_world(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.world.operational_profile, profile)

    def test_simulation_propagates_profile_to_brain(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.brain.operational_profile, profile)

    def test_summary_operational_profile_version_is_int(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIsInstance(summary["config"]["operational_profile"]["version"], int)

    def test_annotate_behavior_rows_with_custom_profile(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        rows = [{"scenario": "night_rest"}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["operational_profile"], "ablation_test_profile")
        self.assertEqual(annotated[0]["operational_profile_version"], 21)
        self.assertEqual(annotated[0]["noise_profile"], "none")
