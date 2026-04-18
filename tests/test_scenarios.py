import unittest
from dataclasses import replace

from scenario_test_utils import ScenarioWorldHelpers
from spider_cortex_sim.scenarios import (
    BenchmarkTier,
    CAPABILITY_PROBE_SCENARIOS,
    FAST_VISUAL_HUNTER_PROFILE,
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    ProbeType,
    SCENARIO_NAMES,
    SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
    get_scenario,
)
from spider_cortex_sim.scenarios import SCENARIOS
from spider_cortex_sim.scenarios.specs import ScenarioSpec
from spider_cortex_sim.simulation import CURRICULUM_FOCUS_SCENARIOS
from spider_cortex_sim.world import ACTION_TO_INDEX


class ScenarioRegressionTest(ScenarioWorldHelpers, unittest.TestCase):

    def test_scenario_registry_exposes_expected_names(self) -> None:
        self.assertEqual(
            set(SCENARIO_NAMES),
            {
                "night_rest",
                "predator_edge",
                "entrance_ambush",
                "open_field_foraging",
                "shelter_blockade",
                "recover_after_failed_chase",
                "corridor_gauntlet",
                "two_shelter_tradeoff",
                "exposed_day_foraging",
                "food_deprivation",
                "visual_olfactory_pincer",
                "olfactory_ambush",
                "visual_hunter_open_field",
                "food_vs_predator_conflict",
                "sleep_vs_exploration_conflict",
            },
        )

    def test_night_rest_reaches_deep_sleep(self) -> None:
        world = self._setup_world("night_rest")
        phases = []
        for _ in range(3):
            _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
            phases.append(info["state"]["sleep_phase"])
        self.assertEqual(phases[-1], "DEEP_SLEEP")
        self.assertLess(world.state.sleep_debt, 0.60)

    def test_predator_edge_triggers_alert_memory(self) -> None:
        world = self._setup_world("predator_edge")
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertTrue(info["state"]["predator_memory"]["target"] is not None)
        self.assertIn(world.lizard.mode, {"ORIENT", "CHASE", "INVESTIGATE"})

    def test_entrance_ambush_keeps_lizard_waiting_outside(self) -> None:
        world = self._setup_world("entrance_ambush")
        entrance = next(iter(world.shelter_entrance_cells))
        for _ in range(3):
            world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(world.lizard.mode, "WAIT")
        self.assertEqual(world.lizard.wait_target, entrance)
        self.assertNotIn(world.lizard_pos(), world.shelter_cells)

    def test_open_field_foraging_reduces_food_distance(self) -> None:
        """
        Verify that in the "open_field_foraging" scenario the spider reduces its observed distance to the nearest food within four move steps and is not on shelter afterward.

        The test checks the scenario uses the "exposed_feeding_ground" map, records the initial observed `meta["food_dist"]`, advances the world up to four steps moving toward the first configured food position, and then asserts the observed food distance is smaller than the initial value and that the spider is not on a shelter cell.
        """
        world = self._setup_world("open_field_foraging")
        self.assertEqual(world.map_template_name, "exposed_feeding_ground")
        initial_food_dist = world.observe()["meta"]["food_dist"]
        target = world.food_positions[0]
        for _ in range(4):
            world.step(ACTION_TO_INDEX[self._move_towards(world, target)])
        self.assertLess(world.observe()["meta"]["food_dist"], initial_food_dist)
        self.assertFalse(world.on_shelter())

    def test_shelter_blockade_preserves_shelter_boundary(self) -> None:
        world = self._setup_world("shelter_blockade")
        for _ in range(4):
            world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(world.lizard.mode, "WAIT")
        self.assertNotIn(world.lizard_pos(), world.shelter_cells)
        self.assertTrue(world.inside_shelter() or world.deep_shelter())

    def test_recover_after_failed_chase_enters_recover_state(self) -> None:
        world = self._setup_world("recover_after_failed_chase")
        modes = []
        for _ in range(6):
            world.step(ACTION_TO_INDEX["STAY"])
            modes.append(world.lizard.mode)
        self.assertIn("WAIT", modes)
        self.assertIn("RECOVER", modes)

    def test_weak_scenarios_expose_diagnostic_metadata(self) -> None:
        """
        Validate that selected scenarios expose required diagnostic metadata.

        Asserts that for each of the scenarios "open_field_foraging", "corridor_gauntlet", "exposed_day_foraging", and "food_deprivation" the attributes `diagnostic_focus`, `success_interpretation`, `failure_interpretation`, and `budget_note` are truthy.
        """
        for name in (
            "open_field_foraging",
            "corridor_gauntlet",
            "exposed_day_foraging",
            "food_deprivation",
        ):
            scenario = get_scenario(name)
            self.assertTrue(scenario.diagnostic_focus)
            self.assertTrue(scenario.success_interpretation)
            self.assertTrue(scenario.failure_interpretation)
            self.assertTrue(scenario.budget_note)

    def test_scenario_probe_metadata_defaults_to_primary_emergence_gate(self) -> None:
        scenario = get_scenario("night_rest")
        self.assertEqual(scenario.probe_type, ProbeType.EMERGENCE_GATE.value)
        self.assertEqual(scenario.benchmark_tier, BenchmarkTier.PRIMARY.value)
        self.assertIsNone(scenario.target_skill)
        self.assertIsNone(scenario.geometry_assumptions)
        self.assertIsNone(scenario.acceptable_partial_progress)

    def test_scenario_probe_type_rejects_unknown_value(self) -> None:
        scenario = get_scenario("night_rest")

        with self.assertRaisesRegex(ValueError, "Invalid probe_type.*not_a_probe"):
            replace(scenario, probe_type="not_a_probe")

    def test_scenario_benchmark_tier_rejects_unknown_value(self) -> None:
        scenario = get_scenario("night_rest")

        with self.assertRaisesRegex(ValueError, "Invalid benchmark_tier.*not_a_tier"):
            replace(scenario, benchmark_tier="not_a_tier")

    def test_scenario_probe_metadata_normalizes_enum_inputs(self) -> None:
        scenario = get_scenario("food_deprivation")

        updated = replace(
            scenario,
            probe_type=ProbeType.CAPABILITY_PROBE,
            benchmark_tier=BenchmarkTier.CAPABILITY,
        )

        self.assertEqual(updated.probe_type, ProbeType.CAPABILITY_PROBE.value)
        self.assertEqual(updated.benchmark_tier, BenchmarkTier.CAPABILITY.value)

    def test_capability_probe_rejects_missing_required_metadata(self) -> None:
        scenario = get_scenario("night_rest")

        with self.assertRaisesRegex(
            ValueError,
            "Invalid capability probe metadata.*target_skill",
        ):
            replace(
                scenario,
                probe_type=ProbeType.CAPABILITY_PROBE.value,
                benchmark_tier=BenchmarkTier.CAPABILITY.value,
            )

    def test_capability_probe_rejects_blank_required_metadata(self) -> None:
        scenario = get_scenario("food_deprivation")

        for field in (
            "target_skill",
            "geometry_assumptions",
            "acceptable_partial_progress",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    f"Invalid capability probe metadata.*{field}",
                ):
                    replace(scenario, **{field: ""})

    def test_capability_probe_rejects_non_string_required_metadata(self) -> None:
        scenario = get_scenario("food_deprivation")

        for field, value in (
            ("target_skill", 123),
            ("geometry_assumptions", {}),
            ("acceptable_partial_progress", []),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ValueError,
                    f"Invalid capability probe metadata.*{field}",
                ):
                    replace(scenario, **{field: value})

    def test_capability_probe_rejects_non_capability_benchmark_tier(self) -> None:
        scenario = get_scenario("food_deprivation")

        with self.assertRaisesRegex(
            ValueError,
            "Invalid capability probe metadata.*benchmark_tier must be 'capability'",
        ):
            replace(scenario, benchmark_tier=BenchmarkTier.PRIMARY.value)

    def test_capability_probe_scenarios_expose_probe_framing(self) -> None:
        expected = (
            "open_field_foraging",
            "corridor_gauntlet",
            "exposed_day_foraging",
            "food_deprivation",
        )
        expected_target_skills = {
            "open_field_foraging": "food_vector_acquisition_exposed",
            "corridor_gauntlet": "corridor_navigation_under_threat",
            "exposed_day_foraging": "daytime_foraging_under_patrol",
            "food_deprivation": "hunger_driven_commitment",
        }

        self.assertEqual(CAPABILITY_PROBE_SCENARIOS, expected)
        for name in CAPABILITY_PROBE_SCENARIOS:
            with self.subTest(name=name):
                scenario = get_scenario(name)
                self.assertEqual(scenario.probe_type, ProbeType.CAPABILITY_PROBE.value)
                self.assertEqual(scenario.benchmark_tier, BenchmarkTier.CAPABILITY.value)
                self.assertEqual(scenario.target_skill, expected_target_skills[name])
                self.assertTrue(scenario.geometry_assumptions)
                self.assertTrue(scenario.acceptable_partial_progress)

    def test_capability_probe_scenarios_have_probe_metadata(self) -> None:
        for name in CAPABILITY_PROBE_SCENARIOS:
            with self.subTest(name=name):
                scenario = get_scenario(name)
                self.assertEqual(scenario.probe_type, ProbeType.CAPABILITY_PROBE.value)
                self.assertTrue(scenario.target_skill)
                self.assertTrue(scenario.geometry_assumptions)
                self.assertTrue(scenario.acceptable_partial_progress)

    def test_capability_probes_have_benchmark_tier_capability(self) -> None:
        for name in CAPABILITY_PROBE_SCENARIOS:
            with self.subTest(name=name):
                self.assertEqual(
                    get_scenario(name).benchmark_tier,
                    BenchmarkTier.CAPABILITY.value,
                )

    def test_capability_probe_scenarios_match_curriculum_focus(self) -> None:
        self.assertEqual(
            set(CAPABILITY_PROBE_SCENARIOS),
            set(CURRICULUM_FOCUS_SCENARIOS),
        )

    def test_exposed_day_foraging_metadata_documents_geometry_hypothesis(self) -> None:
        scenario = get_scenario("exposed_day_foraging")
        self.assertIn("Hypothesis: previous geometry placed lizard on food", scenario.diagnostic_focus)
        self.assertIn("failure_mode", scenario.failure_interpretation)
        self.assertIn("heuristic lizard-food/frontier spacing", scenario.failure_interpretation)
        self.assertIn("Geometry fix", scenario.budget_note)
        self.assertIn("threat-radius heuristic", scenario.budget_note)
        self.assertIn("first food-visible frontier", scenario.budget_note)
        self.assertIn("arbitration or training limits", scenario.budget_note)

    def test_corridor_gauntlet_metadata_documents_failure_modes(self) -> None:
        scenario = get_scenario("corridor_gauntlet")
        self.assertIn("failure_mode classification", scenario.diagnostic_focus)
        self.assertIn("positive food-distance progress", scenario.success_interpretation)
        self.assertIn("zero predator contacts", scenario.success_interpretation)
        self.assertIn("survival to episode end", scenario.success_interpretation)
        for failure_mode in (
            '"success"',
            '"frozen_in_shelter"',
            '"contact_failure_died"',
            '"contact_failure_survived"',
            '"survived_no_progress"',
            '"progress_then_died"',
            '"scoring_mismatch"',
        ):
            with self.subTest(failure_mode=failure_mode):
                self.assertIn(failure_mode, scenario.failure_interpretation)

    def test_food_deprivation_metadata_documents_timing_hypothesis(self) -> None:
        """
        Assert that the `food_deprivation` scenario's metadata documents the expected timing hypothesis and related diagnostic phrases.

        Verifies that diagnostic metadata separates commitment, timing, and orientation
        failures while preserving existing failure-mode and budget interpretation.
        """
        scenario = get_scenario("food_deprivation")
        self.assertIn("hunger-driven commitment failures", scenario.diagnostic_focus)
        self.assertIn("timing failures", scenario.diagnostic_focus)
        self.assertIn("orientation failures", scenario.diagnostic_focus)
        self.assertIn("failure_mode", scenario.failure_interpretation)
        self.assertIn("geometric race condition", scenario.budget_note)

class ConflictScenarioMetadataTest(unittest.TestCase):
    """Tests for the new conflict scenarios added in this PR."""

    def test_scenario_metadata_and_behavior_checks(self) -> None:
        cases = [
            ("food_vs_predator_conflict", "exposed_feeding_ground"),
            ("sleep_vs_exploration_conflict", "central_burrow"),
        ]
        for name, expected_map in cases:
            with self.subTest(scenario=name):
                scenario = get_scenario(name)
                self.assertTrue(scenario.diagnostic_focus)
                self.assertTrue(scenario.success_interpretation)
                self.assertTrue(scenario.failure_interpretation)
                self.assertTrue(scenario.budget_note)
                self.assertEqual(scenario.map_template, expected_map)
                self.assertGreater(scenario.max_steps, 0)
                self.assertLessEqual(scenario.max_steps, 24)
                self.assertEqual(len(scenario.behavior_checks), 3)

class MultiPredatorScenarioMetadataTest(unittest.TestCase):
    """Tests for the new predator-specialization scenarios."""

    def test_scenario_metadata_and_behavior_checks(self) -> None:
        cases = [
            ("visual_olfactory_pincer", "exposed_feeding_ground"),
            ("olfactory_ambush", "entrance_funnel"),
            ("visual_hunter_open_field", "exposed_feeding_ground"),
        ]
        for name, expected_map in cases:
            with self.subTest(scenario=name):
                scenario = get_scenario(name)
                self.assertTrue(scenario.diagnostic_focus)
                self.assertTrue(scenario.success_interpretation)
                self.assertTrue(scenario.failure_interpretation)
                self.assertTrue(scenario.budget_note)
                self.assertEqual(scenario.map_template, expected_map)
                self.assertGreater(scenario.max_steps, 0)
                self.assertEqual(len(scenario.behavior_checks), 3)


class GetScenarioTest(unittest.TestCase):
    """Tests for get_scenario() function from the new __init__.py module."""

    def test_known_scenario_returns_spec(self) -> None:
        """get_scenario returns a ScenarioSpec for a known name."""
        spec = get_scenario("night_rest")
        self.assertIsInstance(spec, ScenarioSpec)

    def test_unknown_scenario_raises_value_error(self) -> None:
        """get_scenario raises ValueError for an unrecognized scenario name."""
        with self.assertRaises(ValueError):
            get_scenario("not_a_real_scenario_name_xyz")

    def test_error_message_includes_scenario_name(self) -> None:
        """ValueError message includes the bad scenario name."""
        bad_name = "completely_unknown_scenario"
        with self.assertRaisesRegex(ValueError, bad_name):
            get_scenario(bad_name)

    def test_all_scenario_names_retrievable(self) -> None:
        """Every name in SCENARIO_NAMES can be retrieved via get_scenario."""
        for name in SCENARIO_NAMES:
            with self.subTest(name=name):
                spec = get_scenario(name)
                self.assertEqual(spec.name, name)

    def test_returns_same_object_as_dict(self) -> None:
        """get_scenario returns the same ScenarioSpec as SCENARIOS[name]."""
        for name in SCENARIO_NAMES:
            with self.subTest(name=name):
                self.assertIs(get_scenario(name), SCENARIOS[name])

    def test_empty_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_scenario("")

    def test_case_sensitive_lookup(self) -> None:
        """get_scenario is case-sensitive."""
        with self.assertRaises(ValueError):
            get_scenario("Night_Rest")


class ScenariosInitExportsTest(unittest.TestCase):
    """Tests that the __init__.py module correctly exports constants and types."""

    def test_fast_visual_hunter_profile_exported(self) -> None:
        """FAST_VISUAL_HUNTER_PROFILE is accessible from the scenarios package."""
        self.assertIsNotNone(FAST_VISUAL_HUNTER_PROFILE)
        self.assertEqual(FAST_VISUAL_HUNTER_PROFILE.name, "fast_visual_hunter")
        self.assertEqual(FAST_VISUAL_HUNTER_PROFILE.detection_style, "visual")

    def test_fast_visual_hunter_has_faster_move_interval(self) -> None:
        """FAST_VISUAL_HUNTER_PROFILE has move_interval=1."""
        self.assertEqual(FAST_VISUAL_HUNTER_PROFILE.move_interval, 1)

    def test_food_deprivation_initial_hunger_exported(self) -> None:
        """FOOD_DEPRIVATION_INITIAL_HUNGER is accessible from the scenarios package."""
        self.assertAlmostEqual(FOOD_DEPRIVATION_INITIAL_HUNGER, 0.96)

    def test_night_rest_initial_sleep_debt_exported(self) -> None:
        """NIGHT_REST_INITIAL_SLEEP_DEBT is accessible from the scenarios package."""
        self.assertAlmostEqual(NIGHT_REST_INITIAL_SLEEP_DEBT, 0.60)

    def test_sleep_vs_exploration_initial_sleep_debt_exported(self) -> None:
        """SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT is accessible from the scenarios package."""
        self.assertAlmostEqual(SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT, 0.92)

    def test_scenarios_dict_is_non_empty(self) -> None:
        """SCENARIOS dict has at least one entry."""
        self.assertGreater(len(SCENARIOS), 0)

    def test_scenario_names_match_scenarios_dict_keys(self) -> None:
        """SCENARIO_NAMES is equal to tuple of SCENARIOS.keys()."""
        self.assertEqual(SCENARIO_NAMES, tuple(SCENARIOS.keys()))

    def test_all_scenarios_are_spec_instances(self) -> None:
        """Every value in SCENARIOS is a ScenarioSpec instance."""
        for name, spec in SCENARIOS.items():
            with self.subTest(name=name):
                self.assertIsInstance(spec, ScenarioSpec)

    def test_capability_probe_scenarios_is_subset_of_scenario_names(self) -> None:
        """Every capability probe scenario is in SCENARIO_NAMES."""
        for name in CAPABILITY_PROBE_SCENARIOS:
            self.assertIn(name, SCENARIO_NAMES)

    def test_probe_type_enum_values_accessible(self) -> None:
        """ProbeType enum values work correctly."""
        self.assertEqual(ProbeType.EMERGENCE_GATE.value, "emergence_gate")
        self.assertEqual(ProbeType.CAPABILITY_PROBE.value, "capability_probe")
        self.assertEqual(ProbeType.CURRICULUM_FOCUS.value, "curriculum_focus")

    def test_benchmark_tier_enum_values_accessible(self) -> None:
        """BenchmarkTier enum values work correctly."""
        self.assertEqual(BenchmarkTier.PRIMARY.value, "primary")
        self.assertEqual(BenchmarkTier.CAPABILITY.value, "capability")
        self.assertEqual(BenchmarkTier.DIAGNOSTIC.value, "diagnostic")
