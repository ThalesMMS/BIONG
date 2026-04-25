from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import (
    A4_FINE_MODULES,
    BrainAblationConfig,
    COARSE_ROLLUP_MODULES,
    MODULE_NAMES,
    MULTI_PREDATOR_SCENARIOS,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    VISUAL_PREDATOR_SCENARIOS,
    OLFACTORY_PREDATOR_SCENARIOS,
    MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    canonical_ablation_scenario_groups,
    compare_predator_type_ablation_performance,
    canonical_ablation_variant_names,
    default_brain_config,
    resolve_ablation_scenario_group,
    resolve_ablation_configs,
    _safe_float,
    _mean,
    _scenario_success_rate,
)
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import RecurrentProposalNetwork
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.simulation import SpiderSimulation

from tests.fixtures.ablations import _blank_mapping, _build_observation, _profile_with_updates

class AnnotateBehaviorRowsTest(unittest.TestCase):
    """Tests for SpiderSimulation._annotate_behavior_rows (new in this PR)."""

    def test_annotate_adds_ablation_variant(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertIn("ablation_variant", annotated[0])

    def test_annotate_adds_ablation_architecture(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertIn("ablation_architecture", annotated[0])

    def test_annotate_adds_operational_profile_metadata(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["operational_profile"], "default_v1")
        self.assertEqual(annotated[0]["operational_profile_version"], 1)
        self.assertEqual(annotated[0]["noise_profile"], "none")
        self.assertIn("noise_profile_config", annotated[0])

    def test_annotate_variant_matches_config_name(self) -> None:
        config = BrainAblationConfig(name="drop_sleep_center", architecture="modular", module_dropout=0.0, disabled_modules=("sleep_center",))
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        rows = [{"scenario": "night_rest", "success": 0}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["ablation_variant"], "drop_sleep_center")

    def test_annotate_architecture_matches_config(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        rows = [{"scenario": "night_rest", "success": 0}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["ablation_architecture"], "monolithic")

    def test_annotate_preserves_original_row_content(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1, "failures": []}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["scenario"], "night_rest")
        self.assertEqual(annotated[0]["success"], 1)

    def test_annotate_does_not_mutate_original_rows(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest"}]
        sim._annotate_behavior_rows(rows)
        self.assertNotIn("ablation_variant", rows[0])

    def test_annotate_empty_rows_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        annotated = sim._annotate_behavior_rows([])
        self.assertEqual(annotated, [])

    def test_annotate_multiple_rows(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest"}, {"scenario": "predator_edge"}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(len(annotated), 2)
        for row in annotated:
            self.assertIn("ablation_variant", row)

class BuildSummaryBrainConfigTest(unittest.TestCase):
    """Tests that _build_summary now includes brain config info."""

    def test_summary_config_has_brain_key(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("brain", summary["config"])

    def test_summary_brain_config_architecture_matches(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["architecture"], "modular")

    def test_summary_config_has_operational_profile(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("operational_profile", summary["config"])
        self.assertEqual(summary["config"]["operational_profile"]["name"], "default_v1")
        self.assertIn("version", summary["config"]["operational_profile"])
        self.assertEqual(summary["config"]["operational_profile"]["version"], 1)
        self.assertIn("noise_profile", summary["config"])
        self.assertEqual(summary["config"]["noise_profile"]["name"], "none")

    def test_summary_brain_config_name_matches(self) -> None:
        """
        Verify that SpiderSimulation._build_summary includes the default brain config name "modular_full".
        """
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["name"], "modular_full")

    def test_summary_has_architecture_traceability(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["architecture_version"], SpiderBrain.ARCHITECTURE_VERSION)
        self.assertTrue(summary["config"]["architecture_fingerprint"])

    def test_summary_includes_reward_audit(self) -> None:
        """
        Verifies that SpiderSimulation._build_summary includes a reward_audit section with expected keys and values.
        
        Asserts that the summary contains a top-level "reward_audit" entry, that its "current_profile" is "classic", that "predator_dist" appears in "observation_signals", and that "austere" is listed among "reward_profiles".
        """
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("reward_audit", summary)
        self.assertEqual(summary["reward_audit"]["current_profile"], "classic")
        self.assertIn("predator_dist", summary["reward_audit"]["observation_signals"])
        self.assertIn("austere", summary["reward_audit"]["reward_profiles"])
        self.assertIn("comparison", summary["reward_audit"])
        self.assertIn("classic", summary["reward_audit"]["comparison"]["profiles"])
        self.assertIn("shaping_reduction_status", summary)
        self.assertEqual(
            summary["shaping_reduction_status"]["roadmap_target_count"],
            len(
                summary["reward_audit"]["reward_profiles"]["classic"][
                    "reduction_roadmap_status"
                ]
            ),
        )

    def test_summary_monolithic_brain_config_in_summary(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["architecture"], "monolithic")

class LocalCreditOnlyVariantTest(unittest.TestCase):
    """Tests for the local_credit_only ablation variant (new in this PR)."""

    def test_uses_local_credit_only_is_false_for_modular_full(self) -> None:
        config = BrainAblationConfig(name="modular_full")
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_no_module_dropout(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", module_dropout=0.0)
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_no_module_reflexes(self) -> None:
        config = BrainAblationConfig(name="no_module_reflexes", enable_reflexes=False)
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_monolithic_policy(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic")
        self.assertFalse(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_true_only_for_local_only_strategy(self) -> None:
        config = BrainAblationConfig(
            name="custom_local_credit",
            credit_strategy="local_only",
        )
        self.assertTrue(config.uses_local_credit_only)

    def test_uses_local_credit_only_is_false_for_drop_variants(self) -> None:
        for module_name in MODULE_NAMES:
            config = BrainAblationConfig(
                name=f"drop_{module_name}",
                disabled_modules=(module_name,),
            )
            self.assertFalse(
                config.uses_local_credit_only,
                f"Expected drop_{module_name} to have uses_local_credit_only=False",
            )

    def test_local_credit_only_is_modular_not_monolithic(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertTrue(lc.is_modular)
        self.assertFalse(lc.is_monolithic)

    def test_local_credit_only_has_reflexes_and_aux_targets_enabled(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertTrue(lc.enable_reflexes)
        self.assertTrue(lc.enable_auxiliary_targets)

    def test_local_credit_only_uses_a4_fine_topology(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertEqual(set(lc.disabled_modules), set(COARSE_ROLLUP_MODULES))

    def test_local_credit_only_has_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertAlmostEqual(lc.reflex_scale, 1.0)

    def test_local_credit_only_uses_local_only_strategy(self) -> None:
        configs = canonical_ablation_configs()
        lc = configs["local_credit_only"]
        self.assertEqual(lc.credit_strategy, "local_only")
        self.assertTrue(lc.uses_local_credit_only)
        self.assertFalse(lc.uses_counterfactual_credit)

    def test_local_credit_only_appears_after_no_module_reflexes_in_canonical_order(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertIn("local_credit_only", names)
        self.assertIn("no_module_reflexes", names)
        idx_lc = names.index("local_credit_only")
        idx_nr = names.index("no_module_reflexes")
        self.assertGreater(
            idx_lc,
            idx_nr,
            "local_credit_only should appear after no_module_reflexes in canonical order",
        )

    def test_local_credit_only_appears_before_monolithic_policy_in_canonical_order(self) -> None:
        """
        Assert that the canonical ablation variant ordering lists "local_credit_only" before "monolithic_policy".
        
        Verifies the relative positions of these two canonical variant names returned by canonical_ablation_variant_names().
        """
        names = canonical_ablation_variant_names()
        idx_lc = names.index("local_credit_only")
        idx_mono = names.index("monolithic_policy")
        self.assertLess(
            idx_lc,
            idx_mono,
            "local_credit_only should appear before monolithic_policy in canonical order",
        )

    def test_uses_local_credit_only_is_not_tied_to_variant_name(self) -> None:
        config = BrainAblationConfig(
            name="local_credit_only",
            architecture="modular",
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=(),
            reflex_scale=1.0,
            module_reflex_scales={},
        )
        self.assertFalse(config.uses_local_credit_only)

    def test_resolve_ablation_configs_returns_local_credit_only_by_name(self) -> None:
        configs = resolve_ablation_configs(["local_credit_only"])
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "local_credit_only")
        self.assertTrue(configs[0].uses_local_credit_only)

    def test_local_credit_only_to_summary_includes_name(self) -> None:
        configs = canonical_ablation_configs()
        summary = configs["local_credit_only"].to_summary()
        self.assertEqual(summary["name"], "local_credit_only")
        self.assertEqual(summary["architecture"], "modular")
        self.assertEqual(summary["credit_strategy"], "local_only")

    def test_local_credit_only_custom_dropout_propagates(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.12)
        lc = configs["local_credit_only"]
        self.assertAlmostEqual(lc.module_dropout, 0.12)

class CanonicalModularFullLearnedArbitrationTest(unittest.TestCase):
    """Tests that the modular_full canonical config explicitly enables learned arbitration."""

    def test_modular_full_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        mf = configs["modular_full"]
        self.assertTrue(mf.use_learned_arbitration)

    def test_no_module_dropout_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        self.assertTrue(configs["no_module_dropout"].use_learned_arbitration)

    def test_no_module_reflexes_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        self.assertTrue(configs["no_module_reflexes"].use_learned_arbitration)

    def test_fixed_arbitration_baseline_has_use_learned_arbitration_false(self) -> None:
        configs = canonical_ablation_configs()
        self.assertFalse(configs["fixed_arbitration_baseline"].use_learned_arbitration)

    def test_learned_arbitration_no_regularization_has_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        self.assertTrue(configs["learned_arbitration_no_regularization"].use_learned_arbitration)

    def test_drop_variants_have_use_learned_arbitration_true(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in A4_FINE_MODULES:
            config = configs[f"drop_{module_name}"]
            self.assertTrue(
                config.use_learned_arbitration,
                f"drop_{module_name} should have use_learned_arbitration=True",
            )

    def test_to_summary_for_modular_full_includes_use_learned_arbitration(self) -> None:
        configs = canonical_ablation_configs()
        summary = configs["modular_full"].to_summary()
        self.assertIn("use_learned_arbitration", summary)
        self.assertTrue(summary["use_learned_arbitration"])

    def test_to_summary_for_fixed_baseline_has_false(self) -> None:
        configs = canonical_ablation_configs()
        summary = configs["fixed_arbitration_baseline"].to_summary()
        self.assertFalse(summary["use_learned_arbitration"])

class CreditStrategyLearnDiagnosticsTest(unittest.TestCase):
    """Tests verifying learn() diagnostics for the new credit-strategy fields."""

    @staticmethod
    def _build_obs() -> dict[str, np.ndarray]:
        return _build_observation()

    def test_learn_stats_sorted_credit_weights_keys(self) -> None:
        """module_credit_weights returned by learn() are sorted by module name."""
        brain = SpiderBrain(
            seed=1,
            config=BrainAblationConfig(
                name="test",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy="broadcast",
            ),
        )
        obs = self._build_obs()
        # sample=True so training mode is active and networks cache state for backward
        step = brain.act(obs, sample=True)
        stats = brain.learn(step, reward=0.0, next_observation=obs, done=True)

        keys = list(stats["module_credit_weights"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_learn_stats_sorted_gradient_norm_keys(self) -> None:
        """module_gradient_norms returned by learn() are sorted by module name."""
        brain = SpiderBrain(
            seed=2,
            config=BrainAblationConfig(
                name="test",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy="broadcast",
            ),
        )
        obs = self._build_obs()
        # sample=True so training mode is active and networks cache state for backward
        step = brain.act(obs, sample=True)
        stats = brain.learn(step, reward=0.0, next_observation=obs, done=True)

        keys = list(stats["module_gradient_norms"].keys())
        self.assertEqual(keys, sorted(keys))

    def test_learn_raises_value_error_for_reflex_only_policy_mode(self) -> None:
        """learn() raises ValueError when the decision has policy_mode != 'normal'."""
        brain = SpiderBrain(
            seed=3,
            config=BrainAblationConfig(
                name="test",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=True,
                enable_auxiliary_targets=False,
                credit_strategy="broadcast",
            ),
        )
        obs = self._build_obs()
        step = brain.act(obs, sample=False, policy_mode="normal")
        step.policy_mode = "reflex_only"

        with self.assertRaises(ValueError):
            brain.learn(step, reward=0.0, next_observation=obs, done=True)

class SafeFloatTest(unittest.TestCase):
    """Tests for ablations._safe_float() - new in this PR."""

    def test_valid_int_converts_to_float(self) -> None:
        self.assertEqual(_safe_float(3), 3.0)

    def test_valid_float_returns_float(self) -> None:
        self.assertAlmostEqual(_safe_float(0.75), 0.75)

    def test_string_numeric_converts(self) -> None:
        self.assertAlmostEqual(_safe_float("0.5"), 0.5)

    def test_none_returns_zero(self) -> None:
        self.assertEqual(_safe_float(None), 0.0)

    def test_nan_returns_zero(self) -> None:
        import math
        self.assertEqual(_safe_float(float("nan")), 0.0)

    def test_positive_infinity_returns_zero(self) -> None:
        self.assertEqual(_safe_float(float("inf")), 0.0)

    def test_negative_infinity_returns_zero(self) -> None:
        self.assertEqual(_safe_float(float("-inf")), 0.0)

    def test_non_numeric_string_returns_zero(self) -> None:
        self.assertEqual(_safe_float("not_a_number"), 0.0)

    def test_empty_string_returns_zero(self) -> None:
        self.assertEqual(_safe_float(""), 0.0)

    def test_zero_returns_zero_float(self) -> None:
        self.assertEqual(_safe_float(0), 0.0)

    def test_negative_finite_value_returns_float(self) -> None:
        self.assertAlmostEqual(_safe_float(-1.5), -1.5)

    def test_bool_true_converts_to_one(self) -> None:
        self.assertAlmostEqual(_safe_float(True), 1.0)

    def test_bool_false_converts_to_zero(self) -> None:
        self.assertEqual(_safe_float(False), 0.0)

class AblationMeanTest(unittest.TestCase):
    """Tests for ablations._mean() - new in this PR."""

    def test_empty_sequence_returns_zero(self) -> None:
        self.assertEqual(_mean([]), 0.0)

    def test_single_element(self) -> None:
        self.assertAlmostEqual(_mean([0.7]), 0.7)

    def test_multiple_elements(self) -> None:
        self.assertAlmostEqual(_mean([0.0, 1.0, 0.5]), 0.5)

    def test_all_zeros(self) -> None:
        self.assertAlmostEqual(_mean([0.0, 0.0, 0.0]), 0.0)

    def test_all_ones(self) -> None:
        self.assertAlmostEqual(_mean([1.0, 1.0, 1.0]), 1.0)

    def test_tuple_input(self) -> None:
        self.assertAlmostEqual(_mean((0.2, 0.4, 0.6)), 0.4)

    def test_result_is_float(self) -> None:
        result = _mean([1, 2, 3])
        self.assertIsInstance(result, float)

class ScenarioSuccessRateTest(unittest.TestCase):
    """Tests for ablations._scenario_success_rate() - new in this PR."""

    def test_reads_from_suite_key(self) -> None:
        payload = {
            "suite": {
                "visual_olfactory_pincer": {"success_rate": 0.6},
            }
        }
        result = _scenario_success_rate(payload, "visual_olfactory_pincer")
        self.assertAlmostEqual(result, 0.6)

    def test_reads_from_legacy_scenarios_key_when_suite_is_not_mapping(self) -> None:
        # The legacy_scenarios path is only reached when "suite" is not a mapping
        payload = {
            "suite": "invalid",
            "legacy_scenarios": {
                "olfactory_ambush": {"success_rate": 0.4},
            }
        }
        result = _scenario_success_rate(payload, "olfactory_ambush")
        self.assertAlmostEqual(result, 0.4)

    def test_suite_takes_precedence_over_legacy(self) -> None:
        payload = {
            "suite": {"s1": {"success_rate": 0.8}},
            "legacy_scenarios": {"s1": {"success_rate": 0.2}},
        }
        result = _scenario_success_rate(payload, "s1")
        self.assertAlmostEqual(result, 0.8)

    def test_missing_scenario_in_suite_returns_none(self) -> None:
        payload = {"suite": {}}
        result = _scenario_success_rate(payload, "nonexistent_scenario")
        self.assertIsNone(result)

    def test_empty_payload_returns_none(self) -> None:
        result = _scenario_success_rate({}, "visual_olfactory_pincer")
        self.assertIsNone(result)

    def test_suite_not_a_mapping_falls_through_to_legacy(self) -> None:
        payload = {
            "suite": "invalid",
            "legacy_scenarios": {"s1": {"success_rate": 0.3}},
        }
        result = _scenario_success_rate(payload, "s1")
        self.assertAlmostEqual(result, 0.3)

    def test_missing_success_rate_key_returns_none(self) -> None:
        payload = {"suite": {"s1": {}}}
        result = _scenario_success_rate(payload, "s1")
        self.assertIsNone(result)

    def test_both_suite_and_legacy_missing_returns_none(self) -> None:
        result = _scenario_success_rate({"other_key": {}}, "any_scenario")
        self.assertIsNone(result)

    def test_explicit_none_success_rate_returns_none(self) -> None:
        payload = {"suite": {"s1": {"success_rate": None}}}
        result = _scenario_success_rate(payload, "s1")
        self.assertIsNone(result)

    def test_invalid_success_rate_returns_none(self) -> None:
        payload = {"suite": {"s1": {"success_rate": "not-a-number"}}}
        result = _scenario_success_rate(payload, "s1")
        self.assertIsNone(result)

class ComparePredatorTypeAblationEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for compare_predator_type_ablation_performance() - new in this PR."""

    def test_non_mapping_variants_value_returns_unavailable(self) -> None:
        # The only way to get available=False is variants not being a Mapping
        result = compare_predator_type_ablation_performance({"variants": [1, 2, 3]})
        self.assertFalse(result["available"])
        self.assertEqual(result["comparisons"], {})

    def test_non_mapping_variants_returns_unavailable(self) -> None:
        result = compare_predator_type_ablation_performance({"variants": "not_a_mapping"})
        self.assertFalse(result["available"])

    def test_variant_not_in_payload_is_skipped(self) -> None:
        # A missing variant has no scenario runs, so predator-type comparisons should
        # only include variants with real multi-predator scenario data.
        payload = {
            "variants": {
                "drop_sensory_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.5},
                        "olfactory_ambush": {"success_rate": 0.3},
                        "visual_hunter_open_field": {"success_rate": 0.7},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        self.assertTrue(result["available"])
        self.assertIn("drop_sensory_cortex", result["comparisons"])
        self.assertNotIn("drop_visual_cortex", result["comparisons"])

    def test_custom_variant_names(self) -> None:
        payload = {
            "variants": {
                "my_custom_variant": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.6},
                        "olfactory_ambush": {"success_rate": 0.4},
                        "visual_hunter_open_field": {"success_rate": 0.8},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(
            payload,
            variant_names=["my_custom_variant"],
        )
        self.assertTrue(result["available"])
        self.assertIn("my_custom_variant", result["comparisons"])

    def test_scenario_groups_always_present_in_result(self) -> None:
        result = compare_predator_type_ablation_performance({})
        self.assertIn("scenario_groups", result)
        self.assertIn("multi_predator_ecology", result["scenario_groups"])

    def test_no_scenarios_present_is_unavailable(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {}
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        self.assertFalse(result["available"])
        self.assertEqual(result["comparisons"], {})

    def test_visual_minus_olfactory_is_rounded_to_six_decimals(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 1 / 3},
                        "visual_hunter_open_field": {"success_rate": 1 / 9},
                        "olfactory_ambush": {"success_rate": 0.0},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        diff = result["comparisons"]["drop_visual_cortex"]["visual_minus_olfactory_success_rate"]
        expected = round((((1 / 3) + (1 / 9)) / 2.0) - (((1 / 3) + 0.0) / 2.0), 6)
        self.assertEqual(diff, expected)

    def test_visual_minus_olfactory_is_none_when_a_group_lacks_success_rate_data(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_hunter_open_field": {"success_rate": 0.25},
                    }
                }
            }
        }

        result = compare_predator_type_ablation_performance(payload)

        comparison = result["comparisons"]["drop_visual_cortex"]
        self.assertEqual(comparison["visual_predator_scenarios"]["scenario_count"], 1)
        self.assertEqual(comparison["olfactory_predator_scenarios"]["scenario_count"], 0)
        self.assertIsNone(comparison["visual_minus_olfactory_success_rate"])

    def test_visual_minus_olfactory_delta_is_none_when_a_group_lacks_delta_data(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_hunter_open_field": {"success_rate": 0.25},
                        "olfactory_ambush": {"success_rate": 0.5},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex_vs_modular_full": {
                    "scenarios": {
                        "visual_hunter_open_field": {
                            "scenario_success_rate_delta": -0.3,
                        },
                    }
                }
            },
        }

        result = compare_predator_type_ablation_performance(payload)

        comparison = result["comparisons"]["drop_visual_cortex"]
        self.assertEqual(comparison["visual_predator_scenarios"]["scenario_delta_count"], 1)
        self.assertEqual(comparison["olfactory_predator_scenarios"]["scenario_delta_count"], 0)
        self.assertIsNone(comparison["visual_minus_olfactory_success_rate_delta"])

    def test_invalid_success_rate_is_not_counted_in_group_metrics(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": "bad"},
                        "visual_hunter_open_field": {"success_rate": 0.25},
                        "olfactory_ambush": {"success_rate": 0.5},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        visual_group = result["comparisons"]["drop_visual_cortex"]["visual_predator_scenarios"]
        self.assertEqual(visual_group["scenario_count"], 1)
        self.assertEqual(visual_group["mean_success_rate"], 0.25)

    def test_delta_computed_from_deltas_vs_reference(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex_vs_modular_full": {
                    "scenarios": {
                        "visual_olfactory_pincer": {
                            "scenario_success_rate_delta": -0.4,
                        },
                        "visual_hunter_open_field": {
                            "scenario_success_rate_delta": -0.6,
                        },
                        "olfactory_ambush": {"scenario_success_rate_delta": 0.1},
                    }
                }
            },
        }
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        # Visual group: pincer -0.4, open_field -0.6 → mean = -0.5
        # Olfactory group: pincer -0.4, ambush 0.1 → mean = -0.15
        visual_delta = comparison["visual_predator_scenarios"]["mean_success_rate_delta"]
        olfactory_delta = comparison["olfactory_predator_scenarios"]["mean_success_rate_delta"]
        self.assertAlmostEqual(visual_delta, -0.5, places=5)
        self.assertAlmostEqual(olfactory_delta, -0.15, places=5)

    def test_delta_uses_variant_vs_reference_keys_and_scenario_success_field(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex_vs_modular_full": {
                    "scenarios": {
                        "visual_olfactory_pincer": {
                            "scenario_success_rate_delta": -0.4,
                        },
                        "visual_hunter_open_field": {
                            "scenario_success_rate_delta": -0.6,
                        },
                        "olfactory_ambush": {
                            "scenario_success_rate_delta": 0.1,
                        },
                    }
                }
            },
        }

        result = compare_predator_type_ablation_performance(payload)

        comparison = result["comparisons"]["drop_visual_cortex"]
        self.assertAlmostEqual(
            comparison["visual_predator_scenarios"]["mean_success_rate_delta"],
            -0.5,
            places=5,
        )
        self.assertAlmostEqual(
            comparison["olfactory_predator_scenarios"]["mean_success_rate_delta"],
            -0.15,
            places=5,
        )

    def test_invalid_success_rate_delta_is_not_counted_in_group_metrics(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex_vs_modular_full": {
                    "scenarios": {
                        "visual_olfactory_pincer": {
                            "scenario_success_rate_delta": "bad",
                        },
                        "visual_hunter_open_field": {
                            "scenario_success_rate_delta": -0.6,
                        },
                        "olfactory_ambush": {
                            "scenario_success_rate_delta": 0.1,
                        },
                    }
                }
            },
        }
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        self.assertAlmostEqual(
            comparison["visual_predator_scenarios"]["mean_success_rate_delta"],
            -0.6,
            places=5,
        )
        self.assertAlmostEqual(
            comparison["olfactory_predator_scenarios"]["mean_success_rate_delta"],
            0.1,
            places=5,
        )

    def test_legacy_scenarios_key_works_when_suite_not_mapping(self) -> None:
        # legacy_scenarios only reached when "suite" is not a Mapping
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": "not_a_mapping",
                    "legacy_scenarios": {
                        "visual_olfactory_pincer": {"success_rate": 0.3},
                        "visual_hunter_open_field": {"success_rate": 0.5},
                        "olfactory_ambush": {"success_rate": 0.6},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        self.assertTrue(result["available"])
        comparison = result["comparisons"]["drop_visual_cortex"]
        # visual scenarios: pincer=0.3, open_field=0.5 → mean=0.4
        self.assertAlmostEqual(
            comparison["visual_predator_scenarios"]["mean_success_rate"], 0.4
        )
