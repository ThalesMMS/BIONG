import unittest
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import canonical_ablation_configs
from spider_cortex_sim.comparison import (
    build_ablation_deltas,
    build_learning_evidence_deltas,
    build_learning_evidence_summary,
    build_predator_type_specialization_summary,
    build_reward_audit,
    build_reward_audit_comparison,
    austere_survival_gate_passed,
    compare_ablation_suite,
    compare_behavior_suite,
    compare_configurations,
    compare_learning_evidence,
    compare_noise_robustness,
    compare_reward_profiles,
    compare_training_regimes,
    condition_compact_summary,
    profile_comparison_metrics,
)
from spider_cortex_sim.curriculum import CURRICULUM_FOCUS_SCENARIOS
from spider_cortex_sim.learning_evidence import LearningEvidenceConditionSpec
from spider_cortex_sim.noise import RobustnessMatrixSpec
from spider_cortex_sim.reward import SCENARIO_AUSTERE_REQUIREMENTS, SHAPING_GAP_POLICY
from spider_cortex_sim.simulation import SpiderSimulation

class ComparisonWorkflowTest(unittest.TestCase):
    def test_behavior_comparison_reports_profiles_maps_and_matrix(self) -> None:
            comparisons, rows = compare_behavior_suite(
                episodes=0,
                evaluation_episodes=0,
                reward_profiles=["classic", "ecological"],
                map_templates=["central_burrow", "two_shelters"],
                seeds=(7,),
                names=("night_rest",),
            )

            self.assertEqual(comparisons["seeds"], [7])
            self.assertEqual(comparisons["scenario_names"], ["night_rest"])
            self.assertIn("classic", comparisons["reward_profiles"])
            self.assertIn("two_shelters", comparisons["map_templates"])
            self.assertIn("central_burrow", comparisons["matrix"]["classic"])
            self.assertIn("summary", comparisons["reward_profiles"]["classic"])
            self.assertEqual(comparisons["noise_profile"], "none")
            self.assertIn("noise_profile_config", comparisons["reward_profiles"]["classic"])
            self.assertTrue(rows)
            self.assertIn("ablation_variant", rows[0])
            self.assertIn("ablation_architecture", rows[0])
            self.assertIn("noise_profile", rows[0])
            self.assertIn("noise_profile_config", rows[0])

    def test_compare_ablation_suite_reports_reference_deltas_and_rows(self) -> None:
            """
            Verify that compare_ablation_suite reports the reference variant, includes requested variants and scenarios, provides deltas against the reference for each variant, and returns non-empty rows that contain `ablation_variant` and `ablation_architecture` columns.

            Asserts that:
            - `reference_variant` is "modular_full" and `scenario_names` contains "night_rest".
            - The payload's `variants` includes "modular_full", "no_module_reflexes", and "monolithic_policy".
            - The "monolithic_policy" variant includes a `config`, and deltas vs reference include a `summary`.
            - The "no_module_reflexes" deltas list the "night_rest" scenario.
            - `rows` is non-empty and the first row contains `ablation_variant` and `ablation_architecture`.
            """
            payload, rows = compare_ablation_suite(
                episodes=0,
                evaluation_episodes=0,
                variant_names=["no_module_reflexes", "local_credit_only", "monolithic_policy"],
                names=("night_rest",),
                seeds=(7,),
            )

            self.assertEqual(payload["reference_variant"], "modular_full")
            self.assertEqual(payload["seeds"], [7])
            self.assertEqual(payload["scenario_names"], ["night_rest"])
            self.assertIn("modular_full", payload["variants"])
            self.assertIn("no_module_reflexes", payload["variants"])
            self.assertIn("local_credit_only", payload["variants"])
            self.assertIn("monolithic_policy", payload["variants"])
            self.assertIn("config", payload["variants"]["monolithic_policy"])
            self.assertIn("summary", payload["deltas_vs_reference"]["monolithic_policy"])
            self.assertIn("summary", payload["deltas_vs_reference"]["local_credit_only"])
            self.assertIn("night_rest", payload["deltas_vs_reference"]["no_module_reflexes"]["scenarios"])
            self.assertTrue(rows)
            self.assertIn("ablation_variant", rows[0])
            self.assertIn("ablation_architecture", rows[0])
            self.assertIn("budget_profile", rows[0])
            self.assertIn("benchmark_strength", rows[0])
            self.assertIn("checkpoint_source", rows[0])
            self.assertIn("noise_profile", rows[0])
            self.assertIn("metric_module_contribution_alert_center", rows[0])

    def test_compare_configurations_uses_budget_profile_defaults(self) -> None:
            """
            Ensures compare_configurations applies budget-profile defaults when a known profile is provided.

            Asserts the returned comparison payload sets `budget_profile` to the supplied value, uses the expected default `benchmark_strength` for that profile (`"quick"` for `"smoke"`), and resolves `seeds` to `[7]`.
            """
            comparisons = compare_configurations(
                budget_profile="smoke",
                reward_profiles=["classic"],
                map_templates=["central_burrow"],
            )

            self.assertEqual(comparisons["budget_profile"], "smoke")
            self.assertEqual(comparisons["benchmark_strength"], "quick")
            self.assertEqual(comparisons["seeds"], [7])
            self.assertEqual(comparisons["noise_profile"], "none")

    def test_compare_behavior_suite_uses_budget_profile_defaults(self) -> None:
            payload, rows = compare_behavior_suite(
                budget_profile="smoke",
                reward_profiles=["classic"],
                map_templates=["central_burrow"],
                names=("night_rest",),
            )

            self.assertEqual(payload["budget_profile"], "smoke")
            self.assertEqual(payload["benchmark_strength"], "quick")
            self.assertEqual(payload["seeds"], [7])
            self.assertEqual(payload["episodes_per_scenario"], 1)
            self.assertEqual(payload["noise_profile"], "none")
            self.assertTrue(rows)
            self.assertEqual(rows[0]["budget_profile"], "smoke")
            self.assertEqual(rows[0]["benchmark_strength"], "quick")
            self.assertEqual(rows[0]["checkpoint_source"], "final")
            self.assertEqual(rows[0]["noise_profile"], "none")
            self.assertIn("noise_profile_config", rows[0])

    def test_compare_noise_robustness_returns_matrix_and_rows(self) -> None:
            payload, rows = compare_noise_robustness(
                budget_profile="smoke",
                reward_profile="classic",
                map_template="central_burrow",
                seeds=(7,),
                names=("night_rest",),
                robustness_matrix=RobustnessMatrixSpec(
                    train_conditions=("none", "low"),
                    eval_conditions=("none", "high"),
                ),
            )

            self.assertEqual(payload["budget_profile"], "smoke")
            self.assertEqual(payload["seeds"], [7])
            self.assertEqual(payload["scenario_names"], ["night_rest"])
            self.assertEqual(
                payload["matrix_spec"]["train_conditions"],
                ["none", "low"],
            )
            self.assertEqual(
                payload["matrix_spec"]["eval_conditions"],
                ["none", "high"],
            )
            self.assertEqual(payload["matrix_spec"]["cell_count"], 4)
            self.assertIn("summary", payload["matrix"]["none"]["none"])
            self.assertEqual(
                payload["matrix"]["low"]["high"]["train_noise_profile"],
                "low",
            )
            self.assertEqual(
                payload["matrix"]["low"]["high"]["eval_noise_profile"],
                "high",
            )
            self.assertIn("train_marginals", payload)
            self.assertIn("eval_marginals", payload)
            self.assertIn("robustness_score", payload)
            self.assertIn("diagonal_score", payload)
            self.assertIn("off_diagonal_score", payload)
            self.assertTrue(rows)
            self.assertEqual(len(rows), payload["matrix_spec"]["cell_count"])
            observed_pairs = {
                (
                    row["train_noise_profile"],
                    row["eval_noise_profile"],
                )
                for row in rows
            }
            self.assertEqual(
                observed_pairs,
                {
                    ("none", "none"),
                    ("none", "high"),
                    ("low", "none"),
                    ("low", "high"),
                },
            )

    def _compare_training_regimes_focus_case(
            self,
            curriculum_profile: str,
        ) -> tuple[dict[str, object], list[dict[str, object]], list[str]]:
            """
            Run compare_training_regimes for a given curriculum profile using a fixed set of scenarios and compute the expected focus scenarios.

            Parameters:
                curriculum_profile (str): Curriculum profile name passed to compare_training_regimes.

            Returns:
                payload (dict): The comparison payload returned by compare_training_regimes.
                rows (list[dict]): The list of row dictionaries returned by compare_training_regimes.
                expected_focus_scenarios (list[str]): Focus scenarios from CURRICULUM_FOCUS_SCENARIOS that intersect with the fixed scenario set.
            """
            scenario_names = (
                "night_rest",
                "open_field_foraging",
                "corridor_gauntlet",
                "exposed_day_foraging",
                "food_deprivation",
            )
            expected_focus_scenarios = [
                name for name in CURRICULUM_FOCUS_SCENARIOS if name in scenario_names
            ]
            payload, rows = compare_training_regimes(
                budget_profile="smoke",
                reward_profile="classic",
                map_template="central_burrow",
                names=scenario_names,
                curriculum_profile=curriculum_profile,
            )
            return payload, rows, expected_focus_scenarios

    def test_compare_training_regimes_v1_reports_focus_scenarios_and_rows(self) -> None:
            payload, rows, expected_focus_scenarios = (
                self._compare_training_regimes_focus_case("ecological_v1")
            )

            self.assertEqual(payload["budget_profile"], "smoke")
            self.assertEqual(payload["seeds"], [7])
            self.assertEqual(payload["reference_regime"], "flat")
            self.assertEqual(payload["curriculum_profile"], "ecological_v1")
            self.assertIn("flat", payload["regimes"])
            self.assertIn("curriculum", payload["regimes"])
            self.assertEqual(
                payload["regimes"]["flat"]["episode_allocation"],
                payload["regimes"]["curriculum"]["episode_allocation"],
            )
            self.assertEqual(payload["focus_scenarios"], expected_focus_scenarios)
            self.assertTrue(rows)
            for field in (
                "training_regime",
                "curriculum_profile",
                "curriculum_phase",
                "curriculum_skill",
                "curriculum_phase_status",
                "curriculum_promotion_reason",
            ):
                self.assertIn(field, rows[0])
            curriculum_rows = [
                row for row in rows if row["training_regime"] == "curriculum"
            ]
            self.assertTrue(curriculum_rows)
            for field in (
                "curriculum_profile",
                "curriculum_phase",
                "curriculum_skill",
                "curriculum_phase_status",
            ):
                self.assertNotIn(curriculum_rows[0][field], {"", "none"})

    def test_compare_training_regimes_v2_reports_check_specs_and_rows(self) -> None:
            v2_payload, v2_rows, expected_focus_scenarios = (
                self._compare_training_regimes_focus_case("ecological_v2")
            )

            self.assertEqual(v2_payload["curriculum_profile"], "ecological_v2")
            self.assertIn("flat", v2_payload["regimes"])
            self.assertIn("curriculum", v2_payload["regimes"])
            self.assertEqual(v2_payload["focus_scenarios"], expected_focus_scenarios)
            curriculum_summary = v2_payload["regimes"]["curriculum"]["curriculum"]
            phases = curriculum_summary["phases"]
            self.assertEqual(len(phases), 4)
            self.assertTrue(
                any(
                    phase["training_scenarios"] != phase["promotion_scenarios"]
                    for phase in phases
                )
            )
            self.assertTrue(
                all(phase["promotion_check_specs"] for phase in phases)
            )
            promotion_reasons = [
                phase["promotion_reason"]
                for phase in phases
                if phase["promotion_reason"]
            ]
            self.assertTrue(promotion_reasons)
            self.assertTrue(
                all(
                    reason == "all_checks_passed"
                    or reason == "any_check_passed"
                    or reason == "threshold_fallback"
                    or reason.startswith("check_failed:")
                    for reason in promotion_reasons
                )
            )
            self.assertTrue(
                any(
                    reason == "all_checks_passed"
                    or reason.startswith("check_failed:")
                    for reason in promotion_reasons
                )
            )
            v2_curriculum_rows = [
                row for row in v2_rows if row["training_regime"] == "curriculum"
            ]
            self.assertTrue(v2_curriculum_rows)
            self.assertTrue(
                any(row["curriculum_skill"] for row in v2_curriculum_rows)
            )
            for row in v2_curriculum_rows:
                self.assertIn("curriculum_promotion_reason", row)
                self.assertIsInstance(row["curriculum_promotion_reason"], str)

    def test_compare_training_regimes_none_profile_raises_value_error(self) -> None:
            with self.assertRaises(ValueError):
                compare_training_regimes(
                    budget_profile="smoke",
                    curriculum_profile="none",
                )

    def test_compare_training_regimes_invalid_profile_raises_value_error(self) -> None:
            with self.assertRaises(ValueError):
                compare_training_regimes(
                    budget_profile="smoke",
                    curriculum_profile="bad_profile",
                )

    def test_compare_training_regimes_empty_seeds_raises_english_error(self) -> None:
            class EmptySeedBudget:
                scenario_episodes = 1
                behavior_seeds: tuple[int, ...] = ()

            with self.assertRaisesRegex(
                ValueError,
                r"compare_training_regimes\(\) requires at least one seed\.",
            ), mock.patch(
                "spider_cortex_sim.comparison.resolve_budget",
                return_value=EmptySeedBudget(),
            ):
                compare_training_regimes(
                    budget_profile="smoke",
                    curriculum_profile="ecological_v1",
                )

    def test_compare_training_regimes_zero_budget_preserves_curriculum_metadata(self) -> None:
            payload, rows = compare_training_regimes(
                budget_profile="smoke",
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                curriculum_profile="ecological_v1",
            )

            self.assertEqual(
                payload["regimes"]["curriculum"]["training_regime"]["mode"],
                "curriculum",
            )
            self.assertEqual(
                payload["regimes"]["curriculum"]["training_regime"]["curriculum_profile"],
                "ecological_v1",
            )
            curriculum_rows = [
                row for row in rows if row["training_regime"] == "curriculum"
            ]
            self.assertTrue(curriculum_rows)
            for row in curriculum_rows:
                self.assertEqual(row["curriculum_profile"], "ecological_v1")
                self.assertEqual(row["curriculum_phase"], "")
                self.assertEqual(row["curriculum_skill"], "")
                self.assertEqual(row["curriculum_phase_status"], "")
                self.assertEqual(row["curriculum_promotion_reason"], "")

    def test_compare_training_regimes_records_per_seed_metadata(self) -> None:
            payload, _ = compare_training_regimes(
                budget_profile="smoke",
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(7, 17),
                curriculum_profile="ecological_v2",
            )

            flat_payload = payload["regimes"]["flat"]
            curriculum_payload = payload["regimes"]["curriculum"]
            self.assertEqual(len(flat_payload["training_regimes"]), 2)
            self.assertEqual(len(curriculum_payload["training_regimes"]), 2)
            self.assertEqual(len(curriculum_payload["curriculum_runs"]), 2)
            self.assertEqual(
                [item["seed"] for item in flat_payload["training_regimes"]],
                [7, 17],
            )
            self.assertEqual(
                [item["seed"] for item in curriculum_payload["training_regimes"]],
                [7, 17],
            )
            self.assertEqual(
                [item["seed"] for item in curriculum_payload["curriculum_runs"]],
                [7, 17],
            )
            self.assertTrue(
                all(
                    item["curriculum_profile"] == "ecological_v2"
                    for item in curriculum_payload["training_regimes"]
                )
            )
            self.assertTrue(
                all(
                    item["profile"] == "ecological_v2"
                    for item in curriculum_payload["curriculum_runs"]
                )
            )

    def test_compare_training_regimes_returns_deltas_vs_flat_key(self) -> None:
            payload, _ = compare_training_regimes(
                budget_profile="smoke",
                names=("night_rest",),
                curriculum_profile="ecological_v1",
            )
            self.assertIn("deltas_vs_flat", payload)

    def test_compare_training_regimes_focus_summary_has_both_regimes(self) -> None:
            payload, _ = compare_training_regimes(
                budget_profile="smoke",
                names=("open_field_foraging", "night_rest"),
                curriculum_profile="ecological_v1",
            )
            self.assertIn("focus_summary", payload)
            self.assertIn("flat", payload["focus_summary"])
            self.assertIn("curriculum", payload["focus_summary"])

    def test_compare_training_regimes_focus_scenarios_subset_of_names(self) -> None:
            # When only non-focus scenarios are requested, focus_scenarios should be empty
            payload, _ = compare_training_regimes(
                budget_profile="smoke",
                names=("night_rest", "predator_edge"),
                curriculum_profile="ecological_v1",
            )
            self.assertEqual(payload["focus_scenarios"], [])

    def test_compare_training_regimes_scenario_names_recorded_in_payload(self) -> None:
            names = ("night_rest", "open_field_foraging")
            payload, _ = compare_training_regimes(
                budget_profile="smoke",
                names=names,
                curriculum_profile="ecological_v1",
            )
            self.assertEqual(payload["scenario_names"], list(names))

    def test_compare_training_regimes_seeds_recorded_in_payload(self) -> None:
            payload, _ = compare_training_regimes(
                budget_profile="smoke",
                names=("night_rest",),
                seeds=(42,),
                curriculum_profile="ecological_v1",
            )
            self.assertIn(42, payload["seeds"])

    def test_compare_ablation_suite_uses_budget_profile_defaults(self) -> None:
            """
            Verify compare_ablation_suite applies budget-profile defaults and annotates output rows.

            Asserts the returned payload uses the provided budget_profile ('smoke') and default benchmark_strength ('quick'),
            that seeds default to [7], and episodes_per_scenario defaults to 1. Also asserts rows are non-empty and the first
            row is annotated with budget_profile 'smoke', benchmark_strength 'quick', and noise_profile 'none'.
            """
            payload, rows = compare_ablation_suite(
                budget_profile="smoke",
                variant_names=["monolithic_policy"],
                names=("night_rest",),
            )

            self.assertEqual(payload["budget_profile"], "smoke")
            self.assertEqual(payload["benchmark_strength"], "quick")
            self.assertEqual(payload["seeds"], [7])
            self.assertEqual(payload["episodes_per_scenario"], 1)
            self.assertEqual(payload["noise_profile"], "none")
            self.assertTrue(rows)
            self.assertEqual(rows[0]["budget_profile"], "smoke")
            self.assertEqual(rows[0]["benchmark_strength"], "quick")
            self.assertEqual(rows[0]["noise_profile"], "none")
            self.assertIn("noise_profile_config", rows[0])

    def test_compare_learning_evidence_reports_conditions_deltas_and_rows(self) -> None:
            payload, rows = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(7,),
            )

            self.assertEqual(payload["reference_condition"], "trained_without_reflex_support")
            self.assertEqual(payload["budget_profile"], "smoke")
            self.assertEqual(payload["long_budget_profile"], "smoke")
            self.assertEqual(payload["scenario_names"], ["night_rest"])
            self.assertIn("trained_final", payload["conditions"])
            self.assertIn("trained_without_reflex_support", payload["conditions"])
            self.assertIn("trained_reflex_annealed", payload["conditions"])
            self.assertIn("trained_late_finetuning", payload["conditions"])
            self.assertIn("random_init", payload["conditions"])
            self.assertIn("reflex_only", payload["conditions"])
            self.assertIn("freeze_half_budget", payload["conditions"])
            self.assertIn("trained_long_budget", payload["conditions"])
            self.assertIn("summary", payload["deltas_vs_reference"]["random_init"])
            self.assertIn(
                "mean_reward_delta",
                payload["deltas_vs_reference"]["random_init"]["summary"],
            )
            self.assertIn("has_learning_evidence", payload["evidence_summary"])
            self.assertTrue(rows)
            self.assertIn("learning_evidence_condition", rows[0])
            self.assertIn("learning_evidence_policy_mode", rows[0])
            self.assertIn("learning_evidence_checkpoint_source", rows[0])

    def test_compare_learning_evidence_freeze_half_budget_records_freeze_point(self) -> None:
            payload, rows = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(7,),
                condition_names=("freeze_half_budget",),
            )

            freeze_payload = payload["conditions"]["freeze_half_budget"]
            self.assertEqual(freeze_payload["train_episodes"], 3)
            self.assertEqual(freeze_payload["frozen_after_episode"], 3)
            freeze_rows = [
                row
                for row in rows
                if row["learning_evidence_condition"] == "freeze_half_budget"
            ]
            self.assertTrue(freeze_rows)
            self.assertEqual(freeze_rows[0]["learning_evidence_frozen_after_episode"], 3)

    def test_compare_learning_evidence_freeze_half_budget_allows_zero_training(self) -> None:
            payload, rows = compare_learning_evidence(
                episodes=0,
                evaluation_episodes=0,
                max_steps=60,
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(7,),
                condition_names=("freeze_half_budget",),
            )

            freeze_payload = payload["conditions"]["freeze_half_budget"]
            self.assertEqual(freeze_payload["train_episodes"], 0)
            self.assertEqual(freeze_payload["frozen_after_episode"], 0)
            freeze_rows = [
                row
                for row in rows
                if row["learning_evidence_condition"] == "freeze_half_budget"
            ]
            self.assertTrue(freeze_rows)
            self.assertEqual(freeze_rows[0]["learning_evidence_train_episodes"], 0)
            self.assertEqual(freeze_rows[0]["learning_evidence_frozen_after_episode"], 0)

    def test_compare_learning_evidence_freeze_half_budget_uses_resolved_regime_total(self) -> None:
            requested_episodes = 6
            training_regime_name = "late_finetuning"
            condition = LearningEvidenceConditionSpec(
                name="freeze_late_finetuning",
                description="Freeze after a regime-resolved partial budget.",
                train_budget="freeze_half",
                training_regime=training_regime_name,
                eval_reflex_scale=0.0,
            )
            with mock.patch(
                "spider_cortex_sim.comparison.resolve_learning_evidence_conditions",
                return_value=[condition],
            ):
                payload, rows = compare_learning_evidence(
                    episodes=requested_episodes,
                    evaluation_episodes=0,
                    max_steps=1,
                    long_budget_profile="smoke",
                    names=("night_rest",),
                    seeds=(7,),
                    condition_names=("freeze_late_finetuning",),
                    episodes_per_scenario=1,
                )

            freeze_payload = payload["conditions"]["freeze_late_finetuning"]
            expected_sim = SpiderSimulation(seed=7, max_steps=1)
            expected_sim.train(
                max(0, requested_episodes // 2),
                evaluation_episodes=0,
                capture_evaluation_trace=False,
                training_regime=training_regime_name,
            )
            expected_train_episodes = int(
                expected_sim._latest_training_regime_summary["resolved_budget"][
                    "total_training_episodes"
                ]
            )
            self.assertEqual(freeze_payload["train_episodes"], expected_train_episodes)
            self.assertEqual(
                freeze_payload["frozen_after_episode"],
                expected_train_episodes,
            )
            freeze_rows = [
                row
                for row in rows
                if row["learning_evidence_condition"] == "freeze_late_finetuning"
            ]
            self.assertTrue(freeze_rows)
            self.assertEqual(
                freeze_rows[0]["learning_evidence_train_episodes"],
                expected_train_episodes,
            )
            self.assertEqual(
                freeze_rows[0]["learning_evidence_frozen_after_episode"],
                expected_train_episodes,
            )

    def test_learning_evidence_summary_requires_explicit_condition_keys(self) -> None:
            summary = build_learning_evidence_summary(
                {
                    "trained_without_reflex_support": {
                        "summary": {
                            "scenario_success_rate": 1.0,
                            "episode_success_rate": 1.0,
                            "mean_reward": 1.0,
                        }
                    }
                },
                reference_condition="trained_without_reflex_support",
            )

            self.assertFalse(summary["supports_primary_evidence"])
            self.assertFalse(summary["has_learning_evidence"])

    def test_condition_compact_summary_derives_mean_reward_from_legacy_scenarios(self) -> None:
            compact = condition_compact_summary(
                {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.25,
                    },
                    "legacy_scenarios": {
                        "night_rest": {"mean_reward": 1.5},
                        "predator_edge": {"mean_reward": -0.5},
                    },
                }
            )

            self.assertEqual(compact["scenario_success_rate"], 0.5)
            self.assertEqual(compact["episode_success_rate"], 0.25)
            self.assertEqual(compact["mean_reward"], 0.5)

    def test_compare_learning_evidence_auto_includes_no_reflex_reference_in_explicit_subset(self) -> None:
            payload, _ = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(7,),
                condition_names=("random_init",),
            )

            self.assertIn("trained_without_reflex_support", payload["conditions"])
            self.assertIn("random_init", payload["conditions"])
            self.assertEqual(payload["reference_condition"], "trained_without_reflex_support")

    def test_training_guardrail_matrix_stays_finite_and_minimally_viable(self) -> None:
            comparisons = compare_configurations(
                max_steps=90,
                episodes=12,
                evaluation_episodes=2,
                reward_profiles=["classic", "ecological"],
                map_templates=["central_burrow"],
            )

            self.assertEqual(comparisons["seeds"], [7, 17, 29])
            for profile in ("classic", "ecological"):
                stats = comparisons["reward_profiles"][profile]
                self.assertTrue(np.isfinite(stats["mean_reward"]))
                self.assertIn("mean_night_role_distribution", stats)
                viability = (
                    stats["mean_food"]
                    + stats["mean_night_shelter_occupancy_rate"]
                    + stats["survival_rate"]
                )
                self.assertGreater(viability, 0.15)

    def test_comparison_workflow_reports_profiles_and_maps(self) -> None:
            """
            Verify the configuration comparison workflow exposes reported reward profiles, map templates, and a result matrix linking profiles to maps.

            Calls compare_configurations with a small sweep and asserts:
            - the reported `reward_profiles` contains "classic" and "ecological",
            - the reported `map_templates` contains the three requested templates,
            - the comparison `matrix` includes an entry for "classic" and a mapping from "classic" to "two_shelters".
            """
            comparisons = compare_configurations(
                max_steps=40,
                episodes=2,
                evaluation_episodes=1,
                reward_profiles=["classic", "ecological"],
                map_templates=["central_burrow", "two_shelters", "exposed_feeding_ground"],
                seeds=(7,),
            )

            self.assertEqual(set(comparisons["reward_profiles"].keys()), {"classic", "ecological"})
            self.assertEqual(set(comparisons["map_templates"].keys()), {"central_burrow", "two_shelters", "exposed_feeding_ground"})
            self.assertIn("classic", comparisons["matrix"])
            self.assertIn("two_shelters", comparisons["matrix"]["classic"])

class BehaviorComparisonAuditTest(unittest.TestCase):
    @staticmethod
    def _synthetic_reward_profile_comparison() -> dict[str, object]:
            return {
                "reward_profiles": {
                    "classic": {
                        "summary": {
                            "scenario_success_rate": 0.8,
                            "episode_success_rate": 0.8,
                        },
                        "suite": {
                            "night_rest": {"success_rate": 1.0, "episodes": 2},
                            "food_vs_predator_conflict": {
                                "success_rate": 0.6,
                                "episodes": 2,
                            },
                        },
                    },
                    "austere": {
                        "summary": {
                            "scenario_success_rate": 0.25,
                            "episode_success_rate": 0.25,
                        },
                        "suite": {
                            "night_rest": {"success_rate": 0.0, "episodes": 2},
                            "food_vs_predator_conflict": {
                                "success_rate": 0.5,
                                "episodes": 2,
                            },
                        },
                    },
                }
            }
    def test_compare_behavior_suite_includes_shaping_audit(self) -> None:
            payload, _rows = compare_behavior_suite(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(7,),
                reward_profiles=("classic", "austere"),
            )
            self.assertIn("reward_audit", payload)
            self.assertEqual(payload["reward_audit"]["minimal_profile"], "austere")
            self.assertIn("comparison", payload["reward_audit"])
            self.assertIn(
                "classic",
                payload["reward_audit"]["comparison"]["deltas_vs_minimal"],
            )
            self.assertIn("behavior_survival", payload)
            self.assertIn("austere_survival_summary", payload)
            self.assertIn("gap_policy_check", payload)
            self.assertIn("shaping_dependent_behaviors", payload)
            self.assertTrue(payload["behavior_survival"]["available"])
            self.assertTrue(payload["austere_survival_summary"]["available"])
    def test_compare_behavior_suite_gap_policy_check_uses_policy(self) -> None:
            payload, _rows = compare_behavior_suite(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(7,),
                reward_profiles=("classic", "austere"),
            )

            self.assertIn("gap_policy_check", payload)
            self.assertEqual(payload["gap_policy_check"]["policy"], SHAPING_GAP_POLICY)
            self.assertIn("classic", payload["gap_policy_check"]["checked_profiles"])
            self.assertIn("night_rest", payload["gap_policy_check"]["checked_scenarios"])
    def test_shaping_dependent_behaviors_identifies_gap_exceeding_scenarios(self) -> None:
            comparison = build_reward_audit_comparison(
                self._synthetic_reward_profile_comparison()
            )

            dependent = comparison["shaping_dependent_behaviors"]
            self.assertEqual(len(dependent), 1)
            self.assertEqual(dependent[0]["scenario"], "night_rest")
            self.assertEqual(dependent[0]["profile"], "classic")
            self.assertGreater(
                dependent[0]["success_rate_delta"],
                SHAPING_GAP_POLICY["max_scenario_success_rate_delta"][
                    "classic_minus_austere"
                ],
            )
    def test_gate_scenario_austere_failure_increments_gate_fail_count(self) -> None:
            comparison = build_reward_audit_comparison(
                self._synthetic_reward_profile_comparison()
            )

            summary = comparison["austere_survival_summary"]
            self.assertEqual(summary["gate_fail_count"], 1)
            self.assertEqual(summary["gate_pass_count"], 0)
            self.assertEqual(summary["observed_gate_count"], 1)
            self.assertEqual(
                summary["expected_gate_count"],
                sum(
                    1
                    for requirement in SCENARIO_AUSTERE_REQUIREMENTS.values()
                    if requirement["requirement_level"] == "gate"
                ),
            )
            self.assertFalse(summary["gate_coverage_complete"])
            self.assertTrue(summary["gap_policy_violations"])
    def test_austere_gate_pass_requires_complete_gate_coverage(self) -> None:
            payload = self._synthetic_reward_profile_comparison()
            austere_suite = payload["reward_profiles"]["austere"]["suite"]
            austere_suite["night_rest"]["success_rate"] = 1.0

            comparison = build_reward_audit_comparison(payload)

            summary = comparison["austere_survival_summary"]
            self.assertEqual(summary["gate_fail_count"], 0)
            self.assertFalse(summary["gate_coverage_complete"])
            self.assertFalse(austere_survival_gate_passed(comparison))
    def test_compare_behavior_suite_omits_minimal_baseline_when_austere_not_compared(self) -> None:
            payload, _rows = compare_behavior_suite(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(7,),
                reward_profiles=("classic", "ecological"),
            )
            self.assertIsNone(payload["reward_audit"]["minimal_profile"])
            self.assertEqual(
                payload["reward_audit"]["comparison"]["deltas_vs_minimal"],
                {},
            )

class SimulationCompareConfigurationsBudgetTest(unittest.TestCase):
    """Tests that compare_configurations properly handles budget profiles."""

    def test_compare_configurations_with_no_budget_profile_uses_custom(self) -> None:
        comparisons = compare_configurations(
            episodes=0,
            evaluation_episodes=0,
            reward_profiles=["classic"],
            map_templates=["central_burrow"],
            seeds=(7,),
        )
        # With no budget_profile, it uses 'custom'
        self.assertEqual(comparisons["budget_profile"], "custom")
        self.assertEqual(comparisons["benchmark_strength"], "custom")

    def test_compare_configurations_budget_profile_in_result(self) -> None:
        comparisons = compare_configurations(
            budget_profile="smoke",
            reward_profiles=["classic"],
            map_templates=["central_burrow"],
        )
        self.assertIn("budget_profile", comparisons)
        self.assertIn("benchmark_strength", comparisons)
        self.assertIn("seeds", comparisons)

    def test_compare_configurations_seeds_from_budget_profile_when_none(self) -> None:
        # When seeds=None and budget_profile='smoke', seeds come from profile comparison_seeds
        comparisons = compare_configurations(
            budget_profile="smoke",
            episodes=0,
            evaluation_episodes=0,
            reward_profiles=["classic"],
            map_templates=["central_burrow"],
        )
        self.assertEqual(comparisons["seeds"], [7])

    def test_compare_reward_profiles_promotes_austere_survival(self) -> None:
        comparisons = compare_reward_profiles(
            episodes=0,
            evaluation_episodes=0,
            max_steps=5,
            reward_profiles=["classic", "austere"],
            map_templates=["central_burrow"],
            seeds=(7,),
        )

        self.assertIn("behavior_survival", comparisons)
        self.assertIn("austere_survival_summary", comparisons)
        self.assertIn("gap_policy_check", comparisons)
        self.assertIn("shaping_dependent_behaviors", comparisons)
        self.assertIn("episodes_detail", comparisons["reward_profiles"]["austere"])
        self.assertIn(
            "episodes_detail",
            comparisons["map_templates"]["central_burrow"],
        )
        self.assertFalse(comparisons["behavior_survival"]["available"])
