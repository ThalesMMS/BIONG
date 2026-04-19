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


class LearningEvidenceComparisonHelpersTest(unittest.TestCase):
    def test_condition_compact_summary_none_returns_zeros(self) -> None:
            result = condition_compact_summary(None)
            self.assertEqual(result["scenario_success_rate"], 0.0)
            self.assertEqual(result["episode_success_rate"], 0.0)
            self.assertEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_non_dict_returns_zeros(self) -> None:
            result = condition_compact_summary("not_a_dict")  # type: ignore[arg-type]
            self.assertEqual(result["scenario_success_rate"], 0.0)
            self.assertEqual(result["episode_success_rate"], 0.0)
            self.assertEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_missing_summary_key_returns_zeros(self) -> None:
            result = condition_compact_summary({"policy_mode": "normal"})
            self.assertEqual(result["scenario_success_rate"], 0.0)
            self.assertEqual(result["episode_success_rate"], 0.0)
            self.assertEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_extracts_values(self) -> None:
            payload = {
                "summary": {
                    "scenario_success_rate": 0.8,
                    "episode_success_rate": 0.75,
                    "mean_reward": 12.5,
                }
            }
            result = condition_compact_summary(payload)
            self.assertAlmostEqual(result["scenario_success_rate"], 0.8)
            self.assertAlmostEqual(result["episode_success_rate"], 0.75)
            self.assertAlmostEqual(result["mean_reward"], 12.5)

    def test_condition_compact_summary_partial_keys_default_to_zero(self) -> None:
            payload = {"summary": {"scenario_success_rate": 0.5}}
            result = condition_compact_summary(payload)
            self.assertAlmostEqual(result["scenario_success_rate"], 0.5)
            self.assertAlmostEqual(result["episode_success_rate"], 0.0)
            self.assertAlmostEqual(result["mean_reward"], 0.0)

    def test_condition_compact_summary_non_dict_summary_returns_zeros(self) -> None:
            payload = {"summary": "not_a_dict"}
            result = condition_compact_summary(payload)
            self.assertEqual(result["scenario_success_rate"], 0.0)

    def test_build_learning_evidence_deltas_skipped_condition(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.7,
                        "mean_reward": 10.0,
                    },
                    "suite": {},
                },
                "random_init": {
                    "skipped": True,
                    "reason": "architecture mismatch",
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )
            self.assertTrue(deltas["random_init"]["skipped"])
            self.assertEqual(deltas["random_init"]["reason"], "architecture mismatch")

    def test_build_learning_evidence_deltas_reference_has_zero_delta(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.7,
                        "mean_reward": 10.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.9}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )
            ref_delta = deltas["trained_final"]
            self.assertAlmostEqual(ref_delta["summary"]["scenario_success_rate_delta"], 0.0)
            self.assertAlmostEqual(ref_delta["summary"]["episode_success_rate_delta"], 0.0)
            self.assertAlmostEqual(ref_delta["summary"]["mean_reward_delta"], 0.0)

    def test_build_learning_evidence_deltas_correct_delta_values(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {
                        "scenario_success_rate": 1.0,
                        "episode_success_rate": 0.9,
                    },
                    "suite": {"night_rest": {"success_rate": 1.0}},
                    "legacy_scenarios": {"night_rest": {"mean_reward": 20.0}},
                },
                "random_init": {
                    "summary": {
                        "scenario_success_rate": 0.4,
                        "episode_success_rate": 0.3,
                    },
                    "suite": {"night_rest": {"success_rate": 0.4}},
                    "legacy_scenarios": {"night_rest": {"mean_reward": 5.0}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )
            random_delta = deltas["random_init"]
            self.assertAlmostEqual(
                random_delta["summary"]["scenario_success_rate_delta"], 0.4 - 1.0
            )
            self.assertAlmostEqual(
                random_delta["summary"]["mean_reward_delta"], 5.0 - 20.0
            )
            self.assertAlmostEqual(
                random_delta["scenarios"]["night_rest"]["success_rate_delta"], 0.4 - 1.0
            )

    def test_compare_learning_evidence_skips_reflex_only_when_reflexes_disabled(self) -> None:
            payload, _ = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(7,),
                brain_config=canonical_ablation_configs()["no_module_reflexes"],
                condition_names=("reflex_only",),
            )

            self.assertIn("trained_without_reflex_support", payload["conditions"])
            self.assertIn("reflex_only", payload["conditions"])
            self.assertTrue(payload["conditions"]["reflex_only"]["skipped"])
            self.assertIn(
                "reflexes disabled",
                payload["conditions"]["reflex_only"]["reason"],
            )

    def test_build_learning_evidence_deltas_condition_without_summary_is_skipped(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.7, "mean_reward": 5.0},
                    "suite": {},
                },
                "random_init": {"policy_mode": "normal"},  # no "summary" key
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=[],
            )
            self.assertTrue(deltas["random_init"]["skipped"])

    def test_build_learning_evidence_deltas_missing_reference_skips_all_conditions(self) -> None:
            conditions = {
                "random_init": {
                    "summary": {
                        "scenario_success_rate": 0.4,
                        "episode_success_rate": 0.3,
                        "mean_reward": 5.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.4}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )

            self.assertTrue(deltas["random_init"]["skipped"])
            self.assertIn("missing or skipped", deltas["random_init"]["reason"])

    def test_build_learning_evidence_deltas_skipped_reference_skips_all_conditions(self) -> None:
            conditions = {
                "trained_final": {
                    "skipped": True,
                    "reason": "architecture mismatch",
                },
                "random_init": {
                    "summary": {
                        "scenario_success_rate": 0.4,
                        "episode_success_rate": 0.3,
                        "mean_reward": 5.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.4}},
                },
            }
            deltas = build_learning_evidence_deltas(
                conditions,
                reference_condition="trained_final",
                scenario_names=["night_rest"],
            )

            self.assertTrue(deltas["trained_final"]["skipped"])
            self.assertTrue(deltas["random_init"]["skipped"])
            self.assertIn("missing or skipped", deltas["random_init"]["reason"])

    def test_build_learning_evidence_summary_has_learning_evidence_true(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertTrue(result["has_learning_evidence"])
            self.assertTrue(result["supports_primary_evidence"])
            self.assertEqual(result["primary_gate_metric"], "scenario_success_rate")
            self.assertEqual(result["reference_condition"], "trained_without_reflex_support")
            self.assertEqual(
                result["primary_condition"],
                result["trained_without_reflex_support"],
            )
            self.assertAlmostEqual(
                result["trained_vs_random_init"]["scenario_success_rate_delta"],
                0.7 - 0.3,
            )

    def test_build_learning_evidence_summary_has_learning_evidence_false_when_trained_not_better(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertFalse(result["has_learning_evidence"])

    def test_build_learning_evidence_summary_reflex_only_not_available_sets_primary_supported_false(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "skipped": True,
                    "reason": "monolithic architecture",
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertFalse(result["supports_primary_evidence"])
            self.assertFalse(result["has_learning_evidence"])

    def test_build_learning_evidence_summary_notes_include_gate_note(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            notes = result["notes"]
            self.assertTrue(any("trained_without_reflex_support" in note for note in notes))
            self.assertTrue(any("scenario_success_rate" in note for note in notes))

    def test_build_learning_evidence_summary_contains_delta_blocks(self) -> None:
            conditions = {
                "trained_final": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.8, "mean_reward": 15.0},
                },
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertIn("trained_vs_random_init", result)
            self.assertIn("trained_vs_reflex_only", result)
            self.assertIn("scenario_success_rate_delta", result["trained_vs_random_init"])

    def test_build_learning_evidence_summary_reports_uncertainty(self) -> None:
            conditions = {
                "trained_without_reflex_support": {
                    "summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.6, "mean_reward": 10.0},
                    "seed_level": [
                        {"metric_name": "scenario_success_rate", "seed": 1, "value": 0.6, "condition": "trained_without_reflex_support", "scenario": None},
                        {"metric_name": "scenario_success_rate", "seed": 2, "value": 0.8, "condition": "trained_without_reflex_support", "scenario": None},
                    ],
                    "uncertainty": {
                        "scenario_success_rate": {
                            "mean": 0.7,
                            "ci_lower": 0.6,
                            "ci_upper": 0.8,
                            "std_error": 0.1,
                            "n_seeds": 2,
                            "confidence_level": 0.95,
                            "seed_values": [0.6, 0.8],
                        }
                    },
                },
                "random_init": {
                    "summary": {"scenario_success_rate": 0.3, "episode_success_rate": 0.2, "mean_reward": 2.0},
                    "seed_level": [
                        {"metric_name": "scenario_success_rate", "seed": 1, "value": 0.2, "condition": "random_init", "scenario": None},
                        {"metric_name": "scenario_success_rate", "seed": 2, "value": 0.4, "condition": "random_init", "scenario": None},
                    ],
                    "uncertainty": {
                        "scenario_success_rate": {
                            "mean": 0.3,
                            "ci_lower": 0.2,
                            "ci_upper": 0.4,
                            "std_error": 0.1,
                            "n_seeds": 2,
                            "confidence_level": 0.95,
                            "seed_values": [0.2, 0.4],
                        }
                    },
                },
                "reflex_only": {
                    "summary": {"scenario_success_rate": 0.4, "episode_success_rate": 0.3, "mean_reward": 3.0},
                },
            }
            result = build_learning_evidence_summary(
                conditions, reference_condition="trained_without_reflex_support"
            )
            self.assertIn("seed_level", result)
            self.assertIn("uncertainty", result)
            delta_uncertainty = result["uncertainty"]["trained_vs_random_init"][
                "scenario_success_rate_delta"
            ]
            self.assertEqual(delta_uncertainty["n_seeds"], 2)
            self.assertAlmostEqual(delta_uncertainty["mean"], 0.4)

    def test_compare_learning_evidence_conditions_have_required_fields(self) -> None:
            payload, _rows = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=(
                    "trained_final",
                    "trained_reflex_annealed",
                    "random_init",
                ),
            )
            for cond_name in ("trained_final", "trained_reflex_annealed", "random_init"):
                cond = payload["conditions"][cond_name]
                self.assertIn("policy_mode", cond)
                self.assertIn("training_regime", cond)
                self.assertIn("train_episodes", cond)
                self.assertIn("checkpoint_source", cond)
                self.assertIn("budget_profile", cond)
                self.assertIn("skipped", cond)
            self.assertEqual(
                payload["conditions"]["trained_reflex_annealed"]["training_regime"],
                "reflex_annealed",
            )

    def test_compare_learning_evidence_random_init_has_zero_train_episodes(self) -> None:
            payload, _ = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=("random_init",),
            )
            self.assertEqual(payload["conditions"]["random_init"]["train_episodes"], 0)

    def test_compare_learning_evidence_rows_contain_all_learning_evidence_columns(self) -> None:
            _, rows = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=("trained_final", "trained_reflex_annealed"),
            )
            self.assertTrue(rows)
            expected_keys = [
                "learning_evidence_condition",
                "learning_evidence_policy_mode",
                "learning_evidence_training_regime",
                "learning_evidence_train_episodes",
                "learning_evidence_frozen_after_episode",
                "learning_evidence_checkpoint_source",
                "learning_evidence_budget_profile",
                "learning_evidence_budget_benchmark_strength",
            ]
            for key in expected_keys:
                self.assertIn(key, rows[0])
            regime_rows = [
                row
                for row in rows
                if row["learning_evidence_condition"] == "trained_reflex_annealed"
            ]
            self.assertTrue(regime_rows)
            self.assertTrue(
                all(
                    row["learning_evidence_training_regime"] == "reflex_annealed"
                    for row in regime_rows
                )
            )

    def test_compare_learning_evidence_evidence_summary_has_gate_fields(self) -> None:
            payload, _ = compare_learning_evidence(
                budget_profile="smoke",
                long_budget_profile="smoke",
                names=("night_rest",),
                seeds=(3,),
                condition_names=("trained_final", "random_init", "reflex_only"),
            )
            ev = payload["evidence_summary"]
            self.assertIn("has_learning_evidence", ev)
            self.assertIn("supports_primary_evidence", ev)
            self.assertIn("primary_gate_metric", ev)
            self.assertEqual(ev["primary_gate_metric"], "scenario_success_rate")



class BuildAblationDeltasTest(unittest.TestCase):
    """Tests for comparison.build_ablation_deltas (new in this PR)."""

    def _make_variants(self) -> dict:
        def _variant(scenario_sr: float, episode_sr: float, scenario_success: float):
            return {
                "summary": {
                    "scenario_success_rate": scenario_sr,
                    "episode_success_rate": episode_sr,
                },
                "suite": {
                    "night_rest": {"success_rate": scenario_success},
                },
            }
        return {
            "modular_full": _variant(0.50, 0.60, 0.50),
            "no_module_reflexes": _variant(0.30, 0.40, 0.20),
        }

    def test_reference_variant_delta_is_zero(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        ref_delta = deltas["modular_full"]["summary"]
        self.assertAlmostEqual(ref_delta["scenario_success_rate_delta"], 0.0)
        self.assertAlmostEqual(ref_delta["episode_success_rate_delta"], 0.0)

    def test_nonreference_variant_delta_is_computed_correctly(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        d = deltas["no_module_reflexes"]["summary"]
        self.assertAlmostEqual(d["scenario_success_rate_delta"], 0.30 - 0.50, places=5)
        self.assertAlmostEqual(d["episode_success_rate_delta"], 0.40 - 0.60, places=5)

    def test_scenario_level_delta_computed(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        scenario_delta = deltas["no_module_reflexes"]["scenarios"]["night_rest"]["success_rate_delta"]
        self.assertAlmostEqual(scenario_delta, 0.20 - 0.50, places=5)

    def test_reference_scenario_delta_is_zero(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        self.assertAlmostEqual(
            deltas["modular_full"]["scenarios"]["night_rest"]["success_rate_delta"], 0.0
        )

    def test_all_variants_present_in_deltas(self) -> None:
        variants = self._make_variants()
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        self.assertIn("modular_full", deltas)
        self.assertIn("no_module_reflexes", deltas)

    def test_uncertainty_and_seed_level_deltas_are_reported(self) -> None:
        variants = self._make_variants()
        variants["modular_full"]["seed_level"] = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.4,
                "condition": "modular_full",
                "scenario": None,
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.6,
                "condition": "modular_full",
                "scenario": None,
            },
        ]
        variants["no_module_reflexes"]["seed_level"] = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.2,
                "condition": "no_module_reflexes",
                "scenario": None,
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.4,
                "condition": "no_module_reflexes",
                "scenario": None,
            },
        ]
        deltas = build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        result = deltas["no_module_reflexes"]
        self.assertIn("uncertainty", result)
        self.assertIn("seed_level", result)
        uncertainty = result["uncertainty"]["scenario_success_rate_delta"]
        self.assertEqual(uncertainty["n_seeds"], 2)
        self.assertAlmostEqual(uncertainty["mean"], -0.2)

class PredatorTypeSpecializationUncertaintyTest(unittest.TestCase):
    def _variant(self, condition: str, values: dict[str, tuple[float, float]]) -> dict:
        suite = {
            scenario: {"success_rate": sum(seed_values) / len(seed_values)}
            for scenario, seed_values in values.items()
        }
        seed_level: list[dict[str, object]] = [
            {
                "metric_name": "specialization_score",
                "seed": 1,
                "value": 0.2,
                "condition": condition,
                "scenario": None,
            },
            {
                "metric_name": "specialization_score",
                "seed": 2,
                "value": 0.4,
                "condition": condition,
                "scenario": None,
            },
        ]
        for scenario, seed_values in values.items():
            self.assertEqual(2, len(seed_values))
            scenario_rows = [
                {
                    "metric_name": "scenario_success_rate",
                    "seed": seed,
                    "value": value,
                    "condition": condition,
                    "scenario": scenario,
                }
                for seed, value in zip((1, 2), seed_values)
            ]
            suite[scenario]["seed_level"] = scenario_rows
            seed_level.extend(scenario_rows)
        return {
            "summary": {
                "scenario_success_rate": 0.5,
                "episode_success_rate": 0.5,
            },
            "suite": suite,
            "seed_level": seed_level,
        }

    def test_predator_type_specialization_summary_reports_uncertainty(self) -> None:
        variants = {
            "modular_full": self._variant(
                "modular_full",
                {
                    "visual_olfactory_pincer": (0.5, 0.5),
                    "visual_hunter_open_field": (0.6, 0.8),
                    "olfactory_ambush": (0.4, 0.2),
                },
            ),
            "drop_visual_cortex": self._variant(
                "drop_visual_cortex",
                {
                    "visual_olfactory_pincer": (0.2, 0.2),
                    "visual_hunter_open_field": (0.1, 0.3),
                    "olfactory_ambush": (0.7, 0.5),
                },
            ),
        }
        deltas = build_ablation_deltas(
            variants,
            reference_variant="modular_full",
            scenario_names=[
                "visual_olfactory_pincer",
                "visual_hunter_open_field",
                "olfactory_ambush",
            ],
        )

        result = build_predator_type_specialization_summary(
            variants,
            reference_variant="modular_full",
            deltas_vs_reference=deltas,
        )

        self.assertTrue(result["available"])
        uncertainty = result["uncertainty"]["drop_visual_cortex"]
        self.assertEqual(
            uncertainty["visual_minus_olfactory_success_rate"]["n_seeds"],
            2,
        )
        self.assertEqual(uncertainty["specialization_score"]["n_seeds"], 2)

class CompareAblationSuiteAlwaysIncludesReferenceTest(unittest.TestCase):
    """Tests that compare_ablation_suite always includes modular_full as reference."""

    def test_reference_always_present_when_only_variant_requested(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["no_module_dropout"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("modular_full", payload["variants"])
        self.assertEqual(payload["reference_variant"], "modular_full")

    def test_reference_not_duplicated_when_explicitly_requested(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["modular_full", "monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        variant_keys = list(payload["variants"].keys())
        self.assertEqual(variant_keys.count("modular_full"), 1)

    def test_deltas_always_has_reference_entry(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["no_module_reflexes"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("modular_full", payload["deltas_vs_reference"])

    def test_scenario_names_in_payload(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest", "predator_edge"),
            seeds=(7,),
        )
        self.assertEqual(payload["scenario_names"], ["night_rest", "predator_edge"])

    def test_seeds_in_payload(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7, 17),
        )
        self.assertEqual(payload["seeds"], [7, 17])

    def test_rows_contain_ablation_columns(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["local_credit_only", "monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("ablation_variant", row)
            self.assertIn("ablation_architecture", row)
            self.assertIn("metric_module_contribution_alert_center", row)
            self.assertIn("metric_dominant_module", row)
            self.assertIn("metric_effective_module_count", row)

    def test_payload_includes_local_credit_only_variant(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["local_credit_only"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("local_credit_only", payload["variants"])
        self.assertEqual(
            payload["variants"]["local_credit_only"]["config"]["name"],
            "local_credit_only",
        )
        self.assertIn(
            "mean_module_contribution_share",
            payload["variants"]["local_credit_only"]["legacy_scenarios"]["night_rest"],
        )

    def test_rows_contain_operational_profile_columns(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("operational_profile", row)
            self.assertIn("operational_profile_version", row)
            self.assertIn("noise_profile", row)
            self.assertIn("noise_profile_config", row)

    def test_rows_contain_architecture_traceability_columns(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("architecture_version", row)
            self.assertIn("architecture_fingerprint", row)

    def test_payload_includes_without_reflex_support_and_rows_track_eval_scale(self) -> None:
        """
        Verify that compare_ablation_suite marks the primary evaluation as "without_reflex_support" and that reflex-scale variant rows include both evaluation runs with eval_reflex_scale == 0.0 and eval_reflex_scale > 0.0.

        This test calls compare_ablation_suite for the single variant "reflex_scale_0_50" and asserts:
        - The payload's variant entries for "modular_full" and "reflex_scale_0_50" include "without_reflex_support".
        - The overall primary evaluation is "without_reflex_support" and the reference evaluation reflex scale is 0.0.
        - The modular_full variant's primary evaluation and its summary.eval_reflex_scale are set to "without_reflex_support" and 0.0 respectively.
        - The reflex_scale_0_50 config reflex_scale is approximately 0.5.
        - The returned rows contain entries for the "reflex_scale_0_50" variant and include at least one row with eval_reflex_scale == 0.0 and at least one row with eval_reflex_scale > 0.0.
        """
        payload, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["reflex_scale_0_50"],
            names=("night_rest",),
            seeds=(7,),
        )

        self.assertIn("without_reflex_support", payload["variants"]["modular_full"])
        self.assertIn("without_reflex_support", payload["variants"]["reflex_scale_0_50"])
        self.assertEqual(payload["primary_evaluation"], "without_reflex_support")
        self.assertEqual(payload["reference_eval_reflex_scale"], 0.0)
        self.assertEqual(
            payload["variants"]["modular_full"]["primary_evaluation"],
            "without_reflex_support",
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["summary"]["eval_reflex_scale"],
            0.0,
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["summary"]["competence_type"],
            "self_sufficient",
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["without_reflex_support"]["summary"][
                "competence_type"
            ],
            "self_sufficient",
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["with_reflex_support"]["summary"][
                "competence_type"
            ],
            "scaffolded",
        )
        self.assertAlmostEqual(
            float(payload["variants"]["reflex_scale_0_50"]["config"]["reflex_scale"]),
            0.5,
        )
        reflex_scale_rows = [
            row for row in rows if row["ablation_variant"] == "reflex_scale_0_50"
        ]
        self.assertTrue(reflex_scale_rows)
        self.assertTrue(
            any(float(row["eval_reflex_scale"]) == 0.0 for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(float(row["eval_reflex_scale"]) > 0.0 for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(row["competence_type"] == "self_sufficient" for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(row["competence_type"] == "scaffolded" for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(row["is_primary_benchmark"] is True for row in reflex_scale_rows)
        )

    def test_without_reflex_support_reuses_behavior_base_index(self) -> None:
        """
        Verifies that behavior evaluations run without reflex support reuse the same base_index across related suite executions.

        Patches SpiderSimulation._execute_behavior_suite to record (brain config name, current_reflex_scale, base_index) for each executed suite, runs compare_ablation_suite for a reflex-scale variant, and asserts the sequence of recorded calls and their base_index values match the expected order: first the reference variant with reflex support and without, then the reflex-scale variant with reflex support and without — all using the same base_index (300000).
        """
        original_execute = SpiderSimulation._execute_behavior_suite
        recorded_calls: list[tuple[str, float, int]] = []

        def wrapped_execute(
            self,
            *,
            names,
            episodes_per_scenario,
            capture_trace,
            debug_trace,
            base_index=100_000,
        ):
            recorded_calls.append(
                (
                    self.brain.config.name,
                    float(self.brain.current_reflex_scale),
                    int(base_index),
                )
            )
            return original_execute(
                self,
                names=names,
                episodes_per_scenario=episodes_per_scenario,
                capture_trace=capture_trace,
                debug_trace=debug_trace,
                base_index=base_index,
            )

        SpiderSimulation._execute_behavior_suite = wrapped_execute  # type: ignore[method-assign]
        try:
            compare_ablation_suite(
                episodes=0,
                evaluation_episodes=0,
                variant_names=["reflex_scale_0_50"],
                names=("night_rest",),
                seeds=(7,),
            )
        finally:
            SpiderSimulation._execute_behavior_suite = original_execute  # type: ignore[method-assign]

        self.assertEqual(
            recorded_calls,
            [
                ("modular_full", 1.0, 300_000),
                ("modular_full", 0.0, 300_000),
                ("reflex_scale_0_50", 0.5, 300_000),
                ("reflex_scale_0_50", 0.0, 300_000),
            ],
        )

    def test_rows_operational_profile_value_is_default(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        for row in rows:
            self.assertEqual(row["operational_profile"], "default_v1")
            self.assertEqual(row["operational_profile_version"], 1)
            self.assertEqual(row["noise_profile"], "none")
            self.assertIn("noise_profile_config", row)

class BuildRewardAuditTest(unittest.TestCase):
    """Tests for comparison.build_reward_audit and related helpers."""

    def test_build_reward_audit_returns_required_top_level_keys(self) -> None:
        audit = build_reward_audit()
        expected_keys = {
            "current_profile", "minimal_profile", "reward_components",
            "observation_signals", "memory_signals", "reward_profiles", "notes",
        }
        self.assertTrue(expected_keys.issubset(set(audit.keys())))

    def test_build_reward_audit_current_profile_recorded(self) -> None:
        audit = build_reward_audit(current_profile="classic")
        self.assertEqual(audit["current_profile"], "classic")

    def test_build_reward_audit_none_current_profile(self) -> None:
        audit = build_reward_audit(current_profile=None)
        self.assertIsNone(audit["current_profile"])

    def test_build_reward_audit_minimal_profile_is_austere(self) -> None:
        audit = build_reward_audit()
        self.assertEqual(audit["minimal_profile"], "austere")

    def test_build_reward_audit_reward_profiles_contains_all_known_profiles(self) -> None:
        from spider_cortex_sim.reward import REWARD_PROFILES
        audit = build_reward_audit()
        for profile_name in REWARD_PROFILES:
            self.assertIn(
                profile_name,
                audit["reward_profiles"],
                f"Profile {profile_name!r} missing from reward_audit['reward_profiles']",
            )

    def test_build_reward_audit_observation_signals_contains_predator_dist(self) -> None:
        audit = build_reward_audit()
        self.assertIn("predator_dist", audit["observation_signals"])

    def test_build_reward_audit_memory_signals_contains_shelter_memory(self) -> None:
        audit = build_reward_audit()
        self.assertIn("shelter_memory", audit["memory_signals"])

    def test_build_reward_audit_with_comparison_payload_includes_comparison(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.6,
                        "episode_success_rate": 0.55,
                    }
                },
            }
        }
        audit = build_reward_audit(comparison_payload=payload)
        self.assertIn("comparison", audit)

    def test_build_reward_audit_without_comparison_payload_no_comparison_key(self) -> None:
        audit = build_reward_audit()
        self.assertNotIn("comparison", audit)

    def test_build_reward_audit_notes_is_non_empty_list(self) -> None:
        audit = build_reward_audit()
        self.assertIsInstance(audit["notes"], list)
        self.assertGreater(len(audit["notes"]), 0)

    def test_build_reward_audit_reward_components_contains_food_progress(self) -> None:
        audit = build_reward_audit()
        self.assertIn("food_progress", audit["reward_components"])

class BuildRewardAuditComparisonTest(unittest.TestCase):
    """Tests for comparison.build_reward_audit_comparison."""

    def _payload_with_austere_suite(self) -> dict[str, object]:
        """
        Builds a test payload containing two reward profiles, `classic` and `austere`, each with summary metrics and per-scenario suite results.

        The returned dictionary has the top-level key `"reward_profiles"` mapping profile names to objects with:
        - `summary`: contains `scenario_success_rate` and `episode_success_rate`.
        - `suite`: maps scenario names to `{ "success_rate": float, "episodes": int }` entries.

        Returns:
            payload (dict[str, object]): A payload suitable for testing reward-audit comparison logic, including `classic` and `austere` profiles with differing per-scenario outcomes.
        """
        return {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.9,
                        "episode_success_rate": 0.9,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        "open_field_foraging": {"success_rate": 1.0, "episodes": 2},
                    },
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.5,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        "open_field_foraging": {"success_rate": 0.0, "episodes": 2},
                    },
                },
            }
        }

    def test_returns_none_for_none_input(self) -> None:
        result = build_reward_audit_comparison(None)
        self.assertIsNone(result)

    def test_returns_none_for_non_dict_input(self) -> None:
        result = build_reward_audit_comparison("not_a_dict")
        self.assertIsNone(result)

    def test_returns_none_when_reward_profiles_missing(self) -> None:
        result = build_reward_audit_comparison({"other_key": {}})
        self.assertIsNone(result)

    def test_returns_none_when_reward_profiles_empty(self) -> None:
        result = build_reward_audit_comparison({"reward_profiles": {}})
        self.assertIsNone(result)

    def test_returns_none_when_reward_profiles_not_dict(self) -> None:
        result = build_reward_audit_comparison({"reward_profiles": "bad"})
        self.assertIsNone(result)

    def test_minimal_profile_is_none_when_austere_absent(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.7,
                        "episode_success_rate": 0.6,
                    }
                }
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertIsNotNone(result)
        self.assertIsNone(result["minimal_profile"])
        self.assertEqual(result["deltas_vs_minimal"], {})

    def test_minimal_profile_is_austere_when_present(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertEqual(result["minimal_profile"], "austere")

    def test_deltas_vs_minimal_computed_for_all_profiles(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertIn("classic", result["deltas_vs_minimal"])
        self.assertIn("austere", result["deltas_vs_minimal"])

    def test_delta_keys_are_correct(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        delta_keys = set(result["deltas_vs_minimal"]["classic"].keys())
        self.assertEqual(
            delta_keys,
            {"scenario_success_rate_delta", "episode_success_rate_delta", "mean_reward_delta"},
        )

    def test_austere_delta_vs_itself_is_zero(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        austere_delta = result["deltas_vs_minimal"]["austere"]
        self.assertAlmostEqual(austere_delta["scenario_success_rate_delta"], 0.0)
        self.assertAlmostEqual(austere_delta["episode_success_rate_delta"], 0.0)
        self.assertAlmostEqual(austere_delta["mean_reward_delta"], 0.0)

    def test_classic_delta_vs_austere_is_positive_when_classic_has_higher_rate(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    }
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    }
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        classic_delta = result["deltas_vs_minimal"]["classic"]
        self.assertAlmostEqual(classic_delta["scenario_success_rate_delta"], 0.3)
        self.assertAlmostEqual(classic_delta["episode_success_rate_delta"], 0.3)

    def test_result_profiles_are_sorted(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {"summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.75}},
                "austere": {"summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.45}},
                "ecological": {"summary": {"scenario_success_rate": 0.7, "episode_success_rate": 0.65}},
            }
        }
        result = build_reward_audit_comparison(payload)
        profile_keys = list(result["profiles"].keys())
        self.assertEqual(profile_keys, sorted(profile_keys))

    def test_result_has_notes_list(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {"summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.75}},
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertIsInstance(result["notes"], list)
        self.assertGreater(len(result["notes"]), 0)

    def test_behavior_survival_field_present_with_austere_profile(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertIn("behavior_survival", result)
        self.assertTrue(result["behavior_survival"]["available"])
        self.assertIn("austere_survival_summary", result)
        self.assertIn("gap_policy_check", result)
        self.assertIn("shaping_dependent_behaviors", result)

    def test_behavior_survival_can_be_derived_from_episode_details(self) -> None:
        result = build_reward_audit_comparison(
            {
                "reward_profiles": {
                    "classic": {
                        "summary": {
                            "scenario_success_rate": 1.0,
                            "episode_success_rate": 1.0,
                        },
                        "episodes_detail": [
                            {
                                "scenario": "night_rest",
                                "alive": True,
                                "total_reward": 1.0,
                            }
                        ],
                    },
                    "austere": {
                        "summary": {
                            "scenario_success_rate": 1.0,
                            "episode_success_rate": 1.0,
                        },
                        "episodes_detail": [
                            {
                                "scenario": "night_rest",
                                "alive": True,
                                "total_reward": 0.5,
                            }
                        ],
                    },
                }
            }
        )
        self.assertTrue(result["behavior_survival"]["available"])
        self.assertEqual(
            result["behavior_survival"]["scenarios"]["night_rest"][
                "austere_success_rate"
            ],
            1.0,
        )

    def test_austere_survival_summary_reports_gate_counts(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        summary = result["austere_survival_summary"]
        self.assertTrue(summary["available"])
        self.assertAlmostEqual(summary["overall_survival_rate"], 0.5)
        self.assertEqual(summary["gate_pass_count"], 1)
        self.assertEqual(summary["gate_fail_count"], 0)
        self.assertEqual(summary["observed_gate_count"], 1)
        self.assertFalse(summary["gate_coverage_complete"])

    def test_gap_policy_check_is_promoted_in_comparison(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertTrue(result["gap_policy_check"]["violations"])
        self.assertEqual(
            result["austere_survival_summary"]["gap_policy_violations"],
            result["gap_policy_check"]["violations"],
        )

    def test_shaping_dependent_behaviors_reports_scenario_gaps(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        dependent = result["shaping_dependent_behaviors"]
        self.assertEqual(len(dependent), 1)
        self.assertEqual(dependent[0]["scenario"], "open_field_foraging")
        self.assertEqual(dependent[0]["profile"], "classic")
        self.assertAlmostEqual(dependent[0]["success_rate_delta"], 1.0)

    def test_behavior_survival_flag_true_above_threshold(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        survival = result["behavior_survival"]
        self.assertTrue(survival["scenarios"]["night_rest"]["survives"])
        self.assertFalse(survival["scenarios"]["open_field_foraging"]["survives"])

    def test_behavior_survival_rate_summary_is_computed(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertAlmostEqual(result["behavior_survival"]["survival_rate"], 0.5)
        self.assertAlmostEqual(result["survival_rate"], 0.5)

    def test_behavior_survival_reports_austere_scenario_success(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        survival = result["behavior_survival"]

        self.assertTrue(survival["available"])
        self.assertEqual(survival["scenario_count"], 2)
        self.assertEqual(survival["surviving_scenario_count"], 1)
        self.assertAlmostEqual(survival["survival_rate"], 0.5)
        self.assertAlmostEqual(result["survival_rate"], 0.5)
        self.assertTrue(survival["scenarios"]["night_rest"]["survives"])
        self.assertFalse(survival["scenarios"]["open_field_foraging"]["survives"])

    def test_behavior_survival_threshold_is_minimal_shaping_survival_threshold(self) -> None:
        from spider_cortex_sim.reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertAlmostEqual(
            result["behavior_survival"]["survival_threshold"],
            MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )

    def test_behavior_survival_minimal_profile_set_correctly(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertEqual(result["behavior_survival"]["minimal_profile"], "austere")

    def test_behavior_survival_not_available_when_no_austere_profile(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 0.8, "episodes": 2},
                    },
                },
                "ecological": {
                    "summary": {
                        "scenario_success_rate": 0.7,
                        "episode_success_rate": 0.65,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 0.7, "episodes": 2},
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertFalse(result["behavior_survival"]["available"])
        self.assertAlmostEqual(result["behavior_survival"]["survival_rate"], 0.0)

    def test_behavior_survival_not_available_when_austere_has_no_suite(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.75,
                    },
                    "suite": {
                        "night_rest": {"success_rate": 0.8, "episodes": 2},
                    },
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.45,
                    },
                    # no "suite" key
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertFalse(result["behavior_survival"]["available"])

    def test_behavior_survival_scenario_exactly_at_threshold_survives(self) -> None:
        from spider_cortex_sim.reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.8},
                    "suite": {"night_rest": {"success_rate": 0.8, "episodes": 2}},
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                        "episode_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                    },
                    "suite": {
                        "night_rest": {
                            "success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                            "episodes": 2,
                        }
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        survival = result["behavior_survival"]
        self.assertTrue(survival["scenarios"]["night_rest"]["survives"])

    def test_behavior_survival_scenario_just_below_threshold_does_not_survive(self) -> None:
        from spider_cortex_sim.reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.8},
                    "suite": {"night_rest": {"success_rate": 0.8, "episodes": 2}},
                },
                "austere": {
                    "summary": {
                        "scenario_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD - 0.01,
                        "episode_success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD - 0.01,
                    },
                    "suite": {
                        "night_rest": {
                            "success_rate": MINIMAL_SHAPING_SURVIVAL_THRESHOLD - 0.01,
                            "episodes": 2,
                        }
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        survival = result["behavior_survival"]
        self.assertFalse(survival["scenarios"]["night_rest"]["survives"])

    def test_behavior_survival_episodes_count_captured(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        survival = result["behavior_survival"]
        self.assertEqual(survival["scenarios"]["night_rest"]["episodes"], 2)
        self.assertEqual(survival["scenarios"]["open_field_foraging"]["episodes"], 2)

    def test_survival_rate_top_level_matches_behavior_survival_rate(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        self.assertAlmostEqual(
            result["survival_rate"],
            result["behavior_survival"]["survival_rate"],
        )

    def test_behavior_survival_scenario_count_uses_austere_suite_only(self) -> None:
        # Classic has an additional scenario not in austere suite
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.9, "episode_success_rate": 0.9},
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        "extra_scenario": {"success_rate": 1.0, "episodes": 2},
                    },
                },
                "austere": {
                    "summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.5},
                    "suite": {
                        "night_rest": {"success_rate": 1.0, "episodes": 2},
                        # extra_scenario present in classic but missing from austere suite
                    },
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        survival = result["behavior_survival"]
        self.assertEqual(survival["scenario_count"], 1)
        self.assertIn("night_rest", survival["scenarios"])
        self.assertNotIn("extra_scenario", survival["scenarios"])

    def test_notes_contains_behavior_survival_note(self) -> None:
        result = build_reward_audit_comparison(
            self._payload_with_austere_suite()
        )
        notes_text = " ".join(result["notes"])
        self.assertIn("behavior_survival", notes_text)

    def test_behavior_survival_zero_scenarios_when_austere_suite_is_empty_dict(self) -> None:
        payload = {
            "reward_profiles": {
                "classic": {
                    "summary": {"scenario_success_rate": 0.8, "episode_success_rate": 0.8},
                    "suite": {"night_rest": {"success_rate": 0.8, "episodes": 2}},
                },
                "austere": {
                    "summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.5},
                    "suite": {},  # empty suite
                },
            }
        }
        result = build_reward_audit_comparison(payload)
        self.assertFalse(result["behavior_survival"]["available"])

class ProfileComparisonMetricsTest(unittest.TestCase):
    """Tests for comparison.profile_comparison_metrics."""

    def test_returns_zero_defaults_for_none_input(self) -> None:
        result = profile_comparison_metrics(None)
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)
        self.assertEqual(result["mean_reward"], 0.0)

    def test_returns_zero_defaults_for_non_dict_input(self) -> None:
        result = profile_comparison_metrics("bad")
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)

    def test_extracts_from_summary_subdict(self) -> None:
        payload = {
            "summary": {
                "scenario_success_rate": 0.75,
                "episode_success_rate": 0.60,
            }
        }
        result = profile_comparison_metrics(payload)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.75)
        self.assertAlmostEqual(result["episode_success_rate"], 0.60)

    def test_extracts_from_flat_payload_when_no_summary(self) -> None:
        payload = {
            "scenario_success_rate": 0.55,
            "episode_success_rate": 0.40,
            "mean_reward": 1.23,
        }
        result = profile_comparison_metrics(payload)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.55)
        self.assertAlmostEqual(result["episode_success_rate"], 0.40)
        self.assertAlmostEqual(result["mean_reward"], 1.23)

    def test_returns_three_metric_keys(self) -> None:
        result = profile_comparison_metrics({})
        self.assertEqual(
            set(result.keys()),
            {"scenario_success_rate", "episode_success_rate", "mean_reward"},
        )

    def test_missing_rate_fields_default_to_zero(self) -> None:
        payload = {"summary": {}}
        result = profile_comparison_metrics(payload)
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)

    def test_values_are_float(self) -> None:
        payload = {
            "summary": {
                "scenario_success_rate": 1,
                "episode_success_rate": 0,
            }
        }
        result = profile_comparison_metrics(payload)
        self.assertIsInstance(result["scenario_success_rate"], float)
        self.assertIsInstance(result["episode_success_rate"], float)


if __name__ == "__main__":
    unittest.main()
