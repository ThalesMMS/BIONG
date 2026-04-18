import unittest

from spider_cortex_sim.comparison import (
    build_learning_evidence_summary,
    compare_learning_evidence,
    compare_training_regimes,
)
from spider_cortex_sim.learning_evidence import (
    LearningEvidenceConditionSpec,
    canonical_learning_evidence_condition_names,
    canonical_learning_evidence_conditions,
    resolve_learning_evidence_conditions,
)
from spider_cortex_sim.metrics import summarize_behavior_suite
from spider_cortex_sim.simulation import (
    EXPERIMENT_OF_RECORD_REGIME,
    SpiderSimulation,
)


class LearningEvidenceConditionSpecTest(unittest.TestCase):
    def test_frozen_dataclass_is_immutable(self) -> None:
        spec = LearningEvidenceConditionSpec(name="x", description="y")
        with self.assertRaises((AttributeError, TypeError)):
            spec.name = "z"  # type: ignore[misc]

    def test_default_policy_mode_is_normal(self) -> None:
        spec = LearningEvidenceConditionSpec(name="a", description="b")
        self.assertEqual(spec.policy_mode, "normal")

    def test_default_train_budget_is_base(self) -> None:
        spec = LearningEvidenceConditionSpec(name="a", description="b")
        self.assertEqual(spec.train_budget, "base")

    def test_default_eval_reflex_scale_is_none(self) -> None:
        spec = LearningEvidenceConditionSpec(name="a", description="b")
        self.assertIsNone(spec.eval_reflex_scale)

    def test_default_training_regime_is_none(self) -> None:
        spec = LearningEvidenceConditionSpec(name="a", description="b")
        self.assertIsNone(spec.training_regime)

    def test_default_checkpoint_source_is_final(self) -> None:
        spec = LearningEvidenceConditionSpec(name="a", description="b")
        self.assertEqual(spec.checkpoint_source, "final")

    def test_default_supports_architectures(self) -> None:
        spec = LearningEvidenceConditionSpec(name="a", description="b")
        self.assertIn("modular", spec.supports_architectures)
        self.assertIn("monolithic", spec.supports_architectures)

    def test_custom_fields_are_stored(self) -> None:
        spec = LearningEvidenceConditionSpec(
            name="reflex_only",
            description="desc",
            policy_mode="reflex_only",
            train_budget="none",
            training_regime="reflex_annealed",
            eval_reflex_scale=0.0,
            checkpoint_source="initial",
            supports_architectures=("modular",),
        )
        self.assertEqual(spec.name, "reflex_only")
        self.assertEqual(spec.policy_mode, "reflex_only")
        self.assertEqual(spec.train_budget, "none")
        self.assertEqual(spec.training_regime, "reflex_annealed")
        self.assertEqual(spec.eval_reflex_scale, 0.0)
        self.assertEqual(spec.checkpoint_source, "initial")
        self.assertEqual(spec.supports_architectures, ("modular",))


class CanonicalLearningEvidenceConditionsTest(unittest.TestCase):
    EXPECTED_NAMES = (
        "trained_final",
        "trained_without_reflex_support",
        "trained_reflex_annealed",
        "trained_late_finetuning",
        "random_init",
        "reflex_only",
        "freeze_half_budget",
        "trained_long_budget",
    )

    def test_returns_all_expected_condition_names(self) -> None:
        registry = canonical_learning_evidence_conditions()
        for name in self.EXPECTED_NAMES:
            self.assertIn(name, registry)

    def test_returns_dict_of_condition_specs(self) -> None:
        registry = canonical_learning_evidence_conditions()
        for spec in registry.values():
            self.assertIsInstance(spec, LearningEvidenceConditionSpec)

    def test_trained_final_uses_base_budget_and_final_source(self) -> None:
        spec = canonical_learning_evidence_conditions()["trained_final"]
        self.assertEqual(spec.train_budget, "base")
        self.assertEqual(spec.checkpoint_source, "final")
        self.assertEqual(spec.policy_mode, "normal")
        self.assertIsNone(spec.eval_reflex_scale)
        description = spec.description.lower()
        self.assertIn("default runtime configuration", description)
        self.assertIn("secondary diagnostic", description)

    def test_trained_without_reflex_support_has_zero_eval_reflex_scale(self) -> None:
        spec = canonical_learning_evidence_conditions()["trained_without_reflex_support"]
        self.assertEqual(spec.eval_reflex_scale, 0.0)
        self.assertEqual(spec.train_budget, "base")
        self.assertEqual(spec.checkpoint_source, "final")
        description = spec.description.lower()
        self.assertIn("primary", description)
        self.assertIn("reflex support disabled", description)

    def test_trained_reflex_annealed_uses_named_regime(self) -> None:
        spec = canonical_learning_evidence_conditions()["trained_reflex_annealed"]
        self.assertEqual(spec.training_regime, "reflex_annealed")
        self.assertEqual(spec.eval_reflex_scale, 0.0)
        self.assertEqual(spec.train_budget, "base")

    def test_trained_late_finetuning_uses_experiment_regime(self) -> None:
        spec = canonical_learning_evidence_conditions()["trained_late_finetuning"]
        self.assertEqual(spec.training_regime, "late_finetuning")
        self.assertEqual(spec.eval_reflex_scale, 0.0)
        self.assertEqual(spec.train_budget, "base")

    def test_trained_without_reflex_support_is_primary_gating_condition(self) -> None:
        """
        Verifies that when 'trained_without_reflex_support' is used as the reference, the learning-evidence summary selects it as the primary condition and indicates no learning evidence.
        
        Constructs a set of four condition summaries with differing performance metrics, calls build_learning_evidence_summary with reference_condition="trained_without_reflex_support", and asserts:
        - the summary's "reference_condition" equals "trained_without_reflex_support",
        - the summary's "primary_condition" matches the entry for "trained_without_reflex_support",
        - the summary's "has_learning_evidence" is False.
        """
        conditions = {
            "trained_final": {
                "summary": {
                    "scenario_success_rate": 0.95,
                    "episode_success_rate": 0.95,
                    "mean_reward": 9.5,
                }
            },
            "trained_without_reflex_support": {
                "summary": {
                    "scenario_success_rate": 0.25,
                    "episode_success_rate": 0.25,
                    "mean_reward": 2.5,
                }
            },
            "random_init": {
                "summary": {
                    "scenario_success_rate": 0.10,
                    "episode_success_rate": 0.10,
                    "mean_reward": 1.0,
                }
            },
            "reflex_only": {
                "summary": {
                    "scenario_success_rate": 0.50,
                    "episode_success_rate": 0.50,
                    "mean_reward": 5.0,
                }
            },
        }

        summary = build_learning_evidence_summary(
            conditions,
            reference_condition="trained_without_reflex_support",
        )

        self.assertEqual(summary["reference_condition"], "trained_without_reflex_support")
        self.assertEqual(
            summary["primary_condition"],
            summary["trained_without_reflex_support"],
        )
        self.assertFalse(summary["has_learning_evidence"])

    def test_random_init_has_no_training(self) -> None:
        spec = canonical_learning_evidence_conditions()["random_init"]
        self.assertEqual(spec.train_budget, "none")
        self.assertEqual(spec.checkpoint_source, "initial")
        self.assertIsNone(spec.eval_reflex_scale)

    def test_reflex_only_uses_reflex_only_policy_mode(self) -> None:
        spec = canonical_learning_evidence_conditions()["reflex_only"]
        self.assertEqual(spec.policy_mode, "reflex_only")
        self.assertEqual(spec.train_budget, "none")
        self.assertEqual(spec.checkpoint_source, "initial")

    def test_reflex_only_supports_only_modular_architecture(self) -> None:
        spec = canonical_learning_evidence_conditions()["reflex_only"]
        self.assertEqual(spec.supports_architectures, ("modular",))
        self.assertNotIn("monolithic", spec.supports_architectures)

    def test_freeze_half_budget_uses_freeze_half_train_budget(self) -> None:
        spec = canonical_learning_evidence_conditions()["freeze_half_budget"]
        self.assertEqual(spec.train_budget, "freeze_half")
        self.assertEqual(spec.checkpoint_source, "frozen_half_budget")

    def test_trained_long_budget_uses_long_train_budget(self) -> None:
        spec = canonical_learning_evidence_conditions()["trained_long_budget"]
        self.assertEqual(spec.train_budget, "long")
        self.assertEqual(spec.checkpoint_source, "final")

    def test_names_match_spec_name_attribute(self) -> None:
        registry = canonical_learning_evidence_conditions()
        for key, spec in registry.items():
            self.assertEqual(key, spec.name)

    def test_returns_new_dict_each_call(self) -> None:
        a = canonical_learning_evidence_conditions()
        b = canonical_learning_evidence_conditions()
        self.assertIsNot(a, b)


class CanonicalLearningEvidenceConditionNamesTest(unittest.TestCase):
    def test_returns_tuple(self) -> None:
        names = canonical_learning_evidence_condition_names()
        self.assertIsInstance(names, tuple)

    def test_contains_all_expected_names(self) -> None:
        names = canonical_learning_evidence_condition_names()
        for expected in (
            "trained_final",
            "trained_without_reflex_support",
            "trained_reflex_annealed",
            "trained_late_finetuning",
            "random_init",
            "reflex_only",
            "freeze_half_budget",
            "trained_long_budget",
        ):
            self.assertIn(expected, names)

    def test_matches_canonical_conditions_keys(self) -> None:
        names = canonical_learning_evidence_condition_names()
        keys = tuple(canonical_learning_evidence_conditions().keys())
        self.assertEqual(names, keys)


class ResolveLearningEvidenceConditionsTest(unittest.TestCase):
    def test_none_returns_all_conditions_in_order(self) -> None:
        specs = resolve_learning_evidence_conditions(None)
        expected_names = list(canonical_learning_evidence_condition_names())
        self.assertEqual([s.name for s in specs], expected_names)

    def test_empty_list_returns_empty(self) -> None:
        specs = resolve_learning_evidence_conditions([])
        self.assertEqual(specs, [])

    def test_single_name_returns_single_spec(self) -> None:
        specs = resolve_learning_evidence_conditions(["trained_final"])
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "trained_final")

    def test_multiple_names_returns_specs_in_order(self) -> None:
        names = ["random_init", "reflex_only", "trained_final"]
        specs = resolve_learning_evidence_conditions(names)
        self.assertEqual([s.name for s in specs], names)

    def test_returns_correct_spec_objects(self) -> None:
        registry = canonical_learning_evidence_conditions()
        specs = resolve_learning_evidence_conditions(["freeze_half_budget"])
        self.assertEqual(specs[0], registry["freeze_half_budget"])

    def test_invalid_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_learning_evidence_conditions(["nonexistent_condition"])

    def test_invalid_name_message_contains_condition_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "bad_condition"):
            resolve_learning_evidence_conditions(["bad_condition"])


class LearningEvidenceRegimeConditionIntegrationTest(unittest.TestCase):
    def test_regime_condition_trains_with_named_regime(self) -> None:
        payload, rows = compare_learning_evidence(
            budget_profile="smoke",
            episodes=1,
            evaluation_episodes=0,
            names=("night_rest",),
            condition_names=("trained_reflex_annealed",),
            seeds=(7,),
            episodes_per_scenario=1,
        )

        condition = payload["conditions"]["trained_reflex_annealed"]
        self.assertEqual(condition["training_regime"], "reflex_annealed")
        self.assertEqual(condition["summary"]["competence_type"], "self_sufficient")
        self.assertEqual(condition["summary"]["eval_reflex_scale"], 0.0)
        regime_rows = [
            row
            for row in rows
            if row["learning_evidence_condition"] == "trained_reflex_annealed"
        ]
        self.assertTrue(regime_rows)
        self.assertTrue(
            all(row["training_regime_name"] == "reflex_annealed" for row in regime_rows)
        )
        self.assertTrue(
            all(
                row["learning_evidence_training_regime"] == "reflex_annealed"
                for row in regime_rows
            )
        )

    def test_default_eval_scale_records_runtime_scale(self) -> None:
        payload, rows = compare_learning_evidence(
            budget_profile="smoke",
            episodes=1,
            evaluation_episodes=0,
            names=("night_rest",),
            condition_names=("trained_final",),
            seeds=(7,),
            episodes_per_scenario=1,
        )

        condition = payload["conditions"]["trained_final"]
        self.assertEqual(condition["summary"]["eval_reflex_scale"], 1.0)
        self.assertEqual(condition["summary"]["competence_type"], "scaffolded")
        trained_rows = [
            row for row in rows if row["learning_evidence_condition"] == "trained_final"
        ]
        self.assertTrue(trained_rows)
        self.assertTrue(
            all(float(row["eval_reflex_scale"]) == 1.0 for row in trained_rows)
        )
        self.assertTrue(
            all(row["competence_type"] == "scaffolded" for row in trained_rows)
        )

    def test_mix_of_valid_and_invalid_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_learning_evidence_conditions(["trained_final", "not_a_condition"])

    def test_duplicate_names_returns_duplicate_specs(self) -> None:
        specs = resolve_learning_evidence_conditions(["trained_final", "trained_final"])
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].name, "trained_final")
        self.assertEqual(specs[1].name, "trained_final")

    def test_coerces_names_to_str(self) -> None:
        # Passing an iterable that yields non-str should be coerced
        class StrLike:
            def __str__(self) -> str:
                return "random_init"

        specs = resolve_learning_evidence_conditions([StrLike()])
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "random_init")


class TrainingRegimeComparisonPayloadTest(unittest.TestCase):
    def test_compare_training_regimes_returns_structured_payload(self) -> None:
        payload, rows = compare_training_regimes(
            regime_names=["reflex_annealed"],
            budget_profile="smoke",
            episodes=1,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
            episodes_per_scenario=1,
        )

        self.assertEqual(payload["comparison_type"], "training_regimes")
        self.assertEqual(payload["reference_regime"], "baseline")
        self.assertEqual(
            payload["experiment_of_record_regime"],
            EXPERIMENT_OF_RECORD_REGIME,
        )
        self.assertIn("baseline", payload["regimes"])
        self.assertIn("reflex_annealed", payload["regimes"])
        reflex_payload = payload["regimes"]["reflex_annealed"]
        self.assertIn("success_rates", reflex_payload)
        self.assertIn("self_sufficient", reflex_payload["success_rates"])
        self.assertIn("scaffolded", reflex_payload["success_rates"])
        self.assertEqual(
            reflex_payload["primary_benchmark"]["summary"]["competence_type"],
            "self_sufficient",
        )
        self.assertIn("scenario_success_rate_delta", reflex_payload["competence_gap"])
        self.assertIn("reflex_annealed", payload["deltas_vs_baseline"])
        self.assertTrue(rows)
        self.assertTrue(any(row["competence_type"] == "self_sufficient" for row in rows))
        self.assertTrue(any(row["competence_type"] == "scaffolded" for row in rows))


class CompetenceLabelingEvaluationSummaryTest(unittest.TestCase):
    def test_summarize_behavior_suite_defaults_to_mixed(self) -> None:
        summary = summarize_behavior_suite({})

        self.assertEqual(summary["competence_type"], "mixed")

    def test_evaluation_summary_labels_self_sufficient_when_reflex_scale_zero(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=1)

        payload, _, rows = sim.evaluate_behavior_suite(
            names=["night_rest"],
            episodes_per_scenario=1,
            eval_reflex_scale=0.0,
        )

        self.assertEqual(payload["summary"]["competence_type"], "self_sufficient")
        self.assertEqual(payload["summary"]["eval_reflex_scale"], 0.0)
        self.assertTrue(rows)
        self.assertTrue(all(row["competence_type"] == "self_sufficient" for row in rows))
        self.assertTrue(all(row["is_primary_benchmark"] for row in rows))

    def test_evaluation_summary_labels_scaffolded_when_reflex_scale_positive(self) -> None:
        """
        Verifies that evaluate_behavior_suite labels competence as "scaffolded" when eval_reflex_scale is positive.
        
        Asserts the payload summary's `competence_type` is "scaffolded" and `eval_reflex_scale` equals 0.5, that rows are returned, every row has `competence_type == "scaffolded"`, and no row is marked as the primary benchmark (`is_primary_benchmark` is False).
        """
        sim = SpiderSimulation(seed=7, max_steps=1)

        payload, _, rows = sim.evaluate_behavior_suite(
            names=["night_rest"],
            episodes_per_scenario=1,
            eval_reflex_scale=0.5,
        )

        self.assertEqual(payload["summary"]["competence_type"], "scaffolded")
        self.assertEqual(payload["summary"]["eval_reflex_scale"], 0.5)
        self.assertTrue(rows)
        self.assertTrue(all(row["competence_type"] == "scaffolded" for row in rows))
        self.assertTrue(all(not row["is_primary_benchmark"] for row in rows))


if __name__ == "__main__":
    unittest.main()
