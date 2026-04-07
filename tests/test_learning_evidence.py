import unittest

from spider_cortex_sim.learning_evidence import (
    LearningEvidenceConditionSpec,
    canonical_learning_evidence_condition_names,
    canonical_learning_evidence_conditions,
    resolve_learning_evidence_conditions,
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
            eval_reflex_scale=0.0,
            checkpoint_source="initial",
            supports_architectures=("modular",),
        )
        self.assertEqual(spec.name, "reflex_only")
        self.assertEqual(spec.policy_mode, "reflex_only")
        self.assertEqual(spec.train_budget, "none")
        self.assertEqual(spec.eval_reflex_scale, 0.0)
        self.assertEqual(spec.checkpoint_source, "initial")
        self.assertEqual(spec.supports_architectures, ("modular",))


class CanonicalLearningEvidenceConditionsTest(unittest.TestCase):
    EXPECTED_NAMES = (
        "trained_final",
        "trained_without_reflex_support",
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

    def test_trained_without_reflex_support_has_zero_eval_reflex_scale(self) -> None:
        spec = canonical_learning_evidence_conditions()["trained_without_reflex_support"]
        self.assertEqual(spec.eval_reflex_scale, 0.0)
        self.assertEqual(spec.train_budget, "base")
        self.assertEqual(spec.checkpoint_source, "final")

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


if __name__ == "__main__":
    unittest.main()