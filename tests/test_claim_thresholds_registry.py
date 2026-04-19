import json
import unittest

from spider_cortex_sim.claim_tests import (
    ClaimTestSpec,
    ScaffoldAssessment,
    ScaffoldSupportLevel,
    WARM_START_MINIMAL_THRESHOLD,
    assess_scaffold_support,
    canonical_claim_tests,
    claim_test_names,
    primary_claim_test_names,
    resolve_claim_tests,
)
from spider_cortex_sim.claim_evaluation import (
    SPECIALIZATION_ENGAGEMENT_CHECKS,
    build_claim_test_summary,
    claim_count_threshold,
    claim_leakage_audit_summary,
    claim_noise_subset_scores,
    claim_payload_config_summary,
    claim_payload_eval_reflex_scale,
    claim_registry_entry,
    claim_skip_result,
    claim_subset_scenario_success_rate,
    claim_test_source,
    claim_threshold_from_operator,
    claim_threshold_from_phrase,
    evaluate_claim_test,
    extract_claim_config_for_scaffold_assessment,
    run_claim_test_suite,
)
from spider_cortex_sim.comparison import (
    compare_noise_robustness,
    metric_seed_values_from_payload,
)
from spider_cortex_sim.claim_eval.thresholds import (
    condense_claim_test_summary as threshold_condense_claim_test_summary,
)
from spider_cortex_sim.noise import RobustnessMatrixSpec
from spider_cortex_sim.reward import SCENARIO_AUSTERE_REQUIREMENTS
from spider_cortex_sim.simulation import CAPABILITY_PROBE_SCENARIOS

from tests.fixtures.claim_tests import (
    _behavior_payload,
    _representation_metrics,
    _scaffold_config_summary,
    _add_seed_level_success_rates,
    _austere_payload,
)

class ClaimThresholdFromOperatorTest(unittest.TestCase):
    """Tests for claim_threshold_from_operator()."""

    def test_thresholds_module_reexports_condense_claim_test_summary(self) -> None:
        result = threshold_condense_claim_test_summary(
            {
                "claims": {"claim": {"status": "passed", "passed": True}},
                "summary": {"claims_passed": 1},
            }
        )

        self.assertEqual(result["claims_passed"], 1)

    def test_gte_operator_extracts_value(self) -> None:
        value = claim_threshold_from_operator("score >= 0.75", ">=")
        self.assertAlmostEqual(value, 0.75)

    def test_lte_operator_extracts_value(self) -> None:
        value = claim_threshold_from_operator("gap <= 0.15", "<=")
        self.assertAlmostEqual(value, 0.15)

    def test_gt_operator_extracts_value(self) -> None:
        value = claim_threshold_from_operator("delta > 0.10", ">")
        self.assertAlmostEqual(value, 0.10)

    def test_no_match_returns_none(self) -> None:
        value = claim_threshold_from_operator("score is good", ">=")
        self.assertIsNone(value)

    def test_integer_value_is_parsed(self) -> None:
        value = claim_threshold_from_operator("count >= 3", ">=")
        self.assertAlmostEqual(value, 3.0)

    def test_negative_value_is_parsed(self) -> None:
        value = claim_threshold_from_operator("delta >= -0.05", ">=")
        self.assertAlmostEqual(value, -0.05)

    def test_operator_with_spaces_is_parsed(self) -> None:
        value = claim_threshold_from_operator("score >=  0.60", ">=")
        self.assertAlmostEqual(value, 0.60)

    def test_first_match_is_returned_when_multiple(self) -> None:
        value = claim_threshold_from_operator("a >= 0.5 and b >= 0.8", ">=")
        self.assertAlmostEqual(value, 0.5)

class ClaimThresholdFromPhraseTest(unittest.TestCase):
    """Tests for claim_threshold_from_phrase()."""

    def test_by_at_least_phrase_extracts_value(self) -> None:
        value = claim_threshold_from_phrase(
            "trained agent improves by at least 0.15", "by at least"
        )
        self.assertAlmostEqual(value, 0.15)

    def test_no_match_returns_none(self) -> None:
        value = claim_threshold_from_phrase("score is good", "by at least")
        self.assertIsNone(value)

    def test_custom_phrase_extracts_value(self) -> None:
        value = claim_threshold_from_phrase(
            "representation_specialization_score >= 0.10", "representation_specialization_score >="
        )
        self.assertAlmostEqual(value, 0.10)

    def test_negative_threshold_is_parsed(self) -> None:
        value = claim_threshold_from_phrase("delta changes by at least -0.05", "by at least")
        self.assertAlmostEqual(value, -0.05)

    def test_phrase_with_spaces_after_matched(self) -> None:
        value = claim_threshold_from_phrase("improves by at least  0.20", "by at least")
        self.assertAlmostEqual(value, 0.20)

class ClaimCountThresholdTest(unittest.TestCase):
    """Tests for claim_count_threshold()."""

    def test_extracts_count_from_at_least_phrase(self) -> None:
        count = claim_count_threshold("at least 2 of the 3 scenarios must pass")
        self.assertEqual(count, 2)

    def test_no_match_returns_none(self) -> None:
        count = claim_count_threshold("all scenarios must pass")
        self.assertIsNone(count)

    def test_larger_count_is_parsed(self) -> None:
        count = claim_count_threshold("at least 5 of the 7 checks must be engaged")
        self.assertEqual(count, 5)

    def test_count_of_one_is_parsed(self) -> None:
        count = claim_count_threshold("at least 1 of the 3 checks engaged")
        self.assertEqual(count, 1)

    def test_partial_phrase_no_of_the_returns_none(self) -> None:
        count = claim_count_threshold("at least 2 checks engaged")
        self.assertIsNone(count)

class ClaimRegistryEntryTest(unittest.TestCase):
    """Tests for claim_registry_entry()."""

    def test_valid_entry_returned_with_no_error(self) -> None:
        payload = {
            "variants": {
                "modular_full": {"config": {}, "suite": {}}
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNotNone(entry)
        self.assertIsNone(error)

    def test_none_payload_returns_error(self) -> None:
        entry, error = claim_registry_entry(
            None, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)
        self.assertIn("variants", error)

    def test_missing_registry_key_returns_error(self) -> None:
        payload = {"conditions": {"ref": {}}}
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)

    def test_missing_entry_name_returns_error(self) -> None:
        payload = {"variants": {"other_variant": {}}}
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)
        self.assertIn("modular_full", error)

    def test_skipped_entry_returns_none_with_skip_reason(self) -> None:
        payload = {
            "variants": {
                "skipped_variant": {
                    "skipped": True,
                    "reason": "Not enough data",
                }
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="skipped_variant"
        )
        self.assertIsNone(entry)
        self.assertEqual(error, "Not enough data")

    def test_skipped_without_reason_returns_default_message(self) -> None:
        payload = {
            "variants": {
                "skipped_variant": {"skipped": True}
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="skipped_variant"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)

    def test_non_dict_registry_value_returns_error(self) -> None:
        payload = {"variants": "not_a_dict"}
        entry, error = claim_registry_entry(
            payload, registry_key="variants", entry_name="modular_full"
        )
        self.assertIsNone(entry)
        self.assertIsNotNone(error)

    def test_conditions_registry_key_works(self) -> None:
        payload = {
            "conditions": {
                "random_init": {"config": {}, "suite": {}}
            }
        }
        entry, error = claim_registry_entry(
            payload, registry_key="conditions", entry_name="random_init"
        )
        self.assertIsNotNone(entry)
        self.assertIsNone(error)
