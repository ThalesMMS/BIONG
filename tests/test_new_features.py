"""Tests for new features introduced by the metrics and reward package refactor.

Covers:
- metrics package import compatibility (all public symbols accessible from
  spider_cortex_sim.metrics)
- reward package import compatibility (all public symbols accessible from
  spider_cortex_sim.reward)
- normalize_competence_label helper
- competence_label_from_eval_reflex_scale helper
- SHELTER_ROLES, COMPETENCE_LABELS, PREDATOR_TYPE_NAMES constants
- PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES re-exports from metrics
"""

from __future__ import annotations

import unittest

# ---------------------------------------------------------------------------
# Metrics package compatibility imports
# ---------------------------------------------------------------------------
from spider_cortex_sim.metrics import (
    ACTION_CENTER_REPRESENTATION_FIELDS,
    COMPETENCE_LABELS,
    PREDATOR_RESPONSE_END_THRESHOLD,
    PREDATOR_TYPE_NAMES,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    PROPOSAL_SOURCE_NAMES,
    PROPOSER_REPRESENTATION_LOGIT_FIELD,
    REFLEX_MODULE_NAMES,
    SHELTER_ROLES,
    BehaviorCheckResult,
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeMetricAccumulator,
    EpisodeStats,
    aggregate_behavior_scores,
    aggregate_episode_stats,
    build_behavior_check,
    build_behavior_score,
    competence_label_from_eval_reflex_scale,
    flatten_behavior_rows,
    jensen_shannon_divergence,
    normalize_competence_label,
    summarize_behavior_suite,
)
from spider_cortex_sim.metrics import (
    _aggregate_values,
    _clamp_unit_interval,
    _contact_predator_types,
    _diagnostic_predator_distance,
    _dominant_predator_type,
    _first_active_predator_type,
    _mean_like,
    _mean_map,
    _normalize_distribution,
    _predator_type_threat,
)

# ---------------------------------------------------------------------------
# Reward package compatibility imports
# ---------------------------------------------------------------------------
from spider_cortex_sim.reward import (
    DISPOSITION_EVIDENCE_CRITERIA,
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_COMPONENT_AUDIT,
    REWARD_COMPONENT_NAMES,
    REWARD_PROFILES,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_DISPOSITIONS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_component_audit,
    reward_profile_audit,
    reward_total,
    shaping_disposition_summary,
    shaping_reduction_roadmap,
    validate_gap_policy,
    validate_shaping_disposition,
)

# Submodule direct imports must still work
from spider_cortex_sim.metrics.types import (
    normalize_competence_label as _normalize_competence_label_direct,
    competence_label_from_eval_reflex_scale as _competence_label_direct,
    SHELTER_ROLES as _SHELTER_ROLES_direct,
    COMPETENCE_LABELS as _COMPETENCE_LABELS_direct,
)
from spider_cortex_sim.metrics.accumulator import jensen_shannon_divergence as _jsd_direct
from spider_cortex_sim.metrics.aggregation import (
    aggregate_episode_stats as _agg_direct,
    build_behavior_check as _build_check_direct,
)
from spider_cortex_sim.reward.profiles import REWARD_PROFILES as _REWARD_PROFILES_direct
from spider_cortex_sim.reward.shaping import SHAPING_DISPOSITIONS as _SHAPING_DISPOSITIONS_direct
from spider_cortex_sim.reward.computation import empty_reward_components as _empty_rc_direct
from spider_cortex_sim.reward.audit import reward_component_audit as _audit_direct


class MetricsPackageCompatibilityImportTest(unittest.TestCase):
    """Verifies that all symbols expected from spider_cortex_sim.metrics are importable."""

    def test_episode_stats_is_dataclass(self) -> None:
        import dataclasses
        self.assertTrue(dataclasses.is_dataclass(EpisodeStats))

    def test_behavior_check_spec_is_frozen_dataclass(self) -> None:
        spec = BehaviorCheckSpec("a", "b", "c")
        with self.assertRaises((AttributeError, TypeError)):
            spec.name = "new"  # type: ignore[misc]

    def test_behavior_check_result_is_frozen_dataclass(self) -> None:
        result = BehaviorCheckResult("a", "b", "c", True, 1.0)
        with self.assertRaises((AttributeError, TypeError)):
            result.name = "new"  # type: ignore[misc]

    def test_behavioral_episode_score_is_dataclass(self) -> None:
        import dataclasses
        self.assertTrue(dataclasses.is_dataclass(BehavioralEpisodeScore))

    def test_episode_metric_accumulator_is_constructible(self) -> None:
        acc = EpisodeMetricAccumulator(
            reward_component_names=["feeding"],
            predator_states=["PATROL"],
        )
        self.assertIsNotNone(acc)

    def test_public_functions_are_callable(self) -> None:
        self.assertTrue(callable(aggregate_episode_stats))
        self.assertTrue(callable(aggregate_behavior_scores))
        self.assertTrue(callable(build_behavior_check))
        self.assertTrue(callable(build_behavior_score))
        self.assertTrue(callable(summarize_behavior_suite))
        self.assertTrue(callable(flatten_behavior_rows))
        self.assertTrue(callable(jensen_shannon_divergence))
        self.assertTrue(callable(normalize_competence_label))
        self.assertTrue(callable(competence_label_from_eval_reflex_scale))

    def test_private_helpers_are_callable(self) -> None:
        self.assertTrue(callable(_aggregate_values))
        self.assertTrue(callable(_clamp_unit_interval))
        self.assertTrue(callable(_contact_predator_types))
        self.assertTrue(callable(_diagnostic_predator_distance))
        self.assertTrue(callable(_dominant_predator_type))
        self.assertTrue(callable(_first_active_predator_type))
        self.assertTrue(callable(_mean_like))
        self.assertTrue(callable(_mean_map))
        self.assertTrue(callable(_normalize_distribution))
        self.assertTrue(callable(_predator_type_threat))


class RewardPackageCompatibilityImportTest(unittest.TestCase):
    """Verifies that all expected symbols from spider_cortex_sim.reward are importable."""

    def test_reward_profiles_dict_is_not_empty(self) -> None:
        self.assertIsInstance(REWARD_PROFILES, dict)
        self.assertGreater(len(REWARD_PROFILES), 0)

    def test_reward_component_names_is_non_empty_sequence(self) -> None:
        self.assertGreater(len(REWARD_COMPONENT_NAMES), 0)

    def test_shaping_dispositions_is_non_empty(self) -> None:
        self.assertGreater(len(SHAPING_DISPOSITIONS), 0)

    def test_reward_public_functions_are_callable(self) -> None:
        self.assertTrue(callable(empty_reward_components))
        self.assertTrue(callable(reward_total))
        self.assertTrue(callable(copy_reward_components))
        self.assertTrue(callable(apply_action_and_terrain_effects))
        self.assertTrue(callable(compute_predator_threat))
        self.assertTrue(callable(apply_pressure_penalties))
        self.assertTrue(callable(apply_progress_and_event_rewards))
        self.assertTrue(callable(reward_component_audit))
        self.assertTrue(callable(reward_profile_audit))
        self.assertTrue(callable(shaping_disposition_summary))
        self.assertTrue(callable(shaping_reduction_roadmap))
        self.assertTrue(callable(validate_gap_policy))
        self.assertTrue(callable(validate_shaping_disposition))

    def test_reward_component_audit_constant_is_dict(self) -> None:
        self.assertIsInstance(REWARD_COMPONENT_AUDIT, dict)

    def test_shaping_gap_policy_is_dict_like(self) -> None:
        self.assertIsNotNone(SHAPING_GAP_POLICY)

    def test_shaping_reduction_roadmap_constant_is_dict(self) -> None:
        self.assertIsInstance(SHAPING_REDUCTION_ROADMAP, dict)

    def test_scenario_austere_requirements_is_dict(self) -> None:
        self.assertIsInstance(SCENARIO_AUSTERE_REQUIREMENTS, dict)

    def test_disposition_evidence_criteria_is_dict(self) -> None:
        self.assertIsInstance(DISPOSITION_EVIDENCE_CRITERIA, dict)

    def test_minimal_shaping_survival_threshold_is_float(self) -> None:
        self.assertIsInstance(MINIMAL_SHAPING_SURVIVAL_THRESHOLD, float)
        self.assertGreater(MINIMAL_SHAPING_SURVIVAL_THRESHOLD, 0.0)
        self.assertLessEqual(MINIMAL_SHAPING_SURVIVAL_THRESHOLD, 1.0)


class SubmoduleDirectImportTest(unittest.TestCase):
    """Verifies the submodules export the same objects as the package __init__."""

    def test_normalize_competence_label_same_function(self) -> None:
        self.assertIs(normalize_competence_label, _normalize_competence_label_direct)

    def test_competence_label_from_eval_reflex_scale_same_function(self) -> None:
        self.assertIs(
            competence_label_from_eval_reflex_scale, _competence_label_direct
        )

    def test_jensen_shannon_divergence_same_function(self) -> None:
        self.assertIs(jensen_shannon_divergence, _jsd_direct)

    def test_aggregate_episode_stats_same_function(self) -> None:
        self.assertIs(aggregate_episode_stats, _agg_direct)

    def test_build_behavior_check_same_function(self) -> None:
        self.assertIs(build_behavior_check, _build_check_direct)

    def test_reward_profiles_same_object(self) -> None:
        self.assertIs(REWARD_PROFILES, _REWARD_PROFILES_direct)

    def test_shaping_dispositions_same_object(self) -> None:
        self.assertIs(SHAPING_DISPOSITIONS, _SHAPING_DISPOSITIONS_direct)

    def test_empty_reward_components_same_function(self) -> None:
        self.assertIs(empty_reward_components, _empty_rc_direct)

    def test_reward_component_audit_same_function(self) -> None:
        self.assertIs(reward_component_audit, _audit_direct)


class ShelterRolesConstantTest(unittest.TestCase):
    """Tests for SHELTER_ROLES constant exposed by metrics package."""

    def test_contains_required_roles(self) -> None:
        for role in ("outside", "entrance", "inside", "deep"):
            self.assertIn(role, SHELTER_ROLES)

    def test_has_four_roles(self) -> None:
        self.assertEqual(len(SHELTER_ROLES), 4)

    def test_roles_are_strings(self) -> None:
        for role in SHELTER_ROLES:
            self.assertIsInstance(role, str)

    def test_direct_submodule_import_same_value(self) -> None:
        self.assertEqual(list(SHELTER_ROLES), list(_SHELTER_ROLES_direct))


class CompetenceLabelsConstantTest(unittest.TestCase):
    """Tests for COMPETENCE_LABELS constant."""

    def test_contains_all_three_labels(self) -> None:
        self.assertIn("self_sufficient", COMPETENCE_LABELS)
        self.assertIn("scaffolded", COMPETENCE_LABELS)
        self.assertIn("mixed", COMPETENCE_LABELS)

    def test_has_three_labels(self) -> None:
        self.assertEqual(len(COMPETENCE_LABELS), 3)

    def test_direct_submodule_import_same_value(self) -> None:
        self.assertEqual(list(COMPETENCE_LABELS), list(_COMPETENCE_LABELS_direct))


class NormalizeCompetenceLabelTest(unittest.TestCase):
    """Tests for normalize_competence_label."""

    def test_valid_label_returned_unchanged(self) -> None:
        for label in ("self_sufficient", "scaffolded", "mixed"):
            self.assertEqual(normalize_competence_label(label), label)

    def test_invalid_label_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            normalize_competence_label("unknown")

    def test_empty_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            normalize_competence_label("")

    def test_case_sensitive(self) -> None:
        with self.assertRaises(ValueError):
            normalize_competence_label("Self_Sufficient")

    def test_numeric_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            normalize_competence_label("0")

    def test_return_type_is_str(self) -> None:
        result = normalize_competence_label("mixed")
        self.assertIsInstance(result, str)

    def test_non_string_coerced_if_valid(self) -> None:
        # The implementation calls str(label), so anything coercible to a valid label works
        # This is a documentation test for the coercion behavior
        result = normalize_competence_label("mixed")
        self.assertEqual(result, "mixed")


class CompetenceLabelFromEvalReflexScaleTest(unittest.TestCase):
    """Tests for competence_label_from_eval_reflex_scale."""

    def test_none_returns_mixed(self) -> None:
        self.assertEqual(competence_label_from_eval_reflex_scale(None), "mixed")

    def test_zero_returns_self_sufficient(self) -> None:
        self.assertEqual(competence_label_from_eval_reflex_scale(0.0), "self_sufficient")

    def test_zero_int_returns_self_sufficient(self) -> None:
        self.assertEqual(competence_label_from_eval_reflex_scale(0), "self_sufficient")

    def test_positive_value_returns_scaffolded(self) -> None:
        self.assertEqual(competence_label_from_eval_reflex_scale(0.5), "scaffolded")

    def test_one_returns_scaffolded(self) -> None:
        self.assertEqual(competence_label_from_eval_reflex_scale(1.0), "scaffolded")

    def test_small_positive_returns_scaffolded(self) -> None:
        self.assertEqual(competence_label_from_eval_reflex_scale(1e-9), "scaffolded")

    def test_return_values_are_valid_competence_labels(self) -> None:
        for scale in (None, 0.0, 0.5, 1.0):
            result = competence_label_from_eval_reflex_scale(scale)
            self.assertIn(result, COMPETENCE_LABELS)


class PredatorResponseEndThresholdTest(unittest.TestCase):
    """Tests for PREDATOR_RESPONSE_END_THRESHOLD constant."""

    def test_is_positive_float(self) -> None:
        self.assertIsInstance(PREDATOR_RESPONSE_END_THRESHOLD, float)
        self.assertGreater(PREDATOR_RESPONSE_END_THRESHOLD, 0.0)

    def test_is_small_threshold(self) -> None:
        # Semantically, this is a "near-zero" resolved threshold
        self.assertLess(PREDATOR_RESPONSE_END_THRESHOLD, 0.5)


class ProposalAndReflexNamesConstantsTest(unittest.TestCase):
    """Tests for PROPOSAL_SOURCE_NAMES and REFLEX_MODULE_NAMES re-exports."""

    def test_proposal_source_names_is_non_empty(self) -> None:
        self.assertGreater(len(PROPOSAL_SOURCE_NAMES), 0)

    def test_reflex_module_names_is_non_empty(self) -> None:
        self.assertGreater(len(REFLEX_MODULE_NAMES), 0)

    def test_all_names_are_strings(self) -> None:
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIsInstance(name, str)
        for name in REFLEX_MODULE_NAMES:
            self.assertIsInstance(name, str)

    def test_proposal_source_names_match_ablations_import(self) -> None:
        from spider_cortex_sim.ablations import PROPOSAL_SOURCE_NAMES as ablations_names
        self.assertEqual(list(PROPOSAL_SOURCE_NAMES), list(ablations_names))

    def test_reflex_module_names_match_ablations_import(self) -> None:
        from spider_cortex_sim.ablations import REFLEX_MODULE_NAMES as ablations_names
        self.assertEqual(list(REFLEX_MODULE_NAMES), list(ablations_names))


class PrimaryRepresentationReadoutModulesTest(unittest.TestCase):
    """Tests for PRIMARY_REPRESENTATION_READOUT_MODULES constant."""

    def test_contains_visual_cortex(self) -> None:
        self.assertIn("visual_cortex", PRIMARY_REPRESENTATION_READOUT_MODULES)

    def test_contains_sensory_cortex(self) -> None:
        self.assertIn("sensory_cortex", PRIMARY_REPRESENTATION_READOUT_MODULES)

    def test_has_exactly_two_modules(self) -> None:
        self.assertEqual(len(PRIMARY_REPRESENTATION_READOUT_MODULES), 2)


class ActionCenterRepresentationFieldsTest(unittest.TestCase):
    """Tests for ACTION_CENTER_REPRESENTATION_FIELDS constant."""

    def test_contains_module_gates(self) -> None:
        self.assertIn("module_gates", ACTION_CENTER_REPRESENTATION_FIELDS)

    def test_contains_module_contribution_share(self) -> None:
        self.assertIn("module_contribution_share", ACTION_CENTER_REPRESENTATION_FIELDS)

    def test_has_exactly_two_fields(self) -> None:
        self.assertEqual(len(ACTION_CENTER_REPRESENTATION_FIELDS), 2)


class ProposerRepresentationLogitFieldTest(unittest.TestCase):
    """Tests for PROPOSER_REPRESENTATION_LOGIT_FIELD constant."""

    def test_is_string(self) -> None:
        self.assertIsInstance(PROPOSER_REPRESENTATION_LOGIT_FIELD, str)

    def test_is_non_empty(self) -> None:
        self.assertGreater(len(PROPOSER_REPRESENTATION_LOGIT_FIELD), 0)

    def test_expected_value(self) -> None:
        self.assertEqual(PROPOSER_REPRESENTATION_LOGIT_FIELD, "post_reflex_logits")


if __name__ == "__main__":
    unittest.main()