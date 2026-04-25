"""Focused metrics and behavior-evaluation tests."""

from __future__ import annotations

import unittest
from typing import Dict
from unittest.mock import MagicMock

from spider_cortex_sim.ablations import PROPOSAL_SOURCE_NAMES
from spider_cortex_sim.metrics import (
    ACTION_CENTER_REPRESENTATION_FIELDS,
    EpisodeMetricAccumulator,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    REFLEX_MODULE_NAMES,
    _contact_predator_types,
    _diagnostic_predator_distance,
    _diagnostic_predator_distance_for_type,
    _dominant_predator_type,
    _first_active_predator_type,
    _normalize_distribution,
    _predator_type_threat,
    _softmax_probabilities,
    jensen_shannon_divergence,
)
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES

from tests.fixtures.metrics_accumulator import _make_accumulator, _make_fake_meta, _make_fake_info, _make_fake_state

class EpisodeMetricAccumulatorReflexTest(unittest.TestCase):
    """Tests for EpisodeMetricAccumulator.record_decision reflex tracking (new in this PR)."""

    def _make_accumulator(self) -> EpisodeMetricAccumulator:
        """
        Create an EpisodeMetricAccumulator configured for tests with no reward components and a single predator state "PATROL".
        
        Returns:
            EpisodeMetricAccumulator: An accumulator initialized with reward_component_names=[] and predator_states=['PATROL'].
        """
        return EpisodeMetricAccumulator(
            reward_component_names=[],
            predator_states=["PATROL"],
        )

    def _make_decision(
        self,
        *,
        final_reflex_override: bool = False,
        module_results: list | None = None,
        arbitration_decision: object | None = None,
    ) -> object:
        """
        Create a lightweight fake decision object for tests with specified reflex and arbitration fields.
        
        Parameters:
        	final_reflex_override (bool): Whether the decision's `final_reflex_override` attribute is True.
        	module_results (list | None): Sequence of module-result-like objects to assign to `module_results`; empty list if None.
        	arbitration_decision (object | None): Optional arbitration decision object to assign to `arbitration_decision`.
        
        Returns:
        	decision (object): An object with attributes `final_reflex_override`, `module_results`, and `arbitration_decision`.
        """
        class FakeDecision:
            pass

        decision = FakeDecision()
        decision.final_reflex_override = final_reflex_override
        decision.module_results = module_results or []
        decision.arbitration_decision = arbitration_decision
        return decision

    def _make_arbitration(
        self,
        *,
        module_contribution_share: dict[str, float] | None = None,
        dominant_module: str = "alert_center",
        dominant_module_share: float = 0.0,
        effective_module_count: float = 0.0,
        module_agreement_rate: float = 0.0,
        module_disagreement_rate: float = 0.0,
    ) -> object:
        """
        Create a lightweight fake arbitration object with configurable arbitration-related attributes for tests.
        
        Parameters:
            module_contribution_share (dict[str, float] | None): Mapping of module name to its contribution share; defaults to empty dict.
            dominant_module (str): Name of the dominant module.
            dominant_module_share (float): Contribution share of the dominant module.
            effective_module_count (float): Effective number of contributing modules (can be fractional).
            module_agreement_rate (float): Fraction of modules in agreement.
            module_disagreement_rate (float): Fraction of modules in disagreement.
        
        Returns:
            object: An object whose attributes mirror the provided arguments:
                - module_contribution_share
                - dominant_module
                - dominant_module_share
                - effective_module_count
                - module_agreement_rate
                - module_disagreement_rate
        """
        class FakeArbitration:
            pass

        arbitration = FakeArbitration()
        arbitration.module_contribution_share = module_contribution_share or {}
        arbitration.dominant_module = dominant_module
        arbitration.dominant_module_share = dominant_module_share
        arbitration.effective_module_count = effective_module_count
        arbitration.module_agreement_rate = module_agreement_rate
        arbitration.module_disagreement_rate = module_disagreement_rate
        return arbitration

    def _make_module_result(
        self,
        name: str,
        *,
        reflex_applied: bool = False,
        module_reflex_override: bool = False,
        module_reflex_dominance: float = 0.0,
    ) -> object:
        """
        Create a lightweight fake module result object with specified reflex and dominance attributes.
        
        Parameters:
            name (str): Identifier for the fake module result.
            reflex_applied (bool): Whether the module applied a reflex for this decision.
            module_reflex_override (bool): Whether the module's reflex was marked as an override.
            module_reflex_dominance (float): Dominance score contributed by the module when reflexes are considered.
        
        Returns:
            object: An object with attributes `name`, `reflex_applied`, `module_reflex_override`, and `module_reflex_dominance`.
        """
        class FakeModuleResult:
            pass

        result = FakeModuleResult()
        result.name = name
        result.reflex_applied = reflex_applied
        result.module_reflex_override = module_reflex_override
        result.module_reflex_dominance = module_reflex_dominance
        return result

    def _record_motor_transition(
        self,
        acc: EpisodeMetricAccumulator,
        *,
        occurred: bool,
        terrain: str,
        orientation_alignment: float,
        terrain_difficulty: float,
    ) -> None:
        """
        Record a synthetic motor transition on the given accumulator for unit tests.
        
        This helper builds a minimal observation metadata, a fake state, and an info payload
        containing motor execution and motor slip fields, then calls acc.record_transition
        with those values so tests can assert accumulator behavior for motor slips and related metrics.
        
        Parameters:
            acc (EpisodeMetricAccumulator): The accumulator to record the transition into.
            occurred (bool): Whether a motor slip occurred (sets both `motor_noise_applied` and `motor_slip["occurred"]`).
            terrain (str): Terrain label stored as `motor_slip["terrain"]`.
            orientation_alignment (float): Value recorded under both `motor_execution_components["orientation_alignment"]` and `motor_slip["components"]["orientation_alignment"]`.
            terrain_difficulty (float): Value recorded under both `motor_execution_components["terrain_difficulty"]` and `motor_slip["components"]["terrain_difficulty"]`.
        """
        class FakeState:
            sleep_debt = 0.0
            last_move_dx = 0
            last_move_dy = 0

        meta = {
            "food_dist": 3,
            "shelter_dist": 2,
            "predator_visible": False,
            "diagnostic": {"diagnostic_predator_dist": 9},
            "night": False,
            "shelter_role": "outside",
            "on_shelter": False,
        }
        acc.record_transition(
            step=0,
            observation_meta=meta,
            next_meta=meta,
            info={
                "reward_components": {},
                "predator_contact": False,
                "motor_noise_applied": occurred,
                "motor_execution_components": {
                    "orientation_alignment": orientation_alignment,
                    "terrain_difficulty": terrain_difficulty,
                },
                "motor_slip": {
                    "occurred": occurred,
                    "terrain": terrain,
                    "components": {
                        "orientation_alignment": orientation_alignment,
                        "terrain_difficulty": terrain_difficulty,
                    },
                },
            },
            state=FakeState(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

    def test_decision_steps_increments_each_call(self) -> None:
        acc = self._make_accumulator()
        self.assertEqual(acc.decision_steps, 0)
        acc.record_decision(self._make_decision())
        self.assertEqual(acc.decision_steps, 1)
        acc.record_decision(self._make_decision())
        self.assertEqual(acc.decision_steps, 2)

    def test_no_reflex_applied_leaves_reflex_steps_zero(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=False),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 0)

    def test_reflex_applied_increments_reflex_steps(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 1)

    def test_reflex_steps_counts_steps_not_modules(self) -> None:
        # Two modules both apply reflex in one step: reflex_steps should be 1, not 2
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
                self._make_module_result("hunger_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 1)

    def test_final_reflex_override_increments_counter(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision(final_reflex_override=True))
        self.assertEqual(acc.final_reflex_override_steps, 1)

    def test_final_reflex_override_false_does_not_increment(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision(final_reflex_override=False))
        self.assertEqual(acc.final_reflex_override_steps, 0)

    def test_module_reflex_usage_tracked_by_module_name(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.module_reflex_usage_steps["alert_center"], 1)

    def test_module_reflex_override_tracked_by_module_name(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", module_reflex_override=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.module_reflex_override_steps["alert_center"], 1)

    def test_module_reflex_dominance_accumulated(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", module_reflex_dominance=0.4),
            ]
        )
        acc.record_decision(decision)
        self.assertAlmostEqual(acc.module_reflex_dominance_sums["alert_center"], 0.4)

    def test_unknown_module_name_in_results_is_ignored(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("nonexistent_module", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 0)

    def test_snapshot_includes_reflex_usage_rate(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        snapshot = acc.snapshot()
        self.assertIn("reflex_usage_rate", snapshot)
        self.assertAlmostEqual(snapshot["reflex_usage_rate"], 1.0)

    def test_snapshot_reflex_usage_rate_is_zero_when_no_reflex(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision())
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["reflex_usage_rate"], 0.0)

    def test_snapshot_includes_final_reflex_override_rate(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision(final_reflex_override=True))
        acc.record_decision(self._make_decision(final_reflex_override=False))
        snapshot = acc.snapshot()
        self.assertIn("final_reflex_override_rate", snapshot)
        self.assertAlmostEqual(snapshot["final_reflex_override_rate"], 0.5)

    def test_snapshot_includes_module_reflex_usage_rates(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision())
        snapshot = acc.snapshot()
        self.assertIn("module_reflex_usage_rates", snapshot)
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, snapshot["module_reflex_usage_rates"])

    def test_snapshot_includes_mean_reflex_dominance(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision())
        snapshot = acc.snapshot()
        self.assertIn("mean_reflex_dominance", snapshot)
        self.assertIsInstance(snapshot["mean_reflex_dominance"], float)

    def test_module_contribution_share_accumulated_from_arbitration(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(
            self._make_decision(
                arbitration_decision=self._make_arbitration(
                    module_contribution_share={"alert_center": 0.75},
                )
            )
        )
        self.assertAlmostEqual(
            acc.module_contribution_share_sums["alert_center"],
            0.75,
        )

    def test_snapshot_includes_independence_metrics(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(
            self._make_decision(
                arbitration_decision=self._make_arbitration(
                    module_contribution_share={"alert_center": 0.7, "hunger_center": 0.3},
                    dominant_module="alert_center",
                    dominant_module_share=0.7,
                    effective_module_count=1.72,
                    module_agreement_rate=0.8,
                    module_disagreement_rate=0.2,
                )
            )
        )
        snapshot = acc.snapshot()
        self.assertIn("module_contribution_share", snapshot)
        self.assertIn("dominant_module_distribution", snapshot)
        self.assertEqual(snapshot["dominant_module"], "alert_center")
        self.assertAlmostEqual(snapshot["dominant_module_share"], 0.7)
        self.assertAlmostEqual(snapshot["effective_module_count"], 1.72)
        self.assertAlmostEqual(snapshot["module_agreement_rate"], 0.8)
        self.assertAlmostEqual(snapshot["module_disagreement_rate"], 0.2)
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, snapshot["module_contribution_share"])
            self.assertIn(name, snapshot["dominant_module_distribution"])

    def test_snapshot_dominant_module_empty_when_no_decisions(self) -> None:
        acc = self._make_accumulator()

        snapshot = acc.snapshot()

        self.assertEqual(snapshot["dominant_module"], "")

    def test_snapshot_owner_alignment_uses_passed_scenario(self) -> None:
        acc = self._make_accumulator()
        acc._current_scenario_name = "night_rest"

        snapshot = acc.snapshot(scenario="olfactory_ambush")

        self.assertEqual(
            snapshot["expected_owner_modules"],
            ["sensory_cortex", "alert_center"],
        )

    def test_record_learning_accumulates_credit_weights_and_gradient_norms(self) -> None:
        acc = self._make_accumulator()
        acc.record_learning(
            {
                "module_credit_weights": {"alert_center": 0.25},
                "module_gradient_norms": {"alert_center": 1.5},
            }
        )
        acc.record_learning(
            {
                "module_credit_weights": {"alert_center": 0.75},
                "module_gradient_norms": {"alert_center": 2.5},
            }
        )
        snapshot = acc.snapshot()
        self.assertAlmostEqual(
            snapshot["mean_module_credit_weights"]["alert_center"],
            0.5,
        )
        self.assertAlmostEqual(
            snapshot["module_gradient_norm_means"]["alert_center"],
            2.0,
        )

    def test_snapshot_includes_learning_credit_maps(self) -> None:
        acc = self._make_accumulator()
        snapshot = acc.snapshot()
        self.assertIn("mean_module_credit_weights", snapshot)
        self.assertIn("module_gradient_norm_means", snapshot)
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, snapshot["mean_module_credit_weights"])
            self.assertIn(name, snapshot["module_gradient_norm_means"])

    def test_snapshot_denominator_uses_max_one_when_no_decisions(self) -> None:
        # With zero decisions, snapshot should still not raise division-by-zero
        acc = self._make_accumulator()
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["reflex_usage_rate"], 0.0)
        self.assertAlmostEqual(snapshot["final_reflex_override_rate"], 0.0)

    def test_snapshot_includes_motor_execution_metrics(self) -> None:
        acc = self._make_accumulator()
        self._record_motor_transition(
            acc,
            occurred=False,
            terrain="open",
            orientation_alignment=1.0,
            terrain_difficulty=0.0,
        )
        self._record_motor_transition(
            acc,
            occurred=True,
            terrain="narrow",
            orientation_alignment=0.25,
            terrain_difficulty=0.7,
        )
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["motor_slip_rate"], 0.5)
        self.assertAlmostEqual(snapshot["mean_orientation_alignment"], 0.625)
        self.assertAlmostEqual(snapshot["mean_terrain_difficulty"], 0.35)
        self.assertAlmostEqual(snapshot["terrain_slip_rates"]["open"], 0.0)
        self.assertAlmostEqual(snapshot["terrain_slip_rates"]["narrow"], 1.0)

    def test_post_init_initializes_all_module_reflex_dicts(self) -> None:
        acc = self._make_accumulator()
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, acc.module_reflex_usage_steps)
            self.assertIn(name, acc.module_reflex_override_steps)
            self.assertIn(name, acc.module_reflex_dominance_sums)
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, acc.module_credit_weight_sums)
            self.assertIn(name, acc.module_gradient_norm_sums)

class RecordLearningEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for EpisodeMetricAccumulator.record_learning()."""

    def _make_accumulator(self) -> EpisodeMetricAccumulator:
        """
        Create a minimal EpisodeMetricAccumulator configured for tests.
        
        Returns:
            EpisodeMetricAccumulator: An accumulator initialized with reward component names ["food", "sleep"] and predator states ["PATROL", "CHASE"].
        """
        return EpisodeMetricAccumulator(
            reward_component_names=["food", "sleep"],
            predator_states=["PATROL", "CHASE"],
        )

    def test_record_learning_increments_learning_steps(self) -> None:
        """learning_steps counter increments once per record_learning call."""
        acc = self._make_accumulator()
        self.assertEqual(acc.learning_steps, 0)
        acc.record_learning({"module_credit_weights": {}, "module_gradient_norms": {}})
        self.assertEqual(acc.learning_steps, 1)
        acc.record_learning({"module_credit_weights": {}, "module_gradient_norms": {}})
        self.assertEqual(acc.learning_steps, 2)

    def test_record_learning_with_no_credit_weights_key(self) -> None:
        """record_learning is a no-op on sums when credit_weights key is absent."""
        acc = self._make_accumulator()
        for name in PROPOSAL_SOURCE_NAMES:
            acc.module_credit_weight_sums[name] = 0.0
        acc.record_learning({"module_gradient_norms": {"alert_center": 1.0}})
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(acc.module_credit_weight_sums[name], 0.0)

    def test_record_learning_with_non_mapping_credit_weights_ignored(self) -> None:
        """A non-mapping value for 'module_credit_weights' is silently ignored."""
        acc = self._make_accumulator()
        # Should not raise
        acc.record_learning({"module_credit_weights": [1.0, 2.0]})
        acc.record_learning({"module_credit_weights": None})
        acc.record_learning({"module_credit_weights": 42.0})
        # All sums should remain 0
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(acc.module_credit_weight_sums[name], 0.0)

    def test_record_learning_with_non_mapping_gradient_norms_ignored(self) -> None:
        """A non-mapping value for 'module_gradient_norms' is silently ignored."""
        acc = self._make_accumulator()
        acc.record_learning({"module_gradient_norms": "not_a_mapping"})
        acc.record_learning({"module_gradient_norms": 99.9})
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(acc.module_gradient_norm_sums[name], 0.0)

    def test_record_learning_with_unknown_module_name_creates_new_entry(self) -> None:
        """record_learning adds new entries for module names not in PROPOSAL_SOURCE_NAMES."""
        acc = self._make_accumulator()
        acc.record_learning(
            {"module_credit_weights": {"unknown_module": 0.5}}
        )
        self.assertIn("unknown_module", acc.module_credit_weight_sums)
        self.assertAlmostEqual(acc.module_credit_weight_sums["unknown_module"], 0.5)

    def test_snapshot_mean_credit_weights_zero_when_no_learning_steps(self) -> None:
        """With no record_learning calls, mean_module_credit_weights are 0.0 (denominator = max(1, 0) = 1)."""
        acc = self._make_accumulator()
        snapshot = acc.snapshot()
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(
                snapshot["mean_module_credit_weights"][name],
                0.0,
            )

    def test_snapshot_mean_credit_weights_averages_over_learning_steps(self) -> None:
        """Mean credit weights average over all learning steps, not decision steps."""
        acc = self._make_accumulator()
        # Record 3 learning steps for a module
        for value in (0.2, 0.4, 0.6):
            acc.record_learning({"module_credit_weights": {"alert_center": value}})
        snapshot = acc.snapshot()
        # Mean of (0.2 + 0.4 + 0.6) / 3 = 0.4, not 0.2/1 = 0.2
        self.assertAlmostEqual(snapshot["mean_module_credit_weights"]["alert_center"], 0.4)

    def test_snapshot_gradient_norm_means_average_correctly(self) -> None:
        """module_gradient_norm_means averages over learning steps, not decision steps."""
        acc = self._make_accumulator()
        for value in (1.0, 3.0):
            acc.record_learning({"module_gradient_norms": {"visual_cortex": value}})
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["module_gradient_norm_means"]["visual_cortex"], 2.0)

    def test_snapshot_counterfactual_credit_weights_average_correctly(self) -> None:
        acc = self._make_accumulator()
        for value in (0.25, 0.75):
            acc.record_learning(
                {"counterfactual_credit_weights": {"alert_center": value}}
            )
        snapshot = acc.snapshot()
        self.assertAlmostEqual(
            snapshot["mean_counterfactual_credit_weights"]["alert_center"],
            0.5,
        )

    def test_record_learning_multiple_modules_in_same_step(self) -> None:
        """record_learning accumulates all modules provided in a single call."""
        acc = self._make_accumulator()
        acc.record_learning(
            {
                "module_credit_weights": {
                    "alert_center": 0.5,
                    "visual_cortex": 0.3,
                },
                "module_gradient_norms": {
                    "alert_center": 2.0,
                    "visual_cortex": 1.0,
                },
                "counterfactual_credit_weights": {
                    "alert_center": 0.6,
                    "visual_cortex": 0.4,
                },
            }
        )
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["mean_module_credit_weights"]["alert_center"], 0.5)
        self.assertAlmostEqual(snapshot["mean_module_credit_weights"]["visual_cortex"], 0.3)
        self.assertAlmostEqual(snapshot["module_gradient_norm_means"]["alert_center"], 2.0)
        self.assertAlmostEqual(snapshot["module_gradient_norm_means"]["visual_cortex"], 1.0)
        self.assertAlmostEqual(
            snapshot["mean_counterfactual_credit_weights"]["alert_center"],
            0.6,
        )
        self.assertAlmostEqual(
            snapshot["mean_counterfactual_credit_weights"]["visual_cortex"],
            0.4,
        )

    def test_record_learning_empty_dict_still_increments_counter(self) -> None:
        """An empty learn_stats dict still increments learning_steps."""
        acc = self._make_accumulator()
        acc.record_learning({})
        self.assertEqual(acc.learning_steps, 1)

class MetricsAccumulatorNewFieldsTest(unittest.TestCase):
    """Tests for new EpisodeMetricAccumulator fields introduced in this PR."""

    def test_initial_distance_fields_are_none(self) -> None:
        acc = _make_accumulator()
        self.assertIsNone(acc.initial_food_dist)
        self.assertIsNone(acc.initial_shelter_dist)
        self.assertIsNone(acc.final_food_dist)
        self.assertIsNone(acc.final_shelter_dist)

    def test_initial_predator_mode_transitions_is_zero(self) -> None:
        acc = _make_accumulator()
        self.assertEqual(acc.predator_mode_transitions, 0)

    def test_record_transition_captures_initial_food_and_shelter_dist(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(food_dist=8, shelter_dist=4)
        next_meta = _make_fake_meta(food_dist=7, shelter_dist=5)
        acc.record_transition(
            step=0,
            observation_meta=obs_meta,
            next_meta=next_meta,
            info=_make_fake_info(),
            state=_make_fake_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )
        self.assertEqual(acc.initial_food_dist, 8)
        self.assertEqual(acc.initial_shelter_dist, 4)
        self.assertEqual(acc.final_food_dist, 7)
        self.assertEqual(acc.final_shelter_dist, 5)

    def test_record_transition_does_not_overwrite_initial_distances(self) -> None:
        acc = _make_accumulator()
        obs_meta1 = _make_fake_meta(food_dist=10, shelter_dist=6)
        next_meta1 = _make_fake_meta(food_dist=9, shelter_dist=7)
        obs_meta2 = _make_fake_meta(food_dist=9, shelter_dist=7)
        next_meta2 = _make_fake_meta(food_dist=5, shelter_dist=2)
        acc.record_transition(step=0, observation_meta=obs_meta1, next_meta=next_meta1,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        acc.record_transition(step=1, observation_meta=obs_meta2, next_meta=next_meta2,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        # Initial should still be from step 0
        self.assertEqual(acc.initial_food_dist, 10)
        self.assertEqual(acc.initial_shelter_dist, 6)
        # Final should be updated to step 1's next_meta
        self.assertEqual(acc.final_food_dist, 5)
        self.assertEqual(acc.final_shelter_dist, 2)

    def test_predator_mode_transition_counted_on_state_change(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="ORIENT")
        self.assertEqual(acc.predator_mode_transitions, 1)

    def test_predator_mode_transition_not_counted_when_same_state(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="CHASE", predator_state="CHASE")
        self.assertEqual(acc.predator_mode_transitions, 0)

    def test_predator_mode_transitions_accumulate_over_multiple_steps(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        transitions = [
            ("PATROL", "ORIENT"),    # +1
            ("ORIENT", "ORIENT"),    # no change
            ("ORIENT", "CHASE"),     # +1
            ("CHASE", "WAIT"),       # +1
            ("WAIT", "PATROL"),      # +1
        ]
        for before, after in transitions:
            acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                                  info=_make_fake_info(), state=_make_fake_state(),
                                  predator_state_before=before, predator_state=after)
        self.assertEqual(acc.predator_mode_transitions, 4)

    def test_snapshot_includes_predator_mode_transitions(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="ORIENT")
        snap = acc.snapshot()
        self.assertIn("predator_mode_transitions", snap)
        self.assertEqual(snap["predator_mode_transitions"], 1)

    def test_snapshot_includes_dominant_predator_state(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        # Record mostly CHASE ticks
        for _ in range(5):
            acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                                  info=_make_fake_info(), state=_make_fake_state(),
                                  predator_state_before="PATROL", predator_state="CHASE")
        acc.record_transition(step=5, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="CHASE", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertIn("dominant_predator_state", snap)
        self.assertEqual(snap["dominant_predator_state"], "CHASE")

    def test_snapshot_dominant_predator_state_defaults_to_patrol_when_no_ticks(self) -> None:
        acc = _make_accumulator()
        # Don't record any transitions - all ticks are 0
        snap = acc.snapshot()
        self.assertEqual(snap["dominant_predator_state"], "PATROL")

    def test_snapshot_dominant_predator_state_uses_patrol_fallback_when_all_ticks_zero(self) -> None:
        acc = EpisodeMetricAccumulator(
            reward_component_names=REWARD_COMPONENT_NAMES,
            predator_states=["CHASE", "PATROL"],
        )

        snap = acc.snapshot()

        self.assertEqual(snap["dominant_predator_state"], "PATROL")

    def test_snapshot_food_distance_delta_positive_when_approaching_food(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(food_dist=10)
        next_meta = _make_fake_meta(food_dist=7)
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertAlmostEqual(snap["food_distance_delta"], 3.0)

    def test_snapshot_food_distance_delta_negative_when_moving_away(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(food_dist=3)
        next_meta = _make_fake_meta(food_dist=7)
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertAlmostEqual(snap["food_distance_delta"], -4.0)

    def test_snapshot_shelter_distance_delta_tracks_correctly(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(shelter_dist=8)
        next_meta = _make_fake_meta(shelter_dist=5)
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertAlmostEqual(snap["shelter_distance_delta"], 3.0)

    def test_record_transition_tracks_predator_contact_by_type(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(predator_visible=False)
        next_meta = _make_fake_meta(predator_visible=True, predator_dist=1)
        next_meta["dominant_predator_type_label"] = "visual"
        next_meta["visual_predator_threat"] = 0.7
        next_meta["olfactory_predator_threat"] = 0.0
        next_meta["predators"] = [
            {
                "x": 6,
                "y": 6,
                "profile": {"detection_style": "visual"},
            },
            {
                "x": 2,
                "y": 3,
                "profile": {"detection_style": "olfactory"},
            }
        ]
        state = _make_fake_state()
        state.x = 2
        state.y = 3

        acc.record_transition(
            step=0,
            observation_meta=obs_meta,
            next_meta=next_meta,
            info=_make_fake_info(predator_contact=True),
            state=state,
            predator_state_before="PATROL",
            predator_state="CHASE",
        )

        snap = acc.snapshot()
        self.assertEqual(snap["predator_contacts_by_type"]["visual"], 0)
        self.assertEqual(snap["predator_contacts_by_type"]["olfactory"], 1)

class JensenShannonDivergenceTest(unittest.TestCase):
    """Tests for the representation-specialization JS divergence helper."""

    def test_softmax_ignores_non_finite_logits(self) -> None:
        result = _softmax_probabilities([1.0, float("inf"), float("-inf")])
        self.assertEqual(result, [1.0, 0.0, 0.0])

    def test_softmax_all_non_finite_logits_returns_zeros(self) -> None:
        result = _softmax_probabilities([float("nan"), float("inf")])
        self.assertEqual(result, [0.0, 0.0])

    def test_identical_distributions_return_zero(self) -> None:
        result = jensen_shannon_divergence([0.7, 0.2, 0.1], [0.7, 0.2, 0.1])
        self.assertAlmostEqual(result, 0.0)

    def test_disjoint_distributions_approach_one(self) -> None:
        result = jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(result, 1.0)

    def test_partially_overlapping_distributions_return_intermediate_value(self) -> None:
        result = jensen_shannon_divergence([0.6, 0.4], [0.4, 0.6])
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_single_element_distributions_return_zero_for_identical_mass(self) -> None:
        result = jensen_shannon_divergence([1.0], [1.0])
        self.assertAlmostEqual(result, 0.0)

    def test_single_element_zero_mass_input_degrades_gracefully(self) -> None:
        result = jensen_shannon_divergence([1.0], [0.0])
        self.assertAlmostEqual(result, 0.0)

    def test_empty_distributions_return_zero(self) -> None:
        result = jensen_shannon_divergence([], [])
        self.assertAlmostEqual(result, 0.0)

    def test_mismatched_lengths_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            jensen_shannon_divergence([1.0, 0.0], [1.0])

    def test_non_finite_values_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            jensen_shannon_divergence([1.0, float("nan")], [1.0, 0.0])

    def test_predator_distance_for_type_falls_back_on_invalid_spider_position(self) -> None:
        meta = {
            "diagnostic": {"diagnostic_predator_dist": 9},
            "predators": [
                {
                    "x": 2,
                    "y": 3,
                    "profile": {"detection_style": "visual"},
                }
            ],
        }
        state = MagicMock()
        state.x = "bad"
        state.y = 1

        result = _diagnostic_predator_distance_for_type(
            meta,
            "visual",
            state=state,
        )

        self.assertEqual(result, 9)
