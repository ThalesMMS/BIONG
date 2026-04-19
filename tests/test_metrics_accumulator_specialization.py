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
    _dominant_predator_type,
    _first_active_predator_type,
    _normalize_distribution,
    _predator_type_threat,
    jensen_shannon_divergence,
)
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES

from tests.fixtures.metrics_accumulator import _make_accumulator, _make_fake_meta, _make_fake_info, _make_fake_state

class RepresentationSpecializationAccumulatorTest(unittest.TestCase):
    """Tests for predator-type-conditioned representation accumulation."""

    @staticmethod
    def _make_module_result(name: str, logits: list[float]) -> object:
        """
        Create a fake module result object for tests.
        
        Parameters:
            name (str): Module name to set on the fake result.
            logits (list[float]): Values to copy into the `post_reflex_logits` field.
        
        Returns:
            object: A MagicMock with attributes `name`, `post_reflex_logits`, `reflex_applied`, `module_reflex_override`, and `module_reflex_dominance` initialized for use in accumulator tests.
        """
        result = MagicMock()
        result.name = name
        result.post_reflex_logits = list(logits)
        result.reflex_applied = False
        result.module_reflex_override = False
        result.module_reflex_dominance = 0.0
        return result

    @staticmethod
    def _make_decision(
        *,
        visual_logits: list[float],
        sensory_logits: list[float],
        module_gates: Dict[str, float],
        module_contribution_share: Dict[str, float],
    ) -> object:
        """
        Create a test decision-like object containing two module results and an arbitration summary.
        
        Parameters:
            visual_logits (list[float]): Logit vector for the visual_cortex module's proposer.
            sensory_logits (list[float]): Logit vector for the sensory_cortex module's proposer.
            module_gates (Dict[str, float]): Mapping of module gate values keyed by module name.
            module_contribution_share (Dict[str, float]): Mapping of module contribution shares keyed by module name.
        
        Returns:
            object: A MagicMock acting as a decision with attributes:
                - module_results: list of two module result objects for "visual_cortex" and "sensory_cortex" (each containing the provided logits).
                - final_reflex_override: False
                - arbitration_decision: an object with `module_gates`, `module_contribution_share`, `dominant_module`, `dominant_module_share`, `effective_module_count`, `module_agreement_rate`, and `module_disagreement_rate`.
        """
        arbitration = MagicMock()
        arbitration.module_gates = dict(module_gates)
        arbitration.module_contribution_share = dict(module_contribution_share)
        arbitration.dominant_module = "visual_cortex"
        arbitration.dominant_module_share = float(
            module_contribution_share.get("visual_cortex", 0.0)
        )
        arbitration.effective_module_count = 2.0
        arbitration.module_agreement_rate = 0.5
        arbitration.module_disagreement_rate = 0.5
        decision = MagicMock()
        decision.module_results = [
            RepresentationSpecializationAccumulatorTest._make_module_result(
                "visual_cortex",
                visual_logits,
            ),
            RepresentationSpecializationAccumulatorTest._make_module_result(
                "sensory_cortex",
                sensory_logits,
            ),
        ]
        decision.final_reflex_override = False
        decision.arbitration_decision = arbitration
        return decision

    @staticmethod
    def _make_final_state() -> object:
        """
        Create a mock "final" agent state for tests with common end-of-episode attributes preset.
        
        The returned MagicMock has attributes typically inspected by tests set to representative final-state values:
        - steps_alive: 2
        - food_eaten, sleep_events, shelter_entries, alert_events, predator_contacts, predator_sightings, predator_escapes: 0
        - hunger, fatigue, sleep_debt: 0.0
        - health: 1.0
        
        Returns:
            mock_state (MagicMock): A MagicMock configured with the above attributes.
        """
        state = MagicMock()
        state.steps_alive = 2
        state.food_eaten = 0
        state.sleep_events = 0
        state.shelter_entries = 0
        state.alert_events = 0
        state.predator_contacts = 0
        state.predator_sightings = 0
        state.predator_escapes = 0
        state.hunger = 0.0
        state.fatigue = 0.0
        state.sleep_debt = 0.0
        state.health = 1.0
        return state

    def test_snapshot_defaults_representation_metrics_when_no_predator_readout_recorded(self) -> None:
        acc = _make_accumulator()
        snap = acc.snapshot()
        self.assertEqual(snap["proposer_divergence_by_module"], {})
        self.assertEqual(snap["action_center_gate_differential"], {})
        self.assertEqual(snap["action_center_contribution_differential"], {})
        self.assertEqual(snap["mean_proposer_probs_by_predator_type"], {})
        self.assertAlmostEqual(snap["representation_specialization_score"], 0.0)

    def test_representation_readout_uses_observation_meta_predator_label(self) -> None:
        acc = _make_accumulator()
        acc.record_decision(
            self._make_decision(
                visual_logits=[4.0, 0.0, 0.0],
                sensory_logits=[0.0, 4.0, 0.0],
                module_gates={"visual_cortex": 0.8, "sensory_cortex": 0.2},
                module_contribution_share={"visual_cortex": 0.7, "sensory_cortex": 0.3},
            )
        )
        obs_meta = _make_fake_meta(predator_visible=True)
        obs_meta["dominant_predator_type_label"] = "visual"
        obs_meta["visual_predator_threat"] = 0.9
        obs_meta["olfactory_predator_threat"] = 0.1
        next_meta = _make_fake_meta(predator_visible=True)
        next_meta["dominant_predator_type_label"] = "olfactory"
        next_meta["visual_predator_threat"] = 0.1
        next_meta["olfactory_predator_threat"] = 0.9

        acc.record_transition(
            step=0,
            observation_meta=obs_meta,
            next_meta=next_meta,
            info=_make_fake_info(),
            state=_make_fake_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        snap = acc.snapshot()
        self.assertIn("visual", snap["mean_proposer_probs_by_predator_type"])
        self.assertNotIn("olfactory", snap["mean_proposer_probs_by_predator_type"])

    def test_snapshot_and_finalize_include_representation_specialization_metrics(self) -> None:
        acc = _make_accumulator()
        self.assertEqual(
            ACTION_CENTER_REPRESENTATION_FIELDS,
            ("module_gates", "module_contribution_share"),
        )
        self.assertEqual(
            PRIMARY_REPRESENTATION_READOUT_MODULES,
            ("visual_cortex", "sensory_cortex"),
        )

        visual_obs = _make_fake_meta(predator_visible=True)
        visual_obs["dominant_predator_type_label"] = "visual"
        visual_obs["visual_predator_threat"] = 0.9
        visual_obs["olfactory_predator_threat"] = 0.0
        olfactory_obs = _make_fake_meta(predator_visible=True)
        olfactory_obs["dominant_predator_type_label"] = "olfactory"
        olfactory_obs["visual_predator_threat"] = 0.0
        olfactory_obs["olfactory_predator_threat"] = 0.9

        acc.record_decision(
            self._make_decision(
                visual_logits=[5.0, 0.0, 0.0],
                sensory_logits=[0.0, 5.0, 0.0],
                module_gates={"visual_cortex": 0.8, "sensory_cortex": 0.2},
                module_contribution_share={"visual_cortex": 0.75, "sensory_cortex": 0.25},
            )
        )
        acc.record_transition(
            step=0,
            observation_meta=visual_obs,
            next_meta=visual_obs,
            info=_make_fake_info(),
            state=_make_fake_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        acc.record_decision(
            self._make_decision(
                visual_logits=[0.0, 5.0, 0.0],
                sensory_logits=[5.0, 0.0, 0.0],
                module_gates={"visual_cortex": 0.2, "sensory_cortex": 0.8},
                module_contribution_share={"visual_cortex": 0.25, "sensory_cortex": 0.75},
            )
        )
        acc.record_transition(
            step=1,
            observation_meta=olfactory_obs,
            next_meta=olfactory_obs,
            info=_make_fake_info(),
            state=_make_fake_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        snap = acc.snapshot()
        self.assertIn("visual", snap["mean_proposer_probs_by_predator_type"])
        self.assertIn("olfactory", snap["mean_proposer_probs_by_predator_type"])
        self.assertGreater(
            snap["mean_proposer_probs_by_predator_type"]["visual"]["visual_cortex"][0],
            0.9,
        )
        self.assertGreater(
            snap["mean_proposer_probs_by_predator_type"]["olfactory"]["sensory_cortex"][0],
            0.9,
        )
        self.assertGreater(
            snap["proposer_divergence_by_module"]["visual_cortex"],
            0.8,
        )
        self.assertGreater(
            snap["proposer_divergence_by_module"]["sensory_cortex"],
            0.8,
        )
        self.assertAlmostEqual(
            snap["action_center_gate_differential"]["visual_cortex"],
            0.6,
        )
        self.assertAlmostEqual(
            snap["action_center_gate_differential"]["sensory_cortex"],
            -0.6,
        )
        self.assertAlmostEqual(
            snap["action_center_contribution_differential"]["visual_cortex"],
            0.5,
        )
        self.assertAlmostEqual(
            snap["action_center_contribution_differential"]["sensory_cortex"],
            -0.5,
        )
        self.assertAlmostEqual(
            snap["representation_specialization_score"],
            sum(snap["proposer_divergence_by_module"].values()) / 2.0,
        )

        stats = acc.finalize(
            episode=0,
            seed=7,
            training=False,
            scenario="visual_olfactory_pincer",
            total_reward=0.0,
            state=self._make_final_state(),
        )
        self.assertEqual(
            stats.proposer_divergence_by_module,
            snap["proposer_divergence_by_module"],
        )
        self.assertEqual(
            stats.action_center_gate_differential,
            snap["action_center_gate_differential"],
        )
        self.assertEqual(
            stats.action_center_contribution_differential,
            snap["action_center_contribution_differential"],
        )
        self.assertEqual(
            stats.mean_proposer_probs_by_predator_type,
            snap["mean_proposer_probs_by_predator_type"],
        )
        self.assertAlmostEqual(
            stats.representation_specialization_score,
            snap["representation_specialization_score"],
        )

class NormalizeDistributionTest(unittest.TestCase):
    """Tests for metrics._normalize_distribution() - new in this PR."""

    def test_empty_mapping_returns_empty(self) -> None:
        result = _normalize_distribution({})
        self.assertEqual(result, {})

    def test_single_item_returns_one(self) -> None:
        result = _normalize_distribution({"a": 5})
        self.assertAlmostEqual(result["a"], 1.0)

    def test_two_equal_items_return_half_each(self) -> None:
        result = _normalize_distribution({"a": 3, "b": 3})
        self.assertAlmostEqual(result["a"], 0.5)
        self.assertAlmostEqual(result["b"], 0.5)

    def test_zero_total_returns_all_zeros(self) -> None:
        result = _normalize_distribution({"a": 0, "b": 0})
        self.assertAlmostEqual(result["a"], 0.0)
        self.assertAlmostEqual(result["b"], 0.0)

    def test_proportions_sum_to_one(self) -> None:
        result = _normalize_distribution({"a": 1, "b": 2, "c": 7})
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0)

    def test_keys_are_strings(self) -> None:
        result = _normalize_distribution({"module_a": 4, "module_b": 6})
        for key in result:
            self.assertIsInstance(key, str)

    def test_values_are_floats(self) -> None:
        result = _normalize_distribution({"a": 10})
        for value in result.values():
            self.assertIsInstance(value, float)

    def test_three_unequal_counts(self) -> None:
        result = _normalize_distribution({"a": 1, "b": 2, "c": 7})
        self.assertAlmostEqual(result["a"], 0.1)
        self.assertAlmostEqual(result["b"], 0.2)
        self.assertAlmostEqual(result["c"], 0.7)

class PredatorTypeThreatTest(unittest.TestCase):
    """Tests for metrics._predator_type_threat() - new in this PR."""

    def test_returns_visual_threat_when_present(self) -> None:
        meta = {"visual_predator_threat": 0.6}
        self.assertAlmostEqual(_predator_type_threat(meta, "visual"), 0.6)

    def test_returns_olfactory_threat_when_present(self) -> None:
        meta = {"olfactory_predator_threat": 0.3}
        self.assertAlmostEqual(_predator_type_threat(meta, "olfactory"), 0.3)

    def test_missing_key_returns_zero(self) -> None:
        self.assertAlmostEqual(_predator_type_threat({}, "visual"), 0.0)

    def test_none_value_returns_zero(self) -> None:
        meta = {"visual_predator_threat": None}
        self.assertAlmostEqual(_predator_type_threat(meta, "visual"), 0.0)

    def test_zero_value_returns_zero(self) -> None:
        meta = {"visual_predator_threat": 0.0}
        self.assertAlmostEqual(_predator_type_threat(meta, "visual"), 0.0)

    def test_constructs_key_correctly(self) -> None:
        meta = {"custom_predator_threat": 0.9}
        self.assertAlmostEqual(_predator_type_threat(meta, "custom"), 0.9)

class DominantPredatorTypeTest(unittest.TestCase):
    """Tests for metrics._dominant_predator_type() - new in this PR."""

    def test_label_takes_precedence(self) -> None:
        meta = {
            "dominant_predator_type_label": "visual",
            "visual_predator_threat": 0.1,
            "olfactory_predator_threat": 0.9,
        }
        result = _dominant_predator_type(meta)
        self.assertEqual(result, "visual")

    def test_label_case_insensitive(self) -> None:
        meta = {"dominant_predator_type_label": "OLFACTORY"}
        result = _dominant_predator_type(meta)
        self.assertEqual(result, "olfactory")

    def test_invalid_label_falls_back_to_threat_comparison(self) -> None:
        meta = {
            "dominant_predator_type_label": "unknown",
            "visual_predator_threat": 0.3,
            "olfactory_predator_threat": 0.7,
        }
        result = _dominant_predator_type(meta)
        self.assertEqual(result, "olfactory")

    def test_visual_wins_when_higher(self) -> None:
        meta = {"visual_predator_threat": 0.8, "olfactory_predator_threat": 0.2}
        self.assertEqual(_dominant_predator_type(meta), "visual")

    def test_olfactory_wins_when_higher(self) -> None:
        meta = {"visual_predator_threat": 0.2, "olfactory_predator_threat": 0.8}
        self.assertEqual(_dominant_predator_type(meta), "olfactory")

    def test_both_zero_returns_empty_string(self) -> None:
        meta = {"visual_predator_threat": 0.0, "olfactory_predator_threat": 0.0}
        self.assertEqual(_dominant_predator_type(meta), "")

    def test_empty_meta_returns_empty_string(self) -> None:
        self.assertEqual(_dominant_predator_type({}), "")

    def test_equal_threats_returns_visual(self) -> None:
        # When equal, visual is returned (olfactory > visual is False)
        meta = {"visual_predator_threat": 0.5, "olfactory_predator_threat": 0.5}
        result = _dominant_predator_type(meta)
        self.assertEqual(result, "visual")

class DiagnosticPredatorDistanceTest(unittest.TestCase):
    """Tests for metrics._diagnostic_predator_distance() - new in this PR."""

    def test_reads_diagnostic_dist(self) -> None:
        meta = {"diagnostic": {"diagnostic_predator_dist": 5}}
        self.assertEqual(_diagnostic_predator_distance(meta), 5)

    def test_missing_diagnostic_returns_zero(self) -> None:
        self.assertEqual(_diagnostic_predator_distance({}), 0)

    def test_non_mapping_diagnostic_returns_zero(self) -> None:
        meta = {"diagnostic": "not_a_mapping"}
        self.assertEqual(_diagnostic_predator_distance(meta), 0)

    def test_missing_dist_key_returns_zero(self) -> None:
        meta = {"diagnostic": {}}
        self.assertEqual(_diagnostic_predator_distance(meta), 0)

    def test_none_dist_returns_zero(self) -> None:
        meta = {"diagnostic": {"diagnostic_predator_dist": None}}
        self.assertEqual(_diagnostic_predator_distance(meta), 0)

    def test_non_numeric_dist_returns_zero(self) -> None:
        meta = {"diagnostic": {"diagnostic_predator_dist": "not-a-number"}}
        self.assertEqual(_diagnostic_predator_distance(meta), 0)

    def test_nonfinite_dist_returns_zero(self) -> None:
        meta = {"diagnostic": {"diagnostic_predator_dist": float("inf")}}
        self.assertEqual(_diagnostic_predator_distance(meta), 0)

    def test_float_dist_is_truncated_to_int(self) -> None:
        meta = {"diagnostic": {"diagnostic_predator_dist": 3.9}}
        self.assertEqual(_diagnostic_predator_distance(meta), 3)

class FirstActivePredatorTypeTest(unittest.TestCase):
    """Tests for metrics._first_active_predator_type() - new in this PR."""

    def test_returns_visual_when_present(self) -> None:
        active = {"visual": {"start_step": 1}, "olfactory": {"start_step": 2}}
        self.assertEqual(_first_active_predator_type(active), "visual")

    def test_returns_olfactory_when_only_olfactory(self) -> None:
        active = {"olfactory": {"start_step": 1}}
        self.assertEqual(_first_active_predator_type(active), "olfactory")

    def test_returns_empty_when_no_predator_types(self) -> None:
        active = {"unknown_type": {}}
        self.assertEqual(_first_active_predator_type(active), "")

    def test_returns_empty_when_mapping_empty(self) -> None:
        self.assertEqual(_first_active_predator_type({}), "")

    def test_visual_preferred_over_olfactory_order(self) -> None:
        # PREDATOR_TYPE_NAMES order: visual comes first
        active = {"olfactory": {}, "visual": {}}
        self.assertEqual(_first_active_predator_type(active), "visual")

class ContactPredatorTypesTest(unittest.TestCase):
    """Tests for metrics._contact_predator_types() - new in this PR."""

    def _make_state(self, x: int = 5, y: int = 5) -> object:
        """
        Create a fake state object with x and y coordinate attributes for tests.
        
        Parameters:
            x (int): X coordinate value for the fake state (default 5).
            y (int): Y coordinate value for the fake state (default 5).
        
        Returns:
            state (unittest.mock.MagicMock): A mock object with integer attributes `x` and `y`.
        """
        from unittest.mock import MagicMock
        state = MagicMock()
        state.x = x
        state.y = y
        return state

    def test_returns_visual_when_visual_predator_at_same_cell(self) -> None:
        meta = {
            "predators": [
                {"x": 5, "y": 5, "profile": {"detection_style": "visual"}}
            ]
        }
        state = self._make_state(5, 5)
        result = _contact_predator_types(meta, state=state)
        self.assertIn("visual", result)

    def test_returns_olfactory_when_olfactory_predator_at_same_cell(self) -> None:
        meta = {
            "predators": [
                {"x": 3, "y": 4, "profile": {"detection_style": "olfactory"}}
            ]
        }
        state = self._make_state(3, 4)
        result = _contact_predator_types(meta, state=state)
        self.assertIn("olfactory", result)

    def test_returns_empty_when_predator_at_different_cell(self) -> None:
        meta = {
            "predators": [
                {"x": 1, "y": 1, "profile": {"detection_style": "visual"}}
            ],
            "visual_predator_threat": 0.0,
            "olfactory_predator_threat": 0.0,
        }
        state = self._make_state(5, 5)
        result = _contact_predator_types(meta, state=state)
        self.assertEqual(result, [])

    def test_no_state_coords_falls_back_to_dominant_type(self) -> None:
        from unittest.mock import MagicMock
        state = MagicMock()
        state.x = None
        state.y = None
        meta = {"visual_predator_threat": 0.8, "olfactory_predator_threat": 0.0}
        result = _contact_predator_types(meta, state=state)
        self.assertEqual(result, ["visual"])

    def test_multiple_predators_at_same_cell_returns_all_types(self) -> None:
        meta = {
            "predators": [
                {"x": 2, "y": 2, "profile": {"detection_style": "visual"}},
                {"x": 2, "y": 2, "profile": {"detection_style": "olfactory"}},
            ]
        }
        state = self._make_state(2, 2)
        result = _contact_predator_types(meta, state=state)
        self.assertIn("visual", result)
        self.assertIn("olfactory", result)

    def test_no_predators_list_falls_back_to_dominant_type(self) -> None:
        meta = {
            "visual_predator_threat": 0.0,
            "olfactory_predator_threat": 0.6,
        }
        state = self._make_state(5, 5)
        result = _contact_predator_types(meta, state=state)
        self.assertEqual(result, ["olfactory"])

    def test_predator_without_profile_mapping_is_skipped(self) -> None:
        meta = {
            "predators": [
                {"x": 5, "y": 5, "profile": "not_a_mapping"},
            ],
            "visual_predator_threat": 0.0,
            "olfactory_predator_threat": 0.0,
        }
        state = self._make_state(5, 5)
        result = _contact_predator_types(meta, state=state)
        self.assertEqual(result, [])

    def test_predator_with_malformed_coordinates_is_skipped(self) -> None:
        meta = {
            "predators": [
                {"x": None, "y": 5, "profile": {"detection_style": "visual"}},
                {"x": "", "y": 5, "profile": {"detection_style": "olfactory"}},
                {"x": "not-a-number", "y": 5, "profile": {"detection_style": "visual"}},
            ],
            "visual_predator_threat": 0.0,
            "olfactory_predator_threat": 0.0,
        }
        state = self._make_state(5, 5)
        result = _contact_predator_types(meta, state=state)
        self.assertEqual(result, [])

    def test_unknown_detection_style_not_included(self) -> None:
        meta = {
            "predators": [
                {"x": 5, "y": 5, "profile": {"detection_style": "unknown_style"}},
            ],
            "visual_predator_threat": 0.0,
            "olfactory_predator_threat": 0.0,
        }
        state = self._make_state(5, 5)
        result = _contact_predator_types(meta, state=state)
        self.assertNotIn("unknown_style", result)

class EpisodeAccumulatorPredatorTypeFieldsTest(unittest.TestCase):
    """Tests for EpisodeMetricAccumulator new predator-type fields and initialization."""

    def test_accumulator_initializes_contacts_by_type(self) -> None:
        acc = _make_accumulator()
        self.assertIn("visual", acc.predator_contacts_by_type)
        self.assertIn("olfactory", acc.predator_contacts_by_type)
        self.assertEqual(acc.predator_contacts_by_type["visual"], 0)
        self.assertEqual(acc.predator_contacts_by_type["olfactory"], 0)

    def test_accumulator_initializes_escapes_by_type(self) -> None:
        acc = _make_accumulator()
        self.assertIn("visual", acc.predator_escapes_by_type)
        self.assertIn("olfactory", acc.predator_escapes_by_type)

    def test_accumulator_initializes_response_latencies_by_type(self) -> None:
        acc = _make_accumulator()
        self.assertIn("visual", acc.predator_response_latencies_by_type)
        self.assertIsInstance(acc.predator_response_latencies_by_type["visual"], list)

    def test_accumulator_initializes_module_response_by_predator_type_counts(self) -> None:
        acc = _make_accumulator()
        self.assertIn("visual", acc.module_response_by_predator_type_counts)
        self.assertIn("olfactory", acc.module_response_by_predator_type_counts)

    def test_snapshot_includes_predator_contacts_by_type(self) -> None:
        acc = _make_accumulator()
        snap = acc.snapshot()
        self.assertIn("predator_contacts_by_type", snap)
        self.assertIn("visual", snap["predator_contacts_by_type"])
        self.assertIn("olfactory", snap["predator_contacts_by_type"])

    def test_snapshot_includes_predator_escapes_by_type(self) -> None:
        acc = _make_accumulator()
        snap = acc.snapshot()
        self.assertIn("predator_escapes_by_type", snap)

    def test_record_transition_preserves_escape_type_when_response_window_closes(self) -> None:
        acc = _make_accumulator()
        state = _make_fake_state()

        obs_meta = _make_fake_meta(predator_visible=False, predator_dist=3)
        obs_meta["visual_predator_threat"] = 0.0
        obs_meta["olfactory_predator_threat"] = 0.0

        threat_meta = _make_fake_meta(predator_visible=True, predator_dist=2)
        threat_meta["visual_predator_threat"] = 0.8
        threat_meta["olfactory_predator_threat"] = 0.0
        threat_meta["dominant_predator_type_label"] = "visual"

        acc.record_transition(
            step=0,
            observation_meta=obs_meta,
            next_meta=threat_meta,
            info=_make_fake_info(),
            state=state,
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        escape_info = _make_fake_info()
        escape_info["predator_escape"] = True
        escape_meta = _make_fake_meta(predator_visible=False, predator_dist=5)
        escape_meta["visual_predator_threat"] = 0.0
        escape_meta["olfactory_predator_threat"] = 0.0

        acc.record_transition(
            step=1,
            observation_meta=threat_meta,
            next_meta=escape_meta,
            info=escape_info,
            state=state,
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        self.assertEqual(acc.predator_escapes_by_type["visual"], 1)
        self.assertEqual(acc.predator_escapes_by_type["olfactory"], 0)

    def test_record_transition_tolerates_missing_diagnostic_predator_distance(self) -> None:
        acc = _make_accumulator()
        state = _make_fake_state()

        obs_meta = _make_fake_meta(predator_visible=False)
        obs_meta.pop("diagnostic")
        threat_meta = _make_fake_meta(predator_visible=True)
        threat_meta.pop("diagnostic")

        acc.record_transition(
            step=0,
            observation_meta=obs_meta,
            next_meta=threat_meta,
            info=_make_fake_info(),
            state=state,
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        self.assertEqual(acc.active_predator_response["start_distance"], 0)

    def test_snapshot_includes_predator_response_latency_by_type(self) -> None:
        acc = _make_accumulator()
        snap = acc.snapshot()
        self.assertIn("predator_response_latency_by_type", snap)

    def test_snapshot_includes_module_response_by_predator_type(self) -> None:
        acc = _make_accumulator()
        snap = acc.snapshot()
        self.assertIn("module_response_by_predator_type", snap)
