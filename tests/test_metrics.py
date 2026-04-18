from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from spider_cortex_sim.metrics import EpisodeMetricAccumulator, jensen_shannon_divergence
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exp_values = [math.exp(value - peak) for value in values]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def _make_accumulator() -> EpisodeMetricAccumulator:
    return EpisodeMetricAccumulator(
        reward_component_names=REWARD_COMPONENT_NAMES,
        predator_states=PREDATOR_STATES,
    )


def _make_meta(
    *,
    dominant_predator_type_label: str = "",
    visual_predator_threat: float = 0.0,
    olfactory_predator_threat: float = 0.0,
    predator_visible: bool = False,
) -> dict[str, object]:
    return {
        "food_dist": 5,
        "shelter_dist": 3,
        "night": False,
        "predator_visible": predator_visible,
        "shelter_role": "outside",
        "on_shelter": False,
        "dominant_predator_type_label": dominant_predator_type_label,
        "visual_predator_threat": visual_predator_threat,
        "olfactory_predator_threat": olfactory_predator_threat,
        "diagnostic": {
            "diagnostic_predator_dist": 4,
            "diagnostic_home_dx": 0.0,
            "diagnostic_home_dy": 0.0,
            "diagnostic_home_dist": 3.0,
        },
    }


def _make_info(*, predator_contact: bool = False) -> dict[str, object]:
    return {
        "reward_components": {name: 0.0 for name in REWARD_COMPONENT_NAMES},
        "predator_contact": predator_contact,
        "predator_escape": False,
    }


def _make_state() -> object:
    return SimpleNamespace(
        sleep_debt=0.0,
        last_move_dx=0,
        last_move_dy=0,
        x=0,
        y=0,
    )


def _make_decision(
    *,
    visual_logits: list[float],
    sensory_logits: list[float],
    visual_gate: float = 0.5,
    sensory_gate: float = 0.5,
    visual_contribution: float = 0.5,
    sensory_contribution: float = 0.5,
) -> object:
    return SimpleNamespace(
        module_results=[
            SimpleNamespace(
                name="visual_cortex",
                post_reflex_logits=list(visual_logits),
                reflex_applied=False,
                module_reflex_override=False,
                module_reflex_dominance=0.0,
            ),
            SimpleNamespace(
                name="sensory_cortex",
                post_reflex_logits=list(sensory_logits),
                reflex_applied=False,
                module_reflex_override=False,
                module_reflex_dominance=0.0,
            ),
        ],
        final_reflex_override=False,
        arbitration_decision=SimpleNamespace(
            module_gates={
                "visual_cortex": visual_gate,
                "sensory_cortex": sensory_gate,
            },
            module_contribution_share={
                "visual_cortex": visual_contribution,
                "sensory_cortex": sensory_contribution,
            },
            dominant_module="visual_cortex",
            dominant_module_share=visual_contribution,
            effective_module_count=2.0,
            module_agreement_rate=0.5,
            module_disagreement_rate=0.5,
        ),
    )


class JensenShannonDivergenceUtilityTest(unittest.TestCase):
    def test_identical_distributions_return_zero(self) -> None:
        self.assertAlmostEqual(
            jensen_shannon_divergence([0.7, 0.2, 0.1], [0.7, 0.2, 0.1]),
            0.0,
        )

    def test_disjoint_one_hot_distributions_return_one(self) -> None:
        self.assertAlmostEqual(
            jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]),
            1.0,
        )

    def test_known_binary_case_matches_base_two_reference_value(self) -> None:
        self.assertAlmostEqual(
            jensen_shannon_divergence([1.0, 0.0], [0.5, 0.5]),
            0.311278,
            places=6,
        )


class EpisodeMetricAccumulatorRepresentationTest(unittest.TestCase):
    def test_record_transition_accumulates_proposer_logits_by_predator_type(self) -> None:
        acc = _make_accumulator()
        acc.record_decision(
            _make_decision(
                visual_logits=[3.0, 1.0, -2.0],
                sensory_logits=[-1.0, 2.0, 0.5],
            )
        )

        visual_meta = _make_meta(
            dominant_predator_type_label="visual",
            visual_predator_threat=0.9,
            predator_visible=True,
        )
        acc.record_transition(
            step=0,
            observation_meta=visual_meta,
            next_meta=visual_meta,
            info=_make_info(),
            state=_make_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        self.assertEqual(
            acc.proposer_logits_by_predator_type["visual"]["visual_cortex"],
            [3.0, 1.0, -2.0],
        )
        self.assertEqual(
            acc.proposer_logits_by_predator_type["visual"]["sensory_cortex"],
            [-1.0, 2.0, 0.5],
        )
        self.assertEqual(
            acc.proposer_steps_by_predator_type["visual"]["visual_cortex"],
            1,
        )

    def test_snapshot_computes_divergence_from_mean_probability_vectors(self) -> None:
        acc = _make_accumulator()

        visual_meta = _make_meta(
            dominant_predator_type_label="visual",
            visual_predator_threat=1.0,
            predator_visible=True,
        )
        olfactory_meta = _make_meta(
            dominant_predator_type_label="olfactory",
            olfactory_predator_threat=1.0,
            predator_visible=True,
        )

        acc.record_decision(
            _make_decision(
                visual_logits=[2.0, 0.0],
                sensory_logits=[0.0, 2.0],
                visual_gate=0.8,
                sensory_gate=0.2,
                visual_contribution=0.7,
                sensory_contribution=0.3,
            )
        )
        acc.record_transition(
            step=0,
            observation_meta=visual_meta,
            next_meta=visual_meta,
            info=_make_info(),
            state=_make_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        acc.record_decision(
            _make_decision(
                visual_logits=[0.0, 2.0],
                sensory_logits=[2.0, 0.0],
                visual_gate=0.2,
                sensory_gate=0.8,
                visual_contribution=0.3,
                sensory_contribution=0.7,
            )
        )
        acc.record_transition(
            step=1,
            observation_meta=olfactory_meta,
            next_meta=olfactory_meta,
            info=_make_info(),
            state=_make_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        snapshot = acc.snapshot()
        expected_visual_divergence = jensen_shannon_divergence(
            _softmax([2.0, 0.0]),
            _softmax([0.0, 2.0]),
        )
        self.assertAlmostEqual(
            snapshot["proposer_divergence_by_module"]["visual_cortex"],
            expected_visual_divergence,
            places=6,
        )
        self.assertAlmostEqual(
            snapshot["action_center_gate_differential"]["visual_cortex"],
            0.6,
        )
        self.assertAlmostEqual(
            snapshot["action_center_contribution_differential"]["sensory_cortex"],
            -0.4,
        )
        self.assertGreater(
            snapshot["representation_specialization_score"],
            0.0,
        )
        self.assertAlmostEqual(
            snapshot["representation_specialization_score"],
            (
                snapshot["proposer_divergence_by_module"]["visual_cortex"]
                + snapshot["proposer_divergence_by_module"]["sensory_cortex"]
            )
            / 2.0,
            places=6,
        )

    def test_no_predator_encounters_leave_representation_metrics_empty(self) -> None:
        acc = _make_accumulator()
        acc.record_decision(
            _make_decision(
                visual_logits=[1.0, 0.0],
                sensory_logits=[0.0, 1.0],
            )
        )
        no_threat_meta = _make_meta(dominant_predator_type_label="visual")
        acc.record_transition(
            step=0,
            observation_meta=no_threat_meta,
            next_meta=no_threat_meta,
            info=_make_info(),
            state=_make_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        snapshot = acc.snapshot()
        self.assertEqual(snapshot["mean_proposer_probs_by_predator_type"], {})
        self.assertEqual(snapshot["proposer_divergence_by_module"], {})
        self.assertEqual(snapshot["representation_specialization_score"], 0.0)

    def test_single_predator_type_only_does_not_emit_divergence(self) -> None:
        acc = _make_accumulator()
        visual_meta = _make_meta(
            dominant_predator_type_label="visual",
            visual_predator_threat=0.9,
            predator_visible=True,
        )
        acc.record_decision(
            _make_decision(
                visual_logits=[3.0, 0.0],
                sensory_logits=[0.0, 3.0],
            )
        )
        acc.record_transition(
            step=0,
            observation_meta=visual_meta,
            next_meta=visual_meta,
            info=_make_info(),
            state=_make_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

        snapshot = acc.snapshot()
        self.assertIn("visual", snapshot["mean_proposer_probs_by_predator_type"])
        self.assertEqual(snapshot["proposer_divergence_by_module"], {})
        self.assertEqual(snapshot["representation_specialization_score"], 0.0)

    def test_zero_logits_produce_zero_divergence(self) -> None:
        acc = _make_accumulator()
        visual_meta = _make_meta(
            dominant_predator_type_label="visual",
            visual_predator_threat=1.0,
            predator_visible=True,
        )
        olfactory_meta = _make_meta(
            dominant_predator_type_label="olfactory",
            olfactory_predator_threat=1.0,
            predator_visible=True,
        )

        for step, meta in enumerate((visual_meta, olfactory_meta)):
            acc.record_decision(
                _make_decision(
                    visual_logits=[0.0, 0.0, 0.0],
                    sensory_logits=[0.0, 0.0, 0.0],
                )
            )
            acc.record_transition(
                step=step,
                observation_meta=meta,
                next_meta=meta,
                info=_make_info(),
                state=_make_state(),
                predator_state_before="PATROL",
                predator_state="PATROL",
            )

        snapshot = acc.snapshot()
        self.assertAlmostEqual(
            snapshot["proposer_divergence_by_module"]["visual_cortex"],
            0.0,
        )
        self.assertAlmostEqual(
            snapshot["proposer_divergence_by_module"]["sensory_cortex"],
            0.0,
        )
        self.assertAlmostEqual(snapshot["representation_specialization_score"], 0.0)
