from __future__ import annotations

import unittest

from spider_cortex_sim.offline_analysis.extraction.ladder import (
    CAPACITY_MATCH_RATIO_THRESHOLD,
    COMPARISON_CONFOUNDED,
    COMPARISON_SUPPORTED,
    COMPARISON_UNDERPOWERED,
    evaluate_variant_comparison_support,
)


def _payload_with_seed_items(
    *,
    total_trainable: int,
    architecture_signature: dict[str, object] | None = None,
    distillation_enabled: bool = False,
    loss_override_penalty_weight: float = 0.0,
    seed_count: int = 3,
    seed_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if seed_items is None:
        seed_items = [
            {"seed": str(index), "value": 0.4 + (index * 0.1)}
            for index in range(seed_count)
        ]
    seed_level = [
        {
            "metric_name": "scenario_success_rate",
            "seed": item["seed"],
            "value": item["value"],
        }
        for item in seed_items
    ]
    return {
        "summary": {"scenario_success_rate": 0.5},
        "seed_level": seed_level,
        "uncertainty": {
            "scenario_success_rate": {
                "seed_values": seed_items
            }
        },
        "parameter_counts": {"total_trainable": total_trainable},
        "variant_metadata": {
            "capacity": {"parameter_count_total": total_trainable},
            "interface_access": {
                "architecture_signature": architecture_signature or {"obs": "v1"}
            },
            "credit_assignment": {
                "mechanism": {
                    "distillation": {"enabled": distillation_enabled},
                    "loss_structure": {
                        "loss_override_penalty_weight": loss_override_penalty_weight,
                        "loss_dominance_penalty_weight": 0.0,
                        "arbitration_regularization_weight": 0.0,
                        "arbitration_valence_regularization_weight": 0.0,
                    },
                }
            },
        },
    }


class OfflineAnalysisLadderConfoundControlTest(unittest.TestCase):
    def test_underpowered_when_missing_seed_items(self) -> None:
        baseline = {"summary": {"scenario_success_rate": 0.5}, "parameter_counts": {"total_trainable": 100}}
        comparison = {"summary": {"scenario_success_rate": 0.5}, "parameter_counts": {"total_trainable": 100}}

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_UNDERPOWERED)
        self.assertTrue(any("missing per-seed" in item for item in result["diagnostics"]))

    def test_underpowered_when_insufficient_seeds(self) -> None:
        baseline = _payload_with_seed_items(total_trainable=100)
        comparison = _payload_with_seed_items(total_trainable=100, seed_count=2)

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_UNDERPOWERED)
        self.assertTrue(any("insufficient seeds" in item for item in result["diagnostics"]))

    def test_confounded_when_capacity_mismatch(self) -> None:
        self.assertLess(CAPACITY_MATCH_RATIO_THRESHOLD, 10)
        baseline = _payload_with_seed_items(total_trainable=100)
        comparison = _payload_with_seed_items(total_trainable=1000)

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_CONFOUNDED)
        self.assertEqual(result["confounds"]["capacity"], "mismatch")

    def test_confounded_when_interface_signature_differs(self) -> None:
        baseline = _payload_with_seed_items(
            total_trainable=100,
            architecture_signature={"obs": "v1"},
        )
        comparison = _payload_with_seed_items(
            total_trainable=100,
            architecture_signature={"obs": "v2"},
        )

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_CONFOUNDED)
        self.assertEqual(result["confounds"]["interface_access"], "mismatch")

    def test_confounded_when_credit_assignment_differs(self) -> None:
        baseline = _payload_with_seed_items(total_trainable=100, distillation_enabled=False)
        comparison = _payload_with_seed_items(total_trainable=100, distillation_enabled=True)

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_CONFOUNDED)
        self.assertEqual(result["confounds"]["credit_assignment"], "mismatch")

    def test_underpowered_when_no_overlapping_seeds(self) -> None:
        baseline = _payload_with_seed_items(total_trainable=100)
        comparison = _payload_with_seed_items(
            total_trainable=100,
            seed_items=[
                {"seed": "3", "value": 0.4},
                {"seed": "4", "value": 0.5},
                {"seed": "5", "value": 0.6},
            ],
        )

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_UNDERPOWERED)
        self.assertTrue(any("no overlapping" in item for item in result["diagnostics"]))

    def test_unknown_confounds_when_metadata_missing(self) -> None:
        baseline = _payload_with_seed_items(total_trainable=100)
        comparison = _payload_with_seed_items(total_trainable=100)
        del baseline["variant_metadata"]
        del comparison["variant_metadata"]

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_SUPPORTED)
        self.assertEqual(result["confounds"]["interface_access"], "unknown")
        self.assertEqual(result["confounds"]["credit_assignment"], "unknown")

    def test_supported_when_matched_metadata_and_sufficient_seeds(self) -> None:
        baseline = _payload_with_seed_items(total_trainable=100)
        comparison = _payload_with_seed_items(total_trainable=110)

        result = evaluate_variant_comparison_support(
            baseline_variant="baseline",
            comparison_variant="comparison",
            baseline_payload=baseline,
            comparison_payload=comparison,
        )

        self.assertEqual(result["status"], COMPARISON_SUPPORTED)
        self.assertEqual(result["confounds"]["capacity"], "matched")
        self.assertEqual(result["confounds"]["interface_access"], "matched")
        self.assertEqual(result["confounds"]["credit_assignment"], "matched")
