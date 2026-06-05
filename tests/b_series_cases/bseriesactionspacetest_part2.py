from __future__ import annotations

from .shared import *


class BSeriesActionSpaceTestPart2(unittest.TestCase):
    def test_diagnostic_catalog_registers_b78_vestibular_balance_variants(self) -> None:
        variants = [
            "b78_vestibular_balance_h48_bridge_policy",
            "b78_head_stabilization_h48_bridge_policy",
            "b78_locomotor_confidence_h48_bridge_policy",
            "b78_vestibular_balance_h56_bridge_policy",
            "b78_genetic_vestibular_balance_h48_bridge_policy",
        ]
        try:
            h48, stabilizing, confidence, h56, genetic = resolve_ablation_configs(
                variants
            )
        except KeyError as exc:
            self.fail(str(exc))

        for config in (h48, stabilizing, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 78)
            self.assertEqual(config.b_parent_level, 77)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b78_balance_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "vestibular_balance")
        self.assertEqual(stabilizing.b_controller_profile, "head_stabilization")
        self.assertEqual(confidence.b_controller_profile, "locomotor_confidence")
        self.assertEqual(genetic.b_controller_profile, "genetic_vestibular_balance")
