from __future__ import annotations

from .shared import *


class BSeriesCheckpointTestPart6(unittest.TestCase):
    def test_b78_transfer_reports_source_parent_and_coverage(self) -> None:
        build_b78 = getattr(
            b_series_evolution_module,
            "build_b78_vestibular_balance_config",
            None,
        )
        self.assertIsNotNone(build_b78)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b77_olivary_error_source(tmpdir)
            variants = (
                ("b78_vestibular_balance_h48_bridge_policy", 1.0),
                ("b78_head_stabilization_h48_bridge_policy", 1.0),
                ("b78_locomotor_confidence_h48_bridge_policy", 1.0),
                ("b78_vestibular_balance_h56_bridge_policy", 0.85),
                ("b78_genetic_vestibular_balance_h48_bridge_policy", 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b78(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=436 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 78)
                self.assertEqual(report["parent_level"], 77)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])
