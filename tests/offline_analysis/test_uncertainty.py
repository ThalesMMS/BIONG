from __future__ import annotations

import math
import unittest

from spider_cortex_sim.offline_analysis.uncertainty import (
    _delta_uncertainty_from_seed_values,
    _payload_metric_seed_values,
)

class OfflineAnalysisUncertaintyTablesTest(unittest.TestCase):
    def test_payload_metric_seed_values_accepts_mapping_seed_values(self) -> None:
        values = _payload_metric_seed_values(
            {
                "uncertainty": {
                    "scenario_success_rate": {
                        "seed_values": [
                            {"seed": 1, "value": 0.25},
                            {"seed": 2, "value": "0.75"},
                        ]
                    }
                }
            },
            "scenario_success_rate",
        )

        self.assertEqual(values, [0.25, 0.75])
    def test_delta_uncertainty_pairs_seed_values_by_seed_key(self) -> None:
        uncertainty = _delta_uncertainty_from_seed_values(
            [("seed-b", 0.2), ("seed-a", 0.1), ("seed-c", 0.4)],
            [("seed-a", 0.5), ("seed-c", 0.9), ("seed-d", 1.0)],
            raw_delta=0.45,
        )

        self.assertEqual(uncertainty["mean"], 0.45)
        self.assertEqual(uncertainty["n_seeds"], 2)
        self.assertEqual(
            uncertainty["seed_values"],
            [
                {"seed": "seed-a", "value": 0.4},
                {"seed": "seed-c", "value": 0.5},
            ],
        )
        self.assertIsInstance(uncertainty["ci_lower"], float)
        self.assertIsInstance(uncertainty["ci_upper"], float)
        self.assertTrue(math.isfinite(uncertainty["ci_lower"]))
        self.assertTrue(math.isfinite(uncertainty["ci_upper"]))
        self.assertLessEqual(uncertainty["ci_lower"], uncertainty["ci_upper"])
    def test_delta_uncertainty_unkeyed_values_use_unpaired_bootstrap(self) -> None:
        uncertainty = _delta_uncertainty_from_seed_values(
            [0.1, 0.2],
            [0.7, 0.9],
        )

        self.assertAlmostEqual(uncertainty["mean"], 0.65)
        self.assertEqual(uncertainty["n_seeds"], 2)
        self.assertEqual(uncertainty["seed_values"], [])
        self.assertIsInstance(uncertainty["ci_lower"], float)
        self.assertIsInstance(uncertainty["ci_upper"], float)
