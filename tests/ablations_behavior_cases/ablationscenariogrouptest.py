from __future__ import annotations

from .shared import *


class AblationScenarioGroupTest(unittest.TestCase):
    def test_multi_predator_group_resolves_expected_scenarios(self) -> None:
        self.assertEqual(
            resolve_ablation_scenario_group("multi_predator_ecology"),
            MULTI_PREDATOR_SCENARIOS,
        )

    def test_compare_predator_type_ablation_performance_summarizes_visual_vs_olfactory_groups(self) -> None:
        """
        Verifies that compare_predator_type_ablation_performance summarizes mean success rates for visual and olfactory predator scenario groups and produces the expected sign change in visual_minus_olfactory_success_rate between visual and sensory cortex drops.
        
        Constructs a payload with per-variant scenario success rates and deltas, calls compare_predator_type_ablation_performance, and asserts that the result is available, that the grouped mean success rates for visual and olfactory predator scenarios match expected values, and that the visual-minus-olfactory difference is negative for the visual-cortex drop and positive for the sensory-cortex drop.
        """
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                },
                "drop_sensory_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.3},
                        "visual_hunter_open_field": {"success_rate": 0.9},
                        "olfactory_ambush": {"success_rate": 0.2},
                    }
                },
            },
            "deltas_vs_reference": {
                "drop_visual_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": -0.3},
                        "visual_hunter_open_field": {"success_rate_delta": -0.6},
                        "olfactory_ambush": {"success_rate_delta": -0.1},
                    }
                },
                "drop_sensory_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": -0.2},
                        "visual_hunter_open_field": {"success_rate_delta": -0.1},
                        "olfactory_ambush": {"success_rate_delta": -0.7},
                    }
                },
            },
        }

        result = compare_predator_type_ablation_performance(payload)

        self.assertTrue(result["available"])
        visual_drop = result["comparisons"]["drop_visual_cortex"]
        sensory_drop = result["comparisons"]["drop_sensory_cortex"]
        self.assertAlmostEqual(
            visual_drop["visual_predator_scenarios"]["mean_success_rate"],
            0.15,
        )
        self.assertAlmostEqual(
            visual_drop["olfactory_predator_scenarios"]["mean_success_rate"],
            0.5,
        )
        self.assertLess(
            visual_drop["visual_minus_olfactory_success_rate"],
            0.0,
        )
        self.assertGreater(
            sensory_drop["visual_minus_olfactory_success_rate"],
            0.0,
        )
