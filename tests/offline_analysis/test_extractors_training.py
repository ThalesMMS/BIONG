import unittest

from spider_cortex_sim.offline_analysis.extractors import (
    build_primary_benchmark,
    extract_comparisons,
    extract_scenario_success,
)


class OfflineAnalysisTrainingExtractorTest(unittest.TestCase):
    def test_behavior_csv_failures_support_current_and_legacy_delimiters(self) -> None:
        report = extract_scenario_success(
            {},
            [
                {
                    "scenario": "hard_case",
                    "success": False,
                    "failures": "starved,eaten;timeout",
                }
            ],
        )

        self.assertEqual(report["source"], "behavior_csv")
        self.assertEqual(
            report["scenarios"][0]["failures"],
            ["eaten", "starved", "timeout"],
        )

    def test_behavior_csv_checks_ignore_inapplicable_empty_columns(self) -> None:
        report = extract_scenario_success(
            {},
            [
                {
                    "scenario": "escape",
                    "success": True,
                    "check_escape_reached_passed": "True",
                    "check_food_found_passed": "",
                },
                {
                    "scenario": "escape",
                    "success": True,
                    "check_escape_reached_passed": "",
                    "check_food_found_passed": "",
                },
                {
                    "scenario": "forage",
                    "success": True,
                    "check_escape_reached_passed": "",
                    "check_food_found_passed": "True",
                },
            ],
        )
        scenarios = {
            item["scenario"]: item
            for item in report["scenarios"]
        }

        self.assertEqual(set(scenarios["escape"]["checks"]), {"escape_reached"})
        self.assertEqual(
            scenarios["escape"]["checks"]["escape_reached"]["pass_rate"],
            1.0,
        )
        self.assertEqual(set(scenarios["forage"]["checks"]), {"food_found"})

    def test_comparison_fallback_requires_full_scenario_success(self) -> None:
        rows = [
            {
                "reward_profile": "classic",
                "evaluation_map": "central_burrow",
                "scenario": scenario,
                "success": success,
            }
            for scenario in ("a", "b")
            for success in (True, False)
        ]

        report = extract_comparisons({}, rows)
        summary = report["reward_profiles"]["classic"]["summary"]

        self.assertEqual(summary["scenario_success_rate"], 0.0)
        self.assertEqual(summary["episode_success_rate"], 0.5)

    def test_primary_benchmark_fallback_requires_full_scenario_success(self) -> None:
        benchmark = build_primary_benchmark(
            {},
            {
                "source": "behavior_csv",
                "scenarios": [
                    {"scenario": "a", "success_rate": 0.5},
                    {"scenario": "b", "success_rate": 0.5},
                ],
            },
            {},
        )

        self.assertEqual(benchmark["scenario_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
