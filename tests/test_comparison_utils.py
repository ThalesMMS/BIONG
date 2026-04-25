from copy import deepcopy
from pathlib import Path
import unittest
from unittest import mock

from spider_cortex_sim.budget_profiles import resolve_budget
from spider_cortex_sim.comparison_utils import (
    build_checkpoint_path,
    build_run_budget_summary,
    create_comparison_simulation,
)


class ComparisonBootstrapHelperTest(unittest.TestCase):
    def test_build_run_budget_summary_patches_resolved_fields(self) -> None:
        budget = resolve_budget(
            profile="smoke",
            episodes=3,
            eval_episodes=2,
            max_steps=42,
            scenario_episodes=4,
            checkpoint_interval=5,
            behavior_seeds=(1, 2),
            ablation_seeds=(3,),
        )

        for behavior_seeds, ablation_seeds in (
            ((7, 17), (29,)),
            ([41], [53, 59]),
        ):
            with self.subTest(
                behavior_seeds=behavior_seeds,
                ablation_seeds=ablation_seeds,
            ):
                base_summary = deepcopy(budget.to_summary())
                summary = build_run_budget_summary(
                    budget,
                    scenario_episodes=2,
                    behavior_seeds=behavior_seeds,
                    ablation_seeds=ablation_seeds,
                )
                self.assertEqual(
                    budget.to_summary(),
                    base_summary,
                    "build_run_budget_summary() must not mutate budget.",
                )

                expected = budget.to_summary()
                expected["resolved"]["scenario_episodes"] = 2
                expected["resolved"]["behavior_seeds"] = list(behavior_seeds)
                expected["resolved"]["ablation_seeds"] = list(ablation_seeds)

                self.assertEqual(summary, expected)
                self.assertEqual(summary["profile"], base_summary["profile"])
                self.assertEqual(
                    summary["benchmark_strength"],
                    base_summary["benchmark_strength"],
                )
                self.assertEqual(summary["overrides"], base_summary["overrides"])
                self.assertEqual(
                    summary["resolved"]["max_steps"],
                    base_summary["resolved"]["max_steps"],
                )
                self.assertEqual(
                    summary["resolved"]["checkpoint_interval"],
                    base_summary["resolved"]["checkpoint_interval"],
                )

    def test_create_comparison_simulation_calls_constructor_without_brain_config(
        self,
    ) -> None:
        budget_summary = {"profile": "smoke", "resolved": {"max_steps": 60}}

        with mock.patch(
            "spider_cortex_sim.comparison_utils.SpiderSimulation"
        ) as simulation_cls:
            result = create_comparison_simulation(
                width=8,
                height=9,
                food_count=3,
                day_length=10,
                night_length=5,
                gamma=0.91,
                module_lr=0.02,
                motor_lr=0.03,
                module_dropout=0.25,
                operational_profile="standard",
                noise_profile="low",
                reward_profile="classic",
                map_template="central_burrow",
                max_steps=60,
                budget_profile_name="smoke",
                benchmark_strength="quick",
                budget_summary=budget_summary,
                seed=11,
            )

        self.assertIs(result, simulation_cls.return_value)
        simulation_cls.assert_called_once_with(
            width=8,
            height=9,
            food_count=3,
            day_length=10,
            night_length=5,
            max_steps=60,
            seed=11,
            gamma=0.91,
            module_lr=0.02,
            motor_lr=0.03,
            module_dropout=0.25,
            operational_profile="standard",
            noise_profile="low",
            reward_profile="classic",
            map_template="central_burrow",
            brain_config=None,
            budget_profile_name="smoke",
            benchmark_strength="quick",
            budget_summary=budget_summary,
        )

    def test_create_comparison_simulation_calls_constructor_with_brain_config(
        self,
    ) -> None:
        budget_summary = {"profile": "custom", "resolved": {"max_steps": 30}}
        brain_config = object()

        with mock.patch(
            "spider_cortex_sim.comparison_utils.SpiderSimulation"
        ) as simulation_cls:
            result = create_comparison_simulation(
                width=12,
                height=13,
                food_count=4,
                day_length=18,
                night_length=12,
                gamma=0.96,
                module_lr=0.01,
                motor_lr=0.012,
                module_dropout=0.05,
                operational_profile=None,
                noise_profile=None,
                reward_profile="austere",
                map_template="open_field",
                max_steps=30,
                budget_profile_name="custom",
                benchmark_strength="custom",
                budget_summary=budget_summary,
                seed=7,
                brain_config=brain_config,
            )

        self.assertIs(result, simulation_cls.return_value)
        simulation_cls.assert_called_once_with(
            width=12,
            height=13,
            food_count=4,
            day_length=18,
            night_length=12,
            max_steps=30,
            seed=7,
            gamma=0.96,
            module_lr=0.01,
            motor_lr=0.012,
            module_dropout=0.05,
            operational_profile=None,
            noise_profile=None,
            reward_profile="austere",
            map_template="open_field",
            brain_config=brain_config,
            budget_profile_name="custom",
            benchmark_strength="custom",
            budget_summary=budget_summary,
        )

    def test_build_checkpoint_path_joins_root_workflow_and_run_key(self) -> None:
        root = Path("tests/checkpoints")

        cases = (
            (root, "ablation_compare", "modular_full__seed_7"),
            (str(root), "behavior_compare", "classic__central_burrow__seed_7"),
            (root, "training_regime_compare", "baseline__seed_7"),
            (root, "learning_evidence", "trained__seed_7"),
            (root, "noise_robustness", "low__seed_7__fingerprint"),
        )

        for case_root, workflow, run_key in cases:
            with self.subTest(workflow=workflow):
                self.assertEqual(
                    build_checkpoint_path(case_root, workflow, run_key),
                    Path(case_root) / workflow / run_key,
                )


if __name__ == "__main__":
    unittest.main()
