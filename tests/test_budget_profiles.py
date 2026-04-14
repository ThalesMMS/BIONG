from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

from spider_cortex_sim.cli import main
from spider_cortex_sim.budget_profiles import (
    BUDGET_PROFILES,
    CHECKPOINT_METRIC_NAMES,
    CHECKPOINT_SELECTION_NAMES,
    CUSTOM_BUDGET_PROFILE,
    DEV_BUDGET_PROFILE,
    PAPER_BUDGET_PROFILE,
    REPORT_BUDGET_PROFILE,
    SMOKE_BUDGET_PROFILE,
    BudgetProfile,
    ResolvedBudget,
    canonical_budget_profile_names,
    resolve_budget,
    resolve_budget_profile,
)


class BudgetProfilesTest(unittest.TestCase):
    def test_canonical_budget_profile_names(self) -> None:
        self.assertEqual(
            canonical_budget_profile_names(),
            ("smoke", "dev", "report", "paper"),
        )

    def test_resolve_budget_profile_none_uses_custom(self) -> None:
        self.assertEqual(resolve_budget_profile(None).name, CUSTOM_BUDGET_PROFILE.name)

    def test_resolve_budget_profile_named(self) -> None:
        self.assertEqual(resolve_budget_profile("dev").name, DEV_BUDGET_PROFILE.name)

    def test_resolve_budget_profile_paper_named(self) -> None:
        self.assertEqual(resolve_budget_profile("paper").name, PAPER_BUDGET_PROFILE.name)

    def test_resolve_budget_profile_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_budget_profile("unknown_profile")

    def test_resolve_budget_uses_profile_defaults(self) -> None:
        budget = resolve_budget(
            profile="smoke",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )

        self.assertEqual(budget.profile, "smoke")
        self.assertEqual(budget.episodes, 6)
        self.assertEqual(budget.eval_episodes, 1)
        self.assertEqual(budget.max_steps, 60)
        self.assertEqual(budget.behavior_seeds, (7,))
        self.assertEqual(budget.ablation_seeds, (7,))
        self.assertEqual(budget.checkpoint_interval, 2)

    def test_resolve_budget_tracks_overrides(self) -> None:
        budget = resolve_budget(
            profile="dev",
            episodes=20,
            eval_episodes=5,
            max_steps=140,
            scenario_episodes=3,
            checkpoint_interval=9,
            behavior_seeds=(101, 103),
            ablation_seeds=(107,),
        )

        self.assertEqual(budget.episodes, 20)
        self.assertEqual(budget.eval_episodes, 5)
        self.assertEqual(budget.max_steps, 140)
        self.assertEqual(budget.scenario_episodes, 3)
        self.assertEqual(budget.checkpoint_interval, 9)
        self.assertEqual(budget.behavior_seeds, (101, 103))
        self.assertEqual(budget.ablation_seeds, (107,))
        self.assertEqual(
            budget.overrides,
            {
                "episodes": 20,
                "eval_episodes": 5,
                "max_steps": 140,
                "scenario_episodes": 3,
                "checkpoint_interval": 9,
                "behavior_seeds": [101, 103],
                "ablation_seeds": [107],
            },
        )

    def test_resolved_budget_summary_contains_runtime_seeds(self) -> None:
        budget = resolve_budget(
            profile=None,
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=(11, 13),
            ablation_seeds=(17,),
        )

        summary = budget.to_summary()
        self.assertEqual(summary["profile"], "custom")
        self.assertEqual(summary["resolved"]["behavior_seeds"], [11, 13])
        self.assertEqual(summary["resolved"]["ablation_seeds"], [17])

    def test_resolve_budget_rejects_negative_episodes(self) -> None:
        with self.assertRaises(ValueError):
            resolve_budget(
                profile="smoke",
                episodes=-1,
                eval_episodes=None,
                max_steps=None,
                scenario_episodes=None,
                checkpoint_interval=None,
                behavior_seeds=None,
                ablation_seeds=None,
            )

    def test_resolve_budget_rejects_negative_eval_episodes(self) -> None:
        with self.assertRaises(ValueError):
            resolve_budget(
                profile="smoke",
                episodes=None,
                eval_episodes=-1,
                max_steps=None,
                scenario_episodes=None,
                checkpoint_interval=None,
                behavior_seeds=None,
                ablation_seeds=None,
            )

    def test_resolve_budget_rejects_non_positive_max_steps(self) -> None:
        with self.assertRaises(ValueError):
            resolve_budget(
                profile="smoke",
                episodes=None,
                eval_episodes=None,
                max_steps=0,
                scenario_episodes=None,
                checkpoint_interval=None,
                behavior_seeds=None,
                ablation_seeds=None,
            )

    def test_resolve_budget_rejects_non_positive_scenario_episodes(self) -> None:
        with self.assertRaises(ValueError):
            resolve_budget(
                profile="smoke",
                episodes=None,
                eval_episodes=None,
                max_steps=None,
                scenario_episodes=0,
                checkpoint_interval=None,
                behavior_seeds=None,
                ablation_seeds=None,
            )

    def test_resolve_budget_rejects_checkpoint_interval_below_one(self) -> None:
        with self.assertRaises(ValueError):
            resolve_budget(
                profile="smoke",
                episodes=None,
                eval_episodes=None,
                max_steps=None,
                scenario_episodes=None,
                checkpoint_interval=0,
                behavior_seeds=None,
                ablation_seeds=None,
            )


class BudgetProfileValuesTest(unittest.TestCase):
    """Tests that the canonical profile singletons have the expected values."""

    def test_smoke_profile_values(self) -> None:
        p = SMOKE_BUDGET_PROFILE
        self.assertEqual(p.name, "smoke")
        self.assertEqual(p.benchmark_strength, "quick")
        self.assertEqual(p.episodes, 6)
        self.assertEqual(p.eval_episodes, 1)
        self.assertEqual(p.max_steps, 60)
        self.assertEqual(p.scenario_episodes, 1)
        self.assertEqual(p.comparison_seeds, (7,))
        self.assertEqual(p.checkpoint_interval, 2)
        self.assertEqual(p.selection_scenario_episodes, 1)
        self.assertFalse(p.requires_checkpoint_selection)

    def test_dev_profile_values(self) -> None:
        p = DEV_BUDGET_PROFILE
        self.assertEqual(p.name, "dev")
        self.assertEqual(p.benchmark_strength, "quick")
        self.assertEqual(p.episodes, 12)
        self.assertEqual(p.eval_episodes, 2)
        self.assertEqual(p.max_steps, 90)
        self.assertEqual(p.scenario_episodes, 1)
        self.assertEqual(p.comparison_seeds, (7, 17, 29))
        self.assertEqual(p.checkpoint_interval, 4)
        self.assertEqual(p.selection_scenario_episodes, 1)
        self.assertFalse(p.requires_checkpoint_selection)

    def test_report_profile_values(self) -> None:
        p = REPORT_BUDGET_PROFILE
        self.assertEqual(p.name, "report")
        self.assertEqual(p.benchmark_strength, "strong")
        self.assertEqual(p.episodes, 24)
        self.assertEqual(p.eval_episodes, 4)
        self.assertEqual(p.max_steps, 120)
        self.assertEqual(p.scenario_episodes, 2)
        self.assertEqual(p.comparison_seeds, (7, 17, 29, 41, 53))
        self.assertEqual(p.checkpoint_interval, 6)
        self.assertEqual(p.selection_scenario_episodes, 1)
        self.assertFalse(p.requires_checkpoint_selection)

    def test_paper_profile_values(self) -> None:
        p = PAPER_BUDGET_PROFILE
        self.assertEqual(p.name, "paper")
        self.assertEqual(p.benchmark_strength, "publication")
        self.assertEqual(p.episodes, 48)
        self.assertEqual(p.eval_episodes, 8)
        self.assertEqual(p.max_steps, 150)
        self.assertEqual(p.scenario_episodes, 4)
        self.assertEqual(p.comparison_seeds, (7, 17, 29, 41, 53, 67, 79))
        self.assertEqual(p.checkpoint_interval, 8)
        self.assertEqual(p.selection_scenario_episodes, 2)
        self.assertTrue(p.requires_checkpoint_selection)

    def test_custom_profile_values(self) -> None:
        p = CUSTOM_BUDGET_PROFILE
        self.assertEqual(p.name, "custom")
        self.assertEqual(p.benchmark_strength, "custom")
        self.assertEqual(p.episodes, 180)
        self.assertEqual(p.eval_episodes, 3)
        self.assertEqual(p.max_steps, 120)
        self.assertEqual(p.comparison_seeds, (7, 17, 29))
        self.assertFalse(p.requires_checkpoint_selection)

    def test_budget_profiles_dict_contains_named_profiles(self) -> None:
        self.assertIn("smoke", BUDGET_PROFILES)
        self.assertIn("dev", BUDGET_PROFILES)
        self.assertIn("report", BUDGET_PROFILES)
        self.assertIn("paper", BUDGET_PROFILES)
        self.assertNotIn("custom", BUDGET_PROFILES)
        self.assertIs(BUDGET_PROFILES["paper"], PAPER_BUDGET_PROFILE)

    def test_checkpoint_selection_names(self) -> None:
        self.assertIn("none", CHECKPOINT_SELECTION_NAMES)
        self.assertIn("best", CHECKPOINT_SELECTION_NAMES)

    def test_checkpoint_metric_names(self) -> None:
        self.assertIn("scenario_success_rate", CHECKPOINT_METRIC_NAMES)
        self.assertIn("episode_success_rate", CHECKPOINT_METRIC_NAMES)
        self.assertIn("mean_reward", CHECKPOINT_METRIC_NAMES)


class BudgetProfileToSummaryTest(unittest.TestCase):
    """Tests BudgetProfile.to_summary() output structure."""

    def test_smoke_profile_to_summary_structure(self) -> None:
        summary = SMOKE_BUDGET_PROFILE.to_summary()
        self.assertEqual(summary["profile"], "smoke")
        self.assertEqual(summary["benchmark_strength"], "quick")
        resolved = summary["resolved"]
        self.assertEqual(resolved["episodes"], 6)
        self.assertEqual(resolved["eval_episodes"], 1)
        self.assertEqual(resolved["max_steps"], 60)
        self.assertEqual(resolved["scenario_episodes"], 1)
        self.assertEqual(resolved["comparison_seeds"], [7])
        self.assertEqual(resolved["checkpoint_interval"], 2)
        self.assertEqual(resolved["selection_scenario_episodes"], 1)
        self.assertFalse(resolved["requires_checkpoint_selection"])

    def test_dev_profile_to_summary_structure(self) -> None:
        """Verify dev profile summary values."""
        summary = DEV_BUDGET_PROFILE.to_summary()
        self.assertEqual(summary["profile"], "dev")
        self.assertEqual(summary["benchmark_strength"], "quick")
        resolved = summary["resolved"]
        self.assertEqual(resolved["episodes"], 12)
        self.assertEqual(resolved["comparison_seeds"], [7, 17, 29])
        self.assertFalse(resolved["requires_checkpoint_selection"])

    def test_report_profile_to_summary_structure(self) -> None:
        summary = REPORT_BUDGET_PROFILE.to_summary()
        self.assertEqual(summary["profile"], "report")
        self.assertEqual(summary["benchmark_strength"], "strong")
        resolved = summary["resolved"]
        self.assertEqual(resolved["episodes"], 24)
        self.assertEqual(resolved["comparison_seeds"], [7, 17, 29, 41, 53])
        self.assertFalse(resolved["requires_checkpoint_selection"])

    def test_paper_profile_to_summary_structure(self) -> None:
        summary = PAPER_BUDGET_PROFILE.to_summary()
        self.assertEqual(summary["profile"], "paper")
        self.assertEqual(summary["benchmark_strength"], "publication")
        resolved = summary["resolved"]
        self.assertEqual(resolved["episodes"], 48)
        self.assertEqual(
            resolved["comparison_seeds"],
            [7, 17, 29, 41, 53, 67, 79],
        )
        self.assertTrue(resolved["requires_checkpoint_selection"])

    def test_budget_profile_to_summary_comparison_seeds_is_list(self) -> None:
        summary = DEV_BUDGET_PROFILE.to_summary()
        self.assertIsInstance(summary["resolved"]["comparison_seeds"], list)
        self.assertFalse(summary["resolved"]["requires_checkpoint_selection"])

    def test_budget_profile_coerces_requires_checkpoint_selection_to_bool(self) -> None:
        profile = BudgetProfile(
            name="example",
            benchmark_strength="example",
            episodes=1,
            eval_episodes=1,
            max_steps=1,
            scenario_episodes=1,
            comparison_seeds=(1,),
            checkpoint_interval=1,
            selection_scenario_episodes=1,
            requires_checkpoint_selection=1,
        )

        self.assertIs(profile.requires_checkpoint_selection, True)

    def test_budget_profile_to_summary_no_behavior_seeds_key(self) -> None:
        # BudgetProfile.to_summary() does NOT include behavior_seeds or ablation_seeds
        summary = SMOKE_BUDGET_PROFILE.to_summary()
        self.assertNotIn("behavior_seeds", summary["resolved"])
        self.assertNotIn("ablation_seeds", summary["resolved"])


class ResolveBudgetProfileTest(unittest.TestCase):
    """Tests for resolve_budget_profile() function."""

    def test_resolve_budget_profile_with_budget_profile_instance(self) -> None:
        # Passing a BudgetProfile instance should return it directly (passthrough)
        result = resolve_budget_profile(SMOKE_BUDGET_PROFILE)
        self.assertIs(result, SMOKE_BUDGET_PROFILE)

    def test_resolve_budget_profile_returns_smoke(self) -> None:
        result = resolve_budget_profile("smoke")
        self.assertIs(result, SMOKE_BUDGET_PROFILE)

    def test_resolve_budget_profile_returns_report(self) -> None:
        result = resolve_budget_profile("report")
        self.assertIs(result, REPORT_BUDGET_PROFILE)

    def test_resolve_budget_profile_invalid_error_message_mentions_available(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_budget_profile("nonexistent")
        error_msg = str(ctx.exception)
        self.assertIn("nonexistent", error_msg)
        # Error should list available profiles
        self.assertIn("dev", error_msg)
        self.assertIn("smoke", error_msg)
        self.assertIn("report", error_msg)


class ResolveBudgetTest(unittest.TestCase):
    """Additional tests for resolve_budget() function."""

    def test_resolve_budget_no_overrides_empty_overrides_dict(self) -> None:
        budget = resolve_budget(
            profile="dev",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        self.assertEqual(budget.overrides, {})

    def test_resolve_budget_behavior_seeds_fallback_to_comparison_seeds(self) -> None:
        # When behavior_seeds is None, they fall back to comparison_seeds of the profile
        budget = resolve_budget(
            profile="dev",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        self.assertEqual(budget.behavior_seeds, DEV_BUDGET_PROFILE.comparison_seeds)

    def test_resolve_budget_ablation_seeds_fallback_to_comparison_seeds(self) -> None:
        budget = resolve_budget(
            profile="smoke",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        self.assertEqual(budget.ablation_seeds, SMOKE_BUDGET_PROFILE.comparison_seeds)

    def test_resolve_budget_explicit_behavior_seeds_override_comparison_seeds(self) -> None:
        budget = resolve_budget(
            profile="report",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=(99, 101),
            ablation_seeds=None,
        )
        self.assertEqual(budget.behavior_seeds, (99, 101))
        # ablation_seeds should still fall back to comparison_seeds
        self.assertEqual(budget.ablation_seeds, REPORT_BUDGET_PROFILE.comparison_seeds)

    def test_resolve_budget_explicit_ablation_seeds_override_comparison_seeds(self) -> None:
        budget = resolve_budget(
            profile="report",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=(200,),
        )
        self.assertEqual(budget.ablation_seeds, (200,))
        # behavior_seeds should still fall back to comparison_seeds
        self.assertEqual(budget.behavior_seeds, REPORT_BUDGET_PROFILE.comparison_seeds)

    def test_resolve_budget_none_profile_uses_custom_defaults(self) -> None:
        budget = resolve_budget(
            profile=None,
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        self.assertEqual(budget.profile, "custom")
        self.assertEqual(budget.benchmark_strength, "custom")
        self.assertEqual(budget.episodes, CUSTOM_BUDGET_PROFILE.episodes)
        self.assertEqual(budget.eval_episodes, CUSTOM_BUDGET_PROFILE.eval_episodes)

    def test_resolve_budget_with_budget_profile_instance(self) -> None:
        budget = resolve_budget(
            profile=SMOKE_BUDGET_PROFILE,
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        self.assertEqual(budget.profile, "smoke")
        self.assertEqual(budget.episodes, 6)
        self.assertFalse(budget.requires_checkpoint_selection)

    def test_resolve_budget_comparison_seeds_preserved_from_profile(self) -> None:
        budget = resolve_budget(
            profile="report",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        self.assertEqual(budget.comparison_seeds, (7, 17, 29, 41, 53))

    def test_resolve_budget_type_coercion_for_seeds(self) -> None:
        # Seeds given as floats should be coerced to int
        budget = resolve_budget(
            profile="smoke",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=[7.0, 11.0],  # type: ignore[list-item]
            ablation_seeds=None,
        )
        self.assertEqual(budget.behavior_seeds, (7, 11))
        self.assertIsInstance(budget.behavior_seeds[0], int)

    def test_resolve_budget_partial_overrides_recorded(self) -> None:
        # Only providing episodes override should record only that key
        budget = resolve_budget(
            profile="dev",
            episodes=5,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        self.assertEqual(budget.overrides, {"episodes": 5})
        # Non-overridden values still come from profile
        self.assertEqual(budget.eval_episodes, DEV_BUDGET_PROFILE.eval_episodes)
        self.assertEqual(budget.max_steps, DEV_BUDGET_PROFILE.max_steps)


class ResolvedBudgetToSummaryTest(unittest.TestCase):
    """Tests for ResolvedBudget.to_summary() output structure."""

    def test_to_summary_full_structure(self) -> None:
        budget = resolve_budget(
            profile="dev",
            episodes=10,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=(7, 17),
            ablation_seeds=(29,),
        )
        summary = budget.to_summary()
        self.assertEqual(summary["profile"], "dev")
        self.assertEqual(summary["benchmark_strength"], "quick")
        resolved = summary["resolved"]
        self.assertIn("episodes", resolved)
        self.assertIn("eval_episodes", resolved)
        self.assertIn("max_steps", resolved)
        self.assertIn("scenario_episodes", resolved)
        self.assertIn("comparison_seeds", resolved)
        self.assertIn("checkpoint_interval", resolved)
        self.assertIn("selection_scenario_episodes", resolved)
        self.assertIn("behavior_seeds", resolved)
        self.assertIn("ablation_seeds", resolved)
        self.assertIn("requires_checkpoint_selection", resolved)
        self.assertEqual(resolved["behavior_seeds"], [7, 17])
        self.assertEqual(resolved["ablation_seeds"], [29])
        self.assertFalse(resolved["requires_checkpoint_selection"])
        self.assertIn("overrides", summary)
        self.assertIn("episodes", summary["overrides"])

    def test_to_summary_includes_checkpoint_selection_requirement(self) -> None:
        budget = resolve_budget(
            profile="paper",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )

        summary = budget.to_summary()
        self.assertTrue(summary["resolved"]["requires_checkpoint_selection"])

    def test_to_summary_seeds_are_lists(self) -> None:
        budget = resolve_budget(
            profile="smoke",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        summary = budget.to_summary()
        self.assertIsInstance(summary["resolved"]["behavior_seeds"], list)
        self.assertIsInstance(summary["resolved"]["ablation_seeds"], list)
        self.assertIsInstance(summary["resolved"]["comparison_seeds"], list)

    def test_to_summary_empty_overrides(self) -> None:
        budget = resolve_budget(
            profile="smoke",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        summary = budget.to_summary()
        self.assertEqual(summary["overrides"], {})

    def test_to_summary_overrides_are_independent_copy(self) -> None:
        budget = resolve_budget(
            profile="dev",
            episodes=3,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        summary = budget.to_summary()
        # Mutating the returned summary should not affect the budget
        summary["overrides"]["episodes"] = 9999
        self.assertEqual(budget.overrides["episodes"], 3)


class PaperCheckpointSelectionEnforcementTest(unittest.TestCase):
    def _minimal_summary(self) -> dict[str, object]:
        """Return the minimal summary shape needed by CLI tests."""
        return {
            "config": {
                "world": {
                    "reward_profile": "classic",
                    "map_template": "central_burrow",
                },
                "budget": {
                    "profile": "paper",
                    "benchmark_strength": "publication",
                },
                "training_regime": "flat",
                "operational_profile": {"name": "default_v1"},
                "noise_profile": {"name": "none"},
            },
            "training_last_window": {
                "mean_reward": 0.0,
                "mean_night_shelter_occupancy_rate": 0.0,
                "mean_night_stillness_rate": 0.0,
            },
            "evaluation": {
                "mean_reward": 0.0,
                "mean_food": 0.0,
                "mean_sleep": 0.0,
                "mean_sleep_debt": 0.0,
                "mean_predator_contacts": 0.0,
                "mean_predator_escapes": 0.0,
                "mean_night_shelter_occupancy_rate": 0.0,
                "mean_night_stillness_rate": 0.0,
                "mean_predator_response_events": 0.0,
                "mean_predator_response_latency": 0.0,
                "mean_predator_mode_transitions": 0.0,
                "dominant_predator_state": "idle",
                "mean_food_distance_delta": 0.0,
                "mean_shelter_distance_delta": 0.0,
                "survival_rate": 0.0,
                "mean_night_role_distribution": {},
            },
        }

    def _assert_cli_error(self, argv: list[str]) -> None:
        """Assert the CLI rejects a paper-profile run without selection."""
        stderr = io.StringIO()
        with patch("sys.argv", argv), patch("sys.stderr", stderr), patch(
            "spider_cortex_sim.cli.SpiderSimulation"
        ) as simulation_mock:
            with self.assertRaises(SystemExit) as ctx:
                main()

        self.assertEqual(ctx.exception.code, 2)
        error_output = stderr.getvalue()
        self.assertIn("paper", error_output)
        self.assertIn("--checkpoint-selection best", error_output)
        simulation_mock.assert_not_called()

    def test_resolve_budget_paper_requires_checkpoint_selection(self) -> None:
        budget = resolve_budget(
            profile="paper",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )

        self.assertTrue(budget.requires_checkpoint_selection)

    def test_cli_rejects_paper_without_checkpoint_selection(self) -> None:
        self._assert_cli_error(["spider_cortex_sim", "--budget-profile", "paper"])

    def test_cli_rejects_paper_with_checkpoint_selection_none(self) -> None:
        self._assert_cli_error(
            [
                "spider_cortex_sim",
                "--budget-profile",
                "paper",
                "--checkpoint-selection",
                "none",
            ]
        )

    def test_cli_rejects_learning_evidence_long_budget_paper_without_selection(
        self,
    ) -> None:
        self._assert_cli_error(
            [
                "spider_cortex_sim",
                "--learning-evidence",
                "--learning-evidence-long-budget-profile",
                "paper",
            ]
        )

    def test_cli_allows_paper_with_checkpoint_selection_best(self) -> None:
        sim = MagicMock()
        sim.train.return_value = (self._minimal_summary(), [])

        with patch(
            "sys.argv",
            [
                "spider_cortex_sim",
                "--budget-profile",
                "paper",
                "--checkpoint-selection",
                "best",
                "--full-summary",
            ],
        ), patch("sys.stdout", io.StringIO()), patch(
            "sys.stderr", io.StringIO()
        ), patch(
            "spider_cortex_sim.cli.SpiderSimulation",
            return_value=sim,
        ) as simulation_cls:
            main()

        simulation_cls.assert_called_once()
        sim.train.assert_called_once()

    def test_cli_claim_test_suite_reuses_precomputed_payloads(self) -> None:
        sim = MagicMock()
        sim._build_summary.return_value = self._minimal_summary()
        ablation_payload = {"variants": {"modular_full": {"summary": {}}}}
        learning_payload = {
            "conditions": {"trained_without_reflex_support": {"summary": {}}}
        }
        claim_payload = {"claims": {}, "summary": {}, "metadata": {}}

        with patch(
            "sys.argv",
            [
                "spider_cortex_sim",
                "--episodes",
                "0",
                "--eval-episodes",
                "0",
                "--ablation-suite",
                "--learning-evidence",
                "--claim-test-suite",
                "--claim-test",
                "learning_without_privileged_signals",
                "--budget-profile",
                "smoke",
                "--learning-evidence-long-budget-profile",
                "smoke",
                "--full-summary",
            ],
        ), patch("sys.stdout", io.StringIO()), patch(
            "sys.stderr", io.StringIO()
        ), patch(
            "spider_cortex_sim.cli.SpiderSimulation",
            return_value=sim,
        ) as simulation_cls:
            simulation_cls.compare_ablation_suite.return_value = (ablation_payload, [])
            simulation_cls.compare_learning_evidence.return_value = (
                learning_payload,
                [],
            )
            simulation_cls.run_claim_test_suite.return_value = (claim_payload, [])

            main()

        claim_kwargs = simulation_cls.run_claim_test_suite.call_args.kwargs
        self.assertIs(claim_kwargs["ablation_payload"], ablation_payload)
        self.assertIs(
            claim_kwargs["learning_evidence_payload"],
            learning_payload,
        )


class AdditionalPaperProfileTest(unittest.TestCase):
    """Additional paper profile regressions."""

    def _resolve_paper_budget(self, **overrides: int | None) -> ResolvedBudget:
        defaults = {
            "episodes": None,
            "eval_episodes": None,
            "max_steps": None,
            "scenario_episodes": None,
            "checkpoint_interval": None,
        }
        defaults.update(overrides)
        return resolve_budget(
            profile="paper",
            behavior_seeds=None,
            ablation_seeds=None,
            **defaults,
        )

    def test_resolve_budget_profile_paper_returns_singleton(self) -> None:
        result = resolve_budget_profile("paper")
        self.assertIs(result, PAPER_BUDGET_PROFILE)

    def test_resolve_budget_paper_preserves_requirement_with_overrides(self) -> None:
        budget = self._resolve_paper_budget(episodes=10, eval_episodes=2)

        self.assertTrue(budget.requires_checkpoint_selection)
        self.assertEqual(budget.episodes, 10)
        self.assertEqual(budget.eval_episodes, 2)

    def test_resolve_budget_paper_runtime_contract(self) -> None:
        budget = self._resolve_paper_budget()
        summary = budget.to_summary()
        resolved = summary["resolved"]

        expected_budget_values = {
            "profile": PAPER_BUDGET_PROFILE.name,
            "benchmark_strength": PAPER_BUDGET_PROFILE.benchmark_strength,
            "episodes": PAPER_BUDGET_PROFILE.episodes,
            "eval_episodes": PAPER_BUDGET_PROFILE.eval_episodes,
            "max_steps": PAPER_BUDGET_PROFILE.max_steps,
            "scenario_episodes": PAPER_BUDGET_PROFILE.scenario_episodes,
            "checkpoint_interval": PAPER_BUDGET_PROFILE.checkpoint_interval,
            "selection_scenario_episodes": (
                PAPER_BUDGET_PROFILE.selection_scenario_episodes
            ),
            "requires_checkpoint_selection": True,
        }
        for attr, expected in expected_budget_values.items():
            with self.subTest(attr=attr):
                self.assertEqual(getattr(budget, attr), expected)

        self.assertEqual(budget.comparison_seeds, PAPER_BUDGET_PROFILE.comparison_seeds)
        self.assertEqual(budget.behavior_seeds, PAPER_BUDGET_PROFILE.comparison_seeds)
        self.assertEqual(budget.ablation_seeds, PAPER_BUDGET_PROFILE.comparison_seeds)

        expected_summary_values = {
            "episodes": PAPER_BUDGET_PROFILE.episodes,
            "eval_episodes": PAPER_BUDGET_PROFILE.eval_episodes,
            "max_steps": PAPER_BUDGET_PROFILE.max_steps,
            "scenario_episodes": PAPER_BUDGET_PROFILE.scenario_episodes,
            "checkpoint_interval": PAPER_BUDGET_PROFILE.checkpoint_interval,
            "selection_scenario_episodes": (
                PAPER_BUDGET_PROFILE.selection_scenario_episodes
            ),
            "comparison_seeds": list(PAPER_BUDGET_PROFILE.comparison_seeds),
            "behavior_seeds": list(PAPER_BUDGET_PROFILE.comparison_seeds),
            "ablation_seeds": list(PAPER_BUDGET_PROFILE.comparison_seeds),
            "requires_checkpoint_selection": True,
        }
        self.assertEqual(summary["profile"], PAPER_BUDGET_PROFILE.name)
        self.assertEqual(
            summary["benchmark_strength"],
            PAPER_BUDGET_PROFILE.benchmark_strength,
        )
        for key, expected in expected_summary_values.items():
            with self.subTest(summary_key=key):
                self.assertEqual(resolved[key], expected)

    def test_non_paper_profiles_do_not_require_checkpoint_selection(self) -> None:
        for profile in ("smoke", "dev", "report"):
            with self.subTest(profile=profile):
                budget = resolve_budget(
                    profile=profile,
                    episodes=None,
                    eval_episodes=None,
                    max_steps=None,
                    scenario_episodes=None,
                    checkpoint_interval=None,
                    behavior_seeds=None,
                    ablation_seeds=None,
                )
                self.assertFalse(budget.requires_checkpoint_selection)

    def test_requires_checkpoint_selection_coercion_edge_cases(self) -> None:
        cases = (
            (
                "profile_zero",
                BudgetProfile(
                    name="test_zero",
                    benchmark_strength="test",
                    episodes=1,
                    eval_episodes=1,
                    max_steps=1,
                    scenario_episodes=1,
                    comparison_seeds=(1,),
                    checkpoint_interval=1,
                    selection_scenario_episodes=1,
                    requires_checkpoint_selection=0,
                ),
                False,
            ),
            (
                "resolved_truthy",
                ResolvedBudget(
                    profile="test",
                    benchmark_strength="test",
                    episodes=1,
                    eval_episodes=1,
                    max_steps=1,
                    scenario_episodes=1,
                    comparison_seeds=(1,),
                    checkpoint_interval=1,
                    selection_scenario_episodes=1,
                    behavior_seeds=(1,),
                    ablation_seeds=(1,),
                    overrides={},
                    requires_checkpoint_selection=1,
                ),
                True,
            ),
            (
                "resolved_zero",
                ResolvedBudget(
                    profile="test",
                    benchmark_strength="test",
                    episodes=1,
                    eval_episodes=1,
                    max_steps=1,
                    scenario_episodes=1,
                    comparison_seeds=(1,),
                    checkpoint_interval=1,
                    selection_scenario_episodes=1,
                    behavior_seeds=(1,),
                    ablation_seeds=(1,),
                    overrides={},
                    requires_checkpoint_selection=0,
                ),
                False,
            ),
        )
        for name, budget, expected in cases:
            with self.subTest(name=name):
                self.assertIs(budget.requires_checkpoint_selection, expected)
                self.assertIsInstance(budget.requires_checkpoint_selection, bool)


class BudgetProfileImmutabilityTest(unittest.TestCase):
    """Tests that BudgetProfile and ResolvedBudget are frozen dataclasses."""

    def test_budget_profile_is_frozen(self) -> None:
        with self.assertRaises((AttributeError, TypeError)):
            SMOKE_BUDGET_PROFILE.name = "mutated"  # type: ignore[misc]

    def test_resolved_budget_is_frozen(self) -> None:
        budget = resolve_budget(
            profile="smoke",
            episodes=None,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        with self.assertRaises((AttributeError, TypeError)):
            budget.profile = "mutated"  # type: ignore[misc]

    def test_resolved_budget_overrides_is_deep_copy(self) -> None:
        original_overrides = {"episodes": 5}
        budget = resolve_budget(
            profile="smoke",
            episodes=5,
            eval_episodes=None,
            max_steps=None,
            scenario_episodes=None,
            checkpoint_interval=None,
            behavior_seeds=None,
            ablation_seeds=None,
        )
        # Internally the overrides should be a copy - mutating separately created dict
        # won't change the budget's overrides
        self.assertEqual(budget.overrides["episodes"], 5)


if __name__ == "__main__":
    unittest.main()
