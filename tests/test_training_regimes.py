import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spider_cortex_sim.cli import build_parser
from spider_cortex_sim.comparison import compare_training_regimes
from spider_cortex_sim.simulation import (
    EXPERIMENT_OF_RECORD_REGIME,
    SpiderSimulation,
)
from spider_cortex_sim.training_regimes import (
    AnnealingSchedule,
    TRAINING_REGIMES,
    TrainingRegimeSpec,
    canonical_training_regime_names,
    resolve_training_regime,
)


def run_spider_cortex_sim(
    args: list[str],
    *,
    check: bool,
    timeout: float | None = 30.0,
) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "spider_cortex_sim", *args],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "."},
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


class TrainingRegimeSpecTest(unittest.TestCase):
    def test_canonical_names_include_requested_regimes(self) -> None:
        self.assertEqual(
            canonical_training_regime_names(),
            ("baseline", "reflex_annealed", "late_finetuning"),
        )

    def test_baseline_matches_current_training_defaults(self) -> None:
        spec = resolve_training_regime("baseline")
        self.assertEqual(spec.annealing_schedule, AnnealingSchedule.NONE)
        self.assertEqual(spec.anneal_target_scale, 1.0)
        self.assertEqual(spec.anneal_warmup_fraction, 0.0)
        self.assertEqual(spec.finetuning_episodes, 0)
        self.assertEqual(spec.finetuning_reflex_scale, 0.0)
        self.assertEqual(spec.loss_override_penalty_weight, 0.0)
        self.assertEqual(spec.loss_dominance_penalty_weight, 0.0)

    def test_reflex_annealed_targets_zero_without_finetuning(self) -> None:
        spec = resolve_training_regime("reflex_annealed")
        self.assertEqual(spec.annealing_schedule, AnnealingSchedule.LINEAR)
        self.assertAlmostEqual(spec.anneal_target_scale, 0.0)
        self.assertEqual(spec.anneal_warmup_fraction, 0.0)
        self.assertEqual(spec.finetuning_episodes, 0)
        self.assertEqual(spec.loss_override_penalty_weight, 0.0)
        self.assertEqual(spec.loss_dominance_penalty_weight, 0.0)

    def test_late_finetuning_adds_no_reflex_learning_phase(self) -> None:
        spec = resolve_training_regime("late_finetuning")
        self.assertEqual(spec.annealing_schedule, AnnealingSchedule.LINEAR)
        self.assertAlmostEqual(spec.anneal_target_scale, 0.0)
        self.assertGreater(spec.finetuning_episodes, 0)
        self.assertAlmostEqual(spec.finetuning_reflex_scale, 0.0)
        self.assertEqual(spec.loss_override_penalty_weight, 0.0)
        self.assertEqual(spec.loss_dominance_penalty_weight, 0.0)

    def test_distillation_is_reserved_but_not_cli_canonical(self) -> None:
        self.assertIn("distillation", TRAINING_REGIMES)
        self.assertNotIn("distillation", canonical_training_regime_names())
        with self.assertRaises(NotImplementedError):
            resolve_training_regime("distillation")

    def test_unknown_regime_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_training_regime("unknown")

    def test_invalid_schedule_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TrainingRegimeSpec(
                name="bad",
                annealing_schedule="exp",
                anneal_target_scale=0.0,
                anneal_warmup_fraction=0.0,
                finetuning_episodes=0,
                finetuning_reflex_scale=0.0,
                loss_override_penalty_weight=0.0,
                loss_dominance_penalty_weight=0.0,
            )

    def test_negative_penalty_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            TrainingRegimeSpec(
                name="bad",
                annealing_schedule=AnnealingSchedule.LINEAR,
                anneal_target_scale=0.0,
                anneal_warmup_fraction=0.0,
                finetuning_episodes=0,
                finetuning_reflex_scale=0.0,
                loss_override_penalty_weight=-0.1,
                loss_dominance_penalty_weight=0.0,
            )

    def test_non_numeric_float_fields_raise_field_value_error(self) -> None:
        base_kwargs = {
            "name": "bad",
            "annealing_schedule": AnnealingSchedule.LINEAR,
            "anneal_target_scale": 0.0,
            "anneal_warmup_fraction": 0.0,
            "finetuning_episodes": 0,
            "finetuning_reflex_scale": 0.0,
            "loss_override_penalty_weight": 0.0,
            "loss_dominance_penalty_weight": 0.0,
        }
        for field_name, value in (
            ("anneal_target_scale", None),
            ("loss_override_penalty_weight", "bad"),
        ):
            kwargs = dict(base_kwargs)
            kwargs[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{field_name} must be a number",
                ):
                    TrainingRegimeSpec(**kwargs)

    def test_none_schedule_rejects_warmup_fraction(self) -> None:
        with self.assertRaises(ValueError):
            TrainingRegimeSpec(
                name="bad",
                annealing_schedule=AnnealingSchedule.NONE,
                anneal_target_scale=1.0,
                anneal_warmup_fraction=0.1,
                finetuning_episodes=0,
                finetuning_reflex_scale=0.0,
                loss_override_penalty_weight=0.0,
                loss_dominance_penalty_weight=0.0,
            )

    def test_active_schedule_rejects_warmup_fraction_at_or_above_one(self) -> None:
        for warmup_fraction in (1.0, 1.1):
            with self.subTest(warmup_fraction=warmup_fraction):
                with self.assertRaises(ValueError):
                    TrainingRegimeSpec(
                        name="bad",
                        annealing_schedule=AnnealingSchedule.LINEAR,
                        anneal_target_scale=0.0,
                        anneal_warmup_fraction=warmup_fraction,
                        finetuning_episodes=0,
                        finetuning_reflex_scale=0.0,
                        loss_override_penalty_weight=0.0,
                        loss_dominance_penalty_weight=0.0,
                    )

    def test_finetuning_episodes_must_be_integer(self) -> None:
        for finetuning_episodes in (True, 1.5, "1"):
            with self.subTest(finetuning_episodes=finetuning_episodes):
                with self.assertRaises(ValueError):
                    TrainingRegimeSpec(
                        name="bad",
                        annealing_schedule=AnnealingSchedule.LINEAR,
                        anneal_target_scale=0.0,
                        anneal_warmup_fraction=0.0,
                        finetuning_episodes=finetuning_episodes,
                        finetuning_reflex_scale=0.0,
                        loss_override_penalty_weight=0.0,
                        loss_dominance_penalty_weight=0.0,
                    )

    def test_finetuning_requires_active_completed_anneal(self) -> None:
        """Reject finetuning unless annealing is active and completes to target."""
        with self.assertRaises(ValueError):
            TrainingRegimeSpec(
                name="bad",
                annealing_schedule=AnnealingSchedule.NONE,
                anneal_target_scale=1.0,
                anneal_warmup_fraction=0.0,
                finetuning_episodes=1,
                finetuning_reflex_scale=0.0,
                loss_override_penalty_weight=0.0,
                loss_dominance_penalty_weight=0.0,
            )
        with self.assertRaises(ValueError):
            TrainingRegimeSpec(
                name="bad",
                annealing_schedule=AnnealingSchedule.LINEAR,
                anneal_target_scale=0.5,
                anneal_warmup_fraction=0.0,
                finetuning_episodes=1,
                finetuning_reflex_scale=0.0,
                loss_override_penalty_weight=0.0,
                loss_dominance_penalty_weight=0.0,
            )


class TrainingRegimeScheduleTest(unittest.TestCase):
    def test_linear_schedule_progression(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        scales = [
            sim._set_training_episode_reflex_scale(
                episode_index=episode_index,
                total_episodes=5,
                final_scale=0.0,
                schedule=AnnealingSchedule.LINEAR,
            )
            for episode_index in range(5)
        ]

        self.assertEqual(scales, [1.0, 0.75, 0.5, 0.25, 0.0])

    def test_cosine_schedule_reaches_target(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        start = sim._set_training_episode_reflex_scale(
            episode_index=0,
            total_episodes=5,
            final_scale=0.0,
            schedule=AnnealingSchedule.COSINE,
        )
        end = sim._set_training_episode_reflex_scale(
            episode_index=4,
            total_episodes=5,
            final_scale=0.0,
            schedule=AnnealingSchedule.COSINE,
        )
        self.assertAlmostEqual(start, 1.0)
        middle = sim._set_training_episode_reflex_scale(
            episode_index=2,
            total_episodes=5,
            final_scale=0.0,
            schedule=AnnealingSchedule.COSINE,
        )
        self.assertAlmostEqual(middle, 0.5)
        self.assertAlmostEqual(end, 0.0)

    def test_warmup_holds_start_scale_before_annealing(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        warmup_scale = sim._set_training_episode_reflex_scale(
            episode_index=1,
            total_episodes=6,
            final_scale=0.0,
            schedule=AnnealingSchedule.LINEAR,
            warmup_fraction=0.5,
        )
        later_scale = sim._set_training_episode_reflex_scale(
            episode_index=4,
            total_episodes=6,
            final_scale=0.0,
            schedule=AnnealingSchedule.LINEAR,
            warmup_fraction=0.5,
        )
        self.assertAlmostEqual(warmup_scale, 1.0)
        self.assertLess(later_scale, 1.0)


class TrainingRegimeSimulationTest(unittest.TestCase):
    def test_named_regime_is_recorded_in_summary(self) -> None:
        """Record named regimes and their resolved reflex schedule in summaries."""
        sim = SpiderSimulation(seed=7, max_steps=1)
        summary, _ = sim.train(
            episodes=2,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            training_regime="reflex_annealed",
        )
        regime = summary["config"]["training_regime"]
        schedule = summary["config"]["reflex_schedule"]
        self.assertEqual(regime["name"], "reflex_annealed")
        self.assertEqual(regime["annealing_schedule"], "linear")
        self.assertEqual(schedule["mode"], "linear")
        self.assertEqual(schedule["warmup_fraction"], 0.0)
        self.assertAlmostEqual(schedule["final_scale"], 0.0)

    def test_explicit_regime_runs_late_finetuning_phase(self) -> None:
        """Run explicit late-finetuning specs and report the resolved budget."""
        spec = TrainingRegimeSpec(
            name="tiny_late_finetuning",
            annealing_schedule=AnnealingSchedule.LINEAR,
            anneal_target_scale=0.0,
            anneal_warmup_fraction=0.0,
            finetuning_episodes=2,
            finetuning_reflex_scale=0.0,
            loss_override_penalty_weight=0.0,
            loss_dominance_penalty_weight=0.0,
        )
        sim = SpiderSimulation(seed=7, max_steps=1)
        summary, _ = sim.train(
            episodes=1,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            training_regime=spec,
        )
        self.assertEqual(summary["training"]["episodes"], 3)
        regime = summary["config"]["training_regime"]
        self.assertEqual(regime["name"], "tiny_late_finetuning")
        self.assertEqual(regime["resolved_budget"]["main_training_episodes"], 1)
        self.assertEqual(regime["resolved_budget"]["finetuning_episodes"], 2)
        self.assertEqual(regime["resolved_budget"]["total_training_episodes"], 3)
        self.assertAlmostEqual(sim.brain.current_reflex_scale, 0.0)

    def test_late_finetuning_summary_is_experiment_of_record(self) -> None:
        """Mark late-finetuning summaries as the experiment of record."""
        spec = TrainingRegimeSpec(
            name=EXPERIMENT_OF_RECORD_REGIME,
            annealing_schedule=AnnealingSchedule.LINEAR,
            anneal_target_scale=0.0,
            anneal_warmup_fraction=0.0,
            finetuning_episodes=1,
            finetuning_reflex_scale=0.0,
            loss_override_penalty_weight=0.0,
            loss_dominance_penalty_weight=0.0,
        )
        sim = SpiderSimulation(seed=7, max_steps=1)
        summary, _ = sim.train(
            episodes=1,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            training_regime=spec,
        )

        self.assertTrue(summary["is_experiment_of_record"])
        self.assertTrue(
            summary["config"]["training_regime"]["is_experiment_of_record"]
        )

    def test_training_regime_and_manual_anneal_are_rejected_together(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=1)
        with self.assertRaises(ValueError):
            sim.train(
                episodes=1,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
                training_regime="baseline",
                reflex_anneal_final_scale=0.5,
            )

    def test_named_regime_comparison_reports_competence_gaps(self) -> None:
        payload, rows = compare_training_regimes(
            regime_names=["reflex_annealed", EXPERIMENT_OF_RECORD_REGIME],
            budget_profile="smoke",
            episodes=1,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
            episodes_per_scenario=1,
        )

        self.assertEqual(payload["comparison_type"], "training_regimes")
        self.assertEqual(payload["reference_regime"], "baseline")
        self.assertEqual(
            payload["experiment_of_record_regime"],
            EXPERIMENT_OF_RECORD_REGIME,
        )
        self.assertIn("baseline", payload["regimes"])
        self.assertIn("reflex_annealed", payload["regimes"])
        self.assertIn(EXPERIMENT_OF_RECORD_REGIME, payload["regimes"])
        self.assertTrue(
            payload["regimes"][EXPERIMENT_OF_RECORD_REGIME][
                "is_experiment_of_record"
            ]
        )
        self.assertIn(
            "scenario_success_rate_delta",
            payload["regimes"]["reflex_annealed"]["competence_gap"],
        )
        self.assertIn("reflex_annealed", payload["deltas_vs_baseline"])
        self.assertTrue(rows)
        self.assertTrue(
            any(row["competence_type"] == "self_sufficient" for row in rows)
        )
        self.assertTrue(any(row["competence_type"] == "scaffolded" for row in rows))

    def test_regime_comparison_scaffolded_scale_uses_trained_runtime_scale(self) -> None:
        """
        Verify that scaffolded evaluations record the trained runtime reflex scale for the "reflex_annealed" regime.
        
        Asserts that the regime payload's scaffolded summary `eval_reflex_scale` is 0.0 and that every scaffolded row for `reflex_annealed` has an `eval_reflex_scale` value equal to 0.0 when converted to float.
        """
        payload, rows = compare_training_regimes(
            regime_names=["reflex_annealed"],
            episodes=2,
            evaluation_episodes=0,
            max_steps=1,
            names=("night_rest",),
            seeds=(7,),
            episodes_per_scenario=1,
        )

        reflex_payload = payload["regimes"]["reflex_annealed"]
        self.assertEqual(
            reflex_payload["scaffolded"]["summary"]["eval_reflex_scale"],
            0.0,
        )
        scaffolded_rows = [
            row
            for row in rows
            if (
                row.get("training_regime_name") == "reflex_annealed"
                and row.get("competence_type") == "scaffolded"
            )
        ]
        self.assertTrue(scaffolded_rows)
        self.assertTrue(
            all(float(row["eval_reflex_scale"]) == 0.0 for row in scaffolded_rows)
        )


class TrainingRegimeCliTest(unittest.TestCase):
    def test_training_regime_parser_arg(self) -> None:
        args = build_parser().parse_args(["--training-regime", "reflex_annealed"])
        self.assertEqual(args.training_regime, "reflex_annealed")

    def test_training_regime_parser_rejects_reserved_distillation(self) -> None:
        parser = build_parser()
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--training-regime", "distillation"])

    def test_training_regime_and_reflex_anneal_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--training-regime",
                    "reflex_annealed",
                    "--reflex-anneal-final-scale",
                    "0.2",
                ]
            )

    def test_experiment_of_record_parser_arg(self) -> None:
        args = build_parser().parse_args(["--experiment-of-record"])
        self.assertTrue(args.experiment_of_record)

    def test_experiment_of_record_rejects_custom_reflex_flags(self) -> None:
        for reflex_args in (
            ["--reflex-scale", "0.5"],
            ["--module-reflex-scale", "alert_center=0.5"],
        ):
            with self.subTest(reflex_args=reflex_args):
                proc = run_spider_cortex_sim(
                    [
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        "--experiment-of-record",
                        *reflex_args,
                    ],
                    check=False,
                )

                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("--experiment-of-record", proc.stderr)
                self.assertIn("custom reflex", proc.stderr.lower())
                self.assertIn(EXPERIMENT_OF_RECORD_REGIME, proc.stderr)

    def test_named_regime_error_mentions_training_regime_for_unsupported_workflow(self) -> None:
        proc = run_spider_cortex_sim(
            [
                "--episodes",
                "0",
                "--eval-episodes",
                "0",
                "--ablation-variant",
                "monolithic_policy",
                "--training-regime",
                "baseline",
            ],
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--training-regime", proc.stderr)
        self.assertIn("named training regime", proc.stderr.lower())

    def test_experiment_of_record_cli_selects_regime_and_direct_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            proc = run_spider_cortex_sim(
                [
                    "--episodes",
                    "1",
                    "--eval-episodes",
                    "0",
                    "--max-steps",
                    "1",
                    "--behavior-scenario",
                    "night_rest",
                    "--experiment-of-record",
                    "--summary",
                    str(summary_path),
                    "--full-summary",
                ],
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertTrue(payload["is_experiment_of_record"])
            self.assertEqual(
                payload["config"]["training_regime"]["name"],
                EXPERIMENT_OF_RECORD_REGIME,
            )
            self.assertEqual(payload["checkpointing"]["selection"], "best")
            self.assertEqual(payload["checkpointing"]["penalty_mode"], "direct")
            self.assertEqual(
                payload["checkpointing"]["penalty_config"][
                    "override_penalty_weight"
                ],
                1.0,
            )
            self.assertTrue(summary_path.exists())
            with summary_path.open("r", encoding="utf-8") as handle:
                persisted_payload = json.load(handle)
            self.assertTrue(persisted_payload["is_experiment_of_record"])
            self.assertEqual(
                persisted_payload["config"]["training_regime"]["name"],
                EXPERIMENT_OF_RECORD_REGIME,
            )
            self.assertEqual(
                persisted_payload["checkpointing"]["selection"],
                "best",
            )
            self.assertEqual(
                persisted_payload["checkpointing"]["penalty_mode"],
                "direct",
            )
            self.assertEqual(
                persisted_payload["checkpointing"]["penalty_config"][
                    "override_penalty_weight"
                ],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
