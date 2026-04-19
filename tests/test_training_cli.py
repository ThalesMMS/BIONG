import unittest
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig, canonical_ablation_configs
from spider_cortex_sim.agent import BrainStep, SpiderBrain
from spider_cortex_sim.curriculum import (
    CurriculumPhaseDefinition,
    PromotionCheckCriteria,
)
from spider_cortex_sim.export import save_behavior_csv, save_summary
from spider_cortex_sim.interfaces import ACTION_TO_INDEX, LOCOMOTION_ACTIONS
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.scenarios import SCENARIO_NAMES
from spider_cortex_sim.simulation import SpiderSimulation
from spider_cortex_sim.world import SpiderWorld

from tests.fixtures.training import SpiderTrainingTestBase

class SpiderTrainingCliTest(SpiderTrainingTestBase):
    def test_cli_invalid_reflex_scale_validation_is_reported_as_usage_error(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "spider_cortex_sim",
                "--reflex-scale",
                "nan",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--reflex-scale", proc.stderr)
        self.assertRegex(proc.stderr.lower(), r"finit")

    def test_cli_brain_config_validation_is_reported_as_usage_error(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "spider_cortex_sim",
                "--module-reflex-scale",
                "motor_cortex_context=0.5",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("module_reflex_scales", proc.stderr)
        self.assertIn("invalid", proc.stderr.lower())

    def test_cli_rejects_custom_reflex_flags_for_ablation_workflows(self) -> None:
        """
        Verify the CLI rejects custom reflex flags when running an ablation workflow.
        
        Runs the package CLI with an ablation variant and a custom `--reflex-scale`, then asserts the process exits with a non-zero status and that stderr indicates the ablation context does not support the custom reflex flag.
        """
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "spider_cortex_sim",
                "--episodes",
                "0",
                "--eval-episodes",
                "0",
                "--ablation-variant",
                "monolithic_policy",
                "--reflex-scale",
                "0.5",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertRegex(proc.stderr.lower(), r"ablation|abla")
        self.assertRegex(proc.stderr.lower(), r"suport|support")

    def test_cli_rejects_custom_reflex_flags_for_claim_test_suite(self) -> None:
        for claim_args in (
            ["--claim-test-suite"],
            ["--claim-test", "learning_without_privileged_signals"],
        ):
            with self.subTest(claim_args=claim_args):
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        *claim_args,
                        "--reflex-scale",
                        "0.5",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("--claim-test-suite", proc.stderr)
                self.assertRegex(proc.stderr.lower(), r"suport|support")

    def test_cli_behavior_suite_emits_json_and_csv(self) -> None:
        """
        Run the package CLI and assert it emits a behavior-evaluation JSON payload and writes a CSV with expected headers.
        
        Executes the CLI for a minimal run (no training episodes) requesting the `night_rest` behavior scenario and a full summary, then:
        - Asserts the stdout JSON contains a top-level `behavior_evaluation` with a `suite` entry that includes `night_rest`.
        - Asserts a CSV file was created and its header row contains key columns including `scenario`, `scenario_map`, `evaluation_map`, `budget_profile`, `benchmark_strength`, `architecture_version`, `architecture_fingerprint`, `operational_profile`, `operational_profile_version`, `noise_profile`, `checkpoint_source`, and `check_deep_night_shelter_passed`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "behavior.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "0",
                    "--eval-episodes",
                    "0",
                    "--behavior-scenario",
                    "night_rest",
                    "--budget-profile",
                    "smoke",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertIn("behavior_evaluation", payload)
            self.assertIn("night_rest", payload["behavior_evaluation"]["suite"])
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("scenario", header)
            self.assertIn("scenario_map", header)
            self.assertIn("evaluation_map", header)
            self.assertIn("budget_profile", header)
            self.assertIn("benchmark_strength", header)
            self.assertIn("architecture_version", header)
            self.assertIn("architecture_fingerprint", header)
            self.assertIn("operational_profile", header)
            self.assertIn("operational_profile_version", header)
            self.assertIn("noise_profile", header)
            self.assertIn("checkpoint_source", header)
            self.assertIn("check_deep_night_shelter_passed", header)

    def test_cli_ablation_variant_emits_summary_and_csv_columns(self) -> None:
        """
        Verify the CLI emits an ablation summary JSON and a CSV containing expected ablation and metadata columns.
        
        Asserts the JSON payload includes ablation metadata: `reference_variant` equal to `"modular_full"`, `budget_profile` and `benchmark_strength` resolved for the smoke profile, `seeds` equal to `[7]`, `scenario_names` matching SCENARIO_NAMES, presence of the requested `monolithic_policy` variant, and an empty `suite`. Asserts the generated CSV exists and its header contains ablation and metadata columns including: `ablation_variant`, `ablation_architecture`, `budget_profile`, `benchmark_strength`, `architecture_version`, `architecture_fingerprint`, `operational_profile`, `operational_profile_version`, `noise_profile`, and `checkpoint_source`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ablation.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "0",
                    "--eval-episodes",
                    "0",
                    "--ablation-variant",
                    "monolithic_policy",
                    "--budget-profile",
                    "smoke",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            ablations = payload["behavior_evaluation"]["ablations"]
            self.assertEqual(ablations["reference_variant"], "modular_full")
            self.assertEqual(ablations["budget_profile"], "smoke")
            self.assertEqual(ablations["benchmark_strength"], "quick")
            self.assertEqual(ablations["seeds"], [7])
            self.assertEqual(ablations["scenario_names"], list(SCENARIO_NAMES))
            self.assertIn("monolithic_policy", ablations["variants"])
            self.assertEqual(payload["behavior_evaluation"]["suite"], {})
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("ablation_variant", header)
            self.assertIn("ablation_architecture", header)
            self.assertIn("budget_profile", header)
            self.assertIn("benchmark_strength", header)
            self.assertIn("architecture_version", header)
            self.assertIn("architecture_fingerprint", header)
            self.assertIn("operational_profile", header)
            self.assertIn("operational_profile_version", header)
            self.assertIn("noise_profile", header)
            self.assertIn("checkpoint_source", header)

    def test_cli_learning_evidence_emits_summary_and_csv_columns(self) -> None:
        """
        Verify the CLI produces a learning-evidence JSON summary and a CSV file containing expected learning-evidence columns.
        
        Runs the command-line tool with learning-evidence and related flags, then asserts the produced JSON contains a learning_evidence section with:
        - reference_condition equal to "trained_without_reflex_support"
        - budget_profile and long_budget_profile set to "smoke"
        - a "trained_final" condition present
        
        Also asserts a CSV file is written and its header contains the columns:
        "learning_evidence_condition", "learning_evidence_policy_mode", "learning_evidence_train_episodes", "learning_evidence_frozen_after_episode", and "learning_evidence_checkpoint_source".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "learning_evidence.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "0",
                    "--eval-episodes",
                    "0",
                    "--learning-evidence",
                    "--behavior-scenario",
                    "night_rest",
                    "--budget-profile",
                    "smoke",
                    "--learning-evidence-long-budget-profile",
                    "smoke",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            learning_evidence = payload["behavior_evaluation"]["learning_evidence"]
            self.assertEqual(
                learning_evidence["reference_condition"],
                "trained_without_reflex_support",
            )
            self.assertEqual(learning_evidence["budget_profile"], "smoke")
            self.assertEqual(learning_evidence["long_budget_profile"], "smoke")
            self.assertIn("trained_final", learning_evidence["conditions"])
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("learning_evidence_condition", header)
            self.assertIn("learning_evidence_policy_mode", header)
            self.assertIn("learning_evidence_train_episodes", header)
            self.assertIn("learning_evidence_frozen_after_episode", header)
            self.assertIn("learning_evidence_checkpoint_source", header)

    def test_cli_claim_test_suite_emits_summary_and_csv_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "claim_tests.csv"
            proc = subprocess.run(
                self._claim_suite_base_argv()
                + ["--full-summary", "--behavior-csv", str(csv_path)],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            claim_tests = payload["behavior_evaluation"]["claim_tests"]
            self.assertIn("claims", claim_tests)
            self.assertIn("summary", claim_tests)
            self.assertIn("metadata", claim_tests)
            self.assertIn(
                "learning_without_privileged_signals",
                claim_tests["claims"],
            )
            self.assertIn("claims_passed", claim_tests["summary"])
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("claim_test", header)
            self.assertIn("claim_test_status", header)
            self.assertIn("claim_test_passed", header)
            self.assertIn("claim_test_primary_metric", header)

    def test_cli_claim_test_suite_short_summary_is_condensed(self) -> None:
        """
        Verify that running the CLI claim-test-suite without requesting a full summary returns a condensed summary structure.
        
        The test invokes the CLI for the specified claim test and asserts the JSON output includes top-level claim-test summary fields:
        `claims`, `claims_passed`, `claims_failed`, and `all_primary_claims_passed`. It also asserts the per-claim entry for
        `learning_without_privileged_signals` is condensed to boolean `passed` and `skipped` keys.
        """
        proc = subprocess.run(
            self._claim_suite_base_argv(),
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(proc.stdout)
        claim_tests = payload["behavior_evaluation"]["claim_tests"]
        self.assertIn("claims", claim_tests)
        self.assertIn("claims_passed", claim_tests)
        self.assertIn("claims_failed", claim_tests)
        self.assertIn("claims_skipped", claim_tests)
        self.assertIn("all_primary_claims_passed", claim_tests)
        self.assertEqual(
            set(claim_tests["claims"]["learning_without_privileged_signals"].keys()),
            {"passed", "skipped"},
        )

    def test_cli_claim_test_without_suite_flag_runs_claim_suite(self) -> None:
        proc = subprocess.run(
            self._claim_suite_base_argv(include_suite_flag=False),
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "."},
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(proc.stdout)
        self.assertIn("claim_tests", payload["behavior_evaluation"])
        self.assertIn(
            "learning_without_privileged_signals",
            payload["behavior_evaluation"]["claim_tests"]["claims"],
        )

    def test_cli_curriculum_emits_summary_comparison_and_csv_columns(self) -> None:
        """
        Verifies the CLI emits a curriculum training summary, includes a curriculum comparison in the behavior evaluation, and writes a CSV with curriculum-related columns.
        
        Asserts that the produced JSON payload sets the training regime mode to "curriculum", contains a "curriculum" section, and that the behavior evaluation includes a "curriculum_comparison" with `curriculum_profile` equal to "ecological_v1". Also asserts a CSV file is created and its header contains the columns: "training_regime", "curriculum_profile", "curriculum_phase", "curriculum_skill", "curriculum_phase_status", and "curriculum_promotion_reason".
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "curriculum.csv"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--episodes",
                    "2",
                    "--eval-episodes",
                    "1",
                    "--behavior-scenario",
                    "night_rest",
                    "--budget-profile",
                    "smoke",
                    "--curriculum-profile",
                    "ecological_v1",
                    "--full-summary",
                    "--behavior-csv",
                    str(csv_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertEqual(payload["config"]["training_regime"]["mode"], "curriculum")
            self.assertIn("curriculum", payload)
            self.assertIn("curriculum_comparison", payload["behavior_evaluation"])
            self.assertEqual(
                payload["behavior_evaluation"]["curriculum_comparison"]["curriculum_profile"],
                "ecological_v1",
            )
            self.assertTrue(csv_path.exists())
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("training_regime", header)
            self.assertIn("curriculum_profile", header)
            self.assertIn("curriculum_phase", header)
            self.assertIn("curriculum_skill", header)
            self.assertIn("curriculum_phase_status", header)
            self.assertIn("curriculum_promotion_reason", header)

    def test_cli_noise_robustness_emits_summary_and_csv_columns(self) -> None:
        """
        Verify the CLI --noise-robustness option emits a JSON robustness matrix summary and a CSV with the expected noise-profile columns.
        
        Asserts that the JSON payload contains a `behavior_evaluation.robustness_matrix` with a `matrix_spec.cell_count` of 16 and includes `matrix`, `train_marginals`, `eval_marginals`, and `robustness_score`. Also asserts the emitted CSV file exists and its header contains `train_noise_profile`, `train_noise_profile_config`, `eval_noise_profile`, and `eval_noise_profile_config`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            for flag in ("--noise-robustness", "--behavior-noise-robustness"):
                with self.subTest(flag=flag):
                    csv_path = Path(tmpdir) / f"{flag.lstrip('-').replace('-', '_')}.csv"
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            "--full-summary",
                            "--behavior-csv",
                            str(csv_path),
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                        check=True,
                    )

                    payload = json.loads(proc.stdout)
                    self.assertEqual(
                        payload["config"]["training_regime"],
                        {"name": "noise_robustness"},
                    )
                    self.assertIn("summary", payload["behavior_evaluation"])
                    self.assertIn("suite", payload["behavior_evaluation"])
                    self.assertIn("legacy_scenarios", payload["behavior_evaluation"])
                    robustness = payload["behavior_evaluation"]["robustness_matrix"]
                    self.assertEqual(robustness["matrix_spec"]["cell_count"], 16)
                    self.assertIn("matrix", robustness)
                    self.assertIn("train_marginals", robustness)
                    self.assertIn("eval_marginals", robustness)
                    self.assertIn("robustness_score", robustness)
                    self.assertTrue(csv_path.exists())
                    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
                    self.assertIn("train_noise_profile", header)
                    self.assertIn("train_noise_profile_config", header)
                    self.assertIn("eval_noise_profile", header)
                    self.assertIn("eval_noise_profile_config", header)

    def test_cli_noise_robustness_short_summary_reports_score_and_dimensions(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                proc = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        flag,
                        "--behavior-scenario",
                        "night_rest",
                        "--budget-profile",
                        "smoke",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                    check=True,
                )

                payload = json.loads(proc.stdout)
                robustness = payload["behavior_evaluation"]["robustness_matrix"]
                self.assertIn("robustness_score", robustness)
                self.assertEqual(robustness["matrix_dimensions"], "4x4")
                self.assertEqual(payload["training_regime"], {"name": "noise_robustness"})

    def test_cli_noise_robustness_rejects_trace_output(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                with tempfile.TemporaryDirectory() as tmpdir:
                    trace_path = Path(tmpdir) / "trace.json"
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            "--trace",
                            str(trace_path),
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                    )

                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(
                        "--trace is not supported with --noise-robustness.",
                        proc.stderr,
                    )
                    self.assertFalse(trace_path.exists())

    def test_cli_noise_robustness_rejects_render_and_debug_flags(self) -> None:
        conflict_cases = (
            (["--render-eval"], "--render-eval"),
            (["--debug-trace"], "--debug-trace"),
        )
        for robustness_flag in ("--noise-robustness", "--behavior-noise-robustness"):
            for extra_args, rejected_flag in conflict_cases:
                with self.subTest(
                    robustness_flag=robustness_flag,
                    rejected_flag=rejected_flag,
                ):
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            robustness_flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            *extra_args,
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                    )

                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(
                        f"{rejected_flag} is not supported with --noise-robustness.",
                        proc.stderr,
                    )

    def test_cli_noise_robustness_rejects_curriculum_profile(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                proc = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        flag,
                        "--behavior-scenario",
                        "night_rest",
                        "--budget-profile",
                        "smoke",
                        "--curriculum-profile",
                        "ecological_v1",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                )

                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    "--curriculum-profile is not supported with --noise-robustness.",
                    proc.stderr,
                )

    def test_cli_noise_robustness_rejects_noise_profile(self) -> None:
        for flag in ("--noise-robustness", "--behavior-noise-robustness"):
            with self.subTest(flag=flag):
                proc = subprocess.run(  # noqa: S603
                    [
                        sys.executable,
                        "-m",
                        "spider_cortex_sim",
                        "--episodes",
                        "0",
                        "--eval-episodes",
                        "0",
                        flag,
                        "--behavior-scenario",
                        "night_rest",
                        "--budget-profile",
                        "smoke",
                        "--noise-profile",
                        "low",
                    ],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONPATH": "."},
                    text=True,
                    capture_output=True,
                )

                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(
                    "--noise-profile is not supported with --noise-robustness.",
                    proc.stderr,
                )

    def test_cli_noise_robustness_rejects_incompatible_workflow_flags(self) -> None:
        """
        Verifies the CLI rejects workflow flags that are incompatible with noise-robustness modes.
        
        This test runs the CLI with each of the noise-robustness entry points (`--noise-robustness`
        and `--behavior-noise-robustness`) combined with a set of mutually incompatible workflow
        flags (e.g., comparison, ablation, learning-evidence, claim-test-suite) and asserts the
        process exits with a non-zero status and emits an error message indicating the specific
        flag is not supported with noise-robustness.
        """
        conflict_cases = (
            (["--compare-profiles"], "--compare-profiles"),
            (["--compare-maps"], "--compare-maps"),
            (["--behavior-compare-profiles"], "--behavior-compare-profiles"),
            (["--behavior-compare-maps"], "--behavior-compare-maps"),
            (["--ablation-suite"], "--ablation-suite"),
            (["--ablation-variant", "monolithic_policy"], "--ablation-variant"),
            (["--learning-evidence"], "--learning-evidence"),
            (["--claim-test-suite"], "--claim-test-suite"),
            (
                ["--claim-test", "learning_without_privileged_signals"],
                "--claim-test-suite",
            ),
        )
        for robustness_flag in ("--noise-robustness", "--behavior-noise-robustness"):
            for extra_args, rejected_flag in conflict_cases:
                with self.subTest(
                    robustness_flag=robustness_flag,
                    rejected_flag=rejected_flag,
                ):
                    proc = subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "spider_cortex_sim",
                            "--episodes",
                            "0",
                            "--eval-episodes",
                            "0",
                            robustness_flag,
                            "--behavior-scenario",
                            "night_rest",
                            "--budget-profile",
                            "smoke",
                            *extra_args,
                        ],
                        cwd=Path(__file__).resolve().parents[1],
                        env={**os.environ, "PYTHONPATH": "."},
                        text=True,
                        capture_output=True,
                    )

                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn(
                        f"{rejected_flag} is not supported with --noise-robustness.",
                        proc.stderr,
                    )

    def test_offline_analysis_module_emits_report_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            behavior_csv = Path(tmpdir) / "behavior.csv"
            report_dir = Path(tmpdir) / "offline_report"
            summary_path = Path(tmpdir) / "summary.json"

            sim = SpiderSimulation(seed=11, max_steps=8)
            payload, _, rows = sim.evaluate_behavior_suite(["night_rest"])
            summary, _ = sim.train(
                episodes=0,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
            )
            summary["behavior_evaluation"] = payload
            save_behavior_csv(rows, behavior_csv)
            save_summary(summary, summary_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim.offline_analysis",
                    "--summary",
                    str(summary_path),
                    "--behavior-csv",
                    str(behavior_csv),
                    "--output-dir",
                    str(report_dir),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertIn("generated_files", payload)
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "report.json").exists())
            self.assertTrue((report_dir / "training_eval.svg").exists())
            self.assertTrue((report_dir / "scenario_success.svg").exists())
            self.assertTrue((report_dir / "robustness_matrix.svg").exists())
            self.assertTrue((report_dir / "scenario_checks.csv").exists())
            self.assertTrue((report_dir / "reward_components.csv").exists())
            report_md = (report_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("night_rest", report_md)

    def test_cli_does_not_synthesize_checkpointing_without_selection_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "summary.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spider_cortex_sim",
                    "--budget-profile",
                    "smoke",
                    "--checkpoint-selection",
                    "best",
                    "--summary",
                    str(summary_path),
                    "--full-summary",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "PYTHONPATH": "."},
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(proc.stdout)
            self.assertNotIn("checkpointing", payload)
            file_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertNotIn("checkpointing", file_payload)
