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

class SpiderTrainingCheckpointingTest(SpiderTrainingTestBase):
    def test_best_checkpoint_selection_restores_selected_checkpoint(self) -> None:
        """
        Verifies that selecting the "best" checkpoint during training persists, restores, and annotates the chosen checkpoint.
        
        Creates scripted checkpoint candidates with differing evaluation metrics, runs training with checkpoint selection set to "best", and asserts:
        - the training summary records the checkpointing selection, metric, and evaluation reflex scale,
        - the selected checkpoint is the one with the highest metric and its metadata (episode and eval_reflex_scale) is recorded,
        - restoring from the saved "best" and "last" checkpoints reproduces the expected model parameters (best matches the post-selection model, last differs),
        - behavior evaluations are annotated with checkpoint_source == "best".
        """
        sim = SpiderSimulation(seed=61, max_steps=12)

        def scripted_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names: Sequence[str],
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: Optional[float] = None,
        ) -> dict[str, object]:
            del scenario_names, metric, selection_scenario_episodes, eval_reflex_scale
            checkpoint_name = f"episode_{int(episode):05d}"
            checkpoint_path = root_dir / checkpoint_name
            sim.brain.save(checkpoint_path)
            score = 0.9 if int(episode) == 1 else 0.1
            return {
                "name": checkpoint_name,
                "episode": int(episode),
                "path": checkpoint_path,
                "metric": "scenario_success_rate",
                "scenario_success_rate": score,
                "episode_success_rate": score,
                "mean_reward": score,
            }

        sim._capture_checkpoint_candidate = scripted_capture_checkpoint  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as tmpdir:
            summary, _ = sim.train(
                2,
                evaluation_episodes=0,
                capture_evaluation_trace=False,
                checkpoint_selection="best",
                checkpoint_metric="scenario_success_rate",
                checkpoint_interval=1,
                checkpoint_dir=Path(tmpdir),
                checkpoint_scenario_names=["night_rest"],
                selection_scenario_episodes=1,
            )

            self.assertIn("checkpointing", summary)
            self.assertEqual(summary["checkpointing"]["selection"], "best")
            self.assertEqual(summary["checkpointing"]["metric"], "scenario_success_rate")
            self.assertEqual(summary["checkpointing"]["eval_reflex_scale"], 0.0)
            self.assertEqual(summary["checkpointing"]["selected_checkpoint"]["episode"], 1)
            self.assertEqual(
                summary["checkpointing"]["selected_checkpoint"]["eval_reflex_scale"],
                0.0,
            )

            best_sim = SpiderSimulation(seed=61, max_steps=12)
            best_sim.brain.load(Path(tmpdir) / "best")
            last_sim = SpiderSimulation(seed=61, max_steps=12)
            last_sim.brain.load(Path(tmpdir) / "last")

            np.testing.assert_allclose(sim.brain.motor_cortex.W1, best_sim.brain.motor_cortex.W1)
            self.assertFalse(
                np.allclose(sim.brain.motor_cortex.W1, last_sim.brain.motor_cortex.W1)
            )

            _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
            self.assertEqual(rows[0]["checkpoint_source"], "best")

    def test_direct_checkpoint_penalty_records_composite_scores(self) -> None:
        sim = SpiderSimulation(seed=62, max_steps=5)

        def scripted_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names: Sequence[str],
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: Optional[float] = None,
        ) -> dict[str, object]:
            """
            Persist a scripted checkpoint to disk and return a synthetic checkpoint-candidate metadata dictionary used by tests.
            
            Parameters:
                root_dir (Path): Directory where the checkpoint subdirectory will be created.
                episode (int): Episode index used to name the checkpoint (formatted as "episode_XXXXX").
                scenario_names: Ignored in this scripted implementation (kept for API compatibility).
                metric (str): Ignored; the returned candidate uses the fixed `"scenario_success_rate"` metric.
                selection_scenario_episodes (int): Ignored in this scripted implementation.
                eval_reflex_scale (float | None): Ignored in this scripted implementation.
            
            Returns:
                dict: A checkpoint-candidate mapping with these keys:
                    - name (str): Checkpoint directory name (e.g., "episode_00001").
                    - episode (int): Episode index.
                    - path (Path): Filesystem path to the saved checkpoint directory.
                    - metric (str): Primary metric name (`"scenario_success_rate"`).
                    - scenario_success_rate (float): Simulated scenario success rate (0.8 for episode 1, 0.7 otherwise).
                    - episode_success_rate (float): Simulated episode success rate (0.5).
                    - mean_reward (float): Simulated mean reward (0.25).
                    - evaluation_summary (dict): Nested evaluation info containing:
                        - mean_final_reflex_override_rate (float)
                        - mean_reflex_dominance (float)
            """
            del scenario_names, metric, selection_scenario_episodes, eval_reflex_scale
            checkpoint_name = f"episode_{int(episode):05d}"
            checkpoint_path = root_dir / checkpoint_name
            sim.brain.save(checkpoint_path)
            if int(episode) == 1:
                scenario_success_rate = 0.8
                override_rate = 0.9
            else:
                scenario_success_rate = 0.7
                override_rate = 0.0
            return {
                "name": checkpoint_name,
                "episode": int(episode),
                "path": checkpoint_path,
                "metric": "scenario_success_rate",
                "scenario_success_rate": scenario_success_rate,
                "episode_success_rate": 0.5,
                "mean_reward": 0.25,
                "evaluation_summary": {
                    "mean_final_reflex_override_rate": override_rate,
                    "mean_reflex_dominance": 0.0,
                },
            }

        sim._capture_checkpoint_candidate = scripted_capture_checkpoint  # type: ignore[method-assign]

        summary, _ = sim.train(
            2,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            checkpoint_selection="best",
            checkpoint_metric="scenario_success_rate",
            checkpoint_interval=1,
            checkpoint_scenario_names=["night_rest"],
            selection_scenario_episodes=1,
            checkpoint_override_penalty=1.0,
            checkpoint_penalty_mode="direct",
        )

        checkpointing = summary["checkpointing"]
        self.assertEqual(checkpointing["penalty_mode"], "direct")
        self.assertEqual(
            checkpointing["penalty_config"],
            {
                "metric": "scenario_success_rate",
                "override_penalty_weight": 1.0,
                "dominance_penalty_weight": 0.0,
                "penalty_mode": "direct",
            },
        )
        self.assertEqual(checkpointing["selected_checkpoint"]["episode"], 2)
        self.assertAlmostEqual(
            checkpointing["selected_checkpoint"]["composite_score"],
            0.7,
        )
        generated = checkpointing["generated_checkpoints"]
        self.assertTrue(all("composite_score" in candidate for candidate in generated))
        first_candidate = next(
            candidate for candidate in generated if candidate["episode"] == 1
        )
        self.assertAlmostEqual(first_candidate["composite_score"], -0.1)

    def test_best_checkpoint_selection_keeps_post_training_eval_offset(self) -> None:
        sim = SpiderSimulation(seed=71, max_steps=6)
        original_run_episode = sim.run_episode
        recorded_eval_indices: list[int] = []

        def wrapped_run_episode(
            episode_index: int,
            *,
            training: bool,
            sample: bool,
            render: bool = False,
            capture_trace: bool = False,
            scenario_name: str | None = None,
            debug_trace: bool = False,
            policy_mode: str = "normal",
        ):
            if not training and scenario_name is None:
                recorded_eval_indices.append(int(episode_index))
            return original_run_episode(
                episode_index,
                training=training,
                sample=sample,
                render=render,
                capture_trace=capture_trace,
                scenario_name=scenario_name,
                debug_trace=debug_trace,
                policy_mode=policy_mode,
            )

        def scripted_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names,
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: float | None = None,
        ):
            del scenario_names, metric, selection_scenario_episodes, eval_reflex_scale
            checkpoint_name = f"episode_{int(episode):05d}"
            checkpoint_path = root_dir / checkpoint_name
            sim.brain.save(checkpoint_path)
            return {
                "name": checkpoint_name,
                "episode": int(episode),
                "path": checkpoint_path,
                "metric": "scenario_success_rate",
                "scenario_success_rate": float(episode),
                "episode_success_rate": float(episode),
                "mean_reward": float(episode),
            }

        sim.run_episode = wrapped_run_episode  # type: ignore[method-assign]
        sim._capture_checkpoint_candidate = scripted_capture_checkpoint  # type: ignore[method-assign]

        sim.train(
            2,
            evaluation_episodes=2,
            capture_evaluation_trace=False,
            checkpoint_selection="best",
            checkpoint_interval=1,
            checkpoint_scenario_names=["night_rest"],
            selection_scenario_episodes=1,
        )

        self.assertEqual(recorded_eval_indices, [2, 3, 2, 3])

    def test_checkpoint_candidate_scoring_uses_no_reflex_scale(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        recorded_eval_scales: list[float | None] = []
        original_capture = sim._capture_checkpoint_candidate

        def wrapped_capture_checkpoint(
            *,
            root_dir: Path,
            episode: int,
            scenario_names,
            metric: str,
            selection_scenario_episodes: int,
            eval_reflex_scale: float | None = None,
        ):
            recorded_eval_scales.append(eval_reflex_scale)
            return original_capture(
                root_dir=root_dir,
                episode=episode,
                scenario_names=scenario_names,
                metric=metric,
                selection_scenario_episodes=selection_scenario_episodes,
                eval_reflex_scale=eval_reflex_scale,
            )

        sim._capture_checkpoint_candidate = wrapped_capture_checkpoint  # type: ignore[method-assign]

        sim.train(
            3,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
            checkpoint_selection="best",
            checkpoint_interval=1,
            checkpoint_scenario_names=["night_rest"],
            selection_scenario_episodes=1,
            reflex_anneal_final_scale=0.25,
        )

        self.assertTrue(recorded_eval_scales)
        for value in recorded_eval_scales:
            self.assertAlmostEqual(float(value), 0.0)

class CheckpointSourceAnnotationTest(unittest.TestCase):
    """Tests that checkpoint_source is properly annotated in behavior rows."""

    def test_evaluate_behavior_suite_rows_have_checkpoint_source(self) -> None:
        """
        Verify evaluate_behavior_suite produces non-empty rows annotated with checkpoint_source and noise_profile defaults.
        
        Asserts that the returned rows list is non-empty, that the first row contains a "checkpoint_source" field equal to "final", and that the "noise_profile" field equals "none".
        """
        sim = SpiderSimulation(seed=7, max_steps=10)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertIn("checkpoint_source", rows[0])
        self.assertEqual(rows[0]["checkpoint_source"], "final")
        self.assertEqual(rows[0]["noise_profile"], "none")

    def test_checkpoint_source_final_in_rows_after_regular_train(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        sim.train(episodes=2, evaluation_episodes=0, capture_evaluation_trace=False)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertEqual(rows[0]["checkpoint_source"], "final")

    def test_budget_profile_annotation_in_behavior_rows(self) -> None:
        sim = SpiderSimulation(
            seed=7,
            max_steps=10,
            budget_profile_name="smoke",
            benchmark_strength="quick",
        )
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertEqual(rows[0]["budget_profile"], "smoke")
        self.assertEqual(rows[0]["benchmark_strength"], "quick")

    def test_annotate_behavior_rows_includes_budget_fields(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10, budget_profile_name="dev", benchmark_strength="quick")
        raw_rows = [{"some_key": "some_val"}]
        annotated = sim._annotate_behavior_rows(raw_rows)
        self.assertTrue(annotated)
        self.assertIn("budget_profile", annotated[0])
        self.assertIn("benchmark_strength", annotated[0])
        self.assertIn("checkpoint_source", annotated[0])
        self.assertIn("ablation_variant", annotated[0])
        self.assertIn("ablation_architecture", annotated[0])
        self.assertIn("operational_profile", annotated[0])
        self.assertIn("noise_profile", annotated[0])
        self.assertIn("eval_noise_profile", annotated[0])

    def test_annotate_behavior_rows_merges_extra_metadata(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        raw_rows = [{"existing_key": "val"}]
        extra = {
            "learning_evidence_condition": "trained_final",
            "learning_evidence_policy_mode": "normal",
            "learning_evidence_train_episodes": 6,
        }
        annotated = sim._annotate_behavior_rows(raw_rows, extra_metadata=extra)
        self.assertEqual(len(annotated), 1)
        self.assertEqual(annotated[0]["learning_evidence_condition"], "trained_final")
        self.assertEqual(annotated[0]["learning_evidence_policy_mode"], "normal")
        self.assertEqual(annotated[0]["learning_evidence_train_episodes"], 6)
        # Original keys preserved
        self.assertEqual(annotated[0]["existing_key"], "val")

    def test_annotate_behavior_rows_extra_metadata_none_does_not_add_keys(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        raw_rows = [{"only_key": "v"}]
        annotated = sim._annotate_behavior_rows(raw_rows, extra_metadata=None)
        self.assertNotIn("learning_evidence_condition", annotated[0])

    def test_annotate_behavior_rows_extra_metadata_does_not_mutate_original(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10)
        original = {"k": "v"}
        annotated = sim._annotate_behavior_rows([original], extra_metadata={"extra": 1})
        self.assertNotIn("extra", original)
        self.assertIn("extra", annotated[0])

    def test_swap_eval_noise_profile_restores_original_world_noise(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10, noise_profile="low")
        self.assertEqual(sim.world.noise_profile.name, "low")

        with sim._swap_eval_noise_profile("high"):
            self.assertEqual(sim.world.noise_profile.name, "high")

        self.assertEqual(sim.world.noise_profile.name, "low")

        with self.assertRaises(RuntimeError):
            with sim._swap_eval_noise_profile("high"):
                self.assertEqual(sim.world.noise_profile.name, "high")
                raise RuntimeError("exercise context-manager cleanup")

        self.assertEqual(sim.world.noise_profile.name, "low")

    def test_annotate_behavior_rows_includes_train_and_eval_noise_fields(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=10, noise_profile="low")

        with sim._swap_eval_noise_profile("high"):
            annotated = sim._annotate_behavior_rows(
                [{"scenario": "night_rest"}],
                train_noise_profile="low",
            )

        self.assertEqual(annotated[0]["train_noise_profile"], "low")
        self.assertEqual(annotated[0]["eval_noise_profile"], "high")
        self.assertEqual(annotated[0]["noise_profile"], "high")
        self.assertIn("train_noise_profile_config", annotated[0])
        self.assertIn("eval_noise_profile_config", annotated[0])

    def test_consume_episodes_without_learning_returns_stats_list(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        history = sim._consume_episodes_without_learning(episodes=2, episode_start=0)
        self.assertEqual(len(history), 2)

    def test_consume_episodes_without_learning_zero_episodes_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        history = sim._consume_episodes_without_learning(episodes=0)
        self.assertEqual(history, [])

    def test_consume_episodes_without_learning_negative_episodes_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        history = sim._consume_episodes_without_learning(episodes=-5)
        self.assertEqual(history, [])

    def test_consume_episodes_without_learning_does_not_update_weights(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        weights_before = sim.brain.motor_cortex.W1.copy()
        sim._consume_episodes_without_learning(episodes=2, episode_start=0)
        np.testing.assert_array_equal(weights_before, sim.brain.motor_cortex.W1)

    def test_consume_episodes_without_learning_accepts_policy_mode_reflex_only(self) -> None:
        sim = SpiderSimulation(seed=5, max_steps=8)
        # reflex_only is only valid for modular architecture; default brain is modular
        history = sim._consume_episodes_without_learning(
            episodes=1, episode_start=0, policy_mode="reflex_only"
        )
        self.assertEqual(len(history), 1)
