import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.distillation import DistillationConfig, DistillationLossConfig
from spider_cortex_sim.distillation.teacher import load_teacher_checkpoint
from spider_cortex_sim.nn import softmax
from spider_cortex_sim.nn_utils import kl_divergence
from spider_cortex_sim.simulation import SpiderSimulation


class DistillationEndToEndTest(unittest.TestCase):
    def test_valence_gradient_uses_distillation_temperature(self) -> None:
        sim = SpiderSimulation(seed=17, max_steps=2)
        brain = sim.brain
        arbitration_network = brain.arbitration_network
        self.assertIsNotNone(arbitration_network)
        assert arbitration_network is not None
        observation = sim.world.reset(seed=17)
        teacher_logits = np.array([1.5, -0.5, 0.25, 0.75], dtype=float)
        temperature = 2.0
        loss_weights = DistillationLossConfig(
            final_policy_weight=0.0,
            action_center_weight=0.0,
            valence_weight=0.25,
            proposal_weight=0.0,
        )
        captured: dict[str, object] = {}
        original_act = brain.act

        def capture_act(*args, **kwargs):
            decision = original_act(*args, **kwargs)
            captured["decision"] = decision
            return decision

        def capture_backward(**kwargs):
            captured["grad_valence_logits"] = np.asarray(
                kwargs["grad_valence_logits"],
                dtype=float,
            ).copy()
            return np.zeros(arbitration_network.input_dim, dtype=float)

        with (
            patch.object(brain, "act", side_effect=capture_act),
            patch.object(arbitration_network, "backward", side_effect=capture_backward),
        ):
            brain.distill_step(
                observation,
                teacher_policy=np.zeros(0, dtype=float),
                teacher_valence_logits=teacher_logits,
                temperature=temperature,
                loss_weights=loss_weights,
            )

        decision = captured["decision"]
        student_logits = np.array(
            [
                decision.arbitration_decision.valence_logits[name]
                for name in brain.VALENCE_ORDER
            ],
            dtype=float,
        )
        teacher_probs = softmax(teacher_logits, temperature=temperature)
        expected = loss_weights.valence_weight * (
            softmax(student_logits, temperature=temperature) - teacher_probs
        )
        temperature_one = loss_weights.valence_weight * (
            softmax(student_logits) - teacher_probs
        )

        np.testing.assert_allclose(captured["grad_valence_logits"], expected)
        self.assertFalse(
            np.allclose(captured["grad_valence_logits"], temperature_one)
        )

    def _teacher_checkpoint_path(self, architecture: str) -> str:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        checkpoint_dir = Path(tmpdir.name) / architecture
        teacher_config = BrainAblationConfig(
            name=f"{architecture}_teacher",
            architecture=architecture,
            module_dropout=0.0,
        )
        teacher_sim = SpiderSimulation(
            seed=7,
            max_steps=6,
            brain_config=teacher_config,
        )
        teacher_sim.train(
            episodes=1,
            evaluation_episodes=0,
            capture_evaluation_trace=False,
        )
        teacher_sim.brain.save(checkpoint_dir)
        return str(checkpoint_dir)

    def _mean_policy_kl(
        self,
        sim: SpiderSimulation,
        dataset,
    ) -> float:
        last_episode = None
        total_kl = 0.0
        sim.brain.reset_hidden_states()
        for sample in dataset:
            if last_episode is None or int(sample.episode) != last_episode:
                sim.brain.reset_hidden_states()
            decision = sim.brain.act(
                sample.observation,
                bus=None,
                sample=False,
                policy_mode="normal",
                training=False,
            )
            total_kl += kl_divergence(
                decision.total_logits,
                softmax(sample.teacher_total_logits),
            )
            last_episode = int(sample.episode)
        sample_count = max(1, len(dataset))
        return total_kl / sample_count

    def _assert_teacher_can_be_distilled(self, architecture: str) -> None:
        checkpoint_dir = self._teacher_checkpoint_path(architecture)
        teacher_brain, teacher_metadata = load_teacher_checkpoint(checkpoint_dir)
        student_sim = SpiderSimulation(seed=17, max_steps=6)
        dataset = student_sim.collect_teacher_rollout(
            teacher_brain,
            episodes=1,
            teacher_metadata=teacher_metadata,
            scenario_name="night_rest",
        )

        self.assertGreater(len(dataset), 0)
        pre_kl = self._mean_policy_kl(student_sim, dataset)
        distillation_config = DistillationConfig(
            teacher_checkpoint=checkpoint_dir,
            dataset_episodes=1,
            distillation_epochs=2,
            shuffle=False,
        )
        run_summary = student_sim._execute_distillation_phase(
            dataset=dataset,
            distillation_config=distillation_config,
        )
        post_kl = self._mean_policy_kl(student_sim, dataset)

        self.assertLess(post_kl, pre_kl)
        self.assertIn("history", run_summary)
        self.assertLessEqual(
            run_summary["final_epoch"]["mean_policy_kl"],
            run_summary["history"][0]["mean_policy_kl"],
        )

    def test_true_monolithic_teacher_distills_into_modular_student(self) -> None:
        self._assert_teacher_can_be_distilled("true_monolithic")

    def test_monolithic_teacher_distills_into_modular_student(self) -> None:
        self._assert_teacher_can_be_distilled("monolithic")
