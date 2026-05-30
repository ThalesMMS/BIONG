from __future__ import annotations

from .shared import *


class SpiderSimulationBrainConfigTest(unittest.TestCase):
    """Tests for the brain_config parameter added to SpiderSimulation."""

    def test_default_brain_config_is_modular_full(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        self.assertEqual(sim.brain_config.name, "modular_full")
        self.assertEqual(sim.brain_config.architecture, "modular")

    def test_custom_brain_config_stored(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", architecture="modular", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.brain_config.name, "no_module_dropout")

    def test_module_dropout_comes_from_brain_config(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", architecture="modular", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.module_dropout, 0.0)

    def test_training_td_update_publishes_credit_diagnostics(self) -> None:
        """
        Runs a short training episode and asserts that temporal-difference update diagnostics containing module credit information are published to the simulation bus.
        
        Verifies at least one "td_update" message is present and that its payload includes:
        - "module_credit_weights" and "module_gradient_norms" keys,
        - "credit_strategy" equal to "route_mask",
        - "route_mask_enabled" set to True.
        """
        sim = SpiderSimulation(seed=7, max_steps=2)

        sim.run_episode(
            episode_index=0,
            training=True,
            sample=False,
        )

        td_updates = [
            message.payload
            for message in sim.bus.history()
            if message.topic == "td_update"
        ]
        self.assertGreater(len(td_updates), 0)
        payload = td_updates[0]
        self.assertIn("module_credit_weights", payload)
        self.assertIn("module_gradient_norms", payload)
        self.assertEqual(payload["credit_strategy"], "route_mask")
        self.assertTrue(payload["route_mask_enabled"])

    def test_true_monolithic_phase_replay_training_episode_applies_replay_passes(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_teacher_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        sim = SpiderSimulation(seed=7, max_steps=18, brain_config=config)

        sim.run_episode(
            episode_index=0,
            training=True,
            sample=False,
            scenario_name="continuous_survival_post_rest_inside_v1",
        )

        td_updates = [
            message.payload
            for message in sim.bus.history()
            if message.topic == "td_update"
        ]
        self.assertGreater(len(td_updates), 0)
        self.assertTrue(
            any(
                int(payload.get("continuation_replay_passes_applied", 0)) > 0
                for payload in td_updates
            )
        )

    def test_brain_config_propagated_to_brain(self) -> None:
        config = BrainAblationConfig(
            name="drop_alert_center",
            architecture="modular",
            module_dropout=0.0,
            disabled_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.brain.config.name, "drop_alert_center")
        self.assertIn("alert_center", sim.brain.config.disabled_modules)

    def test_annotate_behavior_rows_uses_effective_eval_reflex_scale(self) -> None:
        config = BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            reflex_scale=0.0,
        )
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)

        rows = sim._annotate_behavior_rows([{"scenario": "night_rest"}], eval_reflex_scale=1.0)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["eval_reflex_scale"]), 0.0)
