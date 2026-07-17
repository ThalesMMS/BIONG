import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spider_cortex_sim.b_series_legacy import LegacyB0Simulation
from spider_cortex_sim.gui import SpiderGUI
from spider_cortex_sim.gui.controller import GUIController
from spider_cortex_sim.gui.constants import BOTTOM_BAR_HEIGHT, TOP_BAR_HEIGHT
from spider_cortex_sim.gui.models import (
    GUI_MODEL_SPECS_BY_ID,
    GUIRunConfig,
    build_runtime_adapter_for_checkpoint,
    discover_gui_checkpoints,
)
from spider_cortex_sim.simulation import SpiderSimulation
from spider_cortex_sim.world import ACTIONS


class GUIControllerTest(unittest.TestCase):
    def _write_minimal_checkpoint_metadata(
        self,
        directory: Path,
        *,
        name: str = "test_checkpoint",
        architecture: str = "modular",
    ) -> None:
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "architecture_fingerprint": "fake-fingerprint",
                    "ablation_config": {
                        "name": name,
                        "architecture": architecture,
                    },
                    "total_parameters": 3,
                    "modules": {
                        "visual_cortex": {
                            "type": "proposal",
                        }
                    },
                }
            )
        )
        (directory / "visual_cortex.npz").write_bytes(b"not-a-real-npz")

    def test_discover_gui_checkpoints_reads_metadata_and_ignores_incomplete_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "runs" / "candidate" / "best"
            self._write_minimal_checkpoint_metadata(valid, name="candidate_a")
            invalid = root / "runs" / "missing_weights" / "best"
            invalid.mkdir(parents=True)
            (invalid / "metadata.json").write_text(
                json.dumps({"ablation_config": {"name": "missing_weights"}})
            )

            specs = discover_gui_checkpoints([root])

        self.assertEqual([spec.variant for spec in specs], ["candidate_a"])
        self.assertEqual(specs[0].path, valid)
        self.assertEqual(specs[0].architecture, "modular")
        self.assertEqual(specs[0].total_parameters, 3)

    def test_build_runtime_adapter_for_checkpoint_reconstructs_b78_config(self) -> None:
        checkpoint = Path(
            "artifacts/b_series/evolution/"
            "b78_vestibular_balance_h48_bridge_policy/seed_7/best"
        )
        specs = discover_gui_checkpoints([checkpoint])

        runtime = build_runtime_adapter_for_checkpoint(
            GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=7),
            specs[0],
        )

        self.assertEqual(runtime.brain.config.name, "b78_vestibular_balance_h48_bridge_policy")
        self.assertEqual(runtime.brain.config.architecture, "b_series")
        self.assertEqual(runtime.brain.config.b_level, 78)
        self.assertEqual(runtime.brain.load(checkpoint), ["b_series_policy"])

    def test_load_selected_checkpoint_evaluate_mode_does_not_learn(self) -> None:
        checkpoint = Path(
            "artifacts/b_series/evolution/"
            "b78_vestibular_balance_h48_bridge_policy/seed_7/best"
        )
        controller = GUIController(
            run_config=GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=7)
        )
        controller.available_checkpoint_specs = discover_gui_checkpoints([checkpoint])
        controller.selected_checkpoint_index = 0
        controller.configure_run(train_episodes=1, eval_episodes=1)

        self.assertTrue(controller.load_selected_checkpoint(mode="evaluate"))
        seen_learn_calls: list[bool] = []
        controller.brain.learn = lambda *args, **kwargs: seen_learn_calls.append(True)

        controller._do_step()

        self.assertEqual(controller.brain.config.b_level, 78)
        self.assertEqual(controller.phase, "evaluation")
        self.assertEqual(seen_learn_calls, [])

    def test_load_selected_checkpoint_train_mode_preserves_learning_flow(self) -> None:
        checkpoint = Path(
            "artifacts/b_series/evolution/"
            "b78_vestibular_balance_h48_bridge_policy/seed_7/best"
        )
        controller = GUIController(
            run_config=GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=7)
        )
        controller.available_checkpoint_specs = discover_gui_checkpoints([checkpoint])
        controller.selected_checkpoint_index = 0
        controller.configure_run(train_episodes=1, eval_episodes=1)

        self.assertTrue(controller.load_selected_checkpoint(mode="train"))
        seen_learn_calls: list[bool] = []
        controller.brain.learn = lambda *args, **kwargs: seen_learn_calls.append(True)

        controller._do_step()

        self.assertEqual(controller.brain.config.b_level, 78)
        self.assertEqual(controller.phase, "training")
        self.assertEqual(seen_learn_calls, [True])

    def test_legacy_load_fallback_honors_evaluate_mode(self) -> None:
        controller = GUIController(
            run_config=GUIRunConfig(width=5, height=5, food_count=1, max_steps=1, seed=7)
        )
        controller.configure_run(train_episodes=1, eval_episodes=0)
        controller.training_rewards.append(3.0)
        seen_learn_calls: list[bool] = []
        controller.brain.learn = lambda *args, **kwargs: seen_learn_calls.append(True)

        with (
            patch(
                "spider_cortex_sim.gui.controller.discover_gui_checkpoints",
                return_value=[],
            ),
            patch.object(controller.brain, "load", return_value=["visual_cortex"]),
        ):
            loaded = controller.load_brain("legacy_brain", mode="evaluate")

        self.assertTrue(loaded)
        self.assertEqual(controller.checkpoint_load_mode, "evaluate")
        self.assertEqual(controller.phase, "evaluation")
        self.assertEqual(controller.current_episode, 0)
        self.assertEqual(controller.training_rewards, [])
        self.assertTrue(controller.paused)
        self.assertGreaterEqual(controller.total_eval_episodes, 1)

        controller._do_step()
        self.assertEqual(seen_learn_calls, [])

    def test_load_selected_checkpoint_reports_incompatible_load_without_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "bad" / "best"
            self._write_minimal_checkpoint_metadata(
                checkpoint,
                name="bad_true_monolithic",
                architecture="true_monolithic",
            )

            controller = GUIController(
                run_config=GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=7)
            )
            original_brain = controller.brain
            controller.available_checkpoint_specs = discover_gui_checkpoints([checkpoint])
            controller.selected_checkpoint_index = 0

            self.assertFalse(controller.load_selected_checkpoint(mode="evaluate"))

        self.assertIs(controller.brain, original_brain)
        self.assertTrue(controller.toast_is_error)
        self.assertIn("Load error", controller.toast_text)

    def test_gui_model_registry_exposes_a0_and_b0_variants(self) -> None:
        self.assertIn("a0_true_monolithic", GUI_MODEL_SPECS_BY_ID)
        self.assertIn("a0_owned_option", GUI_MODEL_SPECS_BY_ID)
        self.assertIn("b0_current_bridge", GUI_MODEL_SPECS_BY_ID)
        self.assertIn("b0_legacy_semantic", GUI_MODEL_SPECS_BY_ID)

        self.assertEqual(
            GUI_MODEL_SPECS_BY_ID["b0_current_bridge"].runtime_kind,
            "current_world",
        )
        self.assertTrue(
            GUI_MODEL_SPECS_BY_ID[
                "b0_current_bridge"
            ].evolution_transfer_compatible
        )
        self.assertEqual(
            GUI_MODEL_SPECS_BY_ID["b0_legacy_semantic"].runtime_kind,
            "legacy_b0",
        )
        self.assertFalse(
            GUI_MODEL_SPECS_BY_ID[
                "b0_legacy_semantic"
            ].evolution_transfer_compatible
        )

    def test_rebuild_buttons_updates_resize_dependent_references(self) -> None:
        class FakeRect:
            def __init__(self, x, y, w, h) -> None:
                self.x = x
                self.y = y
                self.w = w
                self.h = h

        gui = SpiderGUI.__new__(SpiderGUI)
        gui.controller = SimpleNamespace(
            world=SimpleNamespace(height=10),
            cell_size=20,
        )
        gui.win_w = 360
        old_buttons = {"pause": object()}
        gui.buttons = old_buttons
        gui.renderer = SimpleNamespace(buttons=old_buttons, btn_pause=old_buttons["pause"])

        with patch("spider_cortex_sim.gui.widgets.pygame", SimpleNamespace(Rect=FakeRect)):
            gui._rebuild_buttons()

        self.assertIs(gui.events.buttons, gui.buttons)
        self.assertIs(gui.renderer.buttons, gui.buttons)
        self.assertIs(gui.renderer.btn_pause, gui.buttons["pause"])
        self.assertEqual(gui.buttons["pause"].rect.y, TOP_BAR_HEIGHT + 10 * 20 + 4)
        self.assertLessEqual(
            gui.buttons["load"].rect.x + gui.buttons["load"].rect.w,
            gui.win_w,
        )

        gui.win_w = 5
        with patch("spider_cortex_sim.gui.widgets.pygame", SimpleNamespace(Rect=FakeRect)):
            gui._rebuild_buttons()

        self.assertEqual(gui.buttons["pause"].rect.x, 0)
        self.assertLessEqual(
            gui.buttons["load"].rect.x + gui.buttons["load"].rect.w,
            gui.win_w,
        )

    def test_initial_window_height_includes_bottom_bar(self) -> None:
        sim = SpiderSimulation(width=5, height=6, food_count=1, max_steps=1, seed=3)
        controller = GUIController(sim)

        expected = TOP_BAR_HEIGHT + (sim.world.height * controller.cell_size) + BOTTOM_BAR_HEIGHT
        self.assertEqual(controller.win_h, expected)

    def test_layout_docks_left_sidebar_before_grid_and_right_panel(self) -> None:
        sim = SpiderSimulation(width=5, height=6, food_count=1, max_steps=1, seed=3)
        controller = GUIController(sim)
        controller.apply_window_size(controller.win_w, controller.win_h)

        grid_w = sim.world.width * controller.cell_size
        self.assertEqual(controller.grid_offset_x, controller.left_sidebar_width)
        self.assertEqual(controller.panel_x, controller.left_sidebar_width + grid_w)

    def test_scroll_panel_uses_current_cell_size(self) -> None:
        sim = SpiderSimulation(width=5, height=10, food_count=1, max_steps=1, seed=3)
        controller = GUIController(sim)
        controller.cell_size = 20
        controller.panel_content_height = 500

        controller.scroll_panel(999)

        self.assertEqual(controller.panel_scroll, 300)

    def test_training_step_marks_time_limit_as_terminal_for_learning(self) -> None:
        sim = SpiderSimulation(width=5, height=5, food_count=1, max_steps=1, seed=3)
        controller = GUIController(sim)
        controller.configure_run(train_episodes=1, eval_episodes=0)
        seen_done: list[bool] = []

        def record_learn(decision, reward, next_observation, done):
            seen_done.append(done)
            return {}

        sim.brain.learn = record_learn

        controller._do_step()

        self.assertEqual(seen_done, [True])
        self.assertTrue(controller.episode_done)

    def test_zero_training_budget_starts_evaluation_without_learning(self) -> None:
        sim = SpiderSimulation(width=5, height=5, food_count=1, max_steps=1, seed=3)
        controller = GUIController(sim)
        seen_learn_calls: list[bool] = []
        sim.brain.learn = lambda *args, **kwargs: seen_learn_calls.append(True)

        controller.configure_run(train_episodes=0, eval_episodes=3)
        controller._do_step()

        self.assertEqual(controller.phase, "evaluation")
        self.assertEqual(controller.current_episode, 0)
        self.assertEqual(seen_learn_calls, [])

    def test_zero_evaluation_budget_finishes_after_last_training_episode(self) -> None:
        sim = SpiderSimulation(width=5, height=5, food_count=1, max_steps=1, seed=3)
        controller = GUIController(sim)
        controller.configure_run(train_episodes=1, eval_episodes=0)
        controller.episode_done = True

        with patch.object(controller, "_start_episode", wraps=controller._start_episode) as start:
            controller._advance_episode()

        self.assertEqual(controller.phase, "done")
        self.assertTrue(controller.paused)
        start.assert_not_called()

    def test_zero_training_and_evaluation_budgets_start_done(self) -> None:
        sim = SpiderSimulation(width=5, height=5, food_count=1, max_steps=1, seed=3)
        controller = GUIController(sim)

        with patch.object(controller, "_start_episode", wraps=controller._start_episode) as start:
            controller.configure_run(train_episodes=0, eval_episodes=0)

        self.assertEqual(controller.phase, "done")
        self.assertTrue(controller.paused)
        self.assertIsNone(controller.observation)
        start.assert_not_called()

    def test_model_switch_recreates_runtime_and_can_select_legacy_b0(self) -> None:
        run_config = GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=3)
        controller = GUIController(run_config=run_config)
        controller.configure_run(train_episodes=1, eval_episodes=0)
        original_sim = controller.sim

        controller.apply_model("b0_legacy_semantic")

        self.assertIsInstance(controller.sim, LegacyB0Simulation)
        self.assertIsNot(controller.sim, original_sim)
        self.assertEqual(controller.active_model.id, "b0_legacy_semantic")
        self.assertEqual(controller.phase, "training")
        self.assertEqual(controller.current_episode, 0)
        self.assertEqual(controller.current_step, 0)
        self.assertIsNotNone(controller.observation)

    def test_b0_current_bridge_world_step_receives_only_primitive_action(self) -> None:
        run_config = GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=3)
        controller = GUIController(run_config=run_config, model_id="b0_current_bridge")
        controller.configure_run(train_episodes=1, eval_episodes=0)
        seen_action_indexes: list[int] = []
        original_step = controller.world.step

        def record_step(action_idx: int):
            seen_action_indexes.append(action_idx)
            self.assertIsInstance(action_idx, int)
            self.assertGreaterEqual(action_idx, 0)
            self.assertLess(action_idx, len(ACTIONS))
            return original_step(action_idx)

        controller.world.step = record_step

        controller._do_step()

        self.assertEqual(len(seen_action_indexes), 1)
        self.assertIsNotNone(controller.last_decision.semantic_action)
        self.assertIn(controller.last_decision.bridge_primitive_action, ACTIONS)

    def test_b0_current_evolution_snapshot_records_transfer_metadata(self) -> None:
        run_config = GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=3)
        controller = GUIController(run_config=run_config, model_id="b0_current_bridge")
        controller.configure_run(train_episodes=1, eval_episodes=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = controller.save_evolution_snapshot(tmp)
            self.assertIsNotNone(path)
            payload = json.loads((path / "gui_snapshot.json").read_text())

        self.assertEqual(payload["evolution"]["process_name"], "Evolution")
        self.assertEqual(payload["evolution"]["source_model_id"], "b0_current_bridge")
        self.assertTrue(payload["evolution"]["transfer_compatible"])
        self.assertEqual(payload["ablation_config"]["b_level"], 0)
        self.assertEqual(payload["ablation_config"]["b_mode"], "current_bridge")

    def test_b0_legacy_evolution_snapshot_marks_non_transferable(self) -> None:
        run_config = GUIRunConfig(width=5, height=5, food_count=1, max_steps=2, seed=3)
        controller = GUIController(run_config=run_config, model_id="b0_legacy_semantic")
        controller.configure_run(train_episodes=1, eval_episodes=0)
        controller._do_step()

        with tempfile.TemporaryDirectory() as tmp:
            path = controller.save_evolution_snapshot(tmp)
            self.assertIsNotNone(path)
            payload = json.loads((path / "gui_snapshot.json").read_text())
            self.assertTrue((path / "legacy_b0_weights.npz").exists())
            self.assertTrue((path / "legacy_b0_metadata.json").exists())

        self.assertEqual(payload["evolution"]["source_model_id"], "b0_legacy_semantic")
        self.assertFalse(payload["evolution"]["transfer_compatible"])
        self.assertEqual(payload["legacy_checkpoint"], "legacy_b0_weights.npz")


if __name__ == "__main__":
    unittest.main()
