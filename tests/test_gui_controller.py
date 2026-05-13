import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spider_cortex_sim.b_series_legacy import LegacyB0Simulation
from spider_cortex_sim.gui import SpiderGUI
from spider_cortex_sim.gui.controller import GUIController
from spider_cortex_sim.gui.constants import BOTTOM_BAR_HEIGHT, TOP_BAR_HEIGHT
from spider_cortex_sim.gui.models import GUI_MODEL_SPECS_BY_ID, GUIRunConfig
from spider_cortex_sim.simulation import SpiderSimulation
from spider_cortex_sim.world import ACTIONS


class GUIControllerTest(unittest.TestCase):
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
