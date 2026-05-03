import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spider_cortex_sim.gui import SpiderGUI
from spider_cortex_sim.gui.controller import GUIController
from spider_cortex_sim.gui.constants import BOTTOM_BAR_HEIGHT, TOP_BAR_HEIGHT
from spider_cortex_sim.simulation import SpiderSimulation


class GUIControllerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
