import unittest

from spider_cortex_sim.gui.controller import GUIController
from spider_cortex_sim.simulation import SpiderSimulation


class GUIControllerTest(unittest.TestCase):
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
