from __future__ import annotations

from .shared import *


class BSeriesLegacyHarnessTest(unittest.TestCase):
    def test_legacy_harness_runs_short_training_smoke(self) -> None:
        sim = LegacyB0Simulation(max_steps=20, seed=5)
        summary, trace = sim.train(
            2,
            evaluation_episodes=1,
            capture_evaluation_trace=True,
        )

        self.assertEqual(summary["b_level"], 0)
        self.assertEqual(summary["b_mode"], "legacy_semantic")
        self.assertEqual(summary["semantic_actions"], list(B_SEMANTIC_ACTIONS))
        self.assertEqual(summary["evaluation"]["episodes"], 1)
        self.assertGreater(len(trace), 0)
