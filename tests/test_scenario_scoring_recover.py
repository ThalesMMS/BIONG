from __future__ import annotations

import unittest

from spider_cortex_sim.scenarios import get_scenario

from tests.fixtures.scenario_trace_builders import _make_episode_stats


class RecoverAfterFailedChaseScoringTest(unittest.TestCase):
    def test_wait_before_recover_does_not_satisfy_return_check(self) -> None:
        spec = get_scenario("recover_after_failed_chase")
        stats = _make_episode_stats(scenario="recover_after_failed_chase", alive=True)
        trace = [
            {"state": {"lizard_mode": "WAIT"}},
            {"state": {"lizard_mode": "RECOVER"}},
        ]

        score = spec.score_episode(stats, trace)

        self.assertTrue(score.checks["predator_enters_recover"].passed)
        self.assertFalse(score.checks["predator_returns_to_wait"].passed)


if __name__ == "__main__":
    unittest.main()
