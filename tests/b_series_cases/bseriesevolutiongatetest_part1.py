from __future__ import annotations

from .shared import *
from .bseriesevolutiongatetest_helpers import BSeriesEvolutionGateTestHelpers



class BSeriesEvolutionGateTestPart1(BSeriesEvolutionGateTestHelpers, unittest.TestCase):
    def test_b4_canonical_gate_accepts_retention_without_forcing_improvement(
        self,
    ) -> None:
        results = [
            self._b4_result(
                0,
                steps=300,
                alive=True,
                food=14,
                sleep=54,
                shelter=10,
                contacts=2,
            ),
            self._b4_result(
                1,
                steps=300,
                alive=True,
                food=15,
                sleep=46,
                shelter=11,
                contacts=4,
            ),
            self._b4_result(
                2,
                steps=214,
                alive=False,
                food=8,
                sleep=37,
                shelter=8,
                contacts=1,
            ),
            self._b4_result(
                3,
                steps=244,
                alive=False,
                food=9,
                sleep=40,
                shelter=12,
                contacts=3,
            ),
            self._b4_result(
                4,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                5,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                6,
                steps=215,
                alive=False,
                food=8,
                sleep=37,
                shelter=7,
                contacts=1,
            ),
            self._b4_result(
                7,
                steps=300,
                alive=True,
                food=12,
                sleep=38,
                shelter=5,
                contacts=4,
            ),
            self._b4_result(
                8,
                steps=49,
                alive=False,
                food=2,
                sleep=0,
                shelter=2,
                contacts=1,
            ),
            self._b4_result(
                9,
                steps=300,
                alive=True,
                food=13,
                sleep=54,
                shelter=11,
                contacts=4,
            ),
        ]

        gate = b4_canonical_multi_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        aggregate = gate["aggregate"]
        self.assertEqual(aggregate["completed_horizons"], 4)
        self.assertEqual(aggregate["min_steps"], 49)
        self.assertEqual(aggregate["total_predator_contacts"], 24)
        self.assertEqual(aggregate["food_cycle_episodes"], 9)
        self.assertEqual(aggregate["sleep_cycle_episodes"], 9)
        self.assertEqual(aggregate["shelter_cycle_episodes"], 10)

    def test_b4_canonical_gate_rejects_b3_anchor_regression(self) -> None:
        results = [
            self._b4_result(
                0,
                steps=168,
                alive=False,
                food=8,
                sleep=21,
                shelter=2,
                contacts=0,
            ),
            self._b4_result(
                1,
                steps=300,
                alive=True,
                food=14,
                sleep=93,
                shelter=7,
                contacts=3,
            ),
            self._b4_result(
                2,
                steps=300,
                alive=True,
                food=8,
                sleep=37,
                shelter=8,
                contacts=1,
            ),
            self._b4_result(
                3,
                steps=300,
                alive=True,
                food=9,
                sleep=40,
                shelter=12,
                contacts=3,
            ),
            self._b4_result(
                4,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                5,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                6,
                steps=215,
                alive=False,
                food=8,
                sleep=37,
                shelter=7,
                contacts=1,
            ),
            self._b4_result(
                7,
                steps=300,
                alive=True,
                food=12,
                sleep=38,
                shelter=5,
                contacts=4,
            ),
            self._b4_result(
                8,
                steps=49,
                alive=False,
                food=2,
                sleep=0,
                shelter=2,
                contacts=1,
            ),
            self._b4_result(
                9,
                steps=300,
                alive=True,
                food=13,
                sleep=54,
                shelter=11,
                contacts=4,
            ),
        ]

        gate = b4_canonical_multi_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn("canonical_ep0:b3_anchor_completed_horizon", gate["failures"])
