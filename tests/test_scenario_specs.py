from __future__ import annotations

import unittest

from spider_cortex_sim.metrics import BehavioralEpisodeScore
from spider_cortex_sim.scenarios.specs import ScenarioSpec


def _score_episode(stats, trace):
    return BehavioralEpisodeScore(
        episode=0,
        seed=0,
        scenario="scenario",
        objective="objective",
        success=True,
        checks={},
        behavior_metrics={},
        failures=[],
    )


def _scenario_spec(**overrides) -> ScenarioSpec:
    fields = {
        "name": "scenario",
        "description": "description",
        "objective": "objective",
        "behavior_checks": (),
        "diagnostic_focus": "focus",
        "success_interpretation": "success",
        "failure_interpretation": "failure",
        "budget_note": "budget",
        "max_steps": 1,
        "map_template": "central_burrow",
        "setup": lambda world: None,
        "score_episode": _score_episode,
    }
    fields.update(overrides)
    return ScenarioSpec(**fields)


class ScenarioSpecExpectedOwnersTest(unittest.TestCase):
    def test_expected_owner_modules_are_normalized(self) -> None:
        spec = _scenario_spec(
            expected_owner_modules=(" sleep_center ", "visual.cortex")
        )

        self.assertEqual(
            spec.expected_owner_modules,
            ("sleep_center", "visual.cortex"),
        )

    def test_expected_owner_modules_rejects_bare_string(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a string"):
            _scenario_spec(expected_owner_modules="sleep_center")

    def test_expected_owner_modules_rejects_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "not bytes"):
            _scenario_spec(expected_owner_modules=b"sleep_center")

    def test_expected_owner_modules_rejects_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence of module names"):
            _scenario_spec(expected_owner_modules=None)

    def test_expected_owner_modules_rejects_non_iterable(self) -> None:
        with self.assertRaisesRegex(ValueError, "iterable sequence"):
            _scenario_spec(expected_owner_modules=123)

    def test_expected_owner_modules_rejects_invalid_entries(self) -> None:
        for value in ("", "   ", object(), "1bad", "bad-name"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "expected_owner_modules"):
                    _scenario_spec(expected_owner_modules=(value,))

    def test_expected_owner_modules_rejects_duplicates(self) -> None:
        for value in (
            ("sleep_center", "sleep_center"),
            ("sleep_center", " sleep_center "),
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "duplicate module"):
                    _scenario_spec(expected_owner_modules=value)


if __name__ == "__main__":
    unittest.main()
