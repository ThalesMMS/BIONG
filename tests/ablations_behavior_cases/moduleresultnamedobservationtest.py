from __future__ import annotations

from .shared import *


class ModuleResultNamedObservationTest(unittest.TestCase):
    """Tests for ModuleResult.named_observation (new None-guard added in this PR)."""

    def test_named_observation_returns_empty_dict_for_none_interface(self) -> None:
        result = ModuleResult(
            interface=None,
            name="monolithic_policy",
            observation_key="monolithic_policy",
            observation=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            logits=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            probs=np.full(
                len(LOCOMOTION_ACTIONS),
                1.0 / len(LOCOMOTION_ACTIONS),
                dtype=float,
            ),
            active=True,
        )
        self.assertEqual(result.named_observation(), {})

    def test_named_observation_works_with_valid_interface(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        obs = np.zeros(interface.input_dim, dtype=float)
        result = ModuleResult(
            interface=interface,
            name="alert_center",
            observation_key=interface.observation_key,
            observation=obs,
            logits=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            probs=np.full(
                len(LOCOMOTION_ACTIONS),
                1.0 / len(LOCOMOTION_ACTIONS),
                dtype=float,
            ),
            active=True,
        )
        named = result.named_observation()
        self.assertIsInstance(named, dict)
        self.assertIn("predator_visible", named)
