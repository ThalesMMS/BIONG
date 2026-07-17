import unittest

import numpy as np

from spider_cortex_sim.nn import (
    RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
)


class AffordancePositionMetricsTest(unittest.TestCase):
    def _network(
        self,
        *,
        transition_rollout: bool,
    ) -> RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork:
        return RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=3,
            hidden_dim=4,
            output_dim=2,
            rng=np.random.default_rng(217),
            transition_rollout_prediction_head=transition_rollout,
            transition_rollout_prediction_feedback=transition_rollout,
            name="metrics_check",
        )

    def test_count_parameters_includes_transition_rollout_arrays(self) -> None:
        enabled = self._network(transition_rollout=True)
        disabled = self._network(transition_rollout=False)
        rollout_arrays = (
            enabled.W2_transition_rollout_prediction,
            enabled.b2_transition_rollout_prediction,
            enabled.W_transition_rollout_prediction_feedback,
            enabled.b_transition_rollout_prediction_feedback,
        )

        self.assertEqual(
            enabled.count_parameters() - disabled.count_parameters(),
            sum(array.size for array in rollout_arrays),
        )

    def test_parameter_norm_includes_shelter_column_and_rollout_arrays(self) -> None:
        enabled = self._network(transition_rollout=True)
        zeroed = self._network(transition_rollout=True)
        parameter_names = (
            "W2_shelter_column",
            "b2_shelter_column",
            "W2_transition_rollout_prediction",
            "b2_transition_rollout_prediction",
            "W_transition_rollout_prediction_feedback",
            "b_transition_rollout_prediction_feedback",
        )
        expected_squared_delta = sum(
            float(np.sum(getattr(enabled, name) ** 2))
            for name in parameter_names
        )
        for name in parameter_names:
            getattr(zeroed, name).fill(0.0)

        actual_squared_delta = (
            enabled.parameter_norm() ** 2 - zeroed.parameter_norm() ** 2
        )
        self.assertAlmostEqual(
            actual_squared_delta,
            expected_squared_delta,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
