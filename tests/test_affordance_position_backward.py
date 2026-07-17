import unittest

import numpy as np

from spider_cortex_sim.nn import (
    RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
)


class AffordancePositionBackwardTest(unittest.TestCase):
    def test_grad_x_matches_finite_difference_without_decoder_state(self) -> None:
        x = np.array([0.4, -0.7, 1.1], dtype=float)
        grad_policy = np.array([0.3, -0.2], dtype=float)
        epsilon = 1e-6

        for controller_state in (False, True):
            with self.subTest(option_action_controller_state=controller_state):
                network = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
                    input_dim=x.size,
                    hidden_dim=4,
                    output_dim=grad_policy.size,
                    rng=np.random.default_rng(215),
                    event_buffer_size=2,
                    option_ttl=4,
                    option_action_head=True,
                    option_decoder_state=False,
                    option_action_controller_state=controller_state,
                    name="gradient_check",
                )
                network.reset_hidden_state()
                network.forward(x, store_cache=True)
                analytic = network.backward(
                    grad_policy,
                    grad_value=0.0,
                    lr=0.0,
                )
                numeric = np.zeros_like(x)

                for index in range(x.size):
                    plus = x.copy()
                    minus = x.copy()
                    plus[index] += epsilon
                    minus[index] -= epsilon
                    network.reset_hidden_state()
                    plus_loss = float(
                        np.dot(network.forward(plus, store_cache=False)[0], grad_policy)
                    )
                    network.reset_hidden_state()
                    minus_loss = float(
                        np.dot(network.forward(minus, store_cache=False)[0], grad_policy)
                    )
                    numeric[index] = (plus_loss - minus_loss) / (2.0 * epsilon)

                np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
