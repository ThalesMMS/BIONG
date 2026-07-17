import unittest

import numpy as np

from spider_cortex_sim.nn import (
    RecurrentEventAttentionTrueMonolithicNetwork,
    RecurrentTrueMonolithicNetwork,
)


class RecurrentMonolithicValueOnlyTest(unittest.TestCase):
    def test_recurrent_value_only_preserves_hidden_state(self) -> None:
        network = RecurrentTrueMonolithicNetwork(
            input_dim=3,
            hidden_dim=4,
            output_dim=2,
            rng=np.random.default_rng(219),
        )
        hidden_state = np.array([0.1, -0.2, 0.3, -0.4], dtype=float)
        network.set_hidden_state(hidden_state)

        network.value_only(np.array([0.5, -0.25, 0.75], dtype=float))

        np.testing.assert_array_equal(network.get_hidden_state(), hidden_state)

    def test_event_attention_value_only_preserves_runtime_diagnostics(self) -> None:
        network = RecurrentEventAttentionTrueMonolithicNetwork(
            input_dim=3,
            hidden_dim=4,
            output_dim=2,
            rng=np.random.default_rng(220),
            event_buffer_size=2,
        )
        hidden_state = np.array([0.4, -0.3, 0.2, -0.1], dtype=float)
        attention_summary = {
            "event_attention_top_type": "REST_STARTED",
            "event_attention_top_age": 3,
            "event_attention_entropy": 0.25,
        }
        network.set_hidden_state(hidden_state)
        network.last_attention_summary = attention_summary.copy()

        network.value_only(np.array([-0.5, 0.25, 0.75], dtype=float))

        np.testing.assert_array_equal(network.get_hidden_state(), hidden_state)
        self.assertEqual(network.last_attention_summary, attention_summary)


if __name__ == "__main__":
    unittest.main()
