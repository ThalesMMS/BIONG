from __future__ import annotations

import unittest

import numpy as np

from spider_cortex_sim.interfaces import ACTION_TO_INDEX, MODULE_INTERFACE_BY_NAME, ModuleInterface
from spider_cortex_sim.modules import MODULE_HIDDEN_DIMS
from spider_cortex_sim.nn import ProposalNetwork


def _get_interface_for_module(module_name: str) -> ModuleInterface:
    return MODULE_INTERFACE_BY_NAME[module_name]


def _create_fresh_network(module_name: str, seed: int) -> ProposalNetwork:
    interface = _get_interface_for_module(module_name)
    return ProposalNetwork(
        input_dim=interface.input_dim,
        hidden_dim=MODULE_HIDDEN_DIMS[module_name],
        output_dim=len(ACTION_TO_INDEX),
        rng=np.random.default_rng(seed),
        name=f"{module_name}_test_network",
    )


def _assert_action_favored(
    test_case: unittest.TestCase,
    logits: np.ndarray,
    expected_actions: tuple[str, ...],
    margin: float = 0.5,
) -> None:
    expected_indices = tuple(ACTION_TO_INDEX[action_name] for action_name in expected_actions)
    unexpected_indices = [
        idx
        for idx in range(logits.shape[0])
        if idx not in expected_indices
    ]
    unexpected_max = float(np.max(logits[unexpected_indices]))
    for action_name, action_idx in zip(expected_actions, expected_indices, strict=True):
        test_case.assertGreaterEqual(
            float(logits[action_idx]) - unexpected_max,
            margin,
            f"Expected action {action_name} was not favored by margin {margin}.",
        )
