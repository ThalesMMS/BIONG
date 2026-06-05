from __future__ import annotations

import unittest
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from spider_cortex_sim.brain.b_series_trace_fields import (
    B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES,
    B_SERIES_EPISODE_TRACE_FIELD_NAMES,
)
from spider_cortex_sim.brain.runtime_act_payload import _BrainRuntimeActPayloadMixin
from spider_cortex_sim.brain.types import BrainStep
from spider_cortex_sim.modules import ModuleResult
from spider_cortex_sim.simulation_episode_b_series_trace import (
    append_b_series_trace_fields,
)


class BSeriesTraceFieldRegistryTest(unittest.TestCase):
    def test_registry_covers_brain_step_direct_policy_trace_fields(self) -> None:
        step_field_names = tuple(field.name for field in fields(BrainStep))
        start = step_field_names.index("b_current_threat_pressure")
        end = step_field_names.index("b78_genetic_candidate") + 1

        self.assertEqual(
            B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES,
            step_field_names[start:end],
        )

    def test_runtime_payload_uses_registry_defaults_for_inactive_b_series(self) -> None:
        runtime = _RuntimePayloadHarness(is_b_series=False)

        payload = runtime._b_series_step_payload(
            {},
            np.array([1.0, 2.0], dtype=float),
            np.array([0.25, 0.75], dtype=float),
        )

        for name in B_SERIES_DIRECT_POLICY_PAYLOAD_FIELD_NAMES:
            self.assertIn(name, payload)
            self.assertIsNone(payload[name])
        self.assertEqual(payload["b_level"], -1)
        self.assertIsNone(payload["b_effective_level"])
        self.assertIsNone(payload["semantic_action"])
        self.assertEqual(payload["semantic_action_idx"], -1)
        self.assertEqual(payload["semantic_override_count"], 0)
        self.assertEqual(payload["blocked_mask"], {})
        self.assertEqual(payload["food_delta_used"], 0.0)
        self.assertEqual(payload["shelter_delta_used"], 0.0)

    def test_episode_trace_export_preserves_registered_keys(self) -> None:
        decision = BrainStep(
            module_results=[],
            action_center_logits=np.zeros(1, dtype=float),
            action_center_policy=np.ones(1, dtype=float),
            motor_correction_logits=np.zeros(1, dtype=float),
            b_level=6,
            b_effective_level="B6",
            b_mode="bridge",
            b_parent_level=5,
            b_current_threat_pressure=0.2,
            b6_controller_profile="fused_risk",
            b6_decision="return_to_shelter",
            semantic_action="MOVE_TO_SHELTER",
            learned_semantic_action="STAY",
            semantic_action_source="threat_guard",
            semantic_action_reason="threat_pressure",
            semantic_override_count=2,
            semantic_logits=np.array([0.1234567, 1.9876543], dtype=float),
            bridge_primitive_action="MOVE_LEFT",
            bridge_reason="toward_shelter",
            blocked_mask={"MOVE_RIGHT": True},
            food_delta_used=0.3333333,
            shelter_delta_used=-0.4444444,
            external_override_count=3,
        )
        item: dict[str, object] = {}

        append_b_series_trace_fields(item, decision)

        self.assertEqual(tuple(item), B_SERIES_EPISODE_TRACE_FIELD_NAMES)
        self.assertEqual(item["b_level"], 6)
        self.assertEqual(item["b_current_threat_pressure"], 0.2)
        self.assertEqual(item["b6_controller_profile"], "fused_risk")
        self.assertEqual(item["b6_decision"], "return_to_shelter")
        self.assertEqual(item["semantic_logits"], [0.123457, 1.987654])
        self.assertEqual(item["blocked_mask"], {"MOVE_RIGHT": True})
        self.assertEqual(item["food_delta_used"], 0.333333)
        self.assertEqual(item["shelter_delta_used"], -0.444444)
        self.assertEqual(item["external_override_count"], 3)


class _RuntimePayloadHarness(_BrainRuntimeActPayloadMixin):
    def __init__(self, *, is_b_series: bool) -> None:
        self.config = SimpleNamespace(
            is_b_series=is_b_series,
            b_level=6,
            b_mode="bridge",
        )
