from __future__ import annotations

from .shared import *

from spider_cortex_sim.b_series import B0_CURRENT_BRIDGE_POLICY_NAME
from spider_cortex_sim.brain.b_series_progression import (
    find_b_series_semantic_progression_descriptor,
)


class BSeriesProgressionDescriptorTest(unittest.TestCase):
    def test_semantic_progression_descriptors_route_representative_b0_to_b6_modes(
        self,
    ) -> None:
        b0_descriptor = find_b_series_semantic_progression_descriptor(
            b_level=0,
            config_name=B0_CURRENT_BRIDGE_POLICY_NAME,
            b_mode="current_bridge",
        )
        self.assertIsNotNone(b0_descriptor)
        assert b0_descriptor is not None
        self.assertEqual(
            b0_descriptor.selector_name,
            "_b0_current_simple_semantic_action",
        )
        self.assertEqual(
            b0_descriptor.effective_level_for(
                B_CURRENT_BRIDGE_SELECTION_SOURCE,
                fallback="B0",
            ),
            B_CURRENT_BRIDGE_EFFECTIVE_LEVEL,
        )

        self.assertIsNone(
            find_b_series_semantic_progression_descriptor(
                b_level=1,
                config_name=B1_CAPACITY_H48_POLICY_NAME,
                b_mode="",
            )
        )

        b1_descriptor = find_b_series_semantic_progression_descriptor(
            b_level=1,
            config_name=B1_THREAT_GUARD_POLICY_NAME,
            b_mode="",
        )
        self.assertIsNotNone(b1_descriptor)
        assert b1_descriptor is not None
        self.assertEqual(
            b1_descriptor.selector_name,
            "_b1_threat_guard_semantic_action",
        )
        self.assertEqual(
            b1_descriptor.effective_level_for(
                B1_THREAT_GUARD_SELECTION_SOURCE,
                fallback="B1",
            ),
            B1_THREAT_GUARD_EFFECTIVE_LEVEL,
        )

        b2_descriptor = find_b_series_semantic_progression_descriptor(
            b_level=2,
            config_name=B2_TEMPORAL_THREAT_H56_POLICY_NAME,
            b_mode="",
        )
        self.assertIsNotNone(b2_descriptor)
        assert b2_descriptor is not None
        self.assertEqual(
            b2_descriptor.selector_name,
            "_b2_temporal_threat_semantic_action",
        )
        self.assertEqual(
            b2_descriptor.effective_level_for(
                B2_TEMPORAL_THREAT_SELECTION_SOURCE,
                fallback="B2",
            ),
            B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL,
        )

        b6_descriptor = find_b_series_semantic_progression_descriptor(
            b_level=6,
            config_name=B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
            b_mode="",
        )
        self.assertIsNotNone(b6_descriptor)
        assert b6_descriptor is not None
        self.assertEqual(
            b6_descriptor.selector_name,
            "_b6_risk_corridor_semantic_action",
        )
        self.assertEqual(
            b6_descriptor.effective_level_for(
                B6_RISK_CORRIDOR_SELECTION_SOURCE,
                fallback="B6",
            ),
            B6_RISK_CORRIDOR_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            b6_descriptor.effective_level_for(
                B6_RECURRENT_MEMORY_SELECTION_SOURCE,
                fallback="B6",
            ),
            B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            b6_descriptor.effective_level_for(
                B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE,
                fallback="B6",
            ),
            B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL,
        )
