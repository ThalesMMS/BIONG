from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..b_series import (
    B_CURRENT_BRIDGE_EFFECTIVE_LEVEL,
    B1_THREAT_GUARD_EFFECTIVE_LEVEL,
    B1_THREAT_GUARD_POLICY_NAME,
    B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL,
    B2_TEMPORAL_THREAT_H48_POLICY_NAME,
    B2_TEMPORAL_THREAT_H56_POLICY_NAME,
    B2_TEMPORAL_THREAT_H64_POLICY_NAME,
    B3_CONTACT_MEMORY_EFFECTIVE_LEVEL,
    B3_CONTACT_MEMORY_H48_POLICY_NAME,
    B3_CONTACT_MEMORY_H56_POLICY_NAME,
    B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
    B3_RECURRENT_GUARD_EFFECTIVE_LEVEL,
    B3_RECURRENT_GUARD_H48_POLICY_NAME,
    B4_GENETIC_RECOVERY_H48_POLICY_NAME,
    B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
    B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL,
    B4_RECOVERY_BALANCE_H48_POLICY_NAME,
    B4_RECOVERY_BALANCE_H56_POLICY_NAME,
    B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
    B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL,
    B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
    B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
    B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL,
    B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
    B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE,
    B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
    B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME,
    B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
    B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
    B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
    B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL,
    B6_RECURRENT_MEMORY_SELECTION_SOURCE,
    B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
    B6_RISK_CORRIDOR_EFFECTIVE_LEVEL,
    B6_RISK_CORRIDOR_H56_POLICY_NAME,
    B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
    B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME,
)


@dataclass(frozen=True)
class BSeriesSemanticProgressionDescriptor:
    level: int
    selector_name: str
    effective_level: str | None = None
    policy_names: frozenset[str] = frozenset()
    b_mode: str | None = None
    effective_level_by_source: Mapping[str, str] = MappingProxyType({})
    updates_temporal_threat_trace: bool = True

    def matches(self, *, b_level: int, config_name: str, b_mode: str) -> bool:
        if b_level != self.level:
            return False
        if self.b_mode is not None and b_mode != self.b_mode:
            return False
        if self.policy_names and config_name not in self.policy_names:
            return False
        return True

    def effective_level_for(self, semantic_action_source: str, *, fallback: str) -> str:
        if semantic_action_source in self.effective_level_by_source:
            return self.effective_level_by_source[semantic_action_source]
        if self.effective_level is not None:
            return self.effective_level
        return fallback


B_SERIES_SEMANTIC_PROGRESSION_DESCRIPTORS: tuple[
    BSeriesSemanticProgressionDescriptor,
    ...,
] = (
    BSeriesSemanticProgressionDescriptor(
        level=0,
        selector_name="_b0_current_simple_semantic_action",
        effective_level=B_CURRENT_BRIDGE_EFFECTIVE_LEVEL,
        b_mode="current_bridge",
        updates_temporal_threat_trace=False,
    ),
    BSeriesSemanticProgressionDescriptor(
        level=1,
        selector_name="_b1_threat_guard_semantic_action",
        effective_level=B1_THREAT_GUARD_EFFECTIVE_LEVEL,
        policy_names=frozenset({B1_THREAT_GUARD_POLICY_NAME}),
        updates_temporal_threat_trace=False,
    ),
    BSeriesSemanticProgressionDescriptor(
        level=2,
        selector_name="_b2_temporal_threat_semantic_action",
        effective_level=B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL,
        policy_names=frozenset(
            {
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                B2_TEMPORAL_THREAT_H56_POLICY_NAME,
                B2_TEMPORAL_THREAT_H64_POLICY_NAME,
            }
        ),
    ),
    BSeriesSemanticProgressionDescriptor(
        level=3,
        selector_name="_b3_contact_memory_semantic_action",
        effective_level=B3_CONTACT_MEMORY_EFFECTIVE_LEVEL,
        policy_names=frozenset(
            {
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_H56_POLICY_NAME,
            }
        ),
    ),
    BSeriesSemanticProgressionDescriptor(
        level=3,
        selector_name="_b3_recurrent_guard_semantic_action",
        effective_level=B3_RECURRENT_GUARD_EFFECTIVE_LEVEL,
        policy_names=frozenset({B3_RECURRENT_GUARD_H48_POLICY_NAME}),
    ),
    BSeriesSemanticProgressionDescriptor(
        level=4,
        selector_name="_b4_recovery_balance_semantic_action",
        effective_level=B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL,
        policy_names=frozenset(
            {
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
                B4_RECOVERY_BALANCE_H56_POLICY_NAME,
                B4_GENETIC_RECOVERY_H48_POLICY_NAME,
            }
        ),
    ),
    BSeriesSemanticProgressionDescriptor(
        level=5,
        selector_name="_b5_homeostatic_arbiter_semantic_action",
        effective_level=B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL,
        policy_names=frozenset(
            {
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
                B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
                B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
            }
        ),
    ),
    BSeriesSemanticProgressionDescriptor(
        level=6,
        selector_name="_b6_risk_corridor_semantic_action",
        effective_level=B6_RISK_CORRIDOR_EFFECTIVE_LEVEL,
        policy_names=frozenset(
            {
                B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
                B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
                B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME,
                B6_RISK_CORRIDOR_H56_POLICY_NAME,
                B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME,
                B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
                B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
                B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
                B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
                B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
                B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
            }
        ),
        effective_level_by_source=MappingProxyType(
            {
                B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE: (
                    B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL
                ),
                B6_RECURRENT_MEMORY_SELECTION_SOURCE: (
                    B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL
                ),
            }
        ),
    ),
)


def find_b_series_semantic_progression_descriptor(
    *,
    b_level: int,
    config_name: str,
    b_mode: str,
) -> BSeriesSemanticProgressionDescriptor | None:
    for descriptor in B_SERIES_SEMANTIC_PROGRESSION_DESCRIPTORS:
        if descriptor.matches(
            b_level=b_level,
            config_name=config_name,
            b_mode=b_mode,
        ):
            return descriptor
    return None
