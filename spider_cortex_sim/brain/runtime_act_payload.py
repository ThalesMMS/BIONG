from __future__ import annotations

from .runtime_shared import *


from .b_series_trace_fields import (
    b_series_direct_policy_payload,
    b_series_inactive_runtime_payload,
)


class _BrainRuntimeActPayloadMixin:
    def _b_series_step_payload(
        self,
        direct_policy_trace_payload: Dict[str, object],
        semantic_logits: np.ndarray,
        semantic_policy: np.ndarray,
    ) -> dict[str, object]:
        if not self.config.is_b_series:
            return b_series_inactive_runtime_payload()

        payload = b_series_direct_policy_payload(direct_policy_trace_payload)
        payload.update(
            {
                "b_level": int(self.config.b_level),
                "b_effective_level": (
                    str(direct_policy_trace_payload.get("b_effective_level"))
                    if direct_policy_trace_payload.get("b_effective_level") is not None
                    else None
                ),
                "b_mode": str(self.config.b_mode),
                "b_parent_level": direct_policy_trace_payload.get("b_parent_level"),
                "b_transfer_source_checkpoint": direct_policy_trace_payload.get(
                    "b_transfer_source_checkpoint"
                ),
                "b_transfer_coverage": direct_policy_trace_payload.get(
                    "b_transfer_coverage"
                ),
                "semantic_action": direct_policy_trace_payload.get("semantic_action"),
                "semantic_action_idx": int(
                    direct_policy_trace_payload.get("semantic_action_idx", -1)
                ),
                "learned_semantic_action": direct_policy_trace_payload.get(
                    "learned_semantic_action"
                ),
                "learned_semantic_action_idx": int(
                    direct_policy_trace_payload.get("learned_semantic_action_idx", -1)
                ),
                "semantic_action_source": direct_policy_trace_payload.get(
                    "semantic_action_source"
                ),
                "semantic_action_reason": direct_policy_trace_payload.get(
                    "semantic_action_reason"
                ),
                "semantic_override_count": int(
                    direct_policy_trace_payload.get("semantic_override_count", 0)
                ),
                "semantic_logits": np.asarray(semantic_logits, dtype=float).copy(),
                "semantic_policy": np.asarray(semantic_policy, dtype=float).copy(),
                "bridge_primitive_action": direct_policy_trace_payload.get(
                    "bridge_primitive_action"
                ),
                "bridge_reason": direct_policy_trace_payload.get("bridge_reason"),
                "blocked_mask": dict(direct_policy_trace_payload.get("blocked_mask", {})),
                "food_delta_used": float(
                    direct_policy_trace_payload.get("food_delta_used", 0.0)
                ),
                "shelter_delta_used": float(
                    direct_policy_trace_payload.get("shelter_delta_used", 0.0)
                ),
            }
        )
        return payload
