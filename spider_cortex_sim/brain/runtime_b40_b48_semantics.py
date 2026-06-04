from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart6Mixin:
    def _b40_global_workspace_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b39_attention_binding_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b40_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "global_workspace"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b40_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        binding_strength = float(trace_payload.get("b39_binding_strength", 0.0) or 0.0)
        coherence = float(trace_payload.get("b39_cross_factor_coherence", 0.0) or 0.0)
        bound_context = float(trace_payload.get("b39_bound_context", 0.0) or 0.0)
        binding_gain = float(trace_payload.get("b39_binding_gain", 0.0) or 0.0)
        b39_decision = str(trace_payload.get("b39_decision", "preserve_b38"))

        decay = float(params["b40_workspace_decay"])
        previous_activation = float(getattr(self, "_b40_workspace_activation", 0.0))
        previous_context = float(getattr(self, "_b40_context_availability", 0.0))
        previous_stability = float(getattr(self, "_b40_workspace_stability", 0.0))
        workspace_activation = float(
            np.clip(
                previous_activation * decay
                + max(0.0, binding_strength) * float(params["b40_activation_gain"])
                + max(0.0, coherence) * 0.22,
                -1.0,
                1.0,
            )
        )
        broadcast_gain = float(
            np.clip(
                max(0.0, binding_gain) * float(params["b40_broadcast_gain"])
                + max(0.0, workspace_activation) * 0.26,
                -1.0,
                1.0,
            )
        )
        context_availability = float(
            np.clip(
                previous_context * decay
                + max(0.0, bound_context) * float(params["b40_context_gain"])
                + broadcast_gain * 0.22,
                -1.0,
                1.0,
            )
        )
        workspace_stability = float(
            np.clip(
                previous_stability * decay
                + workspace_activation * 0.30
                + context_availability * 0.26
                + broadcast_gain * 0.20
                - max(0.0, -coherence) * 0.18,
                -1.0,
                1.0,
            )
        )
        workspace_score = float(
            np.clip(
                workspace_activation * 0.34
                + broadcast_gain * 0.24
                + context_availability * 0.22
                + workspace_stability * 0.20,
                -1.0,
                1.0,
            )
        )
        workspace_lock = int(getattr(self, "_b40_workspace_lock", 0))
        decision_label = "preserve_b39"

        if corridor_map:
            if (
                b39_decision in {"attention_binding_commit", "continue_binding_lock"}
                and workspace_score >= float(params["b40_workspace_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                workspace_lock = max(
                    workspace_lock,
                    int(params["b40_workspace_lock_ticks"]),
                )
                decision_label = "global_workspace_commit"
                reason = "b40_global_workspace_commit"
            elif workspace_score <= float(params["b40_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "global_workspace_abort"
                reason = "b40_global_workspace_abort"
            elif workspace_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_workspace_lock"
                reason = "b40_continue_workspace_lock"

        trace_payload.update(
            {
                "b40_controller_profile": profile,
                "b40_workspace_activation": round(float(workspace_activation), 6),
                "b40_broadcast_gain": round(float(broadcast_gain), 6),
                "b40_context_availability": round(float(context_availability), 6),
                "b40_workspace_stability": round(float(workspace_stability), 6),
                "b40_workspace_lock": int(workspace_lock),
                "b40_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b40_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b40_genetic_candidate"] = int(params["ga_candidate"])

        self._b40_workspace_activation = float(workspace_activation)
        self._b40_context_availability = float(context_availability)
        self._b40_workspace_stability = float(workspace_stability)
        self._b40_workspace_lock = max(0, int(workspace_lock) - 1)
        self._b40_last_tick = int(tick)
        return (
            semantic_action,
            B40_GLOBAL_WORKSPACE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b41_controller_params(self) -> dict[str, float]:
        params = self._b40_controller_params()
        defaults = {
            "b41_executive_decay": 0.86,
            "b41_selection_gain": 0.34,
            "b41_inhibition_gain": 0.30,
            "b41_goal_context_gain": 0.32,
            "b41_selection_threshold": 0.12,
            "b41_abort_threshold": -0.18,
            "b41_executive_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "executive_workspace"
        )
        if profile == "inhibitory_control":
            defaults.update({"b41_inhibition_gain": 0.38, "b41_abort_threshold": -0.16})
        elif profile == "goal_context_selector":
            defaults.update(
                {"b41_goal_context_gain": 0.40, "b41_selection_threshold": 0.10}
            )
        elif profile == "executive_workspace_h56":
            defaults.update({"b41_executive_decay": 0.89, "b41_executive_lock_ticks": 6.0})
        elif profile == "genetic_executive_workspace":
            defaults.update({"b41_selection_gain": 0.38, "b41_goal_context_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b41_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b41_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b41_executive_selection = 0.0
        self._b41_goal_context = 0.0
        self._b41_executive_stability = 0.0
        self._b41_executive_lock = 0
        self._b41_last_tick = int(tick)

    def _b41_executive_workspace_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b40_global_workspace_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b41_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "executive_workspace"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b41_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        workspace_activation = float(
            trace_payload.get("b40_workspace_activation", 0.0) or 0.0
        )
        broadcast_gain = float(trace_payload.get("b40_broadcast_gain", 0.0) or 0.0)
        context_availability = float(
            trace_payload.get("b40_context_availability", 0.0) or 0.0
        )
        workspace_stability = float(
            trace_payload.get("b40_workspace_stability", 0.0) or 0.0
        )
        b40_decision = str(trace_payload.get("b40_decision", "preserve_b39"))

        decay = float(params["b41_executive_decay"])
        previous_selection = float(getattr(self, "_b41_executive_selection", 0.0))
        previous_context = float(getattr(self, "_b41_goal_context", 0.0))
        previous_stability = float(getattr(self, "_b41_executive_stability", 0.0))
        executive_selection = float(
            np.clip(
                previous_selection * decay
                + max(0.0, workspace_activation) * float(params["b41_selection_gain"])
                + max(0.0, broadcast_gain) * 0.24,
                -1.0,
                1.0,
            )
        )
        inhibitory_pressure = float(
            np.clip(
                max(0.0, -workspace_stability) * float(params["b41_inhibition_gain"])
                + max(0.0, -context_availability) * 0.22,
                0.0,
                1.0,
            )
        )
        goal_context = float(
            np.clip(
                previous_context * decay
                + max(0.0, context_availability) * float(params["b41_goal_context_gain"])
                + executive_selection * 0.22
                - inhibitory_pressure * 0.18,
                -1.0,
                1.0,
            )
        )
        executive_stability = float(
            np.clip(
                previous_stability * decay
                + executive_selection * 0.30
                + goal_context * 0.26
                + workspace_stability * 0.20
                - inhibitory_pressure * 0.20,
                -1.0,
                1.0,
            )
        )
        executive_score = float(
            np.clip(
                executive_selection * 0.34
                + goal_context * 0.26
                + executive_stability * 0.24
                - inhibitory_pressure * 0.16,
                -1.0,
                1.0,
            )
        )
        executive_lock = int(getattr(self, "_b41_executive_lock", 0))
        decision_label = "preserve_b40"

        if corridor_map:
            if (
                b40_decision in {"global_workspace_commit", "continue_workspace_lock"}
                and executive_score >= float(params["b41_selection_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                executive_lock = max(
                    executive_lock,
                    int(params["b41_executive_lock_ticks"]),
                )
                decision_label = "executive_workspace_select"
                reason = "b41_executive_workspace_select"
            elif executive_score <= float(params["b41_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "executive_workspace_abort"
                reason = "b41_executive_workspace_abort"
            elif executive_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_executive_lock"
                reason = "b41_continue_executive_lock"

        trace_payload.update(
            {
                "b41_controller_profile": profile,
                "b41_executive_selection": round(float(executive_selection), 6),
                "b41_inhibitory_pressure": round(float(inhibitory_pressure), 6),
                "b41_goal_context": round(float(goal_context), 6),
                "b41_executive_stability": round(float(executive_stability), 6),
                "b41_executive_lock": int(executive_lock),
                "b41_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b41_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b41_genetic_candidate"] = int(params["ga_candidate"])

        self._b41_executive_selection = float(executive_selection)
        self._b41_goal_context = float(goal_context)
        self._b41_executive_stability = float(executive_stability)
        self._b41_executive_lock = max(0, int(executive_lock) - 1)
        self._b41_last_tick = int(tick)
        return (
            semantic_action,
            B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b42_controller_params(self) -> dict[str, float]:
        params = self._b41_controller_params()
        defaults = {
            "b42_monitor_decay": 0.86,
            "b42_error_gain": 0.34,
            "b42_conflict_gain": 0.30,
            "b42_performance_gain": 0.32,
            "b42_commit_threshold": 0.12,
            "b42_abort_threshold": -0.18,
            "b42_monitor_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "error_monitor")
        if profile == "conflict_monitor":
            defaults.update({"b42_conflict_gain": 0.38, "b42_abort_threshold": -0.16})
        elif profile == "performance_monitor":
            defaults.update(
                {"b42_performance_gain": 0.40, "b42_commit_threshold": 0.10}
            )
        elif profile == "error_monitor_h56":
            defaults.update({"b42_monitor_decay": 0.89, "b42_monitor_lock_ticks": 6.0})
        elif profile == "genetic_error_monitor":
            defaults.update({"b42_error_gain": 0.38, "b42_performance_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b42_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b42_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b42_error_signal = 0.0
        self._b42_performance_context = 0.0
        self._b42_monitor_stability = 0.0
        self._b42_monitor_lock = 0
        self._b42_last_tick = int(tick)

    def _b42_error_monitor_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b41_executive_workspace_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b42_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "error_monitor")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b42_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        executive_selection = float(
            trace_payload.get("b41_executive_selection", 0.0) or 0.0
        )
        inhibitory_pressure = float(
            trace_payload.get("b41_inhibitory_pressure", 0.0) or 0.0
        )
        goal_context = float(trace_payload.get("b41_goal_context", 0.0) or 0.0)
        executive_stability = float(
            trace_payload.get("b41_executive_stability", 0.0) or 0.0
        )
        b41_decision = str(trace_payload.get("b41_decision", "preserve_b40"))

        decay = float(params["b42_monitor_decay"])
        previous_error = float(getattr(self, "_b42_error_signal", 0.0))
        previous_context = float(getattr(self, "_b42_performance_context", 0.0))
        previous_stability = float(getattr(self, "_b42_monitor_stability", 0.0))
        error_signal = float(
            np.clip(
                previous_error * decay
                + max(0.0, 1.0 - executive_stability) * float(params["b42_error_gain"]) * 0.20
                + inhibitory_pressure * 0.18,
                0.0,
                1.0,
            )
        )
        conflict_signal = float(
            np.clip(
                inhibitory_pressure * float(params["b42_conflict_gain"])
                + abs(executive_selection - goal_context) * 0.18,
                0.0,
                1.0,
            )
        )
        performance_context = float(
            np.clip(
                previous_context * decay
                + max(0.0, goal_context) * float(params["b42_performance_gain"])
                + max(0.0, executive_selection) * 0.24
                - conflict_signal * 0.12,
                -1.0,
                1.0,
            )
        )
        monitor_stability = float(
            np.clip(
                previous_stability * decay
                + performance_context * 0.28
                + executive_stability * 0.24
                - error_signal * 0.12
                - conflict_signal * 0.10,
                -1.0,
                1.0,
            )
        )
        monitor_score = float(
            np.clip(
                performance_context * 0.36
                + monitor_stability * 0.28
                + executive_selection * 0.24
                - error_signal * 0.06
                - conflict_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        monitor_lock = int(getattr(self, "_b42_monitor_lock", 0))
        decision_label = "preserve_b41"

        if corridor_map:
            if (
                b41_decision
                in {"executive_workspace_select", "continue_executive_lock"}
                and monitor_score >= float(params["b42_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                monitor_lock = max(monitor_lock, int(params["b42_monitor_lock_ticks"]))
                decision_label = "error_monitor_commit"
                reason = "b42_error_monitor_commit"
            elif monitor_score <= float(params["b42_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "error_monitor_abort"
                reason = "b42_error_monitor_abort"
            elif monitor_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_monitor_lock"
                reason = "b42_continue_monitor_lock"

        trace_payload.update(
            {
                "b42_controller_profile": profile,
                "b42_error_signal": round(float(error_signal), 6),
                "b42_conflict_signal": round(float(conflict_signal), 6),
                "b42_performance_context": round(float(performance_context), 6),
                "b42_monitor_stability": round(float(monitor_stability), 6),
                "b42_monitor_lock": int(monitor_lock),
                "b42_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b42_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b42_genetic_candidate"] = int(params["ga_candidate"])

        self._b42_error_signal = float(error_signal)
        self._b42_performance_context = float(performance_context)
        self._b42_monitor_stability = float(monitor_stability)
        self._b42_monitor_lock = max(0, int(monitor_lock) - 1)
        self._b42_last_tick = int(tick)
        return (
            semantic_action,
            B42_ERROR_MONITOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b43_controller_params(self) -> dict[str, float]:
        params = self._b42_controller_params()
        defaults = {
            "b43_precision_decay": 0.86,
            "b43_precision_gain": 0.34,
            "b43_arousal_gain": 0.30,
            "b43_threshold_gain": 0.32,
            "b43_commit_threshold": 0.12,
            "b43_abort_threshold": -0.18,
            "b43_precision_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "adaptive_precision"
        )
        if profile == "arousal_precision":
            defaults.update({"b43_arousal_gain": 0.38, "b43_abort_threshold": -0.16})
        elif profile == "threshold_adaptation":
            defaults.update({"b43_threshold_gain": 0.40, "b43_commit_threshold": 0.10})
        elif profile == "adaptive_precision_h56":
            defaults.update({"b43_precision_decay": 0.89, "b43_precision_lock_ticks": 6.0})
        elif profile == "genetic_adaptive_precision":
            defaults.update({"b43_precision_gain": 0.38, "b43_threshold_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b43_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b43_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b43_precision_signal = 0.0
        self._b43_arousal_context = 0.0
        self._b43_control_stability = 0.0
        self._b43_precision_lock = 0
        self._b43_last_tick = int(tick)

    def _b43_adaptive_precision_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b42_error_monitor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b43_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "adaptive_precision"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b43_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        error_signal = float(trace_payload.get("b42_error_signal", 0.0) or 0.0)
        conflict_signal = float(trace_payload.get("b42_conflict_signal", 0.0) or 0.0)
        performance_context = float(
            trace_payload.get("b42_performance_context", 0.0) or 0.0
        )
        monitor_stability = float(
            trace_payload.get("b42_monitor_stability", 0.0) or 0.0
        )
        b42_decision = str(trace_payload.get("b42_decision", "preserve_b41"))

        decay = float(params["b43_precision_decay"])
        previous_precision = float(getattr(self, "_b43_precision_signal", 0.0))
        previous_arousal = float(getattr(self, "_b43_arousal_context", 0.0))
        previous_stability = float(getattr(self, "_b43_control_stability", 0.0))
        adaptive_threshold = float(
            np.clip(
                float(params["b43_commit_threshold"])
                + conflict_signal * float(params["b43_threshold_gain"]) * 0.10
                + error_signal * 0.08
                - max(0.0, performance_context) * 0.06,
                -0.25,
                0.45,
            )
        )
        precision_signal = float(
            np.clip(
                previous_precision * decay
                + max(0.0, performance_context) * float(params["b43_precision_gain"])
                + max(0.0, monitor_stability) * 0.22
                - error_signal * 0.08
                - conflict_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        arousal_context = float(
            np.clip(
                previous_arousal * decay
                + (error_signal + conflict_signal) * float(params["b43_arousal_gain"]) * 0.20
                + max(0.0, precision_signal) * 0.18,
                0.0,
                1.0,
            )
        )
        control_stability = float(
            np.clip(
                previous_stability * decay
                + precision_signal * 0.30
                + monitor_stability * 0.24
                + performance_context * 0.20
                - arousal_context * 0.08,
                -1.0,
                1.0,
            )
        )
        precision_score = float(
            np.clip(
                precision_signal * 0.36
                + control_stability * 0.28
                + performance_context * 0.24
                - adaptive_threshold * 0.08
                - arousal_context * 0.04,
                -1.0,
                1.0,
            )
        )
        precision_lock = int(getattr(self, "_b43_precision_lock", 0))
        decision_label = "preserve_b42"

        if corridor_map:
            if (
                b42_decision in {"error_monitor_commit", "continue_monitor_lock"}
                and precision_score >= adaptive_threshold
            ):
                semantic_action = "MOVE_TO_FOOD"
                precision_lock = max(
                    precision_lock,
                    int(params["b43_precision_lock_ticks"]),
                )
                decision_label = "adaptive_precision_commit"
                reason = "b43_adaptive_precision_commit"
            elif precision_score <= float(params["b43_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "adaptive_precision_abort"
                reason = "b43_adaptive_precision_abort"
            elif precision_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_precision_lock"
                reason = "b43_continue_precision_lock"

        trace_payload.update(
            {
                "b43_controller_profile": profile,
                "b43_precision_signal": round(float(precision_signal), 6),
                "b43_adaptive_threshold": round(float(adaptive_threshold), 6),
                "b43_arousal_context": round(float(arousal_context), 6),
                "b43_control_stability": round(float(control_stability), 6),
                "b43_precision_lock": int(precision_lock),
                "b43_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b43_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b43_genetic_candidate"] = int(params["ga_candidate"])

        self._b43_precision_signal = float(precision_signal)
        self._b43_arousal_context = float(arousal_context)
        self._b43_control_stability = float(control_stability)
        self._b43_precision_lock = max(0, int(precision_lock) - 1)
        self._b43_last_tick = int(tick)
        return (
            semantic_action,
            B43_ADAPTIVE_PRECISION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b44_controller_params(self) -> dict[str, float]:
        params = self._b43_controller_params()
        defaults = {
            "b44_relay_decay": 0.86,
            "b44_gate_gain": 0.34,
            "b44_sensory_gain": 0.30,
            "b44_context_gain": 0.32,
            "b44_relay_threshold": 0.12,
            "b44_abort_threshold": -0.18,
            "b44_relay_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "thalamic_relay"
        )
        if profile == "sensory_relay":
            defaults.update({"b44_sensory_gain": 0.38, "b44_abort_threshold": -0.16})
        elif profile == "context_relay":
            defaults.update({"b44_context_gain": 0.40, "b44_relay_threshold": 0.10})
        elif profile == "thalamic_relay_h56":
            defaults.update({"b44_relay_decay": 0.89, "b44_relay_lock_ticks": 6.0})
        elif profile == "genetic_thalamic_relay":
            defaults.update({"b44_gate_gain": 0.38, "b44_context_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b44_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b44_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b44_relay_gate = 0.0
        self._b44_context_relay = 0.0
        self._b44_gate_stability = 0.0
        self._b44_relay_lock = 0
        self._b44_last_tick = int(tick)

    def _b44_thalamic_relay_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b43_adaptive_precision_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b44_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "thalamic_relay")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b44_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        precision_signal = float(trace_payload.get("b43_precision_signal", 0.0) or 0.0)
        adaptive_threshold = float(
            trace_payload.get("b43_adaptive_threshold", 0.0) or 0.0
        )
        arousal_context = float(trace_payload.get("b43_arousal_context", 0.0) or 0.0)
        control_stability = float(
            trace_payload.get("b43_control_stability", 0.0) or 0.0
        )
        b43_decision = str(trace_payload.get("b43_decision", "preserve_b42"))

        decay = float(params["b44_relay_decay"])
        previous_gate = float(getattr(self, "_b44_relay_gate", 0.0))
        previous_context = float(getattr(self, "_b44_context_relay", 0.0))
        previous_stability = float(getattr(self, "_b44_gate_stability", 0.0))
        sensory_precision = float(
            np.clip(
                max(0.0, precision_signal) * float(params["b44_sensory_gain"])
                + max(0.0, control_stability) * 0.24
                - max(0.0, arousal_context - 0.35) * 0.08,
                0.0,
                1.0,
            )
        )
        context_relay = float(
            np.clip(
                previous_context * decay
                + max(0.0, control_stability) * float(params["b44_context_gain"])
                + max(0.0, precision_signal - adaptive_threshold) * 0.22,
                -1.0,
                1.0,
            )
        )
        relay_gate = float(
            np.clip(
                previous_gate * decay
                + sensory_precision * float(params["b44_gate_gain"])
                + context_relay * 0.26
                - adaptive_threshold * 0.08,
                -1.0,
                1.0,
            )
        )
        gate_stability = float(
            np.clip(
                previous_stability * decay
                + relay_gate * 0.34
                + control_stability * 0.22
                + sensory_precision * 0.20
                - arousal_context * 0.06,
                -1.0,
                1.0,
            )
        )
        relay_score = float(
            np.clip(
                relay_gate * 0.34
                + gate_stability * 0.26
                + sensory_precision * 0.22
                + context_relay * 0.18,
                -1.0,
                1.0,
            )
        )
        relay_lock = int(getattr(self, "_b44_relay_lock", 0))
        decision_label = "preserve_b43"

        if corridor_map:
            if (
                b43_decision in {"adaptive_precision_commit", "continue_precision_lock"}
                and relay_score >= float(params["b44_relay_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                relay_lock = max(relay_lock, int(params["b44_relay_lock_ticks"]))
                decision_label = "thalamic_relay_commit"
                reason = "b44_thalamic_relay_commit"
            elif relay_score <= float(params["b44_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "thalamic_relay_abort"
                reason = "b44_thalamic_relay_abort"
            elif relay_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_relay_lock"
                reason = "b44_continue_relay_lock"

        trace_payload.update(
            {
                "b44_controller_profile": profile,
                "b44_relay_gate": round(float(relay_gate), 6),
                "b44_sensory_precision": round(float(sensory_precision), 6),
                "b44_context_relay": round(float(context_relay), 6),
                "b44_gate_stability": round(float(gate_stability), 6),
                "b44_relay_lock": int(relay_lock),
                "b44_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b44_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b44_genetic_candidate"] = int(params["ga_candidate"])

        self._b44_relay_gate = float(relay_gate)
        self._b44_context_relay = float(context_relay)
        self._b44_gate_stability = float(gate_stability)
        self._b44_relay_lock = max(0, int(relay_lock) - 1)
        self._b44_last_tick = int(tick)
        return (
            semantic_action,
            B44_THALAMIC_RELAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b45_controller_params(self) -> dict[str, float]:
        params = self._b44_controller_params()
        defaults = {
            "b45_inhibition_decay": 0.86,
            "b45_inhibitory_gain": 0.34,
            "b45_sensory_filter_gain": 0.30,
            "b45_context_suppression_gain": 0.32,
            "b45_commit_threshold": 0.12,
            "b45_abort_threshold": -0.18,
            "b45_inhibition_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "reticular_inhibition"
        )
        if profile == "sensory_inhibition":
            defaults.update({"b45_sensory_filter_gain": 0.38, "b45_abort_threshold": -0.16})
        elif profile == "context_inhibition":
            defaults.update(
                {"b45_context_suppression_gain": 0.40, "b45_commit_threshold": 0.10}
            )
        elif profile == "reticular_inhibition_h56":
            defaults.update(
                {"b45_inhibition_decay": 0.89, "b45_inhibition_lock_ticks": 6.0}
            )
        elif profile == "genetic_reticular_inhibition":
            defaults.update({"b45_inhibitory_gain": 0.38, "b45_context_suppression_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b45_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b45_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b45_inhibitory_gate = 0.0
        self._b45_context_suppression = 0.0
        self._b45_loop_stability = 0.0
        self._b45_inhibition_lock = 0
        self._b45_last_tick = int(tick)

    def _b45_reticular_inhibition_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b44_thalamic_relay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b45_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "reticular_inhibition"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b45_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        relay_gate = float(trace_payload.get("b44_relay_gate", 0.0) or 0.0)
        sensory_precision = float(trace_payload.get("b44_sensory_precision", 0.0) or 0.0)
        context_relay = float(trace_payload.get("b44_context_relay", 0.0) or 0.0)
        gate_stability = float(trace_payload.get("b44_gate_stability", 0.0) or 0.0)
        b44_decision = str(trace_payload.get("b44_decision", "preserve_b43"))

        decay = float(params["b45_inhibition_decay"])
        previous_gate = float(getattr(self, "_b45_inhibitory_gate", 0.0))
        previous_suppression = float(getattr(self, "_b45_context_suppression", 0.0))
        previous_stability = float(getattr(self, "_b45_loop_stability", 0.0))
        sensory_filter = float(
            np.clip(
                max(0.0, sensory_precision) * float(params["b45_sensory_filter_gain"])
                + max(0.0, relay_gate) * 0.22
                - max(0.0, -context_relay) * 0.08,
                0.0,
                1.0,
            )
        )
        context_suppression = float(
            np.clip(
                previous_suppression * decay
                + max(0.0, context_relay) * float(params["b45_context_suppression_gain"])
                + max(0.0, gate_stability) * 0.22,
                -1.0,
                1.0,
            )
        )
        inhibitory_gate = float(
            np.clip(
                previous_gate * decay
                + sensory_filter * float(params["b45_inhibitory_gain"])
                + context_suppression * 0.24
                + max(0.0, relay_gate) * 0.18,
                -1.0,
                1.0,
            )
        )
        loop_stability = float(
            np.clip(
                previous_stability * decay
                + inhibitory_gate * 0.32
                + gate_stability * 0.24
                + sensory_filter * 0.18
                - max(0.0, -context_suppression) * 0.06,
                -1.0,
                1.0,
            )
        )
        inhibition_score = float(
            np.clip(
                inhibitory_gate * 0.34
                + loop_stability * 0.28
                + sensory_filter * 0.20
                + context_suppression * 0.18,
                -1.0,
                1.0,
            )
        )
        inhibition_lock = int(getattr(self, "_b45_inhibition_lock", 0))
        decision_label = "preserve_b44"

        if corridor_map:
            if (
                b44_decision in {"thalamic_relay_commit", "continue_relay_lock"}
                and inhibition_score >= float(params["b45_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                inhibition_lock = max(
                    inhibition_lock,
                    int(params["b45_inhibition_lock_ticks"]),
                )
                decision_label = "reticular_inhibition_commit"
                reason = "b45_reticular_inhibition_commit"
            elif inhibition_score <= float(params["b45_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "reticular_inhibition_abort"
                reason = "b45_reticular_inhibition_abort"
            elif inhibition_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_inhibition_lock"
                reason = "b45_continue_inhibition_lock"

        trace_payload.update(
            {
                "b45_controller_profile": profile,
                "b45_inhibitory_gate": round(float(inhibitory_gate), 6),
                "b45_sensory_filter": round(float(sensory_filter), 6),
                "b45_context_suppression": round(float(context_suppression), 6),
                "b45_loop_stability": round(float(loop_stability), 6),
                "b45_inhibition_lock": int(inhibition_lock),
                "b45_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b45_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b45_genetic_candidate"] = int(params["ga_candidate"])

        self._b45_inhibitory_gate = float(inhibitory_gate)
        self._b45_context_suppression = float(context_suppression)
        self._b45_loop_stability = float(loop_stability)
        self._b45_inhibition_lock = max(0, int(inhibition_lock) - 1)
        self._b45_last_tick = int(tick)
        return (
            semantic_action,
            B45_RETICULAR_INHIBITION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b46_controller_params(self) -> dict[str, float]:
        params = self._b45_controller_params()
        defaults = {
            "b46_feedback_decay": 0.86,
            "b46_feedback_gain": 0.34,
            "b46_topdown_gain": 0.30,
            "b46_prediction_gain": 0.32,
            "b46_commit_threshold": 0.08,
            "b46_abort_threshold": -0.18,
            "b46_feedback_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "corticothalamic_feedback"
        )
        if profile == "feedback_gain":
            defaults.update({"b46_feedback_gain": 0.38, "b46_abort_threshold": -0.16})
        elif profile == "context_feedback":
            defaults.update({"b46_topdown_gain": 0.40, "b46_commit_threshold": 0.07})
        elif profile == "corticothalamic_feedback_h56":
            defaults.update({"b46_feedback_decay": 0.89, "b46_feedback_lock_ticks": 6.0})
        elif profile == "genetic_corticothalamic_feedback":
            defaults.update({"b46_feedback_gain": 0.38, "b46_prediction_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b46_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b46_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b46_feedback_gain = 0.0
        self._b46_topdown_context = 0.0
        self._b46_feedback_stability = 0.0
        self._b46_feedback_lock = 0
        self._b46_last_tick = int(tick)

    def _b46_corticothalamic_feedback_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b45_reticular_inhibition_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b46_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "corticothalamic_feedback"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b46_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        inhibitory_gate = float(trace_payload.get("b45_inhibitory_gate", 0.0) or 0.0)
        sensory_filter = float(trace_payload.get("b45_sensory_filter", 0.0) or 0.0)
        context_suppression = float(
            trace_payload.get("b45_context_suppression", 0.0) or 0.0
        )
        loop_stability = float(trace_payload.get("b45_loop_stability", 0.0) or 0.0)
        b45_decision = str(trace_payload.get("b45_decision", "preserve_b44"))

        decay = float(params["b46_feedback_decay"])
        previous_gain = float(getattr(self, "_b46_feedback_gain", 0.0))
        previous_context = float(getattr(self, "_b46_topdown_context", 0.0))
        previous_stability = float(getattr(self, "_b46_feedback_stability", 0.0))
        topdown_context = float(
            np.clip(
                previous_context * decay
                + max(0.0, context_suppression) * float(params["b46_topdown_gain"])
                + max(0.0, loop_stability) * 0.22,
                -1.0,
                1.0,
            )
        )
        prediction_match = float(
            np.clip(
                max(0.0, sensory_filter) * float(params["b46_prediction_gain"])
                + max(0.0, inhibitory_gate) * 0.24
                + max(0.0, topdown_context) * 0.18,
                0.0,
                1.0,
            )
        )
        feedback_gain = float(
            np.clip(
                previous_gain * decay
                + prediction_match * float(params["b46_feedback_gain"])
                + topdown_context * 0.24
                + max(0.0, loop_stability) * 0.18,
                -1.0,
                1.0,
            )
        )
        feedback_stability = float(
            np.clip(
                previous_stability * decay
                + feedback_gain * 0.34
                + prediction_match * 0.22
                + loop_stability * 0.24
                - max(0.0, -topdown_context) * 0.06,
                -1.0,
                1.0,
            )
        )
        feedback_score = float(
            np.clip(
                feedback_gain * 0.34
                + feedback_stability * 0.28
                + prediction_match * 0.20
                + topdown_context * 0.18,
                -1.0,
                1.0,
            )
        )
        feedback_lock = int(getattr(self, "_b46_feedback_lock", 0))
        decision_label = "preserve_b45"

        if corridor_map:
            if (
                b45_decision
                in {"reticular_inhibition_commit", "continue_inhibition_lock"}
                and feedback_score >= float(params["b46_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                feedback_lock = max(feedback_lock, int(params["b46_feedback_lock_ticks"]))
                decision_label = "corticothalamic_feedback_commit"
                reason = "b46_corticothalamic_feedback_commit"
            elif feedback_score <= float(params["b46_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "corticothalamic_feedback_abort"
                reason = "b46_corticothalamic_feedback_abort"
            elif feedback_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_feedback_lock"
                reason = "b46_continue_feedback_lock"

        trace_payload.update(
            {
                "b46_controller_profile": profile,
                "b46_feedback_gain": round(float(feedback_gain), 6),
                "b46_topdown_context": round(float(topdown_context), 6),
                "b46_prediction_match": round(float(prediction_match), 6),
                "b46_feedback_stability": round(float(feedback_stability), 6),
                "b46_feedback_lock": int(feedback_lock),
                "b46_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b46_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b46_genetic_candidate"] = int(params["ga_candidate"])

        self._b46_feedback_gain = float(feedback_gain)
        self._b46_topdown_context = float(topdown_context)
        self._b46_feedback_stability = float(feedback_stability)
        self._b46_feedback_lock = max(0, int(feedback_lock) - 1)
        self._b46_last_tick = int(tick)
        return (
            semantic_action,
            B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b47_controller_params(self) -> dict[str, float]:
        params = self._b46_controller_params()
        defaults = {
            "b47_phase_decay": 0.86,
            "b47_phase_gain": 0.32,
            "b47_synchrony_gain": 0.34,
            "b47_coherence_gain": 0.30,
            "b47_commit_threshold": 0.08,
            "b47_abort_threshold": -0.18,
            "b47_phase_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "oscillatory_synchrony"
        )
        if profile == "phase_locking":
            defaults.update({"b47_phase_gain": 0.38, "b47_phase_lock_ticks": 6.0})
        elif profile == "coherence_gate":
            defaults.update({"b47_coherence_gain": 0.38, "b47_commit_threshold": 0.07})
        elif profile == "oscillatory_synchrony_h56":
            defaults.update({"b47_phase_decay": 0.89, "b47_phase_lock_ticks": 6.0})
        elif profile == "genetic_oscillatory_synchrony":
            defaults.update({"b47_synchrony_gain": 0.38, "b47_coherence_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b47_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b47_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b47_phase_alignment = 0.0
        self._b47_synchrony_gain = 0.0
        self._b47_cross_loop_coherence = 0.0
        self._b47_phase_lock = 0
        self._b47_last_tick = int(tick)

    def _b47_oscillatory_synchrony_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b46_corticothalamic_feedback_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b47_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "oscillatory_synchrony"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b47_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        feedback_gain = float(trace_payload.get("b46_feedback_gain", 0.0) or 0.0)
        topdown_context = float(trace_payload.get("b46_topdown_context", 0.0) or 0.0)
        prediction_match = float(trace_payload.get("b46_prediction_match", 0.0) or 0.0)
        feedback_stability = float(
            trace_payload.get("b46_feedback_stability", 0.0) or 0.0
        )
        b46_decision = str(trace_payload.get("b46_decision", "preserve_b45"))

        decay = float(params["b47_phase_decay"])
        previous_phase = float(getattr(self, "_b47_phase_alignment", 0.0))
        previous_synchrony = float(getattr(self, "_b47_synchrony_gain", 0.0))
        previous_coherence = float(getattr(self, "_b47_cross_loop_coherence", 0.0))
        phase_drive = float(np.cos(max(0, tick) * 0.37))
        phase_alignment = float(
            np.clip(
                previous_phase * decay
                + max(0.0, feedback_stability) * float(params["b47_phase_gain"])
                + max(0.0, prediction_match) * 0.22
                + max(0.0, phase_drive) * 0.04,
                -1.0,
                1.0,
            )
        )
        synchrony_gain = float(
            np.clip(
                previous_synchrony * decay
                + max(0.0, feedback_gain) * float(params["b47_synchrony_gain"])
                + max(0.0, phase_alignment) * 0.24,
                -1.0,
                1.0,
            )
        )
        cross_loop_coherence = float(
            np.clip(
                previous_coherence * decay
                + max(0.0, topdown_context) * float(params["b47_coherence_gain"])
                + max(0.0, synchrony_gain) * 0.26
                + max(0.0, prediction_match) * 0.18,
                -1.0,
                1.0,
            )
        )
        synchrony_score = float(
            np.clip(
                phase_alignment * 0.30
                + synchrony_gain * 0.32
                + cross_loop_coherence * 0.26
                + feedback_stability * 0.12,
                -1.0,
                1.0,
            )
        )
        phase_lock = int(getattr(self, "_b47_phase_lock", 0))
        decision_label = "preserve_b46"

        if corridor_map:
            if (
                b46_decision
                in {"corticothalamic_feedback_commit", "continue_feedback_lock"}
                and synchrony_score >= float(params["b47_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                phase_lock = max(phase_lock, int(params["b47_phase_lock_ticks"]))
                decision_label = "oscillatory_synchrony_commit"
                reason = "b47_oscillatory_synchrony_commit"
            elif synchrony_score <= float(params["b47_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "oscillatory_synchrony_abort"
                reason = "b47_oscillatory_synchrony_abort"
            elif phase_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_phase_lock"
                reason = "b47_continue_phase_lock"

        trace_payload.update(
            {
                "b47_controller_profile": profile,
                "b47_phase_alignment": round(float(phase_alignment), 6),
                "b47_synchrony_gain": round(float(synchrony_gain), 6),
                "b47_cross_loop_coherence": round(float(cross_loop_coherence), 6),
                "b47_phase_lock": int(phase_lock),
                "b47_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b47_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b47_genetic_candidate"] = int(params["ga_candidate"])

        self._b47_phase_alignment = float(phase_alignment)
        self._b47_synchrony_gain = float(synchrony_gain)
        self._b47_cross_loop_coherence = float(cross_loop_coherence)
        self._b47_phase_lock = max(0, int(phase_lock) - 1)
        self._b47_last_tick = int(tick)
        return (
            semantic_action,
            B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b48_controller_params(self) -> dict[str, float]:
        params = self._b47_controller_params()
        defaults = {
            "b48_timing_decay": 0.86,
            "b48_error_gain": 0.30,
            "b48_prediction_gain": 0.34,
            "b48_corrective_gain": 0.32,
            "b48_commit_threshold": 0.08,
            "b48_abort_threshold": -0.18,
            "b48_calibration_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cerebellar_timing"
        )
        if profile == "timing_error_correction":
            defaults.update({"b48_error_gain": 0.38, "b48_corrective_gain": 0.36})
        elif profile == "predictive_timing":
            defaults.update({"b48_prediction_gain": 0.40, "b48_commit_threshold": 0.07})
        elif profile == "cerebellar_timing_h56":
            defaults.update({"b48_timing_decay": 0.89, "b48_calibration_lock_ticks": 6.0})
        elif profile == "genetic_cerebellar_timing":
            defaults.update({"b48_prediction_gain": 0.38, "b48_corrective_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b48_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b48_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b48_timing_error = 0.0
        self._b48_predictive_timing = 0.0
        self._b48_corrective_gain = 0.0
        self._b48_calibration_lock = 0
        self._b48_last_tick = int(tick)
