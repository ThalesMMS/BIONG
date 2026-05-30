from __future__ import annotations

from .simulation_episode_shared import *


class _SimulationEpisodeMetricsTraceMixin:
    @staticmethod
    def _episode_stats_behavior_metrics(stats: EpisodeStats) -> Dict[str, object]:
        """
        Produce a flat mapping of behavior-metric fields derived from an EpisodeStats object.
        
        The returned dictionary contains named metric entries (floats, ints, and dicts) such as module dominance/share, routing/router statistics, owner alignment/rank/suppression, motor slip and orientation metrics, per-proposal-source contribution shares (keys prefixed with `module_contribution_`), and per-terrain slip rates (keys prefixed with `terrain_slip_rate_`). These fields are intended for inclusion in scenario scorecards and downstream reporting.
        
        Returns:
            Dict[str, object]: A dictionary mapping metric names to primitive values suitable for reporting.
        """
        metrics: Dict[str, object] = {
            "dominant_module": stats.dominant_module,
            "dominant_module_share": float(stats.dominant_module_share),
            "effective_module_count": float(stats.effective_module_count),
            "gate_entropy": float(stats.gate_entropy),
            "dominance_rate": float(stats.dominance_rate),
            "effective_proposer_count": float(stats.effective_proposer_count),
            "module_agreement_rate": float(stats.module_agreement_rate),
            "module_disagreement_rate": float(stats.module_disagreement_rate),
            "route_active_modules": dict(stats.route_active_modules),
            "router_health": dict(stats.router_health),
            "owner_alignment": float(stats.owner_alignment),
            "owner_rank": int(stats.owner_rank),
            "owner_suppressed_rate": float(stats.owner_suppressed_rate),
            "motor_slip_rate": float(stats.motor_slip_rate),
            "mean_orientation_alignment": float(stats.mean_orientation_alignment),
            "mean_terrain_difficulty": float(stats.mean_terrain_difficulty),
        }
        for module_name in PROPOSAL_SOURCE_NAMES:
            metrics[f"module_contribution_{module_name}"] = float(
                stats.module_contribution_share.get(module_name, 0.0)
            )
        for terrain, rate in sorted(stats.terrain_slip_rates.items()):
            metrics[f"terrain_slip_rate_{terrain}"] = float(rate)
        return metrics

    @staticmethod
    def _attach_motor_execution_info(decision: BrainStep, info: Dict[str, object]) -> None:
        """Attach authoritative post-world motor execution diagnostics to a BrainStep-like object."""
        slip_info = info.get("motor_slip", {})
        if not isinstance(slip_info, dict):
            slip_info = {}
        components = info.get("motor_execution_components", {})
        if not isinstance(components, dict):
            components = slip_info.get("components", {})
        if not isinstance(components, dict):
            components = {}

        slip_occurred = bool(
            slip_info.get("occurred", info.get("motor_noise_applied", False))
        )
        decision.execution_slip_occurred = slip_occurred
        decision.motor_slip_occurred = slip_occurred
        decision.motor_noise_applied = slip_occurred
        decision.slip_reason = str(
            info.get(
                "slip_reason",
                info.get("motor_slip_reason", slip_info.get("reason", "none")),
            )
        )
        decision.orientation_alignment = float(
            components.get(
                "orientation_alignment",
                getattr(decision, "orientation_alignment", 1.0),
            )
        )
        decision.terrain_difficulty = float(
            components.get(
                "terrain_difficulty",
                getattr(decision, "terrain_difficulty", 0.0),
            )
        )
        decision.execution_difficulty = float(
            info.get(
                "motor_execution_difficulty",
                slip_info.get(
                    "execution_difficulty",
                    getattr(decision, "execution_difficulty", 0.0),
                ),
            )
        )

    @staticmethod
    def _motor_stage_summary(history: List[EpisodeStats]) -> Dict[str, object]:
        """Aggregate motor execution observability metrics for summary output."""
        if not history:
            return {
                "episodes": 0,
                "mean_motor_slip_rate": 0.0,
                "mean_orientation_alignment": 0.0,
                "mean_terrain_difficulty": 0.0,
                "mean_terrain_slip_rates": {},
            }
        terrain_names = sorted(
            {
                terrain
                for stats in history
                for terrain in stats.terrain_slip_rates
            }
        )
        return {
            "episodes": len(history),
            "mean_motor_slip_rate": float(
                sum(stats.motor_slip_rate for stats in history) / len(history)
            ),
            "mean_orientation_alignment": float(
                sum(stats.mean_orientation_alignment for stats in history) / len(history)
            ),
            "mean_terrain_difficulty": float(
                sum(stats.mean_terrain_difficulty for stats in history) / len(history)
            ),
            "mean_terrain_slip_rates": {
                terrain: float(
                    sum(
                        stats.terrain_slip_rates.get(terrain, 0.0)
                        for stats in history
                    )
                    / len(history)
                )
                for terrain in terrain_names
            },
        }

    @staticmethod
    def _direct_policy_event_features(
        *,
        meta: Dict[str, object],
        strength: float = 1.0,
    ) -> list[float]:
        diagnostic = meta.get("diagnostic", {})
        if not isinstance(diagnostic, dict):
            diagnostic = {}
        threat = max(
            float(meta.get("visual_predator_threat", 0.0) or 0.0),
            float(meta.get("olfactory_predator_threat", 0.0) or 0.0),
            float(meta.get("predator_visible", 0.0) or 0.0),
        )
        return [
            float(strength),
            float(diagnostic.get("hunger", 0.0) or 0.0),
            float(meta.get("sleep_debt", 0.0) or 0.0),
            1.0 if bool(meta.get("on_shelter", False)) else 0.0,
            float(threat),
        ]

    def _record_direct_policy_events(
        self,
        *,
        step: int,
        decision: BrainStep,
        observation: Dict[str, object],
        next_observation: Dict[str, object],
        current_state: Dict[str, object],
        next_state: Dict[str, object],
        info: Dict[str, object],
    ) -> None:
        self.brain.set_direct_policy_event_clock(step + 1)
        if not self.brain.config.direct_policy_event_attention:
            return
        current_meta = observation["meta"]
        next_meta = next_observation["meta"]
        current_on_shelter = bool(current_meta.get("on_shelter", False))
        next_on_shelter = bool(next_meta.get("on_shelter", False))
        current_sleep_phase = str(current_meta.get("sleep_phase", ""))
        next_sleep_phase = str(next_meta.get("sleep_phase", ""))
        if bool(info.get("ate", False)):
            self.brain.record_direct_policy_event(
                "FOOD_EATEN",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta, strength=1.0),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if current_on_shelter and not next_on_shelter:
            self.brain.record_direct_policy_event(
                "SHELTER_EXIT",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if not current_on_shelter and next_on_shelter:
            self.brain.record_direct_policy_event(
                "SHELTER_RETURN",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if bool(info.get("slept", False)) and current_sleep_phase == "AWAKE":
            self.brain.record_direct_policy_event(
                "REST_STARTED",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if current_sleep_phase != "DEEP_SLEEP" and next_sleep_phase == "DEEP_SLEEP":
            self.brain.record_direct_policy_event(
                "DEEP_SLEEP_REACHED",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            bool(next_meta.get("day", False))
            and next_on_shelter
            and int(next_state.get("sleep_events", 0) or 0) > 0
            and next_sleep_phase == "AWAKE"
        ):
            self.brain.record_direct_policy_event(
                "RECOVERY_COMPLETED",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            bool(current_meta.get("day", False))
            and current_on_shelter
            and int(current_state.get("sleep_events", 0) or 0) > 0
            and current_sleep_phase == "AWAKE"
            and ACTIONS[decision.action_idx] != "STAY"
        ):
            self.brain.record_direct_policy_event(
                "POST_REST_RELEASE_ATTEMPT",
                features=np.asarray(
                    self._direct_policy_event_features(meta=current_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            str(info.get("intended_action", "")).startswith("MOVE_")
            and current_state.get("x") == next_state.get("x")
            and current_state.get("y") == next_state.get("y")
        ):
            self.brain.record_direct_policy_event(
                "BLOCKED_MOVE",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            bool(info.get("predator_contact", False))
            or bool(next_meta.get("predator_visible", False))
            or float(next_state.get("recent_contact", 0.0) or 0.0) > 0.0
            or float(next_state.get("recent_pain", 0.0) or 0.0) > 0.0
        ):
            self.brain.record_direct_policy_event(
                "ACUTE_PREDATOR_THREAT",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta, strength=1.2),
                    dtype=float,
                ),
                tick=step + 1,
            )
        if (
            not bool(next_meta.get("predator_visible", False))
            and float(next_meta.get("predator_memory_age", 1.0) or 1.0) < 1.0
        ):
            self.brain.record_direct_policy_event(
                "RESIDUAL_PREDATOR_MEMORY",
                features=np.asarray(
                    self._direct_policy_event_features(meta=next_meta, strength=0.7),
                    dtype=float,
                ),
                tick=step + 1,
            )
