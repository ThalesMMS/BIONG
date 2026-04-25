from __future__ import annotations

from spider_cortex_sim.offline_analysis.constants import (
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
)

from .conftest import (
    EPISODE_SHAPING_GAP,
    LARGE_SHAPING_GAP,
    MEAN_REWARD_SHAPING_GAP,
)


class OfflineAnalysisToleranceFixtures:
    def _summary_with_shaping_program(self) -> dict[str, object]:
        """
        Create a synthetic summary dict representing a shaping-minimization reward audit used in offline-analysis tests.
        
        The dict contains a top-level "reward_audit" mapping with:
        - "minimal_profile": the selected minimal shaping profile ("austere").
        - "reward_components": example components including category, shaping_risk, shaping_disposition, and optional disposition_rationale.
        - "reward_profiles": per-profile disposition_summary values (total_weight_proxy) for dispositions such as "removed" and "outcome_signal".
        - "comparison": profile deltas vs the minimal profile and a "behavior_survival" block with survival_threshold, counts, survival_rate, and per-scenario survival details.
        
        Returns:
            dict: Synthetic summary suitable for shaping-audit and report-generation tests.
        """
        return {
            "reward_audit": {
                "minimal_profile": "austere",
                "reward_components": {
                    "feeding": {
                        "category": "event",
                        "shaping_risk": "low",
                        "shaping_disposition": "outcome_signal",
                    },
                    "food_progress": {
                        "category": "progress",
                        "shaping_risk": "high",
                        "shaping_disposition": "removed",
                        "disposition_rationale": "Zeroed in the austere profile.",
                    },
                },
                "reward_profiles": {
                    "classic": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 1.2},
                            "outcome_signal": {"total_weight_proxy": 3.48},
                        }
                    },
                    "austere": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 0.0},
                            "outcome_signal": {"total_weight_proxy": 3.48},
                        }
                    },
                },
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": LARGE_SHAPING_GAP,
                            "episode_success_rate_delta": EPISODE_SHAPING_GAP,
                            "mean_reward_delta": MEAN_REWARD_SHAPING_GAP,
                        },
                        "austere": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        },
                    },
                    "behavior_survival": {
                        "available": True,
                        "minimal_profile": "austere",
                        "survival_threshold": DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
                        "scenario_count": 2,
                        "surviving_scenario_count": 1,
                        "survival_rate": 0.5,
                        "scenarios": {
                            "night_rest": {
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 2,
                            },
                            "open_field_foraging": {
                                "austere_success_rate": 0.0,
                                "survives": False,
                                "episodes": 2,
                            },
                        },
                    },
                },
            }
        }

    def _summary_with_credit_analysis_program(self) -> dict[str, object]:
        """
        Constructs a synthetic `behavior_evaluation` payload containing ablation variants for credit-analysis tests.
        
        The returned dictionary has the top-level key "behavior_evaluation" with an "ablations" mapping. "ablations" contains:
        - "reference_variant": the name of the reference variant.
        - "variants": a mapping from variant name to a variant record. Each variant record contains:
          - "config": { "architecture": str, "credit_strategy": str }
          - "summary": aggregate metrics including "scenario_success_rate", "episode_success_rate",
            "mean_module_credit_weights", "module_gradient_norm_means",
            "mean_counterfactual_credit_weights", "dominant_module",
            "mean_dominant_module_share", and "mean_effective_module_count".
          - "suite": per-suite success-rate mappings (e.g., "night_rest": {"success_rate": float}).
        
        Returns:
            dict[str, object]: A dictionary encoding the synthetic behavior evaluation and ablation variants.
        """
        def make_variant(
            *,
            architecture: str,
            credit_strategy: str,
            scenario_success_rate: float,
            episode_success_rate: float,
            mean_module_credit_weights: dict[str, float],
            module_gradient_norm_means: dict[str, float],
            mean_counterfactual_credit_weights: dict[str, float],
            dominant_module: str,
            mean_dominant_module_share: float,
            mean_effective_module_count: float,
            suite_success_rate: float | None = None,
        ) -> dict[str, object]:
            suite_rate = (
                scenario_success_rate
                if suite_success_rate is None
                else suite_success_rate
            )
            return {
                "config": {
                    "architecture": architecture,
                    "credit_strategy": credit_strategy,
                },
                "summary": {
                    "scenario_success_rate": scenario_success_rate,
                    "episode_success_rate": episode_success_rate,
                    "mean_module_credit_weights": mean_module_credit_weights,
                    "module_gradient_norm_means": module_gradient_norm_means,
                    "mean_counterfactual_credit_weights": (
                        mean_counterfactual_credit_weights
                    ),
                    "dominant_module": dominant_module,
                    "mean_dominant_module_share": mean_dominant_module_share,
                    "mean_effective_module_count": mean_effective_module_count,
                },
                "suite": {"night_rest": {"success_rate": suite_rate}},
            }

        return {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "three_center_modular": make_variant(
                            architecture="modular",
                            credit_strategy="broadcast",
                            scenario_success_rate=0.6,
                            episode_success_rate=0.6,
                            mean_module_credit_weights={
                                "perception_center": 0.34,
                                "action_center": 0.33,
                                "context_center": 0.33,
                            },
                            module_gradient_norm_means={
                                "perception_center": 0.4,
                                "action_center": 0.4,
                                "context_center": 0.4,
                            },
                            mean_counterfactual_credit_weights={
                                "perception_center": 0.2,
                                "action_center": 0.2,
                                "context_center": 0.2,
                            },
                            dominant_module="perception_center",
                            mean_dominant_module_share=0.34,
                            mean_effective_module_count=3.0,
                        ),
                        "three_center_modular_local_credit": make_variant(
                            architecture="modular",
                            credit_strategy="local_only",
                            scenario_success_rate=0.25,
                            episode_success_rate=0.25,
                            mean_module_credit_weights={
                                "perception_center": 0.0,
                                "action_center": 0.0,
                                "context_center": 0.0,
                            },
                            module_gradient_norm_means={
                                "perception_center": 0.0,
                                "action_center": 0.0,
                                "context_center": 0.0,
                            },
                            mean_counterfactual_credit_weights={
                                "perception_center": 0.0,
                                "action_center": 0.0,
                                "context_center": 0.0,
                            },
                            dominant_module="",
                            mean_dominant_module_share=0.0,
                            mean_effective_module_count=0.0,
                        ),
                        "three_center_modular_counterfactual": make_variant(
                            architecture="modular",
                            credit_strategy="counterfactual",
                            scenario_success_rate=0.75,
                            episode_success_rate=0.75,
                            mean_module_credit_weights={
                                "perception_center": 0.65,
                                "action_center": 0.2,
                                "context_center": 0.15,
                            },
                            module_gradient_norm_means={
                                "perception_center": 1.8,
                                "action_center": 0.8,
                                "context_center": 0.6,
                            },
                            mean_counterfactual_credit_weights={
                                "perception_center": 0.7,
                                "action_center": 0.2,
                                "context_center": 0.1,
                            },
                            dominant_module="perception_center",
                            mean_dominant_module_share=0.65,
                            mean_effective_module_count=1.9,
                        ),
                        "modular_full": make_variant(
                            architecture="modular",
                            credit_strategy="broadcast",
                            scenario_success_rate=0.5,
                            episode_success_rate=0.5,
                            mean_module_credit_weights={
                                "visual_cortex": 0.26,
                                "motor_cortex": 0.25,
                                "decision_cortex": 0.24,
                                "memory_cortex": 0.25,
                            },
                            module_gradient_norm_means={
                                "visual_cortex": 0.35,
                                "motor_cortex": 0.35,
                                "decision_cortex": 0.35,
                                "memory_cortex": 0.35,
                            },
                            mean_counterfactual_credit_weights={
                                "visual_cortex": 0.15,
                                "motor_cortex": 0.15,
                                "decision_cortex": 0.15,
                                "memory_cortex": 0.15,
                            },
                            dominant_module="visual_cortex",
                            mean_dominant_module_share=0.26,
                            mean_effective_module_count=4.0,
                        ),
                        "local_credit_only": make_variant(
                            architecture="modular",
                            credit_strategy="local_only",
                            scenario_success_rate=0.05,
                            episode_success_rate=0.05,
                            mean_module_credit_weights={
                                "visual_cortex": 0.0,
                                "motor_cortex": 0.0,
                                "decision_cortex": 0.0,
                                "memory_cortex": 0.0,
                            },
                            module_gradient_norm_means={
                                "visual_cortex": 0.0,
                                "motor_cortex": 0.0,
                                "decision_cortex": 0.0,
                                "memory_cortex": 0.0,
                            },
                            mean_counterfactual_credit_weights={
                                "visual_cortex": 0.0,
                                "motor_cortex": 0.0,
                                "decision_cortex": 0.0,
                                "memory_cortex": 0.0,
                            },
                            dominant_module="",
                            mean_dominant_module_share=0.0,
                            mean_effective_module_count=0.0,
                        ),
                        "counterfactual_credit": make_variant(
                            architecture="modular",
                            credit_strategy="counterfactual",
                            scenario_success_rate=0.82,
                            episode_success_rate=0.82,
                            mean_module_credit_weights={
                                "visual_cortex": 0.72,
                                "motor_cortex": 0.12,
                                "decision_cortex": 0.1,
                                "memory_cortex": 0.06,
                            },
                            module_gradient_norm_means={
                                "visual_cortex": 2.1,
                                "motor_cortex": 0.8,
                                "decision_cortex": 0.7,
                                "memory_cortex": 0.5,
                            },
                            mean_counterfactual_credit_weights={
                                "visual_cortex": 0.75,
                                "motor_cortex": 0.1,
                                "decision_cortex": 0.08,
                                "memory_cortex": 0.07,
                            },
                            dominant_module="visual_cortex",
                            mean_dominant_module_share=0.72,
                            mean_effective_module_count=1.7,
                        ),
                    },
                }
            }
        }
