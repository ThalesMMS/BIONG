from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_checkpoint_paths import (
    b0_current_bridge_checkpoint_path,
    b10_prospective_replay_checkpoint_path,
    b11_confidence_arbiter_checkpoint_path,
    b12_predictive_attention_checkpoint_path,
    b13_local_affordance_search_checkpoint_path,
    b14_affordance_uncertainty_checkpoint_path,
    b15_option_critic_checkpoint_path,
    b16_option_ensemble_checkpoint_path,
    b17_neuromodulated_ensemble_checkpoint_path,
    b18_eligibility_trace_checkpoint_path,
    b19_episodic_meta_memory_checkpoint_path,
    b1_threat_guard_checkpoint_path,
    b20_working_memory_gate_checkpoint_path,
    b21_hippocampal_replay_checkpoint_path,
    b22_prospective_replay_checkpoint_path,
    b23_conflict_monitor_checkpoint_path,
    b24_precision_conflict_checkpoint_path,
    b25_metacognitive_confidence_checkpoint_path,
    b26_allostatic_prediction_checkpoint_path,
    b27_arousal_gain_checkpoint_path,
    b28_interoceptive_attention_checkpoint_path,
    b29_salience_competition_checkpoint_path,
    b2_temporal_threat_checkpoint_path,
    b30_basal_ganglia_gate_checkpoint_path,
    b31_dopamine_prediction_error_checkpoint_path,
    b32_actor_critic_value_checkpoint_path,
    b33_td_error_decomposition_checkpoint_path,
    b34_eligibility_credit_checkpoint_path,
    b35_forward_model_value_checkpoint_path,
    b36_latent_belief_state_checkpoint_path,
    b37_state_factor_gate_checkpoint_path,
    b38_factor_attention_checkpoint_path,
    b39_attention_binding_checkpoint_path,
    b3_recurrent_guard_checkpoint_path,
    b40_global_workspace_checkpoint_path,
    b41_executive_workspace_checkpoint_path,
    b42_error_monitor_checkpoint_path,
    b43_adaptive_precision_checkpoint_path,
    b44_thalamic_relay_checkpoint_path,
    b45_reticular_inhibition_checkpoint_path,
    b46_corticothalamic_feedback_checkpoint_path,
    b47_oscillatory_synchrony_checkpoint_path,
    b4_genetic_recovery_checkpoint_path,
    b5_genetic_homeostasis_checkpoint_path,
    b6_fused_risk_recurrent_checkpoint_path,
    b7_affordance_budget_checkpoint_path,
    b8_spatial_affordance_checkpoint_path,
    b9_waypoint_planner_checkpoint_path,
)

from ._b_series_evolution_gates_b1_b6 import (
    _stats_payload,
    b1_easy_gate_result,
    trace_uses_only_primitive_actions,
)

from ._b_series_evolution_gates_b51_b60 import (
    b60_orbitofrontal_value_corridor_gate_result,
)

def b61_amygdala_safety_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b60_orbitofrontal_value_corridor_gate_result(results)
    explicit_decision_set = set(B61_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    safety_value_episodes = 0
    threat_channel_episodes = 0
    confidence_episodes = 0
    balance_episodes = 0
    safety_lock_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        metrics = result.get("metrics", {})
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = int(
            metrics.get("predator_contacts", result.get("predator_contacts", 0)) or 0
        )
        decisions = [
            str(item.get("b61_decision"))
            for item in trace
            if item.get("b61_decision") is not None
        ]
        safety_values = [
            float(item.get("b61_safety_value", 0.0) or 0.0)
            for item in trace
            if item.get("b61_safety_value") is not None
        ]
        threat_values = [
            float(item.get("b61_threat_value", 0.0) or 0.0)
            for item in trace
            if item.get("b61_threat_value") is not None
        ]
        confidence_values = [
            float(item.get("b61_safety_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b61_safety_confidence") is not None
        ]
        balance_values = [
            float(item.get("b61_affective_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b61_affective_balance") is not None
        ]
        locks = [
            int(item.get("b61_safety_lock", 0) or 0)
            for item in trace
            if item.get("b61_safety_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        safety_value = any(abs(value) > 0.0 for value in safety_values)
        threat_channel = bool(threat_values)
        confidence = any(abs(value) > 0.0 for value in confidence_values)
        balance = any(abs(value) > 0.0 for value in balance_values)
        safety_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if safety_value:
            safety_value_episodes += 1
        if threat_channel:
            threat_channel_episodes += 1
        if confidence:
            confidence_episodes += 1
        if balance:
            balance_episodes += 1
        if safety_lock:
            safety_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b61_decision": bool(explicit_decision),
                    "safety_value": bool(safety_value),
                    "threat_channel": bool(threat_channel),
                    "safety_confidence": bool(confidence),
                    "affective_balance": bool(balance),
                    "safety_lock": bool(safety_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "safety_values": safety_values,
                "threat_values": threat_values,
                "safety_confidences": confidence_values,
                "affective_balances": balance_values,
                "safety_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b60_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b61_decision_episodes": explicit_decision_episodes >= 2,
        "safety_value_episodes": safety_value_episodes >= 2,
        "threat_channel_episodes": threat_channel_episodes >= 2,
        "safety_confidence_episodes": confidence_episodes >= 2,
        "affective_balance_episodes": balance_episodes >= 2,
        "safety_lock_episodes": safety_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b61_aggregate:" + name
        for name, ok in aggregate_checks.items()
        if not ok
    )
    passed = not failures
    return {
        "scenario": B6_CORRIDOR_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "base_gate": base_gate,
        "aggregate": {
            "base_b60_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "safety_value_episodes": int(safety_value_episodes),
            "threat_channel_episodes": int(threat_channel_episodes),
            "safety_confidence_episodes": int(confidence_episodes),
            "affective_balance_episodes": int(balance_episodes),
            "safety_lock_episodes": int(safety_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b62_defensive_mode_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b61_amygdala_safety_corridor_gate_result(results)
    explicit_decision_set = set(B62_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    defensive_mode_episodes = 0
    pressure_episodes = 0
    shelter_bias_episodes = 0
    balance_episodes = 0
    lock_or_safe_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        metrics = result.get("metrics", {})
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = int(
            metrics.get("predator_contacts", result.get("predator_contacts", 0)) or 0
        )
        decisions = [
            str(item.get("b62_decision"))
            for item in trace
            if item.get("b62_decision") is not None
        ]
        modes = [
            str(item.get("b62_defensive_mode"))
            for item in trace
            if item.get("b62_defensive_mode") is not None
        ]
        freeze_values = [
            float(item.get("b62_freeze_pressure", 0.0) or 0.0)
            for item in trace
            if item.get("b62_freeze_pressure") is not None
        ]
        flee_values = [
            float(item.get("b62_flee_pressure", 0.0) or 0.0)
            for item in trace
            if item.get("b62_flee_pressure") is not None
        ]
        shelter_values = [
            float(item.get("b62_shelter_bias", 0.0) or 0.0)
            for item in trace
            if item.get("b62_shelter_bias") is not None
        ]
        balance_values = [
            float(item.get("b62_defense_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b62_defense_balance") is not None
        ]
        locks = [
            int(item.get("b62_defense_lock", 0) or 0)
            for item in trace
            if item.get("b62_defense_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        defensive_mode = any(mode not in {"", "None", "preserve"} for mode in modes)
        pressure = any(abs(value) > 0.0 for value in freeze_values + flee_values)
        shelter_bias = any(abs(value) > 0.0 for value in shelter_values)
        balance = any(abs(value) > 0.0 for value in balance_values)
        lock_or_safe = any(lock > 0 for lock in locks) or "defensive_safe_advance" in decisions
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if defensive_mode:
            defensive_mode_episodes += 1
        if pressure:
            pressure_episodes += 1
        if shelter_bias:
            shelter_bias_episodes += 1
        if balance:
            balance_episodes += 1
        if lock_or_safe:
            lock_or_safe_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b62_decision": bool(explicit_decision),
                    "defensive_mode": bool(defensive_mode),
                    "defense_pressure": bool(pressure),
                    "shelter_bias": bool(shelter_bias),
                    "defense_balance": bool(balance),
                    "lock_or_safe_commit": bool(lock_or_safe),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "defensive_modes": modes,
                "freeze_pressures": freeze_values,
                "flee_pressures": flee_values,
                "shelter_biases": shelter_values,
                "defense_balances": balance_values,
                "defense_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b61_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b62_decision_episodes": explicit_decision_episodes >= 2,
        "defensive_mode_episodes": defensive_mode_episodes >= 2,
        "defense_pressure_episodes": pressure_episodes >= 2,
        "shelter_bias_episodes": shelter_bias_episodes >= 2,
        "defense_balance_episodes": balance_episodes >= 2,
        "lock_or_safe_episodes": lock_or_safe_episodes >= 2,
    }
    failures.extend(
        "corridor_b62_aggregate:" + name
        for name, ok in aggregate_checks.items()
        if not ok
    )
    passed = not failures
    return {
        "scenario": B6_CORRIDOR_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "base_gate": base_gate,
        "aggregate": {
            "base_b61_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "defensive_mode_episodes": int(defensive_mode_episodes),
            "defense_pressure_episodes": int(pressure_episodes),
            "shelter_bias_episodes": int(shelter_bias_episodes),
            "defense_balance_episodes": int(balance_episodes),
            "lock_or_safe_episodes": int(lock_or_safe_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def _make_simulation(
    *,
    config: BrainAblationConfig,
    seed: int,
    reward_profile: str,
    operational_profile: str,
    noise_profile: str,
) -> SpiderSimulation:
    return SpiderSimulation(
        seed=int(seed),
        reward_profile=reward_profile,
        operational_profile=operational_profile,
        noise_profile=noise_profile,
        brain_config=config,
    )


def ensure_b0_current_bridge_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 24,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    force: bool = False,
) -> dict[str, object]:
    checkpoint = b0_current_bridge_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    report_path = checkpoint.parent / "source_report.json"
    if not force and weights_path.exists() and metadata_path.exists():
        return {
            "status": "existing",
            "variant": B0_CURRENT_BRIDGE_POLICY_NAME,
            "checkpoint": str(checkpoint),
            "report": str(report_path) if report_path.exists() else None,
        }

    config = resolve_ablation_configs([B0_CURRENT_BRIDGE_POLICY_NAME])[0]
    sim = _make_simulation(
        config=config,
        seed=seed,
        reward_profile=reward_profile,
        operational_profile=operational_profile,
        noise_profile=noise_profile,
    )
    if int(training_episodes) > 0:
        sim.train(
            int(training_episodes),
            evaluation_episodes=0,
            capture_evaluation_trace=False,
        )
    stats, trace = sim.run_episode(
        0,
        training=False,
        sample=False,
        capture_trace=True,
        scenario_name=B1_EASY_SCENARIO,
    )
    sim.brain.save(checkpoint)
    gate = b1_easy_gate_result(stats, trace)
    report = {
        "status": "created",
        "variant": B0_CURRENT_BRIDGE_POLICY_NAME,
        "checkpoint": str(checkpoint),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
        "easy_metrics": _stats_payload(stats),
        "trace_contract": gate["checks"]["primitive_trace"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def require_b1_threat_guard_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b1_threat_guard_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B2 requires the accepted B1 checkpoint at "
            f"{checkpoint}. Run B1 evolution first."
        )
    return {
        "status": "existing",
        "variant": B1_THREAT_GUARD_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b2_temporal_threat_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b2_temporal_threat_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B3 requires the accepted B2 checkpoint at "
            f"{checkpoint}. Run B2 evolution first."
        )
    return {
        "status": "existing",
        "variant": B2_TEMPORAL_THREAT_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b3_recurrent_guard_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b3_recurrent_guard_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B4 requires the accepted B3 checkpoint at "
            f"{checkpoint}. Run B3 evolution first."
        )
    return {
        "status": "existing",
        "variant": B3_RECURRENT_GUARD_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b4_genetic_recovery_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b4_genetic_recovery_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B5 requires the accepted B4 checkpoint at "
            f"{checkpoint}. Run B4 evolution first."
        )
    return {
        "status": "existing",
        "variant": B4_GENETIC_RECOVERY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b5_genetic_homeostasis_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b5_genetic_homeostasis_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B6 requires the accepted B5 checkpoint at "
            f"{checkpoint}. Run B5 evolution first."
        )
    return {
        "status": "existing",
        "variant": B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b6_fused_risk_recurrent_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b6_fused_risk_recurrent_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B7 requires the accepted B6 fused checkpoint at "
            f"{checkpoint}. Run B6 evolution first."
        )
    return {
        "status": "existing",
        "variant": B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b7_affordance_budget_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b7_affordance_budget_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B8 requires the accepted B7 checkpoint at "
            f"{checkpoint}. Run B7 evolution first."
        )
    return {
        "status": "existing",
        "variant": B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b8_spatial_affordance_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b8_spatial_affordance_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B9 requires the accepted B8 checkpoint at "
            f"{checkpoint}. Run B8 evolution first."
        )
    return {
        "status": "existing",
        "variant": B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b9_waypoint_planner_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b9_waypoint_planner_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B10 requires the accepted B9 checkpoint at "
            f"{checkpoint}. Run B9 evolution first."
        )
    return {
        "status": "existing",
        "variant": B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b10_prospective_replay_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b10_prospective_replay_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B11 requires the accepted B10 checkpoint at "
            f"{checkpoint}. Run B10 evolution first."
        )
    return {
        "status": "existing",
        "variant": B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b11_confidence_arbiter_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b11_confidence_arbiter_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B12 requires the accepted B11 checkpoint at "
            f"{checkpoint}. Run B11 evolution first."
        )
    return {
        "status": "existing",
        "variant": B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b12_predictive_attention_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b12_predictive_attention_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B13 requires the accepted B12 checkpoint at "
            f"{checkpoint}. Run B12 evolution first."
        )
    return {
        "status": "existing",
        "variant": B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b13_local_affordance_search_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b13_local_affordance_search_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B14 requires the accepted B13 checkpoint at "
            f"{checkpoint}. Run B13 evolution first."
        )
    return {
        "status": "existing",
        "variant": B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b14_affordance_uncertainty_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b14_affordance_uncertainty_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B15 requires the accepted B14 checkpoint at "
            f"{checkpoint}. Run B14 evolution first."
        )
    return {
        "status": "existing",
        "variant": B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b15_option_critic_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b15_option_critic_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B16 requires the accepted B15 checkpoint at "
            f"{checkpoint}. Run B15 evolution first."
        )
    return {
        "status": "existing",
        "variant": B15_OPTION_CRITIC_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b16_option_ensemble_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b16_option_ensemble_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B17 requires the accepted B16 checkpoint at "
            f"{checkpoint}. Run B16 evolution first."
        )
    return {
        "status": "existing",
        "variant": B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b17_neuromodulated_ensemble_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b17_neuromodulated_ensemble_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B18 requires the accepted B17 checkpoint at "
            f"{checkpoint}. Run B17 evolution first."
        )
    return {
        "status": "existing",
        "variant": B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b18_eligibility_trace_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b18_eligibility_trace_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B19 requires the accepted B18 checkpoint at "
            f"{checkpoint}. Run B18 evolution first."
        )
    return {
        "status": "existing",
        "variant": B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b19_episodic_meta_memory_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b19_episodic_meta_memory_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B20 requires the accepted B19 checkpoint at "
            f"{checkpoint}. Run B19 evolution first."
        )
    return {
        "status": "existing",
        "variant": B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b20_working_memory_gate_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b20_working_memory_gate_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B21 requires the accepted B20 checkpoint at "
            f"{checkpoint}. Run B20 evolution first."
        )
    return {
        "status": "existing",
        "variant": B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b21_hippocampal_replay_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b21_hippocampal_replay_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B22 requires the accepted B21 checkpoint at "
            f"{checkpoint}. Run B21 evolution first."
        )
    return {
        "status": "existing",
        "variant": B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b22_prospective_replay_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b22_prospective_replay_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B23 requires the accepted B22 checkpoint at "
            f"{checkpoint}. Run B22 evolution first."
        )
    return {
        "status": "existing",
        "variant": B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b23_conflict_monitor_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b23_conflict_monitor_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B24 requires the accepted B23 checkpoint at "
            f"{checkpoint}. Run B23 evolution first."
        )
    return {
        "status": "existing",
        "variant": B23_CONFLICT_MONITOR_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b24_precision_conflict_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b24_precision_conflict_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B25 requires the accepted B24 checkpoint at "
            f"{checkpoint}. Run B24 evolution first."
        )
    return {
        "status": "existing",
        "variant": B24_PRECISION_CONFLICT_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b25_metacognitive_confidence_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b25_metacognitive_confidence_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B26 requires the accepted B25 checkpoint at "
            f"{checkpoint}. Run B25 evolution first."
        )
    return {
        "status": "existing",
        "variant": B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b26_allostatic_prediction_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b26_allostatic_prediction_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B27 requires the accepted B26 checkpoint at "
            f"{checkpoint}. Run B26 evolution first."
        )
    return {
        "status": "existing",
        "variant": B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b27_arousal_gain_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b27_arousal_gain_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B28 requires the accepted B27 checkpoint at "
            f"{checkpoint}. Run B27 evolution first."
        )
    return {
        "status": "existing",
        "variant": B27_AROUSAL_GAIN_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b28_interoceptive_attention_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b28_interoceptive_attention_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B29 requires the accepted B28 checkpoint at "
            f"{checkpoint}. Run B28 evolution first."
        )
    return {
        "status": "existing",
        "variant": B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b29_salience_competition_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b29_salience_competition_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B30 requires the accepted B29 checkpoint at "
            f"{checkpoint}. Run B29 evolution first."
        )
    return {
        "status": "existing",
        "variant": B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b30_basal_ganglia_gate_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b30_basal_ganglia_gate_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B31 requires the accepted B30 checkpoint at "
            f"{checkpoint}. Run B30 evolution first."
        )
    return {
        "status": "existing",
        "variant": B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b31_dopamine_prediction_error_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b31_dopamine_prediction_error_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B32 requires the accepted B31 checkpoint at "
            f"{checkpoint}. Run B31 evolution first."
        )
    return {
        "status": "existing",
        "variant": B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b32_actor_critic_value_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b32_actor_critic_value_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B33 requires the accepted B32 checkpoint at "
            f"{checkpoint}. Run B32 evolution first."
        )
    return {
        "status": "existing",
        "variant": B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b33_td_error_decomposition_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b33_td_error_decomposition_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B34 requires the accepted B33 checkpoint at "
            f"{checkpoint}. Run B33 evolution first."
        )
    return {
        "status": "existing",
        "variant": B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b34_eligibility_credit_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b34_eligibility_credit_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B35 requires the accepted B34 checkpoint at "
            f"{checkpoint}. Run B34 evolution first."
        )
    return {
        "status": "existing",
        "variant": B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b35_forward_model_value_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b35_forward_model_value_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B36 requires the accepted B35 checkpoint at "
            f"{checkpoint}. Run B35 evolution first."
        )
    return {
        "status": "existing",
        "variant": B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b36_latent_belief_state_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b36_latent_belief_state_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B37 requires the accepted B36 checkpoint at "
            f"{checkpoint}. Run B36 evolution first."
        )
    return {
        "status": "existing",
        "variant": B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b37_state_factor_gate_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b37_state_factor_gate_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B38 requires the accepted B37 checkpoint at "
            f"{checkpoint}. Run B37 evolution first."
        )
    return {
        "status": "existing",
        "variant": B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b38_factor_attention_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b38_factor_attention_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B39 requires the accepted B38 checkpoint at "
            f"{checkpoint}. Run B38 evolution first."
        )
    return {
        "status": "existing",
        "variant": B38_FACTOR_ATTENTION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b39_attention_binding_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b39_attention_binding_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B40 requires the accepted B39 checkpoint at "
            f"{checkpoint}. Run B39 evolution first."
        )
    return {
        "status": "existing",
        "variant": B39_ATTENTION_BINDING_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b40_global_workspace_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b40_global_workspace_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B41 requires the accepted B40 checkpoint at "
            f"{checkpoint}. Run B40 evolution first."
        )
    return {
        "status": "existing",
        "variant": B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b41_executive_workspace_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b41_executive_workspace_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B42 requires the accepted B41 checkpoint at "
            f"{checkpoint}. Run B41 evolution first."
        )
    return {
        "status": "existing",
        "variant": B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b42_error_monitor_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b42_error_monitor_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B43 requires the accepted B42 checkpoint at "
            f"{checkpoint}. Run B42 evolution first."
        )
    return {
        "status": "existing",
        "variant": B42_ERROR_MONITOR_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b43_adaptive_precision_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b43_adaptive_precision_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B44 requires the accepted B43 checkpoint at "
            f"{checkpoint}. Run B43 evolution first."
        )
    return {
        "status": "existing",
        "variant": B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b44_thalamic_relay_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b44_thalamic_relay_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B45 requires the accepted B44 checkpoint at "
            f"{checkpoint}. Run B44 evolution first."
        )
    return {
        "status": "existing",
        "variant": B44_THALAMIC_RELAY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b45_reticular_inhibition_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b45_reticular_inhibition_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B46 requires the accepted B45 checkpoint at "
            f"{checkpoint}. Run B45 evolution first."
        )
    return {
        "status": "existing",
        "variant": B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b46_corticothalamic_feedback_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b46_corticothalamic_feedback_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B47 requires the accepted B46 checkpoint at "
            f"{checkpoint}. Run B46 evolution first."
        )
    return {
        "status": "existing",
        "variant": B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b47_oscillatory_synchrony_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b47_oscillatory_synchrony_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B48 requires the accepted B47 checkpoint at "
            f"{checkpoint}. Run B47 evolution first."
        )
    return {
        "status": "existing",
        "variant": B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }
