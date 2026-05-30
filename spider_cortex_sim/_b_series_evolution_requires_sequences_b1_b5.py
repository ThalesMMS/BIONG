from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_checkpoint_paths import (
    b48_cerebellar_timing_checkpoint_path,
    b49_striatal_action_gate_checkpoint_path,
    b50_habit_chunking_checkpoint_path,
    b51_dopaminergic_habit_checkpoint_path,
    b52_cholinergic_precision_checkpoint_path,
    b53_noradrenergic_arousal_checkpoint_path,
    b54_serotonergic_patience_checkpoint_path,
    b55_hypothalamic_drive_checkpoint_path,
    b56_hpa_stress_axis_checkpoint_path,
    b57_insular_interoceptive_awareness_checkpoint_path,
    b58_acc_conflict_monitor_checkpoint_path,
    b59_prefrontal_goal_context_checkpoint_path,
    b60_orbitofrontal_outcome_value_checkpoint_path,
    b61_amygdala_safety_value_checkpoint_path,
)

from ._b_series_evolution_config_builders import (
    build_b1_capacity_config,
    build_b2_temporal_threat_config,
    build_b3_contact_memory_config,
    build_b4_recovery_balance_config,
)

from ._b_series_evolution_gates_b1_b6 import (
    _stats_payload,
    b1_easy_gate_result,
    b2_canonical_progress_gate_result,
    b3_canonical_robust_gate_result,
    b4_canonical_multi_gate_result,
    b4_easy_multi_gate_result,
)

from ._b_series_evolution_gates_b61_requires import (
    _make_simulation,
    ensure_b0_current_bridge_checkpoint,
    require_b1_threat_guard_checkpoint,
    require_b2_temporal_threat_checkpoint,
    require_b3_recurrent_guard_checkpoint,
)

def require_b48_cerebellar_timing_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b48_cerebellar_timing_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B49 requires the accepted B48 checkpoint at "
            f"{checkpoint}. Run B48 evolution first."
        )
    return {
        "status": "existing",
        "variant": B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b49_striatal_action_gate_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b49_striatal_action_gate_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B50 requires the accepted B49 checkpoint at "
            f"{checkpoint}. Run B49 evolution first."
        )
    return {
        "status": "existing",
        "variant": B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b50_habit_chunking_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b50_habit_chunking_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B51 requires the accepted B50 checkpoint at "
            f"{checkpoint}. Run B50 evolution first."
        )
    return {
        "status": "existing",
        "variant": B50_HABIT_CHUNKING_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b51_dopaminergic_habit_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b51_dopaminergic_habit_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B52 requires the accepted B51 checkpoint at "
            f"{checkpoint}. Run B51 evolution first."
        )
    return {
        "status": "existing",
        "variant": B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b52_cholinergic_precision_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b52_cholinergic_precision_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B53 requires the accepted B52 checkpoint at "
            f"{checkpoint}. Run B52 evolution first."
        )
    return {
        "status": "existing",
        "variant": B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b53_noradrenergic_arousal_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b53_noradrenergic_arousal_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B54 requires the accepted B53 checkpoint at "
            f"{checkpoint}. Run B53 evolution first."
        )
    return {
        "status": "existing",
        "variant": B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b54_serotonergic_patience_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b54_serotonergic_patience_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B55 requires the accepted B54 checkpoint at "
            f"{checkpoint}. Run B54 evolution first."
        )
    return {
        "status": "existing",
        "variant": B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b55_hypothalamic_drive_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b55_hypothalamic_drive_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B56 requires the accepted B55 checkpoint at "
            f"{checkpoint}. Run B55 evolution first."
        )
    return {
        "status": "existing",
        "variant": B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b56_hpa_stress_axis_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b56_hpa_stress_axis_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B57 requires the accepted B56 checkpoint at "
            f"{checkpoint}. Run B56 evolution first."
        )
    return {
        "status": "existing",
        "variant": B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b57_insular_interoceptive_awareness_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b57_insular_interoceptive_awareness_checkpoint_path(
        root=root,
        seed=seed,
    )
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B58 requires the accepted B57 checkpoint at "
            f"{checkpoint}. Run B57 evolution first."
        )
    return {
        "status": "existing",
        "variant": B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b58_acc_conflict_monitor_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b58_acc_conflict_monitor_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B59 requires the accepted B58 checkpoint at "
            f"{checkpoint}. Run B58 evolution first."
        )
    return {
        "status": "existing",
        "variant": B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b59_prefrontal_goal_context_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b59_prefrontal_goal_context_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B60 requires the accepted B59 checkpoint at "
            f"{checkpoint}. Run B59 evolution first."
        )
    return {
        "status": "existing",
        "variant": B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b60_orbitofrontal_outcome_value_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b60_orbitofrontal_outcome_value_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B61 requires the accepted B60 checkpoint at "
            f"{checkpoint}. Run B60 evolution first."
        )
    return {
        "status": "existing",
        "variant": B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def require_b61_amygdala_safety_value_checkpoint(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> dict[str, object]:
    checkpoint = b61_amygdala_safety_value_checkpoint_path(root=root, seed=seed)
    weights_path = checkpoint / "b_series_policy.npz"
    metadata_path = checkpoint / "metadata.json"
    if not weights_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "B62 requires the accepted B61 checkpoint at "
            f"{checkpoint}. Run B61 evolution first."
        )
    return {
        "status": "existing",
        "variant": B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
        "checkpoint": str(checkpoint),
    }


def run_b1_capacity_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 24,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    run_canonical: bool = True,
) -> dict[str, object]:
    config = build_b1_capacity_config(
        variant_name,
        source_checkpoint=source_checkpoint,
    )
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
    easy_stats, easy_trace = sim.run_episode(
        0,
        training=False,
        sample=False,
        capture_trace=True,
        scenario_name=B1_EASY_SCENARIO,
    )
    gate = b1_easy_gate_result(easy_stats, easy_trace)
    canonical_payload: dict[str, object] | None = None
    if run_canonical:
        canonical_stats, canonical_trace = sim.run_episode(
            1,
            training=False,
            sample=False,
            capture_trace=True,
            scenario_name=B1_CANONICAL_SCENARIO,
        )
        canonical_gate = b1_easy_gate_result(
            canonical_stats,
            canonical_trace,
            scenario_name=B1_CANONICAL_SCENARIO,
        )
        canonical_payload = {
            "metrics": _stats_payload(canonical_stats),
            "trace_contract": canonical_gate["checks"]["primitive_trace"],
            "primitive_violations": canonical_gate["primitive_violations"],
        }

    attempt_dir = Path(root) / variant_name / f"seed_{int(seed)}"
    checkpoint_name = "best" if gate["passed"] else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    transfer_report = dict(sim.brain.b_series_transfer_report or {})
    report = {
        "variant": variant_name,
        "status": gate["status"],
        "discard_reason": ", ".join(gate["failures"]) if not gate["passed"] else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": transfer_report,
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
        "easy_gate": gate,
        "easy_metrics": _stats_payload(easy_stats),
        "canonical_diagnostic": canonical_payload,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def run_b2_temporal_threat_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 48,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    config = build_b2_temporal_threat_config(
        variant_name,
        source_checkpoint=source_checkpoint,
    )
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
    easy_stats, easy_trace = sim.run_episode(
        B2_EASY_EVALUATION_EPISODE,
        training=False,
        sample=False,
        capture_trace=True,
        scenario_name=B1_EASY_SCENARIO,
    )
    easy_gate = b1_easy_gate_result(easy_stats, easy_trace)
    canonical_stats, canonical_trace = sim.run_episode(
        B2_CANONICAL_EVALUATION_EPISODE,
        training=False,
        sample=False,
        capture_trace=True,
        scenario_name=B1_CANONICAL_SCENARIO,
    )
    canonical_gate = b2_canonical_progress_gate_result(
        canonical_stats,
        canonical_trace,
    )
    accepted = bool(easy_gate["passed"] and canonical_gate["passed"])
    attempt_dir = Path(root) / variant_name / f"seed_{int(seed)}"
    checkpoint_name = "best" if accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    if not easy_gate["passed"]:
        discard_failures.append("easy:" + ",".join(easy_gate["failures"]))
    if not canonical_gate["passed"]:
        discard_failures.append("canonical:" + ",".join(canonical_gate["failures"]))
    report = {
        "variant": variant_name,
        "status": "accepted" if accepted else "discarded",
        "discard_reason": "; ".join(discard_failures) if discard_failures else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": dict(sim.brain.b_series_transfer_report or {}),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "easy_evaluation_episode": B2_EASY_EVALUATION_EPISODE,
        "canonical_evaluation_episode": B2_CANONICAL_EVALUATION_EPISODE,
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
        "easy_gate": easy_gate,
        "easy_metrics": _stats_payload(easy_stats),
        "canonical_gate": canonical_gate,
        "canonical_metrics": _stats_payload(canonical_stats),
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def run_b3_contact_memory_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 48,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    config = build_b3_contact_memory_config(
        variant_name,
        source_checkpoint=source_checkpoint,
    )
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
    easy_stats, easy_trace = sim.run_episode(
        B3_EASY_EVALUATION_EPISODE,
        training=False,
        sample=False,
        capture_trace=True,
        scenario_name=B1_EASY_SCENARIO,
    )
    easy_gate = b1_easy_gate_result(easy_stats, easy_trace)
    canonical_results = []
    canonical_passed = True
    for evaluation_episode in B3_CANONICAL_EVALUATION_EPISODES:
        canonical_stats, canonical_trace = sim.run_episode(
            int(evaluation_episode),
            training=False,
            sample=False,
            capture_trace=True,
            scenario_name=B1_CANONICAL_SCENARIO,
        )
        canonical_gate = b3_canonical_robust_gate_result(
            canonical_stats,
            canonical_trace,
            evaluation_episode=int(evaluation_episode),
        )
        canonical_passed = canonical_passed and bool(canonical_gate["passed"])
        canonical_results.append(
            {
                "evaluation_episode": int(evaluation_episode),
                "gate": canonical_gate,
                "metrics": _stats_payload(canonical_stats),
            }
        )
    accepted = bool(easy_gate["passed"] and canonical_passed)
    attempt_dir = Path(root) / variant_name / f"seed_{int(seed)}"
    checkpoint_name = "best" if accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    if not easy_gate["passed"]:
        discard_failures.append("easy:" + ",".join(easy_gate["failures"]))
    for result in canonical_results:
        gate = result["gate"]
        if isinstance(gate, dict) and not gate.get("passed", False):
            discard_failures.append(
                "canonical_ep"
                + str(result["evaluation_episode"])
                + ":"
                + ",".join(gate.get("failures", []))
            )
    report = {
        "variant": variant_name,
        "status": "accepted" if accepted else "discarded",
        "discard_reason": "; ".join(discard_failures) if discard_failures else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": dict(sim.brain.b_series_transfer_report or {}),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "easy_evaluation_episode": B3_EASY_EVALUATION_EPISODE,
        "canonical_evaluation_episodes": list(B3_CANONICAL_EVALUATION_EPISODES),
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
        "easy_gate": easy_gate,
        "easy_metrics": _stats_payload(easy_stats),
        "canonical_results": canonical_results,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _run_episode_payload(
    sim: SpiderSimulation,
    *,
    evaluation_episode: int,
    scenario_name: str,
) -> dict[str, object]:
    stats, trace = sim.run_episode(
        int(evaluation_episode),
        training=False,
        sample=False,
        capture_trace=True,
        scenario_name=scenario_name,
    )
    return {
        "evaluation_episode": int(evaluation_episode),
        "stats": stats,
        "trace": trace,
    }


def _b4_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b4_recovery_balance_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 48,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
    candidate_id: str | None = None,
    promote_if_accepted: bool = True,
) -> dict[str, object]:
    config = build_b4_recovery_balance_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        controller_profile=controller_profile,
        controller_params=controller_params,
    )
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
    easy_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B1_EASY_SCENARIO,
        )
        for episode in B4_EASY_EVALUATION_EPISODES
    ]
    easy_gate = b4_easy_multi_gate_result(easy_results)
    canonical_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B1_CANONICAL_SCENARIO,
        )
        for episode in B4_CANONICAL_EVALUATION_EPISODES
    ]
    canonical_gate = b4_canonical_multi_gate_result(canonical_results)
    accepted = bool(easy_gate["passed"] and canonical_gate["passed"])
    attempt_dir = _b4_attempt_dir(root, variant_name, seed, candidate_id)
    checkpoint_name = "best" if accepted and promote_if_accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    if not easy_gate["passed"]:
        discard_failures.extend(easy_gate["failures"])
    if not canonical_gate["passed"]:
        discard_failures.extend(canonical_gate["failures"])
    report = {
        "variant": variant_name,
        "candidate_id": candidate_id,
        "status": "accepted" if accepted else "discarded",
        "discard_reason": "; ".join(discard_failures) if discard_failures else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": dict(sim.brain.b_series_transfer_report or {}),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "controller_profile": config.b_controller_profile,
        "controller_params": dict(config.b_controller_params),
        "easy_evaluation_episodes": list(B4_EASY_EVALUATION_EPISODES),
        "canonical_evaluation_episodes": list(B4_CANONICAL_EVALUATION_EPISODES),
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
        "easy_gate": easy_gate,
        "canonical_gate": canonical_gate,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def run_b1_capacity_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b0_training_episodes: int = 24,
    b1_training_episodes: int = 24,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    force_b0: bool = False,
    run_canonical: bool = True,
) -> dict[str, object]:
    b0_report = ensure_b0_current_bridge_checkpoint(
        root=root,
        seed=seed,
        training_episodes=b0_training_episodes,
        reward_profile=reward_profile,
        operational_profile=operational_profile,
        noise_profile=noise_profile,
        force=force_b0,
    )
    source_checkpoint = str(b0_report["checkpoint"])
    attempts = []
    accepted_variant: str | None = None
    for variant_name in B1_EVOLUTION_ATTEMPTS:
        attempt = run_b1_capacity_attempt(
            variant_name,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b1_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            run_canonical=run_canonical,
        )
        attempts.append(attempt)
        if attempt["status"] == "accepted":
            accepted_variant = variant_name
            break

    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "b0_source": b0_report,
        "attempts": attempts,
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard this B1 line and try a recurrent threat-aware B1 from the same B0 source"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    (root_path / "b1_evolution_summary.json").write_text(
        summary_text,
        encoding="utf-8",
    )
    (root_path / "b1_capacity_summary.json").write_text(
        summary_text,
        encoding="utf-8",
    )
    return summary


def run_b2_temporal_threat_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b2_training_episodes: int = 48,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    b1_report = require_b1_threat_guard_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b1_report["checkpoint"])
    attempts = []
    accepted_variant: str | None = None
    for variant_name in B2_EVOLUTION_ATTEMPTS:
        attempt = run_b2_temporal_threat_attempt(
            variant_name,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b2_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
        )
        attempts.append(attempt)
        if attempt["status"] == "accepted":
            accepted_variant = variant_name
            break

    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "b1_source": b1_report,
        "canonical_baseline": dict(B2_CANONICAL_BASELINE),
        "attempts": attempts,
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard temporal-threat B2 and try a more structural B2 from the same B1 source"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b2_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def run_b3_contact_memory_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b3_training_episodes: int = 48,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    b2_report = require_b2_temporal_threat_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b2_report["checkpoint"])
    attempts = []
    accepted_variant: str | None = None
    for variant_name in B3_EVOLUTION_ATTEMPTS:
        attempt = run_b3_contact_memory_attempt(
            variant_name,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b3_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
        )
        attempts.append(attempt)
        if attempt["status"] == "accepted":
            accepted_variant = variant_name
            break

    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "b2_source": b2_report,
        "canonical_baselines": dict(B3_CANONICAL_BASELINES),
        "attempts": attempts,
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard contact-memory B3 and try explicit recurrent memory from the same B2 source"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b3_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b4_attempt_fitness(attempt: dict[str, object]) -> float:
    canonical_gate = attempt.get("canonical_gate", {})
    easy_gate = attempt.get("easy_gate", {})
    aggregate = (
        canonical_gate.get("aggregate", {})
        if isinstance(canonical_gate, dict)
        else {}
    )
    completed_horizons = int(aggregate.get("completed_horizons", 0) or 0)
    min_steps = int(aggregate.get("min_steps", 0) or 0)
    total_contacts = int(aggregate.get("total_predator_contacts", 999) or 999)
    score = completed_horizons * 1000.0 + min_steps * 3.0 - total_contacts * 20.0
    if isinstance(easy_gate, dict) and not bool(easy_gate.get("passed", False)):
        score -= 5000.0
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


B4_GENETIC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "recovery_pressure_threshold": (0.45, 0.75),
    "sleep_hunger_max": (0.66, 0.78),
    "sleep_threat_max": (0.30, 0.85),
    "exit_health_min": (0.05, 0.55),
    "exit_threat_max": (0.30, 0.80),
    "hunger_release": (0.78, 0.93),
    "emergency_hunger_release": (0.90, 0.98),
    "contact_hold_hunger_max": (0.76, 0.94),
    "return_threat_min": (0.40, 0.80),
    "return_hunger_max": (0.82, 0.96),
}


def _b4_random_params(rng: random.Random) -> dict[str, float]:
    return {
        key: round(rng.uniform(low, high), 6)
        for key, (low, high) in B4_GENETIC_PARAM_BOUNDS.items()
    }


def _b4_breed_params(
    rng: random.Random,
    parent_a: dict[str, float],
    parent_b: dict[str, float],
) -> dict[str, float]:
    child: dict[str, float] = {}
    for key, (low, high) in B4_GENETIC_PARAM_BOUNDS.items():
        value = parent_a[key] if rng.random() < 0.5 else parent_b[key]
        if rng.random() < 0.35:
            value += rng.gauss(0.0, (high - low) * 0.12)
        child[key] = round(max(low, min(high, float(value))), 6)
    return child


def run_b4_genetic_search(
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    population_size: int = 12,
    generations: int = 4,
    workers: int = 1,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    screening_training_episodes: int = 24,
    confirmation_training_episodes: int = 48,
) -> dict[str, object]:
    rng = random.Random(int(seed) + 4000)
    population = [_b4_random_params(rng) for _ in range(max(1, int(population_size)))]
    generation_reports = []
    screened_attempts: list[dict[str, object]] = []
    for generation in range(max(1, int(generations))):
        jobs = []
        for candidate_index, params in enumerate(population):
            candidate_params = dict(params)
            candidate_params["ga_generation"] = float(generation)
            candidate_params["ga_candidate"] = float(candidate_index)
            candidate_id = f"generation_{generation:02d}_candidate_{candidate_index:02d}"
            jobs.append((candidate_index, candidate_id, candidate_params))
        attempts_by_index: dict[int, dict[str, object]] = {}
        if int(workers) > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as pool:
                future_map = {
                    pool.submit(
                        run_b4_recovery_balance_attempt,
                        B4_GENETIC_RECOVERY_H48_POLICY_NAME,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=screening_training_episodes,
                        reward_profile=reward_profile,
                        operational_profile=operational_profile,
                        noise_profile=noise_profile,
                        controller_profile="genetic_recovery",
                        controller_params=params,
                        candidate_id=candidate_id,
                        promote_if_accepted=False,
                    ): candidate_index
                    for candidate_index, candidate_id, params in jobs
                }
                for future in as_completed(future_map):
                    attempts_by_index[future_map[future]] = future.result()
        else:
            for candidate_index, candidate_id, params in jobs:
                attempts_by_index[candidate_index] = run_b4_recovery_balance_attempt(
                    B4_GENETIC_RECOVERY_H48_POLICY_NAME,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=screening_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    controller_profile="genetic_recovery",
                    controller_params=params,
                    candidate_id=candidate_id,
                    promote_if_accepted=False,
                )
        ranked = sorted(
            attempts_by_index.values(),
            key=_b4_attempt_fitness,
            reverse=True,
        )
        screened_attempts.extend(ranked)
        generation_reports.append(
            {
                "generation": int(generation),
                "best_candidate_id": ranked[0].get("candidate_id") if ranked else None,
                "best_fitness": _b4_attempt_fitness(ranked[0]) if ranked else None,
                "best_status": ranked[0].get("status") if ranked else None,
            }
        )
        elites = [dict(attempt["controller_params"]) for attempt in ranked[:4]]
        if not elites:
            population = [_b4_random_params(rng) for _ in range(max(1, int(population_size)))]
            continue
        next_population = elites[:]
        while len(next_population) < max(1, int(population_size)):
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            next_population.append(_b4_breed_params(rng, parent_a, parent_b))
        population = next_population[: max(1, int(population_size))]

    finalists = sorted(screened_attempts, key=_b4_attempt_fitness, reverse=True)[:3]
    confirmed_attempts = []
    accepted_attempt: dict[str, object] | None = None
    for finalist_index, finalist in enumerate(finalists):
        params = dict(finalist.get("controller_params", {}))
        params["ga_generation"] = float(params.get("ga_generation", 0.0))
        params["ga_candidate"] = float(params.get("ga_candidate", finalist_index))
        attempt = run_b4_recovery_balance_attempt(
            B4_GENETIC_RECOVERY_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=confirmation_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_recovery",
            controller_params=params,
            candidate_id=f"finalist_{finalist_index:02d}",
            promote_if_accepted=False,
        )
        confirmed_attempts.append(attempt)
        if attempt["status"] == "accepted":
            accepted_attempt = run_b4_recovery_balance_attempt(
                B4_GENETIC_RECOVERY_H48_POLICY_NAME,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=confirmation_training_episodes,
                reward_profile=reward_profile,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                controller_profile="genetic_recovery",
                controller_params=params,
                candidate_id=None,
                promote_if_accepted=True,
            )
            break
    if accepted_attempt is None and confirmed_attempts:
        best = max(confirmed_attempts, key=_b4_attempt_fitness)
        accepted_attempt = run_b4_recovery_balance_attempt(
            B4_GENETIC_RECOVERY_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=confirmation_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_recovery",
            controller_params=dict(best.get("controller_params", {})),
            candidate_id=None,
            promote_if_accepted=True,
        )
    search_summary = {
        "status": (
            "accepted"
            if accepted_attempt is not None
            and accepted_attempt.get("status") == "accepted"
            else "discarded"
        ),
        "accepted_attempt": accepted_attempt,
        "generation_reports": generation_reports,
        "confirmed_attempts": confirmed_attempts,
        "screened_attempt_count": len(screened_attempts),
    }
    search_dir = Path(root) / B4_GENETIC_RECOVERY_H48_POLICY_NAME / f"seed_{int(seed)}"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "ga_search_report.json").write_text(
        json.dumps(search_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return search_summary


def run_b4_recovery_balance_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b4_training_episodes: int = 48,
    b4_workers: int = 1,
    b4_search: str = "hybrid",
    b4_ga_population: int = 12,
    b4_ga_generations: int = 4,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    b3_report = require_b3_recurrent_guard_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b3_report["checkpoint"])
    attempts: list[dict[str, object]] = []
    accepted_variant: str | None = None
    fixed_attempts: list[dict[str, object]] = []
    if b4_search in {"fixed", "hybrid"}:
        jobs = list(B4_FIXED_EVOLUTION_ATTEMPTS)
        attempts_by_name: dict[str, dict[str, object]] = {}
        if int(b4_workers) > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(b4_workers), len(jobs))) as pool:
                future_map = {
                    pool.submit(
                        run_b4_recovery_balance_attempt,
                        variant_name,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=b4_training_episodes,
                        reward_profile=reward_profile,
                        operational_profile=operational_profile,
                        noise_profile=noise_profile,
                    ): variant_name
                    for variant_name in jobs
                }
                for future in as_completed(future_map):
                    attempts_by_name[future_map[future]] = future.result()
        else:
            for variant_name in jobs:
                attempts_by_name[variant_name] = run_b4_recovery_balance_attempt(
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b4_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                )
        fixed_attempts = [attempts_by_name[name] for name in jobs]
        attempts.extend(fixed_attempts)
        for attempt in fixed_attempts:
            if attempt["status"] == "accepted":
                accepted_variant = str(attempt["variant"])
                break

    genetic_summary = None
    if accepted_variant is None and b4_search in {"ga", "hybrid"}:
        genetic_summary = run_b4_genetic_search(
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            population_size=b4_ga_population,
            generations=b4_ga_generations,
            workers=b4_workers,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            screening_training_episodes=max(1, int(b4_training_episodes) // 2),
            confirmation_training_episodes=b4_training_episodes,
        )
        accepted_attempt = genetic_summary.get("accepted_attempt")
        if isinstance(accepted_attempt, dict):
            attempts.append(accepted_attempt)
            if accepted_attempt.get("status") == "accepted":
                accepted_variant = str(accepted_attempt["variant"])

    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "b3_source": b3_report,
        "b4_search": b4_search,
        "b4_workers": int(b4_workers),
        "canonical_baseline": dict(B4_CANONICAL_BASELINE),
        "attempts": attempts,
        "genetic_search": genetic_summary,
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B4 recovery-balance line and try explicit recurrent state memory from the accepted B3 source"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b4_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b5_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base
