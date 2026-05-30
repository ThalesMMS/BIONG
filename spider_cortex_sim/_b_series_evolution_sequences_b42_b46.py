from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_config_builders import (
    build_b43_adaptive_precision_config,
    build_b44_thalamic_relay_config,
    build_b45_reticular_inhibition_config,
    build_b46_corticothalamic_feedback_config,
)

from ._b_series_evolution_gates_b1_b6 import (
    b4_canonical_multi_gate_result,
    b4_easy_multi_gate_result,
    b5_food_deprivation_gate_result,
    b5_sleep_conflict_gate_result,
    b6_food_predator_conflict_gate_result,
)

from ._b_series_evolution_gates_b41_b50 import (
    b43_adaptive_precision_corridor_gate_result,
    b44_thalamic_relay_corridor_gate_result,
    b45_reticular_inhibition_corridor_gate_result,
    b46_corticothalamic_feedback_corridor_gate_result,
)

from ._b_series_evolution_gates_b61_requires import (
    _make_simulation,
    require_b41_executive_workspace_checkpoint,
    require_b42_error_monitor_checkpoint,
    require_b43_adaptive_precision_checkpoint,
    require_b44_thalamic_relay_checkpoint,
)

from ._b_series_evolution_requires_sequences_b1_b5 import (
    _run_episode_payload,
)

from ._b_series_evolution_sequences_b38_b42 import (
    _b42_attempt_fitness,
    run_b42_error_monitor_attempt,
)

def run_b42_error_monitor_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b42_training_episodes: int = 64,
    b42_workers: int = 1,
    b42_search: str = "hybrid",
    b42_ga_population: int = 24,
    b42_ga_generations: int = 8,
    b42_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b42_ga_population, b42_ga_generations, b42_finalists
    b41_report = require_b41_executive_workspace_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b41_report["checkpoint"])
    jobs = list(B42_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b42_search in {"fixed", "hybrid"} and int(b42_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b42_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b42_error_monitor_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b42_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b42_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b42_error_monitor_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b42_training_episodes,
                reward_profile=reward_profile,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                promote_if_accepted=False,
            )
    fixed_attempts = [attempts_by_name[name] for name in jobs if name in attempts_by_name]
    selected = None
    for attempt in fixed_attempts:
        if attempt.get("status") == "accepted":
            selected = attempt
            break
    fallback_attempt = None
    if selected is None and b42_search in {"ga", "hybrid"}:
        fallback_attempt = run_b42_error_monitor_attempt(
            B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b42_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_error_monitor",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b42_error_monitor_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b42_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile=str(selected.get("controller_profile")),
            controller_params=dict(selected.get("controller_params", {})),
            promote_if_accepted=True,
        )
    accepted_variant = (
        str(promoted_attempt["variant"])
        if isinstance(promoted_attempt, dict)
        and promoted_attempt.get("status") == "accepted"
        else None
    )
    attempts = [*fixed_attempts]
    if fallback_attempt is not None:
        attempts.append(fallback_attempt)
    if promoted_attempt is not None and promoted_attempt not in attempts:
        attempts.append(promoted_attempt)
    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "accepted_checkpoint": (
            promoted_attempt.get("checkpoint")
            if isinstance(promoted_attempt, dict)
            else None
        ),
        "b41_source": b41_report,
        "b42_search": b42_search,
        "b42_workers": int(b42_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b42_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B42 error-monitor line and try explicit ACC control heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b42_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b43_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b43_adaptive_precision_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 64,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
    candidate_id: str | None = None,
    promote_if_accepted: bool = True,
) -> dict[str, object]:
    config = build_b43_adaptive_precision_config(
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
        _run_episode_payload(sim, evaluation_episode=episode, scenario_name=B1_EASY_SCENARIO)
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
    food_deprivation_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_FOOD_DEPRIVATION_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    food_deprivation_gate = b5_food_deprivation_gate_result(food_deprivation_results)
    sleep_conflict_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_SLEEP_CONFLICT_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    sleep_conflict_gate = b5_sleep_conflict_gate_result(sleep_conflict_results)
    food_predator_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_FOOD_PREDATOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    food_predator_gate = b6_food_predator_conflict_gate_result(food_predator_results)
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b43_adaptive_precision_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b43_attempt_dir(root, variant_name, seed, candidate_id)
    checkpoint_name = "best" if accepted and promote_if_accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    for gate in (
        easy_gate,
        canonical_gate,
        food_deprivation_gate,
        sleep_conflict_gate,
        food_predator_gate,
        corridor_gate,
    ):
        if not gate["passed"]:
            discard_failures.extend(gate["failures"])
    report = {
        "variant": variant_name,
        "candidate_id": candidate_id,
        "status": "accepted" if accepted else "discarded",
        "promoted": bool(accepted and promote_if_accepted),
        "discard_reason": "; ".join(discard_failures) if discard_failures else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": dict(sim.brain.b_series_transfer_report or {}),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "controller_profile": config.b_controller_profile,
        "controller_params": dict(config.b_controller_params),
        "easy_gate": easy_gate,
        "canonical_gate": canonical_gate,
        "food_deprivation_gate": food_deprivation_gate,
        "sleep_conflict_gate": sleep_conflict_gate,
        "food_predator_gate": food_predator_gate,
        "corridor_gate": corridor_gate,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _b43_attempt_fitness(attempt: dict[str, object]) -> float:
    corridor_gate = attempt.get("corridor_gate", {})
    corridor_aggregate = (
        corridor_gate.get("aggregate", {}) if isinstance(corridor_gate, dict) else {}
    )
    canonical_gate = attempt.get("canonical_gate", {})
    canonical_aggregate = (
        canonical_gate.get("aggregate", {})
        if isinstance(canonical_gate, dict)
        else {}
    )
    score = (
        int(corridor_aggregate.get("explicit_decision_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("precision_signal_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("adaptive_threshold_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("arousal_context_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("control_stability_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("precision_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b43_adaptive_precision_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b43_training_episodes: int = 64,
    b43_workers: int = 1,
    b43_search: str = "hybrid",
    b43_ga_population: int = 24,
    b43_ga_generations: int = 8,
    b43_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b43_ga_population, b43_ga_generations, b43_finalists
    b42_report = require_b42_error_monitor_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b42_report["checkpoint"])
    jobs = list(B43_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b43_search in {"fixed", "hybrid"} and int(b43_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b43_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b43_adaptive_precision_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b43_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b43_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b43_adaptive_precision_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b43_training_episodes,
                reward_profile=reward_profile,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                promote_if_accepted=False,
            )
    fixed_attempts = [attempts_by_name[name] for name in jobs if name in attempts_by_name]
    selected = None
    for attempt in fixed_attempts:
        if attempt.get("status") == "accepted":
            selected = attempt
            break
    fallback_attempt = None
    if selected is None and b43_search in {"ga", "hybrid"}:
        fallback_attempt = run_b43_adaptive_precision_attempt(
            B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b43_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_adaptive_precision",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b43_adaptive_precision_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b43_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile=str(selected.get("controller_profile")),
            controller_params=dict(selected.get("controller_params", {})),
            promote_if_accepted=True,
        )
    accepted_variant = (
        str(promoted_attempt["variant"])
        if isinstance(promoted_attempt, dict)
        and promoted_attempt.get("status") == "accepted"
        else None
    )
    attempts = [*fixed_attempts]
    if fallback_attempt is not None:
        attempts.append(fallback_attempt)
    if promoted_attempt is not None and promoted_attempt not in attempts:
        attempts.append(promoted_attempt)
    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "accepted_checkpoint": (
            promoted_attempt.get("checkpoint")
            if isinstance(promoted_attempt, dict)
            else None
        ),
        "b42_source": b42_report,
        "b43_search": b43_search,
        "b43_workers": int(b43_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b43_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B43 adaptive-precision line and try explicit gain heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b43_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b44_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b44_thalamic_relay_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 64,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
    candidate_id: str | None = None,
    promote_if_accepted: bool = True,
) -> dict[str, object]:
    config = build_b44_thalamic_relay_config(
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
        _run_episode_payload(sim, evaluation_episode=episode, scenario_name=B1_EASY_SCENARIO)
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
    food_deprivation_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_FOOD_DEPRIVATION_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    food_deprivation_gate = b5_food_deprivation_gate_result(food_deprivation_results)
    sleep_conflict_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_SLEEP_CONFLICT_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    sleep_conflict_gate = b5_sleep_conflict_gate_result(sleep_conflict_results)
    food_predator_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_FOOD_PREDATOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    food_predator_gate = b6_food_predator_conflict_gate_result(food_predator_results)
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b44_thalamic_relay_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b44_attempt_dir(root, variant_name, seed, candidate_id)
    checkpoint_name = "best" if accepted and promote_if_accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    for gate in (
        easy_gate,
        canonical_gate,
        food_deprivation_gate,
        sleep_conflict_gate,
        food_predator_gate,
        corridor_gate,
    ):
        if not gate["passed"]:
            discard_failures.extend(gate["failures"])
    report = {
        "variant": variant_name,
        "candidate_id": candidate_id,
        "status": "accepted" if accepted else "discarded",
        "promoted": bool(accepted and promote_if_accepted),
        "discard_reason": "; ".join(discard_failures) if discard_failures else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": dict(sim.brain.b_series_transfer_report or {}),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "controller_profile": config.b_controller_profile,
        "controller_params": dict(config.b_controller_params),
        "easy_gate": easy_gate,
        "canonical_gate": canonical_gate,
        "food_deprivation_gate": food_deprivation_gate,
        "sleep_conflict_gate": sleep_conflict_gate,
        "food_predator_gate": food_predator_gate,
        "corridor_gate": corridor_gate,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _b44_attempt_fitness(attempt: dict[str, object]) -> float:
    corridor_gate = attempt.get("corridor_gate", {})
    corridor_aggregate = (
        corridor_gate.get("aggregate", {}) if isinstance(corridor_gate, dict) else {}
    )
    canonical_gate = attempt.get("canonical_gate", {})
    canonical_aggregate = (
        canonical_gate.get("aggregate", {})
        if isinstance(canonical_gate, dict)
        else {}
    )
    score = (
        int(corridor_aggregate.get("explicit_decision_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("relay_gate_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("sensory_precision_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("context_relay_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("gate_stability_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("relay_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b44_thalamic_relay_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b44_training_episodes: int = 64,
    b44_workers: int = 1,
    b44_search: str = "hybrid",
    b44_ga_population: int = 24,
    b44_ga_generations: int = 8,
    b44_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b44_ga_population, b44_ga_generations, b44_finalists
    b43_report = require_b43_adaptive_precision_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b43_report["checkpoint"])
    jobs = list(B44_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b44_search in {"fixed", "hybrid"} and int(b44_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b44_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b44_thalamic_relay_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b44_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b44_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b44_thalamic_relay_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b44_training_episodes,
                reward_profile=reward_profile,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                promote_if_accepted=False,
            )
    fixed_attempts = [attempts_by_name[name] for name in jobs if name in attempts_by_name]
    selected = None
    for attempt in fixed_attempts:
        if attempt.get("status") == "accepted":
            selected = attempt
            break
    fallback_attempt = None
    if selected is None and b44_search in {"ga", "hybrid"}:
        fallback_attempt = run_b44_thalamic_relay_attempt(
            B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b44_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_thalamic_relay",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b44_thalamic_relay_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b44_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile=str(selected.get("controller_profile")),
            controller_params=dict(selected.get("controller_params", {})),
            promote_if_accepted=True,
        )
    accepted_variant = (
        str(promoted_attempt["variant"])
        if isinstance(promoted_attempt, dict)
        and promoted_attempt.get("status") == "accepted"
        else None
    )
    attempts = [*fixed_attempts]
    if fallback_attempt is not None:
        attempts.append(fallback_attempt)
    if promoted_attempt is not None and promoted_attempt not in attempts:
        attempts.append(promoted_attempt)
    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "accepted_checkpoint": (
            promoted_attempt.get("checkpoint")
            if isinstance(promoted_attempt, dict)
            else None
        ),
        "b43_source": b43_report,
        "b44_search": b44_search,
        "b44_workers": int(b44_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b44_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B44 thalamic-relay line and try explicit relay heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b44_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b45_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b45_reticular_inhibition_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 64,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
    candidate_id: str | None = None,
    promote_if_accepted: bool = True,
) -> dict[str, object]:
    config = build_b45_reticular_inhibition_config(
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
        _run_episode_payload(sim, evaluation_episode=episode, scenario_name=B1_EASY_SCENARIO)
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
    food_deprivation_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_FOOD_DEPRIVATION_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    food_deprivation_gate = b5_food_deprivation_gate_result(food_deprivation_results)
    sleep_conflict_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_SLEEP_CONFLICT_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    sleep_conflict_gate = b5_sleep_conflict_gate_result(sleep_conflict_results)
    food_predator_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_FOOD_PREDATOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    food_predator_gate = b6_food_predator_conflict_gate_result(food_predator_results)
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b45_reticular_inhibition_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b45_attempt_dir(root, variant_name, seed, candidate_id)
    checkpoint_name = "best" if accepted and promote_if_accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    for gate in (
        easy_gate,
        canonical_gate,
        food_deprivation_gate,
        sleep_conflict_gate,
        food_predator_gate,
        corridor_gate,
    ):
        if not gate["passed"]:
            discard_failures.extend(gate["failures"])
    report = {
        "variant": variant_name,
        "candidate_id": candidate_id,
        "status": "accepted" if accepted else "discarded",
        "promoted": bool(accepted and promote_if_accepted),
        "discard_reason": "; ".join(discard_failures) if discard_failures else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": dict(sim.brain.b_series_transfer_report or {}),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "controller_profile": config.b_controller_profile,
        "controller_params": dict(config.b_controller_params),
        "easy_gate": easy_gate,
        "canonical_gate": canonical_gate,
        "food_deprivation_gate": food_deprivation_gate,
        "sleep_conflict_gate": sleep_conflict_gate,
        "food_predator_gate": food_predator_gate,
        "corridor_gate": corridor_gate,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _b45_attempt_fitness(attempt: dict[str, object]) -> float:
    corridor_gate = attempt.get("corridor_gate", {})
    corridor_aggregate = (
        corridor_gate.get("aggregate", {}) if isinstance(corridor_gate, dict) else {}
    )
    canonical_gate = attempt.get("canonical_gate", {})
    canonical_aggregate = (
        canonical_gate.get("aggregate", {})
        if isinstance(canonical_gate, dict)
        else {}
    )
    score = (
        int(corridor_aggregate.get("explicit_decision_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("inhibitory_gate_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("sensory_filter_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("context_suppression_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("loop_stability_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("inhibition_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b45_reticular_inhibition_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b45_training_episodes: int = 64,
    b45_workers: int = 1,
    b45_search: str = "hybrid",
    b45_ga_population: int = 24,
    b45_ga_generations: int = 8,
    b45_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b45_ga_population, b45_ga_generations, b45_finalists
    b44_report = require_b44_thalamic_relay_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b44_report["checkpoint"])
    jobs = list(B45_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b45_search in {"fixed", "hybrid"} and int(b45_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b45_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b45_reticular_inhibition_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b45_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b45_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b45_reticular_inhibition_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b45_training_episodes,
                reward_profile=reward_profile,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                promote_if_accepted=False,
            )
    fixed_attempts = [attempts_by_name[name] for name in jobs if name in attempts_by_name]
    selected = None
    for attempt in fixed_attempts:
        if attempt.get("status") == "accepted":
            selected = attempt
            break
    fallback_attempt = None
    if selected is None and b45_search in {"ga", "hybrid"}:
        fallback_attempt = run_b45_reticular_inhibition_attempt(
            B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b45_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_reticular_inhibition",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b45_reticular_inhibition_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b45_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile=str(selected.get("controller_profile")),
            controller_params=dict(selected.get("controller_params", {})),
            promote_if_accepted=True,
        )
    accepted_variant = (
        str(promoted_attempt["variant"])
        if isinstance(promoted_attempt, dict)
        and promoted_attempt.get("status") == "accepted"
        else None
    )
    attempts = [*fixed_attempts]
    if fallback_attempt is not None:
        attempts.append(fallback_attempt)
    if promoted_attempt is not None and promoted_attempt not in attempts:
        attempts.append(promoted_attempt)
    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "accepted_checkpoint": (
            promoted_attempt.get("checkpoint")
            if isinstance(promoted_attempt, dict)
            else None
        ),
        "b44_source": b44_report,
        "b45_search": b45_search,
        "b45_workers": int(b45_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b45_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B45 reticular-inhibition line and try explicit inhibitory heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b45_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b46_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b46_corticothalamic_feedback_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 64,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
    candidate_id: str | None = None,
    promote_if_accepted: bool = True,
) -> dict[str, object]:
    config = build_b46_corticothalamic_feedback_config(
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
        _run_episode_payload(sim, evaluation_episode=episode, scenario_name=B1_EASY_SCENARIO)
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
    food_deprivation_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_FOOD_DEPRIVATION_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    food_deprivation_gate = b5_food_deprivation_gate_result(food_deprivation_results)
    sleep_conflict_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B5_SLEEP_CONFLICT_SCENARIO,
        )
        for episode in B5_PROBE_EVALUATION_EPISODES
    ]
    sleep_conflict_gate = b5_sleep_conflict_gate_result(sleep_conflict_results)
    food_predator_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_FOOD_PREDATOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    food_predator_gate = b6_food_predator_conflict_gate_result(food_predator_results)
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b46_corticothalamic_feedback_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b46_attempt_dir(root, variant_name, seed, candidate_id)
    checkpoint_name = "best" if accepted and promote_if_accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    for gate in (
        easy_gate,
        canonical_gate,
        food_deprivation_gate,
        sleep_conflict_gate,
        food_predator_gate,
        corridor_gate,
    ):
        if not gate["passed"]:
            discard_failures.extend(gate["failures"])
    report = {
        "variant": variant_name,
        "candidate_id": candidate_id,
        "status": "accepted" if accepted else "discarded",
        "promoted": bool(accepted and promote_if_accepted),
        "discard_reason": "; ".join(discard_failures) if discard_failures else None,
        "checkpoint": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint),
        "transfer": dict(sim.brain.b_series_transfer_report or {}),
        "seed": int(seed),
        "training_episodes": int(training_episodes),
        "controller_profile": config.b_controller_profile,
        "controller_params": dict(config.b_controller_params),
        "easy_gate": easy_gate,
        "canonical_gate": canonical_gate,
        "food_deprivation_gate": food_deprivation_gate,
        "sleep_conflict_gate": sleep_conflict_gate,
        "food_predator_gate": food_predator_gate,
        "corridor_gate": corridor_gate,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _b46_attempt_fitness(attempt: dict[str, object]) -> float:
    corridor_gate = attempt.get("corridor_gate", {})
    corridor_aggregate = (
        corridor_gate.get("aggregate", {}) if isinstance(corridor_gate, dict) else {}
    )
    canonical_gate = attempt.get("canonical_gate", {})
    canonical_aggregate = (
        canonical_gate.get("aggregate", {})
        if isinstance(canonical_gate, dict)
        else {}
    )
    score = (
        int(corridor_aggregate.get("explicit_decision_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("feedback_gain_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("topdown_context_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("prediction_match_episodes", 0) or 0) * 650.0
        + int(corridor_aggregate.get("feedback_stability_episodes", 0) or 0) * 850.0
        + int(corridor_aggregate.get("feedback_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score
