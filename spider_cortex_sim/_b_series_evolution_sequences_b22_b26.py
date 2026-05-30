from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_config_builders import (
    build_b23_conflict_monitor_config,
    build_b24_precision_conflict_config,
    build_b25_metacognitive_confidence_config,
    build_b26_allostatic_prediction_config,
)

from ._b_series_evolution_gates_b1_b6 import (
    b4_canonical_multi_gate_result,
    b4_easy_multi_gate_result,
    b5_food_deprivation_gate_result,
    b5_sleep_conflict_gate_result,
    b6_food_predator_conflict_gate_result,
)

from ._b_series_evolution_gates_b20_b30 import (
    b23_conflict_monitor_corridor_gate_result,
    b24_precision_conflict_corridor_gate_result,
    b25_metacognitive_confidence_corridor_gate_result,
    b26_allostatic_prediction_corridor_gate_result,
)

from ._b_series_evolution_gates_b61_requires import (
    _make_simulation,
    require_b21_hippocampal_replay_checkpoint,
    require_b22_prospective_replay_checkpoint,
    require_b23_conflict_monitor_checkpoint,
    require_b24_precision_conflict_checkpoint,
)

from ._b_series_evolution_requires_sequences_b1_b5 import (
    _run_episode_payload,
)

from ._b_series_evolution_sequences_b18_b22 import (
    _b22_attempt_fitness,
    run_b22_prospective_replay_attempt,
)

def run_b22_prospective_replay_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b22_training_episodes: int = 64,
    b22_workers: int = 1,
    b22_search: str = "hybrid",
    b22_ga_population: int = 24,
    b22_ga_generations: int = 8,
    b22_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b22_ga_population, b22_ga_generations, b22_finalists
    b21_report = require_b21_hippocampal_replay_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b21_report["checkpoint"])
    jobs = list(B22_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b22_search in {"fixed", "hybrid"} and int(b22_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b22_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b22_prospective_replay_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b22_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b22_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b22_prospective_replay_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b22_training_episodes,
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
    if selected is None and b22_search in {"ga", "hybrid"}:
        fallback_attempt = run_b22_prospective_replay_attempt(
            B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b22_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_prospective_replay",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b22_prospective_replay_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b22_training_episodes,
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
        "b21_source": b21_report,
        "b22_search": b22_search,
        "b22_workers": int(b22_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b22_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B22 prospective line and try explicit cortical model rollout"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b22_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b23_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b23_conflict_monitor_attempt(
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
    config = build_b23_conflict_monitor_config(
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
    corridor_gate = b23_conflict_monitor_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b23_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b23_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("conflict_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("monitor_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("conflict_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b23_conflict_monitor_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b23_training_episodes: int = 64,
    b23_workers: int = 1,
    b23_search: str = "hybrid",
    b23_ga_population: int = 24,
    b23_ga_generations: int = 8,
    b23_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b23_ga_population, b23_ga_generations, b23_finalists
    b22_report = require_b22_prospective_replay_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b22_report["checkpoint"])
    jobs = list(B23_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b23_search in {"fixed", "hybrid"} and int(b23_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b23_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b23_conflict_monitor_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b23_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b23_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b23_conflict_monitor_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b23_training_episodes,
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
    if selected is None and b23_search in {"ga", "hybrid"}:
        fallback_attempt = run_b23_conflict_monitor_attempt(
            B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b23_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_conflict_monitor",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b23_conflict_monitor_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b23_training_episodes,
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
        "b22_source": b22_report,
        "b23_search": b23_search,
        "b23_workers": int(b23_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b23_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B23 conflict-monitor line and try explicit prediction-error learning"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b23_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b24_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b24_precision_conflict_attempt(
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
    config = build_b24_precision_conflict_config(
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
    corridor_gate = b24_precision_conflict_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b24_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b24_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("precision_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("precision_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("precision_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b24_precision_conflict_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b24_training_episodes: int = 64,
    b24_workers: int = 1,
    b24_search: str = "hybrid",
    b24_ga_population: int = 24,
    b24_ga_generations: int = 8,
    b24_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b24_ga_population, b24_ga_generations, b24_finalists
    b23_report = require_b23_conflict_monitor_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b23_report["checkpoint"])
    jobs = list(B24_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b24_search in {"fixed", "hybrid"} and int(b24_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b24_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b24_precision_conflict_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b24_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b24_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b24_precision_conflict_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b24_training_episodes,
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
    if selected is None and b24_search in {"ga", "hybrid"}:
        fallback_attempt = run_b24_precision_conflict_attempt(
            B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b24_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_precision_conflict",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b24_precision_conflict_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b24_training_episodes,
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
        "b23_source": b23_report,
        "b24_search": b24_search,
        "b24_workers": int(b24_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b24_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B24 precision-conflict line and try explicit confidence calibration"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b24_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b25_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b25_metacognitive_confidence_attempt(
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
    config = build_b25_metacognitive_confidence_config(
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
    corridor_gate = b25_metacognitive_confidence_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b25_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b25_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("metacognitive_state_episodes", 0) or 0)
        * 1000.0
        + int(corridor_aggregate.get("meta_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("metacognitive_signal_episodes", 0) or 0)
        * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b25_metacognitive_confidence_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b25_training_episodes: int = 64,
    b25_workers: int = 1,
    b25_search: str = "hybrid",
    b25_ga_population: int = 24,
    b25_ga_generations: int = 8,
    b25_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b25_ga_population, b25_ga_generations, b25_finalists
    b24_report = require_b24_precision_conflict_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b24_report["checkpoint"])
    jobs = list(B25_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b25_search in {"fixed", "hybrid"} and int(b25_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b25_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b25_metacognitive_confidence_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b25_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b25_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b25_metacognitive_confidence_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b25_training_episodes,
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
    if selected is None and b25_search in {"ga", "hybrid"}:
        fallback_attempt = run_b25_metacognitive_confidence_attempt(
            B25_GENETIC_METACOGNITION_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b25_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_metacognition",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b25_metacognitive_confidence_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b25_training_episodes,
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
        "b24_source": b24_report,
        "b25_search": b25_search,
        "b25_workers": int(b25_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b25_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B25 metacognitive-confidence line and try explicit observation calibration"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b25_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b26_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b26_allostatic_prediction_attempt(
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
    config = build_b26_allostatic_prediction_config(
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
    corridor_gate = b26_allostatic_prediction_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b26_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b26_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("allostatic_state_episodes", 0) or 0)
        * 1000.0
        + int(corridor_aggregate.get("stability_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("allostatic_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score
