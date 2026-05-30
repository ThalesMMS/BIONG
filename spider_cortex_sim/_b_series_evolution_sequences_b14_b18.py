from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_config_builders import (
    build_b15_option_critic_config,
    build_b16_option_ensemble_config,
    build_b17_neuromodulated_ensemble_config,
    build_b18_eligibility_trace_config,
)

from ._b_series_evolution_gates_b1_b6 import (
    b4_canonical_multi_gate_result,
    b4_easy_multi_gate_result,
    b5_food_deprivation_gate_result,
    b5_sleep_conflict_gate_result,
    b6_food_predator_conflict_gate_result,
)

from ._b_series_evolution_gates_b61_requires import (
    _make_simulation,
    require_b13_local_affordance_search_checkpoint,
    require_b14_affordance_uncertainty_checkpoint,
    require_b15_option_critic_checkpoint,
    require_b16_option_ensemble_checkpoint,
)

from ._b_series_evolution_gates_b7_b19 import (
    b15_option_corridor_gate_result,
    b16_option_ensemble_corridor_gate_result,
    b17_neuromodulated_corridor_gate_result,
    b18_eligibility_corridor_gate_result,
)

from ._b_series_evolution_requires_sequences_b1_b5 import (
    _run_episode_payload,
)

from ._b_series_evolution_sequences_b10_b14 import (
    _b14_attempt_fitness,
    run_b14_affordance_uncertainty_attempt,
)

def run_b14_affordance_uncertainty_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b14_training_episodes: int = 64,
    b14_workers: int = 1,
    b14_search: str = "hybrid",
    b14_ga_population: int = 24,
    b14_ga_generations: int = 8,
    b14_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b14_ga_population, b14_ga_generations, b14_finalists
    b13_report = require_b13_local_affordance_search_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b13_report["checkpoint"])
    jobs = list(B14_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b14_search in {"fixed", "hybrid"} and int(b14_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b14_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b14_affordance_uncertainty_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b14_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b14_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b14_affordance_uncertainty_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b14_training_episodes,
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
    if selected is None and b14_search in {"ga", "hybrid"}:
        fallback_attempt = run_b14_affordance_uncertainty_attempt(
            B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b14_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_uncertainty_search",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b14_affordance_uncertainty_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b14_training_episodes,
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
        "b13_source": b13_report,
        "b14_search": b14_search,
        "b14_workers": int(b14_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b14_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B14 uncertainty line and try explicit affordance ensemble state"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b14_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b15_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b15_option_critic_attempt(
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
    config = build_b15_option_critic_config(
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
    food_deprivation_gate = b5_food_deprivation_gate_result(
        food_deprivation_results
    )
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
    food_predator_gate = b6_food_predator_conflict_gate_result(
        food_predator_results
    )
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b15_option_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b15_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b15_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("option_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("option_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("option_value_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b15_option_critic_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b15_training_episodes: int = 64,
    b15_workers: int = 1,
    b15_search: str = "hybrid",
    b15_ga_population: int = 24,
    b15_ga_generations: int = 8,
    b15_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b15_ga_population, b15_ga_generations, b15_finalists
    b14_report = require_b14_affordance_uncertainty_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b14_report["checkpoint"])
    jobs = list(B15_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b15_search in {"fixed", "hybrid"} and int(b15_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b15_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b15_option_critic_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b15_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b15_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b15_option_critic_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b15_training_episodes,
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
    if selected is None and b15_search in {"ga", "hybrid"}:
        fallback_attempt = run_b15_option_critic_attempt(
            B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b15_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_option_critic",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b15_option_critic_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b15_training_episodes,
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
        "b14_source": b14_report,
        "b15_search": b15_search,
        "b15_workers": int(b15_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b15_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B15 option-critic line and try explicit option ensemble dynamics"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b15_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b16_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b16_option_ensemble_attempt(
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
    config = build_b16_option_ensemble_config(
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
    food_deprivation_gate = b5_food_deprivation_gate_result(
        food_deprivation_results
    )
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
    food_predator_gate = b6_food_predator_conflict_gate_result(
        food_predator_results
    )
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b16_option_ensemble_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b16_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b16_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("ensemble_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("ensemble_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("ensemble_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b16_option_ensemble_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b16_training_episodes: int = 64,
    b16_workers: int = 1,
    b16_search: str = "hybrid",
    b16_ga_population: int = 24,
    b16_ga_generations: int = 8,
    b16_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b16_ga_population, b16_ga_generations, b16_finalists
    b15_report = require_b15_option_critic_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b15_report["checkpoint"])
    jobs = list(B16_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b16_search in {"fixed", "hybrid"} and int(b16_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b16_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b16_option_ensemble_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b16_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b16_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b16_option_ensemble_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b16_training_episodes,
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
    if selected is None and b16_search in {"ga", "hybrid"}:
        fallback_attempt = run_b16_option_ensemble_attempt(
            B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b16_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_option_ensemble",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b16_option_ensemble_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b16_training_episodes,
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
        "b15_source": b15_report,
        "b16_search": b16_search,
        "b16_workers": int(b16_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b16_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B16 ensemble line and try explicit multi-option arbitration"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b16_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b17_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b17_neuromodulated_ensemble_attempt(
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
    config = build_b17_neuromodulated_ensemble_config(
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
    food_deprivation_gate = b5_food_deprivation_gate_result(
        food_deprivation_results
    )
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
    food_predator_gate = b6_food_predator_conflict_gate_result(
        food_predator_results
    )
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b17_neuromodulated_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b17_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b17_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("modulator_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("modulation_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("modulation_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b17_neuromodulated_ensemble_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b17_training_episodes: int = 64,
    b17_workers: int = 1,
    b17_search: str = "hybrid",
    b17_ga_population: int = 24,
    b17_ga_generations: int = 8,
    b17_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b17_ga_population, b17_ga_generations, b17_finalists
    b16_report = require_b16_option_ensemble_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b16_report["checkpoint"])
    jobs = list(B17_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b17_search in {"fixed", "hybrid"} and int(b17_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b17_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b17_neuromodulated_ensemble_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b17_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b17_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b17_neuromodulated_ensemble_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b17_training_episodes,
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
    if selected is None and b17_search in {"ga", "hybrid"}:
        fallback_attempt = run_b17_neuromodulated_ensemble_attempt(
            B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b17_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_neuromodulated_ensemble",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b17_neuromodulated_ensemble_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b17_training_episodes,
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
        "b16_source": b16_report,
        "b17_search": b17_search,
        "b17_workers": int(b17_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b17_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B17 neuromodulation line and try explicit meta-controller recurrence"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b17_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b18_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b18_eligibility_trace_attempt(
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
    config = build_b18_eligibility_trace_config(
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
    food_deprivation_gate = b5_food_deprivation_gate_result(
        food_deprivation_results
    )
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
    food_predator_gate = b6_food_predator_conflict_gate_result(
        food_predator_results
    )
    corridor_results = [
        _run_episode_payload(
            sim,
            evaluation_episode=episode,
            scenario_name=B6_CORRIDOR_SCENARIO,
        )
        for episode in B6_PROBE_EVALUATION_EPISODES
    ]
    corridor_gate = b18_eligibility_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b18_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b18_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("trace_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("trace_lock_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("trace_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score
