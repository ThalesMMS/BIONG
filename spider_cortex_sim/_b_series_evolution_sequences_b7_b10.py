from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_config_builders import (
    build_b10_prospective_replay_config,
    build_b8_spatial_affordance_config,
    build_b9_waypoint_planner_config,
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
    require_b6_fused_risk_recurrent_checkpoint,
    require_b7_affordance_budget_checkpoint,
    require_b8_spatial_affordance_checkpoint,
)

from ._b_series_evolution_gates_b7_b19 import (
    b10_prospective_corridor_gate_result,
    b8_spatial_corridor_gate_result,
    b9_waypoint_corridor_gate_result,
)

from ._b_series_evolution_requires_sequences_b1_b5 import (
    _run_episode_payload,
)

from ._b_series_evolution_sequences_b5_b7 import (
    run_b7_affordance_budget_attempt,
)

def _b7_attempt_fitness(attempt: dict[str, object]) -> float:
    canonical_gate = attempt.get("canonical_gate", {})
    canonical_aggregate = (
        canonical_gate.get("aggregate", {})
        if isinstance(canonical_gate, dict)
        else {}
    )
    food_gate = attempt.get("food_predator_gate", {})
    food_aggregate = (
        food_gate.get("aggregate", {}) if isinstance(food_gate, dict) else {}
    )
    corridor_gate = attempt.get("corridor_gate", {})
    corridor_aggregate = (
        corridor_gate.get("aggregate", {}) if isinstance(corridor_gate, dict) else {}
    )
    completed = int(canonical_aggregate.get("completed_horizons", 0) or 0)
    min_steps = int(canonical_aggregate.get("min_steps", 0) or 0)
    contacts = int(canonical_aggregate.get("total_predator_contacts", 999) or 999)
    threat_priority = int(
        food_aggregate.get("threat_priority_or_suppression_episodes", 0) or 0
    )
    food_progress = int(corridor_aggregate.get("food_progress_episodes", 0) or 0)
    explicit_decisions = int(
        corridor_aggregate.get("explicit_decision_episodes", 0) or 0
    )
    improvements = int(corridor_aggregate.get("improvement_episodes", 0) or 0)
    score = (
        completed * 1000.0
        + min_steps * 3.0
        - contacts * 20.0
        + threat_priority * 500.0
        + food_progress * 700.0
        + explicit_decisions * 900.0
        + improvements * 700.0
    )
    if not bool(attempt.get("easy_gate", {}).get("passed", False)):
        score -= 5000.0
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


B7_GENETIC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "b7_budget_step_cost": (0.06, 0.11),
    "b7_viability_margin": (-0.18, 0.10),
    "b7_abort_health": (0.20, 0.55),
    "b7_recover_health": (0.25, 0.65),
    "b7_food_commit_distance": (8.0, 14.0),
    "b7_commitment_ticks": (4.0, 14.0),
    "b7_recurrent_decay": (0.55, 0.90),
}


def _b7_random_params(rng: random.Random) -> dict[str, float]:
    return {
        key: round(rng.uniform(low, high), 6)
        for key, (low, high) in B7_GENETIC_PARAM_BOUNDS.items()
    }


def _b7_breed_params(
    rng: random.Random,
    parent_a: dict[str, float],
    parent_b: dict[str, float],
) -> dict[str, float]:
    child: dict[str, float] = {}
    for key, (low, high) in B7_GENETIC_PARAM_BOUNDS.items():
        value = float(parent_a.get(key, parent_b.get(key, (low + high) / 2.0)))
        if rng.random() < 0.5:
            value = float(parent_b.get(key, value))
        if rng.random() < 0.35:
            value += rng.gauss(0.0, (high - low) * 0.12)
        child[key] = round(max(low, min(high, value)), 6)
    return child


def run_b7_genetic_search(
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    population_size: int = 24,
    generations: int = 8,
    finalists: int = 6,
    workers: int = 1,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    screening_training_episodes: int = 32,
    confirmation_training_episodes: int = 64,
) -> dict[str, object]:
    rng = random.Random(int(seed) + 7100)
    population = [
        _b7_random_params(rng) for _ in range(max(1, int(population_size)))
    ]
    generation_reports = []
    screened_attempts: list[dict[str, object]] = []
    for generation in range(max(1, int(generations))):
        jobs = []
        for candidate_index, params in enumerate(population):
            candidate_params = dict(params)
            candidate_params["ga_generation"] = float(generation)
            candidate_params["ga_candidate"] = float(candidate_index)
            candidate_id = (
                f"affordance_budget_generation_{generation:02d}_"
                f"candidate_{candidate_index:02d}"
            )
            jobs.append((candidate_index, candidate_id, candidate_params))
        attempts_by_index: dict[int, dict[str, object]] = {}
        if int(workers) > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as pool:
                future_map = {
                    pool.submit(
                        run_b7_affordance_budget_attempt,
                        B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=screening_training_episodes,
                        reward_profile=reward_profile,
                        operational_profile=operational_profile,
                        noise_profile=noise_profile,
                        controller_profile="genetic_affordance_budget",
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
                attempts_by_index[candidate_index] = run_b7_affordance_budget_attempt(
                    B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=screening_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    controller_profile="genetic_affordance_budget",
                    controller_params=params,
                    candidate_id=candidate_id,
                    promote_if_accepted=False,
                )
        ranked = sorted(
            attempts_by_index.values(),
            key=_b7_attempt_fitness,
            reverse=True,
        )
        screened_attempts.extend(ranked)
        generation_reports.append(
            {
                "generation": int(generation),
                "best_candidate_id": ranked[0].get("candidate_id") if ranked else None,
                "best_fitness": _b7_attempt_fitness(ranked[0]) if ranked else None,
                "best_status": ranked[0].get("status") if ranked else None,
            }
        )
        elites = [dict(attempt["controller_params"]) for attempt in ranked[:6]]
        if not elites:
            population = [
                _b7_random_params(rng)
                for _ in range(max(1, int(population_size)))
            ]
            continue
        next_population = elites[:]
        while len(next_population) < max(1, int(population_size)):
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            next_population.append(_b7_breed_params(rng, parent_a, parent_b))
        population = next_population[: max(1, int(population_size))]

    finalist_count = max(1, int(finalists))
    finalist_attempts = sorted(
        screened_attempts,
        key=_b7_attempt_fitness,
        reverse=True,
    )[:finalist_count]
    confirmed_attempts = []
    for finalist_index, finalist in enumerate(finalist_attempts):
        params = dict(finalist.get("controller_params", {}))
        params["ga_generation"] = float(params.get("ga_generation", 0.0))
        params["ga_candidate"] = float(params.get("ga_candidate", finalist_index))
        attempt = run_b7_affordance_budget_attempt(
            B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=confirmation_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_affordance_budget",
            controller_params=params,
            candidate_id=f"affordance_budget_finalist_{finalist_index:02d}",
            promote_if_accepted=False,
        )
        confirmed_attempts.append(attempt)
    accepted_attempts = [
        attempt
        for attempt in confirmed_attempts
        if attempt.get("status") == "accepted"
    ]
    accepted_attempt = (
        max(accepted_attempts, key=_b7_attempt_fitness)
        if accepted_attempts
        else None
    )
    search_summary = {
        "status": "accepted" if accepted_attempt is not None else "discarded",
        "accepted_attempt": accepted_attempt,
        "generation_reports": generation_reports,
        "confirmed_attempts": confirmed_attempts,
        "screened_attempt_count": len(screened_attempts),
        "finalist_count": int(finalist_count),
    }
    search_dir = (
        Path(root)
        / B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME
        / f"seed_{int(seed)}"
    )
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "ga_search_report.json").write_text(
        json.dumps(search_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return search_summary


def run_b7_affordance_budget_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b7_training_episodes: int = 64,
    b7_workers: int = 1,
    b7_search: str = "hybrid",
    b7_ga_population: int = 24,
    b7_ga_generations: int = 8,
    b7_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    b6_report = require_b6_fused_risk_recurrent_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b6_report["checkpoint"])
    fixed_attempts: list[dict[str, object]] = []
    genetic_search: dict[str, object] | None = None
    selected: dict[str, object] | None = None
    if b7_search in {"fixed", "hybrid"}:
        jobs = list(B7_FIXED_EVOLUTION_ATTEMPTS)
        attempts_by_name: dict[str, dict[str, object]] = {}
        if int(b7_workers) > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(b7_workers), len(jobs))) as pool:
                future_map = {
                    pool.submit(
                        run_b7_affordance_budget_attempt,
                        variant_name,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=b7_training_episodes,
                        reward_profile=reward_profile,
                        operational_profile=operational_profile,
                        noise_profile=noise_profile,
                        promote_if_accepted=False,
                    ): variant_name
                    for variant_name in jobs
                }
                for future in as_completed(future_map):
                    attempts_by_name[future_map[future]] = future.result()
        else:
            for variant_name in jobs:
                attempts_by_name[variant_name] = run_b7_affordance_budget_attempt(
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b7_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                )
        fixed_attempts = [attempts_by_name[name] for name in jobs]
        for attempt in fixed_attempts:
            if attempt.get("status") == "accepted":
                selected = attempt
                break

    if selected is None and b7_search in {"ga", "hybrid"}:
        genetic_search = run_b7_genetic_search(
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            population_size=b7_ga_population,
            generations=b7_ga_generations,
            finalists=b7_finalists,
            workers=b7_workers,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            screening_training_episodes=max(1, int(b7_training_episodes) // 2),
            confirmation_training_episodes=b7_training_episodes,
        )
        accepted = genetic_search.get("accepted_attempt")
        if isinstance(accepted, dict):
            selected = accepted

    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b7_affordance_budget_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b7_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile=str(selected.get("controller_profile")),
            controller_params=dict(selected.get("controller_params", {})),
            candidate_id=None,
            promote_if_accepted=True,
        )
    accepted_variant = (
        str(promoted_attempt["variant"])
        if isinstance(promoted_attempt, dict)
        and promoted_attempt.get("status") == "accepted"
        else None
    )
    attempts = [*fixed_attempts]
    if isinstance(genetic_search, dict):
        accepted = genetic_search.get("accepted_attempt")
        if isinstance(accepted, dict):
            attempts.append(accepted)
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
        "b6_source": b6_report,
        "b7_search": b7_search,
        "b7_workers": int(b7_workers),
        "b7_corridor_gate": {
            "scenario": B6_CORRIDOR_SCENARIO,
            "required_food_distance_delta": B7_CORRIDOR_REQUIRED_FOOD_DISTANCE_DELTA,
            "explicit_decisions": list(B7_CORRIDOR_EXPLICIT_DECISIONS),
        },
        "fixed_attempts": fixed_attempts,
        "genetic_search": genetic_search,
        "attempts": attempts,
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B7 affordance-budget line and try explicit spatial observation expansion or formal corridor_gauntlet adjustment for B7/B8"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b7_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b8_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b8_spatial_affordance_attempt(
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
    config = build_b8_spatial_affordance_config(
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
    corridor_gate = b8_spatial_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b8_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b8_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("spatial_map_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("mapped_progress_episodes", 0) or 0) * 1000.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b8_spatial_affordance_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b8_training_episodes: int = 64,
    b8_workers: int = 1,
    b8_search: str = "hybrid",
    b8_ga_population: int = 24,
    b8_ga_generations: int = 8,
    b8_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b8_ga_population, b8_ga_generations, b8_finalists
    b7_report = require_b7_affordance_budget_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b7_report["checkpoint"])
    jobs = list(B8_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b8_search in {"fixed", "hybrid"} and int(b8_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b8_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b8_spatial_affordance_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b8_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b8_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b8_spatial_affordance_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b8_training_episodes,
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
    if selected is None and b8_search in {"ga", "hybrid"}:
        fallback_attempt = run_b8_spatial_affordance_attempt(
            B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b8_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_spatial_affordance",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b8_spatial_affordance_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b8_training_episodes,
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
        "b7_source": b7_report,
        "b8_search": b8_search,
        "b8_workers": int(b8_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b8_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B8 spatial-affordance line and try explicit observation expansion for local corridor geometry"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b8_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b9_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b9_waypoint_planner_attempt(
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
    config = build_b9_waypoint_planner_config(
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
    corridor_gate = b9_waypoint_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b9_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b9_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("route_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("locked_waypoint_episodes", 0) or 0) * 1000.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b9_waypoint_planner_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b9_training_episodes: int = 64,
    b9_workers: int = 1,
    b9_search: str = "hybrid",
    b9_ga_population: int = 24,
    b9_ga_generations: int = 8,
    b9_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b9_ga_population, b9_ga_generations, b9_finalists
    b8_report = require_b8_spatial_affordance_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b8_report["checkpoint"])
    jobs = list(B9_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b9_search in {"fixed", "hybrid"} and int(b9_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b9_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b9_waypoint_planner_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b9_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b9_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b9_waypoint_planner_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b9_training_episodes,
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
    if selected is None and b9_search in {"ga", "hybrid"}:
        fallback_attempt = run_b9_waypoint_planner_attempt(
            B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b9_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_waypoint_planner",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b9_waypoint_planner_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b9_training_episodes,
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
        "b8_source": b8_report,
        "b9_search": b9_search,
        "b9_workers": int(b9_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b9_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B9 waypoint-planner line and try explicit rollout/value search over corridor affordances"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b9_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b10_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b10_prospective_replay_attempt(
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
    config = build_b10_prospective_replay_config(
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
    corridor_gate = b10_prospective_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b10_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b10_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("replay_state_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("committed_plan_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("value_signal_episodes", 0) or 0) * 750.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score
