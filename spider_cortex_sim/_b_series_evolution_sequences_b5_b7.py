from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_config_builders import (
    build_b5_homeostatic_arbiter_config,
    build_b6_risk_corridor_config,
    build_b7_affordance_budget_config,
)

from ._b_series_evolution_gates_b1_b6 import (
    b4_canonical_multi_gate_result,
    b4_easy_multi_gate_result,
    b5_diagnostic_probe_result,
    b5_food_deprivation_gate_result,
    b5_sleep_conflict_gate_result,
    b6_corridor_progress_gate_result,
    b6_food_predator_conflict_gate_result,
)

from ._b_series_evolution_gates_b61_requires import (
    _make_simulation,
    require_b4_genetic_recovery_checkpoint,
    require_b5_genetic_homeostasis_checkpoint,
)

from ._b_series_evolution_gates_b7_b19 import (
    b7_corridor_viability_gate_result,
)

from ._b_series_evolution_requires_sequences_b1_b5 import (
    _b5_attempt_dir,
    _run_episode_payload,
)

def run_b5_homeostatic_arbiter_attempt(
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
    config = build_b5_homeostatic_arbiter_config(
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
    diagnostic_probes = {
        scenario_name: b5_diagnostic_probe_result(
            scenario_name,
            [
                _run_episode_payload(
                    sim,
                    evaluation_episode=episode,
                    scenario_name=scenario_name,
                )
                for episode in B5_PROBE_EVALUATION_EPISODES
            ],
        )
        for scenario_name in B5_DIAGNOSTIC_SCENARIOS
    }
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
    )
    attempt_dir = _b5_attempt_dir(root, variant_name, seed, candidate_id)
    checkpoint_name = "best" if accepted and promote_if_accepted else "discarded"
    checkpoint_path = attempt_dir / checkpoint_name
    sim.brain.save(checkpoint_path)
    discard_failures = []
    for gate in (
        easy_gate,
        canonical_gate,
        food_deprivation_gate,
        sleep_conflict_gate,
    ):
        if not gate["passed"]:
            discard_failures.extend(gate["failures"])
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
        "probe_evaluation_episodes": list(B5_PROBE_EVALUATION_EPISODES),
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
        "easy_gate": easy_gate,
        "canonical_gate": canonical_gate,
        "food_deprivation_gate": food_deprivation_gate,
        "sleep_conflict_gate": sleep_conflict_gate,
        "diagnostic_probes": diagnostic_probes,
    }
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "attempt_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _b5_attempt_fitness(attempt: dict[str, object]) -> float:
    canonical_gate = attempt.get("canonical_gate", {})
    canonical_aggregate = (
        canonical_gate.get("aggregate", {})
        if isinstance(canonical_gate, dict)
        else {}
    )
    food_gate = attempt.get("food_deprivation_gate", {})
    food_aggregate = (
        food_gate.get("aggregate", {}) if isinstance(food_gate, dict) else {}
    )
    sleep_gate = attempt.get("sleep_conflict_gate", {})
    sleep_aggregate = (
        sleep_gate.get("aggregate", {}) if isinstance(sleep_gate, dict) else {}
    )
    completed = int(canonical_aggregate.get("completed_horizons", 0) or 0)
    min_steps = int(canonical_aggregate.get("min_steps", 0) or 0)
    contacts = int(canonical_aggregate.get("total_predator_contacts", 999) or 999)
    food_progress = int(food_aggregate.get("progress_episodes", 0) or 0)
    sleep_movement = int(
        sleep_aggregate.get("post_recovery_movement_episodes", 0) or 0
    )
    score = (
        completed * 1000.0
        + min_steps * 3.0
        - contacts * 20.0
        + food_progress * 700.0
        + sleep_movement * 500.0
    )
    if not bool(attempt.get("easy_gate", {}).get("passed", False)):
        score -= 5000.0
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


B5_GENETIC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "hunger_release": (0.80, 0.91),
    "emergency_hunger_release": (0.90, 0.98),
    "forage_threat_max": (0.42, 0.70),
    "forage_lock_ticks": (5.0, 12.0),
    "sleep_pressure_threshold": (0.50, 0.78),
    "sleep_hunger_max": (0.68, 0.86),
    "sleep_threat_max": (0.42, 0.72),
    "sleep_lock_ticks": (5.0, 12.0),
    "exit_sleep_pressure_max": (0.38, 0.68),
    "exit_recovery_debt_max": (0.46, 0.74),
    "exit_threat_max": (0.44, 0.72),
    "return_hunger_max": (0.74, 0.90),
}


def _b5_random_params(rng: random.Random) -> dict[str, float]:
    return {
        key: round(rng.uniform(low, high), 6)
        for key, (low, high) in B5_GENETIC_PARAM_BOUNDS.items()
    }


def _b5_breed_params(
    rng: random.Random,
    parent_a: dict[str, float],
    parent_b: dict[str, float],
) -> dict[str, float]:
    child: dict[str, float] = {}
    for key, (low, high) in B5_GENETIC_PARAM_BOUNDS.items():
        value = parent_a[key] if rng.random() < 0.5 else parent_b[key]
        if rng.random() < 0.35:
            value += rng.gauss(0.0, (high - low) * 0.12)
        child[key] = round(max(low, min(high, float(value))), 6)
    return child


def run_b5_genetic_search(
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
    rng = random.Random(int(seed) + 5000)
    population = [_b5_random_params(rng) for _ in range(max(1, int(population_size)))]
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
                        run_b5_homeostatic_arbiter_attempt,
                        B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=screening_training_episodes,
                        reward_profile=reward_profile,
                        operational_profile=operational_profile,
                        noise_profile=noise_profile,
                        controller_profile="genetic_homeostasis",
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
                attempts_by_index[candidate_index] = run_b5_homeostatic_arbiter_attempt(
                    B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=screening_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    controller_profile="genetic_homeostasis",
                    controller_params=params,
                    candidate_id=candidate_id,
                    promote_if_accepted=False,
                )
        ranked = sorted(
            attempts_by_index.values(),
            key=_b5_attempt_fitness,
            reverse=True,
        )
        screened_attempts.extend(ranked)
        generation_reports.append(
            {
                "generation": int(generation),
                "best_candidate_id": ranked[0].get("candidate_id") if ranked else None,
                "best_fitness": _b5_attempt_fitness(ranked[0]) if ranked else None,
                "best_status": ranked[0].get("status") if ranked else None,
            }
        )
        elites = [dict(attempt["controller_params"]) for attempt in ranked[:4]]
        if not elites:
            population = [
                _b5_random_params(rng) for _ in range(max(1, int(population_size)))
            ]
            continue
        next_population = elites[:]
        while len(next_population) < max(1, int(population_size)):
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            next_population.append(_b5_breed_params(rng, parent_a, parent_b))
        population = next_population[: max(1, int(population_size))]

    finalists = sorted(screened_attempts, key=_b5_attempt_fitness, reverse=True)[:3]
    confirmed_attempts = []
    accepted_attempt: dict[str, object] | None = None
    for finalist_index, finalist in enumerate(finalists):
        params = dict(finalist.get("controller_params", {}))
        params["ga_generation"] = float(params.get("ga_generation", 0.0))
        params["ga_candidate"] = float(params.get("ga_candidate", finalist_index))
        attempt = run_b5_homeostatic_arbiter_attempt(
            B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=confirmation_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_homeostasis",
            controller_params=params,
            candidate_id=f"finalist_{finalist_index:02d}",
            promote_if_accepted=False,
        )
        confirmed_attempts.append(attempt)
        if attempt["status"] == "accepted":
            accepted_attempt = run_b5_homeostatic_arbiter_attempt(
                B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=confirmation_training_episodes,
                reward_profile=reward_profile,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                controller_profile="genetic_homeostasis",
                controller_params=params,
                candidate_id=None,
                promote_if_accepted=True,
            )
            break
    if accepted_attempt is None and confirmed_attempts:
        best = max(confirmed_attempts, key=_b5_attempt_fitness)
        accepted_attempt = run_b5_homeostatic_arbiter_attempt(
            B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=confirmation_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_homeostasis",
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
    search_dir = Path(root) / B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME / f"seed_{int(seed)}"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "ga_search_report.json").write_text(
        json.dumps(search_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return search_summary


def run_b5_homeostatic_arbiter_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b5_training_episodes: int = 48,
    b5_workers: int = 1,
    b5_search: str = "hybrid",
    b5_ga_population: int = 12,
    b5_ga_generations: int = 4,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    b4_report = require_b4_genetic_recovery_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b4_report["checkpoint"])
    attempts: list[dict[str, object]] = []
    accepted_variant: str | None = None
    fixed_attempts: list[dict[str, object]] = []
    if b5_search in {"fixed", "hybrid"}:
        jobs = list(B5_FIXED_EVOLUTION_ATTEMPTS)
        attempts_by_name: dict[str, dict[str, object]] = {}
        if int(b5_workers) > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(b5_workers), len(jobs))) as pool:
                future_map = {
                    pool.submit(
                        run_b5_homeostatic_arbiter_attempt,
                        variant_name,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=b5_training_episodes,
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
                attempts_by_name[variant_name] = run_b5_homeostatic_arbiter_attempt(
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b5_training_episodes,
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
    if accepted_variant is None and b5_search in {"ga", "hybrid"}:
        genetic_summary = run_b5_genetic_search(
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            population_size=b5_ga_population,
            generations=b5_ga_generations,
            workers=b5_workers,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            screening_training_episodes=max(1, int(b5_training_episodes) // 2),
            confirmation_training_episodes=b5_training_episodes,
        )
        accepted_attempt = genetic_summary.get("accepted_attempt")
        if isinstance(accepted_attempt, dict):
            attempts.append(accepted_attempt)
            if accepted_attempt.get("status") == "accepted":
                accepted_variant = str(accepted_attempt["variant"])

    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "b4_source": b4_report,
        "b5_search": b5_search,
        "b5_workers": int(b5_workers),
        "canonical_baseline": dict(B4_CANONICAL_BASELINE),
        "attempts": attempts,
        "genetic_search": genetic_summary,
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B5 homeostatic-arbiter line and try a structural interoceptive recurrent module from the accepted B4 source"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b5_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b6_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def b6_family_for_variant(variant_name: str) -> str:
    if variant_name == B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME:
        return "fused_risk_recurrent"
    if variant_name in B6_RECURRENT_EVOLUTION_ATTEMPTS:
        return "recurrent_memory"
    return "risk_corridor"


def _b6_default_profile_for_family(family: str) -> str:
    if family == "fused_risk_recurrent":
        return "fused_risk_recurrent"
    if family == "recurrent_memory":
        return "genetic_recurrent_memory"
    return "genetic_risk_corridor"


def run_b6_risk_corridor_attempt(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    training_episodes: int = 64,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
    controller_family: str | None = None,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
    candidate_id: str | None = None,
    promote_if_accepted: bool = True,
) -> dict[str, object]:
    family = controller_family or b6_family_for_variant(variant_name)
    config = build_b6_risk_corridor_config(
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
    corridor_gate = b6_corridor_progress_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b6_attempt_dir(root, variant_name, seed, candidate_id)
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
        "controller_family": family,
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
        "easy_evaluation_episodes": list(B4_EASY_EVALUATION_EPISODES),
        "canonical_evaluation_episodes": list(B4_CANONICAL_EVALUATION_EPISODES),
        "probe_evaluation_episodes": list(B6_PROBE_EVALUATION_EPISODES),
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
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


def _b6_attempt_fitness(attempt: dict[str, object]) -> float:
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
    threat_exposure = int(food_aggregate.get("threat_exposure_episodes", 0) or 0)
    threat_priority = int(
        food_aggregate.get("threat_priority_or_suppression_episodes", 0) or 0
    )
    corridor_progress = int(corridor_aggregate.get("progress_episodes", 0) or 0)
    corridor_survival = int(
        corridor_aggregate.get("survival_progress_episodes", 0) or 0
    )
    score = (
        completed * 1000.0
        + min_steps * 3.0
        - contacts * 20.0
        + threat_exposure * 500.0
        + threat_priority * 700.0
        + corridor_progress * 500.0
        + corridor_survival * 700.0
    )
    if not bool(attempt.get("easy_gate", {}).get("passed", False)):
        score -= 5000.0
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


B6_GENETIC_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "b6_risk_threshold": (0.10, 0.55),
    "b6_corridor_hunger": (0.78, 0.92),
    "b6_corridor_lock_ticks": (8.0, 18.0),
    "b6_threat_memory_ticks": (6.0, 18.0),
    "b6_return_lock_ticks": (4.0, 14.0),
    "b6_recurrent_decay": (0.55, 0.90),
}


def _b6_random_params(rng: random.Random, *, family: str) -> dict[str, float]:
    params = {
        key: round(rng.uniform(low, high), 6)
        for key, (low, high) in B6_GENETIC_PARAM_BOUNDS.items()
    }
    params["b6_family"] = 2.0 if family == "recurrent_memory" else 1.0
    return params


def _b6_breed_params(
    rng: random.Random,
    parent_a: dict[str, float],
    parent_b: dict[str, float],
    *,
    family: str,
) -> dict[str, float]:
    child: dict[str, float] = {}
    for key, (low, high) in B6_GENETIC_PARAM_BOUNDS.items():
        value = parent_a[key] if rng.random() < 0.5 else parent_b[key]
        if rng.random() < 0.35:
            value += rng.gauss(0.0, (high - low) * 0.12)
        child[key] = round(max(low, min(high, float(value))), 6)
    child["b6_family"] = 2.0 if family == "recurrent_memory" else 1.0
    return child


def run_b6_genetic_search(
    *,
    family: str,
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
    variant_name = (
        B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME
        if family == "recurrent_memory"
        else B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME
    )
    controller_profile = _b6_default_profile_for_family(family)
    rng = random.Random(int(seed) + (6200 if family == "recurrent_memory" else 6100))
    population = [
        _b6_random_params(rng, family=family)
        for _ in range(max(1, int(population_size)))
    ]
    generation_reports = []
    screened_attempts: list[dict[str, object]] = []
    for generation in range(max(1, int(generations))):
        jobs = []
        for candidate_index, params in enumerate(population):
            candidate_params = dict(params)
            candidate_params["ga_generation"] = float(generation)
            candidate_params["ga_candidate"] = float(candidate_index)
            candidate_id = f"{family}_generation_{generation:02d}_candidate_{candidate_index:02d}"
            jobs.append((candidate_index, candidate_id, candidate_params))
        attempts_by_index: dict[int, dict[str, object]] = {}
        if int(workers) > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(workers), len(jobs))) as pool:
                future_map = {
                    pool.submit(
                        run_b6_risk_corridor_attempt,
                        variant_name,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=screening_training_episodes,
                        reward_profile=reward_profile,
                        operational_profile=operational_profile,
                        noise_profile=noise_profile,
                        controller_family=family,
                        controller_profile=controller_profile,
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
                attempts_by_index[candidate_index] = run_b6_risk_corridor_attempt(
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=screening_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    controller_family=family,
                    controller_profile=controller_profile,
                    controller_params=params,
                    candidate_id=candidate_id,
                    promote_if_accepted=False,
                )
        ranked = sorted(
            attempts_by_index.values(),
            key=_b6_attempt_fitness,
            reverse=True,
        )
        screened_attempts.extend(ranked)
        generation_reports.append(
            {
                "family": family,
                "generation": int(generation),
                "best_candidate_id": ranked[0].get("candidate_id") if ranked else None,
                "best_fitness": _b6_attempt_fitness(ranked[0]) if ranked else None,
                "best_status": ranked[0].get("status") if ranked else None,
            }
        )
        elites = [dict(attempt["controller_params"]) for attempt in ranked[:6]]
        if not elites:
            population = [
                _b6_random_params(rng, family=family)
                for _ in range(max(1, int(population_size)))
            ]
            continue
        next_population = elites[:]
        while len(next_population) < max(1, int(population_size)):
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            next_population.append(
                _b6_breed_params(rng, parent_a, parent_b, family=family)
            )
        population = next_population[: max(1, int(population_size))]

    finalist_count = max(1, int(finalists))
    finalist_attempts = sorted(
        screened_attempts,
        key=_b6_attempt_fitness,
        reverse=True,
    )[:finalist_count]
    confirmed_attempts = []
    for finalist_index, finalist in enumerate(finalist_attempts):
        params = dict(finalist.get("controller_params", {}))
        params["ga_generation"] = float(params.get("ga_generation", 0.0))
        params["ga_candidate"] = float(params.get("ga_candidate", finalist_index))
        attempt = run_b6_risk_corridor_attempt(
            variant_name,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=confirmation_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_family=family,
            controller_profile=controller_profile,
            controller_params=params,
            candidate_id=f"{family}_finalist_{finalist_index:02d}",
            promote_if_accepted=False,
        )
        confirmed_attempts.append(attempt)
    accepted_attempts = [
        attempt
        for attempt in confirmed_attempts
        if attempt.get("status") == "accepted"
    ]
    accepted_attempt = (
        max(accepted_attempts, key=_b6_attempt_fitness)
        if accepted_attempts
        else None
    )
    search_summary = {
        "family": family,
        "status": "accepted" if accepted_attempt is not None else "discarded",
        "accepted_attempt": accepted_attempt,
        "generation_reports": generation_reports,
        "confirmed_attempts": confirmed_attempts,
        "screened_attempt_count": len(screened_attempts),
        "finalist_count": int(finalist_count),
    }
    search_dir = Path(root) / variant_name / f"seed_{int(seed)}"
    search_dir.mkdir(parents=True, exist_ok=True)
    (search_dir / "ga_search_report.json").write_text(
        json.dumps(search_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return search_summary


def select_b6_promotion(
    risk_attempts: Sequence[dict[str, object]],
    recurrent_attempts: Sequence[dict[str, object]],
    fusion_attempt: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if fusion_attempt is not None and fusion_attempt.get("status") == "accepted":
        return fusion_attempt
    accepted_attempts = [
        attempt
        for attempt in [*risk_attempts, *recurrent_attempts]
        if attempt.get("status") == "accepted"
    ]
    if not accepted_attempts:
        return None
    return max(accepted_attempts, key=_b6_attempt_fitness)


def run_b6_risk_corridor_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b6_training_episodes: int = 64,
    b6_workers: int = 1,
    b6_search: str = "exhaustive",
    b6_ga_population: int = 24,
    b6_ga_generations: int = 8,
    b6_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    b5_report = require_b5_genetic_homeostasis_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b5_report["checkpoint"])
    fixed_attempts: list[dict[str, object]] = []
    genetic_searches: dict[str, object] = {}
    if b6_search in {"fixed", "exhaustive"}:
        jobs = list(B6_FIXED_EVOLUTION_ATTEMPTS)
        attempts_by_name: dict[str, dict[str, object]] = {}
        if int(b6_workers) > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=min(int(b6_workers), len(jobs))) as pool:
                future_map = {
                    pool.submit(
                        run_b6_risk_corridor_attempt,
                        variant_name,
                        source_checkpoint=source_checkpoint,
                        root=root,
                        seed=seed,
                        training_episodes=b6_training_episodes,
                        reward_profile=reward_profile,
                        operational_profile=operational_profile,
                        noise_profile=noise_profile,
                        controller_family=b6_family_for_variant(variant_name),
                        promote_if_accepted=False,
                    ): variant_name
                    for variant_name in jobs
                }
                for future in as_completed(future_map):
                    attempts_by_name[future_map[future]] = future.result()
        else:
            for variant_name in jobs:
                attempts_by_name[variant_name] = run_b6_risk_corridor_attempt(
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b6_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    controller_family=b6_family_for_variant(variant_name),
                    promote_if_accepted=False,
                )
        fixed_attempts = [attempts_by_name[name] for name in jobs]

    if b6_search in {"ga", "exhaustive"}:
        for family in ("risk_corridor", "recurrent_memory"):
            genetic_searches[family] = run_b6_genetic_search(
                family=family,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                population_size=b6_ga_population,
                generations=b6_ga_generations,
                finalists=b6_finalists,
                workers=b6_workers,
                reward_profile=reward_profile,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                screening_training_episodes=max(1, int(b6_training_episodes) // 2),
                confirmation_training_episodes=b6_training_episodes,
            )

    risk_attempts = [
        attempt
        for attempt in fixed_attempts
        if attempt.get("controller_family") == "risk_corridor"
    ]
    recurrent_attempts = [
        attempt
        for attempt in fixed_attempts
        if attempt.get("controller_family") == "recurrent_memory"
    ]
    for family, attempts in (
        ("risk_corridor", risk_attempts),
        ("recurrent_memory", recurrent_attempts),
    ):
        search = genetic_searches.get(family)
        if isinstance(search, dict):
            accepted = search.get("accepted_attempt")
            if isinstance(accepted, dict):
                attempts.append(accepted)
            for attempt in search.get("confirmed_attempts", []) or []:
                if isinstance(attempt, dict) and attempt is not accepted:
                    attempts.append(attempt)

    accepted_risk = [
        attempt for attempt in risk_attempts if attempt.get("status") == "accepted"
    ]
    accepted_recurrent = [
        attempt
        for attempt in recurrent_attempts
        if attempt.get("status") == "accepted"
    ]
    best_risk = max(accepted_risk, key=_b6_attempt_fitness) if accepted_risk else None
    best_recurrent = (
        max(accepted_recurrent, key=_b6_attempt_fitness)
        if accepted_recurrent
        else None
    )
    fusion_attempt = None
    if best_risk is not None and best_recurrent is not None:
        fused_params = dict(best_risk.get("controller_params", {}))
        fused_params.update(dict(best_recurrent.get("controller_params", {})))
        fused_params["b6_family"] = 3.0
        fusion_attempt = run_b6_risk_corridor_attempt(
            B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b6_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_family="fused_risk_recurrent",
            controller_profile="fused_risk_recurrent",
            controller_params=fused_params,
            candidate_id="fusion_from_best_families",
            promote_if_accepted=False,
        )

    selected = select_b6_promotion(risk_attempts, recurrent_attempts, fusion_attempt)
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b6_risk_corridor_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b6_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_family=str(selected.get("controller_family")),
            controller_profile=str(selected.get("controller_profile")),
            controller_params=dict(selected.get("controller_params", {})),
            candidate_id=None,
            promote_if_accepted=True,
        )
    attempts = [*fixed_attempts]
    for search in genetic_searches.values():
        if isinstance(search, dict):
            accepted = search.get("accepted_attempt")
            if isinstance(accepted, dict):
                attempts.append(accepted)
    if fusion_attempt is not None:
        attempts.append(fusion_attempt)
    if promoted_attempt is not None and promoted_attempt not in attempts:
        attempts.append(promoted_attempt)
    accepted_variant = (
        str(promoted_attempt["variant"])
        if isinstance(promoted_attempt, dict)
        and promoted_attempt.get("status") == "accepted"
        else None
    )
    summary = {
        "status": "accepted" if accepted_variant is not None else "discarded",
        "accepted_variant": accepted_variant,
        "accepted_family": (
            promoted_attempt.get("controller_family")
            if isinstance(promoted_attempt, dict)
            else None
        ),
        "accepted_checkpoint": (
            promoted_attempt.get("checkpoint")
            if isinstance(promoted_attempt, dict)
            else None
        ),
        "b5_source": b5_report,
        "b6_search": b6_search,
        "b6_workers": int(b6_workers),
        "b6_partial_gate": {
            "food_predator_scenario": B6_FOOD_PREDATOR_SCENARIO,
            "corridor_scenario": B6_CORRIDOR_SCENARIO,
            "corridor_baseline_steps": B6_CORRIDOR_BASELINE_STEPS,
        },
        "risk_attempts": risk_attempts,
        "recurrent_attempts": recurrent_attempts,
        "fusion_attempt": fusion_attempt,
        "attempts": attempts,
        "genetic_searches": genetic_searches,
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B6 local hypotheses and try explicit observation/affordance expansion for corridor-risk in B6/B7 from the accepted B5 source"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b6_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b7_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b7_affordance_budget_attempt(
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
    config = build_b7_affordance_budget_config(
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
    corridor_gate = b7_corridor_viability_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b7_attempt_dir(root, variant_name, seed, candidate_id)
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
        "easy_evaluation_episodes": list(B4_EASY_EVALUATION_EPISODES),
        "canonical_evaluation_episodes": list(B4_CANONICAL_EVALUATION_EPISODES),
        "probe_evaluation_episodes": list(B6_PROBE_EVALUATION_EPISODES),
        "reward_profile": reward_profile,
        "operational_profile": operational_profile,
        "noise_profile": noise_profile,
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
