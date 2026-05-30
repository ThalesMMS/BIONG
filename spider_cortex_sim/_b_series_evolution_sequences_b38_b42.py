from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_config_builders import (
    build_b39_attention_binding_config,
    build_b40_global_workspace_config,
    build_b41_executive_workspace_config,
    build_b42_error_monitor_config,
)

from ._b_series_evolution_gates_b1_b6 import (
    b4_canonical_multi_gate_result,
    b4_easy_multi_gate_result,
    b5_food_deprivation_gate_result,
    b5_sleep_conflict_gate_result,
    b6_food_predator_conflict_gate_result,
)

from ._b_series_evolution_gates_b31_b40 import (
    b39_attention_binding_corridor_gate_result,
    b40_global_workspace_corridor_gate_result,
)

from ._b_series_evolution_gates_b41_b50 import (
    b41_executive_workspace_corridor_gate_result,
    b42_error_monitor_corridor_gate_result,
)

from ._b_series_evolution_gates_b61_requires import (
    _make_simulation,
    require_b37_state_factor_gate_checkpoint,
    require_b38_factor_attention_checkpoint,
    require_b39_attention_binding_checkpoint,
    require_b40_global_workspace_checkpoint,
)

from ._b_series_evolution_requires_sequences_b1_b5 import (
    _run_episode_payload,
)

from ._b_series_evolution_sequences_b34_b38 import (
    _b38_attempt_fitness,
    run_b38_factor_attention_attempt,
)

def run_b38_factor_attention_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b38_training_episodes: int = 64,
    b38_workers: int = 1,
    b38_search: str = "hybrid",
    b38_ga_population: int = 24,
    b38_ga_generations: int = 8,
    b38_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b38_ga_population, b38_ga_generations, b38_finalists
    b37_report = require_b37_state_factor_gate_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b37_report["checkpoint"])
    jobs = list(B38_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b38_search in {"fixed", "hybrid"} and int(b38_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b38_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b38_factor_attention_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b38_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b38_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b38_factor_attention_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b38_training_episodes,
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
    if selected is None and b38_search in {"ga", "hybrid"}:
        fallback_attempt = run_b38_factor_attention_attempt(
            B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b38_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_factor_attention",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b38_factor_attention_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b38_training_episodes,
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
        "b37_source": b37_report,
        "b38_search": b38_search,
        "b38_workers": int(b38_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b38_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B38 factor-attention line and try explicit factor heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b38_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b39_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b39_attention_binding_attempt(
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
    config = build_b39_attention_binding_config(
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
    corridor_gate = b39_attention_binding_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b39_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b39_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("binding_strength_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("cross_factor_coherence_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("bound_context_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("binding_gain_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("binding_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b39_attention_binding_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b39_training_episodes: int = 64,
    b39_workers: int = 1,
    b39_search: str = "hybrid",
    b39_ga_population: int = 24,
    b39_ga_generations: int = 8,
    b39_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b39_ga_population, b39_ga_generations, b39_finalists
    b38_report = require_b38_factor_attention_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b38_report["checkpoint"])
    jobs = list(B39_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b39_search in {"fixed", "hybrid"} and int(b39_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b39_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b39_attention_binding_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b39_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b39_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b39_attention_binding_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b39_training_episodes,
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
    if selected is None and b39_search in {"ga", "hybrid"}:
        fallback_attempt = run_b39_attention_binding_attempt(
            B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b39_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_attention_binding",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b39_attention_binding_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b39_training_episodes,
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
        "b38_source": b38_report,
        "b39_search": b39_search,
        "b39_workers": int(b39_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b39_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B39 attention-binding line and try explicit binding heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b39_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b40_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b40_global_workspace_attempt(
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
    config = build_b40_global_workspace_config(
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
    corridor_gate = b40_global_workspace_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b40_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b40_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("workspace_activation_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("broadcast_gain_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("context_availability_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("workspace_stability_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("workspace_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b40_global_workspace_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b40_training_episodes: int = 64,
    b40_workers: int = 1,
    b40_search: str = "hybrid",
    b40_ga_population: int = 24,
    b40_ga_generations: int = 8,
    b40_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b40_ga_population, b40_ga_generations, b40_finalists
    b39_report = require_b39_attention_binding_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b39_report["checkpoint"])
    jobs = list(B40_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b40_search in {"fixed", "hybrid"} and int(b40_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b40_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b40_global_workspace_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b40_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b40_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b40_global_workspace_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b40_training_episodes,
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
    if selected is None and b40_search in {"ga", "hybrid"}:
        fallback_attempt = run_b40_global_workspace_attempt(
            B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b40_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_global_workspace",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b40_global_workspace_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b40_training_episodes,
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
        "b39_source": b39_report,
        "b40_search": b40_search,
        "b40_workers": int(b40_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b40_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B40 global-workspace line and try explicit broadcast heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b40_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b41_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b41_executive_workspace_attempt(
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
    config = build_b41_executive_workspace_config(
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
    corridor_gate = b41_executive_workspace_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b41_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b41_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("executive_selection_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("inhibitory_pressure_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("goal_context_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("executive_stability_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("executive_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score


def run_b41_executive_workspace_sequence(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
    b41_training_episodes: int = 64,
    b41_workers: int = 1,
    b41_search: str = "hybrid",
    b41_ga_population: int = 24,
    b41_ga_generations: int = 8,
    b41_finalists: int = 6,
    reward_profile: str = "ecological",
    operational_profile: str = "default_v1",
    noise_profile: str = "none",
) -> dict[str, object]:
    del b41_ga_population, b41_ga_generations, b41_finalists
    b40_report = require_b40_global_workspace_checkpoint(root=root, seed=seed)
    source_checkpoint = str(b40_report["checkpoint"])
    jobs = list(B41_FIXED_EVOLUTION_ATTEMPTS)
    attempts_by_name: dict[str, dict[str, object]] = {}
    if b41_search in {"fixed", "hybrid"} and int(b41_workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(int(b41_workers), len(jobs))) as pool:
            future_map = {
                pool.submit(
                    run_b41_executive_workspace_attempt,
                    variant_name,
                    source_checkpoint=source_checkpoint,
                    root=root,
                    seed=seed,
                    training_episodes=b41_training_episodes,
                    reward_profile=reward_profile,
                    operational_profile=operational_profile,
                    noise_profile=noise_profile,
                    promote_if_accepted=False,
                ): variant_name
                for variant_name in jobs
            }
            for future in as_completed(future_map):
                attempts_by_name[future_map[future]] = future.result()
    if b41_search in {"fixed", "hybrid"} and len(attempts_by_name) < len(jobs):
        for variant_name in jobs:
            if variant_name in attempts_by_name:
                continue
            attempts_by_name[variant_name] = run_b41_executive_workspace_attempt(
                variant_name,
                source_checkpoint=source_checkpoint,
                root=root,
                seed=seed,
                training_episodes=b41_training_episodes,
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
    if selected is None and b41_search in {"ga", "hybrid"}:
        fallback_attempt = run_b41_executive_workspace_attempt(
            B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b41_training_episodes,
            reward_profile=reward_profile,
            operational_profile=operational_profile,
            noise_profile=noise_profile,
            controller_profile="genetic_executive_workspace",
            candidate_id="fallback_default",
            promote_if_accepted=False,
        )
        if fallback_attempt.get("status") == "accepted":
            selected = fallback_attempt
    promoted_attempt = None
    if selected is not None:
        promoted_attempt = run_b41_executive_workspace_attempt(
            str(selected["variant"]),
            source_checkpoint=source_checkpoint,
            root=root,
            seed=seed,
            training_episodes=b41_training_episodes,
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
        "b40_source": b40_report,
        "b41_search": b41_search,
        "b41_workers": int(b41_workers),
        "attempts": attempts,
        "fixed_attempts": fixed_attempts,
        "fallback_attempt": fallback_attempt,
        "best_attempt": (
            max(attempts, key=_b41_attempt_fitness) if attempts else None
        ),
        "next_recommendation": (
            None
            if accepted_variant is not None
            else "discard B41 executive-workspace line and try explicit control heads"
        ),
    }
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "b41_evolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _b42_attempt_dir(
    root: str | Path,
    variant_name: str,
    seed: int,
    candidate_id: str | None,
) -> Path:
    base = Path(root) / variant_name / f"seed_{int(seed)}"
    if candidate_id:
        return base / "ga_search" / str(candidate_id)
    return base


def run_b42_error_monitor_attempt(
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
    config = build_b42_error_monitor_config(
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
    corridor_gate = b42_error_monitor_corridor_gate_result(corridor_results)
    accepted = bool(
        easy_gate["passed"]
        and canonical_gate["passed"]
        and food_deprivation_gate["passed"]
        and sleep_conflict_gate["passed"]
        and food_predator_gate["passed"]
        and corridor_gate["passed"]
    )
    attempt_dir = _b42_attempt_dir(root, variant_name, seed, candidate_id)
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


def _b42_attempt_fitness(attempt: dict[str, object]) -> float:
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
        + int(corridor_aggregate.get("error_signal_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("conflict_signal_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("performance_context_episodes", 0) or 0) * 1000.0
        + int(corridor_aggregate.get("monitor_stability_episodes", 0) or 0) * 750.0
        + int(corridor_aggregate.get("monitor_lock_episodes", 0) or 0) * 500.0
        + int(canonical_aggregate.get("completed_horizons", 0) or 0) * 500.0
        - int(canonical_aggregate.get("total_predator_contacts", 999) or 999) * 10.0
    )
    if str(attempt.get("status")) == "accepted":
        score += 100000.0
    return score
