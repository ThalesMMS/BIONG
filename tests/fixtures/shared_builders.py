"""Shared builders for metrics and behavior-evaluation tests."""

from __future__ import annotations

from spider_cortex_sim.metrics import (
    BehaviorCheckResult,
    BehavioralEpisodeScore,
    EpisodeStats,
)
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.reward import REWARD_COMPONENT_NAMES


def make_episode_stats(**overrides) -> EpisodeStats:
    defaults = dict(
        episode=0,
        seed=42,
        training=False,
        scenario="test_scenario",
        total_reward=1.0,
        steps=10,
        food_eaten=1,
        sleep_events=0,
        shelter_entries=1,
        alert_events=0,
        predator_contacts=0,
        predator_sightings=0,
        predator_escapes=0,
        night_ticks=5,
        night_shelter_ticks=5,
        night_still_ticks=5,
        night_role_ticks={"outside": 0, "entrance": 0, "inside": 0, "deep": 5},
        night_shelter_occupancy_rate=1.0,
        night_stillness_rate=1.0,
        night_role_distribution={
            "outside": 0.0,
            "entrance": 0.0,
            "inside": 0.0,
            "deep": 1.0,
        },
        predator_response_events=0,
        mean_predator_response_latency=0.0,
        mean_sleep_debt=0.3,
        food_distance_delta=2.0,
        shelter_distance_delta=0.0,
        final_hunger=0.4,
        final_fatigue=0.3,
        final_sleep_debt=0.15,
        final_health=1.0,
        alive=True,
        reward_component_totals={k: 0.0 for k in REWARD_COMPONENT_NAMES},
        predator_state_ticks={s: 0 for s in PREDATOR_STATES},
        predator_mode_transitions=0,
        dominant_predator_state="PATROL",
    )
    defaults.update(overrides)
    return EpisodeStats(**defaults)


def make_behavior_check_result(
    name: str = "check_a",
    passed: bool = True,
    value: object | None = 1.0,
    description: str = "desc",
    expected: str = "true",
) -> BehaviorCheckResult:
    return BehaviorCheckResult(
        name=name,
        description=description,
        expected=expected,
        passed=passed,
        value=value,
    )


def make_behavioral_episode_score(
    episode=0,
    seed=42,
    scenario="test_scenario",
    objective="test_objective",
    success=True,
    checks=None,
    behavior_metrics=None,
    failures=None,
) -> BehavioralEpisodeScore:
    if checks is None:
        checks = {}
    if behavior_metrics is None:
        behavior_metrics = {}
    if failures is None:
        failures = []
    return BehavioralEpisodeScore(
        episode=episode,
        seed=seed,
        scenario=scenario,
        objective=objective,
        success=success,
        checks=checks,
        behavior_metrics=behavior_metrics,
        failures=failures,
    )
