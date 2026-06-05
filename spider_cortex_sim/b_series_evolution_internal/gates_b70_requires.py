from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b69_requires import b70_optic_flow_corridor_gate_result


def b71_tectal_orienting_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b70_optic_flow_corridor_gate_result(results)
    explicit_decision_set = set(B71_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    target_salience_episodes = 0
    orienting_gain_episodes = 0
    collision_veto_episodes = 0
    lock_or_release_episodes = 0
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
            str(item.get("b71_decision"))
            for item in trace
            if item.get("b71_decision") is not None
        ]
        target_saliences = [
            float(item.get("b71_target_salience", 0.0) or 0.0)
            for item in trace
            if item.get("b71_target_salience") is not None
        ]
        orienting_gains = [
            float(item.get("b71_orienting_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b71_orienting_gain") is not None
        ]
        collision_vetoes = [
            float(item.get("b71_collision_veto", 0.0) or 0.0)
            for item in trace
            if item.get("b71_collision_veto") is not None
        ]
        locks = [
            int(item.get("b71_orienting_lock", 0) or 0)
            for item in trace
            if item.get("b71_orienting_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        target_salience = any(abs(value) > 0.0 for value in target_saliences)
        orienting_gain = any(abs(value) > 0.0 for value in orienting_gains)
        collision_veto = any(abs(value) > 0.0 for value in collision_vetoes)
        lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "tectal_recenter",
                "continue_orienting_lock",
                "salient_target_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if target_salience:
            target_salience_episodes += 1
        if orienting_gain:
            orienting_gain_episodes += 1
        if collision_veto:
            collision_veto_episodes += 1
        if lock_or_release:
            lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b71_decision": bool(explicit_decision),
                    "target_salience": bool(target_salience),
                    "orienting_gain": bool(orienting_gain),
                    "collision_veto": bool(collision_veto),
                    "lock_or_release": bool(lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "target_saliences": target_saliences,
                "orienting_gains": orienting_gains,
                "collision_vetoes": collision_vetoes,
                "orienting_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b70_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b71_decision_episodes": explicit_decision_episodes >= 2,
        "target_salience_episodes": target_salience_episodes >= 2,
        "orienting_gain_episodes": orienting_gain_episodes >= 2,
        "collision_veto_episodes": collision_veto_episodes >= 2,
        "lock_or_release_episodes": lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b71_aggregate:" + name
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
            "base_b70_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "target_salience_episodes": int(target_salience_episodes),
            "orienting_gain_episodes": int(orienting_gain_episodes),
            "collision_veto_episodes": int(collision_veto_episodes),
            "lock_or_release_episodes": int(lock_or_release_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
