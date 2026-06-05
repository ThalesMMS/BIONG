from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b71_requires import b72_pulvinar_attention_corridor_gate_result


def b73_reticular_inhibition_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b72_pulvinar_attention_corridor_gate_result(results)
    explicit_decision_set = set(B73_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    inhibitory_tone_episodes = 0
    surround_suppression_episodes = 0
    release_drive_episodes = 0
    inhibition_lock_or_release_episodes = 0
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
            str(item.get("b73_decision"))
            for item in trace
            if item.get("b73_decision") is not None
        ]
        inhibitory_tones = [
            float(item.get("b73_inhibitory_tone", 0.0) or 0.0)
            for item in trace
            if item.get("b73_inhibitory_tone") is not None
        ]
        surround_suppressions = [
            float(item.get("b73_surround_suppression", 0.0) or 0.0)
            for item in trace
            if item.get("b73_surround_suppression") is not None
        ]
        release_drives = [
            float(item.get("b73_release_drive", 0.0) or 0.0)
            for item in trace
            if item.get("b73_release_drive") is not None
        ]
        locks = [
            int(item.get("b73_inhibition_lock", 0) or 0)
            for item in trace
            if item.get("b73_inhibition_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        inhibitory_tone = any(abs(value) > 0.0 for value in inhibitory_tones)
        surround_suppression = any(abs(value) > 0.0 for value in surround_suppressions)
        release_drive = any(abs(value) > 0.0 for value in release_drives)
        inhibition_lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "reticular_surround_hold",
                "continue_inhibition_lock",
                "reticular_focus_release",
                "suppress_distractor_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if inhibitory_tone:
            inhibitory_tone_episodes += 1
        if surround_suppression:
            surround_suppression_episodes += 1
        if release_drive:
            release_drive_episodes += 1
        if inhibition_lock_or_release:
            inhibition_lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b73_decision": bool(explicit_decision),
                    "inhibitory_tone": bool(inhibitory_tone),
                    "surround_suppression": bool(surround_suppression),
                    "release_drive": bool(release_drive),
                    "inhibition_lock_or_release": bool(inhibition_lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "inhibitory_tones": inhibitory_tones,
                "surround_suppressions": surround_suppressions,
                "release_drives": release_drives,
                "inhibition_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b72_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b73_decision_episodes": explicit_decision_episodes >= 2,
        "inhibitory_tone_episodes": inhibitory_tone_episodes >= 2,
        "surround_suppression_episodes": surround_suppression_episodes >= 2,
        "release_drive_episodes": release_drive_episodes >= 2,
        "inhibition_lock_or_release_episodes": inhibition_lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b73_aggregate:" + name
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
            "base_b72_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "inhibitory_tone_episodes": int(inhibitory_tone_episodes),
            "surround_suppression_episodes": int(surround_suppression_episodes),
            "release_drive_episodes": int(release_drive_episodes),
            "inhibition_lock_or_release_episodes": int(inhibition_lock_or_release_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
