from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b73_requires import b74_thalamic_rebound_corridor_gate_result


def b75_basal_thalamic_release_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b74_thalamic_rebound_corridor_gate_result(results)
    explicit_decision_set = set(B75_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    release_timing_drive_episodes = 0
    go_disinhibition_episodes = 0
    nogo_brake_episodes = 0
    burst_window_episodes = 0
    release_lock_or_stride_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = result_predator_contacts(result)
        decisions = [
            str(item.get("b75_decision"))
            for item in trace
            if item.get("b75_decision") is not None
        ]
        timing_drives = [
            float(item.get("b75_release_timing_drive", 0.0) or 0.0)
            for item in trace
            if item.get("b75_release_timing_drive") is not None
        ]
        go_values = [
            float(item.get("b75_go_disinhibition", 0.0) or 0.0)
            for item in trace
            if item.get("b75_go_disinhibition") is not None
        ]
        nogo_values = [
            float(item.get("b75_nogo_brake", 0.0) or 0.0)
            for item in trace
            if item.get("b75_nogo_brake") is not None
        ]
        burst_windows = [
            float(item.get("b75_burst_window", 0.0) or 0.0)
            for item in trace
            if item.get("b75_burst_window") is not None
        ]
        locks = [
            int(item.get("b75_release_lock", 0) or 0)
            for item in trace
            if item.get("b75_release_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        release_timing_drive = any(abs(value) > 0.0 for value in timing_drives)
        go_disinhibition = any(abs(value) > 0.0 for value in go_values)
        nogo_brake = any(abs(value) > 0.0 for value in nogo_values)
        burst_window = any(abs(value) > 0.0 for value in burst_windows)
        release_lock_or_stride = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "basal_thalamic_hold",
                "continue_release_lock",
                "timed_rebound_stride",
                "go_disinhibition_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if release_timing_drive:
            release_timing_drive_episodes += 1
        if go_disinhibition:
            go_disinhibition_episodes += 1
        if nogo_brake:
            nogo_brake_episodes += 1
        if burst_window:
            burst_window_episodes += 1
        if release_lock_or_stride:
            release_lock_or_stride_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b75_decision": bool(explicit_decision),
                    "release_timing_drive": bool(release_timing_drive),
                    "go_disinhibition": bool(go_disinhibition),
                    "nogo_brake": bool(nogo_brake),
                    "burst_window": bool(burst_window),
                    "release_lock_or_stride": bool(release_lock_or_stride),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "release_timing_drives": timing_drives,
                "go_disinhibitions": go_values,
                "nogo_brakes": nogo_values,
                "burst_windows": burst_windows,
                "release_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b74_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b75_decision_episodes": explicit_decision_episodes >= 2,
        "release_timing_drive_episodes": release_timing_drive_episodes >= 2,
        "go_disinhibition_episodes": go_disinhibition_episodes >= 2,
        "nogo_brake_episodes": nogo_brake_episodes >= 2,
        "burst_window_episodes": burst_window_episodes >= 2,
        "release_lock_or_stride_episodes": release_lock_or_stride_episodes >= 2,
    }
    failures.extend(
        "corridor_b75_aggregate:" + name
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
            "base_b74_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "release_timing_drive_episodes": int(release_timing_drive_episodes),
            "go_disinhibition_episodes": int(go_disinhibition_episodes),
            "nogo_brake_episodes": int(nogo_brake_episodes),
            "burst_window_episodes": int(burst_window_episodes),
            "release_lock_or_stride_episodes": int(
                release_lock_or_stride_episodes
            ),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
