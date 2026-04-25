from __future__ import annotations

from .common import *

def _score_food_vs_predator_conflict(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Score whether predator threat takes precedence over food-seeking and whether the agent avoids predator contact.
    
    Extracts action-selection payloads from the provided trace, identifies payloads that indicate a dangerous predator, and computes three checks: (1) threat prioritization in selected valence, (2) suppression of hunger gating when threat is selected, and (3) survival with zero predator contacts. The returned score contains the scenario objective, the three checks, and behavior metrics such as danger_tick_count, threat_priority_rate, foraging_suppressed_rate, mean_hunger_gate_under_threat, predator_contacts, and alive.
    
    Parameters:
        stats (EpisodeStats): Episode summary statistics used for survival and contact checks.
        trace (Sequence[Dict[str, object]]): Execution trace from which action-selection payloads are extracted.
    
    Returns:
        BehavioralEpisodeScore: Score object containing the objective, the three behavior checks, and aggregated behavior metrics.
    """
    payloads = _trace_action_selection_payloads(trace)
    dangerous_payloads: list[Dict[str, object]] = []
    for payload in payloads:
        predator_visible = _payload_float(payload, "evidence", "threat", "predator_visible")
        predator_proximity = _payload_float(payload, "evidence", "threat", "predator_proximity")
        predator_certainty = _payload_float(payload, "evidence", "threat", "predator_certainty")
        if predator_visible is None or predator_proximity is None or predator_certainty is None:
            continue
        if predator_visible >= 0.5 and (
            predator_proximity >= 0.3 or predator_certainty >= 0.35
        ):
            dangerous_payloads.append(payload)
    threat_priority_rate = float(
        sum(
            1.0
            for payload in dangerous_payloads
            if _payload_text(payload, "winning_valence") == "threat"
        ) / max(1, len(dangerous_payloads))
    )
    dangerous_hunger_gates = [
        hunger_gate
        for payload in dangerous_payloads
        if (hunger_gate := _payload_float(payload, "module_gates", "hunger_center")) is not None
    ]
    threat_wins = sum(
        1
        for payload in dangerous_payloads
        if _payload_text(payload, "winning_valence") == "threat"
    )
    foraging_suppressed_rate = float(
        sum(
            1.0
            for payload in dangerous_payloads
            if _payload_text(payload, "winning_valence") == "threat"
            and (hunger_gate := _payload_float(payload, "module_gates", "hunger_center")) is not None
            and hunger_gate < 0.5
        ) / max(1, threat_wins)
    )
    threat_priority = bool(
        len(dangerous_payloads) >= 1 and threat_priority_rate >= CONFLICT_PASS_RATE
    )
    foraging_suppressed = bool(
        len(dangerous_hunger_gates) >= 1
        and foraging_suppressed_rate >= CONFLICT_PASS_RATE
    )
    survives_without_contact = bool(stats.alive and int(stats.predator_contacts) == 0)
    checks = (
        build_behavior_check(FOOD_VS_PREDATOR_CONFLICT_CHECKS[0], passed=threat_priority, value=threat_priority),
        build_behavior_check(FOOD_VS_PREDATOR_CONFLICT_CHECKS[1], passed=foraging_suppressed, value=foraging_suppressed),
        build_behavior_check(
            FOOD_VS_PREDATOR_CONFLICT_CHECKS[2],
            passed=survives_without_contact,
            value={
                "alive": bool(stats.alive),
                "predator_contacts": int(stats.predator_contacts),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate that threat overrides food drive when both compete at the same moment.",
        checks=checks,
        behavior_metrics={
            "danger_tick_count": len(dangerous_payloads),
            "threat_priority_rate": threat_priority_rate,
            "foraging_suppressed_rate": foraging_suppressed_rate,
            "mean_hunger_gate_under_threat": float(
                sum(dangerous_hunger_gates) / max(1, len(dangerous_hunger_gates))
            ),
            "predator_contacts": int(stats.predator_contacts),
            "alive": bool(stats.alive),
        },
    )

__all__ = ["_score_food_vs_predator_conflict"]
