from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_gates_b1_b6 import (
    trace_uses_only_primitive_actions,
)

from ._b_series_evolution_gates_b41_b50 import (
    b50_habit_chunking_corridor_gate_result,
)

def b51_dopaminergic_habit_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b50_habit_chunking_corridor_gate_result(results)
    explicit_decision_set = set(B51_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    prediction_error_episodes = 0
    dopamine_gain_episodes = 0
    habit_modulation_episodes = 0
    modulation_lock_episodes = 0
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
            str(item.get("b51_decision"))
            for item in trace
            if item.get("b51_decision") is not None
        ]
        prediction_errors = [
            float(item.get("b51_prediction_error", 0.0) or 0.0)
            for item in trace
            if item.get("b51_prediction_error") is not None
        ]
        dopamine_gains = [
            float(item.get("b51_dopamine_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b51_dopamine_gain") is not None
        ]
        habit_modulations = [
            float(item.get("b51_habit_modulation", 0.0) or 0.0)
            for item in trace
            if item.get("b51_habit_modulation") is not None
        ]
        locks = [
            int(item.get("b51_modulation_lock", 0) or 0)
            for item in trace
            if item.get("b51_modulation_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        prediction_error = any(abs(value) > 0.0 for value in prediction_errors)
        dopamine_gain = any(abs(value) > 0.0 for value in dopamine_gains)
        habit_modulation = any(abs(value) > 0.0 for value in habit_modulations)
        modulation_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if prediction_error:
            prediction_error_episodes += 1
        if dopamine_gain:
            dopamine_gain_episodes += 1
        if habit_modulation:
            habit_modulation_episodes += 1
        if modulation_lock:
            modulation_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b51_decision": bool(explicit_decision),
                    "prediction_error": bool(prediction_error),
                    "dopamine_gain": bool(dopamine_gain),
                    "habit_modulation": bool(habit_modulation),
                    "modulation_lock": bool(modulation_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "prediction_errors": prediction_errors,
                "dopamine_gains": dopamine_gains,
                "habit_modulations": habit_modulations,
                "modulation_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b50_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b51_decision_episodes": explicit_decision_episodes >= 2,
        "prediction_error_episodes": prediction_error_episodes >= 2,
        "dopamine_gain_episodes": dopamine_gain_episodes >= 2,
        "habit_modulation_episodes": habit_modulation_episodes >= 2,
        "modulation_lock_episodes": modulation_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b51_aggregate:" + name
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
            "base_b50_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "prediction_error_episodes": int(prediction_error_episodes),
            "dopamine_gain_episodes": int(dopamine_gain_episodes),
            "habit_modulation_episodes": int(habit_modulation_episodes),
            "modulation_lock_episodes": int(modulation_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b52_cholinergic_precision_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b51_dopaminergic_habit_corridor_gate_result(results)
    explicit_decision_set = set(B52_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    acetylcholine_episodes = 0
    precision_gain_episodes = 0
    uncertainty_signal_episodes = 0
    attention_lock_episodes = 0
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
            str(item.get("b52_decision"))
            for item in trace
            if item.get("b52_decision") is not None
        ]
        acetylcholine_levels = [
            float(item.get("b52_acetylcholine_level", 0.0) or 0.0)
            for item in trace
            if item.get("b52_acetylcholine_level") is not None
        ]
        precision_gains = [
            float(item.get("b52_precision_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b52_precision_gain") is not None
        ]
        uncertainty_signals = [
            float(item.get("b52_uncertainty_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b52_uncertainty_signal") is not None
        ]
        locks = [
            int(item.get("b52_attention_lock", 0) or 0)
            for item in trace
            if item.get("b52_attention_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        acetylcholine = any(abs(value) > 0.0 for value in acetylcholine_levels)
        precision_gain = any(abs(value) > 0.0 for value in precision_gains)
        uncertainty_signal = any(abs(value) > 0.0 for value in uncertainty_signals)
        attention_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if acetylcholine:
            acetylcholine_episodes += 1
        if precision_gain:
            precision_gain_episodes += 1
        if uncertainty_signal:
            uncertainty_signal_episodes += 1
        if attention_lock:
            attention_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b52_decision": bool(explicit_decision),
                    "acetylcholine_level": bool(acetylcholine),
                    "precision_gain": bool(precision_gain),
                    "uncertainty_signal": bool(uncertainty_signal),
                    "attention_lock": bool(attention_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "acetylcholine_levels": acetylcholine_levels,
                "precision_gains": precision_gains,
                "uncertainty_signals": uncertainty_signals,
                "attention_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b51_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b52_decision_episodes": explicit_decision_episodes >= 2,
        "acetylcholine_level_episodes": acetylcholine_episodes >= 2,
        "precision_gain_episodes": precision_gain_episodes >= 2,
        "uncertainty_signal_episodes": uncertainty_signal_episodes >= 2,
        "attention_lock_episodes": attention_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b52_aggregate:" + name
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
            "base_b51_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "acetylcholine_level_episodes": int(acetylcholine_episodes),
            "precision_gain_episodes": int(precision_gain_episodes),
            "uncertainty_signal_episodes": int(uncertainty_signal_episodes),
            "attention_lock_episodes": int(attention_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b53_noradrenergic_arousal_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b52_cholinergic_precision_corridor_gate_result(results)
    explicit_decision_set = set(B53_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    norepinephrine_episodes = 0
    arousal_gain_episodes = 0
    surprise_signal_episodes = 0
    gain_lock_episodes = 0
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
            str(item.get("b53_decision"))
            for item in trace
            if item.get("b53_decision") is not None
        ]
        norepinephrine_levels = [
            float(item.get("b53_norepinephrine_level", 0.0) or 0.0)
            for item in trace
            if item.get("b53_norepinephrine_level") is not None
        ]
        arousal_gains = [
            float(item.get("b53_arousal_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b53_arousal_gain") is not None
        ]
        surprise_signals = [
            float(item.get("b53_surprise_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b53_surprise_signal") is not None
        ]
        locks = [
            int(item.get("b53_gain_lock", 0) or 0)
            for item in trace
            if item.get("b53_gain_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        norepinephrine = any(abs(value) > 0.0 for value in norepinephrine_levels)
        arousal_gain = any(abs(value) > 0.0 for value in arousal_gains)
        surprise_signal = any(abs(value) > 0.0 for value in surprise_signals)
        gain_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if norepinephrine:
            norepinephrine_episodes += 1
        if arousal_gain:
            arousal_gain_episodes += 1
        if surprise_signal:
            surprise_signal_episodes += 1
        if gain_lock:
            gain_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b53_decision": bool(explicit_decision),
                    "norepinephrine_level": bool(norepinephrine),
                    "arousal_gain": bool(arousal_gain),
                    "surprise_signal": bool(surprise_signal),
                    "gain_lock": bool(gain_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "norepinephrine_levels": norepinephrine_levels,
                "arousal_gains": arousal_gains,
                "surprise_signals": surprise_signals,
                "gain_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b52_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b53_decision_episodes": explicit_decision_episodes >= 2,
        "norepinephrine_level_episodes": norepinephrine_episodes >= 2,
        "arousal_gain_episodes": arousal_gain_episodes >= 2,
        "surprise_signal_episodes": surprise_signal_episodes >= 2,
        "gain_lock_episodes": gain_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b53_aggregate:" + name
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
            "base_b52_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "norepinephrine_level_episodes": int(norepinephrine_episodes),
            "arousal_gain_episodes": int(arousal_gain_episodes),
            "surprise_signal_episodes": int(surprise_signal_episodes),
            "gain_lock_episodes": int(gain_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b54_serotonergic_patience_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b53_noradrenergic_arousal_corridor_gate_result(results)
    explicit_decision_set = set(B54_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    serotonin_episodes = 0
    patience_signal_episodes = 0
    impulse_suppression_episodes = 0
    patience_lock_episodes = 0
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
            str(item.get("b54_decision"))
            for item in trace
            if item.get("b54_decision") is not None
        ]
        serotonin_levels = [
            float(item.get("b54_serotonin_level", 0.0) or 0.0)
            for item in trace
            if item.get("b54_serotonin_level") is not None
        ]
        patience_signals = [
            float(item.get("b54_patience_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b54_patience_signal") is not None
        ]
        impulse_suppressions = [
            float(item.get("b54_impulse_suppression", 0.0) or 0.0)
            for item in trace
            if item.get("b54_impulse_suppression") is not None
        ]
        locks = [
            int(item.get("b54_patience_lock", 0) or 0)
            for item in trace
            if item.get("b54_patience_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        serotonin = any(abs(value) > 0.0 for value in serotonin_levels)
        patience_signal = any(abs(value) > 0.0 for value in patience_signals)
        impulse_suppression = any(abs(value) > 0.0 for value in impulse_suppressions)
        patience_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if serotonin:
            serotonin_episodes += 1
        if patience_signal:
            patience_signal_episodes += 1
        if impulse_suppression:
            impulse_suppression_episodes += 1
        if patience_lock:
            patience_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b54_decision": bool(explicit_decision),
                    "serotonin_level": bool(serotonin),
                    "patience_signal": bool(patience_signal),
                    "impulse_suppression": bool(impulse_suppression),
                    "patience_lock": bool(patience_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "serotonin_levels": serotonin_levels,
                "patience_signals": patience_signals,
                "impulse_suppressions": impulse_suppressions,
                "patience_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b53_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b54_decision_episodes": explicit_decision_episodes >= 2,
        "serotonin_level_episodes": serotonin_episodes >= 2,
        "patience_signal_episodes": patience_signal_episodes >= 2,
        "impulse_suppression_episodes": impulse_suppression_episodes >= 2,
        "patience_lock_episodes": patience_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b54_aggregate:" + name
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
            "base_b53_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "serotonin_level_episodes": int(serotonin_episodes),
            "patience_signal_episodes": int(patience_signal_episodes),
            "impulse_suppression_episodes": int(impulse_suppression_episodes),
            "patience_lock_episodes": int(patience_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b55_hypothalamic_drive_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b54_serotonergic_patience_corridor_gate_result(results)
    explicit_decision_set = set(B55_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    drive_episodes = 0
    satiety_episodes = 0
    recovery_episodes = 0
    balance_episodes = 0
    drive_lock_episodes = 0
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
            str(item.get("b55_decision"))
            for item in trace
            if item.get("b55_decision") is not None
        ]
        drives = [
            float(item.get("b55_hypothalamic_drive", 0.0) or 0.0)
            for item in trace
            if item.get("b55_hypothalamic_drive") is not None
        ]
        satiety_signals = [
            float(item.get("b55_satiety_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b55_satiety_signal") is not None
        ]
        recovery_biases = [
            float(item.get("b55_recovery_bias", 0.0) or 0.0)
            for item in trace
            if item.get("b55_recovery_bias") is not None
        ]
        balances = [
            float(item.get("b55_drive_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b55_drive_balance") is not None
        ]
        locks = [
            int(item.get("b55_drive_lock", 0) or 0)
            for item in trace
            if item.get("b55_drive_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        drive = any(abs(value) > 0.0 for value in drives)
        satiety = any(abs(value) > 0.0 for value in satiety_signals)
        recovery = any(abs(value) > 0.0 for value in recovery_biases)
        balance = any(abs(value) > 0.0 for value in balances)
        drive_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if drive:
            drive_episodes += 1
        if satiety:
            satiety_episodes += 1
        if recovery:
            recovery_episodes += 1
        if balance:
            balance_episodes += 1
        if drive_lock:
            drive_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b55_decision": bool(explicit_decision),
                    "hypothalamic_drive": bool(drive),
                    "satiety_signal": bool(satiety),
                    "recovery_bias": bool(recovery),
                    "drive_balance": bool(balance),
                    "drive_lock": bool(drive_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "hypothalamic_drives": drives,
                "satiety_signals": satiety_signals,
                "recovery_biases": recovery_biases,
                "drive_balances": balances,
                "drive_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b54_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b55_decision_episodes": explicit_decision_episodes >= 2,
        "hypothalamic_drive_episodes": drive_episodes >= 2,
        "satiety_signal_episodes": satiety_episodes >= 2,
        "recovery_bias_episodes": recovery_episodes >= 2,
        "drive_balance_episodes": balance_episodes >= 2,
        "drive_lock_episodes": drive_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b55_aggregate:" + name
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
            "base_b54_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "hypothalamic_drive_episodes": int(drive_episodes),
            "satiety_signal_episodes": int(satiety_episodes),
            "recovery_bias_episodes": int(recovery_episodes),
            "drive_balance_episodes": int(balance_episodes),
            "drive_lock_episodes": int(drive_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b56_hpa_stress_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b55_hypothalamic_drive_corridor_gate_result(results)
    explicit_decision_set = set(B56_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    cortisol_episodes = 0
    stress_load_episodes = 0
    recovery_signal_episodes = 0
    endocrine_balance_episodes = 0
    stress_lock_episodes = 0
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
            str(item.get("b56_decision"))
            for item in trace
            if item.get("b56_decision") is not None
        ]
        cortisol_levels = [
            float(item.get("b56_cortisol_level", 0.0) or 0.0)
            for item in trace
            if item.get("b56_cortisol_level") is not None
        ]
        stress_loads = [
            float(item.get("b56_stress_load", 0.0) or 0.0)
            for item in trace
            if item.get("b56_stress_load") is not None
        ]
        recovery_signals = [
            float(item.get("b56_recovery_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b56_recovery_signal") is not None
        ]
        endocrine_balances = [
            float(item.get("b56_endocrine_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b56_endocrine_balance") is not None
        ]
        locks = [
            int(item.get("b56_stress_lock", 0) or 0)
            for item in trace
            if item.get("b56_stress_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        cortisol = any(abs(value) > 0.0 for value in cortisol_levels)
        stress_load = any(abs(value) > 0.0 for value in stress_loads)
        recovery_signal = any(abs(value) > 0.0 for value in recovery_signals)
        endocrine_balance = any(abs(value) > 0.0 for value in endocrine_balances)
        stress_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if cortisol:
            cortisol_episodes += 1
        if stress_load:
            stress_load_episodes += 1
        if recovery_signal:
            recovery_signal_episodes += 1
        if endocrine_balance:
            endocrine_balance_episodes += 1
        if stress_lock:
            stress_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b56_decision": bool(explicit_decision),
                    "cortisol_level": bool(cortisol),
                    "stress_load": bool(stress_load),
                    "recovery_signal": bool(recovery_signal),
                    "endocrine_balance": bool(endocrine_balance),
                    "stress_lock": bool(stress_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "cortisol_levels": cortisol_levels,
                "stress_loads": stress_loads,
                "recovery_signals": recovery_signals,
                "endocrine_balances": endocrine_balances,
                "stress_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b55_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b56_decision_episodes": explicit_decision_episodes >= 2,
        "cortisol_level_episodes": cortisol_episodes >= 2,
        "stress_load_episodes": stress_load_episodes >= 2,
        "recovery_signal_episodes": recovery_signal_episodes >= 2,
        "endocrine_balance_episodes": endocrine_balance_episodes >= 2,
        "stress_lock_episodes": stress_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b56_aggregate:" + name
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
            "base_b55_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "cortisol_level_episodes": int(cortisol_episodes),
            "stress_load_episodes": int(stress_load_episodes),
            "recovery_signal_episodes": int(recovery_signal_episodes),
            "endocrine_balance_episodes": int(endocrine_balance_episodes),
            "stress_lock_episodes": int(stress_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b57_insular_interoceptive_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b56_hpa_stress_corridor_gate_result(results)
    explicit_decision_set = set(B57_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    awareness_episodes = 0
    salience_episodes = 0
    confidence_episodes = 0
    balance_episodes = 0
    awareness_lock_episodes = 0
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
            str(item.get("b57_decision"))
            for item in trace
            if item.get("b57_decision") is not None
        ]
        awareness_values = [
            float(item.get("b57_interoceptive_awareness", 0.0) or 0.0)
            for item in trace
            if item.get("b57_interoceptive_awareness") is not None
        ]
        salience_values = [
            float(item.get("b57_visceral_salience", 0.0) or 0.0)
            for item in trace
            if item.get("b57_visceral_salience") is not None
        ]
        confidence_values = [
            float(item.get("b57_body_state_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b57_body_state_confidence") is not None
        ]
        balance_values = [
            float(item.get("b57_awareness_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b57_awareness_balance") is not None
        ]
        locks = [
            int(item.get("b57_awareness_lock", 0) or 0)
            for item in trace
            if item.get("b57_awareness_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        awareness = any(abs(value) > 0.0 for value in awareness_values)
        salience = any(abs(value) > 0.0 for value in salience_values)
        confidence = any(abs(value) > 0.0 for value in confidence_values)
        balance = any(abs(value) > 0.0 for value in balance_values)
        awareness_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if awareness:
            awareness_episodes += 1
        if salience:
            salience_episodes += 1
        if confidence:
            confidence_episodes += 1
        if balance:
            balance_episodes += 1
        if awareness_lock:
            awareness_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b57_decision": bool(explicit_decision),
                    "interoceptive_awareness": bool(awareness),
                    "visceral_salience": bool(salience),
                    "body_state_confidence": bool(confidence),
                    "awareness_balance": bool(balance),
                    "awareness_lock": bool(awareness_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "interoceptive_awareness": awareness_values,
                "visceral_salience": salience_values,
                "body_state_confidence": confidence_values,
                "awareness_balances": balance_values,
                "awareness_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b56_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b57_decision_episodes": explicit_decision_episodes >= 2,
        "interoceptive_awareness_episodes": awareness_episodes >= 2,
        "visceral_salience_episodes": salience_episodes >= 2,
        "body_state_confidence_episodes": confidence_episodes >= 2,
        "awareness_balance_episodes": balance_episodes >= 2,
        "awareness_lock_episodes": awareness_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b57_aggregate:" + name
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
            "base_b56_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "interoceptive_awareness_episodes": int(awareness_episodes),
            "visceral_salience_episodes": int(salience_episodes),
            "body_state_confidence_episodes": int(confidence_episodes),
            "awareness_balance_episodes": int(balance_episodes),
            "awareness_lock_episodes": int(awareness_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b58_acc_conflict_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b57_insular_interoceptive_corridor_gate_result(results)
    explicit_decision_set = set(B58_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    conflict_episodes = 0
    error_episodes = 0
    control_episodes = 0
    balance_episodes = 0
    conflict_lock_episodes = 0
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
            str(item.get("b58_decision"))
            for item in trace
            if item.get("b58_decision") is not None
        ]
        conflict_values = [
            float(item.get("b58_conflict_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b58_conflict_signal") is not None
        ]
        error_values = [
            float(item.get("b58_error_likelihood", 0.0) or 0.0)
            for item in trace
            if item.get("b58_error_likelihood") is not None
        ]
        control_values = [
            float(item.get("b58_control_allocation", 0.0) or 0.0)
            for item in trace
            if item.get("b58_control_allocation") is not None
        ]
        balance_values = [
            float(item.get("b58_resolution_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b58_resolution_balance") is not None
        ]
        locks = [
            int(item.get("b58_conflict_lock", 0) or 0)
            for item in trace
            if item.get("b58_conflict_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        conflict = any(abs(value) > 0.0 for value in conflict_values)
        error = any(abs(value) > 0.0 for value in error_values)
        control = any(abs(value) > 0.0 for value in control_values)
        balance = any(abs(value) > 0.0 for value in balance_values)
        conflict_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if conflict:
            conflict_episodes += 1
        if error:
            error_episodes += 1
        if control:
            control_episodes += 1
        if balance:
            balance_episodes += 1
        if conflict_lock:
            conflict_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b58_decision": bool(explicit_decision),
                    "conflict_signal": bool(conflict),
                    "error_likelihood": bool(error),
                    "control_allocation": bool(control),
                    "resolution_balance": bool(balance),
                    "conflict_lock": bool(conflict_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "conflict_signals": conflict_values,
                "error_likelihoods": error_values,
                "control_allocations": control_values,
                "resolution_balances": balance_values,
                "conflict_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b57_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b58_decision_episodes": explicit_decision_episodes >= 2,
        "conflict_signal_episodes": conflict_episodes >= 2,
        "error_likelihood_episodes": error_episodes >= 2,
        "control_allocation_episodes": control_episodes >= 2,
        "resolution_balance_episodes": balance_episodes >= 2,
        "conflict_lock_episodes": conflict_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b58_aggregate:" + name
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
            "base_b57_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "conflict_signal_episodes": int(conflict_episodes),
            "error_likelihood_episodes": int(error_episodes),
            "control_allocation_episodes": int(control_episodes),
            "resolution_balance_episodes": int(balance_episodes),
            "conflict_lock_episodes": int(conflict_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b59_prefrontal_goal_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b58_acc_conflict_corridor_gate_result(results)
    explicit_decision_set = set(B59_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    goal_episodes = 0
    working_set_episodes = 0
    task_confidence_episodes = 0
    balance_episodes = 0
    executive_lock_episodes = 0
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
            str(item.get("b59_decision"))
            for item in trace
            if item.get("b59_decision") is not None
        ]
        goal_values = [
            float(item.get("b59_goal_context", 0.0) or 0.0)
            for item in trace
            if item.get("b59_goal_context") is not None
        ]
        working_values = [
            float(item.get("b59_working_set_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b59_working_set_stability") is not None
        ]
        confidence_values = [
            float(item.get("b59_task_set_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b59_task_set_confidence") is not None
        ]
        balance_values = [
            float(item.get("b59_executive_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b59_executive_balance") is not None
        ]
        locks = [
            int(item.get("b59_executive_lock", 0) or 0)
            for item in trace
            if item.get("b59_executive_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        goal = any(abs(value) > 0.0 for value in goal_values)
        working_set = any(abs(value) > 0.0 for value in working_values)
        task_confidence = any(abs(value) > 0.0 for value in confidence_values)
        balance = any(abs(value) > 0.0 for value in balance_values)
        executive_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if goal:
            goal_episodes += 1
        if working_set:
            working_set_episodes += 1
        if task_confidence:
            task_confidence_episodes += 1
        if balance:
            balance_episodes += 1
        if executive_lock:
            executive_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b59_decision": bool(explicit_decision),
                    "goal_context": bool(goal),
                    "working_set_stability": bool(working_set),
                    "task_set_confidence": bool(task_confidence),
                    "executive_balance": bool(balance),
                    "executive_lock": bool(executive_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "goal_contexts": goal_values,
                "working_set_stability": working_values,
                "task_set_confidences": confidence_values,
                "executive_balances": balance_values,
                "executive_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b58_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b59_decision_episodes": explicit_decision_episodes >= 2,
        "goal_context_episodes": goal_episodes >= 2,
        "working_set_stability_episodes": working_set_episodes >= 2,
        "task_set_confidence_episodes": task_confidence_episodes >= 2,
        "executive_balance_episodes": balance_episodes >= 2,
        "executive_lock_episodes": executive_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b59_aggregate:" + name
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
            "base_b58_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "goal_context_episodes": int(goal_episodes),
            "working_set_stability_episodes": int(working_set_episodes),
            "task_set_confidence_episodes": int(task_confidence_episodes),
            "executive_balance_episodes": int(balance_episodes),
            "executive_lock_episodes": int(executive_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b60_orbitofrontal_value_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b59_prefrontal_goal_corridor_gate_result(results)
    explicit_decision_set = set(B60_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    outcome_value_episodes = 0
    reversal_signal_episodes = 0
    confidence_episodes = 0
    balance_episodes = 0
    value_lock_episodes = 0
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
            str(item.get("b60_decision"))
            for item in trace
            if item.get("b60_decision") is not None
        ]
        outcome_values = [
            float(item.get("b60_outcome_value", 0.0) or 0.0)
            for item in trace
            if item.get("b60_outcome_value") is not None
        ]
        reversal_values = [
            float(item.get("b60_reversal_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b60_reversal_signal") is not None
        ]
        confidence_values = [
            float(item.get("b60_goal_value_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b60_goal_value_confidence") is not None
        ]
        balance_values = [
            float(item.get("b60_value_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b60_value_balance") is not None
        ]
        locks = [
            int(item.get("b60_value_lock", 0) or 0)
            for item in trace
            if item.get("b60_value_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        outcome_value = any(abs(value) > 0.0 for value in outcome_values)
        reversal_signal = bool(reversal_values)
        confidence = any(abs(value) > 0.0 for value in confidence_values)
        balance = any(abs(value) > 0.0 for value in balance_values)
        value_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if outcome_value:
            outcome_value_episodes += 1
        if reversal_signal:
            reversal_signal_episodes += 1
        if confidence:
            confidence_episodes += 1
        if balance:
            balance_episodes += 1
        if value_lock:
            value_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b60_decision": bool(explicit_decision),
                    "outcome_value": bool(outcome_value),
                    "reversal_signal": bool(reversal_signal),
                    "goal_value_confidence": bool(confidence),
                    "value_balance": bool(balance),
                    "value_lock": bool(value_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "outcome_values": outcome_values,
                "reversal_signals": reversal_values,
                "goal_value_confidences": confidence_values,
                "value_balances": balance_values,
                "value_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b59_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b60_decision_episodes": explicit_decision_episodes >= 2,
        "outcome_value_episodes": outcome_value_episodes >= 2,
        "reversal_signal_episodes": reversal_signal_episodes >= 2,
        "goal_value_confidence_episodes": confidence_episodes >= 2,
        "value_balance_episodes": balance_episodes >= 2,
        "value_lock_episodes": value_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b60_aggregate:" + name
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
            "base_b59_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "outcome_value_episodes": int(outcome_value_episodes),
            "reversal_signal_episodes": int(reversal_signal_episodes),
            "goal_value_confidence_episodes": int(confidence_episodes),
            "value_balance_episodes": int(balance_episodes),
            "value_lock_episodes": int(value_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
