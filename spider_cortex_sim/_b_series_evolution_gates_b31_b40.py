from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_gates_b1_b6 import (
    trace_uses_only_primitive_actions,
)

from ._b_series_evolution_gates_b20_b30 import (
    b30_basal_ganglia_gate_corridor_gate_result,
)

def b31_dopamine_prediction_error_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b30_basal_ganglia_gate_corridor_gate_result(results)
    explicit_decision_set = set(B31_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    dopamine_state_episodes = 0
    dopamine_lock_episodes = 0
    dopamine_signal_episodes = 0
    gate_bias_episodes = 0
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
            str(item.get("b31_decision"))
            for item in trace
            if item.get("b31_decision") is not None
        ]
        states = [
            str(item.get("b31_dopamine_state"))
            for item in trace
            if item.get("b31_dopamine_state") is not None
        ]
        locks = [
            int(item.get("b31_dopamine_lock", 0) or 0)
            for item in trace
            if item.get("b31_dopamine_lock") is not None
        ]
        biases = [
            float(item.get("b31_gate_bias", 0.0) or 0.0)
            for item in trace
            if item.get("b31_gate_bias") is not None
        ]
        signals = [
            abs(float(item.get("b31_reward_prediction_error", 0.0) or 0.0))
            + float(item.get("b31_tonic_dopamine", 0.0) or 0.0)
            + float(item.get("b31_phasic_dopamine", 0.0) or 0.0)
            for item in trace
            if item.get("b31_reward_prediction_error") is not None
            or item.get("b31_tonic_dopamine") is not None
            or item.get("b31_phasic_dopamine") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        dopamine_state = any(state != "non_corridor" for state in states)
        dopamine_lock = explicit_decision and any(lock > 0 for lock in locks)
        dopamine_signal = any(signal > 0.0 for signal in signals)
        gate_bias = any(abs(bias) > 0.0 for bias in biases)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if dopamine_state:
            dopamine_state_episodes += 1
        if dopamine_lock:
            dopamine_lock_episodes += 1
        if dopamine_signal:
            dopamine_signal_episodes += 1
        if gate_bias:
            gate_bias_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b31_decision": bool(explicit_decision),
                    "dopamine_state": bool(dopamine_state),
                    "dopamine_lock": bool(dopamine_lock),
                    "dopamine_signal": bool(dopamine_signal),
                    "gate_bias": bool(gate_bias),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "dopamine_states": states,
                "dopamine_locks": locks,
                "gate_biases": biases,
                "dopamine_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b31_decision_episodes": explicit_decision_episodes >= 2,
        "dopamine_state_episodes": dopamine_state_episodes >= 2,
        "dopamine_lock_episodes": dopamine_lock_episodes >= 2,
        "dopamine_signal_episodes": dopamine_signal_episodes >= 2,
        "gate_bias_episodes": gate_bias_episodes >= 2,
    }
    failures.extend(
        "corridor_b31_aggregate:" + name
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
            "base_b30_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "dopamine_state_episodes": int(dopamine_state_episodes),
            "dopamine_lock_episodes": int(dopamine_lock_episodes),
            "dopamine_signal_episodes": int(dopamine_signal_episodes),
            "gate_bias_episodes": int(gate_bias_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b32_actor_critic_value_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b31_dopamine_prediction_error_corridor_gate_result(results)
    explicit_decision_set = set(B32_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    critic_value_episodes = 0
    actor_advantage_episodes = 0
    value_lock_episodes = 0
    policy_bias_episodes = 0
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
            str(item.get("b32_decision"))
            for item in trace
            if item.get("b32_decision") is not None
        ]
        critic_values = [
            float(item.get("b32_critic_value", 0.0) or 0.0)
            for item in trace
            if item.get("b32_critic_value") is not None
        ]
        advantages = [
            float(item.get("b32_actor_advantage", 0.0) or 0.0)
            for item in trace
            if item.get("b32_actor_advantage") is not None
        ]
        locks = [
            int(item.get("b32_value_lock", 0) or 0)
            for item in trace
            if item.get("b32_value_lock") is not None
        ]
        biases = [
            float(item.get("b32_policy_bias", 0.0) or 0.0)
            for item in trace
            if item.get("b32_policy_bias") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        critic_value = any(abs(value) > 0.0 for value in critic_values)
        actor_advantage = any(abs(value) > 0.0 for value in advantages)
        value_lock = explicit_decision and any(lock > 0 for lock in locks)
        policy_bias = any(abs(value) > 0.0 for value in biases)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if critic_value:
            critic_value_episodes += 1
        if actor_advantage:
            actor_advantage_episodes += 1
        if value_lock:
            value_lock_episodes += 1
        if policy_bias:
            policy_bias_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b32_decision": bool(explicit_decision),
                    "critic_value": bool(critic_value),
                    "actor_advantage": bool(actor_advantage),
                    "value_lock": bool(value_lock),
                    "policy_bias": bool(policy_bias),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "critic_values": critic_values,
                "actor_advantages": advantages,
                "value_locks": locks,
                "policy_biases": biases,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b32_decision_episodes": explicit_decision_episodes >= 2,
        "critic_value_episodes": critic_value_episodes >= 2,
        "actor_advantage_episodes": actor_advantage_episodes >= 2,
        "value_lock_episodes": value_lock_episodes >= 2,
        "policy_bias_episodes": policy_bias_episodes >= 2,
    }
    failures.extend(
        "corridor_b32_aggregate:" + name
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
            "base_b31_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "critic_value_episodes": int(critic_value_episodes),
            "actor_advantage_episodes": int(actor_advantage_episodes),
            "value_lock_episodes": int(value_lock_episodes),
            "policy_bias_episodes": int(policy_bias_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b33_td_error_decomposition_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b32_actor_critic_value_corridor_gate_result(results)
    explicit_decision_set = set(B33_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    td_error_episodes = 0
    bootstrap_value_episodes = 0
    reward_trace_episodes = 0
    td_lock_episodes = 0
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
            str(item.get("b33_decision"))
            for item in trace
            if item.get("b33_decision") is not None
        ]
        td_errors = [
            float(item.get("b33_td_error", 0.0) or 0.0)
            for item in trace
            if item.get("b33_td_error") is not None
        ]
        bootstrap_values = [
            float(item.get("b33_bootstrap_value", 0.0) or 0.0)
            for item in trace
            if item.get("b33_bootstrap_value") is not None
        ]
        reward_traces = [
            float(item.get("b33_reward_trace", 0.0) or 0.0)
            for item in trace
            if item.get("b33_reward_trace") is not None
        ]
        locks = [
            int(item.get("b33_td_lock", 0) or 0)
            for item in trace
            if item.get("b33_td_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        td_error = any(abs(value) > 0.0 for value in td_errors)
        bootstrap_value = any(abs(value) > 0.0 for value in bootstrap_values)
        reward_trace = any(abs(value) > 0.0 for value in reward_traces)
        td_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if td_error:
            td_error_episodes += 1
        if bootstrap_value:
            bootstrap_value_episodes += 1
        if reward_trace:
            reward_trace_episodes += 1
        if td_lock:
            td_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b33_decision": bool(explicit_decision),
                    "td_error": bool(td_error),
                    "bootstrap_value": bool(bootstrap_value),
                    "reward_trace": bool(reward_trace),
                    "td_lock": bool(td_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "td_errors": td_errors,
                "bootstrap_values": bootstrap_values,
                "reward_traces": reward_traces,
                "td_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b33_decision_episodes": explicit_decision_episodes >= 2,
        "td_error_episodes": td_error_episodes >= 2,
        "bootstrap_value_episodes": bootstrap_value_episodes >= 2,
        "reward_trace_episodes": reward_trace_episodes >= 2,
        "td_lock_episodes": td_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b33_aggregate:" + name
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
            "base_b32_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "td_error_episodes": int(td_error_episodes),
            "bootstrap_value_episodes": int(bootstrap_value_episodes),
            "reward_trace_episodes": int(reward_trace_episodes),
            "td_lock_episodes": int(td_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b34_eligibility_credit_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b33_td_error_decomposition_corridor_gate_result(results)
    explicit_decision_set = set(B34_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    eligibility_episodes = 0
    credit_assignment_episodes = 0
    synaptic_tag_episodes = 0
    credit_lock_episodes = 0
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
            str(item.get("b34_decision"))
            for item in trace
            if item.get("b34_decision") is not None
        ]
        eligibility_traces = [
            float(item.get("b34_eligibility_trace", 0.0) or 0.0)
            for item in trace
            if item.get("b34_eligibility_trace") is not None
        ]
        credit_assignments = [
            float(item.get("b34_credit_assignment", 0.0) or 0.0)
            for item in trace
            if item.get("b34_credit_assignment") is not None
        ]
        synaptic_tags = [
            float(item.get("b34_synaptic_tag", 0.0) or 0.0)
            for item in trace
            if item.get("b34_synaptic_tag") is not None
        ]
        locks = [
            int(item.get("b34_credit_lock", 0) or 0)
            for item in trace
            if item.get("b34_credit_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        eligibility_trace = any(abs(value) > 0.0 for value in eligibility_traces)
        credit_assignment = any(abs(value) > 0.0 for value in credit_assignments)
        synaptic_tag = any(abs(value) > 0.0 for value in synaptic_tags)
        credit_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if eligibility_trace:
            eligibility_episodes += 1
        if credit_assignment:
            credit_assignment_episodes += 1
        if synaptic_tag:
            synaptic_tag_episodes += 1
        if credit_lock:
            credit_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b34_decision": bool(explicit_decision),
                    "eligibility_trace": bool(eligibility_trace),
                    "credit_assignment": bool(credit_assignment),
                    "synaptic_tag": bool(synaptic_tag),
                    "credit_lock": bool(credit_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "eligibility_traces": eligibility_traces,
                "credit_assignments": credit_assignments,
                "synaptic_tags": synaptic_tags,
                "credit_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b34_decision_episodes": explicit_decision_episodes >= 2,
        "eligibility_trace_episodes": eligibility_episodes >= 2,
        "credit_assignment_episodes": credit_assignment_episodes >= 2,
        "synaptic_tag_episodes": synaptic_tag_episodes >= 2,
        "credit_lock_episodes": credit_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b34_aggregate:" + name
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
            "base_b33_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "eligibility_trace_episodes": int(eligibility_episodes),
            "credit_assignment_episodes": int(credit_assignment_episodes),
            "synaptic_tag_episodes": int(synaptic_tag_episodes),
            "credit_lock_episodes": int(credit_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b35_forward_model_value_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b34_eligibility_credit_corridor_gate_result(results)
    explicit_decision_set = set(B35_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    forward_value_episodes = 0
    transition_error_episodes = 0
    confidence_episodes = 0
    prediction_memory_episodes = 0
    model_lock_episodes = 0
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
            str(item.get("b35_decision"))
            for item in trace
            if item.get("b35_decision") is not None
        ]
        forward_values = [
            float(item.get("b35_forward_value", 0.0) or 0.0)
            for item in trace
            if item.get("b35_forward_value") is not None
        ]
        transition_errors = [
            float(item.get("b35_transition_error", 0.0) or 0.0)
            for item in trace
            if item.get("b35_transition_error") is not None
        ]
        confidences = [
            float(item.get("b35_model_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b35_model_confidence") is not None
        ]
        prediction_memories = [
            float(item.get("b35_prediction_memory", 0.0) or 0.0)
            for item in trace
            if item.get("b35_prediction_memory") is not None
        ]
        locks = [
            int(item.get("b35_model_lock", 0) or 0)
            for item in trace
            if item.get("b35_model_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        forward_value = any(abs(value) > 0.0 for value in forward_values)
        transition_error = any(abs(value) > 0.0 for value in transition_errors)
        confidence = any(abs(value) > 0.0 for value in confidences)
        prediction_memory = any(abs(value) > 0.0 for value in prediction_memories)
        model_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if forward_value:
            forward_value_episodes += 1
        if transition_error:
            transition_error_episodes += 1
        if confidence:
            confidence_episodes += 1
        if prediction_memory:
            prediction_memory_episodes += 1
        if model_lock:
            model_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b35_decision": bool(explicit_decision),
                    "forward_value": bool(forward_value),
                    "transition_error": bool(transition_error),
                    "model_confidence": bool(confidence),
                    "prediction_memory": bool(prediction_memory),
                    "model_lock": bool(model_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "forward_values": forward_values,
                "transition_errors": transition_errors,
                "model_confidences": confidences,
                "prediction_memories": prediction_memories,
                "model_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b35_decision_episodes": explicit_decision_episodes >= 2,
        "forward_value_episodes": forward_value_episodes >= 2,
        "transition_error_episodes": transition_error_episodes >= 2,
        "model_confidence_episodes": confidence_episodes >= 2,
        "prediction_memory_episodes": prediction_memory_episodes >= 2,
        "model_lock_episodes": model_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b35_aggregate:" + name
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
            "base_b34_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "forward_value_episodes": int(forward_value_episodes),
            "transition_error_episodes": int(transition_error_episodes),
            "model_confidence_episodes": int(confidence_episodes),
            "prediction_memory_episodes": int(prediction_memory_episodes),
            "model_lock_episodes": int(model_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b36_latent_belief_state_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b35_forward_model_value_corridor_gate_result(results)
    explicit_decision_set = set(B36_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    latent_state_episodes = 0
    belief_error_episodes = 0
    confidence_episodes = 0
    context_memory_episodes = 0
    belief_lock_episodes = 0
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
            str(item.get("b36_decision"))
            for item in trace
            if item.get("b36_decision") is not None
        ]
        latent_states = [
            float(item.get("b36_latent_state", 0.0) or 0.0)
            for item in trace
            if item.get("b36_latent_state") is not None
        ]
        belief_errors = [
            float(item.get("b36_belief_error", 0.0) or 0.0)
            for item in trace
            if item.get("b36_belief_error") is not None
        ]
        confidences = [
            float(item.get("b36_state_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b36_state_confidence") is not None
        ]
        context_memories = [
            float(item.get("b36_context_memory", 0.0) or 0.0)
            for item in trace
            if item.get("b36_context_memory") is not None
        ]
        locks = [
            int(item.get("b36_belief_lock", 0) or 0)
            for item in trace
            if item.get("b36_belief_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        latent_state = any(abs(value) > 0.0 for value in latent_states)
        belief_error = any(abs(value) > 0.0 for value in belief_errors)
        confidence = any(abs(value) > 0.0 for value in confidences)
        context_memory = any(abs(value) > 0.0 for value in context_memories)
        belief_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if latent_state:
            latent_state_episodes += 1
        if belief_error:
            belief_error_episodes += 1
        if confidence:
            confidence_episodes += 1
        if context_memory:
            context_memory_episodes += 1
        if belief_lock:
            belief_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b36_decision": bool(explicit_decision),
                    "latent_state": bool(latent_state),
                    "belief_error": bool(belief_error),
                    "state_confidence": bool(confidence),
                    "context_memory": bool(context_memory),
                    "belief_lock": bool(belief_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "latent_states": latent_states,
                "belief_errors": belief_errors,
                "state_confidences": confidences,
                "context_memories": context_memories,
                "belief_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b36_decision_episodes": explicit_decision_episodes >= 2,
        "latent_state_episodes": latent_state_episodes >= 2,
        "belief_error_episodes": belief_error_episodes >= 2,
        "state_confidence_episodes": confidence_episodes >= 2,
        "context_memory_episodes": context_memory_episodes >= 2,
        "belief_lock_episodes": belief_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b36_aggregate:" + name
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
            "base_b35_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "latent_state_episodes": int(latent_state_episodes),
            "belief_error_episodes": int(belief_error_episodes),
            "state_confidence_episodes": int(confidence_episodes),
            "context_memory_episodes": int(context_memory_episodes),
            "belief_lock_episodes": int(belief_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b37_state_factor_gate_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b36_latent_belief_state_corridor_gate_result(results)
    explicit_decision_set = set(B37_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    external_factor_episodes = 0
    internal_factor_episodes = 0
    factor_alignment_episodes = 0
    factor_confidence_episodes = 0
    factor_lock_episodes = 0
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
            str(item.get("b37_decision"))
            for item in trace
            if item.get("b37_decision") is not None
        ]
        external_factors = [
            float(item.get("b37_external_state_factor", 0.0) or 0.0)
            for item in trace
            if item.get("b37_external_state_factor") is not None
        ]
        internal_factors = [
            float(item.get("b37_internal_state_factor", 0.0) or 0.0)
            for item in trace
            if item.get("b37_internal_state_factor") is not None
        ]
        alignments = [
            float(item.get("b37_factor_alignment", 0.0) or 0.0)
            for item in trace
            if item.get("b37_factor_alignment") is not None
        ]
        confidences = [
            float(item.get("b37_factor_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b37_factor_confidence") is not None
        ]
        locks = [
            int(item.get("b37_factor_lock", 0) or 0)
            for item in trace
            if item.get("b37_factor_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        external_factor = any(abs(value) > 0.0 for value in external_factors)
        internal_factor = any(abs(value) > 0.0 for value in internal_factors)
        factor_alignment = any(abs(value) > 0.0 for value in alignments)
        factor_confidence = any(abs(value) > 0.0 for value in confidences)
        factor_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if external_factor:
            external_factor_episodes += 1
        if internal_factor:
            internal_factor_episodes += 1
        if factor_alignment:
            factor_alignment_episodes += 1
        if factor_confidence:
            factor_confidence_episodes += 1
        if factor_lock:
            factor_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b37_decision": bool(explicit_decision),
                    "external_state_factor": bool(external_factor),
                    "internal_state_factor": bool(internal_factor),
                    "factor_alignment": bool(factor_alignment),
                    "factor_confidence": bool(factor_confidence),
                    "factor_lock": bool(factor_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "external_state_factors": external_factors,
                "internal_state_factors": internal_factors,
                "factor_alignments": alignments,
                "factor_confidences": confidences,
                "factor_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b37_decision_episodes": explicit_decision_episodes >= 2,
        "external_factor_episodes": external_factor_episodes >= 2,
        "internal_factor_episodes": internal_factor_episodes >= 2,
        "factor_alignment_episodes": factor_alignment_episodes >= 2,
        "factor_confidence_episodes": factor_confidence_episodes >= 2,
        "factor_lock_episodes": factor_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b37_aggregate:" + name
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
            "base_b36_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "external_factor_episodes": int(external_factor_episodes),
            "internal_factor_episodes": int(internal_factor_episodes),
            "factor_alignment_episodes": int(factor_alignment_episodes),
            "factor_confidence_episodes": int(factor_confidence_episodes),
            "factor_lock_episodes": int(factor_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b38_factor_attention_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b37_state_factor_gate_corridor_gate_result(results)
    explicit_decision_set = set(B38_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    external_attention_episodes = 0
    internal_attention_episodes = 0
    attention_balance_episodes = 0
    attention_gain_episodes = 0
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
            str(item.get("b38_decision"))
            for item in trace
            if item.get("b38_decision") is not None
        ]
        external_attentions = [
            float(item.get("b38_external_attention", 0.0) or 0.0)
            for item in trace
            if item.get("b38_external_attention") is not None
        ]
        internal_attentions = [
            float(item.get("b38_internal_attention", 0.0) or 0.0)
            for item in trace
            if item.get("b38_internal_attention") is not None
        ]
        balances = [
            float(item.get("b38_attention_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b38_attention_balance") is not None
        ]
        gains = [
            float(item.get("b38_attention_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b38_attention_gain") is not None
        ]
        locks = [
            int(item.get("b38_attention_lock", 0) or 0)
            for item in trace
            if item.get("b38_attention_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        external_attention = any(abs(value) > 0.0 for value in external_attentions)
        internal_attention = any(abs(value) > 0.0 for value in internal_attentions)
        attention_balance = any(abs(value) > 0.0 for value in balances)
        attention_gain = any(abs(value) > 0.0 for value in gains)
        attention_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if external_attention:
            external_attention_episodes += 1
        if internal_attention:
            internal_attention_episodes += 1
        if attention_balance:
            attention_balance_episodes += 1
        if attention_gain:
            attention_gain_episodes += 1
        if attention_lock:
            attention_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b38_decision": bool(explicit_decision),
                    "external_attention": bool(external_attention),
                    "internal_attention": bool(internal_attention),
                    "attention_balance": bool(attention_balance),
                    "attention_gain": bool(attention_gain),
                    "attention_lock": bool(attention_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "external_attentions": external_attentions,
                "internal_attentions": internal_attentions,
                "attention_balances": balances,
                "attention_gains": gains,
                "attention_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b38_decision_episodes": explicit_decision_episodes >= 2,
        "external_attention_episodes": external_attention_episodes >= 2,
        "internal_attention_episodes": internal_attention_episodes >= 2,
        "attention_balance_episodes": attention_balance_episodes >= 2,
        "attention_gain_episodes": attention_gain_episodes >= 2,
        "attention_lock_episodes": attention_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b38_aggregate:" + name
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
            "base_b37_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "external_attention_episodes": int(external_attention_episodes),
            "internal_attention_episodes": int(internal_attention_episodes),
            "attention_balance_episodes": int(attention_balance_episodes),
            "attention_gain_episodes": int(attention_gain_episodes),
            "attention_lock_episodes": int(attention_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b39_attention_binding_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b38_factor_attention_corridor_gate_result(results)
    explicit_decision_set = set(B39_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    binding_strength_episodes = 0
    coherence_episodes = 0
    bound_context_episodes = 0
    binding_gain_episodes = 0
    binding_lock_episodes = 0
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
            str(item.get("b39_decision"))
            for item in trace
            if item.get("b39_decision") is not None
        ]
        strengths = [
            float(item.get("b39_binding_strength", 0.0) or 0.0)
            for item in trace
            if item.get("b39_binding_strength") is not None
        ]
        coherences = [
            float(item.get("b39_cross_factor_coherence", 0.0) or 0.0)
            for item in trace
            if item.get("b39_cross_factor_coherence") is not None
        ]
        contexts = [
            float(item.get("b39_bound_context", 0.0) or 0.0)
            for item in trace
            if item.get("b39_bound_context") is not None
        ]
        gains = [
            float(item.get("b39_binding_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b39_binding_gain") is not None
        ]
        locks = [
            int(item.get("b39_binding_lock", 0) or 0)
            for item in trace
            if item.get("b39_binding_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        binding_strength = any(abs(value) > 0.0 for value in strengths)
        coherence = any(abs(value) > 0.0 for value in coherences)
        bound_context = any(abs(value) > 0.0 for value in contexts)
        binding_gain = any(abs(value) > 0.0 for value in gains)
        binding_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if binding_strength:
            binding_strength_episodes += 1
        if coherence:
            coherence_episodes += 1
        if bound_context:
            bound_context_episodes += 1
        if binding_gain:
            binding_gain_episodes += 1
        if binding_lock:
            binding_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b39_decision": bool(explicit_decision),
                    "binding_strength": bool(binding_strength),
                    "cross_factor_coherence": bool(coherence),
                    "bound_context": bool(bound_context),
                    "binding_gain": bool(binding_gain),
                    "binding_lock": bool(binding_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "binding_strengths": strengths,
                "cross_factor_coherences": coherences,
                "bound_contexts": contexts,
                "binding_gains": gains,
                "binding_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b39_decision_episodes": explicit_decision_episodes >= 2,
        "binding_strength_episodes": binding_strength_episodes >= 2,
        "cross_factor_coherence_episodes": coherence_episodes >= 2,
        "bound_context_episodes": bound_context_episodes >= 2,
        "binding_gain_episodes": binding_gain_episodes >= 2,
        "binding_lock_episodes": binding_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b39_aggregate:" + name
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
            "base_b38_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "binding_strength_episodes": int(binding_strength_episodes),
            "cross_factor_coherence_episodes": int(coherence_episodes),
            "bound_context_episodes": int(bound_context_episodes),
            "binding_gain_episodes": int(binding_gain_episodes),
            "binding_lock_episodes": int(binding_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b40_global_workspace_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b39_attention_binding_corridor_gate_result(results)
    explicit_decision_set = set(B40_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    activation_episodes = 0
    broadcast_episodes = 0
    context_episodes = 0
    stability_episodes = 0
    lock_episodes = 0
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
            str(item.get("b40_decision"))
            for item in trace
            if item.get("b40_decision") is not None
        ]
        activations = [
            float(item.get("b40_workspace_activation", 0.0) or 0.0)
            for item in trace
            if item.get("b40_workspace_activation") is not None
        ]
        broadcasts = [
            float(item.get("b40_broadcast_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b40_broadcast_gain") is not None
        ]
        contexts = [
            float(item.get("b40_context_availability", 0.0) or 0.0)
            for item in trace
            if item.get("b40_context_availability") is not None
        ]
        stabilities = [
            float(item.get("b40_workspace_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b40_workspace_stability") is not None
        ]
        locks = [
            int(item.get("b40_workspace_lock", 0) or 0)
            for item in trace
            if item.get("b40_workspace_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        activation = any(abs(value) > 0.0 for value in activations)
        broadcast = any(abs(value) > 0.0 for value in broadcasts)
        context = any(abs(value) > 0.0 for value in contexts)
        stability = any(abs(value) > 0.0 for value in stabilities)
        workspace_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if activation:
            activation_episodes += 1
        if broadcast:
            broadcast_episodes += 1
        if context:
            context_episodes += 1
        if stability:
            stability_episodes += 1
        if workspace_lock:
            lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b40_decision": bool(explicit_decision),
                    "workspace_activation": bool(activation),
                    "broadcast_gain": bool(broadcast),
                    "context_availability": bool(context),
                    "workspace_stability": bool(stability),
                    "workspace_lock": bool(workspace_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "workspace_activations": activations,
                "broadcast_gains": broadcasts,
                "context_availabilities": contexts,
                "workspace_stabilities": stabilities,
                "workspace_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b40_decision_episodes": explicit_decision_episodes >= 2,
        "workspace_activation_episodes": activation_episodes >= 2,
        "broadcast_gain_episodes": broadcast_episodes >= 2,
        "context_availability_episodes": context_episodes >= 2,
        "workspace_stability_episodes": stability_episodes >= 2,
        "workspace_lock_episodes": lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b40_aggregate:" + name
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
            "base_b39_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "workspace_activation_episodes": int(activation_episodes),
            "broadcast_gain_episodes": int(broadcast_episodes),
            "context_availability_episodes": int(context_episodes),
            "workspace_stability_episodes": int(stability_episodes),
            "workspace_lock_episodes": int(lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
