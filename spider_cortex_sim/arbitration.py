from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Sequence

import numpy as np

from .ablations import BrainAblationConfig
from .interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACE_BY_NAME,
)
from .modules import ModuleResult
from .nn import ArbitrationNetwork
from .nn_utils import softmax
from .operational_profiles import OperationalProfile


ACTIONS = tuple(LOCOMOTION_ACTIONS)
MONOLITHIC_POLICY_NAME = "monolithic_policy"
ARBITRATION_NETWORK_NAME = "arbitration_network"
VALENCE_ORDER = ("threat", "hunger", "sleep", "exploration")
PRIORITY_GATING_WEIGHTS = MappingProxyType(
    {
        "threat": MappingProxyType(
            {
                "alert_center": 1.0,
                "hunger_center": 0.18,
                "sleep_center": 0.14,
                "visual_cortex": 0.58,
                "sensory_cortex": 0.68,
                MONOLITHIC_POLICY_NAME: 1.0,
            }
        ),
        "hunger": MappingProxyType(
            {
                "alert_center": 0.32,
                "hunger_center": 1.0,
                "sleep_center": 0.34,
                "visual_cortex": 0.74,
                "sensory_cortex": 0.7,
                MONOLITHIC_POLICY_NAME: 1.0,
            }
        ),
        "sleep": MappingProxyType(
            {
                "alert_center": 0.3,
                "hunger_center": 0.24,
                "sleep_center": 1.0,
                "visual_cortex": 0.48,
                "sensory_cortex": 0.56,
                MONOLITHIC_POLICY_NAME: 1.0,
            }
        ),
        "exploration": MappingProxyType(
            {
                "alert_center": 0.42,
                "hunger_center": 0.55,
                "sleep_center": 0.55,
                "visual_cortex": 0.96,
                "sensory_cortex": 0.92,
                MONOLITHIC_POLICY_NAME: 1.0,
            }
        ),
    }
)
ARBITRATION_EVIDENCE_FIELDS = MappingProxyType(
    {
        "threat": (
            "predator_visible",
            "predator_certainty",
            "predator_motion_salience",
            "recent_contact",
            "recent_pain",
            "predator_smell_strength",
        ),
        "hunger": (
            "hunger",
            "on_food",
            "food_visible",
            "food_certainty",
            "food_smell_strength",
            "food_memory_freshness",
        ),
        "sleep": (
            "fatigue",
            "sleep_debt",
            "night",
            "on_shelter",
            "shelter_role_level",
            "shelter_path_confidence",
        ),
        "exploration": (
            "safety_margin",
            "residual_drive",
            "day",
            "off_shelter",
            "visual_openness",
            "food_smell_directionality",
        ),
    }
)
ARBITRATION_GATE_MODULE_ORDER = (
    "alert_center",
    "hunger_center",
    "sleep_center",
    "visual_cortex",
    "sensory_cortex",
    MONOLITHIC_POLICY_NAME,
)
VALENCE_EVIDENCE_WEIGHTS: MappingProxyType = MappingProxyType(
    {
        "threat": MappingProxyType(
            {
                "predator_visible": 0.3,
                "predator_certainty": 0.22,
                "predator_motion_salience": 0.12,
                "recent_contact": 0.14,
                "recent_pain": 0.1,
                "predator_smell_strength": 0.12,
            }
        ),
        "hunger": MappingProxyType(
            {
                "hunger": 0.38,
                "on_food": 0.14,
                "food_visible": 0.16,
                "food_certainty": 0.1,
                "food_smell_strength": 0.12,
                "food_memory_freshness": 0.1,
            }
        ),
        "sleep": MappingProxyType(
            {
                "fatigue": 0.26,
                "sleep_debt": 0.24,
                "night": 0.14,
                "on_shelter": 0.12,
                "shelter_role_level": 0.12,
                "shelter_path_confidence": 0.12,
            }
        ),
        "exploration": MappingProxyType(
            {
                "residual_drive": 0.46,
                "safety_margin": 0.18,
                "day": 0.14,
                "off_shelter": 0.1,
                "visual_openness": 0.06,
                "food_smell_directionality": 0.06,
            }
        ),
    }
)


@dataclass(frozen=True)
class ValenceScore:
    name: str
    score: float
    evidence: Dict[str, float]

    def to_payload(self) -> Dict[str, object]:
        """
        Serialize the valence score into a JSON-serializable payload with rounded numeric values.
        """
        return {
            "name": self.name,
            "score": round(float(self.score), 6),
            "evidence": {
                key: round(float(value), 6)
                for key, value in sorted(self.evidence.items())
            },
        }


@dataclass(frozen=True)
class ArbitrationDecision:
    strategy: str
    winning_valence: str
    valence_scores: Dict[str, float]
    module_gates: Dict[str, float]
    suppressed_modules: list[str]
    evidence: Dict[str, Dict[str, float]]
    intent_before_gating_idx: int
    intent_after_gating_idx: int
    valence_logits: Dict[str, float] = field(default_factory=dict)
    base_gates: Dict[str, float] = field(default_factory=dict)
    gate_adjustments: Dict[str, float] = field(default_factory=dict)
    arbitration_value: float = 0.0
    learned_adjustment: bool = False
    guards_applied: bool = False
    food_bias_applied: bool = False
    food_bias_action: str | None = None
    module_contribution_share: Dict[str, float] = field(default_factory=dict)
    dominant_module: str = ""
    dominant_module_share: float = 0.0
    effective_module_count: float = 0.0
    module_agreement_rate: float = 0.0
    module_disagreement_rate: float = 0.0

    def to_payload(self) -> Dict[str, object]:
        """
        Serialize the arbitration decision to a JSON-safe dictionary for telemetry or persistence.
        """
        evidence_payload = {
            key: {
                inner_key: round(float(inner_value), 6)
                for inner_key, inner_value in sorted(values.items())
            }
            for key, values in sorted(self.evidence.items())
        }
        threat_payload = evidence_payload.get("threat")
        if threat_payload is not None and "predator_proximity" not in threat_payload:
            # Legacy conflict scorers key on predator_proximity. The learned
            # arbitration evidence no longer consumes distance, so traces export
            # a conservative compatibility value and let certainty carry danger.
            threat_payload["predator_proximity"] = 0.0
            evidence_payload["threat"] = {
                key: threat_payload[key]
                for key in sorted(threat_payload)
            }
        return {
            "strategy": self.strategy,
            "winning_valence": self.winning_valence,
            "valence_scores": {
                key: round(float(value), 6)
                for key, value in sorted(self.valence_scores.items())
            },
            "valence_logits": {
                key: round(float(value), 6)
                for key, value in sorted(self.valence_logits.items())
            },
            "base_gates": {
                key: round(float(value), 6)
                for key, value in sorted(self.base_gates.items())
            },
            "gate_adjustments": {
                key: round(float(value), 6)
                for key, value in sorted(self.gate_adjustments.items())
            },
            "module_gates": {
                key: round(float(value), 6)
                for key, value in sorted(self.module_gates.items())
            },
            "arbitration_value": round(float(self.arbitration_value), 6),
            "learned_adjustment": bool(self.learned_adjustment),
            "guards_applied": bool(self.guards_applied),
            "food_bias_applied": bool(self.food_bias_applied),
            "food_bias_action": self.food_bias_action,
            "module_contribution_share": {
                key: round(float(value), 6)
                for key, value in sorted(self.module_contribution_share.items())
            },
            "dominant_module": self.dominant_module,
            "dominant_module_share": round(float(self.dominant_module_share), 6),
            "effective_module_count": round(float(self.effective_module_count), 6),
            "module_agreement_rate": round(float(self.module_agreement_rate), 6),
            "module_disagreement_rate": round(float(self.module_disagreement_rate), 6),
            "suppressed_modules": list(self.suppressed_modules),
            "evidence": evidence_payload,
            "intent_before_gating": ACTIONS[self.intent_before_gating_idx],
            "intent_after_gating": ACTIONS[self.intent_after_gating_idx],
        }


def clamp_unit(value: float) -> float:
    """
    Clamp a numeric value into the unit interval [0.0, 1.0].
    """
    return float(min(1.0, max(0.0, value)))


def deterministic_valence_winner(
    normalized_scores: Mapping[str, float],
    *,
    valence_order: Sequence[str] = VALENCE_ORDER,
) -> str:
    """Return the highest-scoring valence, with earlier VALENCE_ORDER entries winning ties."""
    return max(
        valence_order,
        key=lambda name: (normalized_scores[name], -valence_order.index(name)),
    )


def arbitration_evidence_input_dim(
    *,
    arbitration_evidence_fields: Mapping[str, Sequence[str]] = ARBITRATION_EVIDENCE_FIELDS,
) -> int:
    """
    Compute the total number of scalar signals in the flattened arbitration evidence vector.
    """
    return sum(len(fields) for fields in arbitration_evidence_fields.values())


def arbitration_evidence_signal_names(
    *,
    valence_order: Sequence[str] = VALENCE_ORDER,
    arbitration_evidence_fields: Mapping[str, Sequence[str]] = ARBITRATION_EVIDENCE_FIELDS,
) -> tuple[str, ...]:
    """
    Return the flattened arbitration evidence signal names in the exact order expected by the arbitration network.
    """
    return tuple(
        f"{valence}.{evidence_field}"
        for valence in valence_order
        for evidence_field in arbitration_evidence_fields[valence]
    )


def arbitration_evidence_vector(
    evidence: Mapping[str, Mapping[str, float]],
    *,
    valence_order: Sequence[str] = VALENCE_ORDER,
    arbitration_evidence_fields: Mapping[str, Sequence[str]] = ARBITRATION_EVIDENCE_FIELDS,
) -> np.ndarray:
    """
    Flatten arbitration evidence into a 1D numpy array matching the exact input order expected by the ArbitrationNetwork.
    """
    signal_names = arbitration_evidence_signal_names(
        valence_order=valence_order,
        arbitration_evidence_fields=arbitration_evidence_fields,
    )
    if signal_names != ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES:
        raise RuntimeError("Arbitration evidence order does not match ArbitrationNetwork input order.")
    return np.array(
        [
            float(evidence[valence][evidence_field])
            for valence in valence_order
            for evidence_field in arbitration_evidence_fields[valence]
        ],
        dtype=float,
    )


def warm_start_arbitration_network(
    arbitration_network: ArbitrationNetwork,
    *,
    ablation_config: BrainAblationConfig,
    warm_start_scale: float | None = None,
    valence_order: Sequence[str] = VALENCE_ORDER,
    arbitration_evidence_fields: Mapping[str, Sequence[str]] = ARBITRATION_EVIDENCE_FIELDS,
    valence_evidence_weights: Mapping[str, Mapping[str, float]] = VALENCE_EVIDENCE_WEIGHTS,
) -> None:
    """
    Initialize the arbitration network so its initial valence logits approximate the legacy fixed-formula scores.
    """
    net = arbitration_network
    scale = (
        float(ablation_config.warm_start_scale)
        if warm_start_scale is None
        else float(warm_start_scale)
    )
    if not math.isfinite(scale):
        raise ValueError("warm_start_scale must be finite.")
    if scale < 0.0 or scale > 1.0:
        raise ValueError("warm_start_scale must be in [0.0, 1.0].")
    if scale == 0.0:
        return
    signal_names = arbitration_evidence_signal_names(
        valence_order=valence_order,
        arbitration_evidence_fields=arbitration_evidence_fields,
    )
    if signal_names != ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES:
        raise RuntimeError("Arbitration evidence order does not match ArbitrationNetwork input order.")
    if net.hidden_dim < net.input_dim:
        raise ValueError("Arbitration warm start requires hidden_dim >= input_dim.")

    net.W1.fill(0.0)
    net.b1.fill(0.0)
    net.W2_valence.fill(0.0)
    net.b2_valence.fill(0.0)
    net.W2_gate.fill(0.0)
    net.b2_gate.fill(0.0)
    net.W2_value.fill(0.0)
    net.b2_value.fill(0.0)

    copy_scale = 0.1
    valence_logit_scale = 6.0
    per_side_scale = math.sqrt(scale)
    for index in range(net.input_dim):
        net.W1[index, index] = copy_scale * per_side_scale

    input_index = 0
    for evidence_valence in valence_order:
        weights = valence_evidence_weights[evidence_valence]
        for evidence_field in arbitration_evidence_fields[evidence_valence]:
            for output_index, output_valence in enumerate(valence_order):
                net.W2_valence[output_index, input_index] = (
                    per_side_scale
                    * valence_logit_scale
                    * weights.get(evidence_field, 0.0)
                    / copy_scale
                    if output_valence == evidence_valence
                    else 0.0
                )
            input_index += 1
    net.cache = None


def fixed_formula_valence_scores_from_evidence(
    evidence: Mapping[str, Mapping[str, float]],
    *,
    valence_order: Sequence[str] = VALENCE_ORDER,
    valence_evidence_weights: Mapping[str, Mapping[str, float]] = VALENCE_EVIDENCE_WEIGHTS,
    clamp_fn=clamp_unit,
) -> np.ndarray:
    """
    Compute a deterministic, fixed-formula valence distribution from structured evidence for arbitration regularization.
    """
    raw_scores = {
        valence: clamp_fn(
            sum(
                weight * evidence[valence][evidence_field]
                for evidence_field, weight in valence_evidence_weights[valence].items()
            )
        )
        for valence in valence_order
    }
    total = float(sum(raw_scores.values()))
    if total <= 1e-8:
        fallback = np.zeros(len(valence_order), dtype=float)
        fallback[valence_order.index("exploration")] = 1.0
        return fallback
    return np.array(
        [
            float(raw_scores[name] / total)
            for name in valence_order
        ],
        dtype=float,
    )


def _bound_observation(
    interface_name: str,
    observations: Mapping[str, np.ndarray],
) -> dict[str, float]:
    interface = MODULE_INTERFACE_BY_NAME.get(interface_name)
    if interface is None:
        raise KeyError(f"Unknown interface: {interface_name}")
    sanitized_obs = np.nan_to_num(
        np.asarray(observations[interface.observation_key], dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    return interface.bind_values(sanitized_obs)


def _bound_action_context(observations: Mapping[str, np.ndarray]) -> dict[str, float]:
    sanitized_obs = np.nan_to_num(
        np.asarray(observations[ACTION_CONTEXT_INTERFACE.observation_key], dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    return ACTION_CONTEXT_INTERFACE.bind_values(sanitized_obs)


def proposal_contribution_share(
    module_results: list[ModuleResult],
    gated_logits: Sequence[np.ndarray],
) -> Dict[str, float]:
    """
    Compute normalized proposal-contribution shares from gated logits.
    """
    active_results = [result for result in module_results if result.active]
    if not active_results:
        return {
            result.name: 0.0
            for result in module_results
        }
    magnitudes = {
        result.name: float(np.sum(np.abs(gated_logit)))
        for result, gated_logit in zip(module_results, gated_logits, strict=True)
        if result.active
    }
    total = float(sum(magnitudes.values()))
    if total <= 1e-8:
        uniform = 1.0 / float(len(active_results))
        return {
            result.name: (uniform if result.active else 0.0)
            for result in module_results
        }
    return {
        result.name: (
            float(magnitudes.get(result.name, 0.0) / total)
            if result.active
            else 0.0
        )
        for result in module_results
    }


def compute_arbitration(
    observations: Mapping[str, np.ndarray],
    module_results: list[ModuleResult],
    *,
    arbitration_network: ArbitrationNetwork,
    ablation_config: BrainAblationConfig,
    arbitration_rng: np.random.Generator,
    operational_profile: OperationalProfile | None = None,
    training: bool = False,
    store_cache: bool = True,
    clamp_fn=clamp_unit,
    valence_order: Sequence[str] = VALENCE_ORDER,
    arbitration_evidence_fields: Mapping[str, Sequence[str]] = ARBITRATION_EVIDENCE_FIELDS,
    valence_evidence_weights: Mapping[str, Mapping[str, float]] = VALENCE_EVIDENCE_WEIGHTS,
    arbitration_gate_module_order: Sequence[str] = ARBITRATION_GATE_MODULE_ORDER,
    priority_gating_weights: Mapping[str, Mapping[str, float]] = PRIORITY_GATING_WEIGHTS,
) -> ArbitrationDecision:
    """
    Compute an arbitration decision: choose a valence and produce per-module gate weights and diagnostics.
    """
    action_context = _bound_action_context(observations)
    visual = _bound_observation("visual_cortex", observations)
    sensory = _bound_observation("sensory_cortex", observations)
    hunger = _bound_observation("hunger_center", observations)
    sleep = _bound_observation("sleep_center", observations)
    alert = _bound_observation("alert_center", observations)

    threat_evidence = {
        "predator_visible": action_context["predator_visible"],
        "predator_certainty": action_context["predator_certainty"],
        "predator_motion_salience": alert["predator_motion_salience"],
        "recent_contact": action_context["recent_contact"],
        "recent_pain": action_context["recent_pain"],
        "predator_smell_strength": alert["predator_smell_strength"],
    }
    threat_raw = clamp_fn(
        sum(
            valence_evidence_weights["threat"][key] * threat_evidence[key]
            for key in threat_evidence.keys()
        )
    )

    hunger_memory_freshness = clamp_fn(1.0 - hunger["food_memory_age"])
    hunger_evidence = {
        "hunger": action_context["hunger"],
        "on_food": action_context["on_food"],
        "food_visible": hunger["food_visible"],
        "food_certainty": hunger["food_certainty"],
        "food_smell_strength": hunger["food_smell_strength"],
        "food_memory_freshness": hunger_memory_freshness,
    }
    hunger_raw = clamp_fn(
        sum(
            valence_evidence_weights["hunger"][key] * hunger_evidence[key]
            for key in hunger_evidence.keys()
        )
    )

    sleep_path_confidence = clamp_fn(
        max(
            sleep["shelter_trace_strength"],
            1.0 - sleep["shelter_memory_age"],
        )
    )
    sleep_evidence = {
        "fatigue": action_context["fatigue"],
        "sleep_debt": action_context["sleep_debt"],
        "night": action_context["night"],
        "on_shelter": action_context["on_shelter"],
        "shelter_role_level": action_context["shelter_role_level"],
        "shelter_path_confidence": sleep_path_confidence,
    }
    sleep_raw = clamp_fn(
        sum(
            valence_evidence_weights["sleep"][key] * sleep_evidence[key]
            for key in sleep_evidence.keys()
        )
    )

    exploration_safety = clamp_fn(1.0 - threat_raw)
    exploration_residual = clamp_fn(1.0 - max(threat_raw, hunger_raw, sleep_raw))
    exploration_evidence = {
        "safety_margin": exploration_safety,
        "residual_drive": exploration_residual,
        "day": action_context["day"],
        "off_shelter": clamp_fn(1.0 - action_context["on_shelter"]),
        "visual_openness": max(
            visual["food_visible"],
            visual["shelter_visible"],
            visual["predator_visible"],
        ),
        "food_smell_directionality": clamp_fn(
            abs(sensory["food_smell_dx"]) + abs(sensory["food_smell_dy"])
        ),
    }
    exploration_raw = clamp_fn(
        sum(
            valence_evidence_weights["exploration"][key] * exploration_evidence[key]
            for key in exploration_evidence.keys()
        )
    )

    evidence = {
        "threat": threat_evidence,
        "hunger": hunger_evidence,
        "sleep": sleep_evidence,
        "exploration": exploration_evidence,
    }
    if ablation_config.use_learned_arbitration:
        evidence_vector = arbitration_evidence_vector(
            evidence,
            valence_order=valence_order,
            arbitration_evidence_fields=arbitration_evidence_fields,
        )
        valence_logits_array, gate_adjustments_array, arbitration_value = arbitration_network.forward(
            evidence_vector,
            store_cache=store_cache,
        )
        valence_probs = softmax(valence_logits_array)
        normalized_scores = {
            name: float(valence_probs[index])
            for index, name in enumerate(valence_order)
        }
        valence_logits = {
            name: float(valence_logits_array[index])
            for index, name in enumerate(valence_order)
        }
        if training:
            winning_index = int(arbitration_rng.choice(len(valence_order), p=valence_probs))
            winning_valence = valence_order[winning_index]
        else:
            winning_valence = deterministic_valence_winner(
                normalized_scores,
                valence_order=valence_order,
            )

        adjustment_by_module = {
            name: float(gate_adjustments_array[index])
            for index, name in enumerate(arbitration_gate_module_order)
        }
    else:
        arbitration_network.cache = None
        valence_probs = fixed_formula_valence_scores_from_evidence(
            evidence,
            valence_order=valence_order,
            valence_evidence_weights=valence_evidence_weights,
            clamp_fn=clamp_fn,
        )
        normalized_scores = {
            name: float(valence_probs[index])
            for index, name in enumerate(valence_order)
        }
        valence_logits = {
            "threat": float(threat_raw),
            "hunger": float(hunger_raw),
            "sleep": float(sleep_raw),
            "exploration": float(exploration_raw),
        }
        winning_valence = deterministic_valence_winner(
            normalized_scores,
            valence_order=valence_order,
        )
        adjustment_by_module = {
            name: 1.0
            for name in arbitration_gate_module_order
        }
        arbitration_value = 0.0
    guards_applied = False
    if ablation_config.enable_deterministic_guards:
        if (
            not training
            and threat_evidence["predator_visible"] >= 0.5
            and threat_evidence["predator_certainty"] >= 0.35
        ):
            winning_valence = "threat"
            guards_applied = True
        elif (
            not training
            and threat_raw < 0.25
            and hunger_evidence["hunger"] >= 0.5
            and sleep_evidence["night"] < 0.5
        ):
            winning_valence = "hunger"
            guards_applied = True

    pre_gating_logits = np.sum(
        np.stack([result.logits for result in module_results], axis=0),
        axis=0,
    )
    intent_before_gating_idx = int(np.argmax(pre_gating_logits))
    module_gates: Dict[str, float] = {}
    base_gates: Dict[str, float] = {}
    gate_adjustments: Dict[str, float] = {}
    for result in module_results:
        base_gate = priority_gate_weight_for(
            winning_valence,
            result.name,
            priority_gating_weights=priority_gating_weights,
        )
        adjustment = adjustment_by_module.get(result.name, 1.0)
        base_gates[result.name] = base_gate
        gate_adjustments[result.name] = adjustment
        module_gates[result.name] = float(np.clip(base_gate * adjustment, 0.0, 1.0))
    suppressed_modules = [
        result.name
        for result in module_results
        if result.active and module_gates[result.name] < 0.999
    ]
    gated_logits = [
        module_gates[result.name] * result.logits
        for result in module_results
    ]
    intent_after_gating_idx = int(
        np.argmax(np.sum(np.stack(gated_logits, axis=0), axis=0))
    )
    module_contribution_share = proposal_contribution_share(
        module_results,
        gated_logits,
    )
    dominant_module = ""
    dominant_module_share = 0.0
    effective_module_count = 0.0
    positive_shares = [
        share
        for share in module_contribution_share.values()
        if share > 1e-8
    ]
    if positive_shares:
        dominant_module = max(
            module_results,
            key=lambda result: (
                module_contribution_share[result.name],
                -module_results.index(result),
            ),
        ).name
        dominant_module_share = float(module_contribution_share[dominant_module])
        effective_module_count = float(
            1.0 / sum(share * share for share in positive_shares)
        )
    active_results = [result for result in module_results if result.active]
    if active_results:
        agreeing_modules = sum(
            1
            for result, gated_logit in zip(module_results, gated_logits, strict=True)
            if result.active and int(np.argmax(gated_logit)) == intent_after_gating_idx
        )
        module_agreement_rate = float(agreeing_modules / len(active_results))
    else:
        module_agreement_rate = 0.0
    module_disagreement_rate = float(1.0 - module_agreement_rate) if active_results else 0.0
    return ArbitrationDecision(
        strategy="priority_gating",
        winning_valence=winning_valence,
        valence_scores=normalized_scores,
        module_gates=module_gates,
        module_contribution_share=module_contribution_share,
        dominant_module=dominant_module,
        dominant_module_share=dominant_module_share,
        effective_module_count=effective_module_count,
        module_agreement_rate=module_agreement_rate,
        module_disagreement_rate=module_disagreement_rate,
        suppressed_modules=suppressed_modules,
        evidence=evidence,
        intent_before_gating_idx=intent_before_gating_idx,
        intent_after_gating_idx=intent_after_gating_idx,
        valence_logits=valence_logits,
        base_gates=base_gates,
        gate_adjustments=gate_adjustments,
        arbitration_value=float(arbitration_value),
        learned_adjustment=bool(ablation_config.use_learned_arbitration),
        guards_applied=guards_applied,
    )


def priority_gate_weight_for(
    winning_valence: str,
    module_name: str,
    *,
    priority_gating_weights: Mapping[str, Mapping[str, float]] = PRIORITY_GATING_WEIGHTS,
) -> float:
    """
    Get the configured priority gate weight for a module for the specified winning valence.
    """
    valence_weights = priority_gating_weights.get(winning_valence)
    if valence_weights is None:
        raise ValueError(
            f"Priority gating weights missing winning valence '{winning_valence}'."
        )
    if module_name not in valence_weights:
        raise ValueError(
            f"Priority gating weights missing module '{module_name}' "
            f"for winning valence '{winning_valence}'."
        )
    return float(valence_weights[module_name])


def arbitration_gate_weight_for(
    arbitration: ArbitrationDecision,
    module_name: str,
) -> float:
    """Return the gate weight exported by an arbitration decision for one module."""
    if module_name not in arbitration.module_gates:
        raise ValueError(
            f"Arbitration decision missing gate weight for module '{module_name}' "
            f"under winning valence '{arbitration.winning_valence}'."
        )
    return float(arbitration.module_gates[module_name])


def apply_priority_gating(
    module_results: list[ModuleResult],
    arbitration: ArbitrationDecision,
    *,
    module_valence_roles: Mapping[str, str],
) -> None:
    """
    Apply per-module priority gating diagnostics without overwriting raw proposal logits.
    """
    intent_before_gating = ACTIONS[arbitration.intent_before_gating_idx]
    intent_after_gating = ACTIONS[arbitration.intent_after_gating_idx]
    for result in module_results:
        raw_logits = result.logits.copy()
        result.valence_role = module_valence_roles.get(result.name, "support")
        gate_weight = arbitration_gate_weight_for(arbitration, result.name)
        result.gate_weight = gate_weight
        result.contribution_share = float(
            arbitration.module_contribution_share.get(result.name, 0.0)
        )
        result.gated_logits = gate_weight * raw_logits
        result.logits = raw_logits
        result.intent_before_gating = intent_before_gating
        result.intent_after_gating = intent_after_gating
