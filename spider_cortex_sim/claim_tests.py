from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence


_LEARNING_WITHOUT_PRIVILEGED_SIGNAL_SCENARIOS: tuple[str, ...] = (
    "night_rest",
    "predator_edge",
    "entrance_ambush",
    "shelter_blockade",
    "two_shelter_tradeoff",
)

_ESCAPE_WITHOUT_REFLEX_SUPPORT_SCENARIOS: tuple[str, ...] = (
    "predator_edge",
    "entrance_ambush",
    "shelter_blockade",
)

_MEMORY_IMPROVES_SHELTER_RETURN_SCENARIOS: tuple[str, ...] = (
    "night_rest",
    "two_shelter_tradeoff",
)

_NOISE_PRESERVES_THREAT_VALENCE_SCENARIOS: tuple[str, ...] = (
    "predator_edge",
    "entrance_ambush",
    "shelter_blockade",
    "visual_olfactory_pincer",
    "olfactory_ambush",
    "visual_hunter_open_field",
    "food_vs_predator_conflict",
)

_SPECIALIZATION_EMERGENCE_SCENARIOS: tuple[str, ...] = (
    "visual_olfactory_pincer",
    "olfactory_ambush",
    "visual_hunter_open_field",
)


@dataclass(frozen=True)
class ClaimTestSpec:
    name: str
    hypothesis: str
    description: str
    protocol: str
    reference_condition: str
    comparison_conditions: tuple[str, ...]
    primary_metric: str
    success_criterion: str
    effect_size_metric: str
    scenarios: tuple[str, ...]
    primary: bool = False

    def __post_init__(self) -> None:
        """
        Normalize and coerce post-initialization fields to their canonical types.
        
        Converts textual fields (name, hypothesis, description, protocol, reference_condition,
        primary_metric, success_criterion, effect_size_metric) to `str`, and normalizes
        `comparison_conditions` and `scenarios` to tuples of `str`. Updates the instance
        attributes in place so the frozen dataclass stores canonical, immutable types.
        """
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "hypothesis", str(self.hypothesis))
        object.__setattr__(self, "description", str(self.description))
        object.__setattr__(self, "protocol", str(self.protocol))
        object.__setattr__(self, "reference_condition", str(self.reference_condition))
        object.__setattr__(
            self,
            "comparison_conditions",
            tuple(str(name) for name in self.comparison_conditions),
        )
        object.__setattr__(self, "primary_metric", str(self.primary_metric))
        object.__setattr__(self, "success_criterion", str(self.success_criterion))
        object.__setattr__(self, "effect_size_metric", str(self.effect_size_metric))
        object.__setattr__(
            self,
            "scenarios",
            tuple(str(name) for name in self.scenarios),
        )
        object.__setattr__(self, "primary", bool(self.primary))


_CLAIM_TEST_REGISTRY: Dict[str, ClaimTestSpec] = {
    "learning_without_privileged_signals": ClaimTestSpec(
        name="learning_without_privileged_signals",
        hypothesis=(
            "Learned behavior should still outperform an untrained policy after "
            "privileged reflex support is removed."
        ),
        description=(
            "Primary emergence gate for whether the trained policy shows genuine "
            "learning without privileged signals."
        ),
        protocol=(
            "Compose the learning-evidence workflow with reference_condition="
            "'random_init' and comparison_conditions=('trained_without_reflex_support',) "
            "over the canonical claim scenarios, then fail the claim if the leakage "
            "audit reports any privileged-signal leakage."
        ),
        reference_condition="random_init",
        comparison_conditions=("trained_without_reflex_support",),
        primary_metric="scenario_success_rate",
        success_criterion=(
            "Pass when trained_without_reflex_support improves "
            "scenario_success_rate over random_init by at least 0.15 and the "
            "leakage audit reports zero privileged-signal leakage findings."
        ),
        effect_size_metric="scenario_success_rate_delta",
        scenarios=_LEARNING_WITHOUT_PRIVILEGED_SIGNAL_SCENARIOS,
        primary=True,
    ),
    "escape_without_reflex_support": ClaimTestSpec(
        name="escape_without_reflex_support",
        hypothesis=(
            "Escape behavior should remain sustained by learned policy even when "
            "reflex-only support is not available."
        ),
        description=(
            "Predator-response claim test for whether no-reflex behavior stays "
            "stronger than the reflex-only baseline."
        ),
        protocol=(
            "Compose the learning-evidence workflow with reference_condition="
            "'reflex_only' and comparison_conditions=('trained_without_reflex_support',) "
            "restricted to predator_edge, entrance_ambush, and shelter_blockade."
        ),
        reference_condition="reflex_only",
        comparison_conditions=("trained_without_reflex_support",),
        primary_metric="predator_response_scenario_success_rate",
        success_criterion=(
            "Pass when trained_without_reflex_support reaches "
            "predator_response_scenario_success_rate >= 0.60 and exceeds "
            "reflex_only by at least 0.10 across the escape scenarios."
        ),
        effect_size_metric="predator_response_scenario_success_rate_delta",
        scenarios=_ESCAPE_WITHOUT_REFLEX_SUPPORT_SCENARIOS,
        primary=True,
    ),
    "memory_improves_shelter_return": ClaimTestSpec(
        name="memory_improves_shelter_return",
        hypothesis=(
            "Within-episode recurrent memory should improve return-to-shelter "
            "behavior in delayed or trade-off-heavy shelter tasks."
        ),
        description=(
            "Shelter-return claim test comparing the recurrent modular ablation "
            "against the feed-forward modular reference."
        ),
        protocol=(
            "Compose the ablation suite with reference_condition='modular_full' "
            "and comparison_conditions=('modular_recurrent',) restricted to "
            "night_rest and two_shelter_tradeoff."
        ),
        reference_condition="modular_full",
        comparison_conditions=("modular_recurrent",),
        primary_metric="shelter_return_scenario_success_rate",
        success_criterion=(
            "Pass when modular_recurrent improves "
            "shelter_return_scenario_success_rate over modular_full by at "
            "least 0.10 across the shelter-return scenarios."
        ),
        effect_size_metric="shelter_return_scenario_success_rate_delta",
        scenarios=_MEMORY_IMPROVES_SHELTER_RETURN_SCENARIOS,
    ),
    "noise_preserves_threat_valence": ClaimTestSpec(
        name="noise_preserves_threat_valence",
        hypothesis=(
            "Threat-oriented arbitration should remain stable under train/eval "
            "noise mismatch rather than collapsing outside matched conditions."
        ),
        description=(
            "Noise-robustness claim test for whether threat-response behavior is "
            "preserved across diagonal and off-diagonal noise conditions."
        ),
        protocol=(
            "Compose the canonical 4x4 noise-robustness matrix on the "
            "threat-response scenarios and compare diagonal versus off-diagonal "
            "aggregate scores."
        ),
        reference_condition="diagonal",
        comparison_conditions=("off_diagonal",),
        primary_metric="threat_response_off_diagonal_score",
        success_criterion=(
            "Pass when threat_response_off_diagonal_score >= 0.60 and "
            "diagonal_minus_off_diagonal_score <= 0.15 across the canonical "
            "noise-robustness matrix."
        ),
        effect_size_metric="diagonal_minus_off_diagonal_score",
        scenarios=_NOISE_PRESERVES_THREAT_VALENCE_SCENARIOS,
    ),
    "specialization_emerges_with_multiple_predators": ClaimTestSpec(
        name="specialization_emerges_with_multiple_predators",
        hypothesis=(
            "Multiple predator ecologies should produce predator-type-specific "
            "specialization rather than one undifferentiated threat pathway."
        ),
        description=(
            "Multi-predator claim test that reads specialization through "
            "ablation differentials and type-specific cortex engagement."
        ),
        protocol=(
            "Compose the predator-type ablation comparison with "
            "reference_condition='modular_full' and comparison_conditions="
            "('drop_visual_cortex', 'drop_sensory_cortex') over "
            "visual_olfactory_pincer, olfactory_ambush, and "
            "visual_hunter_open_field, while also checking type-specific cortex "
            "engagement in the reference traces."
        ),
        reference_condition="modular_full",
        comparison_conditions=("drop_visual_cortex", "drop_sensory_cortex"),
        primary_metric="visual_minus_olfactory_success_rate",
        success_criterion=(
            "Pass when drop_visual_cortex has "
            "visual_minus_olfactory_success_rate <= -0.10, "
            "drop_sensory_cortex has visual_minus_olfactory_success_rate >= 0.10, "
            "and modular_full shows type-specific cortex engagement in at least "
            "2 of the 3 specialization scenarios."
        ),
        effect_size_metric="visual_minus_olfactory_success_rate_delta",
        scenarios=_SPECIALIZATION_EMERGENCE_SCENARIOS,
        primary=True,
    ),
}


def canonical_claim_tests() -> list[ClaimTestSpec]:
    """
    Get the canonical claim-test specifications in registry order.
    
    Returns:
        list[ClaimTestSpec]: The list of registered ClaimTestSpec objects preserving the registry's insertion order.
    """
    return list(_CLAIM_TEST_REGISTRY.values())


def claim_test_names() -> list[str]:
    """
    List available canonical claim-test identifiers in registry order.
    
    Returns:
        list[str]: Claim-test names in the canonical registry, preserved in insertion order.
    """
    return list(_CLAIM_TEST_REGISTRY.keys())


def primary_claim_test_names() -> tuple[str, ...]:
    """Return the canonical primary claim names derived from the registry."""
    return tuple(spec.name for spec in _CLAIM_TEST_REGISTRY.values() if spec.primary)


def resolve_claim_tests(names: Sequence[str] | None) -> list[ClaimTestSpec]:
    """
    Resolve a sequence of claim-test names to their canonical ClaimTestSpec objects.
    
    Parameters:
        names (Sequence[str] | None): Names of claim tests to resolve. If `None`, all available claim tests are returned in registry order.
    
    Returns:
        list[ClaimTestSpec]: The resolved ClaimTestSpec objects in the same order as `names` (or registry order when `names` is `None`).
    
    Raises:
        ValueError: If any requested name is not a known claim test.
    """
    requested_names = claim_test_names() if names is None else [str(name) for name in names]
    missing = [name for name in requested_names if name not in _CLAIM_TEST_REGISTRY]
    if missing:
        raise ValueError(f"Invalid claim tests: {missing}.")
    return [_CLAIM_TEST_REGISTRY[name] for name in requested_names]


__all__ = [
    "ClaimTestSpec",
    "canonical_claim_tests",
    "claim_test_names",
    "primary_claim_test_names",
    "resolve_claim_tests",
]
