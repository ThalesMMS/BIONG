from __future__ import annotations

from .extraction.training import (
    _extract_reward_from_episode,
    extract_training_eval_series,
    _suite_from_summary,
    _scenario_suite_from_rows,
    extract_scenario_success,
)
from .extraction.shaping import (
    _profile_disposition_weights,
    _component_disposition_rows,
    _component_disposition_summary,
    _normalize_behavior_survival,
    extract_shaping_audit,
    _rows_with_minimal_reflex_support,
    _variant_with_minimal_reflex_support,
    _summary_delta,
    _scenario_delta,
    _ablation_deltas_vs_reference,
)
from .extraction.benchmark import (
    CAPACITY_MATCH_RATIO_THRESHOLD,
    compare_capacity_totals,
    extract_architecture_capacity,
    extract_capacity_sweeps,
    extract_model_capacity,
    build_primary_benchmark,
    build_reflex_dependence_indicators,
    _format_failure_indicator,
    _fallback_group_summary,
)
from .extraction.noise import (
    _ordered_noise_conditions,
    _canonicalize_inferred_noise_conditions,
    _noise_robustness_cell_summary,
    _normalize_noise_marginals,
    _noise_robustness_metrics,
    _noise_robustness_rate,
    _noise_robustness_marginal,
    _noise_robustness_mean,
    _observed_noise_robustness_metrics,
    extract_noise_robustness,
)
from .extraction.comparison import (
    extract_comparisons,
)
from .extraction.credit import (
    _compare_credit_across_architectures,
    _interpret_credit_failure,
    extract_credit_metrics,
)
from .extraction.ladder import (
    compute_modularity_conclusion,
    detect_missing_experiments,
    extract_unified_ladder_report,
)
from .extraction.representation import (
    _normalize_module_response_by_predator_type,
    _normalize_float_map,
    _extract_first_present_float_map,
    _extract_first_present_float,
    _representation_interpretation,
    _representation_payload,
    _aggregate_representation_from_scenarios,
    _aggregate_specialization_from_scenarios,
    extract_predator_type_specialization,
    _build_representation_specialization_result,
    extract_representation_specialization,
)
from .extraction.ablation import (
    extract_ablations,
    extract_ladder_comparison,
    extract_ladder_profile_comparison,
    extract_reward_profile_ladder,
    extract_reflex_frequency,
)
