from __future__ import annotations

from .budget_profiles import resolve_budget
from .learning_evidence import resolve_learning_evidence_conditions
from . import comparison_learning as _comparison_learning
from . import comparison_ladder_profiles as _comparison_ladder_profiles
from . import comparison_training as _comparison_training
from .comparison_utils import (
    noise_profile_metadata,
    noise_profile_csv_value,
    safe_float,
    aggregate_with_uncertainty,
    metric_seed_values_from_payload,
    fallback_seed_values,
    paired_seed_delta_rows,
    paired_seed_effect_size_rows,
    values_only,
    seed_level_dicts,
    behavior_metric_seed_rows,
    attach_behavior_seed_statistics,
    condition_compact_summary,
    condition_mean_reward,
    module_response_by_predator_type_from_payload,
    predator_type_specialization_score,
)
from .comparison_noise import (
    with_noise_profile_metadata,
    robustness_matrix_metadata,
    condense_robustness_summary,
    matrix_cell_success_rate,
    robustness_aggregate_metrics,
    profile_comparison_metrics,
    compare_noise_robustness,
)
from .comparison_reward import (
    comparison_suite_from_payload,
    episode_history_reward_payload,
    find_austere_comparison_payload,
    austere_comparison_from_payloads,
    austere_survival_summary,
    shaping_dependent_behaviors,
    austere_survival_gate_passed,
    shaping_reduction_status,
    build_reward_audit_comparison,
    build_reward_audit,
)
from .comparison_representation import (
    representation_specialization_from_payload,
    visual_minus_olfactory_seed_rows,
    build_predator_type_specialization_summary,
)
from .comparison_learning import (
    build_distillation_comparison_report,
    build_learning_evidence_deltas,
    build_learning_evidence_summary,
)
from .comparison_behavior import (
    build_ablation_deltas,
    compare_configurations,
    compare_reward_profiles,
    compare_behavior_suite,
)
from .comparison_training import (
    _compare_named_training_regimes,
)
from .comparison_ablation import (
    compare_ablation_suite,
)
from .comparison_capacity import (
    compare_capacity_sweep,
    compare_capacity_sweeps,
)

def compare_learning_evidence(*args, **kwargs):
    _comparison_learning.resolve_budget = resolve_budget
    _comparison_learning.resolve_learning_evidence_conditions = (
        resolve_learning_evidence_conditions
    )
    return _comparison_learning.compare_learning_evidence(*args, **kwargs)

def compare_ladder_under_profiles(*args, **kwargs):
    _comparison_ladder_profiles.resolve_budget = resolve_budget
    return _comparison_ladder_profiles.compare_ladder_under_profiles(
        *args,
        **kwargs,
    )

def compare_training_regimes(*args, **kwargs):
    _comparison_training.resolve_budget = resolve_budget
    return _comparison_training.compare_training_regimes(*args, **kwargs)
