from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_requires_sequences_b1_b5 import (
    run_b1_capacity_sequence,
    run_b2_temporal_threat_sequence,
    run_b3_contact_memory_sequence,
    run_b4_recovery_balance_sequence,
)

from ._b_series_evolution_sequence_b62 import (
    run_b62_defensive_mode_sequence,
)

from ._b_series_evolution_sequences_b10_b14 import (
    run_b10_prospective_replay_sequence,
    run_b11_confidence_arbiter_sequence,
    run_b12_predictive_attention_sequence,
    run_b13_local_affordance_search_sequence,
)

from ._b_series_evolution_sequences_b14_b18 import (
    run_b14_affordance_uncertainty_sequence,
    run_b15_option_critic_sequence,
    run_b16_option_ensemble_sequence,
    run_b17_neuromodulated_ensemble_sequence,
)

from ._b_series_evolution_sequences_b18_b22 import (
    run_b18_eligibility_trace_sequence,
    run_b19_episodic_meta_memory_sequence,
    run_b20_working_memory_gate_sequence,
    run_b21_hippocampal_replay_sequence,
)

from ._b_series_evolution_sequences_b22_b26 import (
    run_b22_prospective_replay_sequence,
    run_b23_conflict_monitor_sequence,
    run_b24_precision_conflict_sequence,
    run_b25_metacognitive_confidence_sequence,
)

from ._b_series_evolution_sequences_b26_b30 import (
    run_b26_allostatic_prediction_sequence,
    run_b27_arousal_gain_sequence,
    run_b28_interoceptive_attention_sequence,
    run_b29_salience_competition_sequence,
)

from ._b_series_evolution_sequences_b30_b34 import (
    run_b30_basal_ganglia_gate_sequence,
    run_b31_dopamine_prediction_error_sequence,
    run_b32_actor_critic_value_sequence,
    run_b33_td_error_decomposition_sequence,
)

from ._b_series_evolution_sequences_b34_b38 import (
    run_b34_eligibility_credit_sequence,
    run_b35_forward_model_value_sequence,
    run_b36_latent_belief_state_sequence,
    run_b37_state_factor_gate_sequence,
)

from ._b_series_evolution_sequences_b38_b42 import (
    run_b38_factor_attention_sequence,
    run_b39_attention_binding_sequence,
    run_b40_global_workspace_sequence,
    run_b41_executive_workspace_sequence,
)

from ._b_series_evolution_sequences_b42_b46 import (
    run_b42_error_monitor_sequence,
    run_b43_adaptive_precision_sequence,
    run_b44_thalamic_relay_sequence,
    run_b45_reticular_inhibition_sequence,
)

from ._b_series_evolution_sequences_b46_b50 import (
    run_b46_corticothalamic_feedback_sequence,
    run_b47_oscillatory_synchrony_sequence,
    run_b48_cerebellar_timing_sequence,
    run_b49_striatal_action_gate_sequence,
)

from ._b_series_evolution_sequences_b50_b54 import (
    run_b50_habit_chunking_sequence,
    run_b51_dopaminergic_habit_sequence,
    run_b52_cholinergic_precision_sequence,
    run_b53_noradrenergic_arousal_sequence,
)

from ._b_series_evolution_sequences_b54_b58 import (
    run_b54_serotonergic_patience_sequence,
    run_b55_hypothalamic_drive_sequence,
    run_b56_hpa_stress_sequence,
    run_b57_insular_interoceptive_sequence,
)

from ._b_series_evolution_sequences_b58_b62 import (
    run_b58_acc_conflict_sequence,
    run_b59_prefrontal_goal_sequence,
    run_b60_orbitofrontal_value_sequence,
    run_b61_amygdala_safety_sequence,
)

from ._b_series_evolution_sequences_b5_b7 import (
    run_b5_homeostatic_arbiter_sequence,
    run_b6_risk_corridor_sequence,
)

from ._b_series_evolution_sequences_b7_b10 import (
    run_b7_affordance_budget_sequence,
    run_b8_spatial_affordance_sequence,
    run_b9_waypoint_planner_sequence,
)

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run B-series evolution diagnostics from accepted parents."
    )
    parser.add_argument("--target-level", type=int, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62), default=1)
    parser.add_argument("--root", default=str(B_SERIES_EVOLUTION_ROOT))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--b0-training-episodes", type=int, default=24)
    parser.add_argument("--b1-training-episodes", type=int, default=24)
    parser.add_argument("--b2-training-episodes", type=int, default=48)
    parser.add_argument("--b3-training-episodes", type=int, default=48)
    parser.add_argument("--b4-training-episodes", type=int, default=48)
    parser.add_argument("--b4-workers", type=int, default=1)
    parser.add_argument("--b4-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b4-ga-population", type=int, default=12)
    parser.add_argument("--b4-ga-generations", type=int, default=4)
    parser.add_argument("--b5-training-episodes", type=int, default=48)
    parser.add_argument("--b5-workers", type=int, default=1)
    parser.add_argument("--b5-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b5-ga-population", type=int, default=12)
    parser.add_argument("--b5-ga-generations", type=int, default=4)
    parser.add_argument("--b6-training-episodes", type=int, default=64)
    parser.add_argument("--b6-workers", type=int, default=1)
    parser.add_argument("--b6-search", choices=("fixed", "ga", "exhaustive"), default="exhaustive")
    parser.add_argument("--b6-ga-population", type=int, default=24)
    parser.add_argument("--b6-ga-generations", type=int, default=8)
    parser.add_argument("--b6-finalists", type=int, default=6)
    parser.add_argument("--b7-training-episodes", type=int, default=64)
    parser.add_argument("--b7-workers", type=int, default=1)
    parser.add_argument("--b7-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b7-ga-population", type=int, default=24)
    parser.add_argument("--b7-ga-generations", type=int, default=8)
    parser.add_argument("--b7-finalists", type=int, default=6)
    parser.add_argument("--b8-training-episodes", type=int, default=64)
    parser.add_argument("--b8-workers", type=int, default=1)
    parser.add_argument("--b8-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b8-ga-population", type=int, default=24)
    parser.add_argument("--b8-ga-generations", type=int, default=8)
    parser.add_argument("--b8-finalists", type=int, default=6)
    parser.add_argument("--b9-training-episodes", type=int, default=64)
    parser.add_argument("--b9-workers", type=int, default=1)
    parser.add_argument("--b9-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b9-ga-population", type=int, default=24)
    parser.add_argument("--b9-ga-generations", type=int, default=8)
    parser.add_argument("--b9-finalists", type=int, default=6)
    parser.add_argument("--b10-training-episodes", type=int, default=64)
    parser.add_argument("--b10-workers", type=int, default=1)
    parser.add_argument("--b10-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b10-ga-population", type=int, default=24)
    parser.add_argument("--b10-ga-generations", type=int, default=8)
    parser.add_argument("--b10-finalists", type=int, default=6)
    parser.add_argument("--b11-training-episodes", type=int, default=64)
    parser.add_argument("--b11-workers", type=int, default=1)
    parser.add_argument("--b11-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b11-ga-population", type=int, default=24)
    parser.add_argument("--b11-ga-generations", type=int, default=8)
    parser.add_argument("--b11-finalists", type=int, default=6)
    parser.add_argument("--b12-training-episodes", type=int, default=64)
    parser.add_argument("--b12-workers", type=int, default=1)
    parser.add_argument("--b12-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b12-ga-population", type=int, default=24)
    parser.add_argument("--b12-ga-generations", type=int, default=8)
    parser.add_argument("--b12-finalists", type=int, default=6)
    parser.add_argument("--b13-training-episodes", type=int, default=64)
    parser.add_argument("--b13-workers", type=int, default=1)
    parser.add_argument("--b13-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b13-ga-population", type=int, default=24)
    parser.add_argument("--b13-ga-generations", type=int, default=8)
    parser.add_argument("--b13-finalists", type=int, default=6)
    parser.add_argument("--b14-training-episodes", type=int, default=64)
    parser.add_argument("--b14-workers", type=int, default=1)
    parser.add_argument("--b14-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b14-ga-population", type=int, default=24)
    parser.add_argument("--b14-ga-generations", type=int, default=8)
    parser.add_argument("--b14-finalists", type=int, default=6)
    parser.add_argument("--b15-training-episodes", type=int, default=64)
    parser.add_argument("--b15-workers", type=int, default=1)
    parser.add_argument("--b15-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b15-ga-population", type=int, default=24)
    parser.add_argument("--b15-ga-generations", type=int, default=8)
    parser.add_argument("--b15-finalists", type=int, default=6)
    parser.add_argument("--b16-training-episodes", type=int, default=64)
    parser.add_argument("--b16-workers", type=int, default=1)
    parser.add_argument("--b16-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b16-ga-population", type=int, default=24)
    parser.add_argument("--b16-ga-generations", type=int, default=8)
    parser.add_argument("--b16-finalists", type=int, default=6)
    parser.add_argument("--b17-training-episodes", type=int, default=64)
    parser.add_argument("--b17-workers", type=int, default=1)
    parser.add_argument("--b17-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b17-ga-population", type=int, default=24)
    parser.add_argument("--b17-ga-generations", type=int, default=8)
    parser.add_argument("--b17-finalists", type=int, default=6)
    parser.add_argument("--b18-training-episodes", type=int, default=64)
    parser.add_argument("--b18-workers", type=int, default=1)
    parser.add_argument("--b18-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b18-ga-population", type=int, default=24)
    parser.add_argument("--b18-ga-generations", type=int, default=8)
    parser.add_argument("--b18-finalists", type=int, default=6)
    parser.add_argument("--b19-training-episodes", type=int, default=64)
    parser.add_argument("--b19-workers", type=int, default=1)
    parser.add_argument("--b19-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b19-ga-population", type=int, default=24)
    parser.add_argument("--b19-ga-generations", type=int, default=8)
    parser.add_argument("--b19-finalists", type=int, default=6)
    parser.add_argument("--b20-training-episodes", type=int, default=64)
    parser.add_argument("--b20-workers", type=int, default=1)
    parser.add_argument("--b20-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b20-ga-population", type=int, default=24)
    parser.add_argument("--b20-ga-generations", type=int, default=8)
    parser.add_argument("--b20-finalists", type=int, default=6)
    parser.add_argument("--b21-training-episodes", type=int, default=64)
    parser.add_argument("--b21-workers", type=int, default=1)
    parser.add_argument("--b21-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b21-ga-population", type=int, default=24)
    parser.add_argument("--b21-ga-generations", type=int, default=8)
    parser.add_argument("--b21-finalists", type=int, default=6)
    parser.add_argument("--b22-training-episodes", type=int, default=64)
    parser.add_argument("--b22-workers", type=int, default=1)
    parser.add_argument("--b22-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b22-ga-population", type=int, default=24)
    parser.add_argument("--b22-ga-generations", type=int, default=8)
    parser.add_argument("--b22-finalists", type=int, default=6)
    parser.add_argument("--b23-training-episodes", type=int, default=64)
    parser.add_argument("--b23-workers", type=int, default=1)
    parser.add_argument("--b23-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b23-ga-population", type=int, default=24)
    parser.add_argument("--b23-ga-generations", type=int, default=8)
    parser.add_argument("--b23-finalists", type=int, default=6)
    parser.add_argument("--b24-training-episodes", type=int, default=64)
    parser.add_argument("--b24-workers", type=int, default=1)
    parser.add_argument("--b24-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b24-ga-population", type=int, default=24)
    parser.add_argument("--b24-ga-generations", type=int, default=8)
    parser.add_argument("--b24-finalists", type=int, default=6)
    parser.add_argument("--b25-training-episodes", type=int, default=64)
    parser.add_argument("--b25-workers", type=int, default=1)
    parser.add_argument("--b25-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b25-ga-population", type=int, default=24)
    parser.add_argument("--b25-ga-generations", type=int, default=8)
    parser.add_argument("--b25-finalists", type=int, default=6)
    parser.add_argument("--b26-training-episodes", type=int, default=64)
    parser.add_argument("--b26-workers", type=int, default=1)
    parser.add_argument("--b26-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b26-ga-population", type=int, default=24)
    parser.add_argument("--b26-ga-generations", type=int, default=8)
    parser.add_argument("--b26-finalists", type=int, default=6)
    parser.add_argument("--b27-training-episodes", type=int, default=64)
    parser.add_argument("--b27-workers", type=int, default=1)
    parser.add_argument("--b27-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b27-ga-population", type=int, default=24)
    parser.add_argument("--b27-ga-generations", type=int, default=8)
    parser.add_argument("--b27-finalists", type=int, default=6)
    parser.add_argument("--b28-training-episodes", type=int, default=64)
    parser.add_argument("--b28-workers", type=int, default=1)
    parser.add_argument("--b28-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b28-ga-population", type=int, default=24)
    parser.add_argument("--b28-ga-generations", type=int, default=8)
    parser.add_argument("--b28-finalists", type=int, default=6)
    parser.add_argument("--b29-training-episodes", type=int, default=64)
    parser.add_argument("--b29-workers", type=int, default=1)
    parser.add_argument("--b29-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b29-ga-population", type=int, default=24)
    parser.add_argument("--b29-ga-generations", type=int, default=8)
    parser.add_argument("--b29-finalists", type=int, default=6)
    parser.add_argument("--b30-training-episodes", type=int, default=64)
    parser.add_argument("--b30-workers", type=int, default=1)
    parser.add_argument("--b30-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b30-ga-population", type=int, default=24)
    parser.add_argument("--b30-ga-generations", type=int, default=8)
    parser.add_argument("--b30-finalists", type=int, default=6)
    parser.add_argument("--b31-training-episodes", type=int, default=64)
    parser.add_argument("--b31-workers", type=int, default=1)
    parser.add_argument("--b31-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b31-ga-population", type=int, default=24)
    parser.add_argument("--b31-ga-generations", type=int, default=8)
    parser.add_argument("--b31-finalists", type=int, default=6)
    parser.add_argument("--b32-training-episodes", type=int, default=64)
    parser.add_argument("--b32-workers", type=int, default=1)
    parser.add_argument("--b32-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b32-ga-population", type=int, default=24)
    parser.add_argument("--b32-ga-generations", type=int, default=8)
    parser.add_argument("--b32-finalists", type=int, default=6)
    parser.add_argument("--b33-training-episodes", type=int, default=64)
    parser.add_argument("--b33-workers", type=int, default=1)
    parser.add_argument("--b33-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b33-ga-population", type=int, default=24)
    parser.add_argument("--b33-ga-generations", type=int, default=8)
    parser.add_argument("--b33-finalists", type=int, default=6)
    parser.add_argument("--b34-training-episodes", type=int, default=64)
    parser.add_argument("--b34-workers", type=int, default=1)
    parser.add_argument("--b34-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b34-ga-population", type=int, default=24)
    parser.add_argument("--b34-ga-generations", type=int, default=8)
    parser.add_argument("--b34-finalists", type=int, default=6)
    parser.add_argument("--b35-training-episodes", type=int, default=64)
    parser.add_argument("--b35-workers", type=int, default=1)
    parser.add_argument("--b35-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b35-ga-population", type=int, default=24)
    parser.add_argument("--b35-ga-generations", type=int, default=8)
    parser.add_argument("--b35-finalists", type=int, default=6)
    parser.add_argument("--b36-training-episodes", type=int, default=64)
    parser.add_argument("--b36-workers", type=int, default=1)
    parser.add_argument("--b36-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b36-ga-population", type=int, default=24)
    parser.add_argument("--b36-ga-generations", type=int, default=8)
    parser.add_argument("--b36-finalists", type=int, default=6)
    parser.add_argument("--b37-training-episodes", type=int, default=64)
    parser.add_argument("--b37-workers", type=int, default=1)
    parser.add_argument("--b37-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b37-ga-population", type=int, default=24)
    parser.add_argument("--b37-ga-generations", type=int, default=8)
    parser.add_argument("--b37-finalists", type=int, default=6)
    parser.add_argument("--b38-training-episodes", type=int, default=64)
    parser.add_argument("--b38-workers", type=int, default=1)
    parser.add_argument("--b38-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b38-ga-population", type=int, default=24)
    parser.add_argument("--b38-ga-generations", type=int, default=8)
    parser.add_argument("--b38-finalists", type=int, default=6)
    parser.add_argument("--b39-training-episodes", type=int, default=64)
    parser.add_argument("--b39-workers", type=int, default=1)
    parser.add_argument("--b39-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b39-ga-population", type=int, default=24)
    parser.add_argument("--b39-ga-generations", type=int, default=8)
    parser.add_argument("--b39-finalists", type=int, default=6)
    parser.add_argument("--b40-training-episodes", type=int, default=64)
    parser.add_argument("--b40-workers", type=int, default=1)
    parser.add_argument("--b40-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b40-ga-population", type=int, default=24)
    parser.add_argument("--b40-ga-generations", type=int, default=8)
    parser.add_argument("--b40-finalists", type=int, default=6)
    parser.add_argument("--b41-training-episodes", type=int, default=64)
    parser.add_argument("--b41-workers", type=int, default=1)
    parser.add_argument("--b41-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b41-ga-population", type=int, default=24)
    parser.add_argument("--b41-ga-generations", type=int, default=8)
    parser.add_argument("--b41-finalists", type=int, default=6)
    parser.add_argument("--b42-training-episodes", type=int, default=64)
    parser.add_argument("--b42-workers", type=int, default=1)
    parser.add_argument("--b42-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b42-ga-population", type=int, default=24)
    parser.add_argument("--b42-ga-generations", type=int, default=8)
    parser.add_argument("--b42-finalists", type=int, default=6)
    parser.add_argument("--b43-training-episodes", type=int, default=64)
    parser.add_argument("--b43-workers", type=int, default=1)
    parser.add_argument("--b43-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b43-ga-population", type=int, default=24)
    parser.add_argument("--b43-ga-generations", type=int, default=8)
    parser.add_argument("--b43-finalists", type=int, default=6)
    parser.add_argument("--b44-training-episodes", type=int, default=64)
    parser.add_argument("--b44-workers", type=int, default=1)
    parser.add_argument("--b44-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b44-ga-population", type=int, default=24)
    parser.add_argument("--b44-ga-generations", type=int, default=8)
    parser.add_argument("--b44-finalists", type=int, default=6)
    parser.add_argument("--b45-training-episodes", type=int, default=64)
    parser.add_argument("--b45-workers", type=int, default=1)
    parser.add_argument("--b45-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b45-ga-population", type=int, default=24)
    parser.add_argument("--b45-ga-generations", type=int, default=8)
    parser.add_argument("--b45-finalists", type=int, default=6)
    parser.add_argument("--b46-training-episodes", type=int, default=64)
    parser.add_argument("--b46-workers", type=int, default=1)
    parser.add_argument("--b46-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b46-ga-population", type=int, default=24)
    parser.add_argument("--b46-ga-generations", type=int, default=8)
    parser.add_argument("--b46-finalists", type=int, default=6)
    parser.add_argument("--b47-training-episodes", type=int, default=64)
    parser.add_argument("--b47-workers", type=int, default=1)
    parser.add_argument("--b47-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b47-ga-population", type=int, default=24)
    parser.add_argument("--b47-ga-generations", type=int, default=8)
    parser.add_argument("--b47-finalists", type=int, default=6)
    parser.add_argument("--b48-training-episodes", type=int, default=64)
    parser.add_argument("--b48-workers", type=int, default=1)
    parser.add_argument("--b48-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b48-ga-population", type=int, default=24)
    parser.add_argument("--b48-ga-generations", type=int, default=8)
    parser.add_argument("--b48-finalists", type=int, default=6)
    parser.add_argument("--b49-training-episodes", type=int, default=64)
    parser.add_argument("--b49-workers", type=int, default=1)
    parser.add_argument("--b49-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b49-ga-population", type=int, default=24)
    parser.add_argument("--b49-ga-generations", type=int, default=8)
    parser.add_argument("--b49-finalists", type=int, default=6)
    parser.add_argument("--b50-training-episodes", type=int, default=64)
    parser.add_argument("--b50-workers", type=int, default=1)
    parser.add_argument("--b50-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b50-ga-population", type=int, default=24)
    parser.add_argument("--b50-ga-generations", type=int, default=8)
    parser.add_argument("--b50-finalists", type=int, default=6)
    parser.add_argument("--b51-training-episodes", type=int, default=64)
    parser.add_argument("--b51-workers", type=int, default=1)
    parser.add_argument("--b51-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b51-ga-population", type=int, default=24)
    parser.add_argument("--b51-ga-generations", type=int, default=8)
    parser.add_argument("--b51-finalists", type=int, default=6)
    parser.add_argument("--b52-training-episodes", type=int, default=64)
    parser.add_argument("--b52-workers", type=int, default=1)
    parser.add_argument("--b52-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b52-ga-population", type=int, default=24)
    parser.add_argument("--b52-ga-generations", type=int, default=8)
    parser.add_argument("--b52-finalists", type=int, default=6)
    parser.add_argument("--b53-training-episodes", type=int, default=64)
    parser.add_argument("--b53-workers", type=int, default=1)
    parser.add_argument("--b53-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b53-ga-population", type=int, default=24)
    parser.add_argument("--b53-ga-generations", type=int, default=8)
    parser.add_argument("--b53-finalists", type=int, default=6)
    parser.add_argument("--b54-training-episodes", type=int, default=64)
    parser.add_argument("--b54-workers", type=int, default=1)
    parser.add_argument("--b54-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b54-ga-population", type=int, default=24)
    parser.add_argument("--b54-ga-generations", type=int, default=8)
    parser.add_argument("--b54-finalists", type=int, default=6)
    parser.add_argument("--b55-training-episodes", type=int, default=64)
    parser.add_argument("--b55-workers", type=int, default=1)
    parser.add_argument("--b55-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b55-ga-population", type=int, default=24)
    parser.add_argument("--b55-ga-generations", type=int, default=8)
    parser.add_argument("--b55-finalists", type=int, default=6)
    parser.add_argument("--b56-training-episodes", type=int, default=64)
    parser.add_argument("--b56-workers", type=int, default=1)
    parser.add_argument("--b56-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b56-ga-population", type=int, default=24)
    parser.add_argument("--b56-ga-generations", type=int, default=8)
    parser.add_argument("--b56-finalists", type=int, default=6)
    parser.add_argument("--b57-training-episodes", type=int, default=64)
    parser.add_argument("--b57-workers", type=int, default=1)
    parser.add_argument("--b57-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b57-ga-population", type=int, default=24)
    parser.add_argument("--b57-ga-generations", type=int, default=8)
    parser.add_argument("--b57-finalists", type=int, default=6)
    parser.add_argument("--b58-training-episodes", type=int, default=64)
    parser.add_argument("--b58-workers", type=int, default=1)
    parser.add_argument("--b58-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b58-ga-population", type=int, default=24)
    parser.add_argument("--b58-ga-generations", type=int, default=8)
    parser.add_argument("--b58-finalists", type=int, default=6)
    parser.add_argument("--b59-training-episodes", type=int, default=64)
    parser.add_argument("--b59-workers", type=int, default=1)
    parser.add_argument("--b59-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b59-ga-population", type=int, default=24)
    parser.add_argument("--b59-ga-generations", type=int, default=8)
    parser.add_argument("--b59-finalists", type=int, default=6)
    parser.add_argument("--b60-training-episodes", type=int, default=64)
    parser.add_argument("--b60-workers", type=int, default=1)
    parser.add_argument("--b60-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b60-ga-population", type=int, default=24)
    parser.add_argument("--b60-ga-generations", type=int, default=8)
    parser.add_argument("--b60-finalists", type=int, default=6)
    parser.add_argument("--b61-training-episodes", type=int, default=64)
    parser.add_argument("--b61-workers", type=int, default=1)
    parser.add_argument("--b61-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b61-ga-population", type=int, default=24)
    parser.add_argument("--b61-ga-generations", type=int, default=8)
    parser.add_argument("--b61-finalists", type=int, default=6)
    parser.add_argument("--b62-training-episodes", type=int, default=64)
    parser.add_argument("--b62-workers", type=int, default=1)
    parser.add_argument("--b62-search", choices=("fixed", "ga", "hybrid"), default="hybrid")
    parser.add_argument("--b62-ga-population", type=int, default=24)
    parser.add_argument("--b62-ga-generations", type=int, default=8)
    parser.add_argument("--b62-finalists", type=int, default=6)
    parser.add_argument("--reward-profile", default="ecological")
    parser.add_argument("--operational-profile", default="default_v1")
    parser.add_argument("--noise-profile", default="none")
    parser.add_argument("--force-b0", action="store_true")
    parser.add_argument("--skip-canonical", action="store_true")
    args = parser.parse_args(argv)
    if int(args.target_level) == 62:
        summary = run_b62_defensive_mode_sequence(
            root=args.root,
            seed=args.seed,
            b62_training_episodes=args.b62_training_episodes,
            b62_workers=args.b62_workers,
            b62_search=args.b62_search,
            b62_ga_population=args.b62_ga_population,
            b62_ga_generations=args.b62_ga_generations,
            b62_finalists=args.b62_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 61:
        summary = run_b61_amygdala_safety_sequence(
            root=args.root,
            seed=args.seed,
            b61_training_episodes=args.b61_training_episodes,
            b61_workers=args.b61_workers,
            b61_search=args.b61_search,
            b61_ga_population=args.b61_ga_population,
            b61_ga_generations=args.b61_ga_generations,
            b61_finalists=args.b61_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 60:
        summary = run_b60_orbitofrontal_value_sequence(
            root=args.root,
            seed=args.seed,
            b60_training_episodes=args.b60_training_episodes,
            b60_workers=args.b60_workers,
            b60_search=args.b60_search,
            b60_ga_population=args.b60_ga_population,
            b60_ga_generations=args.b60_ga_generations,
            b60_finalists=args.b60_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 59:
        summary = run_b59_prefrontal_goal_sequence(
            root=args.root,
            seed=args.seed,
            b59_training_episodes=args.b59_training_episodes,
            b59_workers=args.b59_workers,
            b59_search=args.b59_search,
            b59_ga_population=args.b59_ga_population,
            b59_ga_generations=args.b59_ga_generations,
            b59_finalists=args.b59_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 58:
        summary = run_b58_acc_conflict_sequence(
            root=args.root,
            seed=args.seed,
            b58_training_episodes=args.b58_training_episodes,
            b58_workers=args.b58_workers,
            b58_search=args.b58_search,
            b58_ga_population=args.b58_ga_population,
            b58_ga_generations=args.b58_ga_generations,
            b58_finalists=args.b58_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 57:
        summary = run_b57_insular_interoceptive_sequence(
            root=args.root,
            seed=args.seed,
            b57_training_episodes=args.b57_training_episodes,
            b57_workers=args.b57_workers,
            b57_search=args.b57_search,
            b57_ga_population=args.b57_ga_population,
            b57_ga_generations=args.b57_ga_generations,
            b57_finalists=args.b57_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 56:
        summary = run_b56_hpa_stress_sequence(
            root=args.root,
            seed=args.seed,
            b56_training_episodes=args.b56_training_episodes,
            b56_workers=args.b56_workers,
            b56_search=args.b56_search,
            b56_ga_population=args.b56_ga_population,
            b56_ga_generations=args.b56_ga_generations,
            b56_finalists=args.b56_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 55:
        summary = run_b55_hypothalamic_drive_sequence(
            root=args.root,
            seed=args.seed,
            b55_training_episodes=args.b55_training_episodes,
            b55_workers=args.b55_workers,
            b55_search=args.b55_search,
            b55_ga_population=args.b55_ga_population,
            b55_ga_generations=args.b55_ga_generations,
            b55_finalists=args.b55_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 54:
        summary = run_b54_serotonergic_patience_sequence(
            root=args.root,
            seed=args.seed,
            b54_training_episodes=args.b54_training_episodes,
            b54_workers=args.b54_workers,
            b54_search=args.b54_search,
            b54_ga_population=args.b54_ga_population,
            b54_ga_generations=args.b54_ga_generations,
            b54_finalists=args.b54_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 53:
        summary = run_b53_noradrenergic_arousal_sequence(
            root=args.root,
            seed=args.seed,
            b53_training_episodes=args.b53_training_episodes,
            b53_workers=args.b53_workers,
            b53_search=args.b53_search,
            b53_ga_population=args.b53_ga_population,
            b53_ga_generations=args.b53_ga_generations,
            b53_finalists=args.b53_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 52:
        summary = run_b52_cholinergic_precision_sequence(
            root=args.root,
            seed=args.seed,
            b52_training_episodes=args.b52_training_episodes,
            b52_workers=args.b52_workers,
            b52_search=args.b52_search,
            b52_ga_population=args.b52_ga_population,
            b52_ga_generations=args.b52_ga_generations,
            b52_finalists=args.b52_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 51:
        summary = run_b51_dopaminergic_habit_sequence(
            root=args.root,
            seed=args.seed,
            b51_training_episodes=args.b51_training_episodes,
            b51_workers=args.b51_workers,
            b51_search=args.b51_search,
            b51_ga_population=args.b51_ga_population,
            b51_ga_generations=args.b51_ga_generations,
            b51_finalists=args.b51_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 50:
        summary = run_b50_habit_chunking_sequence(
            root=args.root,
            seed=args.seed,
            b50_training_episodes=args.b50_training_episodes,
            b50_workers=args.b50_workers,
            b50_search=args.b50_search,
            b50_ga_population=args.b50_ga_population,
            b50_ga_generations=args.b50_ga_generations,
            b50_finalists=args.b50_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 49:
        summary = run_b49_striatal_action_gate_sequence(
            root=args.root,
            seed=args.seed,
            b49_training_episodes=args.b49_training_episodes,
            b49_workers=args.b49_workers,
            b49_search=args.b49_search,
            b49_ga_population=args.b49_ga_population,
            b49_ga_generations=args.b49_ga_generations,
            b49_finalists=args.b49_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 48:
        summary = run_b48_cerebellar_timing_sequence(
            root=args.root,
            seed=args.seed,
            b48_training_episodes=args.b48_training_episodes,
            b48_workers=args.b48_workers,
            b48_search=args.b48_search,
            b48_ga_population=args.b48_ga_population,
            b48_ga_generations=args.b48_ga_generations,
            b48_finalists=args.b48_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 47:
        summary = run_b47_oscillatory_synchrony_sequence(
            root=args.root,
            seed=args.seed,
            b47_training_episodes=args.b47_training_episodes,
            b47_workers=args.b47_workers,
            b47_search=args.b47_search,
            b47_ga_population=args.b47_ga_population,
            b47_ga_generations=args.b47_ga_generations,
            b47_finalists=args.b47_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 46:
        summary = run_b46_corticothalamic_feedback_sequence(
            root=args.root,
            seed=args.seed,
            b46_training_episodes=args.b46_training_episodes,
            b46_workers=args.b46_workers,
            b46_search=args.b46_search,
            b46_ga_population=args.b46_ga_population,
            b46_ga_generations=args.b46_ga_generations,
            b46_finalists=args.b46_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 45:
        summary = run_b45_reticular_inhibition_sequence(
            root=args.root,
            seed=args.seed,
            b45_training_episodes=args.b45_training_episodes,
            b45_workers=args.b45_workers,
            b45_search=args.b45_search,
            b45_ga_population=args.b45_ga_population,
            b45_ga_generations=args.b45_ga_generations,
            b45_finalists=args.b45_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 44:
        summary = run_b44_thalamic_relay_sequence(
            root=args.root,
            seed=args.seed,
            b44_training_episodes=args.b44_training_episodes,
            b44_workers=args.b44_workers,
            b44_search=args.b44_search,
            b44_ga_population=args.b44_ga_population,
            b44_ga_generations=args.b44_ga_generations,
            b44_finalists=args.b44_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 43:
        summary = run_b43_adaptive_precision_sequence(
            root=args.root,
            seed=args.seed,
            b43_training_episodes=args.b43_training_episodes,
            b43_workers=args.b43_workers,
            b43_search=args.b43_search,
            b43_ga_population=args.b43_ga_population,
            b43_ga_generations=args.b43_ga_generations,
            b43_finalists=args.b43_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 42:
        summary = run_b42_error_monitor_sequence(
            root=args.root,
            seed=args.seed,
            b42_training_episodes=args.b42_training_episodes,
            b42_workers=args.b42_workers,
            b42_search=args.b42_search,
            b42_ga_population=args.b42_ga_population,
            b42_ga_generations=args.b42_ga_generations,
            b42_finalists=args.b42_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 41:
        summary = run_b41_executive_workspace_sequence(
            root=args.root,
            seed=args.seed,
            b41_training_episodes=args.b41_training_episodes,
            b41_workers=args.b41_workers,
            b41_search=args.b41_search,
            b41_ga_population=args.b41_ga_population,
            b41_ga_generations=args.b41_ga_generations,
            b41_finalists=args.b41_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 40:
        summary = run_b40_global_workspace_sequence(
            root=args.root,
            seed=args.seed,
            b40_training_episodes=args.b40_training_episodes,
            b40_workers=args.b40_workers,
            b40_search=args.b40_search,
            b40_ga_population=args.b40_ga_population,
            b40_ga_generations=args.b40_ga_generations,
            b40_finalists=args.b40_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 39:
        summary = run_b39_attention_binding_sequence(
            root=args.root,
            seed=args.seed,
            b39_training_episodes=args.b39_training_episodes,
            b39_workers=args.b39_workers,
            b39_search=args.b39_search,
            b39_ga_population=args.b39_ga_population,
            b39_ga_generations=args.b39_ga_generations,
            b39_finalists=args.b39_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 38:
        summary = run_b38_factor_attention_sequence(
            root=args.root,
            seed=args.seed,
            b38_training_episodes=args.b38_training_episodes,
            b38_workers=args.b38_workers,
            b38_search=args.b38_search,
            b38_ga_population=args.b38_ga_population,
            b38_ga_generations=args.b38_ga_generations,
            b38_finalists=args.b38_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 37:
        summary = run_b37_state_factor_gate_sequence(
            root=args.root,
            seed=args.seed,
            b37_training_episodes=args.b37_training_episodes,
            b37_workers=args.b37_workers,
            b37_search=args.b37_search,
            b37_ga_population=args.b37_ga_population,
            b37_ga_generations=args.b37_ga_generations,
            b37_finalists=args.b37_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 36:
        summary = run_b36_latent_belief_state_sequence(
            root=args.root,
            seed=args.seed,
            b36_training_episodes=args.b36_training_episodes,
            b36_workers=args.b36_workers,
            b36_search=args.b36_search,
            b36_ga_population=args.b36_ga_population,
            b36_ga_generations=args.b36_ga_generations,
            b36_finalists=args.b36_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 35:
        summary = run_b35_forward_model_value_sequence(
            root=args.root,
            seed=args.seed,
            b35_training_episodes=args.b35_training_episodes,
            b35_workers=args.b35_workers,
            b35_search=args.b35_search,
            b35_ga_population=args.b35_ga_population,
            b35_ga_generations=args.b35_ga_generations,
            b35_finalists=args.b35_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 34:
        summary = run_b34_eligibility_credit_sequence(
            root=args.root,
            seed=args.seed,
            b34_training_episodes=args.b34_training_episodes,
            b34_workers=args.b34_workers,
            b34_search=args.b34_search,
            b34_ga_population=args.b34_ga_population,
            b34_ga_generations=args.b34_ga_generations,
            b34_finalists=args.b34_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 33:
        summary = run_b33_td_error_decomposition_sequence(
            root=args.root,
            seed=args.seed,
            b33_training_episodes=args.b33_training_episodes,
            b33_workers=args.b33_workers,
            b33_search=args.b33_search,
            b33_ga_population=args.b33_ga_population,
            b33_ga_generations=args.b33_ga_generations,
            b33_finalists=args.b33_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 32:
        summary = run_b32_actor_critic_value_sequence(
            root=args.root,
            seed=args.seed,
            b32_training_episodes=args.b32_training_episodes,
            b32_workers=args.b32_workers,
            b32_search=args.b32_search,
            b32_ga_population=args.b32_ga_population,
            b32_ga_generations=args.b32_ga_generations,
            b32_finalists=args.b32_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 31:
        summary = run_b31_dopamine_prediction_error_sequence(
            root=args.root,
            seed=args.seed,
            b31_training_episodes=args.b31_training_episodes,
            b31_workers=args.b31_workers,
            b31_search=args.b31_search,
            b31_ga_population=args.b31_ga_population,
            b31_ga_generations=args.b31_ga_generations,
            b31_finalists=args.b31_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 30:
        summary = run_b30_basal_ganglia_gate_sequence(
            root=args.root,
            seed=args.seed,
            b30_training_episodes=args.b30_training_episodes,
            b30_workers=args.b30_workers,
            b30_search=args.b30_search,
            b30_ga_population=args.b30_ga_population,
            b30_ga_generations=args.b30_ga_generations,
            b30_finalists=args.b30_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 29:
        summary = run_b29_salience_competition_sequence(
            root=args.root,
            seed=args.seed,
            b29_training_episodes=args.b29_training_episodes,
            b29_workers=args.b29_workers,
            b29_search=args.b29_search,
            b29_ga_population=args.b29_ga_population,
            b29_ga_generations=args.b29_ga_generations,
            b29_finalists=args.b29_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 28:
        summary = run_b28_interoceptive_attention_sequence(
            root=args.root,
            seed=args.seed,
            b28_training_episodes=args.b28_training_episodes,
            b28_workers=args.b28_workers,
            b28_search=args.b28_search,
            b28_ga_population=args.b28_ga_population,
            b28_ga_generations=args.b28_ga_generations,
            b28_finalists=args.b28_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 27:
        summary = run_b27_arousal_gain_sequence(
            root=args.root,
            seed=args.seed,
            b27_training_episodes=args.b27_training_episodes,
            b27_workers=args.b27_workers,
            b27_search=args.b27_search,
            b27_ga_population=args.b27_ga_population,
            b27_ga_generations=args.b27_ga_generations,
            b27_finalists=args.b27_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 26:
        summary = run_b26_allostatic_prediction_sequence(
            root=args.root,
            seed=args.seed,
            b26_training_episodes=args.b26_training_episodes,
            b26_workers=args.b26_workers,
            b26_search=args.b26_search,
            b26_ga_population=args.b26_ga_population,
            b26_ga_generations=args.b26_ga_generations,
            b26_finalists=args.b26_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 25:
        summary = run_b25_metacognitive_confidence_sequence(
            root=args.root,
            seed=args.seed,
            b25_training_episodes=args.b25_training_episodes,
            b25_workers=args.b25_workers,
            b25_search=args.b25_search,
            b25_ga_population=args.b25_ga_population,
            b25_ga_generations=args.b25_ga_generations,
            b25_finalists=args.b25_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 24:
        summary = run_b24_precision_conflict_sequence(
            root=args.root,
            seed=args.seed,
            b24_training_episodes=args.b24_training_episodes,
            b24_workers=args.b24_workers,
            b24_search=args.b24_search,
            b24_ga_population=args.b24_ga_population,
            b24_ga_generations=args.b24_ga_generations,
            b24_finalists=args.b24_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 23:
        summary = run_b23_conflict_monitor_sequence(
            root=args.root,
            seed=args.seed,
            b23_training_episodes=args.b23_training_episodes,
            b23_workers=args.b23_workers,
            b23_search=args.b23_search,
            b23_ga_population=args.b23_ga_population,
            b23_ga_generations=args.b23_ga_generations,
            b23_finalists=args.b23_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 22:
        summary = run_b22_prospective_replay_sequence(
            root=args.root,
            seed=args.seed,
            b22_training_episodes=args.b22_training_episodes,
            b22_workers=args.b22_workers,
            b22_search=args.b22_search,
            b22_ga_population=args.b22_ga_population,
            b22_ga_generations=args.b22_ga_generations,
            b22_finalists=args.b22_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 21:
        summary = run_b21_hippocampal_replay_sequence(
            root=args.root,
            seed=args.seed,
            b21_training_episodes=args.b21_training_episodes,
            b21_workers=args.b21_workers,
            b21_search=args.b21_search,
            b21_ga_population=args.b21_ga_population,
            b21_ga_generations=args.b21_ga_generations,
            b21_finalists=args.b21_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 20:
        summary = run_b20_working_memory_gate_sequence(
            root=args.root,
            seed=args.seed,
            b20_training_episodes=args.b20_training_episodes,
            b20_workers=args.b20_workers,
            b20_search=args.b20_search,
            b20_ga_population=args.b20_ga_population,
            b20_ga_generations=args.b20_ga_generations,
            b20_finalists=args.b20_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 19:
        summary = run_b19_episodic_meta_memory_sequence(
            root=args.root,
            seed=args.seed,
            b19_training_episodes=args.b19_training_episodes,
            b19_workers=args.b19_workers,
            b19_search=args.b19_search,
            b19_ga_population=args.b19_ga_population,
            b19_ga_generations=args.b19_ga_generations,
            b19_finalists=args.b19_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 18:
        summary = run_b18_eligibility_trace_sequence(
            root=args.root,
            seed=args.seed,
            b18_training_episodes=args.b18_training_episodes,
            b18_workers=args.b18_workers,
            b18_search=args.b18_search,
            b18_ga_population=args.b18_ga_population,
            b18_ga_generations=args.b18_ga_generations,
            b18_finalists=args.b18_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 17:
        summary = run_b17_neuromodulated_ensemble_sequence(
            root=args.root,
            seed=args.seed,
            b17_training_episodes=args.b17_training_episodes,
            b17_workers=args.b17_workers,
            b17_search=args.b17_search,
            b17_ga_population=args.b17_ga_population,
            b17_ga_generations=args.b17_ga_generations,
            b17_finalists=args.b17_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 16:
        summary = run_b16_option_ensemble_sequence(
            root=args.root,
            seed=args.seed,
            b16_training_episodes=args.b16_training_episodes,
            b16_workers=args.b16_workers,
            b16_search=args.b16_search,
            b16_ga_population=args.b16_ga_population,
            b16_ga_generations=args.b16_ga_generations,
            b16_finalists=args.b16_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 15:
        summary = run_b15_option_critic_sequence(
            root=args.root,
            seed=args.seed,
            b15_training_episodes=args.b15_training_episodes,
            b15_workers=args.b15_workers,
            b15_search=args.b15_search,
            b15_ga_population=args.b15_ga_population,
            b15_ga_generations=args.b15_ga_generations,
            b15_finalists=args.b15_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 14:
        summary = run_b14_affordance_uncertainty_sequence(
            root=args.root,
            seed=args.seed,
            b14_training_episodes=args.b14_training_episodes,
            b14_workers=args.b14_workers,
            b14_search=args.b14_search,
            b14_ga_population=args.b14_ga_population,
            b14_ga_generations=args.b14_ga_generations,
            b14_finalists=args.b14_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 13:
        summary = run_b13_local_affordance_search_sequence(
            root=args.root,
            seed=args.seed,
            b13_training_episodes=args.b13_training_episodes,
            b13_workers=args.b13_workers,
            b13_search=args.b13_search,
            b13_ga_population=args.b13_ga_population,
            b13_ga_generations=args.b13_ga_generations,
            b13_finalists=args.b13_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 12:
        summary = run_b12_predictive_attention_sequence(
            root=args.root,
            seed=args.seed,
            b12_training_episodes=args.b12_training_episodes,
            b12_workers=args.b12_workers,
            b12_search=args.b12_search,
            b12_ga_population=args.b12_ga_population,
            b12_ga_generations=args.b12_ga_generations,
            b12_finalists=args.b12_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 11:
        summary = run_b11_confidence_arbiter_sequence(
            root=args.root,
            seed=args.seed,
            b11_training_episodes=args.b11_training_episodes,
            b11_workers=args.b11_workers,
            b11_search=args.b11_search,
            b11_ga_population=args.b11_ga_population,
            b11_ga_generations=args.b11_ga_generations,
            b11_finalists=args.b11_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 10:
        summary = run_b10_prospective_replay_sequence(
            root=args.root,
            seed=args.seed,
            b10_training_episodes=args.b10_training_episodes,
            b10_workers=args.b10_workers,
            b10_search=args.b10_search,
            b10_ga_population=args.b10_ga_population,
            b10_ga_generations=args.b10_ga_generations,
            b10_finalists=args.b10_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 9:
        summary = run_b9_waypoint_planner_sequence(
            root=args.root,
            seed=args.seed,
            b9_training_episodes=args.b9_training_episodes,
            b9_workers=args.b9_workers,
            b9_search=args.b9_search,
            b9_ga_population=args.b9_ga_population,
            b9_ga_generations=args.b9_ga_generations,
            b9_finalists=args.b9_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 8:
        summary = run_b8_spatial_affordance_sequence(
            root=args.root,
            seed=args.seed,
            b8_training_episodes=args.b8_training_episodes,
            b8_workers=args.b8_workers,
            b8_search=args.b8_search,
            b8_ga_population=args.b8_ga_population,
            b8_ga_generations=args.b8_ga_generations,
            b8_finalists=args.b8_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 7:
        summary = run_b7_affordance_budget_sequence(
            root=args.root,
            seed=args.seed,
            b7_training_episodes=args.b7_training_episodes,
            b7_workers=args.b7_workers,
            b7_search=args.b7_search,
            b7_ga_population=args.b7_ga_population,
            b7_ga_generations=args.b7_ga_generations,
            b7_finalists=args.b7_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 6:
        summary = run_b6_risk_corridor_sequence(
            root=args.root,
            seed=args.seed,
            b6_training_episodes=args.b6_training_episodes,
            b6_workers=args.b6_workers,
            b6_search=args.b6_search,
            b6_ga_population=args.b6_ga_population,
            b6_ga_generations=args.b6_ga_generations,
            b6_finalists=args.b6_finalists,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 5:
        summary = run_b5_homeostatic_arbiter_sequence(
            root=args.root,
            seed=args.seed,
            b5_training_episodes=args.b5_training_episodes,
            b5_workers=args.b5_workers,
            b5_search=args.b5_search,
            b5_ga_population=args.b5_ga_population,
            b5_ga_generations=args.b5_ga_generations,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 4:
        summary = run_b4_recovery_balance_sequence(
            root=args.root,
            seed=args.seed,
            b4_training_episodes=args.b4_training_episodes,
            b4_workers=args.b4_workers,
            b4_search=args.b4_search,
            b4_ga_population=args.b4_ga_population,
            b4_ga_generations=args.b4_ga_generations,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 3:
        summary = run_b3_contact_memory_sequence(
            root=args.root,
            seed=args.seed,
            b3_training_episodes=args.b3_training_episodes,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    elif int(args.target_level) == 2:
        summary = run_b2_temporal_threat_sequence(
            root=args.root,
            seed=args.seed,
            b2_training_episodes=args.b2_training_episodes,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
        )
    else:
        summary = run_b1_capacity_sequence(
            root=args.root,
            seed=args.seed,
            b0_training_episodes=args.b0_training_episodes,
            b1_training_episodes=args.b1_training_episodes,
            reward_profile=args.reward_profile,
            operational_profile=args.operational_profile,
            noise_profile=args.noise_profile,
            force_b0=bool(args.force_b0),
            run_canonical=not bool(args.skip_canonical),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
