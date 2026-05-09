# Ablation Workflow

This repository exposes a reproducible ablation suite for comparing the current modular architecture against structural variants, including both the direct A0 monolithic baseline and the existing A1 monolithic-with-pipeline baseline. No-reflex performance is the primary benchmark: comparisons use `scenario_success_rate` from rows or payloads evaluated with `eval_reflex_scale=0.0`, while reflex-on results are retained as diagnostics for scaffold dependence. For the official progression protocol that asks when to move from monolithic to progressively modular architectures, see `docs/architectural_ladder.md`.

Read the ladder and this workflow together:

- `docs/architectural_ladder.md` is the diagnostic framework for interpreting where an ablation regression likely comes from as modularity increases
- `docs/ablation_workflow.md` is the detailed execution and reporting guide for canonical variants, competence gap, shaping gap, and benchmark-of-record procedures

## Prerequisites

Python 3.10+ is required for the simulator CLI. Command examples use
`python3.10` explicitly; if your supported interpreter is newer, substitute its
versioned command, such as `python3.11`.

## Canonical Variants

| Variant | Purpose |
| --- | --- |
| `modular_full` | Full modular architecture with `module_dropout`, local reflexes, auxiliary targets, learned arbitration, and broadcast proposal credit. |
| `no_module_dropout` | Keeps the modular architecture but removes inter-module dropout. |
| `no_module_reflexes` | Keeps the modules but disables both reflex-logit injection and the auxiliary targets derived from reflexes. |
| `three_center_modular` | A2 ladder rung: three coarse proposer centers with broadcast proposal credit. |
| `three_center_modular_local_credit` | A2 ladder rung with the same three-center topology, but proposer modules receive only local credit. |
| `three_center_modular_counterfactual` | A2 ladder rung with the same three-center topology, but shared proposal credit is weighted by leave-one-out counterfactual importance. |
| `modular_recurrent` | Keeps the modular architecture and reflex pipeline, but swaps `alert_center`, `sleep_center`, and `hunger_center` to recurrent proposers with per-episode hidden state. |
| `modular_recurrent_all` | Makes all five proposer modules recurrent, so every proposal stream can accumulate within-episode context. |
| `local_credit_only` | Preserves the current modular inference path but removes the broadcast of global policy gradients across proposer modules during training. |
| `counterfactual_credit` | Preserves the current modular inference path but replaces uniform broadcast credit with leave-one-out counterfactual proposal credit. |
| `fixed_arbitration_baseline` | Keeps the current modules but disables learned arbitration and uses the fixed valence formulas and priority gates only. |
| `learned_arbitration_no_regularization` | Keeps learned arbitration active but sets arbitration gate and valence regularization to zero. |
| `reflex_scale_0_25`, `reflex_scale_0_50`, `reflex_scale_0_75` | Keep the modular architecture but progressively reduce global reflex strength. |
| `drop_visual_cortex`, `drop_sensory_cortex`, `drop_hunger_center`, `drop_sleep_center`, `drop_alert_center` | Remove one proposer module at a time from locomotion proposal and learning. |
| `monolithic_policy` | Concatenates the modular observations into a single vector and runs a single proposer before the same `action_center` and `motor_cortex` used by the modular architecture. |
| `true_monolithic_policy` | The simplest baseline: a single network maps concatenated observations directly to action logits plus a value head, with no downstream pipeline. |
| `behavior_tree_oracle_policy` | Diagnostic Stage-2 solvability oracle: an evaluation-only behavior-tree controller with explicit ecological phases and shortest-path local planning. It is not trainable and must never be counted as final acceptance evidence; it only answers whether the canonical ecology is solvable by explicit control under the current rules. |
| `true_monolithic_recurrent_policy` | Diagnostic A0-R variant: same observation vector and primitive action space as `true_monolithic_policy`, but with a runtime-only recurrent hidden state that resets between episodes. |
| `true_monolithic_recurrent_phase_policy` | Diagnostic A0-RP variant: recurrent direct policy plus an auxiliary phase-classification head over derived ecological phases such as `RESTING`, `RECOVERED_IN_SHELTER`, and `POST_REST_REACTIVATE`; the phase head is supervised for diagnostics only and does not script actions. |
| `true_monolithic_event_attention_policy` | Diagnostic A0-RE variant: recurrent direct policy plus a small within-episode event buffer and learned attention over discrete events such as `REST_STARTED`, `RECOVERY_COMPLETED`, `SHELTER_RETURN`, `POST_REST_RELEASE_ATTEMPT`, and `BLOCKED_MOVE`. |
| `true_monolithic_option_policy` | Diagnostic A0-OPT variant: recurrent direct policy plus event attention and a small learned option-commitment head over macro-intents such as `FORAGE`, `RETURN_TO_SHELTER`, `DEEPEN_IN_SHELTER`, `REST`, `POST_REST_REACTIVATE`, and `ESCAPE`. |
| `true_monolithic_option_affordance_policy` | Diagnostic A0-AFF variant: the same option-conditioned controller plus auxiliary local-affordance heads that predict per-action blockedness and resulting shelter-role transition on the current map geometry, without feeding any new observation channels into the policy yet. |
| `true_monolithic_option_affordance_feedback_policy` | Diagnostic A0-AFF-FB variant: the same option-conditioned controller with auxiliary local-affordance heads, plus an internal feedback path that lets predicted blockedness and shelter-role transition shape option and policy logits without adding new observation channels. |
| `true_monolithic_option_affordance_geometry_policy` | Diagnostic A0-AFF-GEO variant: the same affordance-feedback controller plus decomposed per-action shelter-geometry heads for `deepen_shelter`, `toward_entrance`, and `toward_outside`, still without adding new external observation channels. |
| `true_monolithic_option_affordance_topology_policy` | Diagnostic A0-AFF-TOP variant: the same affordance-feedback controller plus an auxiliary per-action shelter-column topology head over `outside`, `left`, `center`, and `right`, so sheltered continuation can be decomposed into role plus local column without adding new observation channels. |
| `true_monolithic_option_affordance_position_policy` | Diagnostic A0-AFF-POS variant: the same affordance-feedback controller plus an auxiliary joint per-action shelter-position head over exact local shelter classes such as `inside_center` and `entrance_right`, replacing separate local factors with direct sheltered-position supervision while keeping the policy observation surface fixed. |
| `true_monolithic_option_affordance_position_teacher_policy` | Diagnostic A0-TCH variant: the same position-head controller plus a sparse training-only handoff teacher loss derived from recovered-shelter probe logic. The teacher provides `MOVE_UP -> STAY -> MOVE_DOWN` supervision at narrow recovered-shelter moments, but runtime behavior still comes entirely from the learned policy. |
| `true_monolithic_option_affordance_position_teacher_option_policy` | Diagnostic A0-TCH-OPT variant: the same position-head controller plus the sparse handoff teacher and an additional training-only option-head teacher over recovered-shelter reactivation windows. Runtime behavior still comes entirely from the learned policy; the extra supervision only biases the option logits during training. |
| `true_monolithic_executive_option_policy` | Diagnostic Stage-4 executive bundle: no food-direction helper, persistent phase/option state, explicit phase-to-option and previous-option feedback, option cooldown, option decoder/dynamics state, and a separate option-conditioned primitive-action backbone. It is the smallest current direct-policy bundle that tries to move action ownership away from helper-shaped logits without adding route-script runtime behavior. |
| `true_monolithic_executive_option_guarded_policy` | Diagnostic Stage-4 executive-guarded bundle: the same executive controller as `true_monolithic_executive_option_policy`, but with threat-escape and sleep/rest guards still enabled while food-direction shaping is explicitly suppressed. It tests whether helpers can remain as safety constraints without retaking forage ownership. |
| `true_monolithic_executive_option_physiology_gated_policy` | Diagnostic Stage-4 executive physiology-gated bundle: the same no-food-bias executive controller as `true_monolithic_executive_option_policy`, but with an internal option-selection gate over existing sleep, threat, and shelter signals so rest/return/reactivation pressure is explicit inside the controller rather than injected as runtime action shaping. |
| `true_monolithic_executive_option_release_policy` | Diagnostic Stage-4 executive release bundle: the physiology-gated executive controller plus an internal option-conditioned action gate that uses the existing affordance/geometry/shelter-position heads to turn `REST` into real `STAY`, and `POST_REST_REACTIVATE` into an explicit local release action away from shelter without restoring runtime food-direction shaping. |
| `true_monolithic_executive_option_masked_release_policy` | Diagnostic Stage-4 executive masked-release bundle: the release controller plus an internal option-conditioned locomotion mask, so movement and rest options can suppress orientation leakage and force the learned controller to choose among locally valid locomotion actions or `STAY` rather than drifting into orientation-only basins. |
| `true_monolithic_executive_option_event_release_policy` | Diagnostic Stage-4 executive event-release bundle: the masked-release controller plus a bounded event-driven release latch, so real termination events such as `DEEP_SLEEP_REACHED`, `RECOVERY_COMPLETED`, or sheltered `BLOCKED_MOVE` can force a short `POST_REST_REACTIVATE` handoff instead of letting the controller fall back into another sheltered static-gating basin. |
| `true_monolithic_executive_option_transition_release_policy` | Diagnostic Stage-4 executive transition-release bundle: the event-release controller plus explicit action-level release commitment during that bounded latch, so the controller owns the actual sheltered-to-outside locomotion transition instead of only relabeling the option after a termination event. |
| `true_monolithic_executive_option_release_phase_policy` | Diagnostic Stage-4 executive release-phase bundle: the transition-release controller plus an explicit startup/post-rest release phase state that can arm the sheltered-to-outside transition at episode start and after recovery, so the controller owns the release phase itself rather than only reacting to a later termination event. |
| `true_monolithic_executive_option_release_progression_policy` | Diagnostic Stage-4 executive release-progression bundle: the release-phase controller plus explicit entrance-to-outside progression inside that release phase, so the controller keeps ownership after reaching shelter entrance instead of dropping into orientation or falling back into shelter. |
| `true_monolithic_executive_option_release_exit_contract_policy` | Diagnostic Stage-4 executive release-exit-contract bundle: the release-phase controller plus an explicit map-faithful local exit contract for `central_burrow`, so sheltered release prefers the real exit actions (`MOVE_UP` from deep/inside, `MOVE_LEFT` from entrance) instead of relying on the failed indirect entrance proxy. |
| `true_monolithic_executive_option_release_substate_policy` | Diagnostic Stage-4 executive release-substate bundle: the release-phase controller plus the map-faithful `central_burrow` exit contract and a controller-owned sheltered progression substate that preserves release ownership through `deep -> inside -> entrance -> outside` instead of expiring before the actual exit step. |
| `true_monolithic_executive_option_post_exit_continuation_policy` | Diagnostic Stage-4 executive post-exit-continuation bundle: the release-substate controller plus bounded post-exit continuation, so `POST_REST_REACTIVATE` is not terminated immediately on `SHELTER_EXIT` and the first outside steps can be owned before any return/forage handoff. |
| `true_monolithic_executive_option_post_exit_food_guidance_policy` | Diagnostic Stage-4 executive outside-food-guidance bundle: the post-exit-continuation controller plus bounded use of existing food direction signals during the first outside continuation window, so the executive can choose a productive first outside step instead of the generic `MOVE_DOWN / STAY` collapse. |
| `true_monolithic_executive_option_post_exit_food_commitment_policy` | Diagnostic Stage-4 executive outside-food-commitment bundle: the post-exit-food-guidance controller plus a longer and refreshable outside commitment window while live food guidance exists, so the executive can attempt a real second-cycle forage instead of timing out back into `MOVE_DOWN`. |
| `true_monolithic_executive_option_post_exit_food_progression_policy` | Diagnostic Stage-4 executive outside-food-progression bundle: the post-exit-food-guidance controller plus a narrow anti-reversal progression rule for stale-memory outside states, so the executive does not immediately undo its first useful sidestep and can climb toward the nearest food instead of falling back into the old corridor loop. |
| `true_monolithic_executive_option_post_exit_food_heading_progression_policy` | Diagnostic Stage-4 executive outside-food-heading-progression bundle: the post-exit-food-guidance controller plus a narrow heading-aware climb rule, so after a rightward recentering step the executive can take the productive `MOVE_UP` immediately instead of spending the next tick on `ORIENT_UP` and timing out back into shelter. |
| `true_monolithic_executive_option_post_exit_smell_progression_policy` | Diagnostic Stage-4 executive outside-smell-progression bundle: the post-exit-food-guidance controller plus a three-step outside continuation window and a smell-guided `MOVE_RIGHT -> MOVE_UP` climb rule, so the controller can convert the live early outside smell-follow family into a bounded upward probe without relying on fresh memory. |
| `true_monolithic_executive_option_post_exit_corridor_progression_policy` | Diagnostic Stage-4 executive blind-corridor-progression bundle: the post-exit continuation controller plus a bounded seven-step outside contract that owns the first blind canonical corridor leg when there is no live food cue, favoring rightward advancement first and then a downward drop into the food row. |
| `true_monolithic_executive_option_post_exit_corridor_affordance_progression_policy` | Diagnostic Stage-4 executive blind-corridor-affordance-progression bundle: the blind corridor branch plus a local affordance rule that keeps moving right while a downward move would re-enter shelter or hit a wall, then only drops once the downward path stays outside, so the controller can reach the outer food row without immediate shelter re-entry. |
| `true_monolithic_executive_option_post_food_return_policy` | Diagnostic Stage-4 executive post-food-return bundle: the corridor-affordance branch plus an explicit post-meal return contract that forces return-to-shelter / deepen / stay ownership after `food_reached`, testing whether missing post-meal return ownership is the remaining executive gap. |
| `true_monolithic_executive_option_post_food_vector_return_policy` | Diagnostic Stage-4 executive post-food-vector-return bundle: the post-food-return branch plus an explicit shelter-memory-vector action owner after `food_reached`, so once a meal is earned the controller returns by auditable shelterward geometry instead of relying on the learned shelter-position head. |
| `true_monolithic_executive_option_post_food_path_return_policy` | Diagnostic Stage-4 executive post-food-path-return bundle: the post-food-return branch plus a controller-owned outbound path memory that replays inverse locomotion after `food_reached`, testing whether explicit executive return memory can convert a first meal into a safe shelter return without reopening observation shape. |
| `true_monolithic_option_affordance_position_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY variant: the same teacher-option controller plus continuation-only replay-gradient passes inside `learn()`, so the exact failed post-rest and late-cycle targets can be re-optimized several times with fresh forward passes without changing the official evaluation environment or adding new observation channels. |
| `true_monolithic_option_affordance_position_phase_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE variant: the same replay-gradient branch plus a composed phase head on the event-attention / option / affordance / shelter-position controller, so live continuation replay can optimize both option and phase targets together instead of relying on a sparse teacher surface alone. |
| `true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-OFB variant: the same phase-composed replay branch plus a direct phase-to-option feedback path, so runtime option choice can consume the predicted phase distribution instead of treating phase as auxiliary-only state. |
| `true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-OTR variant: the same phase-composed replay branch plus an explicit option-transition path so the next option logits can depend directly on the previous option instead of only through recurrent hidden state. |
| `true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-OCD variant: the same phase-composed replay branch plus a small termination controller that blocks immediate reselection of a just-completed option on successful handoff events like `REST -> recovery_completed`, without expanding the observation interface. |
| `true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-OAH variant: the same phase-composed replay branch plus a learned option-conditioned action head, so each selected option can contribute its own full action-logit slice instead of only a static bias vector. |
| `true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS variant: the same phase-composed replay branch plus an option-specific decoder state, so the selected option can reshape the hidden state seen by the action decoder rather than only adding biases in logit space. |
| `true_monolithic_option_affordance_position_phase_option_decoder_feedback_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-OFB variant: the decoder-state branch plus direct phase-to-option feedback, testing whether a now-stronger official-path phase representation can finally steer option choice when routed straight into the option head. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD variant: the decoder-state branch plus option-specific recurrent hidden dynamics, so the previous option can directly reshape the next hidden transition instead of only affecting logits or decoder projections. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-OAH variant: the option-dynamics branch plus the per-option action head, testing whether stronger option-conditioned state dynamics and stronger option-conditioned action decoding together can move executed behavior. |
| `true_monolithic_option_affordance_position_phase_teacher_option_margin_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-MARGIN variant: the same phase-composed replay branch plus a continuation-only margin objective that explicitly pushes the teacher action above `STAY`, the teacher option above `RETURN_TO_SHELTER`, and the target phase above `INITIAL_FORAGE` on continuation steps. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_affordance_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-LAF-PRB-RPLY-DST variant: the replayable-teacher detached-action-backbone branch plus a small explicit local-affordance input append, giving the direct controller current shelter-role identity and per-action blocked / next-role cues without yet adding a broader spatial observation interface. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_spatial_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-LSP-PRB-RPLY-DST variant: the same replayable-teacher detached-action-backbone branch plus a 3x3 local spatial patch append carrying blockedness, shelter-role level, and food occupancy around the spider. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-LTR-PRB-RPLY-DST variant: the same replayable-teacher detached-action-backbone branch plus per-action one-step transition consequences, exposing food-distance, shelter-distance, predator-distance deltas, and next-cell food occupancy for each primitive move. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_rollout_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-LTR2-PRB-RPLY-DST variant: the same replayable-teacher detached-action-backbone branch plus per-action two-step rollout consequences, exposing best reachable food/shelter/predator deltas and whether food is reachable within two steps. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_phase_option_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-GEO-OFB-PRB-RPLY-DST variant: the same replayable-teacher detached-action-backbone branch plus bounded local geodesic inputs and a direct phase-to-option feedback path, so the predicted phase distribution can steer option logits on the current frontier without adding new global observation channels. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_option_sequence_head_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-GEO-OSH-PRB-RPLY-DST variant: the same replayable-teacher detached-action-backbone branch plus bounded local geodesic inputs and an explicit per-option, per-age action-sequence head, so within-option execution can depend directly on option identity and option age on the current frontier. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_post_rest_probe_trace_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-GEO-PRB-TRACE-DST variant: the same detached-action-backbone geodesic frontier plus the trace-backed post-rest teacher source, but with replayable/frontier/rollout teacher paths disabled so the stronger trace distillation source is the one actually used during continuation curriculum updates. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_action_token_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-GEO-OATK-PRB-RPLY-DST variant: the same replayable-teacher detached-action geodesic frontier plus a persistent executed-action token decoder, so action execution can use its own token-state core on top of the current local-context and post-rest teacher stack. |
| `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_geodesic_inputs_option_transition_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy` | Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-GEO-OTR-PRB-RPLY-DST variant: the same replayable-teacher detached-action-backbone branch plus bounded local geodesic inputs and an explicit option-to-option transition feedback path, so the selected option on the current frontier can depend directly on the previous option without adding new global observation channels. |

Canonical summaries now include `architecture_description` so the comparison output distinguishes:

- `true monolithic direct control`
- `monolithic proposer + action/motor pipeline`
- `full modular with arbitration`

For `true_monolithic_recurrent_phase_policy`, inspect trace-level `debug.phase.target`, `debug.phase.prediction`, `debug.phase.prediction_confidence`, and the surrounding selected-action sequence at the first recovered daytime shelter window. A correct recovered-phase prediction without a second-cycle exit is evidence that representation improved but action credit is still blocking the behavior.

For `true_monolithic_event_attention_policy`, inspect `event_attention_top_type`, `event_attention_top_age`, and `event_attention_entropy` in the `true_monolithic_policy` action-selection trace payload. If canonical post-rest ticks briefly attend to `RECOVERY_COMPLETED` or `POST_REST_RELEASE_ATTEMPT` but the aggregate row stays at the same `10.0`-day / `3`-food / `1`-exit family, the memory surface is present but still not enough to produce a durable second-cycle option.

For `true_monolithic_option_policy`, inspect `selected_option`, `option_age`, `option_termination_reason`, and `option_logits` in the `true_monolithic_policy` action-selection trace payload. If canonical collapses into a long-lived `FORAGE` option that reaches outside/corridor and reacquires food but never returns to rest, treat that as evidence that macro-intent commitment exists but local return/rest affordance is still missing.

For `true_monolithic_option_affordance_policy`, inspect `affordance_blocked_logits` and `affordance_role_logits` alongside the option trace. A useful diagnostic is whether the head correctly scores the recovered in-shelter post-rest window: if the branch restores the healthy one-exit baseline but still marks the critical sheltered `MOVE_RIGHT` continuation as locally blocked or predicts the wrong shelter-role transition, the local geometry signal is still incomplete even with auxiliary supervision.

For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_affordance_post_rest_probe_replayable_teacher_distill_option_replay_policy`, inspect whether the new local-affordance input actually changes executed actions, not just hidden supervision. If canonical still stays `STAY 286 / MOVE_RIGHT 7 / MOVE_LEFT 6 / MOVE_UP 1` with `RETURN_TO_SHELTER 300` and no post-sleep exits, treat that as evidence that a small scalar action-validity / shelter-transition append is not enough. The next branch should move to a richer local spatial or action-consequence input surface rather than more teacher-source reshaping.

For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_spatial_post_rest_probe_replayable_teacher_distill_option_replay_policy`, inspect whether the broader 3x3 patch append changes the official action family at all. If canonical still stays `STAY 286 / MOVE_RIGHT 7 / MOVE_LEFT 6 / MOVE_UP 1` with `RETURN_TO_SHELTER 300`, phases absent in trace, and no post-sleep exits, treat that as evidence that static local spatial state is still too weak. The next branch should move to action-consequence or transition-prediction inputs rather than another purely spatial append.

For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_post_rest_probe_replayable_teacher_distill_option_replay_policy`, inspect whether the one-step transition deltas change executed behavior or only add redundant local hints. If canonical still stays `STAY 286 / MOVE_RIGHT 7 / MOVE_LEFT 6 / MOVE_UP 1` with `RETURN_TO_SHELTER 300`, phases absent in trace, and no post-sleep exits, treat that as evidence that one-step local consequences are still too weak. The next branch should move to richer multi-step action-consequence or learned transition-prediction inputs rather than another small local append.

For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_local_transition_rollout_post_rest_probe_replayable_teacher_distill_option_replay_policy`, inspect whether the two-step rollout hints change the official action family at all. If canonical still stays `STAY 286 / MOVE_RIGHT 7 / MOVE_LEFT 6 / MOVE_UP 1` with `RETURN_TO_SHELTER 300`, phases absent in trace, and no post-sleep exits, treat that as evidence that static local rollout hints are still too weak. The next branch should move to learned transition-prediction or another richer dynamic consequence surface rather than more static appended hints.

For `true_monolithic_option_affordance_feedback_policy`, compare the same affordance diagnostics against the selected option/action trace. If the auxiliary-only branch learned some local structure but remained behaviorally flat, this feedback branch answers the next causal question: can the controller use its own predicted affordance surface once that signal is fed back into option and action selection, or does the affordance representation itself remain too inaccurate?

For `true_monolithic_option_affordance_geometry_policy`, inspect `geometry_logits` together with `affordance_role_logits` in the `true_monolithic_policy` action trace. The critical read is whether recovered inside-shelter post-rest moves stop being mislabeled as transitions toward `outside` or `entrance`; if canonical still sits on the `10.0`-day / `3`-food / `1`-exit row while `MOVE_RIGHT` from inside shelter is predicted as `outside`-leaning, richer geometry supervision is still not enough and the next step should remain inside Stage 5.

For `true_monolithic_option_affordance_topology_policy`, inspect `shelter_column_logits` together with `affordance_role_logits`. The intended diagnostic is whether the recovered inside-shelter post-rest slice stops collapsing distinct cells such as `inside_center` and `inside_right`; if canonical remains on the same healthy one-exit aggregate row while `MOVE_RIGHT` from recovered `(6, 6)` still predicts `center` rather than `right`, the controller still lacks the exact local shelter topology needed for a second cycle.

For `true_monolithic_option_affordance_position_policy`, inspect `shelter_position_logits` directly. This is the Stage-5 saturation check: if even joint exact-position supervision leaves canonical flat at the `10.0`-day / `3`-food / `1`-exit row and the recovered inside-shelter slice still assigns low-confidence or wrong classes such as `entrance_right` or `deep_center` to the critical continuation moves, pure shelter-geometry auxiliaries are no longer the best next branch and the roadmap should move to Stage 6 teacher/distillation.

For `true_monolithic_option_affordance_position_teacher_policy`, do not read success from the presence of teacher metadata alone. The branch is only useful if the official runner changes post-rest behavior under the same learned runtime policy. The key checks are:
- canonical aggregate behavior must improve beyond the one-exit `10.0`-day / `3`-food baseline, or at least show a real post-rest second-cycle exit
- the first `POST_REST_REACTIVATE` selections must occur after rest, not just at initial warm-start ticks
- if canonical remains flat and only pre-rest ticks select `POST_REST_REACTIVATE`, the narrow teacher labels were not enough to internalize the handoff and the next step should likely move to a richer teacher or distillation surface rather than more pure geometry supervision

For `true_monolithic_option_affordance_position_teacher_option_policy`, keep the same bar and add one more diagnostic question: did broader macro-intent supervision actually change official behavior, or did it only change hidden option bias during training? If canonical and easy remain on the same top-level action family and the official rows stay at `10.0` days / `1` exit and `6.0` days / `2` exits respectively, the option teacher alone is still insufficient and the next step should move to Stage-7 continuation curriculum rather than more option-head loss shaping.

For `true_monolithic_option_affordance_position_teacher_option_replay_policy`, use the same official canonical/easy bar and add one training-side audit: did the new replay-gradient passes actually fire on continuation steps? Check `continuation_replay_passes_configured`, `continuation_replay_passes_applied`, `continuation_replay_lr_scale`, `continuation_replay_total_loss`, and `continuation_replay_total_grad_norm` in learning stats. If those are clearly active but canonical and easy still stay on the same top-level action family, then the failure is no longer explainable as “the continuation targets were seen too rarely or optimized too weakly,” and the next branch should move toward a different supervision surface or controller objective rather than more scheduler tuning.

For `true_monolithic_option_affordance_position_phase_teacher_option_replay_policy`, keep the same official canonical/easy bar, but tighten the interpretation: if replay is now clearly active in real training and official evaluation still stays on `INITIAL_FORAGE` / `RETURN_TO_SHELTER` with the unchanged top-level action family, the remaining blocker is no longer “phase replay was architecturally inert.” At that point the next step should move away from plumbing and toward stronger continuation-specific objectives or controller changes.

For `true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy`, ask the narrower controller question: if direct phase-to-option routing flips canonical phase prediction into `POST_REST_REACTIVATE` while option choice still stays on `RETURN_TO_SHELTER` and the official row remains at the same `10.0`-day / `1`-exit family, then phase recognition is no longer the missing control link. The next branch should focus on option commitment or transition structure rather than stronger phase supervision.

For `true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy`, the key read is whether explicit option-to-option transition structure changes official option usage at all. If the post-rest training probe shows diversified option usage but official canonical and easy still stay on the same `RETURN_TO_SHELTER`-dominated traces and the unchanged aggregate rows, then explicit transition logits alone are still insufficient and the next branch should move to stronger commitment or termination control.

For `true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy`, check whether the reselection cooldown changes only latent option identity or actually changes executed actions. If canonical shifts from `REST`-dominated option usage into `RETURN_TO_SHELTER` / `FORAGE` mixes but the top-level action counts still stay on the same `STAY 286 / MOVE_RIGHT 7 / MOVE_LEFT 6 / MOVE_UP 1` family, then the bottleneck has moved downstream of option choice and the next branch should target option-conditioned action control rather than more option-selection logic.

For `true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy`, the key read is whether an explicit per-option action head changes executed actions rather than only controller internals. If official easy and canonical still stay exactly on the old action family while both traces collapse to `RETURN_TO_SHELTER` on every tick, then even stronger downstream option-conditioned action logits are still not enough. The next branch should move beyond additive option-action heads toward a more invasive action-execution controller or option-specific decoder state.

For `true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy`, the key read is whether option-specific decoder state helps the controller internalize post-rest phase on the official path even if behavior is still flat. If canonical and easy phase predictions flip strongly toward `POST_REST_REACTIVATE` while selected option stays pinned to `RETURN_TO_SHELTER` and top-level actions remain unchanged, then phase representation is no longer the main missing piece. The next branch should test whether direct phase-to-option routing can exploit that signal, or otherwise move on to a stronger option-conditioned control backbone.

For `true_monolithic_option_affordance_position_phase_option_decoder_feedback_teacher_option_replay_policy`, the key read is whether reintroducing direct phase-to-option routing on top of the stronger decoder-state phase signal actually changes option choice. If the branch collapses back to `INITIAL_FORAGE` or `LATE_FORAGE` predictions while option/action usage stays unchanged, then the bottleneck is not simply "phase was represented but never routed." The next branch should move away from more phase-routing tweaks and toward option-specific recurrent/action dynamics.

For `true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy`, the key read is whether option-specific recurrent dynamics can move official option identity even if actions stay flat. If canonical flips from `RETURN_TO_SHELTER` into another stable option such as `ESCAPE` while executed actions remain on the same `STAY 286 / MOVE_RIGHT 7 / MOVE_LEFT 6 / MOVE_UP 1` family, then option choice is no longer the bottleneck. The next branch should focus squarely on action realization under option control.

For `true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy`, the key read is whether combining option-specific recurrent dynamics with the per-option action head changes executed behavior. If official easy and canonical still stay exactly on the old action families and option choice collapses back to `RETURN_TO_SHELTER`, then even this stronger composed option-control stack is still insufficient and the next intervention should move toward a more explicit action-sequence controller rather than more hidden/logit composition.

For `true_monolithic_option_affordance_position_phase_teacher_option_margin_replay_policy`, the key read is whether explicitly pushing against the collapsed continuation competitors helps or destabilizes the controller. If easy stays on the old `6.0`-day / `2`-exit family but canonical regresses into a short no-sleep loop with many shelter exits, repeated `RETURN_TO_SHELTER` selection, and early `POST_REST_REACTIVATE` or late-return phase bias, treat that as evidence that stronger continuation pressure alone is not the missing ingredient. The next branch should then move away from "more continuation loss on the same controller" and toward a different controller intervention.

Stage-7 continuation curricula are training-only aids, not new benchmark rows. The current continuation profile is `a0_post_rest_continuation_v1`, which mixes:
- `continuous_survival_post_rest_inside_v1`
- `continuous_survival_post_rest_entrance_v1`

These scenarios are allowed only as curriculum exposures. Final evaluation remains `continuous_survival_canonical`, and any success claim still has to survive there under the official runner.

The continuation-curriculum family now has three increasingly strong exposure profiles:
- `a0_post_rest_continuation_v1`: recovered post-rest handoff exposure only
- `a0_post_rest_continuation_v2`: adds late-cycle return/rest scenarios and keeps canonical transfer mixed into later phases
- `a0_post_rest_continuation_dense_v1`: concentrates phase 3 fully on late-cycle return/rest and delays canonical exposure to the final mixed phase

If `v1`, `v2`, and the denser continuation profile all leave canonical pinned to the same `10.0`-day / `3`-food / `1`-exit row and easy pinned to the same `6.0`-day / `2`-exit row, treat that as evidence that missing continuation exposure alone is no longer the main bottleneck. The next step should then move to a more invasive training intervention, such as continuation-specific promotion gates or stronger phase weighting, rather than adding yet another continuation scenario.

## Recurrent Module Configuration

`BrainAblationConfig.recurrent_modules` is a tuple of proposer-module names that should use `RecurrentProposalNetwork` instead of the default feed-forward `ProposalNetwork`.

- The default is `()`, which keeps every proposer feed-forward.
- Every name must come from the canonical module set: `visual_cortex`, `sensory_cortex`, `hunger_center`, `sleep_center`, and `alert_center`.
- Recurrent modules are only valid for the modular architecture. Monolithic baselines reject `recurrent_modules`.
- Hidden state is runtime-only. Checkpoints save recurrent weights and biases, but not the live hidden vector from the current episode.

Example:

```python
from spider_cortex_sim.ablations import BrainAblationConfig

config = BrainAblationConfig(
    name="custom_recurrent",
    architecture="modular",
    recurrent_modules=("alert_center", "sleep_center"),
)
```

## Episode-Boundary Hidden State Reset

Recurrent proposer state is intentionally scoped to one episode.

- `SpiderSimulation.run_episode()` resets recurrent hidden state immediately after `world.reset()`.
- Behavior-suite evaluation also resets recurrent hidden state before each scenario rollout.
- Hidden state therefore carries context across timesteps within one episode, but never across training episodes, evaluation episodes, or scenario repetitions.

That design keeps recurrent comparisons fair: a recurrent variant may use recent observation history inside a rollout, but it does not get cross-episode memory as an extra channel.

## Interpreting Recurrent Comparisons

Use `modular_full` as the feed-forward reference and read recurrent deltas the same way as other ablations: start with the no-reflex benchmark, then use reflex-on results as secondary diagnostics.

- If `modular_recurrent` or `modular_recurrent_all` improves no-reflex `scenario_success_rate`, the added within-episode memory is helping the learned policy itself.
- If the gain appears only with reflex support enabled, the recurrent state may be amplifying scaffolded reactions more than improving independent control.
- Improvements concentrated in partially observable or delayed-information scenarios are usually more informative than broad gains in every easy scenario.
- Regressions in simple scenarios can mean the extra recurrent dynamics are adding instability or unnecessary statefulness.

The `modular_recurrent` and `modular_recurrent_all` pair is useful for localization: if the partial recurrent variant helps while the all-recurrent variant regresses, the memory benefit is likely concentrated in the threat, sleep, or hunger streams rather than in every proposer.

## Credit Assignment Strategies

`BrainAblationConfig.credit_strategy` controls how the global policy-gradient signal reaches proposer modules:

- `broadcast`: every active proposer receives the shared policy-gradient signal. This is the default and preserves the original modular training behavior.
- `local_only`: proposer modules receive only local gradients, such as action-center input gradients and reflex auxiliary gradients. The global policy-gradient broadcast weight is `0.0` for every module.
- `counterfactual`: each proposer receives a normalized weight based on how much it changed the selected action's probability.

Counterfactual credit uses a leave-one-out estimate. For each module, the learner zeroes that module's logits in the `action_center` input, keeps the other module logits and action context intact, runs the `action_center` forward path without updating its training cache, and compares the actual selected-action probability against the counterfactual selected-action probability. The raw score is:

```text
actual_policy[action_idx] - counterfactual_policy[action_idx]
```

The absolute scores are normalized to sum to `1.0`, so modules that helped the taken action and modules that interfered with it are both represented by their contribution magnitude. This is more principled than uniform broadcast credit because it asks whether the selected action depended on a module's proposal in the current state, rather than assuming every proposer deserved the same global update.

## Post-`action_center` Topology

- Modular variants still differ in the proposal stage and reflex support.
- Explicit arbitration now happens in `action_center`.
- Before the final choice, `action_center` applies valence-based `priority_gating` across `threat`, `hunger`, `sleep`, and `exploration`.
- `motor_cortex` is now a locomotor correction and execution stage, not the primary arbitrator.
- The short design note for this topology is in `docs/action_center_design.md`.

## Conflict Scenarios

- `food_vs_predator_conflict`: validates that threat overrides foraging when food and predator compete at the same moment
- `sleep_vs_exploration_conflict`: validates that sleep overrides residual exploration in a safe nighttime context
- These scenarios read arbitration directly from `trace.messages[*].payload`, so they do not require `--debug-trace`

## Frozen Proposer Integration Testing

Use frozen proposers when the question is not "can this proposer learn locally?" but "does the downstream integration stack use an already-competent proposer correctly?" This isolates learning in `arbitration_network`, `action_center`, and `motor_cortex` from local proposer learning so failures are easier to attribute.

### API

- `SpiderBrain.freeze_proposers(module_names: Sequence[str] | None = None)`: freezes proposer modules for learning while keeping them active for inference. Pass `None` to freeze every active proposer in the current architecture, or pass a sequence of proposer names to freeze only a subset.
- `SpiderBrain.unfreeze_proposers(module_names: Sequence[str] | None = None)`: removes proposer modules from the frozen set. Pass `None` to clear the entire frozen set, or pass a sequence of proposer names to unfreeze only those modules.
- `SpiderBrain.frozen_module_names()`: returns the sorted list of currently frozen proposer-module names.
- `SpiderBrain.is_module_frozen(name: str)`: returns whether a given proposer module is currently frozen.

Frozen proposers still appear in inference outputs and traces. The only change is that proposer gradient updates are skipped during `learn()`.

### Failure Interpretation

- `proposer_failure`: the expected motivational or action tendency is already wrong at the frozen proposer stage. A common signature is that the expected valence wins, but the resulting action is still wrong because the fixed proposer intent was already bad.
- `integration_failure`: the frozen proposer looks locally competent, but the downstream stack mishandles it. Typical signatures are the wrong `winning_valence`, incorrect suppression or promotion in `module_gates`, or an `action_center` / `motor_cortex` outcome that discards the correct proposer intent.

This distinction matters because a frozen-proposer failure is evidence against the proposer checkpoint itself, while an integration failure points at arbitration, gating, credit assignment, or motor-stage correction.

### End-to-End Workflow

1. Load the checkpoint that contains the proposer modules you want to treat as locally competent.
2. Call `freeze_proposers()` for all proposer modules, or only for the specific proposer subset under test.
3. Disable module dropout for the training run as well. Freezing alone does not keep proposers present during `training=True` episodes if modular dropout is still active. Set `module_dropout=0.0` on the simulation or brain config, or use a `no_module_dropout`-style variant when running integration-only experiments.
4. Continue training with only the integration layers updating: `arbitration_network`, `action_center`, and `motor_cortex`.
5. Evaluate on conflict scenarios and inspect `action.selection` traces to confirm:
   - the expected `winning_valence` is selected,
   - the expected proposer dominates or suppressed modules are down-weighted correctly,
   - the final executed action preserves or appropriately corrects the intended action.
6. If the run fails, classify it as proposer or integration failure before changing the model or the training recipe.

Minimal sketch:

```python
from spider_cortex_sim.simulation import SpiderSimulation

sim = SpiderSimulation(seed=7, max_steps=30)
sim.brain.load_checkpoint("/path/to/checkpoint")
sim.brain.freeze_proposers()
sim.brain.module_dropout = 0.0

for episode_idx in range(20):
    sim.run_episode(
        episode_idx,
        training=True,
        sample=True,
        scenario_name="food_vs_predator_conflict",
    )

stats, trace = sim.run_episode(
    999,
    training=False,
    sample=False,
    capture_trace=True,
    scenario_name="food_vs_predator_conflict",
)
```

If you build the simulation with a custom config instead of mutating the instantiated brain, set `module_dropout=0.0` up front or choose a no-dropout ablation variant. The important invariant is that the frozen proposer modules stay both frozen and present during training episodes.

### Summary Output

Frozen module status is recorded in simulation summaries under the top-level `"frozen_modules"` key. The same list also appears under `summary["config"]["frozen_modules"]` so architecture metadata and training metadata stay aligned.

## Main Commands

Run the publication-grade architecture suite and export `summary` plus CSV:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --checkpoint-selection best \
  --ablation-suite \
  --summary spider_ablation_summary.json \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary
```

Run a fast iterative-development pass when you want quick local feedback rather than the benchmark of record:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --full-summary
```

Run only the direct A0 baseline:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --ablation-variant true_monolithic_policy \
  --full-summary
```

`true_monolithic_policy` is the canonical CLI variant for `A0_true_monolithic` from [architectural_ladder.md](./architectural_ladder.md).

Run a single variant against the modular no-reflex reference in one specific scenario for architecture discussion:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --checkpoint-selection best \
  --ablation-variant monolithic_policy \
  --behavior-scenario night_rest \
  --full-summary
```

Run a custom reflex schedule and record the post-training comparison with `eval_reflex_scale=0.0` during iterative development:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --episodes 12 \
  --eval-episodes 2 \
  --reflex-scale 1.0 \
  --module-reflex-scale alert_center=0.5 \
  --reflex-anneal-final-scale 0.25 \
  --summary spider_reflex_schedule_summary.json \
  --full-summary
```

## Training Regimes

Training regimes package the reflex schedule, optional no-reflex fine-tuning phase, and reflex-dependence loss weights into named presets. Use `--training-regime` when the question is about learned competence under a declared protocol; use `--reflex-anneal-final-scale` only for ad hoc schedule experiments. The two flags are mutually exclusive.

| Regime | Training flow | Intended use |
| --- | --- | --- |
| `baseline` | No reflex annealing, no fine-tuning, no added penalties. | Matches the historical default behavior and acts as the reference for regime comparisons. |
| `reflex_annealed` | Linearly anneals reflex scale to `0.0` across the main training budget, with no extra fine-tuning. | Tests whether competence survives after gradually removing reflex support. |
| `late_finetuning` | Linearly anneals reflex scale to `0.0`, then runs additional learning episodes at no-reflex scale. | Canonical no-reflex learning path and the experiment-of-record regime. |
| `distillation` | Collects teacher rollouts, runs offline policy/action-center/valence distillation, then optionally reuses `finetuning_episodes` for RL on the student. Requires a teacher checkpoint. | Use when the question is "can the modular student represent a policy that a monolithic teacher already learned?" |

Named-regime CLI example:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --training-regime reflex_annealed \
  --behavior-suite \
  --summary spider_reflex_annealed_summary.json \
  --behavior-csv spider_reflex_annealed_rows.csv \
  --full-summary
```

Experiment-of-record shortcut:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --experiment-of-record \
  --behavior-suite \
  --summary spider_experiment_of_record_summary.json \
  --behavior-csv spider_experiment_of_record_rows.csv \
  --full-summary
```

Expected competence-gap pattern:

| Regime | Scaffolded reading | Self-sufficient reading | Gap interpretation |
| --- | --- | --- | --- |
| `baseline` | Reflex-supported performance remains useful as a scaffold diagnostic. | No-reflex rows are the actual benchmark for learned policy competence. | A large gap means the policy still leans on reflex support. |
| `reflex_annealed` | Scaffolded performance should remain a secondary diagnostic. | Self-sufficient performance should improve if gradual removal taught the policy to stand alone. | A narrower gap than `baseline` is evidence that annealing helped. |
| `late_finetuning` | Scaffolded results are retained for context, not for claims. | Primary no-reflex benchmark for learning claims. | The target pattern is the smallest gap with the strongest no-reflex success. |
| `distillation` | Keep teacher performance as the reference. | Read the distilled student with reflex support disabled first; then compare optional post-distillation RL fine-tuning against the distilled-only checkpoint. | If distilled no-reflex performance improves over modular RL from scratch, that is evidence that the modular architecture can express the teacher policy and the bottleneck is training rather than representational capacity. |

## Distillation Workflow

Distillation is implemented. Use it when RL alone cannot tell you whether a modular failure is caused by optimization or by limited representational capacity.

Teacher-student flow:

1. Train a teacher checkpoint with `true_monolithic_policy` (A0) or `monolithic_policy` (A1).
2. Save that checkpoint.
3. Run the modular student with `--training-regime distillation` and `--distillation-teacher-checkpoint`.
4. Read the top-level `distillation` summary block for teacher metadata, dataset size, loss curve, and post-distillation comparisons.
5. Compare four references together:
   - monolithic teacher
   - modular RL from scratch
   - modular distilled
   - modular distilled plus RL fine-tuning

Teacher example:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-variant true_monolithic_policy \
  --checkpoint-selection best \
  --checkpoint-dir checkpoints/a0_teacher \
  --full-summary
```

Student example:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --training-regime distillation \
  --distillation-teacher-checkpoint checkpoints/a0_teacher/best \
  --distillation-epochs 5 \
  --summary spider_distillation_summary.json \
  --full-summary
```

Use distillation when:

- modular RL from scratch underperforms a monolithic teacher and you need to separate architectural limits from training dynamics
- you want a direct answer to the question "can the modular architecture express an effective policy?"
- you need post-distillation evidence before deciding whether more RL tuning is worth pursuing

Interpretation guidance:

- If `modular_distilled` already shows no-reflex competence, the modular architecture can represent useful teacher behavior and the remaining bottleneck is mostly training.
- If only `modular_distilled_plus_rl_finetuning` improves, the representation is plausible but the student still needs RL adaptation to exploit it.
- If neither `modular_distilled` nor `modular_distilled_plus_rl_finetuning` closes any meaningful part of the teacher gap, treat that as evidence against the current modular parameterization or supervision targets, not just against the RL recipe.

## Checkpoint Penalty Configuration

Best-checkpoint selection can penalize direct reflex dependence through:

- `--checkpoint-override-penalty`: weight applied to `mean_final_reflex_override_rate`
- `--checkpoint-dominance-penalty`: weight applied to `mean_reflex_dominance`
- `--checkpoint-penalty-mode`: either `tiebreaker` or `direct`

`tiebreaker` mode preserves the legacy six-part ranking key: primary metric, the remaining metrics, then lower reflex override and dominance rates, then later episode. Use it when reproducing older runs or when penalties should only decide otherwise equivalent checkpoints.

`direct` mode ranks by a composite score first:

```text
primary_metric - (override_penalty * override_rate) - (dominance_penalty * dominance)
```

Use `direct` mode when selecting checkpoints for no-reflex claims, especially with `--experiment-of-record`. It lets a lower-dependence checkpoint beat a reflex-heavy one even when their raw success rates are close.

Direct-penalty CLI example:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --checkpoint-selection best \
  --checkpoint-penalty-mode direct \
  --checkpoint-override-penalty 1.0 \
  --checkpoint-dominance-penalty 1.0 \
  --behavior-suite \
  --full-summary
```

## Learning Evidence

The repository also exposes a separate workflow for testing whether a trained checkpoint outperforms suitable controls without relying only on qualitative narrative.

Smoke command:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --behavior-scenario night_rest \
  --full-summary
```

Canonical short-vs-long command:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --learning-evidence-long-budget-profile report \
  --behavior-suite \
  --summary spider_learning_evidence_summary.json \
  --behavior-csv spider_learning_evidence_rows.csv \
  --full-summary
```

Experiment-of-record no-reflex benchmark:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --experiment-of-record \
  --behavior-suite \
  --summary spider_experiment_of_record_summary.json \
  --behavior-csv spider_experiment_of_record_rows.csv \
  --full-summary
```

Canonical conditions:

- `trained_final`: default-runtime diagnostic for the final checkpoint
- `trained_without_reflex_support`: primary gate condition for the final checkpoint evaluated with `eval_reflex_scale=0.0`
- `trained_reflex_annealed`: trains with the named `reflex_annealed` regime and evaluates the final checkpoint with `eval_reflex_scale=0.0`
- `trained_late_finetuning`: trains with the named `late_finetuning` regime and evaluates the final checkpoint with `eval_reflex_scale=0.0`
- `trained_distilled`: trains with the `distillation` regime without RL fine-tuning and evaluates the student with `eval_reflex_scale=0.0`
- `trained_distilled_finetuned`: trains with the `distillation` regime including RL fine-tuning and evaluates the student with `eval_reflex_scale=0.0`
- `random_init`
- `reflex_only`
- `freeze_half_budget`
- `trained_long_budget`

Practical reading:

- `summary["behavior_evaluation"]["learning_evidence"]["reference_condition"]` is `trained_without_reflex_support`
- `evidence_summary["has_learning_evidence"]` uses the no-reflex `scenario_success_rate` as the primary gate
- `summary["is_experiment_of_record"]` is true for the canonical `late_finetuning` training regime
- `trained_without_reflex_support` participates directly in the gate; `trained_final` remains available as a default-runtime diagnostic condition, but it does not drive the primary gate
- `trained_late_finetuning` is the experiment-of-record condition for no-reflex learning claims; reflex-on results remain diagnostics rather than the primary benchmark
- The exported CSV includes `learning_evidence_condition`, `learning_evidence_policy_mode`, `learning_evidence_training_regime`, `learning_evidence_train_episodes`, `learning_evidence_frozen_after_episode`, `learning_evidence_checkpoint_source`, `learning_evidence_budget_profile`, and `learning_evidence_budget_benchmark_strength`

Regime comparison:

- `SpiderSimulation.compare_training_regimes(regime_names=[...])` compares named regimes such as `baseline`, `reflex_annealed`, and `late_finetuning`
- Each regime publishes `self_sufficient` and `scaffolded` blocks, `success_rates`, `competence_gap`, and deltas versus `baseline`
- `late_finetuning` is marked with `is_experiment_of_record: true`

## Claim Tests As Top-Level Gate

The claim-test suite is the top-level experimental gate for the repository. Ablations, learning-evidence comparisons, and the noise matrix remain the primitive measurement workflows, but they are supporting evidence rather than the final scientific output.

CLI entry point:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --claim-test-suite \
  --summary results.json
```

Composition rules:

- `learning_without_privileged_signals` and `escape_without_reflex_support` compose the learning-evidence conditions
- `memory_improves_shelter_return` composes the modular ablation suite
- `noise_preserves_threat_valence` composes the canonical noise-robustness matrix
- `specialization_emerges_with_multiple_predators` composes the predator-type ablation comparison plus type-specific cortex engagement checks

Relationship hierarchy:

- claim tests
- primitives (`learning_evidence`, `ablation`, `noise_robustness`)
- scenarios
- episode metrics and check pass rates

That hierarchy matters for interpretation. Primitive suites answer local questions such as "which module matters?" or "how much does mismatch hurt?" Claim tests answer the synthesis question: do the results, taken together, support the emergence story strongly enough to count as the benchmark-of-record.

## Noise Robustness Protocol

The noise-robustness workflow treats train-time noise and eval-time noise as separate experimental axes. The canonical protocol is a `4 x 4` matrix over `none`, `low`, `medium`, and `high`, where each row trains under one noise profile and each column evaluates the resulting checkpoint under one eval-time noise profile.

Purpose:

- measure whether a policy only works under the exact disturbance profile it saw during training
- separate matched-condition competence from genuine robustness to distribution shift
- expose whether an architecture degrades smoothly as sensory and motor corruption increase

CLI entry point:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile smoke \
  --noise-robustness \
  --behavior-scenario night_rest \
  --full-summary
```

Budget interaction:

- `--budget-profile` still controls training episodes, eval episodes, scenario repetitions, comparison seeds, and checkpoint cadence
- the robustness workflow multiplies that budget across the canonical train conditions, so `smoke` and `dev` are the practical local loops and `paper` is the benchmark-of-record path
- when the selected budget requires checkpoint selection, keep using `--checkpoint-selection best`; the robustness runner will train or reload each train-condition checkpoint under its own directory

Practical reading:

- `summary["behavior_evaluation"]["robustness_matrix"]["matrix"]` is the nested `{train_profile: {eval_profile: payload}}` cell map
- `robustness_score` is the mean success rate across every train/eval cell
- `diagonal_score` is the mean of matched train/eval conditions
- `off_diagonal_score` is the mean of mismatched train/eval conditions
- `train_marginals` and `eval_marginals` show whether sensitivity is driven more by how the policy was trained or how it is being evaluated

Example interpretation:

- diagonal `0.81`, off-diagonal `0.77`: the policy is largely stable under eval-time noise swaps, so the learned behavior is not tightly coupled to one exact corruption level
- diagonal `0.81`, off-diagonal `0.39`: the policy is specialized to matched train/eval conditions, so claims about robustness should be weakened even if same-condition performance looks strong

## Shaping Minimization Program

Beyond structural ablations, the repository treats reward shaping as a first-class scientific variable. The `austere` profile is not merely a contrast condition; it is the minimal-shaping target used to debug whether behavior survives without dense progress guidance. A well-trained agent should show minimal performance gap between `classic` and `austere`, and interpretation should emphasize behaviors that still work under `austere`.

Smoke command:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --scenario-episodes 1 \
  --behavior-compare-profiles \
  --behavior-scenario night_rest \
  --behavior-seeds 7 \
  --full-summary
```

Practical reading:

- `summary["reward_audit"]` aggregates the static inventory of reward and leakage signals
- `summary["reward_audit"]["reward_profiles"]` compares `classic`, `ecological`, and `austere` by shaping category and shaping disposition
- `summary["reward_audit"]["reward_profiles"][profile]["disposition_summary"]` reports how much weight proxy each profile applies to defended, weakened, removed, outcome-signal, and under-investigation components
- `summary["reward_audit"]["comparison"]["deltas_vs_minimal"]` uses `austere` as the minimal-shaping baseline
- `summary["reward_audit"]["comparison"]["behavior_survival"]` reports which scenarios survive under `austere`
- `predator_dist`, `home_vector`, and `shelter_memory` appear explicitly as higher-risk signals because their derivation paths historically included privileged access to unperceived world state

Profile intent:

- `classic`: denser baseline with stronger progress shaping
- `ecological`: reduces part of the guidance while keeping explicit scaffolding
- `austere`: zeros the main progress terms (`food_progress`, `shelter_progress`, `threat_shelter_progress`, `predator_escape`, `day_exploration`) and acts as the target profile for shaping-minimization debugging

Disposition vocabulary:

- `defended`: retained because the term models an ecological constraint, physiological pressure, or universal cost rather than direct progress guidance
- `weakened`: retained in reduced form because it still contains outcome information but has dense shaping risk
- `removed`: zeroed in `austere` because it acts as direct progress guidance or auxiliary scaffolding
- `outcome_signal`: sparse event or terminal outcome signal, such as eating, predator contact, or death
- `under_investigation`: reserved for terms that still need a defense, reduction, or removal decision

Current high/medium-risk dispositions:

| Component | Category | Risk | Disposition | Current rationale |
| --- | --- | --- | --- | --- |
| `terrain_cost` | event | medium | defended | Retained in `austere` as environmental hazard cost rather than waypoint progress. |
| `night_exposure` | event | medium | defended | Retained in `austere` because exposure is an ecological consequence of missing shelter. |
| `hunger_pressure` | internal_pressure | medium | defended | Retained and strengthened in `austere` as homeostatic need without direct food-direction guidance. |
| `fatigue_pressure` | internal_pressure | medium | defended | Retained and strengthened in `austere` as exhaustion pressure rather than destination guidance. |
| `sleep_debt_pressure` | internal_pressure | medium | defended | Retained and strengthened in `austere` as physiological sleep debt while leaving shelter-finding to behavior. |
| `resting` | event | medium | weakened | Direct rest outcome remains, but configurable rest bonuses are reduced relative to `ecological`. |
| `food_progress` | progress | high | removed | Zeroed in `austere` so food seeking depends on sparse feeding outcomes and learned behavior. |
| `shelter_progress` | progress | high | removed | Zeroed in `austere`, including the threat branch, to test shelter behavior without distance guidance. |
| `predator_escape` | progress | high | removed | Zeroed in `austere`, including escape bonus and stay penalty, so survival depends on consequences and predator dynamics. |
| `day_exploration` | progress | medium | removed | Zeroed in `austere` because it is auxiliary locomotion guidance. |
| `shelter_entry` | event | medium | removed | Zeroed in `austere` because boundary-crossing reward still provides navigation scaffolding. |
| `night_shelter_bonus` | event | medium | removed | Zeroed in `austere` because recurring shelter occupancy reward can hide dependence on dense guidance. |
| `homeostasis_penalty` | internal_pressure | medium | defended | Retained and strengthened in `austere` as physiological deterioration rather than destination instruction. |

## Offline Post-Processing

After generating `summary`, `trace`, and `behavior_csv`, the offline runner can assemble a comparable package without notebooks or manual JSON inspection.

Example:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --summary spider_ablation_summary.json \
  --trace spider_ablation_trace.jsonl \
  --debug-trace \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary

PYTHONPATH=. python3.10 -m spider_cortex_sim.offline_analysis \
  --summary spider_ablation_summary.json \
  --trace spider_ablation_trace.jsonl \
  --behavior-csv spider_ablation_rows.csv \
  --output-dir spider_ablation_report
```

For the benchmark-of-record path, write the package during the simulation run:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --checkpoint-selection best \
  --ablation-suite \
  --summary spider_architecture_paper_summary.json \
  --trace spider_architecture_paper_trace.jsonl \
  --behavior-csv spider_architecture_paper_rows.csv \
  --benchmark-package spider_architecture_paper_package \
  --full-summary
```

`--benchmark-package` is intentionally restricted to `--budget-profile paper --checkpoint-selection best`. The package contains the same offline report bundle plus `benchmark_manifest.json`, `resolved_config.json`, `pip_freeze.txt`, `seed_level_rows.csv`, uncertainty-aware aggregate tables, claim-test uncertainty tables, effect-size tables, supporting CSVs, plots, and `limitations.txt`.

For release posture, citation guidance, and the minimal benchmark-of-record
checklist, see [RELEASE.md](../RELEASE.md), [CHANGELOG.md](../CHANGELOG.md),
and [CITATION.cff](../CITATION.cff). For the operational pre-run and post-run
procedure, use [release_checklist.md](./release_checklist.md). `requirements.txt`
remains a development baseline; benchmark-of-record environment capture comes
from the manifest `environment` block plus `pip_freeze.txt`.

The manifest now records benchmark provenance through its `environment` block,
including git commit, git tag, dirty-state flag, Python version, and platform.
`pip_freeze.txt` captures the dependency snapshot as a hashed package artifact.
Use [release_checklist.md](./release_checklist.md) to verify those fields before
archiving or citing a benchmark-of-record bundle.

Expected outputs in `spider_ablation_report/`:

- `report.md` (begins with an **Overview** summary table and a linked **Table of contents**)
- `report.json`
- `training_eval.svg`
- `scenario_success.svg`
- `scenario_checks.csv`
- `reward_components.csv`
- `ablation_comparison.svg`
- `reflex_frequency.svg`

Practical reading:

- `ablation_comparison.svg` summarizes `scenario_success_rate` by variant when `behavior_evaluation.ablations` exists or can be reconstructed from the CSV
- `report.json["primary_benchmark"]` and the `report.md` Primary Benchmark table feature the no-reflex `scenario_success_rate`; the Ablations table includes `eval_reflex_scale` so `0.0` rows are visible
- `report.md` includes `## Shaping Minimization Program`, which surfaces the dense-vs-minimal gap, component dispositions, behavior survival under `austere`, and a warning banner when shaping dependence is high
- benchmark packages add `## Benchmark-of-Record Summary`, `## Claim Test Results with Uncertainty`, and `## Effect Sizes Against Baselines` to the Markdown report
- `scenario_success.svg` and `scenario_checks.csv` help localize regressions by scenario and check without opening the full JSON
- `reward_components.csv` cross-references reward components aggregated from the summary with observed totals from the trace
- `trace[*].event_log` records the explicit tick order, making causal inspection possible without reconstructing implicit mutations
- `reflex_frequency.svg` uses `trace.messages[*].payload.reflex` as its source and becomes richer when the trace was generated with `--debug-trace`

If a block is missing, the runner does not fail. It records the absence in `report.md` and `report.json`.

Confidence intervals are seed-level percentile bootstrap intervals, reported at the confidence level stored in the package manifest and defaulting to 95%. Interpret a wide interval as limited seed evidence, not as a different pass/fail rule: claim tests still pass or fail on point estimates. Effect-size rows report Cohen's d against the named baseline; `negligible`, `small`, `medium`, and `large` labels describe the absolute standardized effect size, while the sign of Cohen's d indicates the direction of the comparison.

## Workflow Notes

- If no `--behavior-scenario` is provided alongside `--ablation-suite` or `--ablation-variant`, the CLI automatically uses the full behavioral suite
- `summary["behavior_evaluation"]["ablations"]` contains the reference used, the evaluated variants, and no-reflex deltas against `modular_full` evaluated with `eval_reflex_scale=0.0`
- Each variant publishes `without_reflex_support`, which re-evaluates the same checkpoint with `eval_reflex_scale=0.0`; reflex-on results are retained under `with_reflex_support` when present
- The exported CSV includes `ablation_variant`, `ablation_architecture`, `reflex_scale`, `reflex_anneal_final_scale`, `eval_reflex_scale`, `budget_profile`, `benchmark_strength`, `checkpoint_source`, `operational_profile`, `operational_profile_version`, and `noise_profile`
- `summary["config"]["operational_profile"]` records the active operational profile name, version, and effective payload; the CLI can pin it with `--operational-profile default_v1`
- `summary["config"]["noise_profile"]` records the active noise profile name and effective payload; the CLI can pin it with `--noise-profile none|low|medium|high`
- `summary["config"]["budget"]` records the resolved profile (`smoke`, `dev`, `report`, `paper`, or `custom`), benchmark strength, comparison seeds, checkpoint-selection requirements, and any manual overrides
- `summary["evaluation"]["self_sufficient"]` and `summary["evaluation"]["primary_benchmark"]` are the no-reflex benchmark blocks for modular training runs with evaluation episodes; `summary["evaluation"]["scaffolded"]` keeps reflex-supported diagnostics, and `summary["evaluation"]["competence_gap"]` records the scaffolded-minus-self-sufficient deltas
- `--checkpoint-selection best` evaluates intermediate checkpoints with `eval_reflex_scale=0.0`, chooses the best by the requested success metric plus the configured reflex-dependence penalty mode, and persists only `best/` and `last/` when `--checkpoint-dir` is provided

## Credit Diagnostics

Learning updates publish per-module credit diagnostics through the `td_update` message payload:

- `credit_strategy`: the active strategy string, useful for verifying that a trace was generated with `broadcast`, `local_only`, or `counterfactual`
- `module_credit_weights`: per-module weights applied to the shared policy-gradient signal on that step
- `module_gradient_norms`: per-module L2 norms of the total proposal gradient sent to each module on that step

Episode metrics aggregate those step-level signals:

- `mean_module_credit_weights`: mean credit weight per module across learning steps in the episode
- `module_gradient_norm_means`: mean proposal-gradient norm per module across learning steps in the episode

Interpretation:

- In `broadcast`, `module_credit_weights` should be `1.0` for each active proposer (active = not dropped for the state) because the global signal is shared uniformly.
- In `local_only`, `module_credit_weights` should be `0.0` for each active proposer (active = not dropped for the state) because modules learn only from local paths.
- In `counterfactual`, larger `module_credit_weights` identify active proposers (active = not dropped for the state) whose logits most changed the chosen action probability in that state. Weights close to `0.0` mean the action would have looked nearly the same without that module's proposal.
- `module_gradient_norm_means` should be read together with the credit weights for each active proposer (active = not dropped for the state). Inactive proposers should show zero credit; zero-credit active proposers may still have nonzero gradient norms because local action-center input gradients or reflex auxiliary gradients still apply.

## Reflex Dependence Failure Signals

The ablation workflow can measure reflex dependence gradually, not only as a binary switch. Treat override and dominance metrics as failure signals: higher values mean the final behavior is still leaning on reflex scaffolding instead of the learned policy.

The main aggregate signals live in `summary["evaluation"]` and in legacy per-scenario blocks:

- `mean_reflex_usage_rate`
- `mean_final_reflex_override_rate`
- `mean_reflex_dominance`
- `mean_module_reflex_usage_rate`
- `mean_module_reflex_override_rate`
- `mean_module_reflex_dominance`
- `mean_module_contribution_share`
- `dominant_module_distribution`
- `mean_dominant_module_share`
- `mean_effective_module_count`
- `mean_module_agreement_rate`
- `mean_module_disagreement_rate`

Interpretation:

- `mean_final_reflex_override_rate` above about `0.10` is a warning that final-action reflex overrides are still common enough to distort the benchmark
- `mean_reflex_dominance` above about `0.25` is a warning that reflex logits remain large relative to learned logits
- In offline reports, the Diagnostics table labels these as `Reflex Dependence: override rate` and `Reflex Dependence: dominance`; `warning` status should be read as scaffold dependence, not as learned robustness
- No-reflex `scenario_success_rate` remains the benchmark of record even when reflex-on diagnostics look better

The enriched `action_center` trace also serializes:

- `module_contribution_share`
- `dominant_module`
- `dominant_module_share`
- `effective_module_count`
- `module_agreement_rate`
- `module_disagreement_rate`

## Optional Curriculum

The main runner accepts `--curriculum-profile ecological_v1` or `--curriculum-profile ecological_v2` to organize training into four reproducible phases before final evaluation. `ecological_v1` remains available for backward compatibility and still promotes phases with the legacy aggregate `success_threshold` fallback.

Short example:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile smoke \
  --curriculum-profile ecological_v2 \
  --behavior-csv spider_curriculum_rows.csv \
  --full-summary
```

`ecological_v2` targets named behavioral sub-skills and gates promotion on named behavior-suite checks. A phase can train on a broader scenario set while promoting on a narrower scenario set, which lets prerequisite behavior be isolated before adding the next exposure.

| phase | skill name | training scenarios | promotion scenarios | promotion checks |
| --- | --- | --- | --- | --- |
| Shelter safety & predator awareness | `predator_response` | `night_rest`, `predator_edge`, `entrance_ambush` | `predator_edge` | `predator_detected`, `predator_reacted` |
| Shelter exit commitment | `shelter_exit` | `entrance_ambush`, `shelter_blockade`, `food_deprivation` | `food_deprivation` | `commits_to_foraging` |
| Food approach under exposure | `food_approach` | `food_deprivation`, `open_field_foraging`, `exposed_day_foraging` | `food_deprivation`, `open_field_foraging`, `exposed_day_foraging` | `approaches_food`, `made_food_progress`, `day_food_progress` |
| Corridor navigation & hunger survival | `corridor_navigation+hunger_commitment` | `corridor_gauntlet`, `food_deprivation` | `corridor_gauntlet`, `food_deprivation` | `corridor_survives`, `corridor_food_progress`, `hunger_reduced`, `survives_deprivation` |

Promotion criteria are serialized in each phase as `promotion_check_specs`. Each entry names the promotion `scenario`, the behavioral `check_name`, the `required_pass_rate`, and an `aggregation` field. Built-in `ecological_v2` phases use `all`, which promotes only when every listed check meets its threshold; `any` is also supported for future phase definitions that should promote when at least one listed check passes. During promotion, the runner reads `payload["suite"][scenario]["checks"][check_name]["pass_rate"]` from `evaluate_behavior_suite()`. When `promotion_check_specs` is empty, the runner uses the older `scenario_success_rate >= success_threshold` rule and records `threshold_fallback`.

Practical reading:

- `summary["config"]["training_regime"]` describes the active regime and resolved budget
- `summary["curriculum"]` lists phases, skill names, per-phase scenarios, serialized promotion criteria, check-level promotion attempts, `final_check_results`, `promotion_reason`, and final status
- `summary["behavior_evaluation"]["curriculum_comparison"]` compares `flat` versus `curriculum` under the same seeds and highlights `open_field_foraging`, `corridor_gauntlet`, `exposed_day_foraging`, and `food_deprivation`
- The exported CSV includes `training_regime`, `curriculum_profile`, `curriculum_phase`, `curriculum_skill`, `curriculum_phase_status`, and `curriculum_promotion_reason`
- `curriculum_skill` names the sub-skill targeted by the latest executed curriculum phase for that row
- `curriculum_promotion_reason` is `all_checks_passed`, `any_check_passed`, `check_failed:<name>`, `threshold_fallback`, or blank when no curriculum phase decision exists

## Recommended Budgets

| profile | intended use | benchmark strength | episodes | eval episodes | max steps | scenario episodes | comparison seeds | checkpoint interval | selection scenario episodes | requires checkpoint selection |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `smoke` | Sanity / CI | `quick` | 6 | 1 | 60 | 1 | `7` | 2 | 1 | `False` |
| `dev` | Fast local iteration | `quick` | 12 | 2 | 90 | 1 | `7/17/29` | 4 | 1 | `False` |
| `report` | Longer local validation | `strong` | 24 | 4 | 120 | 2 | `7/17/29/41/53` | 6 | 1 | `False` |
| `paper` | Publication-grade / architecture claims | `publication` | 48 | 8 | 150 | 4 | `7/17/29/41/53/67/79` | 8 | 2 | `True` |

Canonical commands:

```bash
# sanity / CI
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile smoke \
  --ablation-variant monolithic_policy \
  --behavior-scenario night_rest \
  --full-summary

# fast local iteration (not the benchmark of record)
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --full-summary

# publication-grade / architecture claims benchmark of record
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --checkpoint-selection best \
  --ablation-suite \
  --full-summary

# publication-grade / no-reflex learning experiment of record
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --experiment-of-record \
  --behavior-suite \
  --full-summary
```

## Primary Benchmark

The primary benchmark is always the self-sufficient result: JSON blocks under `evaluation.self_sufficient` and `evaluation.primary_benchmark`, plus CSV rows with `competence_type=self_sufficient`, `is_primary_benchmark=True`, and `eval_reflex_scale=0.0`.

For no-reflex learning claims, use the experiment-of-record workflow:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --experiment-of-record \
  --behavior-suite \
  --summary spider_experiment_of_record_summary.json \
  --behavior-csv spider_experiment_of_record_rows.csv \
  --full-summary
```

`--experiment-of-record` selects `late_finetuning`, enables `--checkpoint-selection best`, and uses direct reflex-dependence penalties. Reflex-supported rows are still exported as `competence_type=scaffolded`, but they are context for the competence gap rather than the claim target.

For architecture claims, keep using the publication-grade ablation path with `--budget-profile paper --checkpoint-selection best`; for learning-without-reflex claims, cite the experiment-of-record path above.

## Architecture Discussion Anchor

For architecture discussion, benchmark screenshots, or any claim that one architecture outperforms another, anchor on the `paper` profile with checkpoint selection enabled. Treat `dev` as an iteration aid and `report` as a longer local validation profile, not as the benchmark of record.

For no-reflex learning claims, the experiment-of-record workflow is `--experiment-of-record`: it selects the `late_finetuning` training regime, enables best-checkpoint selection, uses direct reflex-dependence penalties, and keeps no-reflex `scenario_success_rate` as the canonical benchmark surface. Reflex-on results remain diagnostic scaffolded competence, not the claim target.

Canonical benchmark-of-record command:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile paper \
  --checkpoint-selection best \
  --ablation-suite \
  --summary spider_architecture_paper_summary.json \
  --behavior-csv spider_architecture_paper_rows.csv \
  --full-summary
```

## Claims and Evidence Requirements

For architecture claims, report all of the following together:

- the no-reflex aggregate benchmark (`scenario_success_rate` with `eval_reflex_scale=0.0`) from the ablation workflow
- the budget profile and whether checkpoint selection was enabled
- robustness sensitivity from the noise protocol, including `robustness_score`, `diagonal_score`, and `off_diagonal_score`

Interpretation rule:

- if diagonal performance is strong but off-diagonal performance drops sharply, describe the result as matched-condition competence rather than broad robustness
- if off-diagonal performance remains close to diagonal performance, that is evidence that the architecture is less sensitive to train/eval noise mismatch rather than merely tuned to one corruption regime

## Illustrative Dev Check-In Result

This section remains useful for iterative development and quick local check-ins. It is not the benchmark-of-record path for architecture claims.

Command used to generate the table below:

```bash
PYTHONPATH=. python3.10 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --full-summary
```

Implicit parameters:

- `budget_profile=dev`
- `reward_profile=classic`
- `map_template=central_burrow`
- `ablation_seeds=(7, 17, 29)`
- `scenario_episodes=1`
- full behavioral scenario suite

### Variant Summary

All rows in this table are interpreted as no-reflex rows (`eval_reflex_scale=0.0`). The reference is `modular_full` evaluated with `eval_reflex_scale=0.0`, not `modular_full` with reflex support left on. Rows marked `n/a` are canonical variants added after this illustrative dev snapshot, or variants not included in that run; rerun the command above to regenerate current metrics.

| variant | architecture | eval reflex scale | suite success | episode success | delta suite vs ref | delta episode vs ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `modular_full` | `modular` | 0.00 | 0.30 | 0.37 | +0.00 | +0.00 |
| `no_module_dropout` | `modular` | 0.00 | 0.30 | 0.47 | +0.00 | +0.10 |
| `no_module_reflexes` | `modular` | 0.00 | 0.10 | 0.33 | -0.20 | -0.03 |
| `fixed_arbitration_baseline` | `modular` | 0.00 | n/a | n/a | n/a | n/a |
| `learned_arbitration_no_regularization` | `modular` | 0.00 | n/a | n/a | n/a | n/a |
| `reflex_scale_0_25` | `modular` | 0.00 | 0.10 | 0.30 | -0.20 | -0.07 |
| `reflex_scale_0_50` | `modular` | 0.00 | 0.30 | 0.37 | +0.00 | +0.00 |
| `reflex_scale_0_75` | `modular` | 0.00 | 0.10 | 0.30 | -0.20 | -0.07 |
| `drop_visual_cortex` | `modular` | 0.00 | 0.40 | 0.47 | +0.10 | +0.10 |
| `drop_sensory_cortex` | `modular` | 0.00 | 0.30 | 0.50 | +0.00 | +0.13 |
| `drop_hunger_center` | `modular` | 0.00 | 0.40 | 0.50 | +0.10 | +0.13 |
| `drop_sleep_center` | `modular` | 0.00 | 0.10 | 0.30 | -0.20 | -0.07 |
| `drop_alert_center` | `modular` | 0.00 | 0.10 | 0.37 | -0.20 | +0.00 |
| `local_credit_only` | `modular` | 0.00 | 0.10 | 0.30 | -0.20 | -0.07 |
| `counterfactual_credit` | `modular` | 0.00 | n/a | n/a | n/a | n/a |
| `monolithic_policy` | `monolithic` | 0.00 | 0.10 | 0.30 | -0.20 | -0.07 |

- `fixed_arbitration_baseline`: keeps the current modules but disables learned arbitration and uses the fixed valence formulas and priority gates only
- `learned_arbitration_no_regularization`: keeps learned arbitration active but sets arbitration gate and valence regularization to zero
- `counterfactual_credit`: keeps the full modular architecture but uses leave-one-out policy credit; this snapshot predates that variant, so current metrics are marked `n/a`

### Per-Scenario Comparison

Short legend: `full=modular_full`, `no_drop=no_module_dropout`, `no_reflex=no_module_reflexes`, `fixed_arb=fixed_arbitration_baseline`, `no_arb_reg=learned_arbitration_no_regularization`, `local=local_credit_only`, `cf=counterfactual_credit`, `r025=reflex_scale_0_25`, `r050=reflex_scale_0_50`, `r075=reflex_scale_0_75`, `d_visual=drop_visual_cortex`, `d_sensory=drop_sensory_cortex`, `d_hunger=drop_hunger_center`, `d_sleep=drop_sleep_center`, `d_alert=drop_alert_center`, `mono=monolithic_policy`.

| scenario | full | no_drop | no_reflex | fixed_arb | no_arb_reg | local | cf | r025 | r050 | r075 | d_visual | d_sensory | d_hunger | d_sleep | d_alert | mono |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `night_rest` | 0.00 | 0.33 | 0.00 | n/a | n/a | n/a | n/a | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 | 0.67 | 0.00 |
| `predator_edge` | 1.00 | 1.00 | 1.00 | n/a | n/a | n/a | n/a | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `entrance_ambush` | 0.33 | 1.00 | 0.67 | n/a | n/a | n/a | n/a | 0.67 | 1.00 | 0.33 | 0.67 | 0.67 | 1.00 | 0.33 | 0.33 | 0.67 |
| `open_field_foraging` | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `shelter_blockade` | 0.33 | 0.67 | 0.67 | n/a | n/a | n/a | n/a | 0.33 | 0.67 | 0.33 | 1.00 | 1.00 | 1.00 | 0.33 | 0.33 | 0.33 |
| `recover_after_failed_chase` | 1.00 | 0.67 | 0.67 | n/a | n/a | n/a | n/a | 0.67 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 0.67 |
| `corridor_gauntlet` | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `two_shelter_tradeoff` | 1.00 | 1.00 | 0.33 | n/a | n/a | n/a | n/a | 0.33 | 0.00 | 0.67 | 1.00 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 |
| `exposed_day_foraging` | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `food_deprivation` | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The per-scenario `local` and `cf` columns are marked `n/a` because this
illustrative snapshot did not retain those per-scenario rows.

## Quick Interpretation

- `local_credit_only` separates topology from credit assignment: it keeps current modular inference but restricts training to local credit by `action_center` input slice.
- The `reflex_scale_*` sweep lets you see whether the policy degrades smoothly, collapses early, or remains dependent on reflex overrides after training.
- The primary aggregate comparison is no-reflex `scenario_success_rate`; reflex-on improvements should be treated as diagnostic evidence of remaining scaffold dependence until they survive `eval_reflex_scale=0.0`.
- In traces and debug output, the correct read order is now: modular or monolithic proposals -> valence-based priority gating -> `action_center` -> `motor_cortex` correction -> optional `final_reflex_override` -> final action.
- When the trace is enriched, use `winning_valence`, `module_gates`, `suppressed_modules`, `module_contribution_share`, and `dominant_module` to interpret conflicts and coadaptation without manual inference.
- Gains are not uniformly positive for every form of modularity: with `12` training episodes, some removal ablations (`drop_visual_cortex`, `drop_hunger_center`) score better on the aggregate benchmark. That is a useful result, not a documentation problem.
- The per-scenario table shows where the benchmark is still weak: `open_field_foraging`, `corridor_gauntlet`, `exposed_day_foraging`, and `food_deprivation` remain at `0.00` for every variant under this short configuration. They remain good candidates for increased training budget or revised checks.

## Reading the `0.00` Cases

The weak-signal scenarios are now formal capability probes. Read them through the
scenario-owned `probe_type`, `target_skill`, `geometry_assumptions`, and
`acceptable_partial_progress` fields in `scenarios.py`, not as undifferentiated
benchmark failures. A `0.00` score on a capability probe is interpretable when
`failure_mode`, `progress_band`, and `outcome_band` are present.

| Scenario | Experimental purpose | Failure-mode interpretation |
| --- | --- | --- |
| `open_field_foraging` | Probe `food_vector_acquisition_exposed`: can the spider acquire a food vector after leaving shelter into exposed terrain? | `left_without_food_signal` or `regressed_and_died` points to food-signal or geometry calibration. Positive `food_distance_delta` or shelter exit with a food signal counts as partial acquisition evidence. |
| `corridor_gauntlet` | Probe `corridor_navigation_under_threat`: can the spider exit shelter and commit directionally through a corridor with a lizard on the row? | `frozen_in_shelter`, contact failures, `survived_no_progress`, and `progress_then_died` separate shelter exit, threat handling, navigation, and survival timing. Positive food progress without contact is capability evidence. |
| `exposed_day_foraging` | Probe `daytime_foraging_under_patrol`: can the spider make food progress during daytime with a nearby patrol after frontier-separated placement? | `cautious_inert` means arbitration chose safety over food, while `partial_progress` is acceptable probe behavior showing foraging capability under pressure. Remaining failures should be interpreted as arbitration limits rather than predator-on-food geometry bugs. |
| `food_deprivation` | Probe `hunger_driven_commitment`: can acute hunger drive shelter exit, food approach, and recovery before the homeostatic timer wins? | `no_commitment`, `orientation_failure`, and `timing_failure` distinguish missing hunger commitment, wrong direction, and a calibrated race lost after commitment. `commits_to_foraging=True` with `approaches_food=True` is partial commitment evidence. |

These scenarios are not claim tests. Claim tests target emergence hypotheses:
learning without privileged support, no-reflex escape, memory benefit, noise
stability, and predator-type specialization. The capability probes instead map
capability boundaries, geometry assumptions, and calibration failures. They stay
in the full benchmark with `benchmark_tier="capability"` and
`is_capability_probe=True`, but they are excluded from `ClaimTestSpec.scenarios`.

In practice:

- `suite[scenario]["diagnostics"]["primary_outcome"]` summarizes the dominant outcome
- `suite[scenario]["diagnostics"]["outcome_distribution"]` shows the mix of episode outcomes
- `suite[scenario]["diagnostics"]["primary_failure_mode"]` and `failure_mode_distribution` separate commitment, missing food signal, orientation, timing, and scoring failures when a scenario emits `failure_mode`
- `suite[scenario]["probe_type"]`, `target_skill`, `benchmark_tier`, and `acceptable_partial_progress` record the formal probe framing
- `partial_progress_rate` helps separate a poorly calibrated scenario from one that truly has no useful signal
- In the CSV, use `is_capability_probe`, `scenario_focus`, `metric_progress_band`, `metric_outcome_band`, and `metric_failure_mode` to filter these cases without opening the full JSON

Additional direct-policy diagnostics:

- `true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD variant. Adds option-specific recurrent hidden dynamics so the previous option can directly reshape the next hidden transition instead of only affecting logits or decoder projections.
- `true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-OAH variant. Adds the per-option action head on top of the option-dynamics branch.
- `true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-OSH variant. Adds an explicit per-option, per-age action-sequence head on top of the option-dynamics branch.
- `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR variant. Adds a persistent option-conditioned recurrent decoder state on top of the option-dynamics branch so action execution can maintain its own within-option trajectory state.
- `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-OAT variant. Adds previous-executed-action-conditioned decoder control on top of the decoder-recurrent branch.
- `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-OAC variant. Adds a dedicated executed-action controller state on top of the decoder-recurrent branch and feeds it directly into policy logits.
- `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-OATK variant. Adds a persistent executed-action token state on top of the decoder-recurrent branch and uses that token state as the policy core for action decoding.
- `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-OARC variant. Adds a separate executed-action recurrent core plus its own policy head on top of the decoder-recurrent branch.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SARH variant. Keeps the same controller stack, but detaches the executed-action recurrent head from the decoder path and drives it directly from the base hidden state plus previous action history.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAPP variant. Keeps the detached executed-action recurrent head and also gives it a separate recurrent policy-path projection plus its own policy logits, so executed-action scoring no longer reuses the shared option-action policy head.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB variant. Keeps the detached action path but replaces the policy-path reuse with a separate recurrent executed-action backbone driven directly by observation and previous executed action, with its own logits and persistent state.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_action_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PAT variant. Keeps the separate executed-action backbone from `..._SAB`, but extends teacher supervision into the post-rest forage window so the detached backbone gets direct `MOVE_RIGHT` pressure after successful post-rest release.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_teacher_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRS variant. Keeps the separate executed-action backbone from `..._SAB`, but broadens the post-rest teacher so awake inside/deep states after real sleep also trigger the full release sequence before the forage window.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_replay_boost_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRS-RB variant. Keeps the broader post-rest release-sequence teacher from `..._SAB_PRS`, but raises continuation replay passes and continuation-stage teacher weights specifically when that release sequence is active.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRS-DST variant. Keeps the broader release-sequence teacher and replay-boost stack from `..._SAB_PRS_RB`, but replaces the active-sequence supervision with stage-conditioned soft action / option / phase targets instead of only one-hot labels.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-DST variant. Keeps the detached action backbone and release-sequence teacher stack from `..._SAB_PRS_DST`, but switches the non-local signal source to a tiny scripted teacher dataset collected from recovered-shelter continuation probes and distilled offline during curriculum training.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_sequence_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-SEQ-DST variant. Keeps the detached action backbone and non-local offline distillation hook from `..._SAB_PRB_DST`, but broadens the scripted teacher dataset to multi-step local release / outside traverse / late-forage return / re-rest sequences across the continuation probes.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_family_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-FAM-DST variant. Keeps the detached action backbone and continuation-triggered non-local distillation hook from `..._SAB_PRB_SEQ_DST`, but replaces the local probe snippets with a longer teacher rollout collected on the real canonical and easy continuous-survival scenarios.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trajectory_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-TRAJ-DST variant. Keeps the detached action backbone and continuation-triggered non-local distillation hook from `..._SAB_PRB_HANDOFF_DST`, but replaces the fallback-heavy handoff rollout with a cycle-204-style one-shot redirect teacher that follows the live teacher policy outside the critical states.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_cycle_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-CYCLE-DST variant. Keeps the detached action backbone and continuation-triggered non-local distillation hook from `..._SAB_PRB_TRAJ_DST`, but tries to extend the redirect teacher into a fuller second-cycle state machine with intended forage, return, and re-rest stages.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trace_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-TRACE-DST variant. Keeps the detached action backbone and continuation-triggered non-local distillation hook from `..._SAB_PRB_CYCLE_DST`, but replaces the purely local cycle-state teacher with a trace-backed Cycle 204 handoff prefix and explicit stitched food-seek / return / re-rest teacher stages.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_rollout_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-ROLLOUT-DST variant. Keeps the detached action backbone and continuation-triggered non-local distillation hook from `..._SAB_PRB_TRACE_DST`, but replaces the stitched late-stage teacher tail with a forced Cycle 204 handoff prefix followed by the altered live teacher policy rollout, collecting only post-handoff samples.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_frontier_teacher_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-FRONTIER-DST variant. Intends to replace the altered live rollout tail with a fixed `true_monolithic_policy` frontier checkpoint teacher after the forced Cycle 204 handoff prefix. In the current runtime this branch also records checkpoint-compatibility fallback metadata because the old frontier checkpoint can no longer be loaded directly when its saved architecture version drifts.
- `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_replayable_teacher_distill_option_replay_policy`: Diagnostic A0-TCH-OPT-REPLAY-PHASE-ODS-ORD-ODR-SAB-PRB-RPLY-DST variant. Keeps the detached action backbone and post-handoff distillation hook, but replaces both checkpoint and stitched-trace teachers with a version-stable replayable teacher that directly applies the successful state-based `one_shot_handoff_probe` logic during an online canonical rollout and distills the resulting post-trigger trajectory.

Interpretation notes:

- For `true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy`, the key read is whether option-specific recurrent dynamics can move official option identity even if actions stay flat.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy`, the key read is whether adding the per-option action head changes executed behavior or collapses option choice back into `RETURN_TO_SHELTER`.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy`, the key read is stricter: if even explicit per-option, per-age action logits leave both official traces on the old action family, additive age-conditioned action decoding is still too weak and the next branch should move to a stronger option-conditioned action-execution controller.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy`, the key read is whether a persistent option-conditioned decoder state can finally convert recovered canonical phase into different executed actions. If canonical phase returns to `POST_REST_REACTIVATE` on the official path but option choice and actions still stay frozen, then the remaining blocker is even deeper than decoder-local trajectory memory and the next branch should intervene more directly in executed action control.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy`, the key read is whether previous executed actions provide the missing control signal. If official easy and canonical still stay on the same action family while canonical also stays at `RETURN_TO_SHELTER 300` and `POST_REST_REACTIVATE 300`, then the next branch should move to a more direct executed-action controller state rather than another transition-style feature.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy`, the key read is whether a dedicated controller state on top of the same backbone can finally move executed behavior. If the official action family still stays unchanged and the internal phase read regresses, the next branch should change the executed-action backbone itself rather than add another controller layer.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy`, the key read is whether a more direct executed-action token backbone can finally turn recovered canonical post-rest phase into different behavior. If canonical recovers `POST_REST_REACTIVATE` while easy stays in `INITIAL_FORAGE`, yet both official traces still stay on the unchanged action family with `RETURN_TO_SHELTER` throughout, the next branch should move to a more isolated executed-action core rather than another decoder-local token/controller variation.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy`, the key read is whether a decoder-derived but separately recurrent executed-action core can finally change behavior. If it collapses back to the old `INITIAL_FORAGE` / `LATE_FORAGE` basins while both official traces still stay on the unchanged action family with `RETURN_TO_SHELTER` throughout, the next branch should move beyond decoder-derived action cores and test a fully separate executed-action recurrent policy head.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy`, the key read is whether detaching the executed-action recurrent head from the decoder path can finally make option identity matter for real behavior. If official option identity flips to `FORAGE` while the executed action family still stays unchanged, the next branch should no longer focus on option/decoder coupling alone and should instead give action execution its own recurrent policy path and training objective.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy`, the key read is whether a detached recurrent policy path plus its own policy-path logits can finally turn recovered canonical phase into changed executed behavior. If canonical still stays on the unchanged one-exit action family while returning to `RETURN_TO_SHELTER` at the option layer and `POST_REST_REACTIVATE` at the phase layer, the next branch should move to a more isolated executed-action backbone with its own optimization path rather than another controller-local policy-path variant.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy`, the key read is whether a fully separate executed-action backbone can preserve the useful internal movement from prior branches while finally changing executed behavior. If both official traces still stay on the unchanged action family and canonical phase falls back to `INITIAL_FORAGE`, the controller-composition line is saturated and the next branch should change training signal directly rather than add another execution-backbone variation.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_action_teacher_option_replay_policy`, the key read is whether direct post-rest forage supervision is enough once the detached action backbone exists. If official easy and canonical still stay on the unchanged action family with canonical frozen at `INITIAL_FORAGE`, then supervising only the late forage-window step is too narrow and the next branch should broaden or distill the full post-rest release sequence.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_teacher_option_replay_policy`, the key read is whether broadening the teacher trigger to the whole post-rest release sequence is enough once the detached action backbone exists. If official easy and canonical still stay on the unchanged action family with canonical frozen at `INITIAL_FORAGE`, then trigger broadening alone is exhausted and the next branch should move to stronger sequence distillation or replay pressure rather than another teacher-trigger variant.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_replay_boost_option_replay_policy`, the key read is whether stronger replay pressure on the same release-sequence labels is enough once the detached action backbone and broader trigger already exist. If official easy and canonical still stay on the unchanged action family with canonical frozen at `INITIAL_FORAGE`, then replay-boost tuning is also exhausted and the next branch should move to stronger full-sequence distillation or a different detached action-backbone objective rather than more replay-pass tuning.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_distill_option_replay_policy`, the key read is whether stage-conditioned soft sequence distillation can change the official path after the detached action backbone, broader release trigger, and replay-boost stack are already present. If official easy and canonical still stay on the unchanged action family with canonical frozen at `INITIAL_FORAGE`, then the same release labels are exhausted even under soft distillation, and the next branch should move to a genuinely different detached action-backbone objective such as teacher-dataset distillation from recovered-shelter handoff probes or another non-local training signal.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_distill_option_replay_policy`, the key read is whether a genuinely non-local teacher-rollout signal can move the official path after local release-label shaping is exhausted. If canonical option identity moves but executed actions and phase stay frozen, the probe distillation is live but too weak, and the next branch should strengthen multi-step teacher-rollout distillation or another trajectory-level detached-action objective rather than returning to local label tweaks.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_sequence_distill_option_replay_policy`, the key read is whether broader local multi-step teacher snippets are enough once the tiny non-local probe branch has already moved canonical option identity. If canonical stays in the same `DEEPEN_IN_SHELTER` / `INITIAL_FORAGE` basin with unchanged actions, then broader local snippets are exhausted too and the next branch should move to a more global successful teacher trajectory signal rather than another local-sequence expansion.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_family_distill_option_replay_policy`, the key read is whether moving from local snippets to a global detached-action teacher rollout actually recreates the successful handoff family. If canonical still stays in the same `DEEPEN_IN_SHELTER` / `INITIAL_FORAGE` basin and the live teacher dataset itself stays dominated by fallback-rest behavior without `family_release` / `family_hold` / `family_continue`, then scripted probe-family heuristics are exhausted too and the next branch should use a higher-fidelity successful teacher source rather than another heuristic rollout variant.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_handoff_distill_option_replay_policy`, the key read is whether a higher-fidelity detached-action teacher rollout that really does recreate the intended handoff stages can finally move the official canonical basin. If the live dataset includes `handoff_family_release`, `handoff_family_hold`, and `handoff_family_continue` but canonical still stays in the same `DEEPEN_IN_SHELTER` / `INITIAL_FORAGE` basin with the unchanged one-exit action family, then the bottleneck is no longer missing scripted handoff states. The next branch should move beyond heuristic handoff rollouts and use a fuller successful trajectory or trace-backed teacher source instead.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trajectory_distill_option_replay_policy`, the key read is whether a cycle-204-style one-shot redirect teacher that follows the live policy outside the critical states can finally move the official canonical basin. If the live dataset includes `trajectory_release`, `trajectory_hold`, and `trajectory_continue`, easy recovers from the handoff-branch collapse, but canonical still stays in the same `DEEPEN_IN_SHELTER` / `INITIAL_FORAGE` basin with the unchanged one-exit action family, then critical-state redirect datasets are still too local. The next branch should use a fuller successful multi-exit trajectory or trace-backed second-cycle teacher source that carries farther through forage, return, and re-rest.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_cycle_distill_option_replay_policy`, the key read is whether extending the redirect teacher into an explicit second-cycle state machine actually creates live forage / return / re-rest teacher samples and moves the official canonical basin. If the live dataset still only materializes `cycle_release`, `cycle_hold`, and `cycle_continue`, while canonical still stays in the same `DEEPEN_IN_SHELTER` / `INITIAL_FORAGE` basin with the unchanged one-exit action family, then local teacher-state variants are exhausted too. The next branch should move to a fuller successful multi-exit trajectory or trace-backed second-cycle teacher source that is guaranteed to carry farther through forage, return, and re-rest.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trace_distill_option_replay_policy`, the key read is whether a stronger trace-backed teacher source that really does contain the Cycle 204 prefix plus explicit late-cycle `trace_return` / `trace_rerest` stages can finally move the official canonical basin. If easy improves and the live dataset contains those late stages, but canonical still stays on the unchanged one-exit action family, then the bottleneck is no longer just “the teacher never reaches late-cycle states.” The next branch should move beyond stitched local stage families and use a fuller trace-backed or successful multi-exit teacher source that preserves more of the real downstream rollout after the handoff.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_rollout_distill_option_replay_policy`, the key read is whether using the forced Cycle 204 handoff prefix once and then following the altered live teacher policy farther downstream can finally move the official canonical basin. If the live dataset becomes denser but is still dominated by `rollout_live`, while easy falls back to the old weak family and canonical still stays on the unchanged one-exit action family, then redirected altered-policy tails are also insufficient. The next branch should move to a fuller successful multi-exit or trace-backed teacher source rather than another redirected rollout tail.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_frontier_teacher_distill_option_replay_policy`, the key read is whether the old `true_monolithic_policy` frontier checkpoint is still reusable as a fixed teacher at all. If the collector immediately reports an architecture-version incompatibility and falls back to the redirected rollout teacher while canonical stays on the unchanged one-exit action family, then checkpoint-format drift itself is now part of the blocker. The next branch should stop assuming direct checkpoint reuse and move to a version-stable successful teacher source or an explicit checkpoint-migration path.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_replayable_teacher_distill_option_replay_policy`, the key read is whether a version-stable successful teacher source is enough once checkpoint drift is removed from the picture. If the offline teacher really does recreate `first_up_redirects=1` plus `down_redirects=3`, yet official canonical still stays on the unchanged one-exit action family and easy still has no post-sleep exit, then the blocker is no longer just teacher fidelity. The next branch should move beyond same-observation teacher variants and test real local affordance input expansion.
- Cycle 240 landed on that second outcome: canonical phase recovered to `POST_REST_REACTIVATE 300`, but option choice still stayed `RETURN_TO_SHELTER 300` and executed actions stayed on the unchanged one-exit family, so the next intervention should target action execution more directly rather than adding another decoder-local tweak.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_transition_prediction_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy`, the key read is whether keeping the full local-transition stack but replacing purely static consequence usage with a learned transition-prediction head plus feedback is enough to move the official basin. If official easy and canonical still stay on the unchanged action family with no post-sleep exits, then a one-step learned consequence surface is also too weak and the next branch should move to a richer multi-step learned consequence surface or another stronger dynamic consequence objective rather than more static local appends.
- `A0_TCH_OPT_REPLAY_PHASE_ODS_ORD_ODR_SAB_LTR2_TR2P_PRB_RPLY_DST_true_monolithic_teacher_option` maps to the diagnostic `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_transition_rollout_prediction_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy` ablation.
- For `true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_transition_rollout_prediction_feedback_post_rest_probe_replayable_teacher_distill_option_replay_policy`, the key read is whether adding a learned multi-step rollout-prediction head plus feedback on top of the full local-transition stack can finally move the official basin. If official easy and canonical still stay on the unchanged action family with no post-sleep exits, then appended or auxiliary learned consequence surfaces are exhausted on this line and the next branch should move to a stronger dynamic-consequence objective or another more invasive controller/input intervention rather than another local prediction head.
- Cycle 268 adds the important correction to that interpretation: the original Cycle 267 implementation misrouted the rollout-prediction feedback so it did not actually reach direct policy logits. After correcting the forward path and rerunning the official branch, canonical and easy still stayed flat on the same executed-action families. That means the conclusion is now stronger: even when the learned multi-step rollout-consequence signal really does reach action logits, the official basin does not move. The next branch should therefore move beyond local prediction heads entirely and test a stronger controller/input intervention or another non-local dynamic-consequence objective.
