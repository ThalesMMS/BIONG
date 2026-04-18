# Action Center Design Note

## Goal

Separate behavioral arbitration from locomotor execution while supporting active sensing actions that change body orientation without displacement.

New pipeline:

`specialized modules or monolithic_policy -> learned constrained arbitration -> action_center decision -> motor_cortex correction -> embodiment-aware execution slip -> final action`

## Contracts

- `action_center`
  - input: concatenation of locomotion proposals plus `action_context`
  - output: locomotion logits over `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `STAY`, `ORIENT_UP`, `ORIENT_DOWN`, `ORIENT_LEFT`, `ORIENT_RIGHT`
  - value head: yes
  - explicit arbitration: `action_center` `action.selection` payload with a learned `arbitration_network` over valences `threat`, `hunger`, `sleep`, and `exploration`
  - the learned arbitration network consumes the same structured evidence used by the former fixed formulas, emits valence logits, per-module gate adjustments, and a scalar arbitration value estimate
  - responsibility: choose the intended locomotion direction under behavioral priorities

- `motor_cortex`
  - input: the locomotion intent chosen by `action_center` in one-hot form plus `motor_context`
  - output: corrective logits over the same locomotion and orientation space, conditioned on local embodiment constraints
  - value head: no
  - responsibility: convert the arbitrated intent into a motor-selected action before physical execution
  - execution note: the environment applies embodiment-aware slip after the motor-cortex decision, so the selected action can differ from the executed action

## Contexts

- `action_context`
  - broad bodily and situational state for arbitration: hunger, fatigue, health, pain, contact, food or shelter status, predator cues, day or night cycle, last movement, sleep debt, and shelter depth

- `motor_context`
  - version: 5
  - input dimension: 14
  - local execution or correction context: food or shelter status, predator cues, day or night cycle, last movement, shelter depth, body orientation, local terrain difficulty, fatigue, and momentum
  - ordered signals: `on_food`, `on_shelter`, `predator_visible`, `predator_certainty`, `day`, `night`, `last_move_dx`, `last_move_dy`, `shelter_role_level`, `heading_dx`, `heading_dy`, `terrain_difficulty`, `fatigue`, `momentum`
  - ownership note: `momentum` is execution-relevant body state for `motor_cortex`, not decision-priority context for `action_center`

## Decision Versus Execution

The model separates behavioral intent from physical outcome.

- `action_center` performs decision arbitration. It integrates proposer logits, context, reflex-modulated module outputs, and learned valence gates to choose an intended action.
- `motor_cortex` performs motor correction. It receives the `action_center` intent plus `motor_context` and can shift the selected action before execution.
- The world performs physical execution. It receives the post-`motor_cortex` action as `intended_action`, computes embodiment-aware execution difficulty, and may apply slip before setting `executed_action`.

Trace and `info` payloads preserve this distinction:

- `debug.action_center.selected_intent`: the arbitrated behavioral decision.
- `debug.motor_cortex.motor_selected_action`: the action selected after motor correction.
- `intended_action`: the action submitted to the world for execution.
- `executed_action`: the physical action actually applied after slip.
- `motor_noise_applied` or `execution_slip_occurred`: whether physical execution changed the intended action.

## Embodiment-Aware Execution

The old flat random action flip has been replaced by an execution slip model applied after `motor_cortex`.

Execution difficulty is computed from body orientation, terrain, fatigue, and
bounded execution momentum:

1. The intended movement vector is compared with the spider's body heading.
2. The normalized dot product is mapped into `orientation_alignment` in `[0, 1]`, where `1` is aligned and `0` is opposite.
3. Terrain under the spider is mapped to `terrain_difficulty`: `OPEN -> 0.0`, `CLUTTER -> 0.4`, `NARROW -> 0.7`.
4. Fatigue is read from body state as a `[0, 1]` multiplier.
5. Base execution difficulty is `terrain_difficulty * (1 - orientation_alignment) * (1 + fatigue)`.
6. When `orientation_alignment > 0.7`, bounded `momentum` reduces difficulty with `difficulty *= (1.0 - momentum * 0.5)`.
7. The final difficulty is clipped to `[0, 1]`.

The stateful transition model is specified in `docs/embodiment_design.md`:
aligned movement builds bounded execution momentum, turns and rest decay or
reset it, and the action space remains unchanged.

Sharp heading changes also feed the physiology system. Heading changes of
90 degrees or greater add `turn_fatigue_cost`, 180-degree reversals add
`reverse_fatigue_cost`, and `ORIENT_*` actions pay those costs when they
produce the same large heading changes.

Slip probability uses `action_flip_prob` as a base slip rate and adds embodiment pressure from `orientation_slip_factor`, `terrain_slip_factor`, and `fatigue_slip_factor`. These factors live in `NoiseConfig.motor`, so preset noise profiles can scale physical unreliability without changing the action space.

`ORIENT_*` actions rotate the body heading to the named cardinal direction, do not displace the spider, and bypass motor slip. They still advance the tick and apply the idle physiology increments for hunger and fatigue.

When slip occurs, the executed action is biased toward `STAY` to model momentum loss. Movement slips can also deviate into adjacent directions: vertical movement can slip left or right, and horizontal movement can slip up or down. The model avoids uniform random substitution and avoids reversing the intended direction.

Motor execution diagnostics are exported through `info`, event logs, `BrainStep`, trace `debug.motor_cortex`, `EpisodeStats`, and aggregate summaries. The main fields are `orientation_alignment`, `terrain_difficulty`, `execution_difficulty`, `slip_reason`, `motor_slip_rate`, `mean_orientation_alignment`, `mean_terrain_difficulty`, and per-terrain slip rates.

## Learned Arbitration Network

The arbitration stage keeps the old evidence extraction path and replaces only the fixed valence sum with a small learned controller.

- input: 24 scalars, concatenated as six threat signals, six hunger signals, six sleep signals, and six exploration signals
- trunk: shared tanh hidden layer
- valence head: four logits for `threat`, `hunger`, `sleep`, and `exploration`
- gate head: six per-module multiplicative adjustment factors for `alert_center`, `hunger_center`, `sleep_center`, `visual_cortex`, `sensory_cortex`, and `monolithic_policy`
- value head: one scalar estimate used by TD learning

Gate adjustments are constrained before being multiplied by the fixed base gate. The default bounds are `[0.5, 1.5]`, which lets learning reduce or amplify a module's influence without completely silencing modules or creating unbounded amplification. `BrainAblationConfig.gate_adjustment_bounds` can widen or narrow this interval for named arbitration variants. The final module gate is still clamped to `[0, 1]`.

## Fixed Priors And Learned Outputs

The fixed formulas still compute interpretable threat, hunger, sleep, and exploration evidence. `ArbitrationNetwork` can be warm-started to approximate those formulas, so early behavior can remain close to the previous hand-authored arbitration. `BrainAblationConfig.warm_start_scale` controls how much of that prior is injected into the initial network: `1.0` applies the full warm start, `0.5` halves the copied trunk and valence-head weights, and `0.0` skips the warm start so the arbitration network starts from its random initialization.

At runtime:

1. evidence is extracted from the existing observation bindings
2. evidence values are concatenated in `ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES` order
3. learned valence logits are converted to probabilities with softmax
4. evaluation uses deterministic `VALENCE_ORDER` tie-breaking; training samples from the learned probabilities for exploration
5. `base_gates` come from `PRIORITY_GATING_WEIGHTS[winning_valence]`
6. `module_gates[module] = clamp(base_gates[module] * gate_adjustments[module], 0, 1)`

## Reduced-Steering Benchmark

The benchmark path is intended to measure the learned arbitration policy rather than hidden manual steering. For that reason, `BrainAblationConfig` defaults `enable_deterministic_guards` and `enable_food_direction_bias` to `false`. With those defaults, clear predator pressure and high hunger remain visible as evidence, but they do not override the learned valence selection. If the learned policy selects `hunger`, the final action logits are also left untouched by food-direction steering.

The previous hand-authored scaffolding is still available as an explicit ablation. When `enable_deterministic_guards=True`, clear predator pressure can force `threat`, and safe high hunger can force `hunger`. When `enable_food_direction_bias=True`, deterministic evaluation can add a bounded `+3.0` logit bonus to the action facing the best available food cue, selected in visible, trace, memory, then smell order, when the cue direction magnitude exceeds `0.05`. These switches are diagnostic tools and regression baselines, not the default benchmark behavior.

## Scaffold Classification for Claim Tests

Claim-test reports attach a scaffold classification derived from the ablation config summary and the observed evaluation reflex scale. This taxonomy is reporting-only: it does not change training, simulation rollout behavior, arbitration behavior, or checkpoint selection. It only changes how claim-test outputs are labeled, qualified, or failed in reporting.

The claim suite uses three support levels:

- `MINIMAL_MANUAL` (`minimal_manual`): no deterministic guards, no food-direction bias, `eval_reflex_scale` is absent or `<= 0.0`, `warm_start_scale <= 0.25`, and `use_learned_arbitration=True`.
- `STANDARD_CONSTRAINED` (`standard_constrained`): no hard runtime scaffolds are active, but either `warm_start_scale > 0.25` or `use_learned_arbitration=False`.
- `SCAFFOLDED_RUNTIME` (`scaffolded_runtime`): any of `enable_deterministic_guards=True`, `enable_food_direction_bias=True`, or `eval_reflex_scale > 0.0`.

Primary claims use that level to decide how a passing metric result is reported:

| Support level | Primary-claim reporting outcome |
| --- | --- |
| `MINIMAL_MANUAL` | Keep `passed`, set `claim_severity="full"`, and mark `benchmark_of_record_eligible=true`. |
| `STANDARD_CONSTRAINED` | Keep `passed`, set `claim_severity="qualified"`, and mark `benchmark_of_record_eligible=false`. |
| `SCAFFOLDED_RUNTIME` | Convert a metric pass into claim-test `failed`, set `claim_severity="non_benchmark"`, and emit an explicit scaffold-dependence reason. |

Non-primary claims do not change pass/fail status based on scaffold level. They only inherit the reporting severity that matches the detected level.

The finding labels are computed from the following exact conditions:

| Input field | Exact condition | Finding label |
| --- | --- | --- |
| `enable_deterministic_guards` | `True` | `deterministic_guards_enabled` |
| `enable_food_direction_bias` | `True` | `food_direction_bias_enabled` |
| `warm_start_scale` | `> 0.25` | `warm_start_prior_active` |
| `use_learned_arbitration` | `False` | `fixed_arbitration_runtime` |
| `eval_reflex_scale` | `> 0.0` | `reflex_support_at_eval` |

These labels surface residual scaffolding explicitly in claim rows, summaries, and offline exports without introducing alternate runtime paths.

## Reflexes

- Local reflexes remain in the proposer modules.
- They modulate proposals before `action_center`.
- After that, `action_center` applies module gates according to the winning valence and learned per-module adjustment factors.
- Gate adjustments use `BrainAblationConfig.gate_adjustment_bounds`, with `[0.5, 1.5]` as the default, before being multiplied by the interpretable base gate table and clamped to `[0, 1]`. This lets learning reduce or amplify module influence without allowing complete learned silence or unbounded amplification in the default benchmark.
- `arbitration_regularization_weight` keeps learned gates close to the fixed baseline early in training; it can be annealed downward over training to progressively allow more deviation from the fixed baseline.
- `motor_cortex` does not apply reflexes; it only corrects execution of the arbitrated intent.

## Regularization

The learned gate gradient combines task credit from `action_center` with a prior-preserving term:

`arbitration_regularization_weight * (learned_gate - base_gate)`

This keeps the learned adjustment near the fixed gate table while the controller is immature. The weight is intended to be annealed downward over longer training runs, gradually allowing more deviation from the fixed baseline once behavior is stable.

An optional valence regularizer pulls learned valence probabilities toward the fixed-formula distribution. It can be disabled independently, and the `learned_arbitration_no_regularization` ablation disables both gate and valence regularization.

## Auditability

- The bus or trace serializes 22 top-level fields: `strategy`, `winning_valence`, `valence_scores`, `valence_logits`, `base_gates`, `gate_adjustments`, `module_gates`, `arbitration_value`, `learned_adjustment`, `guards_applied`, `food_bias_applied`, `food_bias_action`, `module_contribution_share`, `dominant_module`, `dominant_module_share`, `effective_module_count`, `module_agreement_rate`, `module_disagreement_rate`, `suppressed_modules`, `evidence`, `intent_before_gating`, and `intent_after_gating`.
- `ModuleResult` also carries `valence_role`, `gate_weight`, `gated_logits`, `intent_before_gating`, and `intent_after_gating` for explicit debugging of each gate's effect.
- The `food_vs_predator_conflict` and `sleep_vs_exploration_conflict` scenarios exercise this arbitration deterministically.

`ArbitrationDecision` fields should be interpreted as follows:

- `valence_logits`: raw learned scores before softmax
- `valence_scores`: softmax probabilities used for selection
- `base_gates`: fixed-prior gates for the selected valence
- `gate_adjustments`: learned multiplicative factors constrained by `gate_adjustment_bounds`, defaulting to `[0.5, 1.5]`
- `module_gates`: final clamped gates applied to module logits
- `arbitration_value`: value estimate produced by the arbitration network
- `learned_adjustment`: `true` when the learned network path is active, `false` for the fixed baseline
- `guards_applied`: `true` only when the deterministic guard ablation overrode the learned valence selection
- `food_bias_applied`: `true` only when the food-direction bias ablation added a final action-logit bonus
- `food_bias_action`: the action that received the food-direction bonus, or `null` when no bias was applied

Trace exports keep the `action.selection` payload backward compatible with existing conflict scorers. In particular, `winning_valence` and `module_gates` remain top-level payload fields, and threat evidence exports a legacy `predator_proximity` key for scorer compatibility even though learned arbitration no longer consumes that distance signal. The newer `guards_applied`, `food_bias_applied`, and `food_bias_action` fields make manual scaffolding explicit when it is enabled, so interpretability comes from diagnostics rather than silent steering.

To inspect learned versus baseline behavior, run the same scenario across the arbitration variants, export traces, and compare `valence_logits`, `base_gates`, `gate_adjustments`, `module_gates`, `guards_applied`, `food_bias_applied`, and scorer metrics such as `threat_priority_rate` or `sleep_priority_rate`.

## Ablations

- `constrained_arbitration`: legacy regression baseline. It enables deterministic guards and food-direction bias, uses `warm_start_scale=1.0`, and keeps gate adjustment bounds at `(0.5, 1.5)`.
- `weaker_prior_arbitration`: new benchmark-of-record arbitration setting. It disables deterministic guards and food-direction bias, uses `warm_start_scale=0.5`, and keeps gate adjustment bounds at `(0.5, 1.5)`.
- `minimal_arbitration`: maximum learned-autonomy setting. It disables deterministic guards and food-direction bias, skips warm start with `warm_start_scale=0.0`, and widens gate adjustment bounds to `(0.1, 2.0)`.
- `fixed_arbitration_baseline`: disables the learned arbitration path and uses the fixed formulas plus unadjusted base gates.
- `learned_arbitration_no_regularization`: keeps learned valence and gate heads active but sets arbitration regularization weights to zero.

`BrainAblationConfig.use_learned_arbitration` defaults to `true` and can be set to `false` for any modular run that needs fixed-formula arbitration. `enable_deterministic_guards` and `enable_food_direction_bias` also default to `false`, so manual steering is opt-in through ablation configuration rather than implicit in benchmark runs.

## Monolithic Policy

- `monolithic_policy` remains the single-proposal baseline.
- It now feeds the same `action_center` and the same `motor_cortex` used by the modular architecture.
- That preserves structural comparability downstream of the proposal stage.

## Compatibility

- `architecture_signature()` now includes top-level `action_center` and `motor_cortex` sections, a `contexts` object containing `action_context` and `motor_context` summaries, and an `arbitration_network` section with learned/fixed mode, evidence input size, hidden size, regularization weight, and parameter-layout fingerprint.
- `ARCHITECTURE_VERSION` was incremented.
- Checkpoints save `arbitration_network.npz`; missing arbitration weights fail with an explicit message. This change does not provide automatic migration.
