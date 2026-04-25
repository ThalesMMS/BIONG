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

Canonical summaries now include `architecture_description` so the comparison output distinguishes:

- `true monolithic direct control`
- `monolithic proposer + action/motor pipeline`
- `full modular with arbitration`

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

- `report.md`
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
