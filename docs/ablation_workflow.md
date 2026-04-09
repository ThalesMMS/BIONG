# Ablation Workflow

This repository exposes a reproducible ablation suite for comparing the current modular architecture against structural variants and a comparable monolithic baseline.

## Canonical Variants

- `modular_full`: full modular architecture with `module_dropout`, local reflexes, and auxiliary targets
- `no_module_dropout`: keeps the modular architecture but removes inter-module dropout
- `no_module_reflexes`: keeps the modules but disables both reflex-logit injection and the auxiliary targets derived from reflexes
- `reflex_scale_0_25`, `reflex_scale_0_50`, `reflex_scale_0_75`: keep the modular architecture but progressively reduce global reflex strength
- `drop_visual_cortex`, `drop_sensory_cortex`, `drop_hunger_center`, `drop_sleep_center`, `drop_alert_center`: remove one proposer module at a time from locomotion proposal and learning
- `local_credit_only`: preserves the current modular inference path but removes the broadcast of global policy gradients across proposer modules during training
- `monolithic_policy`: concatenates the modular observations into a single vector and runs a single proposer before the same `action_center` and `motor_cortex` used by the modular architecture

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

## Main Commands

Run the full canonical suite and export `summary` plus CSV:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --summary spider_ablation_summary.json \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary
```

Run a single variant against the modular reference in one specific scenario:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-variant monolithic_policy \
  --behavior-scenario night_rest \
  --full-summary
```

Run a custom reflex schedule and record the post-training comparison with `eval_reflex_scale=0.0`:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --episodes 12 \
  --eval-episodes 2 \
  --reflex-scale 1.0 \
  --module-reflex-scale alert_center=0.5 \
  --reflex-anneal-final-scale 0.25 \
  --summary spider_reflex_schedule_summary.json \
  --full-summary
```

## Learning Evidence

The repository also exposes a separate workflow for testing whether a trained checkpoint outperforms suitable controls without relying only on qualitative narrative.

Smoke command:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --behavior-scenario night_rest \
  --full-summary
```

Canonical short-vs-long command:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --learning-evidence-long-budget-profile report \
  --behavior-suite \
  --summary spider_learning_evidence_summary.json \
  --behavior-csv spider_learning_evidence_rows.csv \
  --full-summary
```

Canonical conditions:

- `trained_final`
- `trained_without_reflex_support`
- `random_init`
- `reflex_only`
- `freeze_half_budget`
- `trained_long_budget`

Practical reading:

- `summary["behavior_evaluation"]["learning_evidence"]["reference_condition"]` is `trained_final`
- `evidence_summary["has_learning_evidence"]` uses only `scenario_success_rate` as the primary gate
- `trained_without_reflex_support` does not participate in the gate; it measures residual reflex dependence
- The exported CSV includes `learning_evidence_condition`, `learning_evidence_policy_mode`, `learning_evidence_train_episodes`, `learning_evidence_frozen_after_episode`, `learning_evidence_checkpoint_source`, `learning_evidence_budget_profile`, and `learning_evidence_budget_benchmark_strength`

## Reward-Shaping Audit

Beyond structural ablations, the repository also exposes a reproducible readout of reward shaping and observation leakage.

Smoke command:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
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
- `summary["reward_audit"]["reward_profiles"]` compares `classic`, `ecological`, and `austere` by shaping category
- `summary["behavior_evaluation"]["shaping_audit"]["comparison"]["deltas_vs_minimal"]` uses `austere` as the comparison baseline
- `predator_dist`, `home_vector`, and `shelter_memory` appear explicitly as higher-risk signals because they depend on world-owned derivation

Profile intent:

- `classic`: denser baseline with stronger progress shaping
- `ecological`: reduces part of the guidance while keeping explicit scaffolding
- `austere`: zeros the main progress terms (`food_progress`, `shelter_progress`, `threat_shelter_progress`, `predator_escape`, `day_exploration`) to act as a stricter contrast

## Offline Post-Processing

After generating `summary`, `trace`, and `behavior_csv`, the offline runner can assemble a comparable package without notebooks or manual JSON inspection.

Example:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --summary spider_ablation_summary.json \
  --trace spider_ablation_trace.jsonl \
  --debug-trace \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary

PYTHONPATH=. python3 -m spider_cortex_sim.offline_analysis \
  --summary spider_ablation_summary.json \
  --trace spider_ablation_trace.jsonl \
  --behavior-csv spider_ablation_rows.csv \
  --output-dir spider_ablation_report
```

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
- `scenario_success.svg` and `scenario_checks.csv` help localize regressions by scenario and check without opening the full JSON
- `reward_components.csv` cross-references reward components aggregated from the summary with observed totals from the trace
- `trace[*].event_log` records the explicit tick order, making causal inspection possible without reconstructing implicit mutations
- `reflex_frequency.svg` uses `trace.messages[*].payload.reflex` as its source and becomes richer when the trace was generated with `--debug-trace`

If a block is missing, the runner does not fail. It records the absence in `report.md` and `report.json`.

## Workflow Notes

- If no `--behavior-scenario` is provided alongside `--ablation-suite` or `--ablation-variant`, the CLI automatically uses the full behavioral suite
- `summary["behavior_evaluation"]["ablations"]` contains the reference used, the evaluated variants, and the deltas against `modular_full`
- Each variant also publishes `without_reflex_support`, which re-evaluates the same checkpoint with `eval_reflex_scale=0.0`
- The exported CSV includes `ablation_variant`, `ablation_architecture`, `reflex_scale`, `reflex_anneal_final_scale`, `eval_reflex_scale`, `budget_profile`, `benchmark_strength`, `checkpoint_source`, `operational_profile`, `operational_profile_version`, and `noise_profile`
- `summary["config"]["operational_profile"]` records the active operational profile name, version, and effective payload; the CLI can pin it with `--operational-profile default_v1`
- `summary["config"]["noise_profile"]` records the active noise profile name and effective payload; the CLI can pin it with `--noise-profile none|low|medium|high`
- `summary["config"]["budget"]` records the resolved profile (`smoke`, `dev`, `report`, or `custom`), benchmark strength, comparison seeds, and any manual overrides
- `--checkpoint-selection best` evaluates intermediate checkpoints with the behavioral suite, chooses the best by `scenario_success_rate`, and persists only `best/` and `last/` when `--checkpoint-dir` is provided

## Gradual Reflex Sweep

The ablation workflow can measure reflex dominance gradually, not only as a binary switch.

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

The enriched `action_center` trace also serializes:

- `module_contribution_share`
- `dominant_module`
- `dominant_module_share`
- `effective_module_count`
- `module_agreement_rate`
- `module_disagreement_rate`

## Optional Curriculum

The main runner also accepts `--curriculum-profile ecological_v1` to organize training into four reproducible phases before final evaluation.

Short example:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --curriculum-profile ecological_v1 \
  --behavior-csv spider_curriculum_rows.csv \
  --full-summary
```

Practical reading:

- `summary["config"]["training_regime"]` describes the active regime and resolved budget
- `summary["curriculum"]` lists phases, per-phase scenarios, promotion checks, and final status
- `summary["behavior_evaluation"]["curriculum_comparison"]` compares `flat` versus `curriculum` under the same seeds and highlights `open_field_foraging`, `corridor_gauntlet`, `exposed_day_foraging`, and `food_deprivation`
- The exported CSV includes `training_regime`, `curriculum_profile`, `curriculum_phase`, and `curriculum_phase_status`

## Recommended Budgets

- `smoke`: sanity or CI profile with `6` episodes, `1` evaluation run, `60` steps, and seed `7`
- `dev`: fast local benchmark with `12` episodes, `2` evaluation runs, `90` steps, `scenario_episodes=1`, and seeds `7/17/29`
- `report`: stronger benchmark with `24` episodes, `4` evaluation runs, `120` steps, `scenario_episodes=2`, and seeds `7/17/29/41/53`

Canonical commands:

```bash
# sanity / CI
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --ablation-variant monolithic_policy \
  --behavior-scenario night_rest \
  --full-summary

# fast local benchmark
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --full-summary

# stronger benchmark with checkpoint selection
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile report \
  --checkpoint-selection best \
  --ablation-suite \
  --full-summary
```

## Canonical Check-In Result

Command used to generate the table below:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
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

| variant | architecture | suite success | episode success | delta suite vs ref | delta episode vs ref |
| --- | --- | ---: | ---: | ---: | ---: |
| `modular_full` | `modular` | 0.30 | 0.37 | +0.00 | +0.00 |
| `no_module_dropout` | `modular` | 0.30 | 0.47 | +0.00 | +0.10 |
| `no_module_reflexes` | `modular` | 0.10 | 0.33 | -0.20 | -0.03 |
| `reflex_scale_0_25` | `modular` | 0.10 | 0.30 | -0.20 | -0.07 |
| `reflex_scale_0_50` | `modular` | 0.30 | 0.37 | +0.00 | +0.00 |
| `reflex_scale_0_75` | `modular` | 0.10 | 0.30 | -0.20 | -0.07 |
| `drop_visual_cortex` | `modular` | 0.40 | 0.47 | +0.10 | +0.10 |
| `drop_sensory_cortex` | `modular` | 0.30 | 0.50 | +0.00 | +0.13 |
| `drop_hunger_center` | `modular` | 0.40 | 0.50 | +0.10 | +0.13 |
| `drop_sleep_center` | `modular` | 0.10 | 0.30 | -0.20 | -0.07 |
| `drop_alert_center` | `modular` | 0.10 | 0.37 | -0.20 | +0.00 |
| `local_credit_only` | `modular` | 0.10 | 0.30 | -0.20 | -0.07 |
| `monolithic_policy` | `monolithic` | 0.10 | 0.30 | -0.20 | -0.07 |

### Per-Scenario Comparison

Short legend: `full=modular_full`, `no_drop=no_module_dropout`, `no_reflex=no_module_reflexes`, `local=local_credit_only`, `r025=reflex_scale_0_25`, `r050=reflex_scale_0_50`, `r075=reflex_scale_0_75`, `d_visual=drop_visual_cortex`, `d_sensory=drop_sensory_cortex`, `d_hunger=drop_hunger_center`, `d_sleep=drop_sleep_center`, `d_alert=drop_alert_center`, `mono=monolithic_policy`.

| scenario | full | no_drop | no_reflex | r025 | r050 | r075 | d_visual | d_sensory | d_hunger | d_sleep | d_alert | mono |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `night_rest` | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 | 0.67 | 0.00 |
| `predator_edge` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `entrance_ambush` | 0.33 | 1.00 | 0.67 | 0.67 | 1.00 | 0.33 | 0.67 | 0.67 | 1.00 | 0.33 | 0.33 | 0.67 |
| `open_field_foraging` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `shelter_blockade` | 0.33 | 0.67 | 0.67 | 0.33 | 0.67 | 0.33 | 1.00 | 1.00 | 1.00 | 0.33 | 0.33 | 0.33 |
| `recover_after_failed_chase` | 1.00 | 0.67 | 0.67 | 0.67 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 0.67 | 0.67 |
| `corridor_gauntlet` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `two_shelter_tradeoff` | 1.00 | 1.00 | 0.33 | 0.33 | 0.00 | 0.67 | 1.00 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 |
| `exposed_day_foraging` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `food_deprivation` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Quick Interpretation

- `local_credit_only` separates topology from credit assignment: it keeps current modular inference but restricts training to local credit by `action_center` input slice.
- The `reflex_scale_*` sweep lets you see whether the policy degrades smoothly, collapses early, or remains dependent on reflex overrides after training.
- In traces and debug output, the correct read order is now: modular or monolithic proposals -> valence-based priority gating -> `action_center` -> `motor_cortex` correction -> optional `final_reflex_override` -> final action.
- When the trace is enriched, use `winning_valence`, `module_gates`, `suppressed_modules`, `module_contribution_share`, and `dominant_module` to interpret conflicts and coadaptation without manual inference.
- Gains are not uniformly positive for every form of modularity: with `12` training episodes, some removal ablations (`drop_visual_cortex`, `drop_hunger_center`) score better on the aggregate benchmark. That is a useful result, not a documentation problem.
- The per-scenario table shows where the benchmark is still weak: `open_field_foraging`, `corridor_gauntlet`, `exposed_day_foraging`, and `food_deprivation` remain at `0.00` for every variant under this short configuration. They remain good candidates for increased training budget or revised checks.

## Reading the `0.00` Cases

Those four weak-signal scenarios now include `diagnostics`, `progress_band`, and `outcome_band` so every `0.00` is not interpreted as the same failure.

- `open_field_foraging`: `progress_band=regressed` indicates movement away from food; `outcome_band=regressed_and_died` separates that from simple stalling
- `corridor_gauntlet`: when `corridor_avoids_contact` passes but the scenario still scores `0.00`, `outcome_band` distinguishes safe stalling from death after partial progress
- `exposed_day_foraging`: `progress_band=regressed` usually indicates retreat or defensive movement that never turned into food progress
- `food_deprivation`: `approaches_food` can pass while the scenario still remains `0.00`; that now appears as `partial_progress_died`, which signals real approach without enough homeostatic recovery

In practice:

- `suite[scenario]["diagnostics"]["primary_outcome"]` summarizes the dominant outcome
- `suite[scenario]["diagnostics"]["outcome_distribution"]` shows the mix of episode outcomes
- `partial_progress_rate` helps separate a poorly calibrated scenario from one that truly has no useful signal
- In the CSV, use `scenario_focus`, `metric_progress_band`, and `metric_outcome_band` to filter these cases without opening the full JSON
