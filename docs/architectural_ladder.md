# Architectural Ladder Validation Protocol

## Goal

This document defines the official diagnostic ladder for validating progressive modularity before adding more centers or cortices.

The ladder is neither a claim suite nor a request to keep increasing module count. Its purpose is narrower:

- verify that a simpler architecture already learns the target behavior
- localize where performance breaks as modularity is introduced
- prevent "add more modules" from masking failures in reward design, interfaces, connectivity, or credit assignment

The initial goal is therefore not "more modules". The initial goal is to validate that a small set of modules can learn and stay well connected under the current benchmark.

## Scope And Status

The ladder defines six architectural rungs:

- `A0_true_monolithic`
- `A1_monolithic_with_action_motor`
- `A2_three_center`
- `A3_four_center`
- `A4_current_full_modular`
- `A5_future_bio_refined`

Status in the current repository:

- `A0_true_monolithic` maps directly to the existing `true_monolithic_policy` ablation
- `A1_monolithic_with_action_motor` maps directly to the existing `monolithic_policy` ablation
- `A2_three_center` maps directly to the existing `three_center_modular` ablation
- `A3_four_center` maps directly to the existing `four_center_modular` ablation
- `A4_current_full_modular` maps directly to the existing `modular_full` reference
- `A5_future_bio_refined` remains a protocol-defined target for future architecture comparisons rather than a current canonical CLI variant

That distinction matters. The ladder is official as a validation protocol now, even where some intermediate rungs still need implementation work later.

For the detailed ablation-variant definitions, competence-gap reporting, shaping-gap reads, and benchmark-of-record claim procedures, use [ablation_workflow.md](./ablation_workflow.md). The ladder defined here is the diagnostic framework for deciding how to interpret those comparisons across progressively modular architectures.

## Ladder Overview

```text
A0_true_monolithic
  observations -> single end-to-end policy -> action

A1_monolithic_with_action_motor
  shared observations -> monolithic_policy -> action_center -> motor_cortex -> action

A2_three_center
  split observations -> 3 proposer modules -> action_center -> motor_cortex -> action

A3_four_center
  split observations -> 4 proposer modules -> action_center -> motor_cortex -> action

A4_current_full_modular
  split observations -> 5 proposer modules -> action_center -> motor_cortex -> action

A5_future_bio_refined
  split observations -> refined or extra specialized modules -> action_center -> motor_cortex -> action
```

Interpretation rule:

- move up the ladder only after the lower rung is at least stable on its own target gates
- if a higher rung regresses, diagnose the added structural change before adding another rung

## Ladder Rung Definitions

| Rung | Description |
| --- | --- |
| `A0_true_monolithic` | Direct end-to-end policy with no proposer split and no downstream action or motor staging. |
| `A1_monolithic_with_action_motor` | Monolithic proposer with the existing `action_center` and `motor_cortex` downstream pipeline. |
| `A2_three_center` | Three coarse proposer centers: `perception_center`, `homeostasis_center`, and `threat_center`. |
| `A3_four_center` | Four proposer centers splitting perception into visual and olfactory while maintaining unified homeostasis and threat pathways. |
| `A4_current_full_modular` | Full current modular split with separate visual, sensory, hunger, sleep, and alert proposers. |
| `A5_future_bio_refined` | Future biologically refined or further specialized proposer decomposition beyond the current modular split. |

## Rungs

### `A0_true_monolithic`

Single end-to-end policy with no proposer split and no `action_center` or `motor_cortex`.

Primary question:

- does the policy learn at all?

What this rung isolates:

- baseline capacity of the learning algorithm
- basic reward viability
- gross training instability

Failure interpretation:

- if `A0` cannot learn, do not blame modularity
- first suspect reward design, optimization, budget, observation adequacy, or a general capacity mismatch

### `A1_monolithic_with_action_motor`

Single proposal network followed by the existing `action_center` and `motor_cortex` pipeline. This is the current `monolithic_policy` baseline at the implementation level, and the ladder reports it as `A1_monolithic_with_action_motor`.

Primary questions:

- does the policy still learn once the post-proposal pipeline is introduced?
- does `action_center` plus `motor_cortex` preserve or improve a learnable control path?

What this rung isolates:

- the cost or benefit of inserting learned arbitration and execution stages downstream of a single proposer
- whether failures come from the new downstream interface rather than from modular proposal splitting

Failure interpretation:

- if `A0` passes but `A1` fails, the first suspicion is the `action_center` or `motor_cortex` path, not proposer modularity
- likely culprits are action-context sufficiency, motor-context sufficiency, arbitration constraints, or execution-stage interference

### `A2_three_center`

Three proposer modules plus `action_center` and `motor_cortex`. The current implementation alias is `three_center_modular`, using the coarse `perception_center`, `homeostasis_center`, and `threat_center` split as the first modular rung. Credit-assignment comparisons at this rung use `three_center_modular`, `three_center_modular_local_credit`, and `three_center_modular_counterfactual`.

Primary questions:

- does the initial module split of 3 coarse centers preserve learning compared to the monolithic rung?
- is the proposer interface sufficient for useful specialization without over-fragmenting the policy?
- can coarse modules receive and use credit well enough to justify the first split?

What this rung isolates:

- whether the first act of modularization helps or harms learning
- whether the split exposes weak module interfaces, missing cross-module information, or coarse-module credit problems

Promotion gate to `A3`:

- preserve the inherited `A1` gates: `night_rest`, `predator_edge`, `entrance_ambush`, and `shelter_blockade`
- pass `two_shelter_tradeoff` as the first scenario where a coarse split should help rather than merely survive
- demonstrate benefit from visual and olfactory separation rather than only matching coarse `A2` behavior
- only promote to `A3` when these gates are stable enough that further failures are more likely to reflect added specialization than basic A2 instability

Failure interpretation:

- if `A1` passes but `A2` fails, first suspect interface design or credit assignment to the coarse modules
- examples: missing state in one module view, poor arbitration visibility into proposer intent, a split that creates artificial blindness, or reward credit that does not arrive usefully at `perception_center`, `homeostasis_center`, or `threat_center`

### `A3_four_center`

Four proposer modules plus `action_center` and `motor_cortex`, with perception split into `visual_cortex` and `sensory_cortex` while homeostasis and threat remain unified.

Primary questions:

- does adding another specialized stream improve behavior instead of fragmenting it?
- does credit still reach the right sub-policies after another split?

What this rung isolates:

- whether specialization is emerging rather than merely increasing coordination burden
- whether visual and olfactory separation yields usable specialization rather than duplicative or conflicting proposals
- whether broadcast, local, or counterfactual credit assumptions remain adequate at modest modularity

Promotion gate to `A4`:

- preserve the inherited `A2` gates and the `A3` conflict probes
- show that visual and olfactory separation provides a measurable benefit over `A2` rather than only adding coordination cost
- promote to `A4` when further decomposition into `hunger_center`, `sleep_center`, and `alert_center` is expected to add benefit beyond the unified `homeostasis_center`

Failure interpretation:

- if `A2` passes but `A3` fails, visual/olfactory separation adds coordination burden without commensurate specialization benefit
- first suspect credit assignment, arbitration coadaptation, or a connection topology that no longer carries enough shared context
- if conflict scenarios fail specifically here, prefer a credit or gating diagnosis before assuming the whole idea of modularity failed

### `A4_current_full_modular`

Current five-proposer production architecture, represented by `modular_full`.

Primary questions:

- does full modularity preserve or exceed the performance of the lower rungs?
- are all modules contributing in a way that justifies their existence?
- has granularity become excessive?

What this rung isolates:

- whether the current five-module split is the right operating point
- whether some modules are redundant, dominant to the point of collapse, or weakly integrated

Failure interpretation:

- if `A3` passes but `A4` fails, first suspect excessive granularity, over-fragmentation, or unnecessary interface boundaries
- a common corrective action is to merge responsibilities back together before adding any new center

### `A5_future_bio_refined`

Future extensions with more specialized centers, richer sensory splits, or finer bio-inspired subdivisions.

Primary questions:

- does extra biological refinement produce measurable benchmark value?
- is the extra granularity justified by performance, specialization, or interpretability gains?

What this rung isolates:

- whether future biological refinement is real progress or just complexity growth

Failure interpretation:

- if `A4` passes but `A5` fails, refinement is not yet justified
- the correct outcome may be to stop at `A4` or return to a simpler rung rather than forcing additional centers
- candidate signal specifications for blocked `A5` proposals live in [interfaces.md](./interfaces.md); use those interface notes only after the admission gates in this section are satisfied

#### A5 Admission Prerequisites

All of the following gates must pass before any `A5` module proposal is eligible for implementation or benchmark inclusion.

1. `A2` learnability
   `three_center_modular` must reach no-reflex austere `scenario_success_rate >= 0.50` on the austere gate package.
2. `A3` preservation
   `four_center_modular` must preserve or improve `A2` on inherited gates with delta `>= -0.05`.
3. `A4` non-collapse
   `modular_full` must not regress by more than `0.10` versus `A2` on shared gate scenarios.
4. local interface sufficiency
   Each existing proposer must pass its canonical module-local baseline through `evaluate_local_task()` or the equivalent `module_local_tasks.py` workflow.
5. credit interpretability
   `module_contribution_share` must remain differentiated on the relevant scenarios, with contribution variance `> 0.01` and no persistent single-module collapse masquerading as modularity.
6. ablation correspondence
   Existing `drop_<module>` variants must still produce the expected behavioral deficits on the scenarios they are supposed to own.

Failure interpretation:

- If any prerequisite fails, A5 modules are blocked; diagnose lower rungs first.

#### Module Admission Template

Every `A5` proposal must document the following fields before code is added:

- Signal Source: which existing world, memory, trace, or context signals the module reads
- Biological Rationale: why the split is biologically motivated without assuming that biological naming implies engineering value
- Experimental Hypothesis: the measurable claim the new module is expected to satisfy
- Target Scenarios: the specific scenarios that should improve if the hypothesis is correct
- Success Threshold: at least `Cohen's d >= 0.3` versus `A4_current_full_modular` on the target scenarios while inherited lower-rung gates are preserved
- Ablation Design: the exact `drop_<module>` or rung-comparison test that should selectively fail
- Coupling Risk Assessment: the most likely way the proposal duplicates an existing module or creates hidden interface dependence

Universal rules:

- No module may be added if its ablation test cannot distinguish it from an existing module.
- No module may be treated as an automatic improvement just because it is more biologically specific.
- `A5` depends on `A2`, `A3`, and `A4` remaining valid after the proposed change.
- Proposed `A5` modules default to `role="proposal"` and must declare one primary arbitration valence from `threat`, `hunger`, `sleep`, or `exploration`.
- Unless a separate interface RFC says otherwise, new `A5` proposers must emit the existing locomotion and orientation action space rather than inventing a new output contract.

#### Candidate Dossiers

These are blocked candidate proposals, not approved architecture changes. They exist to force each possible `A5` addition to carry a falsifiable hypothesis, a selective ablation, and an explicit coupling risk.

##### `olfactory_center`

- Status: `proposal`, blocked until all `A5` admission prerequisites pass
- Input: `predator_smell_strength`, `predator_smell_dx`, `predator_smell_dy`, `olfactory_predator_threat`, `dominant_predator_olfactory`, `predator_trace_dx`, `predator_trace_dy`, `predator_trace_strength`, `predator_memory_dx`, `predator_memory_dy`, `predator_memory_age`, and `night`
- Output: standard proposer logits over `MOVE_*`, `STAY`, and `ORIENT_*`
- Connection: split smell-led predator pursuit and avoidance signals out of the current `sensory_cortex` and `alert_center`, then feed `action_center` as `role="proposal"` with primary arbitration valence `threat`
- Experimental hypothesis: a dedicated smell-first threat proposer improves odor-led predator avoidance when direct vision is weak or absent, without reducing performance on visual-hunter scenarios
- Target scenarios: `olfactory_visual_dissociation`, `olfactory_ambush`, and `visual_olfactory_pincer`
- Ablation test: `drop_olfactory_center` should selectively reduce success on `olfactory_visual_dissociation` and `olfactory_ambush`, while `visual_hunter_open_field` should remain materially less affected than under `drop_visual_cortex`
- Coupling risk: high overlap with `sensory_cortex` and `alert_center`; if the ablation signature is indistinguishable from dropping either existing module, the proposal is redundant and should be rejected

##### `nociceptive_center`

- Status: `proposal`, blocked until all `A5` admission prerequisites pass
- Input: `recent_pain`, `recent_contact`, `health`, `predator_memory_dx`, `predator_memory_dy`, `predator_memory_age`, `escape_memory_dx`, `escape_memory_dy`, `escape_memory_age`, `on_shelter`, and `night`
- Output: standard proposer logits over `MOVE_*`, `STAY`, and `ORIENT_*`
- Connection: carve post-contact emergency retreat and damage recovery behavior out of `alert_center` into a dedicated proposer that feeds `action_center` with primary arbitration valence `threat`
- Experimental hypothesis: a pain-specialized proposer shortens post-hit escape latency and reduces repeated contact after damage, especially when the predator is no longer visible but the body state still indicates immediate danger
- Target scenarios: `pain_escape_latency`, `predator_edge`, and `entrance_ambush`
- Ablation test: `drop_nociceptive_center` should worsen `pain_escape_latency`, increase repeated-contact rate, or increase post-contact health loss while leaving smell-led detection tasks less affected than `drop_alert_center`
- Coupling risk: very high overlap with `alert_center`; if pain and contact already act only as generic threat evidence, the extra module will just duplicate gating rather than create a separable behavioral deficit

##### `arousal_center`

- Status: `proposal`, blocked until all `A5` admission prerequisites pass
- Input: `day`, `night`, `fatigue`, `sleep_debt`, `sleep_phase_level`, `rest_streak_norm`, `recent_pain`, `predator_motion_salience`, `heading_dx`, `heading_dy`, and `foveal_scan_age`
- Output: standard proposer logits over `MOVE_*`, `STAY`, and `ORIENT_*`
- Connection: add a wakefulness and vigilance proposer that biases scan initiation and state transitions, feeding `action_center` as `role="proposal"` with primary arbitration valence `exploration`
- Experimental hypothesis: a dedicated arousal proposer improves transitions between quiescence, active sensing, and renewed locomotion under changing light or distant threat pressure without stealing the core sheltering role from `sleep_center`
- Target scenarios: `arousal_state_transition`, `sleep_vs_exploration_conflict`, and `food_vs_predator_conflict`
- Ablation test: `drop_arousal_center` should selectively hurt `arousal_state_transition`, reduce timely `ORIENT_*` use, or delay scan refresh after a state change while leaving sustained sleep retention more affected by `drop_sleep_center`
- Coupling risk: high overlap with `sleep_center`, `visual_cortex`, and `action_center` arbitration itself; if the proposal only re-expresses existing valence switching, it should remain an arbitration issue rather than a new proposer module

##### `homeostasis_center`

- Status: control rung and refinement source, not an automatic `A5` addition
- Input: the current coarse `homeostasis_center` interface from `A2` and `A3`, including hunger, fatigue, health, food and shelter occupancy, day or night state, sleep-state signals, and food or shelter trace and memory fields
- Output: standard proposer logits over `MOVE_*`, `STAY`, and `ORIENT_*`
- Connection: keep `homeostasis_center` as the prerequisite coarse control for `A2` and `A3`; if `A4` or any proposed `A5` split regresses, the default corrective action is to merge back toward this unified proposer before adding further specialization
- Experimental hypothesis: if `hunger_center` plus `sleep_center` truly justify their split, they should beat or clearly preserve the unified `homeostasis_center` control on inherited gates; otherwise finer homeostatic segmentation is blocked
- Target scenarios: `two_shelter_tradeoff`, `night_rest`, `food_deprivation`, and `sleep_vs_exploration_conflict`
- Ablation test: use rung comparison rather than a new `drop_homeostasis_center`; if `A4_current_full_modular` or a future `A5` homeostatic refinement fails to beat the `A2` or `A3` control family on inherited gates, revert to the unified control and stop further splitting
- Coupling risk: any new module that still requires most of the coarse `homeostasis_center` signals without yielding a separable ablation is evidence that the split is premature, not that more modules are needed

#### Module Proposal Checklist

Copy this checklist into a proposal file under [module_proposals/](./module_proposals/) before admitting any new `A5` module.

- [ ] `A2` learnability
- [ ] `A3` preservation
- [ ] `A4` non-collapse
- [ ] local interface sufficiency
- [ ] credit interpretability
- [ ] ablation correspondence
- [ ] experimental hypothesis
- [ ] target scenarios
- [ ] success threshold
- [ ] coupling risk

Completed checklists should be stored in [module_proposals/](./module_proposals/).

## Diagnostic Questions By Rung

| Rung | Primary question | If this rung fails after the previous rung passed |
| --- | --- | --- |
| `A0_true_monolithic` | Does the policy learn at all? | Fundamental learning problem: reward, optimization, observation quality, or raw capacity. |
| `A1_monolithic_with_action_motor` | Does the `action_center` and `motor_cortex` pipeline preserve a learnable policy? | Downstream architecture issue: arbitration or execution interface is likely the problem. |
| `A2_three_center` | Does the initial module split of 3 coarse centers preserve learning compared to monolithic? | Interface design or credit assignment problem introduced by the first coarse modular split. |
| `A3_four_center` | Does splitting perception into visual and olfactory streams improve specialization without overwhelming coordination? | Loss of integration between visual and olfactory streams, competing drives, or impaired unified threat and homeostasis responses. |
| `A4_current_full_modular` | Does full modularity preserve performance and maintain healthy contribution balance? | Excessive granularity, redundancy, or over-fragmentation in the current modular split. |
| `A5_future_bio_refined` | Does additional refinement justify its complexity? | New refinement is not justified; complexity outpaced benefit. |

The ladder is intended to answer the practical question: is the observed failure caused by capacity, interface, connection, credit, reward, or excessive granularity?

## Benchmark Package Report

The benchmark package now includes a unified `Architectural Ladder Report` and a machine-readable `unified_ladder_report.json` artifact.

That report is meant to answer the narrow practical question:

- is modularization helping or hurting under the current benchmark?
- is the read confounded by capacity or interface differences?
- which experiments are still missing before claiming modular emergence?

The package-level conclusion is intentionally conservative and uses one of four labels:

- `modularity supported`
- `modularity inconclusive`
- `modularity currently harmful`
- `capacity/interface confounded`

## Capacity Sweep Addendum

Use capacity sweeps before concluding that a rung failed because of interface or credit design.

The repository now supports canonical capacity presets:

- `small`
- `current`
- `medium`
- `large`

Use `--capacity-profile small|current|medium|large` to run a single configuration at a specific hidden-size preset.

Use `--capacity-sweep` to run the capacity matrix across the currently implemented ladder rungs:

- `A0_true_monolithic`
- `A1_monolithic_with_action_motor`
- `A2_three_center`
- `A3_four_center`
- `A4_current_full_modular`

Capacity-sweep exports should be read with these default interpretations:

- performance improves with larger capacity: possible undercapacity
- performance stays flat across capacity: probable interface, connection, arbitration, reward, or credit problem

The benchmark package now records the selected capacity profile in the summary and includes a capacity-vs-performance curve plus parameter and approximate compute-cost traces for the sweep payload.

## Module-Local Sufficiency Testing

### Purpose

Module-local sufficiency tests isolate local interface sufficiency from organism-level integration failures. Their purpose is to answer a narrower question before blaming arbitration, reward competition, cross-module interference, or downstream execution: does this module receive enough information through its own interface to express the intended directional behavior, and can its own proposal network learn that behavior in isolation?

This makes the failure read cleaner. If a module fails in the full organism, the local task result helps separate "the interface does not expose what the module needs" from "the interface is present but the network still does not learn the local mapping."

### Diagnostic Outputs

Each local task reports one of three diagnostic outcomes:

- `interface_insufficient`: the required signals for the local task are absent from the module interface
- `network_insufficient`: the required signals are present, but the isolated proposal network still fails to learn the local task under the local training budget
- `passed`: the isolated proposal network already shows, or can learn, the expected local directional behavior

### Relationship To Ladder Rungs

These tests are pre-`A0` diagnostics. Run them before full-organism ladder evaluation when defining or revising a module interface, and rerun them before escalating an organism-level failure into a ladder-rung diagnosis.

They do not replace the integrated ladder comparisons. A module can pass its local sufficiency tasks and still fail at `A0` through `A5` because of reward design, arbitration, execution-path interference, cross-module competition, or credit assignment. The point is narrower: rule out obvious local interface and network bottlenecks before diagnosing higher-level integration failures.

### Interface Sufficiency Variants

Interface variants are the pre-`A0` diagnostic for deciding whether a module is failing because it lacks the right signals or because the broader organism stack is failing. They define a small level ladder per module, from minimal direct cues (`v1`) up to the canonical production interface (`v4`).

Use them to answer a specific question before architectural promotion work: what is the minimum viable signal set that still lets the module pass its local tasks?

Interpretation rule:

- if `v2` is sufficient and `v1` is not, the signals introduced at `v2` are the first critical additions
- if only `v4` is sufficient, the current local tasks still depend on canonical trace or memory structure and the interface cannot yet be simplified safely
- if `v4` is insufficient, do not move up the architectural ladder yet; fix local task learnability, hidden-size assumptions, or the interface contract first

### Implementation Reference

The current implementation lives in `spider_cortex_sim/module_local_tasks.py`.

### Example Workflow

Recommended workflow when a module underperforms in organism-level testing:

1. identify the failing module from the organism-level trace or benchmark
2. run variant sufficiency for that module from `spider_cortex_sim/module_local_tasks.py`
3. identify the minimum sufficient variant level, if any
4. confirm that the local tasks at that level are `passed`, not merely interface-covered
5. if no reduced variant is sufficient, keep the canonical interface and continue with ladder diagnostics only after local sufficiency is stable
6. if a reduced variant is sufficient, treat the signals first introduced at that level as the minimum viable additions before proceeding to the ladder
7. only then escalate to `A0` through `A5` if the likely bottleneck is now reward shaping, arbitration, execution, cross-module competition, or credit assignment

## Primary Metrics

Use the existing repository metrics and summaries rather than inventing new score surfaces.

### Benchmark Anchor

- no-reflex `scenario_success_rate` is the benchmark of record for ladder comparisons
- when available, read it from no-reflex rows where `eval_reflex_scale=0.0`

### Core Aggregate Metrics

| Diagnostic intent | Canonical field or reading |
| --- | --- |
| Survival rate | `evaluation.survival_rate` |
| Food eaten | `evaluation.mean_food` |
| Hunger reduction | scenario-owned `behavior_metrics.hunger_reduction` where applicable, especially `food_deprivation` |
| Night shelter occupancy | `evaluation.mean_night_shelter_occupancy_rate` and scenario-owned `behavior_metrics.night_shelter_occupancy_rate` |
| Predator response success | no-reflex predator-response `scenario_success_rate`; for claim-style summaries, `predator_response_scenario_success_rate` |
| Predator escape / response efficiency | `evaluation.mean_predator_escapes`, `evaluation.mean_predator_response_events`, `evaluation.mean_predator_response_latency` |
| Competence gap | `evaluation.competence_gap` with basis `scaffolded_minus_self_sufficient` |
| Shaping gap | `classic_minus_austere`, `ecological_minus_austere` |
| Module contribution | `evaluation.mean_module_contribution_share` and trace-level `module_contribution_share` |
| Module dominance | `evaluation.dominant_module_distribution`, `evaluation.mean_dominant_module_share`, trace-level `dominant_module` |
| Effective modular spread | `evaluation.mean_effective_module_count` and trace-level `effective_module_count` |

### Metric Relevance By Rung

| Rung | Primary metrics to read first |
| --- | --- |
| `A0_true_monolithic` | no-reflex `scenario_success_rate`, `evaluation.survival_rate`, `evaluation.mean_food`, `evaluation.mean_night_shelter_occupancy_rate`, predator-response success |
| `A1_monolithic_with_action_motor` | Same as `A0`, plus `evaluation.mean_predator_response_latency` to detect downstream execution or arbitration degradation |
| `A2_three_center` | `A1` metrics plus `evaluation.competence_gap`, `classic_minus_austere`, `ecological_minus_austere` |
| `A3_four_center` | `A2` metrics plus visual-vs-olfactory specialization reads, conflict-scenario readings, and credit-assignment ablations such as `four_center_modular_local_credit` and `four_center_modular_counterfactual` |
| `A4_current_full_modular` | `A3` metrics plus `evaluation.mean_module_contribution_share`, `evaluation.dominant_module_distribution`, `evaluation.mean_dominant_module_share`, `evaluation.mean_effective_module_count` |
| `A5_future_bio_refined` | Same as `A4`, with emphasis on whether new modules improve outcome metrics without collapsing contribution balance |

### Guardrails From Existing Workflow

The ladder reuses the existing shaping and benchmark guardrails documented elsewhere:

- austere survival rate should stay at or above `0.50`
- `classic_minus_austere` scenario success delta should stay at or below `0.20`
- `ecological_minus_austere` scenario success delta should stay at or below `0.15`
- reflex-on improvements remain diagnostic, not benchmark wins, until they survive no-reflex evaluation

## Scenario Roles

The ladder uses scenarios in three ways:

- gate scenarios decide whether a rung is stable enough to progress
- capability probes diagnose narrow skill boundaries without acting as promotion gates
- targeted arbitration probes localize interface or credit problems in conflict handling

### Gate Scenarios

These are the main promotion gates for ladder progress.

| Scenario | Why it is a gate | First rung where it should matter |
| --- | --- | --- |
| `night_rest` | Basic sheltering, sleep recovery, and stable nighttime occupancy | `A0` |
| `predator_edge` | Minimal threat detection and response under direct predator pressure | `A0` |
| `entrance_ambush` | Threat handling when the shelter boundary itself is contested | `A1` |
| `shelter_blockade` | Safety preservation or escape under constrained route pressure | `A1` |
| `two_shelter_tradeoff` | Trade-off between safety and goal progress | `A2` |
| `visual_olfactory_pincer` | Multi-predator specialization under competing sensory niches | `A4` |
| `olfactory_ambush` | Olfactory threat specialization | `A4` |
| `visual_hunter_open_field` | Visual threat specialization in exposed terrain | `A4` |
| `olfactory_visual_dissociation` | A5 placeholder gate for smell-led threat specialization versus visual specialization; defined but not activated until an A5 module proposal names it as a target | `A5` |
| `pain_escape_latency` | A5 placeholder gate for post-contact escape speed and pain-driven retreat; defined but not activated until an A5 module proposal names it as a target | `A5` |
| `arousal_state_transition` | A5 placeholder gate for wakefulness, scan initiation, and state-transition control; defined but not activated until an A5 module proposal names it as a target | `A5` |

### Capability Probes

These remain in the benchmark, but a `0.00` here is diagnostic evidence rather than automatic ladder failure.

| Scenario | Capability boundary it probes | Typical use in the ladder |
| --- | --- | --- |
| `open_field_foraging` | `food_vector_acquisition_exposed` | Read from `A0` onward as food-orientation evidence, not as a promotion gate |
| `corridor_gauntlet` | `corridor_navigation_under_threat` | Becomes more informative from `A2` onward when split policy coordination matters |
| `exposed_day_foraging` | `daytime_foraging_under_patrol` | Helps separate cautious arbitration from pure incapacity |
| `food_deprivation` | `hunger_driven_commitment` | Useful from `A2` onward to test whether hunger behavior survives sparse outcomes |
| `recover_after_failed_chase` | recovery after disruption, not just first response | Useful from `A2` onward as a capability and recovery probe rather than a hard gate |

### Targeted Arbitration Probes

These are best read as focused interface and credit diagnostics, not as global promotion gates.

| Scenario | What it isolates | First rung where it becomes important |
| --- | --- | --- |
| `food_vs_predator_conflict` | whether threat correctly overrides foraging when both are active | `A3` |
| `sleep_vs_exploration_conflict` | whether sleep correctly suppresses residual exploration in a safe night context | `A3` |

Using that split keeps the ladder aligned with existing repository semantics:

- emergence gates remain the main pass or fail surfaces
- capability probes expose calibration and boundary failures
- conflict scenarios localize arbitration failures without pretending to represent the whole ecology

## Recommended Promotion Logic

Use the ladder progressively instead of evaluating every rung against every scenario from the start.

| Rung | Minimum scenario emphasis |
| --- | --- |
| `A0_true_monolithic` | `night_rest`, `predator_edge`, with capability probes used only for basic diagnosis |
| `A1_monolithic_with_action_motor` | `night_rest`, `predator_edge`, `entrance_ambush`, `shelter_blockade` |
| `A2_three_center` | `A1` gates plus `two_shelter_tradeoff`; promote to `A3` only after those gates are stable under `three_center_modular` |
| `A3_four_center` | `A2` coverage plus `food_vs_predator_conflict` and `sleep_vs_exploration_conflict` |
| `A4_current_full_modular` | `A3` coverage plus multi-predator specialization gates |
| `A5_future_bio_refined` | Same as `A4`, but promotion requires clear value over `A4`, not parity |

Practical rule:

- a higher rung should beat or at least preserve the lower rung on its inherited gates before new specialized probes are used as justification for extra complexity

## How To Interpret Failure

Use the first failing rung to localize the likely source of the problem.

| First failing transition | Default diagnosis |
| --- | --- |
| `A0 -> fail` | Learning stack or reward problem before architecture discussion |
| `A0 pass`, `A1 fail` | `action_center` or `motor_cortex` interface problem |
| `A1 pass`, `A2 fail` | First modular split introduced a bad interface or weak connection topology |
| `A2 pass`, `A3 fail` | Credit assignment or coordination burden increased faster than specialization benefit |
| `A3 pass`, `A4 fail` | Current modular split is too granular or poorly integrated |
| `A4 pass`, `A5 fail` | Further biological refinement is not yet justified |

Then refine the diagnosis with the secondary signals:

- large competence gap means scaffold dependence is still carrying performance
- large shaping gap means reward support is still doing too much of the work
- weak module-contribution spread or single-module dominance means nominal modularity may have collapsed into de facto monolithic behavior
- slower predator-response latency with stable survival can indicate viable but poorly connected arbitration or execution
- capability probes with partial progress can indicate calibration problems rather than total incapacity

## Development Workflow Vs. Benchmark Claims

Development is iterative. The ladder is a development protocol first.

Use it during development with fast local runs such as `--budget-profile dev` to answer questions like:

- is the new failure caused by modular splitting or by the downstream arbitration stack?
- did adding a module actually help on inherited gates?
- are we seeing a reward leak, a credit problem, or simple over-fragmentation?

Strong claims should not be made from those short iterative runs.

For strong claims:

- use `--budget-profile paper` or the benchmark-of-record workflow
- use `--ladder-reward-profiles` when the claim is specifically about how A0-A4 behaves under `classic`, `ecological`, and `austere`
- keep `--checkpoint-selection best` on for the ladder cross-profile benchmark package
- keep no-reflex `scenario_success_rate` as the primary surface
- interpret scaffold severity using the current claim taxonomy from [action_center_design.md](./action_center_design.md)

The benchmark package now exports this cross-profile ladder readout as `ladder_profile_comparison.json` and mirrors it in the Markdown report. The key interpretation split is:

- `shaping_dependent`: the rung looks acceptable in denser profiles but loses too much no-reflex success under `austere`
- `austere_survivor`: the rung keeps no-reflex success at or above the austere survival gate
- `fails_with_shaping`: the rung does not reach the basic success gate even under `classic`
- `mixed_or_borderline`: the rung sits near the thresholds or shows mixed outcomes across profiles without a clean gate result

Current claim framing:

| Scaffold class | Reporting consequence |
| --- | --- |
| `MINIMAL_MANUAL` | `claim_severity="full"` |
| `STANDARD_CONSTRAINED` | `claim_severity="qualified"` |
| `SCAFFOLDED_RUNTIME` | `claim_severity="non_benchmark"` |

Protocol rule:

- use the ladder to diagnose and iterate
- use the paper-budget benchmark-of-record workflow to make strong claims

Do not collapse those two purposes into one.

## Operating Principles

- Do not add a new center because a higher rung exists on paper.
- Treat gate-before-add as the default policy: candidate signals may be documented in [interfaces.md](./interfaces.md), but no `A5` module progresses without the prerequisites and checklist in this document.
- Do not interpret a failure at a high rung as evidence that modularity itself is wrong until the lower rung below it passes.
- Prefer validating a few well-connected modules over accumulating poorly justified specialization.
- When a higher rung regresses, the default action is to diagnose or simplify, not to add more architecture.

That is the core purpose of the Architectural Ladder: validate few modules well before defending more modules.
