# Standardized Interfaces

<!-- Generated from spider_cortex_sim.interface_docs.render_interfaces_markdown(); do not edit manually. -->

- Schema version: `1`
- Registry fingerprint: `cf7df9f21d3a0527bd676a63b010ee22a72edd16e86aa52610a2cb564cd45ce8`
- Proposal interfaces: `visual_cortex, sensory_cortex, hunger_center, sleep_center, alert_center, perception_center, homeostasis_center, threat_center, alert_v1, alert_v2, alert_v3, hunger_v1, hunger_v2, hunger_v3, sensory_v1, sensory_v2, sensory_v3, sleep_v1, sleep_v2, sleep_v3, visual_v1, visual_v2, visual_v3`
- Context interfaces: `action_center_context, motor_cortex_context`

## Compatibility

- The current policy is `exact_match_required` for save/load.
- Changes to the name, `observation_key`, signal order, outputs, or version require explicit incompatibility handling.
- There is no automatic migration for older checkpoints.

## Perception And Active Sensing

### Breaking Change: Action Space

- The locomotion action space expanded from 5 to 9 actions.
- Existing indices are unchanged: `MOVE_UP`=0, `MOVE_DOWN`=1, `MOVE_LEFT`=2, `MOVE_RIGHT`=3, `STAY`=4.
- New active-sensing actions are `ORIENT_UP`=5, `ORIENT_DOWN`=6, `ORIENT_LEFT`=7, and `ORIENT_RIGHT`=8.

### Foveal And Peripheral Vision

- Visual targets are classified relative to the spider heading as `foveal`, `peripheral`, or `outside`.
- `fov_half_angle` defaults to `45.0` degrees and defines the foveal cone.
- The `peripheral_half_angle` defaults to `70.0` degrees and defines the outer visual cone.
- A default `peripheral_certainty_penalty` of `0.35` reduces peripheral visual certainty.
- Targets beyond `peripheral_half_angle` are outside the visual field (`visible=0`).
- Heading zones are not exposed as observation fields; agents receive the resulting `visible`, `certainty`, occlusion, and direction signals.
- Smell and olfactory gradients remain omnidirectional and are not gated by heading zones.

### Scan Recency

- `foveal_scan_age` reports how recently the current heading was actively scanned.
- `max_scan_age` defaults to `10.0` ticks for normalizing scan age into `[0, 1]`.
- A value of `0.0` means the current foveal heading was scanned this tick; `1.0` means stale or never scanned.
- `PerceptTrace.heading_dx` and `PerceptTrace.heading_dy` record the body or gaze heading active when a food, shelter, or predator trace was refreshed.
- Trace headings surface as `food_trace_heading_*`, `shelter_trace_heading_*`, and `predator_trace_heading_*` signals.
- Scan recency is self-knowledge derived from the spider's own heading changes, not hidden world-state access.

### ORIENT Actions

- `ORIENT_*` actions update `heading_dx` and `heading_dy` from `ORIENT_HEADINGS`; they are not movement translations in `ACTION_DELTAS`.
- After the heading update, the world refreshes the current tick's perception buffer immediately, before the normal end-of-step tick commit.
- The refresh itself does not advance the tick or run hunger, fatigue, predator, memory, or perceptual-delay stages.
- `ORIENT_*` actions still consume the step: after the refresh, the normal commit advances the tick and applies hunger, fatigue, predator, and perceptual-buffer progression.
- `ORIENT_*` actions do not cause displacement and do not trigger motor slip.

### Percept Traces

- Trace signals carry target direction, decayed strength, and the heading active when the trace was refreshed.
- `*_trace_heading_dx` and `*_trace_heading_dy` are scan context for the trace; they are not target directions.

### Perceptual Delay

- `perceptual_delay_ticks` defaults to `1.0` and returns observations from the previous tick when buffered.
- `perceptual_delay_noise` defaults to `0.5` and scales delay-noise effects.
- `NoiseConfig.delay.certainty_decay_per_tick` defaults to `0.0` and decays certainty as observations age.
- `NoiseConfig.delay.direction_jitter_per_tick` defaults to `0.0` and adds age-proportional jitter to delayed direction signals.
- Delay noise uses a dedicated RNG stream to preserve reproducibility.
- Trace signals such as `food_trace_strength` reflect the delayed percept state, not the current world state.

### Perception Categories

- Direct sight means visual certainty is above the debug threshold and the current scan is fresh.
- Trace means a decayed percept trace is active after direct sight has gone stale or disappeared.
- Delayed means perceptual delay is active and the current payload was not refreshed by an active scan.
- Uncertain means no direct, trace, or delayed classification applies.
- These categories are debug and audit labels from `_perception_category()`; they are not observation signals.

### Explicit Memory Signals

- `food_memory_dx`/`food_memory_dy`/`food_memory_age`, `shelter_memory_*`, `predator_memory_*`, and `escape_memory_*` are derived from perception-grounded memory slots.
- The environment pipeline maintains mechanical timing only: refresh cycles age slots, enforce TTLs, and clear expired targets.
- Data sources are limited to local visual perception, contact events, and movement history; the pipeline cannot inject information the spider has not perceived.
- `spider_cortex_sim.memory.MEMORY_LEAKAGE_AUDIT` is the authoritative classification source for explicit memory ownership and leakage risk.

## Interface Variants

- Interface variants are diagnostic sufficiency probes for local tasks and organism-level ablations; they are not production save/load interfaces.
- Use the variant ladder to ask how much signal a module needs before escalating to broader architectural diagnosis.

### `visual_cortex`

- Progression: `visual_v1` -> `visual_v2` -> `visual_v3` -> `visual_cortex`

| Level | Interface | Signal Count | Added At This Level |
| --- | --- | ---: | --- |
| v1 | `visual_v1` | 5 | `food_visible`, `food_dx`, `food_dy`, `heading_dx`, `heading_dy` |
| v2 | `visual_v2` | 11 | `shelter_visible`, `shelter_dx`, `shelter_dy`, `predator_visible`, `predator_dx`, `predator_dy` |
| v3 | `visual_v3` | 26 | `foveal_scan_age`, `food_trace_strength`, `food_trace_heading_dx`, `food_trace_heading_dy`, `shelter_trace_strength`, `shelter_trace_heading_dx`, `shelter_trace_heading_dy`, `predator_trace_strength`, `predator_trace_heading_dx`, `predator_trace_heading_dy`, `predator_motion_salience`, `visual_predator_threat`, `olfactory_predator_threat`, `day`, `night` |
| v4 | `visual_cortex` | 32 | `food_certainty`, `food_occluded`, `shelter_certainty`, `shelter_occluded`, `predator_certainty`, `predator_occluded` (canonical interface) |

### `sensory_cortex`

- Progression: `sensory_v1` -> `sensory_v2` -> `sensory_v3` -> `sensory_cortex`

| Level | Interface | Signal Count | Added At This Level |
| --- | --- | ---: | --- |
| v1 | `sensory_v1` | 5 | `recent_pain`, `recent_contact`, `predator_smell_strength`, `predator_smell_dx`, `predator_smell_dy` |
| v2 | `sensory_v2` | 8 | `food_smell_strength`, `food_smell_dx`, `food_smell_dy` |
| v3 | `sensory_v3` | 11 | `health`, `hunger`, `fatigue` |
| v4 | `sensory_cortex` | 12 | `light` (canonical interface) |

### `hunger_center`

- Progression: `hunger_v1` -> `hunger_v2` -> `hunger_v3` -> `hunger_center`

| Level | Interface | Signal Count | Added At This Level |
| --- | --- | ---: | --- |
| v1 | `hunger_v1` | 5 | `hunger`, `on_food`, `food_visible`, `food_dx`, `food_dy` |
| v2 | `hunger_v2` | 8 | `food_smell_strength`, `food_smell_dx`, `food_smell_dy` |
| v3 | `hunger_v3` | 13 | `food_trace_dx`, `food_trace_dy`, `food_trace_strength`, `food_trace_heading_dx`, `food_trace_heading_dy` |
| v4 | `hunger_center` | 18 | `food_certainty`, `food_occluded`, `food_memory_dx`, `food_memory_dy`, `food_memory_age` (canonical interface) |

### `sleep_center`

- Progression: `sleep_v1` -> `sleep_v2` -> `sleep_v3` -> `sleep_center`

| Level | Interface | Signal Count | Added At This Level |
| --- | --- | ---: | --- |
| v1 | `sleep_v1` | 3 | `fatigue`, `on_shelter`, `night` |
| v2 | `sleep_v2` | 4 | `shelter_role_level` |
| v3 | `sleep_v3` | 9 | `shelter_trace_dx`, `shelter_trace_dy`, `shelter_trace_strength`, `shelter_trace_heading_dx`, `shelter_trace_heading_dy` |
| v4 | `sleep_center` | 18 | `hunger`, `health`, `recent_pain`, `sleep_phase_level`, `rest_streak_norm`, `sleep_debt`, `shelter_memory_dx`, `shelter_memory_dy`, `shelter_memory_age` (canonical interface) |

### `alert_center`

- Progression: `alert_v1` -> `alert_v2` -> `alert_v3` -> `alert_center`

| Level | Interface | Signal Count | Added At This Level |
| --- | --- | ---: | --- |
| v1 | `alert_v1` | 3 | `predator_visible`, `predator_dx`, `predator_dy` |
| v2 | `alert_v2` | 5 | `predator_certainty`, `predator_occluded` |
| v3 | `alert_v3` | 9 | `predator_smell_strength`, `recent_pain`, `recent_contact`, `on_shelter` |
| v4 | `alert_center` | 27 | `predator_motion_salience`, `visual_predator_threat`, `olfactory_predator_threat`, `dominant_predator_none`, `dominant_predator_visual`, `dominant_predator_olfactory`, `night`, `predator_trace_dx`, `predator_trace_dy`, `predator_trace_strength`, `predator_trace_heading_dx`, `predator_trace_heading_dy`, `predator_memory_dx`, `predator_memory_dy`, `predator_memory_age`, `escape_memory_dx`, `escape_memory_dy`, `escape_memory_age` (canonical interface) |

## A5 Candidate Interfaces

- `A5` candidate modules are gated by the prerequisite checklist, module admission criteria, and operating rules in [architectural_ladder.md](./architectural_ladder.md).
- These candidate entries are protocol-defined proposal surfaces for `A5_future_bio_refined`; they are not part of `interface_registry()`, do not affect save/load compatibility, and do not change the current registry fingerprint.
- Placeholder signals documented here are allowed at the protocol level even when the runtime does not yet expose them. They exist so each future module proposal has a concrete interface hypothesis before implementation starts.
- `homeostasis_center` remains the coarse A2 control surface. Any A5 homeostatic refinement is an alternative experiment path, not an automatic extension of the current proposer stack.

### `olfactory_center`

- Status: `proposal_only`
- Candidate input count: `9`
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Connection: feeds `action_center` with valence role `"foraging"` or `"threat"` based on the dominant smell-driven signal. If the current runtime arbitration vocabulary is kept unchanged, `"foraging"` would map to the existing `hunger` lane.
- Experimental hypothesis: a dedicated smell-specialized proposer will improve odor-led food approach and odor-led threat avoidance in mixed-scent conditions, especially when direct vision is weak or absent.
- Ablation test: `drop_olfactory_center` must selectively reduce performance on smell-dominant tasks such as `olfactory_visual_dissociation` without producing the same failure pattern as `drop_sensory_cortex` or `drop_alert_center`.
- Coupling risk: `HIGH` because the candidate overlaps both the current smell integrator (`sensory_cortex`) and smell-aware threat handling (`alert_center`).

| # | Signal | Source | Range | Description |
| --- | --- | --- | --- | --- |
| 1 | `food_smell_strength` | `sensory_cortex` | `0.0..1.0` | Strength of the local food odor field. |
| 2 | `food_smell_dx` | `sensory_cortex` | `-1.0..1.0` | Horizontal food odor gradient. |
| 3 | `food_smell_dy` | `sensory_cortex` | `-1.0..1.0` | Vertical food odor gradient. |
| 4 | `predator_smell_strength` | `sensory_cortex` + `alert_center` | `0.0..1.0` | Strength of the local predator odor field. |
| 5 | `predator_smell_dx` | `sensory_cortex` | `-1.0..1.0` | Horizontal predator odor gradient. |
| 6 | `predator_smell_dy` | `sensory_cortex` | `-1.0..1.0` | Vertical predator odor gradient. |
| 7 | `olfactory_predator_threat` | `alert_center` | `0.0..1.0` | Aggregated threat estimate from olfactory-oriented predators. |
| 8 | `dominant_predator_olfactory` | `alert_center` | `0.0..1.0` | Indicator that olfactory hunters are the dominant current predator type. |
| 9 | `smell_ambiguity` | derived from `sensory_cortex` + `alert_center` | `0.0..1.0` | Conflict score for mixed food and predator gradients or unstable odor localization. |

### `nociceptive_center`

- Status: `proposal_only`
- Candidate input count: `8`
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Connection: feeds `action_center` with valence role `"threat"` as a body-state emergency proposer.
- Experimental hypothesis: a pain-specialized proposer will shorten escape latency after damage or contact and reduce repeated contact events after the first hit, especially when the predator is no longer directly visible.
- Ablation test: `drop_nociceptive_center` must degrade `pain_escape_latency` or post-contact survival more than smell-led detection tasks, while remaining distinguishable from the broader `drop_alert_center` failure pattern.
- Coupling risk: `MEDIUM` because the candidate overlaps current threat signals but proposes a narrower post-contact recovery role than `alert_center`.

| # | Signal | Source | Range | Description |
| --- | --- | --- | --- | --- |
| 1 | `recent_pain` | `sensory_cortex` + `alert_center` | `0.0..1.0` | Immediate recent pain caused by a predator hit. |
| 2 | `recent_contact` | `sensory_cortex` + `alert_center` | `0.0..1.0` | Immediate recent physical contact with the predator. |
| 3 | `health` | `sensory_cortex` + `alert_center` | `0.0..1.0` | Current body health available for damage-sensitive retreat control. |
| 4 | `damage_rate` | derived from `health` delta | `0.0..1.0` | Short-horizon rate of health loss after contact. |
| 5 | `pain_memory_dx` | derived from `alert_center` memory traces | `-1.0..1.0` | Horizontal direction of the last harmful contact or pain source. |
| 6 | `pain_memory_dy` | derived from `alert_center` memory traces | `-1.0..1.0` | Vertical direction of the last harmful contact or pain source. |
| 7 | `pain_memory_age` | derived from `alert_center` memory traces | `0.0..1.0` | Age of the last harmful contact memory. |
| 8 | `contact_intensity` | derived from `recent_contact` + `recent_pain` | `0.0..1.0` | Combined severity estimate for recent bodily impact. |

### `arousal_center`

- Status: `proposal_only`
- Candidate input count: `10`
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Connection: feeds `action_center` with valence role `"sleep"` when arousal is low and `"exploration"` when arousal is high, acting as a state-transition proposer rather than a direct threat or feeding specialist.
- Experimental hypothesis: a dedicated arousal proposer will improve sleep-wake transitions, scan initiation, and exploratory reactivation under changing circadian and threat pressure without taking over the core sheltering job of `sleep_center`.
- Ablation test: `drop_arousal_center` must selectively hurt `arousal_state_transition`, timely `ORIENT_*` use, or post-rest reactivation without collapsing into the same deficit profile as `drop_sleep_center`.
- Coupling risk: `HIGH` because the candidate overlaps `sleep_center`, contextual arbitration, and activity-state features that may already be expressible without another proposer.

| # | Signal | Source | Range | Description |
| --- | --- | --- | --- | --- |
| 1 | `fatigue` | `sensory_cortex` + `sleep_center` + `action_context` | `0.0..1.0` | Immediate fatigue pressure acting against wakeful exploration. |
| 2 | `sleep_debt` | `sleep_center` + `action_context` | `0.0..1.0` | Accumulated need for restorative sleep. |
| 3 | `night` | `sleep_center` + `action_context` | `0.0..1.0` | Nighttime state indicator. |
| 4 | `day_night_transition` | derived from `day`/`night` history | `0.0..1.0` | Recent circadian transition marker for waking or winding down. |
| 5 | `arousal_level` | derived from `sleep_center` + `action_context` | `0.0..1.0` | Latent wakefulness estimate combining rest and vigilance state. |
| 6 | `threat_induced_arousal` | derived from `sensory_cortex` + `action_context` | `0.0..1.0` | Transient activation caused by distant threat evidence. |
| 7 | `activity_level` | derived from `action_context.last_move_*` history | `0.0..1.0` | Short-horizon locomotor activity or quiescence estimate. |
| 8 | `metabolic_state` | derived from `sensory_cortex` body-state signals | `0.0..1.0` | Compact scalar summarizing available internal energy and recovery pressure. |
| 9 | `circadian_phase` | derived from day/night cycle phase | `0.0..1.0` | Normalized position within the circadian cycle. |
| 10 | `rest_urgency` | derived from `sleep_center` | `0.0..1.0` | Immediate pressure to stay in shelter or resume rest. |

### `homeostasis_center` A5 Refinement Pathway

- Current status: `homeostasis_center` is the coarse `A2_three_center` homeostatic control with `28` canonical signals and remains the rollback target when finer splits regress.
- Refinement admission rule: consider an A5 homeostatic refinement only if the current `A4_current_full_modular` split still shows reproducible interference that the existing `hunger_center` and `sleep_center` separation does not resolve.
- Refinement topology rule: treat `metabolic_regulation_center` and `thermoregulation_center` as alternative sixth-module experiments. Do not admit both at once without new evidence, so the first A5 comparison stays `modular_full` versus a hypothetical 6-module proposer stack.

#### `metabolic_regulation_center`

- Status: `proposal_only`, blocked unless `A4` homeostatic splits prove insufficient
- Candidate input count: `14`
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Connection: would feed `action_center` with valence role `"hunger"` as a narrower metabolic-drive proposer.

| # | Signal | Source | Range | Description |
| --- | --- | --- | --- | --- |
| 1 | `hunger` | `homeostasis_center` subset | `0.0..1.0` | Primary metabolic deficit signal. |
| 2 | `health` | `homeostasis_center` subset | `0.0..1.0` | Body viability constraint on foraging persistence. |
| 3 | `on_food` | `homeostasis_center` subset | `0.0..1.0` | Immediate feeding context. |
| 4 | `food_visible` | `homeostasis_center` subset | `0.0..1.0` | Direct food visibility gate. |
| 5 | `food_certainty` | `homeostasis_center` subset | `0.0..1.0` | Confidence in the visible food cue. |
| 6 | `food_smell_strength` | `homeostasis_center` subset | `0.0..1.0` | Food odor strength. |
| 7 | `food_smell_dx` | `homeostasis_center` subset | `-1.0..1.0` | Horizontal food odor gradient. |
| 8 | `food_smell_dy` | `homeostasis_center` subset | `-1.0..1.0` | Vertical food odor gradient. |
| 9 | `food_trace_dx` | `homeostasis_center` subset | `-1.0..1.0` | Horizontal direction of the short food trace. |
| 10 | `food_trace_dy` | `homeostasis_center` subset | `-1.0..1.0` | Vertical direction of the short food trace. |
| 11 | `food_trace_strength` | `homeostasis_center` subset | `0.0..1.0` | Strength of the short food trace. |
| 12 | `food_memory_dx` | `homeostasis_center` subset | `-1.0..1.0` | Horizontal direction of remembered food. |
| 13 | `food_memory_dy` | `homeostasis_center` subset | `-1.0..1.0` | Vertical direction of remembered food. |
| 14 | `food_memory_age` | `homeostasis_center` subset | `0.0..1.0` | Age of remembered food. |

#### `thermoregulation_center`

- Status: `proposal_only`, blocked unless `A4` homeostatic splits prove insufficient
- Candidate input count: `14`
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Connection: would feed `action_center` with valence role `"sleep"` as a narrower shelter and recovery proposer.

| # | Signal | Source | Range | Description |
| --- | --- | --- | --- | --- |
| 1 | `fatigue` | `homeostasis_center` subset | `0.0..1.0` | Immediate fatigue pressure. |
| 2 | `on_shelter` | `homeostasis_center` subset | `0.0..1.0` | Current shelter occupancy. |
| 3 | `day` | `homeostasis_center` subset | `0.0..1.0` | Daytime exposure proxy. |
| 4 | `night` | `homeostasis_center` subset | `0.0..1.0` | Nighttime sheltering proxy. |
| 5 | `sleep_phase_level` | `homeostasis_center` subset | `0.0..1.0` | Current sleep depth or rest phase. |
| 6 | `rest_streak_norm` | `homeostasis_center` subset | `0.0..1.0` | Recent rest continuity. |
| 7 | `sleep_debt` | `homeostasis_center` subset | `0.0..1.0` | Accumulated unmet rest demand. |
| 8 | `shelter_role_level` | `homeostasis_center` subset | `0.0..1.0` | Current depth inside the shelter. |
| 9 | `shelter_trace_dx` | `homeostasis_center` subset | `-1.0..1.0` | Horizontal direction of the short shelter trace. |
| 10 | `shelter_trace_dy` | `homeostasis_center` subset | `-1.0..1.0` | Vertical direction of the short shelter trace. |
| 11 | `shelter_trace_strength` | `homeostasis_center` subset | `0.0..1.0` | Strength of the short shelter trace. |
| 12 | `shelter_memory_dx` | `homeostasis_center` subset | `-1.0..1.0` | Horizontal direction of remembered shelter. |
| 13 | `shelter_memory_dy` | `homeostasis_center` subset | `-1.0..1.0` | Vertical direction of remembered shelter. |
| 14 | `shelter_memory_age` | `homeostasis_center` subset | `0.0..1.0` | Age of remembered shelter. |

- Refinement experimental hypothesis: only pursue an A5 homeostatic refinement when the current split still shows a reproducible interference pattern, and the candidate sixth-module configuration improves that pattern with `Cohen's d >= 0.3` while preserving inherited gates.
- Ablation design: compare `modular_full` against a hypothetical 6-module A5 configuration that introduces exactly one homeostatic refinement module at a time, then require the added module's removal or rollback comparison to isolate the intended interference reduction under the same `Cohen's d >= 0.3` admission threshold.
- Coupling risk: `HIGH` because the current A4 architecture already decomposes coarse homeostasis into `hunger_center` and `sleep_center`, so additional splitting is likely to duplicate existing behavior unless the interference pattern is cleanly separable.

## `visual_cortex`

- Observation key: `visual`
- Role: `proposal`
- Version: `7`
- Description: Visual proposer driven by food, shelter, and predator cues inside the local visual field.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `food_visible` | -1.0 | 1.0 | 1 when food is detected with sufficient confidence in the visual field. |
| 2 | `food_certainty` | 0.0 | 1.0 | Visual confidence in the perceived food target. |
| 3 | `food_occluded` | 0.0 | 1.0 | 1 when the nearest food is within range but occluded. |
| 4 | `food_dx` | -1.0 | 1.0 | Relative horizontal offset of the detected food. |
| 5 | `food_dy` | -1.0 | 1.0 | Relative vertical offset of the detected food. |
| 6 | `shelter_visible` | -1.0 | 1.0 | 1 when the shelter is visible with sufficient confidence. |
| 7 | `shelter_certainty` | 0.0 | 1.0 | Visual confidence in the perceived shelter. |
| 8 | `shelter_occluded` | 0.0 | 1.0 | 1 when the shelter is within visual range but occluded. |
| 9 | `shelter_dx` | -1.0 | 1.0 | Relative horizontal offset of the detected shelter. |
| 10 | `shelter_dy` | -1.0 | 1.0 | Relative vertical offset of the detected shelter. |
| 11 | `predator_visible` | -1.0 | 1.0 | 1 when the predator is visible with sufficient confidence. |
| 12 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the perceived predator. |
| 13 | `predator_occluded` | 0.0 | 1.0 | 1 when the predator is within visual range but occluded. |
| 14 | `predator_dx` | -1.0 | 1.0 | Relative horizontal offset of the detected predator. |
| 15 | `predator_dy` | -1.0 | 1.0 | Relative vertical offset of the detected predator. |
| 16 | `heading_dx` | -1.0 | 1.0 | Current horizontal body or gaze orientation. |
| 17 | `heading_dy` | -1.0 | 1.0 | Current vertical body or gaze orientation. |
| 18 | `foveal_scan_age` | 0.0 | 1.0 | Normalized ticks since the current foveal heading was actively scanned; 0 is fresh and 1 is stale or never scanned. |
| 19 | `food_trace_strength` | 0.0 | 1.0 | Decayed strength of the short food trace. |
| 20 | `food_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short food trace was refreshed. |
| 21 | `food_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short food trace was refreshed. |
| 22 | `shelter_trace_strength` | 0.0 | 1.0 | Decayed strength of the short shelter trace. |
| 23 | `shelter_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short shelter trace was refreshed. |
| 24 | `shelter_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short shelter trace was refreshed. |
| 25 | `predator_trace_strength` | 0.0 | 1.0 | Decayed strength of the short predator trace. |
| 26 | `predator_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short predator trace was refreshed. |
| 27 | `predator_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short predator trace was refreshed. |
| 28 | `predator_motion_salience` | 0.0 | 1.0 | Explicit motion salience of the predator. |
| 29 | `visual_predator_threat` | 0.0 | 1.0 | Aggregated threat from visually oriented predators. |
| 30 | `olfactory_predator_threat` | 0.0 | 1.0 | Aggregated threat from olfactory predators. |
| 31 | `day` | -1.0 | 1.0 | 1 during daytime. |
| 32 | `night` | -1.0 | 1.0 | 1 during nighttime. |

## `sensory_cortex`

- Observation key: `sensory`
- Role: `proposal`
- Version: `2`
- Description: Sensory proposer that integrates pain, body state, smell, and light.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `recent_pain` | 0.0 | 1.0 | Recent pain caused by a predator attack. |
| 2 | `recent_contact` | 0.0 | 1.0 | Recent physical contact with the predator. |
| 3 | `health` | 0.0 | 1.0 | Body health. |
| 4 | `hunger` | 0.0 | 1.0 | Body hunger. |
| 5 | `fatigue` | 0.0 | 1.0 | Body fatigue. |
| 6 | `food_smell_strength` | 0.0 | 1.0 | Intensity of the food scent. |
| 7 | `food_smell_dx` | -1.0 | 1.0 | Horizontal gradient of the food scent. |
| 8 | `food_smell_dy` | -1.0 | 1.0 | Vertical gradient of the food scent. |
| 9 | `predator_smell_strength` | 0.0 | 1.0 | Intensity of the predator scent. |
| 10 | `predator_smell_dx` | -1.0 | 1.0 | Horizontal gradient of the predator scent. |
| 11 | `predator_smell_dy` | -1.0 | 1.0 | Vertical gradient of the predator scent. |
| 12 | `light` | 0.0 | 1.0 | Simple light level. |

## `hunger_center`

- Observation key: `hunger`
- Role: `proposal`
- Version: `4`
- Description: Homeostatic proposer aimed at foraging and returning to the last perceived food.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `hunger` | 0.0 | 1.0 | Internal hunger state. |
| 2 | `on_food` | 0.0 | 1.0 | 1 when the spider is standing on food. |
| 3 | `food_visible` | 0.0 | 1.0 | 1 when food is visible. |
| 4 | `food_certainty` | 0.0 | 1.0 | Visual confidence in the perceived food. |
| 5 | `food_occluded` | 0.0 | 1.0 | 1 when nearby food exists but is occluded. |
| 6 | `food_dx` | -1.0 | 1.0 | Horizontal direction of the detected food. |
| 7 | `food_dy` | -1.0 | 1.0 | Vertical direction of the detected food. |
| 8 | `food_smell_strength` | 0.0 | 1.0 | Intensity of the food scent. |
| 9 | `food_smell_dx` | -1.0 | 1.0 | Horizontal gradient of the food scent. |
| 10 | `food_smell_dy` | -1.0 | 1.0 | Vertical gradient of the food scent. |
| 11 | `food_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short food trace. |
| 12 | `food_trace_dy` | -1.0 | 1.0 | Vertical direction of the short food trace. |
| 13 | `food_trace_strength` | 0.0 | 1.0 | Decayed strength of the short food trace. |
| 14 | `food_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short food trace was refreshed. |
| 15 | `food_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short food trace was refreshed. |
| 16 | `food_memory_dx` | -1.0 | 1.0 | Horizontal direction of the last seen food. |
| 17 | `food_memory_dy` | -1.0 | 1.0 | Vertical direction of the last seen food. |
| 18 | `food_memory_age` | 0.0 | 1.0 | Normalized age of the food memory. |

## `sleep_center`

- Observation key: `sleep`
- Role: `proposal`
- Version: `5`
- Description: Homeostatic proposer aimed at returning to shelter, resting, and seeking safe depth.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `fatigue` | 0.0 | 1.0 | Internal fatigue state. |
| 2 | `hunger` | 0.0 | 1.0 | Internal hunger state. |
| 3 | `on_shelter` | 0.0 | 1.0 | 1 when the spider is on the shelter. |
| 4 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 5 | `health` | 0.0 | 1.0 | Body health. |
| 6 | `recent_pain` | 0.0 | 1.0 | Recent pain. |
| 7 | `sleep_phase_level` | 0.0 | 1.0 | Current sleep phase level. |
| 8 | `rest_streak_norm` | 0.0 | 1.0 | Recent rest continuity. |
| 9 | `sleep_debt` | 0.0 | 1.0 | Accumulated sleep debt. |
| 10 | `shelter_role_level` | 0.0 | 1.0 | Current depth inside the shelter. |
| 11 | `shelter_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short shelter trace. |
| 12 | `shelter_trace_dy` | -1.0 | 1.0 | Vertical direction of the short shelter trace. |
| 13 | `shelter_trace_strength` | 0.0 | 1.0 | Decayed strength of the short shelter trace. |
| 14 | `shelter_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short shelter trace was refreshed. |
| 15 | `shelter_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short shelter trace was refreshed. |
| 16 | `shelter_memory_dx` | -1.0 | 1.0 | Horizontal direction of the nearest visible shelter cell. |
| 17 | `shelter_memory_dy` | -1.0 | 1.0 | Vertical direction of the nearest visible shelter cell. |
| 18 | `shelter_memory_age` | 0.0 | 1.0 | Normalized age of the nearest visible shelter cell memory. |

## `alert_center`

- Observation key: `alert`
- Role: `proposal`
- Version: `8`
- Description: Defensive proposer aimed at threat response, escape, and shelter prioritization under risk.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `predator_visible` | 0.0 | 1.0 | 1 when the predator is visible. |
| 2 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the perceived predator. |
| 3 | `predator_occluded` | 0.0 | 1.0 | 1 when the predator is within range but occluded. |
| 4 | `predator_dx` | -1.0 | 1.0 | Horizontal direction of the detected predator. |
| 5 | `predator_dy` | -1.0 | 1.0 | Vertical direction of the detected predator. |
| 6 | `predator_smell_strength` | 0.0 | 1.0 | Intensity of the predator scent. |
| 7 | `predator_motion_salience` | 0.0 | 1.0 | Explicit motion salience of the predator. |
| 8 | `visual_predator_threat` | 0.0 | 1.0 | Aggregated threat from visually oriented predators. |
| 9 | `olfactory_predator_threat` | 0.0 | 1.0 | Aggregated threat from olfactory predators. |
| 10 | `dominant_predator_none` | 0.0 | 1.0 | 1 when no predator type is currently dominant. |
| 11 | `dominant_predator_visual` | 0.0 | 1.0 | 1 when visual predators are the dominant threat type. |
| 12 | `dominant_predator_olfactory` | 0.0 | 1.0 | 1 when olfactory predators are the dominant threat type. |
| 13 | `recent_pain` | 0.0 | 1.0 | Recent pain. |
| 14 | `recent_contact` | 0.0 | 1.0 | Recent physical contact. |
| 15 | `on_shelter` | 0.0 | 1.0 | 1 when the spider is on the shelter. |
| 16 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 17 | `predator_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short predator trace. |
| 18 | `predator_trace_dy` | -1.0 | 1.0 | Vertical direction of the short predator trace. |
| 19 | `predator_trace_strength` | 0.0 | 1.0 | Decayed strength of the short predator trace. |
| 20 | `predator_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short predator trace was refreshed. |
| 21 | `predator_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short predator trace was refreshed. |
| 22 | `predator_memory_dx` | -1.0 | 1.0 | Horizontal direction of the predator position perceived during visual detection or inferred from contact event. |
| 23 | `predator_memory_dy` | -1.0 | 1.0 | Vertical direction of the predator position perceived during visual detection or inferred from contact event. |
| 24 | `predator_memory_age` | 0.0 | 1.0 | Normalized age of the predator position perceived during visual detection or inferred from contact event. |
| 25 | `escape_memory_dx` | -1.0 | 1.0 | Horizontal direction of the recent escape target derived from movement history without walkability assumptions. |
| 26 | `escape_memory_dy` | -1.0 | 1.0 | Vertical direction of the recent escape target derived from movement history without walkability assumptions. |
| 27 | `escape_memory_age` | 0.0 | 1.0 | Normalized age of the escape memory derived from movement history without walkability assumptions. |

## `perception_center`

- Observation key: `perception`
- Role: `proposal`
- Version: `1`
- Description: Coarse perception proposer that fuses visual, olfactory, trace, and perceptual-memory cues.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `food_visible` | 0.0 | 1.0 | 1 when food is visible. |
| 2 | `food_certainty` | 0.0 | 1.0 | Visual confidence in the perceived food. |
| 3 | `food_dx` | -1.0 | 1.0 | Horizontal direction of the detected food. |
| 4 | `food_dy` | -1.0 | 1.0 | Vertical direction of the detected food. |
| 5 | `shelter_visible` | 0.0 | 1.0 | 1 when shelter is visible. |
| 6 | `shelter_certainty` | 0.0 | 1.0 | Visual confidence in the perceived shelter. |
| 7 | `shelter_dx` | -1.0 | 1.0 | Horizontal direction of the detected shelter. |
| 8 | `shelter_dy` | -1.0 | 1.0 | Vertical direction of the detected shelter. |
| 9 | `predator_visible` | 0.0 | 1.0 | 1 when a predator is visible. |
| 10 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the perceived predator. |
| 11 | `predator_dx` | -1.0 | 1.0 | Horizontal direction of the detected predator. |
| 12 | `predator_dy` | -1.0 | 1.0 | Vertical direction of the detected predator. |
| 13 | `heading_dx` | -1.0 | 1.0 | Current horizontal body or gaze orientation. |
| 14 | `heading_dy` | -1.0 | 1.0 | Current vertical body or gaze orientation. |
| 15 | `foveal_scan_age` | 0.0 | 1.0 | Normalized ticks since the current foveal heading was actively scanned. |
| 16 | `food_smell_strength` | 0.0 | 1.0 | Intensity of the food scent. |
| 17 | `food_smell_dx` | -1.0 | 1.0 | Horizontal gradient of the food scent. |
| 18 | `food_smell_dy` | -1.0 | 1.0 | Vertical gradient of the food scent. |
| 19 | `predator_smell_strength` | 0.0 | 1.0 | Intensity of the predator scent. |
| 20 | `predator_smell_dx` | -1.0 | 1.0 | Horizontal gradient of the predator scent. |
| 21 | `predator_smell_dy` | -1.0 | 1.0 | Vertical gradient of the predator scent. |
| 22 | `light` | 0.0 | 1.0 | Simple light level. |
| 23 | `day` | 0.0 | 1.0 | 1 during daytime. |
| 24 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 25 | `food_trace_strength` | 0.0 | 1.0 | Decayed strength of the short food trace. |
| 26 | `food_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short food trace was refreshed. |
| 27 | `food_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short food trace was refreshed. |
| 28 | `shelter_trace_strength` | 0.0 | 1.0 | Decayed strength of the short shelter trace. |
| 29 | `shelter_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short shelter trace was refreshed. |
| 30 | `shelter_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short shelter trace was refreshed. |
| 31 | `predator_trace_strength` | 0.0 | 1.0 | Decayed strength of the short predator trace. |
| 32 | `predator_trace_heading_dx` | -1.0 | 1.0 | Horizontal heading active when the short predator trace was refreshed. |
| 33 | `predator_trace_heading_dy` | -1.0 | 1.0 | Vertical heading active when the short predator trace was refreshed. |
| 34 | `food_memory_dx` | -1.0 | 1.0 | Horizontal direction of the last seen food. |
| 35 | `food_memory_dy` | -1.0 | 1.0 | Vertical direction of the last seen food. |
| 36 | `food_memory_age` | 0.0 | 1.0 | Normalized age of the food memory. |
| 37 | `shelter_memory_dx` | -1.0 | 1.0 | Horizontal direction of the remembered shelter. |
| 38 | `shelter_memory_dy` | -1.0 | 1.0 | Vertical direction of the remembered shelter. |
| 39 | `shelter_memory_age` | 0.0 | 1.0 | Normalized age of the shelter memory. |
| 40 | `predator_memory_dx` | -1.0 | 1.0 | Horizontal direction of the remembered predator. |
| 41 | `predator_memory_dy` | -1.0 | 1.0 | Vertical direction of the remembered predator. |
| 42 | `predator_memory_age` | 0.0 | 1.0 | Normalized age of the predator memory. |

## `homeostasis_center`

- Observation key: `homeostasis`
- Role: `proposal`
- Version: `1`
- Description: Coarse homeostatic proposer that blends hunger, fatigue, sheltering, and recovery drives.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `hunger` | 0.0 | 1.0 | Internal hunger state. |
| 2 | `fatigue` | 0.0 | 1.0 | Internal fatigue state. |
| 3 | `health` | 0.0 | 1.0 | Body health. |
| 4 | `on_food` | 0.0 | 1.0 | 1 when the spider is standing on food. |
| 5 | `on_shelter` | 0.0 | 1.0 | 1 when the spider is on the shelter. |
| 6 | `day` | 0.0 | 1.0 | 1 during daytime. |
| 7 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 8 | `sleep_phase_level` | 0.0 | 1.0 | Current sleep phase level. |
| 9 | `rest_streak_norm` | 0.0 | 1.0 | Recent rest continuity. |
| 10 | `sleep_debt` | 0.0 | 1.0 | Accumulated sleep debt. |
| 11 | `shelter_role_level` | 0.0 | 1.0 | Current depth inside the shelter. |
| 12 | `food_visible` | 0.0 | 1.0 | 1 when food is visible. |
| 13 | `food_certainty` | 0.0 | 1.0 | Visual confidence in the perceived food. |
| 14 | `food_smell_strength` | 0.0 | 1.0 | Intensity of the food scent. |
| 15 | `food_smell_dx` | -1.0 | 1.0 | Horizontal gradient of the food scent. |
| 16 | `food_smell_dy` | -1.0 | 1.0 | Vertical gradient of the food scent. |
| 17 | `food_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short food trace. |
| 18 | `food_trace_dy` | -1.0 | 1.0 | Vertical direction of the short food trace. |
| 19 | `food_trace_strength` | 0.0 | 1.0 | Decayed strength of the short food trace. |
| 20 | `food_memory_dx` | -1.0 | 1.0 | Horizontal direction of the last seen food. |
| 21 | `food_memory_dy` | -1.0 | 1.0 | Vertical direction of the last seen food. |
| 22 | `food_memory_age` | 0.0 | 1.0 | Normalized age of the food memory. |
| 23 | `shelter_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short shelter trace. |
| 24 | `shelter_trace_dy` | -1.0 | 1.0 | Vertical direction of the short shelter trace. |
| 25 | `shelter_trace_strength` | 0.0 | 1.0 | Decayed strength of the short shelter trace. |
| 26 | `shelter_memory_dx` | -1.0 | 1.0 | Horizontal direction of the remembered shelter. |
| 27 | `shelter_memory_dy` | -1.0 | 1.0 | Vertical direction of the remembered shelter. |
| 28 | `shelter_memory_age` | 0.0 | 1.0 | Normalized age of the shelter memory. |

## `threat_center`

- Observation key: `threat`
- Role: `proposal`
- Version: `1`
- Description: Coarse threat proposer that blends threat sensing, pain/contact, and escape memory.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `predator_visible` | 0.0 | 1.0 | 1 when a predator is visible. |
| 2 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the perceived predator. |
| 3 | `predator_dx` | -1.0 | 1.0 | Horizontal direction of the detected predator. |
| 4 | `predator_dy` | -1.0 | 1.0 | Vertical direction of the detected predator. |
| 5 | `predator_smell_strength` | 0.0 | 1.0 | Intensity of the predator scent. |
| 6 | `predator_smell_dx` | -1.0 | 1.0 | Horizontal gradient of the predator scent. |
| 7 | `predator_smell_dy` | -1.0 | 1.0 | Vertical gradient of the predator scent. |
| 8 | `predator_motion_salience` | 0.0 | 1.0 | Explicit motion salience of the predator. |
| 9 | `visual_predator_threat` | 0.0 | 1.0 | Aggregated threat from visually oriented predators. |
| 10 | `olfactory_predator_threat` | 0.0 | 1.0 | Aggregated threat from olfactory predators. |
| 11 | `dominant_predator_none` | 0.0 | 1.0 | 1 when no predator type is currently dominant. |
| 12 | `dominant_predator_visual` | 0.0 | 1.0 | 1 when visual predators are the dominant threat type. |
| 13 | `dominant_predator_olfactory` | 0.0 | 1.0 | 1 when olfactory predators are the dominant threat type. |
| 14 | `recent_pain` | 0.0 | 1.0 | Recent pain. |
| 15 | `recent_contact` | 0.0 | 1.0 | Recent physical contact. |
| 16 | `health` | 0.0 | 1.0 | Body health. |
| 17 | `on_shelter` | 0.0 | 1.0 | 1 when the spider is on the shelter. |
| 18 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 19 | `predator_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short predator trace. |
| 20 | `predator_trace_dy` | -1.0 | 1.0 | Vertical direction of the short predator trace. |
| 21 | `predator_trace_strength` | 0.0 | 1.0 | Decayed strength of the short predator trace. |
| 22 | `predator_memory_dx` | -1.0 | 1.0 | Horizontal direction of the remembered predator. |
| 23 | `predator_memory_dy` | -1.0 | 1.0 | Vertical direction of the remembered predator. |
| 24 | `predator_memory_age` | 0.0 | 1.0 | Normalized age of the predator memory. |
| 25 | `escape_memory_dx` | -1.0 | 1.0 | Horizontal direction of the remembered escape target. |
| 26 | `escape_memory_dy` | -1.0 | 1.0 | Vertical direction of the remembered escape target. |
| 27 | `escape_memory_age` | 0.0 | 1.0 | Normalized age of the escape memory. |

## `action_center_context`

- Observation key: `action_context`
- Role: `context`
- Version: `3`
- Description: Raw context used by the action_center to arbitrate competing locomotion proposals.
- Outputs: ``
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `hunger` | 0.0 | 1.0 | Body hunger. |
| 2 | `fatigue` | 0.0 | 1.0 | Body fatigue. |
| 3 | `health` | 0.0 | 1.0 | Body health. |
| 4 | `recent_pain` | 0.0 | 1.0 | Recent pain. |
| 5 | `recent_contact` | 0.0 | 1.0 | Recent contact with the predator. |
| 6 | `on_food` | 0.0 | 1.0 | 1 when standing on food. |
| 7 | `on_shelter` | 0.0 | 1.0 | 1 when standing on the shelter. |
| 8 | `predator_visible` | 0.0 | 1.0 | 1 when the predator is visible. |
| 9 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the predator. |
| 10 | `day` | 0.0 | 1.0 | 1 during daytime. |
| 11 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 12 | `last_move_dx` | -1.0 | 1.0 | Last executed horizontal displacement. |
| 13 | `last_move_dy` | -1.0 | 1.0 | Last executed vertical displacement. |
| 14 | `sleep_debt` | 0.0 | 1.0 | Accumulated sleep debt. |
| 15 | `shelter_role_level` | 0.0 | 1.0 | Current depth inside the shelter. |

## `motor_cortex_context`

- Observation key: `motor_context`
- Role: `context`
- Version: `5`
- Description: Raw context used by the motor_cortex to execute locomotion intent within local embodiment constraints.
- Outputs: ``
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `on_food` | 0.0 | 1.0 | 1 when standing on food. |
| 2 | `on_shelter` | 0.0 | 1.0 | 1 when standing on the shelter. |
| 3 | `predator_visible` | 0.0 | 1.0 | 1 when the predator is visible. |
| 4 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the predator. |
| 5 | `day` | 0.0 | 1.0 | 1 during daytime. |
| 6 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 7 | `last_move_dx` | -1.0 | 1.0 | Last executed horizontal displacement. |
| 8 | `last_move_dy` | -1.0 | 1.0 | Last executed vertical displacement. |
| 9 | `shelter_role_level` | 0.0 | 1.0 | Current depth inside the shelter. |
| 10 | `heading_dx` | -1.0 | 1.0 | Current horizontal body orientation. |
| 11 | `heading_dy` | -1.0 | 1.0 | Current vertical body orientation. |
| 12 | `terrain_difficulty` | 0.0 | 1.0 | Local movement cost from the terrain under the spider. |
| 13 | `fatigue` | 0.0 | 1.0 | Body fatigue affecting execution reliability. |
| 14 | `momentum` | 0.0 | 1.0 | Bounded execution momentum state; motor-only body state, not decision-priority context. |
