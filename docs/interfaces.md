# Standardized Interfaces

<!-- Generated from spider_cortex_sim.interfaces.render_interfaces_markdown(); do not edit manually. -->

- Schema version: `1`
- Registry fingerprint: `177abe8cc5bd584e0995d6855f54f22a96cad8948145411fc877538c830ada10`
- Proposal interfaces: `visual_cortex, sensory_cortex, hunger_center, sleep_center, alert_center`
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
- `fov_half_angle` defaults to `60.0` degrees and defines the foveal cone.
- The `peripheral_half_angle` defaults to `90.0` degrees and defines the outer visual cone.
- A default `peripheral_certainty_penalty` of `0.35` reduces peripheral visual certainty.
- Targets beyond `peripheral_half_angle` are outside the visual field (`visible=0`).
- Heading zones are not exposed as observation fields; agents receive the resulting `visible`, `certainty`, occlusion, and direction signals.
- Smell and olfactory gradients remain omnidirectional and are not gated by heading zones.

### ORIENT Actions

- `ORIENT_*` actions update `heading_dx` and `heading_dy` without displacement.
- Orientation targets live in `ORIENT_HEADINGS`; they are not movement translations in `ACTION_DELTAS`.
- They do not trigger motor slip.
- They still consume a tick, so hunger, fatigue, predators, and perceptual buffers advance normally.

### Perceptual Delay

- `perceptual_delay_ticks` defaults to `1.0` and returns observations from the previous tick when buffered.
- `perceptual_delay_noise` defaults to `0.5` and scales delay-noise effects.
- `NoiseConfig.delay.certainty_decay_per_tick` defaults to `0.0` and decays certainty as observations age.
- `NoiseConfig.delay.direction_jitter_per_tick` defaults to `0.0` and adds age-proportional jitter to delayed direction signals.
- Delay noise uses a dedicated RNG stream to preserve reproducibility.
- Trace signals such as `food_trace_strength` reflect the delayed percept state, not the current world state.

### Explicit Memory Signals

- `food_memory_dx`/`food_memory_dy`/`food_memory_age`, `shelter_memory_*`, `predator_memory_*`, and `escape_memory_*` are derived from perception-grounded memory slots.
- The environment pipeline maintains mechanical timing only: refresh cycles age slots, enforce TTLs, and clear expired targets.
- Data sources are limited to local visual perception, contact events, and movement history; the pipeline cannot inject information the spider has not perceived.
- `spider_cortex_sim.memory.MEMORY_LEAKAGE_AUDIT` is the authoritative classification source for explicit memory ownership and leakage risk.

## `visual_cortex`

- Observation key: `visual`
- Role: `proposal`
- Version: `5`
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
| 18 | `food_trace_strength` | 0.0 | 1.0 | Decayed strength of the short food trace. |
| 19 | `shelter_trace_strength` | 0.0 | 1.0 | Decayed strength of the short shelter trace. |
| 20 | `predator_trace_strength` | 0.0 | 1.0 | Decayed strength of the short predator trace. |
| 21 | `predator_motion_salience` | 0.0 | 1.0 | Explicit motion salience of the predator. |
| 22 | `visual_predator_threat` | 0.0 | 1.0 | Aggregated threat from visually oriented predators. |
| 23 | `olfactory_predator_threat` | 0.0 | 1.0 | Aggregated threat from olfactory predators. |
| 24 | `day` | -1.0 | 1.0 | 1 during daytime. |
| 25 | `night` | -1.0 | 1.0 | 1 during nighttime. |

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
- Version: `3`
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
| 14 | `food_memory_dx` | -1.0 | 1.0 | Horizontal direction of the last seen food. |
| 15 | `food_memory_dy` | -1.0 | 1.0 | Vertical direction of the last seen food. |
| 16 | `food_memory_age` | 0.0 | 1.0 | Normalized age of the food memory. |

## `sleep_center`

- Observation key: `sleep`
- Role: `proposal`
- Version: `4`
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
| 14 | `shelter_memory_dx` | -1.0 | 1.0 | Horizontal direction of the nearest visible shelter cell. |
| 15 | `shelter_memory_dy` | -1.0 | 1.0 | Vertical direction of the nearest visible shelter cell. |
| 16 | `shelter_memory_age` | 0.0 | 1.0 | Normalized age of the nearest visible shelter cell memory. |

## `alert_center`

- Observation key: `alert`
- Role: `proposal`
- Version: `7`
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
| 20 | `predator_memory_dx` | -1.0 | 1.0 | Horizontal direction of the predator position perceived during visual detection or inferred from contact event. |
| 21 | `predator_memory_dy` | -1.0 | 1.0 | Vertical direction of the predator position perceived during visual detection or inferred from contact event. |
| 22 | `predator_memory_age` | 0.0 | 1.0 | Normalized age of the predator position perceived during visual detection or inferred from contact event. |
| 23 | `escape_memory_dx` | -1.0 | 1.0 | Horizontal direction of the recent escape target derived from movement history without walkability assumptions. |
| 24 | `escape_memory_dy` | -1.0 | 1.0 | Vertical direction of the recent escape target derived from movement history without walkability assumptions. |
| 25 | `escape_memory_age` | 0.0 | 1.0 | Normalized age of the escape memory derived from movement history without walkability assumptions. |

## `action_center_context`

- Observation key: `action_context`
- Role: `context`
- Version: `3`
- Description: Raw context used by the action_center to arbitrate competing locomotion proposals.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
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
- Version: `4`
- Description: Raw context used by the motor_cortex to execute locomotion intent within local embodiment constraints.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT`
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
