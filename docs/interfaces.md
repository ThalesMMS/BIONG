# Standardized Interfaces

<!-- Generated from spider_cortex_sim.interfaces.render_interfaces_markdown(); do not edit manually. -->

- Schema version: `1`
- Registry fingerprint: `4157eaaaf4b74099d173af0bd0b811a2a7ddd474de9c325734456ade00386372`
- Proposal interfaces: `visual_cortex, sensory_cortex, hunger_center, sleep_center, alert_center`
- Context interfaces: `action_center_context, motor_cortex_context`

## Compatibility

- The current policy is `exact_match_required` for save/load.
- Changes to the name, `observation_key`, signal order, outputs, or version require explicit incompatibility handling.
- There is no automatic migration for older checkpoints.

## `visual_cortex`

- Observation key: `visual`
- Role: `proposal`
- Version: `2`
- Description: Visual proposer driven by food, shelter, and predator cues inside the local visual field.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
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
| 22 | `day` | -1.0 | 1.0 | 1 during daytime. |
| 23 | `night` | -1.0 | 1.0 | 1 during nighttime. |

## `sensory_cortex`

- Observation key: `sensory`
- Role: `proposal`
- Version: `1`
- Description: Sensory proposer that integrates pain, body state, smell, and light.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
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
- Version: `2`
- Description: Homeostatic proposer aimed at foraging and returning to the last perceived food.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
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
- Version: `2`
- Description: Homeostatic proposer aimed at returning to shelter, resting, and seeking safe depth.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `fatigue` | 0.0 | 1.0 | Internal fatigue state. |
| 2 | `hunger` | 0.0 | 1.0 | Internal hunger state. |
| 3 | `on_shelter` | 0.0 | 1.0 | 1 when the spider is on the shelter. |
| 4 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 5 | `home_dx` | -1.0 | 1.0 | Internal horizontal vector toward the shelter. |
| 6 | `home_dy` | -1.0 | 1.0 | Internal vertical vector toward the shelter. |
| 7 | `home_dist` | 0.0 | 1.0 | Normalized distance to the shelter. |
| 8 | `health` | 0.0 | 1.0 | Body health. |
| 9 | `recent_pain` | 0.0 | 1.0 | Recent pain. |
| 10 | `sleep_phase_level` | 0.0 | 1.0 | Current sleep phase level. |
| 11 | `rest_streak_norm` | 0.0 | 1.0 | Recent rest continuity. |
| 12 | `sleep_debt` | 0.0 | 1.0 | Accumulated sleep debt. |
| 13 | `shelter_role_level` | 0.0 | 1.0 | Current depth inside the shelter. |
| 14 | `shelter_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short shelter trace. |
| 15 | `shelter_trace_dy` | -1.0 | 1.0 | Vertical direction of the short shelter trace. |
| 16 | `shelter_trace_strength` | 0.0 | 1.0 | Decayed strength of the short shelter trace. |
| 17 | `shelter_memory_dx` | -1.0 | 1.0 | Horizontal direction of the remembered safe shelter. |
| 18 | `shelter_memory_dy` | -1.0 | 1.0 | Vertical direction of the remembered safe shelter. |
| 19 | `shelter_memory_age` | 0.0 | 1.0 | Normalized age of the shelter memory. |

## `alert_center`

- Observation key: `alert`
- Role: `proposal`
- Version: `2`
- Description: Defensive proposer aimed at threat response, escape, and shelter prioritization under risk.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `predator_visible` | 0.0 | 1.0 | 1 when the predator is visible. |
| 2 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the perceived predator. |
| 3 | `predator_occluded` | 0.0 | 1.0 | 1 when the predator is within range but occluded. |
| 4 | `predator_dx` | -1.0 | 1.0 | Horizontal direction of the detected predator. |
| 5 | `predator_dy` | -1.0 | 1.0 | Vertical direction of the detected predator. |
| 6 | `predator_dist` | 0.0 | 1.0 | Normalized distance to the predator. |
| 7 | `predator_smell_strength` | 0.0 | 1.0 | Intensity of the predator scent. |
| 8 | `predator_motion_salience` | 0.0 | 1.0 | Explicit motion salience of the predator. |
| 9 | `home_dx` | -1.0 | 1.0 | Internal horizontal vector toward the shelter. |
| 10 | `home_dy` | -1.0 | 1.0 | Internal vertical vector toward the shelter. |
| 11 | `recent_pain` | 0.0 | 1.0 | Recent pain. |
| 12 | `recent_contact` | 0.0 | 1.0 | Recent physical contact. |
| 13 | `on_shelter` | 0.0 | 1.0 | 1 when the spider is on the shelter. |
| 14 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 15 | `predator_trace_dx` | -1.0 | 1.0 | Horizontal direction of the short predator trace. |
| 16 | `predator_trace_dy` | -1.0 | 1.0 | Vertical direction of the short predator trace. |
| 17 | `predator_trace_strength` | 0.0 | 1.0 | Decayed strength of the short predator trace. |
| 18 | `predator_memory_dx` | -1.0 | 1.0 | Horizontal direction of the last seen predator position. |
| 19 | `predator_memory_dy` | -1.0 | 1.0 | Vertical direction of the last seen predator position. |
| 20 | `predator_memory_age` | 0.0 | 1.0 | Normalized age of the predator memory. |
| 21 | `escape_memory_dx` | -1.0 | 1.0 | Horizontal direction of the recent escape route. |
| 22 | `escape_memory_dy` | -1.0 | 1.0 | Vertical direction of the recent escape route. |
| 23 | `escape_memory_age` | 0.0 | 1.0 | Normalized age of the escape memory. |

## `action_center_context`

- Observation key: `action_context`
- Role: `context`
- Version: `1`
- Description: Raw context used by the action_center to arbitrate competing locomotion proposals.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
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
| 10 | `predator_dist` | 0.0 | 1.0 | Normalized distance to the predator. |
| 11 | `day` | 0.0 | 1.0 | 1 during daytime. |
| 12 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 13 | `last_move_dx` | -1.0 | 1.0 | Last executed horizontal displacement. |
| 14 | `last_move_dy` | -1.0 | 1.0 | Last executed vertical displacement. |
| 15 | `sleep_debt` | 0.0 | 1.0 | Accumulated sleep debt. |
| 16 | `shelter_role_level` | 0.0 | 1.0 | Current depth inside the shelter. |

## `motor_cortex_context`

- Observation key: `motor_context`
- Role: `context`
- Version: `1`
- Description: Raw context used by the motor_cortex to correct and execute the locomotion intent.
- Outputs: `MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY`
- Save policy: `exact_match_required`

| # | Signal | Min | Max | Description |
| --- | --- | ---: | ---: | --- |
| 1 | `on_food` | 0.0 | 1.0 | 1 when standing on food. |
| 2 | `on_shelter` | 0.0 | 1.0 | 1 when standing on the shelter. |
| 3 | `predator_visible` | 0.0 | 1.0 | 1 when the predator is visible. |
| 4 | `predator_certainty` | 0.0 | 1.0 | Visual confidence in the predator. |
| 5 | `predator_dist` | 0.0 | 1.0 | Normalized distance to the predator. |
| 6 | `day` | 0.0 | 1.0 | 1 during daytime. |
| 7 | `night` | 0.0 | 1.0 | 1 during nighttime. |
| 8 | `last_move_dx` | -1.0 | 1.0 | Last executed horizontal displacement. |
| 9 | `last_move_dy` | -1.0 | 1.0 | Last executed vertical displacement. |
| 10 | `shelter_role_level` | 0.0 | 1.0 | Current depth inside the shelter. |
