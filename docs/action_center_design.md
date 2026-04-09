# Action Center Design Note

## Goal

Separate behavioral arbitration from locomotor execution without changing the environment's motor space.

New pipeline:

`specialized modules or monolithic_policy -> valence-based priority gating -> action_center -> motor_cortex -> final action`

## Contracts

- `action_center`
  - input: concatenation of locomotion proposals plus `action_context`
  - output: locomotion logits over `MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `STAY`
  - value head: yes
  - explicit arbitration: `priority_gating` with valences `threat`, `hunger`, `sleep`, and `exploration`

- `motor_cortex`
  - input: the locomotion intent chosen by `action_center` in one-hot form plus `motor_context`
  - output: corrective logits over the same locomotion space
  - value head: no

## Contexts

- `action_context`
  - broad bodily and situational state for arbitration: hunger, fatigue, health, pain, contact, food or shelter status, predator cues, day or night cycle, last movement, sleep debt, and shelter depth

- `motor_context`
  - local execution or correction context: food or shelter status, predator cues, day or night cycle, last movement, and shelter depth

## Reflexes

- Local reflexes remain in the proposer modules.
- They modulate proposals before `action_center`.
- After that, `action_center` applies module gates according to the winning valence.
- `motor_cortex` does not apply reflexes; it only corrects execution of the arbitrated intent.

## Auditability

- The bus or trace serializes `winning_valence`, `valence_scores`, `module_gates`, `suppressed_modules`, `evidence`, `intent_before_gating`, and `intent_after_gating`.
- `ModuleResult` also carries `valence_role`, `gate_weight`, `gated_logits`, `intent_before_gating`, and `intent_after_gating` for explicit debugging of each gate's effect.
- The `food_vs_predator_conflict` and `sleep_vs_exploration_conflict` scenarios exercise this arbitration deterministically.

## Monolithic Policy

- `monolithic_policy` remains the single-proposal baseline.
- It now feeds the same `action_center` and the same `motor_cortex` used by the modular architecture.
- That preserves structural comparability downstream of the proposal stage.

## Compatibility

- `architecture_signature()` now includes `action_center`, `action_context`, and the new `motor_context`.
- `ARCHITECTURE_VERSION` was incremented.
- Older checkpoints fail with an explicit message; this change does not provide automatic migration.
