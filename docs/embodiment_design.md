# Embodiment Design Note

## Goal

Deepen motor embodiment without expanding the 9-action locomotion contract.

The next step is a compact stateful execution model centered on `momentum`: a
bounded scalar stored on `SpiderState` in `[0.0, 1.0]`. Momentum captures how
much physically useful movement continuity the spider has accumulated. It is not
a behavioral motive and does not choose goals. It changes how reliably a chosen
motor action can be executed.

Benchmark hypothesis: "Momentum-aware motor correction improves escape
efficiency under sustained threat pressure"

## Model

Momentum is stored as a first-class body state:

- location: `SpiderState.momentum`
- range: `[0.0, 1.0]`
- initial value: `0.0`
- interpretation: `0.0` means no useful movement continuity; `1.0` means fully
  built aligned locomotor continuity

This deliberately avoids explicit speed or gait actions. High momentum behaves
like sustained efficient locomotion, while low momentum behaves like cautious or
interrupted movement. The action space remains:

`MOVE_UP`, `MOVE_DOWN`, `MOVE_LEFT`, `MOVE_RIGHT`, `STAY`, `ORIENT_UP`,
`ORIENT_DOWN`, `ORIENT_LEFT`, `ORIENT_RIGHT`

## Momentum Dynamics

The world updates momentum after resolving the motor-selected action and the
physical execution result. The transition should use the heading before and
after execution plus the executed action category.

- Aligned consecutive movement builds momentum.
- Perpendicular movement bleeds momentum.
- Reversing direction clears momentum.
- `STAY` gradually decays momentum.
- `ORIENT_*` clears momentum because it is explicit reorientation without
  displacement.

The concrete implementation uses these rules:

- aligned movement: increase momentum by `0.15`, then clamp to `1.0`
- 90 degree turn before movement: multiply momentum by `0.5`
- 180 degree reversal: set momentum to `0.0`
- `STAY`: multiply momentum by `0.8`
- `ORIENT_*`: set momentum to `0.0`
- blocked movement: multiply momentum by `0.8` because no physical movement
  continuity was achieved
- restoration in `SETTLING`, `RESTING`, and `DEEP_SLEEP`: reset momentum to
  `0.0`

The design intent is continuity cost, not execution latency. Turning should not
consume an extra tick in the first implementation.

## Turn Cost

Turning cost is based on the heading change needed by the executed action.

- heading change below 90 degrees: momentum may bleed, but no extra physiology
  penalty is required by this design
- heading change of 90 degrees or greater: apply momentum bleed and a minor
  fatigue cost from `turn_fatigue_cost` (`0.002`)
- 180 degree reversal: apply the strongest momentum bleed and a larger minor
  fatigue cost from `reverse_fatigue_cost` (`0.004`)

`ORIENT_*` actions remain pure orientation actions. They do not displace the
spider and should still bypass motor slip, but large heading changes through
`ORIENT_*` pay the same turn-fatigue accounting as movement turns.

Movement actions continue to update heading in the first implementation. The
cost of abrupt heading change is paid through momentum loss and fatigue rather
than through a broader orientation state machine.

## Execution Difficulty

Momentum integrates with `compute_execution_difficulty()` as a protective factor
for well-aligned movement.

- If the intended or executed movement is aligned with the current body heading,
  momentum can reduce execution difficulty and therefore reduce slip pressure.
- Momentum should not help misaligned movement. Poor form remains difficult even
  if the spider had momentum before the turn.
- Momentum is reported as a diagnostic component alongside orientation
  alignment, terrain difficulty, fatigue, execution difficulty, and slip reason.

The intended effect is that sustained escape along a stable heading becomes more
reliable, while frantic direction changes lose that reliability.

## Ownership

### `action_center`

`action_center` owns behavioral decision and priority context. It should not
receive `momentum`.

Momentum is not hunger, sleep debt, threat salience, or exploratory preference.
Adding it to action arbitration would leak execution state into the motive layer
and make it harder to distinguish "what the spider wants to do" from "what its
body can execute well."

### `motor_cortex`

`motor_cortex` owns motor correction. It receives `momentum` through
`MotorContextObservation` as field #14 with range `[0, 1]`.

This lets the motor stage learn corrections such as preserving heading under
predator pressure when high momentum makes continued aligned movement reliable.

### World Execution

The world owns physics and embodiment transitions.

- update `SpiderState.momentum`
- compute heading change and turn angle
- apply momentum build, decay, bleed, or reset
- compute execution difficulty and slip pressure
- keep the 9-action contract intact unless a later benchmark justifies changing
  it

### Physiology

Physiology owns energetic and recovery consequences.

- apply minor fatigue cost for sharp turns
- apply larger minor fatigue cost for reversal
- reset momentum during sleep restoration phases
- keep hunger, fatigue, recovery, and sleep accounting outside the motor
  correction network

Momentum decay during rest should make sleep and sustained rest physically reset
the locomotor state rather than preserve hidden movement advantage.

## Diagnostics

The implementation should expose momentum and turn accounting consistently.

`BrainStep` should include:

- `momentum`

World `info` should include:

- `momentum`
- `momentum_before`
- `momentum_after`
- `turn_angle`
- `turn_fatigue_applied`
- execution-difficulty components including `momentum`

Event logs should include these fields on the action-resolution event:

- `momentum_before`
- `momentum_after`
- `turn_angle`
- `turn_fatigue_applied`

Traces and summaries should aggregate enough data to compare momentum-aware runs
against the current embodiment baseline:

- mean momentum
- momentum at predator contact or escape events
- turn-fatigue totals
- slip rate by momentum band
- escape success and escape latency under sustained threat pressure

## Interface Contract

`MotorContextObservation` reserves field #14 for `momentum`.

- interface: `MOTOR_CONTEXT_INTERFACE`
- interface name: `motor_cortex_context`
- observation key: `motor_context`
- version: `5`
- field #14: `momentum`
- range: `[0, 1]`
- ownership: execution-relevant motor body state

`momentum` must not be added to `ActionContextObservation` or
`ACTION_CONTEXT_INTERFACE`.

## Benchmark

The benchmark should test whether the added state changes behavior
scientifically rather than cosmetically.

Primary hypothesis:

"Momentum-aware motor correction improves escape efficiency under sustained
threat pressure"

A useful benchmark should compare the current embodiment baseline against the
momentum model in predator-pressure scenarios where sustained heading matters.
Expected improvements should be measured through escape success, escape latency,
slip rate during escape, and unnecessary turn frequency. A result only counts as
meaningful if the agent learns to preserve useful motion continuity under threat
without receiving hidden goal information in `motor_context`.
