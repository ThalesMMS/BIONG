# B-Series Evolution: B0 to B5

Updated on 2026-05-16.

This document summarizes only the B-series line that worked, that is, the
checkpoints accepted and reused as transfer sources for the next level.
Discarded candidates are mentioned only when they help explain why the accepted
architecture was chosen.

## Principles Of The B Line

The B-series line is a parallel diagnostic track. It does not replace the A
line by itself. The goal of each `Bx` level is to preserve reasonable
learning/survival while adding a small amount of complexity, modularity, or
bioinspiration relative to `B(x-1)`.

The rules that consolidated from B0 to B5:

- The public action space of `SpiderWorld.step()` remains primitive. No
  semantic action enters the world.
- The six semantic actions remain internal: `MOVE_TO_FOOD`,
  `MOVE_TO_SHELTER`, `EXPLORE`, `STAY`, `EAT`, and `SLEEP`.
- The semantic-primitive bridge transforms the internal intention into an
  `ACTIONS` action.
- Starting at B1, every accepted level is born through transfer learning from
  the accepted `best` checkpoint of the previous level.
- A candidate that does not pass the level gate is saved in `discarded` and
  does not become evolutionary substrate.
- Local validation is focused on B-series and behavior smoke tests; the full
  suite remaining partially red does not invalidate promotion of a B level.

## Accepted Chain

| Level | Accepted architecture | Source used | Coverage | Main gate |
| --- | --- | --- | ---: | --- |
| B0 | `b0_current_bridge_policy` | no previous B source | n/a | full easy horizon |
| B1 | `b1_threat_guard_bridge_policy` | B0 current bridge | 0.666896 | full easy horizon |
| B2 | `b2_temporal_threat_h48_bridge_policy` | B1 threat guard | 1.0 | easy + canonical progress |
| B3 | `b3_recurrent_guard_h48_bridge_policy` | B2 temporal threat | 1.0 | easy + robust canonical ep0/ep1 |
| B4 | `b4_genetic_recovery_h48_bridge_policy` | B3 recurrent guard | 1.0 | easy 0..4 + canonical 0..9 retention |
| B5 | `b5_genetic_homeostasis_h48_bridge_policy` | B4 genetic recovery | 1.0 | easy 0..4 + canonical 0..9 + homeostatic probes |

Accepted checkpoints:

- B0:
  `artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best`
- B1:
  `artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best`
- B2:
  `artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best`
- B3:
  `artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best`
- B4:
  `artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best`
- B5:
  `artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best`

## B0: Current Primitive Semantic Bridge

Accepted architecture: `b0_current_bridge_policy`.

B0 created the minimum substrate of the current line: the policy still produces
and records internal semantic intentions, but the world receives only a
primitive action. The level separates three responsibilities:

- auditable semantic selection;
- local movement bridge toward food/shelter;
- primitive execution in the current world.

The fundamental difference from `b0_legacy_semantic_policy` is that B0 current
does not give macro actions to the environment. `MOVE_TO_FOOD` and
`MOVE_TO_SHELTER` do not enter `SpiderWorld.step()`: they are converted first
into primitive movement.

Recorded result:

- seed: `7`
- training: `24` episodes
- easy scenario: `432` steps alive
- food: `21`
- sleep: `100`
- shelter entries: `14`
- predator contacts: `1`
- final health: `1.0`
- primitive trace: valid

B0 is the evolutionary base of the line, but it is still deliberately simple.
The main complexity is in the bridge and in the basic semantic loop, not in a
more modular neural control system.

## B1: Initial Threat Guard

Accepted architecture: `b1_threat_guard_bridge_policy`.

Source:
`artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best`.

Transfer:

- parent level: `0`
- target level: `1`
- coverage: `0.666896`
- loaded parameters: `6791 / 10183`
- low coverage override: disabled

B1 started as a conservative capacity attempt. The capacity-only candidates
`b1_capacity_h48_bridge_policy` and `b1_capacity_h64_bridge_policy` were
discarded because they did not sustain the easy gate. The accepted candidate
added a minimal threat guard without changing the public world contract.

Main architectural change:

- keeps the six internal semantic actions;
- keeps the primitive bridge;
- increases hidden capacity to `48`;
- adds B1-local threat control with
  `semantic_action_source="b1_threat_guard_controller"`.

Accepted result:

- training: `24` episodes
- easy: `432` steps alive
- food: `21`
- sleep: `72`
- shelter entries: `18`
- predator contacts: `1`
- final health: `1.0`

Canonical diagnosis still weak, used as reference for B2:

- `69` steps
- food: `3`
- sleep: `3`
- shelter entries: `5`
- predator contacts: `5`
- final health: `0.0`

B1 proved that transfer from B0 to a larger network and a simple guard
preserved easy learning, but it still did not solve temporal threat in
canonical.

## B2: Temporal Threat Control

Accepted architecture: `b2_temporal_threat_h48_bridge_policy`.

Source:
`artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best`.

Transfer:

- parent level: `1`
- target level: `2`
- coverage: `1.0`
- loaded parameters: `10183 / 10183`

B2 kept `h48` capacity, the same semantic actions, and the same bridge, but
added a temporal threat controller. The goal was to attack the canonical blocker
observed in B1: predator contact and risk were not handled with enough memory.

Main architectural change:

- `semantic_action_source="b2_temporal_threat_controller"`;
- current threat pressure;
- short predator memory;
- temporal predator trace;
- tendency to return to/hold shelter when recent threat is active.

Accepted result:

- training: `48` episodes
- easy: `432` steps alive, `19` food items, `76` sleep events, `15` shelter
  entries, `1` contact, final health `1.0`
- canonical ep0: `300` steps alive, `13` food items, `46` sleep events, `12`
  shelter entries, `9` contacts, final health `0.063414`

B2 passed because it completed the canonical horizon and showed clear progress
against the B1 baseline, although still with many contacts.

## B3: Recurrent Guard And Contact Memory

Accepted architecture: `b3_recurrent_guard_h48_bridge_policy`.

Source:
`artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best`.

Transfer:

- parent level: `2`
- target level: `3`
- coverage: `1.0`
- loaded parameters: `10183 / 10183`

B3 started by testing direct contact memory. The variants
`b3_contact_memory_h48_bridge_policy`,
`b3_contact_memory_strict_h48_bridge_policy`, and
`b3_contact_memory_h56_bridge_policy` were discarded because they failed easy or
canonical. The accepted architecture was the recurrent `h48` guard, which
preserved the B2 substrate and added a transient per-episode profile.

Main architectural change:

- `semantic_action_source="b3_contact_memory_controller"` in the B3 family;
- post-contact cooldown;
- post-food cooldown;
- hunger-drop detection;
- per-episode recurrent profile, with a stricter guard at the beginning and
  later release.

Accepted result:

- training: `48` episodes
- easy: `432` steps alive, `19` food items, `76` sleep events, `15` shelter
  entries, `1` contact, final health `1.0`
- canonical ep0: `300` steps alive, `15` food items, `39` sleep events, `10`
  shelter entries, `2` contacts, final health `0.827079`
- canonical ep1: `300` steps alive, `16` food items, `65` sleep events, `13`
  shelter entries, `1` contact, final health `1.0`

B3 was the first level to make canonical robust in diagnostic episodes 0 and 1.
The improvement did not come from more capacity, but from transient state memory
and a temporal control profile.

## B4: Multi-Episode Recovery With Genetic Search

Accepted architecture: `b4_genetic_recovery_h48_bridge_policy`.

Source:
`artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best`.

Transfer:

- parent level: `3`
- target level: `4`
- coverage: `1.0`
- loaded parameters: `10183 / 10183`

B4 shifted the focus from local ep0/ep1 robustness to multi-episode retention in
canonical `0..9`. The goal was not necessarily to beat B3 on every number, but
to preserve reasonable survival with a more bioinspired recovery module.

The fixed variants passed easy, but were discarded for insufficient retention
in multi-episode canonical. The small genetic fallback searched recovery
thresholds and promoted the first confirmed profile.

Main architectural change:

- `semantic_action_source="b4_genetic_recovery_controller"`;
- recovery pressure;
- sleep/shelter retention when health, fatigue, or sleep debt indicate risk;
- premature-exit blocking;
- controlled foraging release;
- winning thresholds persisted in `b_controller_params`.

Accepted result:

- training: `48` episodes
- search: hybrid, workers `4`, GA population `12`, generations `4`
- easy `0..4`: passed
- canonical `0..9`: passed retention gate
- completed horizons: `4 / 10`
- minimum steps: `49`
- total contacts: `24`
- episodes with food cycle: `9`
- episodes with sleep cycle: `9`
- episodes with shelter cycle: `10`

Preserved canonical anchors:

- ep0: `300` steps alive, `14` food items, `54` sleep events, `10` shelter
  entries, `2` contacts, final health `1.0`
- ep1: `300` steps alive, `15` food items, `46` sleep events, `11` shelter
  entries, `4` contacts, final health `1.0`

Residual weakness that motivated B5:

- canonical episode `8` still collapsed due to homeostasis/shelter exit without
  enough sleep, near `49` steps.

## B5: Modular Homeostatic Arbiter

Accepted architecture: `b5_genetic_homeostasis_h48_bridge_policy`.

Source:
`artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best`.

Transfer:

- parent level: `4`
- target level: `5`
- coverage: `1.0`
- loaded parameters: `10183 / 10183`

B5 kept the B4 checkpoint as substrate and added a more explicit interoceptive
arbiter. The level separated hunger, sleep, recovery, and threat pressures, with
transient per-episode locks.

The fixed variants passed easy and the two required probes, but were discarded
for insufficient canonical retention. The genetic profile was accepted.

Main architectural change:

- `semantic_action_source="b5_genetic_homeostasis_controller"`;
- `b_effective_level="B5-homeostatic-arbiter"`;
- hunger pressure;
- sleep pressure;
- recovery debt;
- threat gate;
- `sleep_bout_lock`;
- `forage_commitment_lock`;
- homeostatic decision recorded in trace;
- winning thresholds persisted in `b_controller_params`.

Accepted result:

- training: `48` episodes
- search: hybrid, workers `4`, GA population `12`, generations `4`
- easy `0..4`: passed
- canonical `0..9`: passed retention
- completed horizons: `5 / 10`
- minimum steps: `52`
- total contacts: `24`
- episodes with food cycle: `9`
- episodes with sleep cycle: `9`
- episodes with shelter cycle: `10`

Required complex probes:

- `food_deprivation` ep0..2: passed, with `2 / 3` episodes showing progress
- `sleep_vs_exploration_conflict` ep0..2: passed, post-recovery movement metric
  exposed and `3 / 3` episodes with post-recovery movement

Problematic non-blocking diagnostics:

- `food_vs_predator_conflict`: scorers still fail threat priority and foraging
  suppression under threat.
- `corridor_gauntlet`: there is initial progress, but rapid death due to lack
  of sustained survival in the corridor.

B5 is therefore the first level in the line to combine B4 retention, explicit
modular homeostasis, and additional environmental probes without changing the
public action space.

## Reading The Evolution

The successful progression was not a simple escalation of hidden size. The
pattern that worked was:

1. B0 created an auditable bridge between semantic intention and primitive
   action.
2. B1 added a minimal threat guard when capacity-only failed.
3. B2 added temporal threat memory.
4. B3 added episodic contact memory and a recurrent profile.
5. B4 added recovery and genetic search over control thresholds.
6. B5 added modular homeostatic arbitration and harder environmental probes.

Across all accepted levels, the new complexity stayed above the primitive
bridge, without changing the `SpiderWorld.step()` contract.

## Next Steps

### 1. Formalize B6 From The Accepted B5

Proposed required source:
`artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best`.

B6 objective: maintain B5 retention and turn the non-blocking B5 diagnostics
into architectural axes, without requiring a general improvement over B5.

Recommended direction: a bioinspired risk-foraging arbitration module, with
attention to local threat and corridor affordances.

Suggested initial variants:

- `b6_risk_forage_arbiter_h48_bridge_policy`
- `b6_corridor_affordance_guard_h48_bridge_policy`
- `b6_threat_priority_memory_h48_bridge_policy`
- `b6_genetic_risk_homeostasis_h48_bridge_policy` as fallback if the fixed
  profiles fail

### 2. Promote B5 Diagnostics To Partial B6 Gates

The `food_vs_predator_conflict` and `corridor_gauntlet` scenarios should not
require perfect success at first, but they need to stop being only diagnostic
text.

Suggested B6 gate:

- keep easy `0..4`;
- keep canonical `0..9` retention at a B5-level floor;
- keep `food_deprivation`;
- keep `sleep_vs_exploration_conflict`;
- require measurable progress in `food_vs_predator_conflict`;
- require measurable progress in `corridor_gauntlet`.

Possible partial criteria:

- fewer contacts or prioritized threat in `food_vs_predator_conflict`;
- more steps alive or later death in `corridor_gauntlet`;
- preserve primitive trace in all probes.

### 3. Improve Observability Before More Complexity

Before a larger structural change, it is worth adding trace fields for B6:

- risk-versus-hunger decision;
- threat used by the arbiter;
- reason for suppressing foraging;
- corridor commitment;
- local progress in corridor;
- reason for abandoning shelter;
- return-to-shelter lock.

This avoids repeating an old A-line problem: bad behavior without being able to
separate perception failure, decision failure, and bridge failure.

### 4. Preserve The Discard Regime

B6 should keep discarding candidates that do not pass the gate. The first
accepted candidate becomes `best`; discarded candidates cannot become transfer
sources.

Document for each candidate:

- B5 source;
- coverage;
- `accepted` or `discarded` status;
- discard reason;
- easy/canonical/probe metrics;
- genetic parameters, when present.

### 5. Defer Broad Multi-Seed Validation

It still does not seem time to turn B6 into heavy multi-seed validation. The
most useful gate now is incremental architecture with more informative
scenarios. After B6 or B7 stabilizes risk-foraging and corridor behavior, the
natural next step is a short multi-seed confirmation.

### 6. Possible B7 Line

If B6 maintains B5 and improves the threat/corridor diagnostics, B7 can be a
more structural architecture:

- small explicit recurrent memory;
- local affordance submodule;
- clearer separation between interoception and exteroception;
- gate with canonical `0..9` and conflict/corridor probes as required.

The criterion should remain retention with incremental modular complexity, not
direct competition for higher raw score.
