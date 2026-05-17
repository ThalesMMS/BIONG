# B-Series Strategy

## Purpose

The B series is a parallel diagnostic line. It does not replace the A ladder and it does not promote a checkpoint into the current frontier by itself.

The strategy starts from one observation: commit `36191725f3fa18921509133c9167108193ef87b6` could learn because its action space already contained useful ecological intentions. The old policy did not learn primitive navigation from scratch. It learned when to choose labels such as `MOVE_TO_FOOD`, `MOVE_TO_SHELTER`, `EAT`, and `SLEEP`, while the legacy world converted some of those labels into useful behavior.

The B line keeps that lesson, but makes the abstraction explicit and auditable:

- `B0 legacy` reproduces the old semantic benchmark in an isolated harness.
- `B0 current bridge` uses the current world and current primitive action space, but lets the network choose an internal semantic action first.
- `B1+` hardens the setup gradually and initializes from the previous B checkpoint instead of training from scratch.

## B0 Legacy Versus B0 Current Bridge

### `b0_legacy_semantic_policy`

`B0 legacy` is a reproduction harness for the old benchmark. It uses `LegacyB0Simulation`, not the current `SpiderWorld`.

Its public action space inside that harness is exactly the old six semantic actions:

- `MOVE_TO_FOOD`
- `MOVE_TO_SHELTER`
- `EXPLORE`
- `STAY`
- `EAT`
- `SLEEP`

In this mode, `MOVE_TO_FOOD` and `MOVE_TO_SHELTER` are macro actions. The legacy world executes a deterministic local step toward the nearest food or shelter. `EAT` and `SLEEP` are also explicit semantic actions with dense reward. The learning problem is therefore short-horizon and highly aligned with the reward.

This mode answers one question only: can the old level of abstraction still reproduce the old competence signal?

Current smoke result:

- `seed=7`
- `episodes=600`
- `max_steps=120`
- evaluation `mean_food=9.67`
- evaluation `mean_sleep=43.33`
- evaluation `survival_rate=1.0`

That result is comparable to the old commit because it uses the same abstraction level and benchmark family. It is not evidence that the current ecological problem is solved.

### `b0_current_bridge_policy`

`B0 current bridge` runs in the current world through `architecture="b_series"` and `b_mode="current_bridge"`.

The public name stays `B0 current bridge`, but this rung now behaves as a deliberately simple legacy-direct bridge. The learned policy still produces logits over the six old semantic labels for audit, but B0-current uses a trace-visible semantic controller to choose the active legacy intention while the network is still fragile. The trace records both the learned semantic label and the controller-selected semantic label so this is not hidden steering.

The bridge then converts the selected label into one current primitive action:

- `MOVE_TO_FOOD`: choose an unblocked primitive move that improves food distance or food signal.
- `MOVE_TO_SHELTER`: choose an unblocked primitive move that improves shelter or geodesic progress.
- `EXPLORE`: choose an unblocked seeded primitive move.
- `STAY`, `EAT`, `SLEEP`: submit `STAY`.

The current public action space remains the nine primitive actions:

- `MOVE_UP`
- `MOVE_DOWN`
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `STAY`
- `ORIENT_UP`
- `ORIENT_DOWN`
- `ORIENT_LEFT`
- `ORIENT_RIGHT`

`B0 current bridge` does not add `MOVE_TO_FOOD` or `MOVE_TO_SHELTER` to `world.step()`. The bridge is outside the world and trace-visible. That is the main difference from the legacy harness.

This mode answers a narrower question than the first B0-current attempt: can the current world run a legacy-like semantic phase loop with primitive actions at the leaf, before asking the neural semantic head to own the full phase policy unaided?

## Why The Performance Is Not Directly Comparable

`B0 legacy` and `B0 current bridge` measure different difficulties.

`B0 legacy` has planner-like actions in the environment. When the policy chooses `MOVE_TO_FOOD`, the environment takes a useful step toward food. When it chooses `MOVE_TO_SHELTER`, the environment takes a useful step toward shelter. The policy mainly learns intention selection.

`B0 current bridge` keeps the semantic choice inside the controller, then submits a primitive action to the current world. It inherits the harder ecology: current shelter geometry, explicit predator state, primitive locomotion, orientation actions, local blocked movement, delayed consequences, and current physiology. Even though the bridge helps with local movement, eating and sleeping are no longer old semantic world commands. They are current-world consequences of being in the right state and submitting primitive `STAY`.

Performance interpretation:

- If `B0 legacy` works, the old abstraction level is reproducible.
- If `B0 current bridge` improves over A0 under the current runner, the simple semantic controller plus primitive bridge is helping in the current ecology.
- If `B0 current bridge` moves semantic labels but not executed primitive behavior, the next diagnosis is ownership or bridge mapping, not another auxiliary head.
- If `B0 current bridge` survives but cannot produce second-cycle behavior, the next step is a B1 change initialized from B0, not a fresh-from-zero controller.

## Runtime Audit Contract

Every B-current trace should expose enough information to distinguish policy ownership from hidden steering:

- `b_level`
- `b_effective_level`
- `b_mode`
- `semantic_action`
- `learned_semantic_action`
- `semantic_action_source`
- `semantic_action_reason`
- `semantic_override_count`
- `semantic_logits`
- `bridge_primitive_action`
- `bridge_reason`
- `blocked_mask`
- `food_delta_used`
- `shelter_delta_used`
- `external_override_count`

`external_override_count` should stay low. Blocked movement masking is allowed. Routine food seeking and shelter return must come from the selected semantic action plus the bridge, not from final A-line runtime biases.

## Transfer Learning Rule For B1+

B levels after B0 must not train from scratch by default.

The project calls this progression **Evolution**: a simpler source architecture
is saved as an explicit evolutionary source, then the next B level initializes
from that saved state instead of starting from random weights. The name is used
in the GUI and checkpoint metadata to make the biological analogy visible
without changing the technical rule: Evolution means transfer from the previous
B level, with auditable source metadata.

Each B1+ config must declare:

- `b_parent_level`
- `b_transfer_source_checkpoint`
- `b_transfer_min_coverage`
- `b_transfer_allow_low_coverage`

The default source should be the previous B level's `best` checkpoint for the same scenario and seed. A `last` checkpoint can be used only when the fallback is recorded. Transfer is partial by stable parameter name and compatible shape overlap. New or incompatible parameters keep normal initialization.

Training aborts when:

- the checkpoint source is missing
- transfer coverage is below `b_transfer_min_coverage`, default `0.50`
- low coverage is not explicitly allowed

The checkpoint metadata records the transfer report: coverage, loaded keys, partial keys, skipped keys, initialized keys, source fingerprint, and low-coverage override status.

The GUI writes Evolution snapshots under
`artifacts/gui_evolution_snapshots/<variant>/<timestamp>/`. For
`B0 current bridge`, the snapshot contains a normal `SpiderBrain` checkpoint
plus `gui_snapshot.json` with `b_level`, `b_mode`, model id, source variant,
and transfer compatibility. For `B0 legacy`, the snapshot persists the legacy
weights and state for reproduction, but marks `transfer_compatible=false`
because there is no silent shape-compatible path from the legacy semantic
harness into B1-current.

## Current B0-Current Simple Frontier

The current `b0_current_bridge_policy` is intentionally simpler than the earlier owned-option experiments. It uses a legacy-direct semantic controller for the six B0 intentions, keeps the neural semantic logits in trace for audit, and sends only primitive actions to the current world.

As of the 2026-05-13 smoke pass with seed `7`, reward profile `ecological`, operational profile `default_v1`, and noise profile `none`:

- `continuous_survival_easy_v1`: passed the continuous-survival diagnostic with 432 steps, 14.4 simulated days, 21 food events, 100 sleep events, 14 shelter exits/returns, 1 predator contact, and final health 1.0.
- `continuous_survival_canonical`: reached the 300-step horizon alive with 10 completed cycles, 15 food events, 39 sleep events, 7 shelter exits/returns, and final health 0.271, but did not pass because predator contacts remained too high at 16.

Interpretation: the simple B0-current bridge has restored the core loop in the current world. The remaining canonical blocker is acute predator-contact avoidance, not the old failure mode where the agent could not produce repeated food/rest/shelter cycles.

## B1 Capacity-Only Evolution

`B1` starts as the smallest hardening step after B0-current. It keeps the same
six semantic labels and the same primitive bridge, but increases the B-series
hidden size and initializes from the saved B0-current checkpoint. Unlike
`B0-current`, B1 does not use the legacy-direct semantic controller: the active
semantic action is the network-selected semantic action, and traces should show
`semantic_action_source="network_policy"` with `semantic_override_count=0`.

The first capacity-only attempts are:

- `b1_capacity_h48_bridge_policy`: hidden size 48, expected transfer coverage
  around 0.667 from B0-current.
- `b1_capacity_h64_bridge_policy`: hidden size 64, expected transfer coverage
  around 0.500 from B0-current.
- If both capacity-only attempts are discarded, the runner tries
  `b1_threat_guard_bridge_policy`: hidden size 48 with the same transfer source
  and primitive bridge, plus a B1-local threat guard that records
  `semantic_action_source="b1_threat_guard_controller"`.

Both require `b_parent_level=0`, `b_transfer_source_checkpoint`, and
`b_transfer_min_coverage=0.50` with no low-coverage override. The default source
path is `artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best`.

The local runner is:

```bash
python3 -m spider_cortex_sim.b_series_evolution
```

It creates the B0-current source checkpoint when absent, runs the B1 attempts in
order, accepts the first attempt that passes
`continuous_survival_easy_v1`, and writes `attempt_report.json` plus
`b1_evolution_summary.json` (`b1_capacity_summary.json` is kept as a
compatibility alias). A failed attempt is not silently reused as success: it is
saved under a `discarded` checkpoint directory with a discard reason, then the
next attempt is tried. If all registered attempts fail, the next B1 line should
be recurrent and threat-aware while still transferring from the same B0 source.

## B2 Temporal-Threat Evolution

`B2` starts from the accepted B1 checkpoint, not from B0 or from scratch. The
default source is
`artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best`.

The first B2 attempts are temporal-threat variants:

- `b2_temporal_threat_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B1.
- `b2_temporal_threat_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b2_temporal_threat_h64_bridge_policy`: hidden size 64, expected transfer
  coverage around 0.750.

Each B2 attempt keeps the same six semantic labels and primitive bridge. The
active controller records `semantic_action_source="b2_temporal_threat_controller"`
and adds trace fields for current threat, temporal threat, predator-memory
pressure, and predator-trace pressure.

The local B2 runner is:

```bash
python3 -m spider_cortex_sim.b_series_evolution --target-level 2 --b2-training-episodes 48
```

B2 acceptance requires the easy gate plus canonical progress over the accepted
B1 baseline. The canonical progress gate is at least 100 steps, no more than 4
predator contacts, at least 3 food events, 3 sleep events, 5 shelter entries,
and a primitive-only trace. Completing the canonical horizon alive accepts the
canonical progress gate automatically.

Current accepted B2 smoke result, with seed `7`, 48 B2 training episodes, and
canonical evaluation episode `0`:

- accepted variant: `b2_temporal_threat_h48_bridge_policy`
- source checkpoint:
  `artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best`
- checkpoint:
  `artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best`
- transfer coverage: `1.0`
- `continuous_survival_easy_v1`: 432 steps alive, 19 food events, 76 sleep
  events, 15 shelter entries, 1 predator contact, final health 1.0
- `continuous_survival_canonical`: 300 steps alive, 13 food events, 46 sleep
  events, 12 shelter entries, 9 predator contacts, final health 0.063414

## B3 Contact-Memory Evolution

`B3` starts from the accepted B2 checkpoint:
`artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best`.
It keeps the same primitive world action space, the same six internal semantic
actions, and the same primitive bridge.

The first B3 attempts are contact-memory variants:

- `b3_contact_memory_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B2.
- `b3_contact_memory_strict_h48_bridge_policy`: same size with more conservative
  post-contact thresholds.
- `b3_contact_memory_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b3_recurrent_guard_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B2, with an explicit per-episode recurrent guard profile.

The active controller records
`semantic_action_source="b3_contact_memory_controller"` and adds per-episode
trace fields for contact cooldown, post-food cooldown, hunger drop, and the B3
controller profile.

The local B3 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 3 --b3-training-episodes 48
```

B3 acceptance requires the easy gate plus robust canonical checks on evaluation
episodes `0` and `1`. Episode `0` must complete 300 steps alive with no more than
6 predator contacts and non-trivial food/sleep/shelter cycles. Episode `1` must
reach at least 100 steps with no more than 4 predator contacts, at least 3 food
events, 3 sleep events, 5 shelter entries, and a primitive-only trace. For B3,
completing the canonical horizon does not bypass the predator-contact limit.

Current B3 smoke result, with seed `7` and 48 B3 training episodes:

- summary: `artifacts/b_series/evolution/b3_evolution_summary.json`
- status: `accepted`
- accepted variant: `b3_recurrent_guard_h48_bridge_policy`
- checkpoint:
  `artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best`
- transfer coverage: `1.0`
- `b3_contact_memory_h48_bridge_policy`: passed easy, but failed canonical
  episode `0` on survival/horizon and episode `1` on minimum steps
- `b3_contact_memory_strict_h48_bridge_policy`: passed canonical episode `0`,
  but failed easy and exceeded the contact limit in canonical episode `1`
- `b3_contact_memory_h56_bridge_policy`: passed easy, but failed canonical
  episode `0` on survival/horizon and episode `1` on minimum steps
- `b3_recurrent_guard_h48_bridge_policy`: passed easy and canonical episodes
  `0` and `1`
- easy: 432 steps alive, 19 food events, 76 sleep events, 15 shelter entries, 1
  predator contact, final health 1.0
- canonical episode `0`: 300 steps alive, 15 food events, 39 sleep events, 10
  shelter entries, 2 predator contacts, final health 0.827079
- canonical episode `1`: 300 steps alive, 16 food events, 65 sleep events, 13
  shelter entries, 1 predator contact, final health 1.0

## B4 Multi-Episode Recovery Evolution

`B4` starts from the accepted B3 checkpoint:
`artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best`.
It keeps the primitive-only world boundary, the six internal semantic actions,
and the B-series primitive bridge.

The B4 line tests recovery-balance variants first, then a small genetic search
if the fixed profiles fail:

- `b4_recovery_balance_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B3.
- `b4_predator_exit_memory_h48_bridge_policy`: same size with more conservative
  shelter-exit thresholds after threat/contact.
- `b4_recovery_balance_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b4_genetic_recovery_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0, with the winning controller thresholds saved as
  `b_controller_params`.

The B4 controller records either
`semantic_action_source="b4_recovery_balance_controller"` or
`semantic_action_source="b4_genetic_recovery_controller"`. Its trace adds
`b4_controller_profile`, `b4_recovery_pressure`, `b4_sleep_hold`,
`b4_exit_blocked`, `b4_hunger_release`, and optional GA generation/candidate
metadata.

The local B4 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 4 --b4-training-episodes 48 --b4-workers 4 --b4-search hybrid --b4-ga-population 12 --b4-ga-generations 4
```

B4 acceptance requires easy episodes `0..4` to complete 432 steps alive with
non-trivial food/sleep/shelter cycles and primitive-only traces. Canonical
episodes `0..9` are the broader environment diagnostic for B4; this stage
validates retention with a more modular recovery controller, not mandatory
aggregate improvement over B3. Episodes `0` and `1` remain B3 anchors and must
complete 300 steps alive with at most 4 predator contacts and non-trivial
cycles. The `0..9` aggregate must retain the confirmed B3 baseline: at least 4
of 10 canonical episodes complete the horizon, minimum steps are at least 49,
total predator contacts are no more than 26, and food/sleep/shelter cycles are
present in the large majority of episodes.

Current B4 smoke result, with seed `7`, 48 B4 training episodes, workers `4`,
and hybrid search:

- summary: `artifacts/b_series/evolution/b4_evolution_summary.json`
- status: `accepted`
- accepted variant: `b4_genetic_recovery_h48_bridge_policy`
- accepted checkpoint:
  `artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best`
- transfer source:
  `artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best`
- transfer coverage: 1.0, parent level 3, target level 4
- all fixed and genetic candidates passed easy episodes `0..4`
- discarded fixed candidates:
  `b4_recovery_balance_h48_bridge_policy` (0 completed horizons, minimum 50
  steps, 11 total contacts),
  `b4_predator_exit_memory_h48_bridge_policy` (3 completed horizons, minimum
  55 steps, 11 total contacts), and
  `b4_recovery_balance_h56_bridge_policy` (0 completed horizons, minimum 50
  steps, 11 total contacts)
- accepted GA candidate: 4 completed horizons, minimum 49 steps, 24 total
  contacts, 9 food-cycle episodes, 9 sleep-cycle episodes, and 10
  shelter-cycle episodes
- canonical anchors passed: episode `0` completed 300 steps alive with 14 food,
  54 sleep, 10 shelter, and 2 predator contacts; episode `1` completed 300
  steps alive with 15 food, 46 sleep, 11 shelter, and 4 predator contacts
- remaining diagnostic weakness: canonical episode `8` still ends at 49 steps
  with only 2 food events and no sleep, so B5 should target explicit
  homeostatic recovery/generalization rather than treating B4 as broad
  multi-seed robustness

## B5 Homeostatic Arbiter Evolution

`B5` starts from the accepted B4 checkpoint:
`artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best`.
It keeps the primitive-only world boundary, the six internal semantic actions,
and the B-series primitive bridge.

The B5 line adds a homeostatic arbiter above B4 recovery balance:

- `b5_homeostatic_arbiter_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B4.
- `b5_circadian_recovery_h48_bridge_policy`: same size, more conservative
  sleep/recovery thresholds.
- `b5_homeostatic_arbiter_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b5_genetic_homeostasis_h48_bridge_policy`: hidden size 48, used only if fixed
  profiles fail, with winning thresholds saved in `b_controller_params`.

The B5 controller records either
`semantic_action_source="b5_homeostatic_arbiter_controller"` or
`semantic_action_source="b5_genetic_homeostasis_controller"`. Its trace adds
`b5_controller_profile`, `b5_hunger_urgency`, `b5_sleep_pressure`,
`b5_recovery_debt`, `b5_threat_gate`, `b5_sleep_bout_lock`,
`b5_forage_commitment_lock`, `b5_homeostatic_decision`, and optional GA
generation/candidate metadata.

The local B5 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 5 --b5-training-episodes 48 --b5-workers 4 --b5-search hybrid --b5-ga-population 12 --b5-ga-generations 4
```

B5 acceptance keeps the B4 retention gate for easy `0..4` and canonical `0..9`,
then adds blocking probes for `food_deprivation` and
`sleep_vs_exploration_conflict` episodes `0..2`. `food_vs_predator_conflict`
and `corridor_gauntlet` are recorded as non-blocking diagnostics for B6.

Run status for seed `7`, 48 training episodes, hybrid search:

- discarded fixed candidates:
  `b5_homeostatic_arbiter_h48_bridge_policy`,
  `b5_circadian_recovery_h48_bridge_policy`, and
  `b5_homeostatic_arbiter_h56_bridge_policy`. All passed easy and both
  blocking probes, but failed canonical retention through insufficient
  completed horizons and shelter-cycle episodes.
- accepted candidate:
  `b5_genetic_homeostasis_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best`,
  parent level `4`, target level `5`, coverage `1.0`, low coverage disabled.
- accepted canonical aggregate: 5 completed horizons, minimum 52 steps,
  24 total predator contacts, 9 food-cycle episodes, 9 sleep-cycle episodes,
  and 10 shelter-cycle episodes.
- accepted probe aggregate: `food_deprivation` passed with 2/3 progress
  episodes; `sleep_vs_exploration_conflict` passed with the recovery-movement
  metric exposed and 3/3 post-recovery movement episodes.
- non-blocking diagnostics still fail behavior scorers for
  `food_vs_predator_conflict` and `corridor_gauntlet`; these remain B6
  guidance, not B5 blockers.

## B6 Risk-Corridor And Recurrent-Memory Evolution

`B6` starts from the accepted B5 checkpoint:
`artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best`.
It remains an incremental B-series rung: no new primitive actions, no new
semantic actions, and the world still receives only primitive `ACTIONS`.

B6 tests two controller families strongly and independently:

- risk-corridor: `b6_risk_forage_arbiter_h48_bridge_policy`,
  `b6_corridor_survival_guard_h48_bridge_policy`,
  `b6_threat_priority_memory_h48_bridge_policy`,
  `b6_risk_corridor_h56_bridge_policy`, and a GA fallback
  `b6_genetic_risk_corridor_h48_bridge_policy`.
- recurrent-memory: `b6_recurrent_context_h48_bridge_policy`,
  `b6_recurrent_threat_homeostasis_h48_bridge_policy`,
  `b6_recurrent_corridor_guard_h48_bridge_policy`,
  `b6_recurrent_context_h56_bridge_policy`, and a GA fallback
  `b6_genetic_recurrent_memory_h48_bridge_policy`.
- fusion: `b6_fused_risk_recurrent_h48_bridge_policy`, run only when both
  families produce accepted candidates. The fusion still transfers from the
  B5 checkpoint, not from another B6 candidate.

The B6 runtime starts with the B5 homeostatic arbiter and adds a modular layer
for risk pressure, threat priority, foraging suppression, corridor commitment,
corridor progress memory, recurrent threat state, and return locks. It emits
B6-specific `action_center` trace payloads so the existing conflict scorers can
measure threat arbitration without changing the global scorer contract.

The local B6 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 6 --b6-training-episodes 64 --b6-workers 4 --b6-search exhaustive --b6-ga-population 24 --b6-ga-generations 8 --b6-finalists 6
```

B6 acceptance keeps B5 retention gates for easy `0..4`, canonical `0..9`,
`food_deprivation`, and `sleep_vs_exploration_conflict`. It then adds partial
progress gates for:

- `food_vs_predator_conflict`: alive without predator contact, scorer-visible
  danger in at least 2/3 episodes, and threat priority or foraging suppression
  in at least 2/3 episodes.
- `corridor_gauntlet`: primitive trace, zero predator contacts, food-distance
  progress in at least 2/3 episodes, and at least 2/3 episodes either reaching
  16+ steps, surviving to the end, or improving the B5 corridor baseline food
  progress while retaining the B5 step floor. This fallback is explicit because
  the current corridor probe starts with food 16 moves away and the physiology
  budget kills even hand-driven direct routes at tick 14.

Promotion is deterministic: fixed candidates are evaluated in priority order,
both GA families run even if a fixed candidate passes, then fusion is attempted
only if both families have an accepted candidate. Failed candidates are saved
under `discarded`; the promoted candidate is rerun and saved under `best`.

Validated B6 result:

- accepted variant: `b6_fused_risk_recurrent_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b6_fused_risk_recurrent_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best`,
  parent level `5`, target level `6`, coverage `1.0`, low coverage disabled.
- exhaustive run: all eight fixed candidates passed; both GA families passed
  after 24x8 screening plus six finalists; fusion passed and was promoted.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- accepted B6 probes: `food_vs_predator_conflict` passed with 3/3 full scorer
  successes; `corridor_gauntlet` passed partial progress with 3/3 progress
  episodes, 3/3 food-progress-over-B5 episodes, and zero predator contacts.

## B7 Affordance-Budget Evolution

`B7` starts from the accepted B6 fused checkpoint:
`artifacts/b_series/evolution/b6_fused_risk_recurrent_h48_bridge_policy/seed_7/best`.
It keeps the public primitive action space, the six internal semantic actions,
and the same B-series primitive bridge.

The B7 line adds a lightweight bioinspired affordance and homeostatic-budget
layer above B6. It estimates local corridor viability from food distance,
return distance, health, hunger, threat, and recent progress, then records an
explicit decision to continue, abort/return, recover before crossing, or
preserve B6 behavior. These decisions still map to existing semantic actions:
`MOVE_TO_FOOD`, `MOVE_TO_SHELTER`, `STAY`, `EAT`, and `SLEEP`.

The B7 variants are:

- `b7_affordance_budget_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B6.
- `b7_energy_budget_corridor_h48_bridge_policy`: same size, more conservative
  physiological margin.
- `b7_recurrent_affordance_h48_bridge_policy`: same size, short recurrent
  memory of corridor progress.
- `b7_affordance_budget_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b7_genetic_affordance_budget_h48_bridge_policy`: GA fallback over B7
  thresholds, only used if fixed variants fail.

The B7 trace adds `b7_controller_profile`, `b7_affordance_state`,
`b7_energy_budget`, `b7_budget_margin`, `b7_food_steps_estimate`,
`b7_return_steps_estimate`, `b7_corridor_viability`, `b7_abort_return`,
`b7_commitment_lock`, `b7_decision`, and optional GA metadata.

The local B7 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 7 --b7-training-episodes 64 --b7-workers 4 --b7-search hybrid --b7-ga-population 24 --b7-ga-generations 8 --b7-finalists 6
```

B7 acceptance retains the B6/B5 gates for easy `0..4`, canonical `0..9`,
`food_deprivation`, `sleep_vs_exploration_conflict`, and
`food_vs_predator_conflict`. Its new blocking gate is partial progress in
`corridor_gauntlet` episodes `0..2`: zero predator contacts, primitive trace
only, at least 2/3 episodes with food-distance delta at least 13, at least 2/3
episodes with an explicit B7 viability decision, and at least 2/3 episodes with
some improvement axis over the B6 corridor baseline.

Validated B7 result:

- accepted variant: `b7_affordance_budget_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b7_affordance_budget_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b6_fused_risk_recurrent_h48_bridge_policy/seed_7/best`,
  parent level `6`, target level `7`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B7 candidates passed; the first priority
  candidate was promoted, so GA was not run.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation` passed with 2/3 progress episodes;
  `sleep_vs_exploration_conflict` passed with 3/3 post-recovery movement
  episodes; `food_vs_predator_conflict` passed with 3/3 full scorer successes.
- B7 corridor progress: `corridor_gauntlet` episodes `0..2` all had zero
  predator contacts, food-distance delta 13, explicit B7 decisions, and
  abort/return recorded before death. The corridor itself is still not solved;
  B7 validates the viability/abort signal needed for the next rung.

## B8 Spatial-Affordance Map Evolution

`B8` starts from the accepted B7 checkpoint:
`artifacts/b_series/evolution/b7_affordance_budget_h48_bridge_policy/seed_7/best`.
It preserves the primitive-only world boundary and the same six internal
semantic actions.

B8 adds a small spatial/hippocampal-style layer above B7. The new module
tracks a local place-memory scalar, estimates food and return vectors from
local transition consequences, computes a dead-end risk estimate, and records
whether the corridor decision is mapped to a spatial affordance rather than
only to a homeostatic budget signal.

The B8 variants are:

- `b8_spatial_affordance_map_h48_bridge_policy`: hidden size 48, expected
  transfer coverage 1.0 from B7.
- `b8_return_vector_h48_bridge_policy`: same size, more aggressive return
  execution when the return vector is strong.
- `b8_corridor_place_memory_h48_bridge_policy`: same size, stronger recurrent
  place/progress memory.
- `b8_spatial_affordance_map_h56_bridge_policy`: hidden size 56, expected
  transfer coverage around 0.857.
- `b8_genetic_spatial_affordance_h48_bridge_policy`: fallback over B8 spatial
  thresholds.

The B8 trace adds `b8_controller_profile`, `b8_spatial_map_state`,
`b8_local_affordance_score`, `b8_return_vector_strength`,
`b8_corridor_dead_end_risk`, `b8_abort_executed`, `b8_place_memory`,
`b8_decision`, and optional GA metadata.

The local B8 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 8 --b8-training-episodes 64 --b8-workers 4 --b8-search hybrid --b8-ga-population 24 --b8-ga-generations 8 --b8-finalists 6
```

B8 acceptance keeps all B7 retention and probe gates, then requires the corridor
gate to expose B8-specific spatial decisions in at least 2/3 episodes. A B7
clone without `b8_decision` and `b8_spatial_map_state` is rejected even if it
matches the B7 food-distance progress.

Validated B8 result:

- accepted variant: `b8_spatial_affordance_map_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b8_spatial_affordance_map_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b7_affordance_budget_h48_bridge_policy/seed_7/best`,
  parent level `7`, target level `8`, coverage `1.0`, low coverage disabled.
- fixed search: `b8_spatial_affordance_map_h48_bridge_policy`,
  `b8_corridor_place_memory_h48_bridge_policy`, and
  `b8_spatial_affordance_map_h56_bridge_policy` passed; the return-vector
  variant was discarded because it broke the inherited B7 corridor-progress
  gate.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` all retained the B7 acceptance behavior.
- B8 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B7
  partial corridor progress and added 3/3 explicit B8 spatial decisions,
  3/3 spatial-map episodes, and 3/3 mapped-progress episodes. The corridor
  remains a partial-progress problem, but now the controller exposes an explicit
  spatial affordance representation for B9 to use.

## B9 Waypoint-Planner Evolution

`B9` starts from the accepted B8 checkpoint:
`artifacts/b_series/evolution/b8_spatial_affordance_map_h48_bridge_policy/seed_7/best`.
It keeps the primitive-only public world boundary and the same six internal
semantic actions.

B9 adds a small route/waypoint layer above B8. The new module treats the B8
spatial affordance signal as local map evidence, maintains a decaying route
memory, estimates path-integration confidence, and records whether the agent is
committing to a food waypoint, continuing a locked waypoint, replanning a return,
or preserving the B8 spatial decision.

The B9 variants are:

- `b9_waypoint_planner_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B8.
- `b9_path_integration_h48_bridge_policy`: same size, stronger path-integration
  gain and lower route-confidence threshold.
- `b9_route_memory_h48_bridge_policy`: same size, slower route-memory decay and
  longer waypoint commitment.
- `b9_waypoint_planner_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b9_genetic_waypoint_planner_h48_bridge_policy`: fallback over B9 route and
  waypoint thresholds.

The B9 trace adds `b9_controller_profile`, `b9_route_state`,
`b9_route_confidence`, `b9_waypoint_lock`, `b9_path_integrator`,
`b9_replan_signal`, `b9_decision`, and optional GA metadata.

The local B9 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 9 --b9-training-episodes 64 --b9-workers 4 --b9-search hybrid --b9-ga-population 24 --b9-ga-generations 8 --b9-finalists 6
```

B9 acceptance keeps all B8 retention and probe gates, then requires the
corridor gate to expose route-level B9 evidence in at least 2/3 episodes:
explicit B9 waypoint decisions, non-empty route state, and a positive waypoint
lock. A B8 clone without B9 route/waypoint evidence is rejected even if it
matches the inherited corridor progress.

Validated B9 result:

- accepted variant: `b9_waypoint_planner_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b9_waypoint_planner_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b8_spatial_affordance_map_h48_bridge_policy/seed_7/best`,
  parent level `8`, target level `9`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B9 candidates passed; the priority h48 waypoint
  planner was promoted and the non-promoted accepted candidates were retained
  under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` all retained the B8 acceptance behavior.
- B9 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B8
  partial corridor progress and added 3/3 explicit B9 decisions, 3/3 route-state
  episodes, and 3/3 locked-waypoint episodes. The corridor is still a partial
  progress environment, but B9 validates an auditable route/waypoint substrate
  for B10.

## B10 Prospective-Replay Evolution

`B10` starts from the accepted B9 checkpoint:
`artifacts/b_series/evolution/b9_waypoint_planner_h48_bridge_policy/seed_7/best`.
It preserves the primitive-only public action boundary and the same six
internal semantic actions.

B10 adds a hippocampal/prefrontal-style prospective replay layer above B9. It
uses the B9 route confidence and waypoint lock as the current route hypothesis,
maintains a decaying replay-memory value, estimates a short rollout depth,
computes prospective route value versus abort pressure, and records whether the
agent commits to a replayed route, continues a replay commitment, aborts a
failed rollout, or preserves the B9 decision.

The B10 variants are:

- `b10_prospective_replay_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B9.
- `b10_value_route_evaluator_h48_bridge_policy`: same size, stronger value
  weighting and lower value threshold.
- `b10_replay_planner_h48_bridge_policy`: same size, stronger replay memory and
  longer plan commitment.
- `b10_prospective_replay_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b10_genetic_replay_planner_h48_bridge_policy`: fallback over B10 replay and
  value thresholds.

The B10 trace adds `b10_controller_profile`, `b10_replay_state`,
`b10_prospective_value`, `b10_rollout_depth`, `b10_replay_memory`,
`b10_plan_commitment`, `b10_abort_signal`, `b10_decision`, and optional GA
metadata.

The local B10 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 10 --b10-training-episodes 64 --b10-workers 4 --b10-search hybrid --b10-ga-population 24 --b10-ga-generations 8 --b10-finalists 6
```

B10 acceptance keeps all B9 retention and probe gates, then requires the
corridor gate to expose replay-level B10 evidence in at least 2/3 episodes:
explicit B10 decisions, non-empty replay state, positive plan commitment, and a
positive prospective-value signal. A B9 clone without B10 replay evidence is
rejected even if it matches inherited route/waypoint progress.

Validated B10 result:

- accepted variant: `b10_prospective_replay_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b10_prospective_replay_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b9_waypoint_planner_h48_bridge_policy/seed_7/best`,
  parent level `9`, target level `10`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B10 candidates passed; the priority h48
  prospective-replay candidate was promoted and the other accepted candidates
  were retained under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` all retained the B9 acceptance behavior.
- B10 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B9
  partial corridor progress and added 3/3 explicit B10 decisions, 3/3 replay
  states, 3/3 committed-plan episodes, and 3/3 prospective-value episodes. The
  corridor remains unsolved, but B10 validates an auditable prospective replay
  substrate for a later rung with richer observation or local affordance search.

## B11 Confidence-Arbiter Evolution

`B11` starts from the accepted B10 checkpoint:
`artifacts/b_series/evolution/b10_prospective_replay_h48_bridge_policy/seed_7/best`.
It keeps the primitive-only public action space and the six internal semantic
actions.

B11 adds a lightweight metacognitive/neuromodulatory confidence layer above
B10. It treats the B10 replay value and B9 route confidence as plan-evidence,
tracks a decaying confidence memory, estimates uncertainty from abort pressure
and weak route confidence, then records whether the current route is a
high-confidence plan, a gated uncertain plan, a continuing confidence lock, or
unchanged B10 behavior.

The B11 variants are:

- `b11_confidence_arbiter_h48_bridge_policy`: hidden size 48, expected transfer
  coverage 1.0 from B10.
- `b11_uncertainty_gate_h48_bridge_policy`: same size, lower uncertainty and
  confidence thresholds.
- `b11_neuromodulated_replay_h48_bridge_policy`: same size, stronger
  neuromodulatory gain and slower confidence decay.
- `b11_confidence_arbiter_h56_bridge_policy`: hidden size 56, expected transfer
  coverage around 0.857.
- `b11_genetic_confidence_gate_h48_bridge_policy`: fallback over confidence and
  uncertainty thresholds.

The B11 trace adds `b11_controller_profile`, `b11_confidence_state`,
`b11_plan_confidence`, `b11_uncertainty`, `b11_neuromod_signal`,
`b11_confidence_lock`, `b11_decision`, and optional GA metadata.

The local B11 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 11 --b11-training-episodes 64 --b11-workers 4 --b11-search hybrid --b11-ga-population 24 --b11-ga-generations 8 --b11-finalists 6
```

B11 acceptance keeps all B10 retention and probe gates, then requires the
corridor gate to expose confidence-level B11 evidence in at least 2/3 episodes:
explicit B11 decisions, non-empty confidence state, positive confidence lock,
and positive neuromodulatory signal. A B10 clone without confidence evidence is
rejected even if it matches inherited replay progress.

Validated B11 result:

- accepted variant: `b11_confidence_arbiter_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b11_confidence_arbiter_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b10_prospective_replay_h48_bridge_policy/seed_7/best`,
  parent level `10`, target level `11`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B11 candidates passed; the priority h48
  confidence arbiter was promoted and the other accepted candidates were
  retained under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` all retained the B10 acceptance behavior.
- B11 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B10
  partial corridor progress and added 3/3 explicit B11 decisions, 3/3 confidence
  states, 3/3 confidence-lock episodes, and 3/3 neuromodulatory-signal episodes.
  The next useful hardening is likely richer uncertainty/affordance observation
  or a structural local search module, because the inherited corridor behavior
  remains partial progress rather than full completion.

## B12 Predictive-Attention Evolution

`B12` starts from the accepted B11 checkpoint:
`artifacts/b_series/evolution/b11_confidence_arbiter_h48_bridge_policy/seed_7/best`.
It keeps the primitive-only public action space and the same six internal
semantic actions.

B12 adds a predictive-attention layer above B11. It treats B11 confidence as a
top-down plan prior, compares it with expected food-progress evidence from the
current affordance trace, tracks a decaying attention memory, and records
whether the current behavior is an attended food affordance, a prediction-error
return check, a continuing attention lock, or unchanged B11 behavior.

The B12 variants are:

- `b12_predictive_attention_h48_bridge_policy`: hidden size 48, expected
  transfer coverage 1.0 from B11.
- `b12_active_inference_gate_h48_bridge_policy`: same size, lower attention and
  prediction-error thresholds.
- `b12_affordance_attention_h48_bridge_policy`: same size, stronger local
  affordance gain and slower attention decay.
- `b12_predictive_attention_h56_bridge_policy`: hidden size 56, expected
  transfer coverage around 0.857.
- `b12_genetic_attention_gate_h48_bridge_policy`: fallback over attention and
  prediction-error thresholds.

The B12 trace adds `b12_controller_profile`, `b12_attention_state`,
`b12_prediction_error`, `b12_attention_gain`, `b12_expected_progress`,
`b12_search_lock`, `b12_decision`, and optional GA metadata.

The local B12 runner is:

```bash
python3.13 -m spider_cortex_sim.b_series_evolution --target-level 12 --b12-training-episodes 64 --b12-workers 4 --b12-search hybrid --b12-ga-population 24 --b12-ga-generations 8 --b12-finalists 6
```

B12 acceptance keeps all B11 retention and probe gates, then requires the
corridor gate to expose attention/prediction-level B12 evidence in at least 2/3
episodes: explicit B12 decisions, non-empty attention state, positive attention
lock, and a positive attention or prediction-error signal. A B11 clone without
B12 attention evidence is rejected.

Validated B12 result:

- accepted variant: `b12_predictive_attention_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b12_predictive_attention_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b11_confidence_arbiter_h48_bridge_policy/seed_7/best`,
  parent level `11`, target level `12`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B12 candidates passed; the priority h48
  predictive-attention candidate was promoted and the other accepted candidates
  were retained under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` all retained the B11 acceptance behavior.
- B12 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B11
  partial corridor progress and added 3/3 explicit B12 decisions, 3/3 attention
  states, 3/3 attention-lock episodes, and 3/3 prediction-signal episodes. The
  corridor remains partial progress; the new substrate exposes top-down
  attention and prediction-error evidence for a future structural local-search
  or richer observation phase.

## B13: Local Affordance Search

B13 adds a structural local-search layer above B12 predictive attention. It
keeps the same primitive public action boundary and the same six internal
semantic actions, but records a local route score, affordance samples, search
memory, dead-end score, and a short search lock. The biological analogy is a
small sensorimotor affordance sampler layered on top of attention: B12 decides
what is salient, while B13 tracks whether the immediately available corridor
route remains locally viable.

Implementation:

- variants: `b13_local_affordance_search_h48_bridge_policy`,
  `b13_counterfactual_route_h48_bridge_policy`,
  `b13_affordance_sampler_h48_bridge_policy`,
  `b13_local_affordance_search_h56_bridge_policy`, and
  `b13_genetic_local_search_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b12_predictive_attention_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b13_local_affordance_search_controller`.
- trace additions: `b13_controller_profile`, `b13_search_state`,
  `b13_local_route_score`, `b13_affordance_samples`, `b13_search_memory`,
  `b13_dead_end_score`, `b13_search_lock`, and `b13_decision`.
- gate: retain B12/B5 survival and probe behavior, then require corridor
  `0..2` to show explicit B13 local-search decisions, non-trivial search
  states, search locks, and positive local search signals in at least 2/3
  episodes.

Validated B13 result:

- accepted variant: `b13_local_affordance_search_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b13_local_affordance_search_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b12_predictive_attention_h48_bridge_policy/seed_7/best`,
  parent level `12`, target level `13`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B13 candidates passed; deterministic priority
  promoted the h48 local-affordance-search candidate and retained the others
  under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B13 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B12
  corridor gate and added 3/3 explicit B13 decisions, 3/3 search-state
  episodes, 3/3 search-lock episodes, and 3/3 local-search-signal episodes.
  The corridor remains a partial-progress probe; B13 is accepted as a more
  modular local route-evaluation substrate for the next phase.

## B14: Affordance Uncertainty Calibration

B14 adds a confidence and uncertainty calibration layer above B13 local search.
It keeps the same public primitive action boundary and internal semantic action
set, but separates route confidence, uncertainty, and risk-adjusted score before
continuing a corridor commitment or returning from an ambiguous local affordance.
The biological analogy is a metacognitive confidence gate: a downstream module
does not simply follow the local route sampler, it estimates whether that sample
is reliable enough to keep acting on.

Implementation:

- variants: `b14_affordance_uncertainty_h48_bridge_policy`,
  `b14_risk_calibrated_search_h48_bridge_policy`,
  `b14_confidence_weighted_route_h48_bridge_policy`,
  `b14_affordance_uncertainty_h56_bridge_policy`, and
  `b14_genetic_uncertainty_search_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b13_local_affordance_search_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b14_affordance_uncertainty_controller`.
- trace additions: `b14_controller_profile`, `b14_uncertainty_state`,
  `b14_affordance_confidence`, `b14_uncertainty`,
  `b14_risk_adjusted_score`, `b14_commitment_lock`, and `b14_decision`.
- gate: retain B13/B5 behavior, then require corridor `0..2` to show explicit
  B14 decisions, non-trivial uncertainty states, positive confidence/uncertainty
  signals, and commitment locks in at least 2/3 episodes.

Validated B14 result:

- accepted variant: `b14_affordance_uncertainty_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b14_affordance_uncertainty_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b13_local_affordance_search_h48_bridge_policy/seed_7/best`,
  parent level `13`, target level `14`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B14 candidates passed; deterministic priority
  promoted the h48 affordance-uncertainty candidate and retained the others
  under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B14 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B13
  corridor gate and added 3/3 explicit B14 decisions, 3/3 uncertainty-state
  episodes, 3/3 confidence-signal episodes, and 3/3 commitment-lock episodes.

## B15: Option Critic

B15 adds an option-level critic above B14 uncertainty calibration. It keeps the
public primitive action space and the six internal semantic actions unchanged,
but separates option value, termination pressure, persistence score, and a short
option lock. The biological analogy is a basal-ganglia-like gate over a
candidate corridor option: B14 estimates whether the affordance is reliable,
while B15 estimates whether the current option should persist or terminate.

Implementation:

- variants: `b15_option_critic_h48_bridge_policy`,
  `b15_persistence_gate_h48_bridge_policy`,
  `b15_value_gated_option_h48_bridge_policy`,
  `b15_option_critic_h56_bridge_policy`, and
  `b15_genetic_option_critic_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b14_affordance_uncertainty_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b15_option_critic_controller`.
- trace additions: `b15_controller_profile`, `b15_option_state`,
  `b15_option_value`, `b15_termination_pressure`,
  `b15_persistence_score`, `b15_option_lock`, and `b15_decision`.
- gate: retain B14/B5 behavior, then require corridor `0..2` to show explicit
  B15 decisions, option states, option locks, and positive option value or
  termination signals in at least 2/3 episodes.

Validated B15 result:

- accepted variant: `b15_option_critic_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b15_option_critic_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b14_affordance_uncertainty_h48_bridge_policy/seed_7/best`,
  parent level `14`, target level `15`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B15 candidates passed; deterministic priority
  promoted the h48 option-critic candidate and retained the others under
  `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B15 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B14
  corridor gate and added 3/3 explicit B15 decisions, 3/3 option-state
  episodes, 3/3 option-lock episodes, and 3/3 option-value-signal episodes.

## B16: Option Ensemble

B16 adds a population-style option ensemble above the B15 option critic. It
keeps the public primitive action space and the six internal semantic actions
unchanged, but separates continue and return votes, consensus, conflict, and an
ensemble lock. The biological analogy is a small competing population of option
signals: B15 evaluates one corridor option, while B16 arbitrates between
continuing the option, returning, or preserving the inherited B15 decision.

Implementation:

- variants: `b16_option_ensemble_h48_bridge_policy`,
  `b16_competing_options_h48_bridge_policy`,
  `b16_action_set_voter_h48_bridge_policy`,
  `b16_option_ensemble_h56_bridge_policy`, and
  `b16_genetic_option_ensemble_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b15_option_critic_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b16_option_ensemble_controller`.
- trace additions: `b16_controller_profile`, `b16_ensemble_state`,
  `b16_continue_vote`, `b16_return_vote`, `b16_option_votes`,
  `b16_consensus_score`, `b16_conflict_score`, `b16_ensemble_lock`, and
  `b16_decision`.
- gate: retain B15/B5 behavior, then require corridor `0..2` to show explicit
  B16 decisions, non-baseline ensemble states, ensemble locks, and positive
  ensemble signals in at least 2/3 episodes.

Validated B16 result:

- accepted variant: `b16_option_ensemble_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b16_option_ensemble_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b15_option_critic_h48_bridge_policy/seed_7/best`,
  parent level `15`, target level `16`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B16 candidates passed; deterministic priority
  promoted the h48 option-ensemble candidate and retained the others under
  `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B16 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B15
  corridor gate and added 3/3 explicit B16 decisions, 3/3 ensemble-state
  episodes, 3/3 ensemble-lock episodes, and 3/3 ensemble-signal episodes.

## B17: Neuromodulated Ensemble

B17 adds a neuromodulatory layer above the B16 option ensemble. It keeps the
public primitive action space and the six internal semantic actions unchanged,
but separates arousal, homeostatic gain, option gain, conflict release, and a
short modulation lock. The biological analogy is a lightweight global
modulator: B16 arbitrates between option votes, while B17 adjusts the strength
of continuing or releasing that option using threat, conflict, hunger, sleep
debt, and recovery state.

Implementation:

- variants: `b17_neuromodulated_ensemble_h48_bridge_policy`,
  `b17_arousal_gated_options_h48_bridge_policy`,
  `b17_homeostatic_modulator_h48_bridge_policy`,
  `b17_neuromodulated_ensemble_h56_bridge_policy`, and
  `b17_genetic_neuromodulated_ensemble_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b16_option_ensemble_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b17_neuromodulated_ensemble_controller`.
- trace additions: `b17_controller_profile`, `b17_modulator_state`,
  `b17_arousal_signal`, `b17_homeostatic_gain`, `b17_option_gain`,
  `b17_conflict_release`, `b17_modulation_lock`, and `b17_decision`.
- gate: retain B16/B5 behavior, then require corridor `0..2` to show explicit
  B17 decisions, non-baseline modulator states, modulation locks, and positive
  modulation signals in at least 2/3 episodes.

Validated B17 result:

- accepted variant: `b17_neuromodulated_ensemble_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b17_neuromodulated_ensemble_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b16_option_ensemble_h48_bridge_policy/seed_7/best`,
  parent level `16`, target level `17`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B17 candidates passed; deterministic priority
  promoted the h48 neuromodulated-ensemble candidate and retained the others
  under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B17 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B16
  corridor gate and added 3/3 explicit B17 decisions, 3/3 modulator-state
  episodes, 3/3 modulation-lock episodes, and 3/3 modulation-signal episodes.

## B18: Eligibility Trace

B18 adds a short eligibility-trace layer above the B17 neuromodulated ensemble.
It keeps the public primitive action space and the six internal semantic
actions unchanged, but separates an eligibility trace, a reward-prediction
proxy, stability bias, switch pressure, and a trace lock. The biological
analogy is a short synaptic trace that stabilizes recently useful option
commitments without turning them into hard-coded actions.

Implementation:

- variants: `b18_eligibility_trace_h48_bridge_policy`,
  `b18_metastable_arousal_h48_bridge_policy`,
  `b18_synaptic_trace_modulator_h48_bridge_policy`,
  `b18_eligibility_trace_h56_bridge_policy`, and
  `b18_genetic_eligibility_trace_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b17_neuromodulated_ensemble_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b18_eligibility_trace_controller`.
- trace additions: `b18_controller_profile`, `b18_trace_state`,
  `b18_eligibility_trace`, `b18_reward_prediction_proxy`,
  `b18_stability_bias`, `b18_switch_pressure`, `b18_trace_lock`, and
  `b18_decision`.
- gate: retain B17/B5 behavior, then require corridor `0..2` to show explicit
  B18 decisions, non-baseline trace states, trace locks, and positive
  eligibility/stability/switch signals in at least 2/3 episodes.

Validated B18 result:

- accepted variant: `b18_eligibility_trace_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b18_eligibility_trace_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b17_neuromodulated_ensemble_h48_bridge_policy/seed_7/best`,
  parent level `17`, target level `18`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B18 candidates passed; deterministic priority
  promoted the h48 eligibility-trace candidate and retained the others under
  `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B18 corridor progress: `corridor_gauntlet` episodes `0..2` retained the B17
  corridor gate and added 3/3 explicit B18 decisions, 3/3 trace-state episodes,
  3/3 trace-lock episodes, and 3/3 trace-signal episodes.

## B19: Episodic Meta-Memory

B19 adds a short episodic meta-memory layer above the B18 eligibility trace.
It keeps the public primitive action space and the six internal semantic
actions unchanged. The new module consolidates or releases recently useful
option commitments using B18 trace signals, which makes the architecture more
bioinspired without changing the world API.

Implementation:

- variants: `b19_episodic_meta_memory_h48_bridge_policy`,
  `b19_stability_memory_h48_bridge_policy`,
  `b19_switch_suppression_h48_bridge_policy`,
  `b19_episodic_meta_memory_h56_bridge_policy`, and
  `b19_genetic_meta_memory_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b18_eligibility_trace_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b19_episodic_meta_memory_controller`.
- trace additions: `b19_controller_profile`, `b19_memory_state`,
  `b19_episode_memory`, `b19_consolidation_score`, `b19_stability_vote`,
  `b19_switch_suppression`, `b19_memory_lock`, and `b19_decision`.
- gate: retain B18/B5 behavior through easy, canonical, food deprivation,
  sleep conflict, and food-vs-predator probes; in the corridor, require
  primitive-only actions, zero predator contact, and explicit B19 memory
  decisions/states/locks/signals in at least 2/3 episodes. The inherited B18
  corridor chain remains recorded as a diagnostic, not a B19 blocking criterion.

Validated B19 result:

- accepted variant: `b19_episodic_meta_memory_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b19_episodic_meta_memory_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b18_eligibility_trace_h48_bridge_policy/seed_7/best`,
  parent level `18`, target level `19`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B19 candidates passed; deterministic priority
  promoted the h48 episodic meta-memory candidate and retained the others under
  `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B19 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B19 decisions,
  3/3 memory-state episodes, 3/3 memory-lock episodes, and 3/3 memory-signal
  episodes. The B18 base corridor diagnostic is recorded as false because the
  current phase no longer treats the older corridor progress chain as a
  blocking criterion.

## B20: Working Memory Gate

B20 adds a compact working-memory/context-binding layer above the B19 episodic
meta-memory. It keeps the public primitive action space and the six internal
semantic actions unchanged. The new layer stores a short working buffer,
context-binding score, gate vote, release vote, and buffer lock, approximating a
prefrontal working-memory gate that holds or releases an active corridor
commitment.

Implementation:

- variants: `b20_working_memory_gate_h48_bridge_policy`,
  `b20_context_binding_h48_bridge_policy`,
  `b20_stability_buffer_h48_bridge_policy`,
  `b20_working_memory_gate_h56_bridge_policy`, and
  `b20_genetic_working_memory_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b19_episodic_meta_memory_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b20_working_memory_gate_controller`.
- trace additions: `b20_controller_profile`, `b20_buffer_state`,
  `b20_working_buffer`, `b20_context_binding`, `b20_gate_vote`,
  `b20_release_vote`, `b20_buffer_lock`, and `b20_decision`.
- gate: retain B19/B5 behavior through easy, canonical, food deprivation,
  sleep conflict, and food-vs-predator probes; in the corridor, require
  primitive-only actions, zero predator contact, and explicit B20 buffer
  decisions/states/locks/signals in at least 2/3 episodes. The B19 corridor
  result remains recorded as a diagnostic.

Validated B20 result:

- accepted variant: `b20_working_memory_gate_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b20_working_memory_gate_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b19_episodic_meta_memory_h48_bridge_policy/seed_7/best`,
  parent level `19`, target level `20`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B20 candidates passed; deterministic priority
  promoted the h48 working-memory gate candidate and retained the others under
  `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B20 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B20 decisions,
  3/3 buffer-state episodes, 3/3 buffer-lock episodes, and 3/3 buffer-signal
  episodes. The B19 corridor diagnostic remained true.

## B21: Hippocampal Replay

B21 adds a lightweight hippocampal replay and sequence-binding layer above the
B20 working-memory gate. It keeps the public primitive action space and the six
internal semantic actions unchanged. The new module keeps a short sequence
memory, replay score, route-commitment signal, abort prediction, and replay
lock, approximating route rehearsal over the current corridor commitment.

Implementation:

- variants: `b21_hippocampal_replay_h48_bridge_policy`,
  `b21_sequence_binding_h48_bridge_policy`,
  `b21_route_rehearsal_h48_bridge_policy`,
  `b21_hippocampal_replay_h56_bridge_policy`, and
  `b21_genetic_replay_gate_h48_bridge_policy`.
- mandatory transfer source:
  `artifacts/b_series/evolution/b20_working_memory_gate_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b21_hippocampal_replay_controller`.
- trace additions: `b21_controller_profile`, `b21_replay_state`,
  `b21_sequence_memory`, `b21_replay_score`, `b21_route_commitment`,
  `b21_abort_prediction`, `b21_replay_lock`, and `b21_decision`.
- gate: retain B20/B5 behavior through easy, canonical, food deprivation,
  sleep conflict, and food-vs-predator probes; in the corridor, require
  primitive-only actions, zero predator contact, and explicit B21 replay
  decisions/states/locks/signals in at least 2/3 episodes. The B20 corridor
  result remains recorded as a diagnostic.

Validated B21 result:

- accepted variant: `b21_hippocampal_replay_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b21_hippocampal_replay_h48_bridge_policy/seed_7/best`.
- transfer report: source checkpoint
  `artifacts/b_series/evolution/b20_working_memory_gate_h48_bridge_policy/seed_7/best`,
  parent level `20`, target level `21`, coverage `1.0`, low coverage disabled.
- fixed search: all four fixed B21 candidates passed; deterministic priority
  promoted the h48 hippocampal-replay candidate and retained the others under
  `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B21 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B21 decisions,
  3/3 replay-state episodes, 3/3 replay-lock episodes, and 3/3 replay-signal
  episodes. The B20 corridor diagnostic remained true.

## B22: Prospective Map Replay

B22 adds a small prospective simulation layer above B21. Biologically, this is
the next incremental hippocampal/prefrontal step after replay: B21 binds and
replays recent route context, while B22 projects that replay into a short
viability estimate before committing to continue or abort a corridor route.

The public world contract remains unchanged. `SpiderWorld.step()` still receives
only primitive actions from `ACTIONS`, and the internal semantic vocabulary
remains the same six B-series semantic actions.

Implemented B22 variants:

- `b22_prospective_map_replay_h48_bridge_policy`
- `b22_forward_model_gate_h48_bridge_policy`
- `b22_route_viability_sim_h48_bridge_policy`
- `b22_prospective_map_replay_h56_bridge_policy`
- `b22_genetic_prospective_replay_h48_bridge_policy`

Runtime additions:

- accepted controller source: `b22_prospective_replay_controller`.
- effective level: `B22-prospective-map-replay`.
- trace additions: `b22_controller_profile`, `b22_sim_state`,
  `b22_prospective_sim`, `b22_forward_model_score`,
  `b22_viability_projection`, `b22_abort_projection`, `b22_sim_lock`, and
  `b22_decision`.

Validated B22 result:

- accepted variant: `b22_prospective_map_replay_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b22_prospective_map_replay_h48_bridge_policy/seed_7/best`.
- transfer source: B21 accepted checkpoint
  `artifacts/b_series/evolution/b21_hippocampal_replay_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `21`, target level `22`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B22 candidates passed; deterministic priority
  promoted the h48 prospective-map-replay candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B22 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B22 decisions,
  3/3 prospective-sim state episodes, 3/3 sim-lock episodes, and 3/3 sim-signal
  episodes. The B21 corridor diagnostic remained true.

## B23: Conflict Monitor

B23 adds a prefrontal-style conflict monitor above B22. It reads the B22
prospective simulation, estimates prediction error and abort bias, and commits
only when the stability vote remains strong enough. This keeps the architecture
incremental while adding a distinct bioinspired control module for conflict and
prediction-error arbitration.

The public world contract remains unchanged. `SpiderWorld.step()` still receives
only primitive actions from `ACTIONS`, and the internal semantic vocabulary
remains the same six B-series semantic actions.

Implemented B23 variants:

- `b23_conflict_monitor_h48_bridge_policy`
- `b23_error_gated_replay_h48_bridge_policy`
- `b23_abort_conflict_arbiter_h48_bridge_policy`
- `b23_conflict_monitor_h56_bridge_policy`
- `b23_genetic_conflict_monitor_h48_bridge_policy`

Runtime additions:

- accepted controller source: `b23_conflict_monitor_controller`.
- effective level: `B23-conflict-monitor`.
- trace additions: `b23_controller_profile`, `b23_conflict_state`,
  `b23_prediction_error`, `b23_conflict_memory`, `b23_stability_vote`,
  `b23_abort_bias`, `b23_monitor_lock`, and `b23_decision`.

Validated B23 result:

- accepted variant: `b23_conflict_monitor_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b23_conflict_monitor_h48_bridge_policy/seed_7/best`.
- transfer source: B22 accepted checkpoint
  `artifacts/b_series/evolution/b22_prospective_map_replay_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `22`, target level `23`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B23 candidates passed; deterministic priority
  promoted the h48 conflict-monitor candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B23 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B23 decisions,
  3/3 conflict-state episodes, 3/3 monitor-lock episodes, and 3/3 conflict-signal
  episodes. The B22 corridor diagnostic remained true.

## B24: Precision Conflict

B24 adds a precision-weighting layer above the B23 conflict monitor. This is a
predictive-processing style increment: B23 estimates conflict and abort bias,
while B24 estimates reliability/precision of that conflict signal before
continuing or aborting a route commitment.

The public world contract remains unchanged. `SpiderWorld.step()` still receives
only primitive actions from `ACTIONS`, and the internal semantic vocabulary
remains the same six B-series semantic actions.

Implemented B24 variants:

- `b24_precision_conflict_h48_bridge_policy`
- `b24_prediction_precision_gate_h48_bridge_policy`
- `b24_reliability_abort_h48_bridge_policy`
- `b24_precision_conflict_h56_bridge_policy`
- `b24_genetic_precision_conflict_h48_bridge_policy`

Runtime additions:

- accepted controller source: `b24_precision_conflict_controller`.
- effective level: `B24-precision-conflict`.
- trace additions: `b24_controller_profile`, `b24_precision_state`,
  `b24_precision_memory`, `b24_precision_vote`,
  `b24_uncertainty_pressure`, `b24_abort_precision`,
  `b24_precision_lock`, and `b24_decision`.

Validated B24 result:

- accepted variant: `b24_precision_conflict_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b24_precision_conflict_h48_bridge_policy/seed_7/best`.
- transfer source: B23 accepted checkpoint
  `artifacts/b_series/evolution/b23_conflict_monitor_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `23`, target level `24`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B24 candidates passed; deterministic priority
  promoted the h48 precision-conflict candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B24 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B24 decisions,
  3/3 precision-state episodes, 3/3 precision-lock episodes, and 3/3 precision
  signal episodes. The B23 corridor diagnostic remained true.

## B25: Metacognitive Confidence

B25 adds a metacognitive confidence layer above B24. B24 estimates precision of
the conflict signal; B25 estimates second-order confidence/doubt over that
precision before stabilizing a corridor route or withholding it. This keeps the
same primitive public action space and the same six internal semantic actions.

Implemented B25 variants:

- `b25_metacognitive_confidence_h48_bridge_policy`
- `b25_confidence_calibration_h48_bridge_policy`
- `b25_uncertainty_integrator_h48_bridge_policy`
- `b25_metacognitive_confidence_h56_bridge_policy`
- `b25_genetic_metacognition_h48_bridge_policy`

Runtime contract:

- transfer source: accepted B24 checkpoint
  `artifacts/b_series/evolution/b24_precision_conflict_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b25_metacognitive_confidence_controller`.
- effective level: `B25-metacognitive-confidence`.
- trace additions: `b25_controller_profile`, `b25_metacognitive_state`,
  `b25_confidence_memory`, `b25_confidence_vote`, `b25_doubt_pressure`,
  `b25_control_gain`, `b25_meta_lock`, and `b25_decision`.

Validated B25 result:

- accepted variant: `b25_metacognitive_confidence_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b25_metacognitive_confidence_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `24`, target level `25`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B25 candidates passed; deterministic priority
  promoted the h48 metacognitive-confidence candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B25 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B25 decisions,
  3/3 metacognitive-state episodes, 3/3 meta-lock episodes, and 3/3
  metacognitive-signal episodes. The B24 corridor diagnostic remained true.

## B26: Allostatic Prediction

B26 adds an allostatic prediction-error layer above B25. B25 estimates
metacognitive confidence/doubt; B26 estimates homeostatic prediction error,
setpoint pressure, and stability of the current route before sustaining or
aborting a corridor commitment. It keeps the primitive public action space and
the same six internal semantic actions.

Implemented B26 variants:

- `b26_allostatic_prediction_h48_bridge_policy`
- `b26_setpoint_drift_h48_bridge_policy`
- `b26_error_suppression_h48_bridge_policy`
- `b26_allostatic_prediction_h56_bridge_policy`
- `b26_genetic_allostasis_h48_bridge_policy`

Runtime contract:

- transfer source: accepted B25 checkpoint
  `artifacts/b_series/evolution/b25_metacognitive_confidence_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b26_allostatic_prediction_controller`.
- effective level: `B26-allostatic-prediction`.
- trace additions: `b26_controller_profile`, `b26_allostatic_state`,
  `b26_prediction_error`, `b26_setpoint_pressure`, `b26_control_vote`,
  `b26_stability_lock`, and `b26_decision`.

Validated B26 result:

- accepted variant: `b26_allostatic_prediction_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b26_allostatic_prediction_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `25`, target level `26`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B26 candidates passed; deterministic priority
  promoted the h48 allostatic-prediction candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B26 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B26 decisions,
  3/3 allostatic-state episodes, 3/3 stability-lock episodes, and 3/3
  allostatic-signal episodes. The B25 corridor diagnostic remained true.

## B27: Arousal Gain

B27 adds a neuromodulatory arousal-gain layer above B26. B26 estimates
allostatic prediction error and setpoint pressure; B27 modulates commitment gain
and stress pressure before sustaining or aborting a corridor route. It preserves
the primitive public action space and the same six internal semantic actions.

Implemented B27 variants:

- `b27_arousal_gain_h48_bridge_policy`
- `b27_stress_modulation_h48_bridge_policy`
- `b27_energy_arousal_h48_bridge_policy`
- `b27_arousal_gain_h56_bridge_policy`
- `b27_genetic_arousal_h48_bridge_policy`

Runtime contract:

- transfer source: accepted B26 checkpoint
  `artifacts/b_series/evolution/b26_allostatic_prediction_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b27_arousal_gain_controller`.
- effective level: `B27-arousal-gain`.
- trace additions: `b27_controller_profile`, `b27_arousal_state`,
  `b27_arousal_level`, `b27_gain_modulation`, `b27_stress_pressure`,
  `b27_arousal_lock`, and `b27_decision`.

Validated B27 result:

- accepted variant: `b27_arousal_gain_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b27_arousal_gain_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `26`, target level `27`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B27 candidates passed; deterministic priority
  promoted the h48 arousal-gain candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B27 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B27 decisions,
  3/3 arousal-state episodes, 3/3 arousal-lock episodes, and 3/3 arousal-signal
  episodes. The B26 corridor diagnostic remained true.

## B28: Interoceptive Attention

B28 adds an attentional gate above B27. B27 modulates arousal and stress gain;
B28 stabilizes that signal with an interoceptive focus estimate, a distractor
pressure estimate, and a short attention lock before sustaining or aborting a
corridor route. It preserves the primitive public action space and the same six
internal semantic actions.

Implemented B28 variants:

- `b28_interoceptive_attention_h48_bridge_policy`
- `b28_threat_focus_attention_h48_bridge_policy`
- `b28_homeostatic_attention_h48_bridge_policy`
- `b28_interoceptive_attention_h56_bridge_policy`
- `b28_genetic_attention_h48_bridge_policy`

Runtime contract:

- transfer source: accepted B27 checkpoint
  `artifacts/b_series/evolution/b27_arousal_gain_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b28_interoceptive_attention_controller`.
- effective level: `B28-interoceptive-attention`.
- trace additions: `b28_controller_profile`, `b28_attention_state`,
  `b28_interoceptive_focus`, `b28_attention_gain`,
  `b28_distractor_pressure`, `b28_attention_lock`, and `b28_decision`.

Validated B28 result:

- accepted variant: `b28_interoceptive_attention_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b28_interoceptive_attention_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `27`, target level `28`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B28 candidates passed; deterministic priority
  promoted the h48 interoceptive-attention candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B28 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B28 decisions,
  3/3 attention-state episodes, 3/3 attention-lock episodes, and 3/3
  attention-signal episodes. The B27 corridor diagnostic remained true.

## B29: Salience Competition

B29 adds a salience-competition layer above B28. B28 stabilizes attention with
interoceptive focus and distractor pressure; B29 separates threat, homeostatic,
and corridor salience channels, then records the winner before continuing,
holding, or aborting the corridor route. It preserves the primitive public
action space and the same six internal semantic actions.

Implemented B29 variants:

- `b29_salience_competition_h48_bridge_policy`
- `b29_threat_salience_gate_h48_bridge_policy`
- `b29_homeostatic_salience_gate_h48_bridge_policy`
- `b29_salience_competition_h56_bridge_policy`
- `b29_genetic_salience_h48_bridge_policy`

Runtime contract:

- transfer source: accepted B28 checkpoint
  `artifacts/b_series/evolution/b28_interoceptive_attention_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b29_salience_competition_controller`.
- effective level: `B29-salience-competition`.
- trace additions: `b29_controller_profile`, `b29_salience_state`,
  `b29_threat_salience`, `b29_homeostatic_salience`,
  `b29_corridor_salience`, `b29_winner_channel`, `b29_salience_lock`, and
  `b29_decision`.

Validated B29 result:

- accepted variant: `b29_salience_competition_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b29_salience_competition_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `28`, target level `29`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B29 candidates passed; deterministic priority
  promoted the h48 salience-competition candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B29 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B29 decisions,
  3/3 salience-state episodes, 3/3 salience-lock episodes, 3/3 salience-signal
  episodes, and 3/3 winner-channel episodes. The B28 corridor diagnostic
  remained true.

## B30: Basal Ganglia Gate

B30 adds a basal-ganglia-inspired action gate above B29. B29 separates salience
channels and records the winning channel; B30 converts that salience state into
explicit `go` and `no_go` signals, then records which action gate controlled
the corridor decision. It preserves the primitive public action space and the
same six internal semantic actions.

Implemented B30 variants:

- `b30_basal_ganglia_gate_h48_bridge_policy`
- `b30_go_nogo_balance_h48_bridge_policy`
- `b30_threat_inhibition_gate_h48_bridge_policy`
- `b30_basal_ganglia_gate_h56_bridge_policy`
- `b30_genetic_action_gate_h48_bridge_policy`

Runtime contract:

- transfer source: accepted B29 checkpoint
  `artifacts/b_series/evolution/b29_salience_competition_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b30_basal_ganglia_gate_controller`.
- effective level: `B30-basal-ganglia-gate`.
- trace additions: `b30_controller_profile`, `b30_gate_state`,
  `b30_go_signal`, `b30_no_go_signal`, `b30_action_gate`,
  `b30_gate_lock`, and `b30_decision`.

Validated B30 result:

- accepted variant: `b30_basal_ganglia_gate_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b30_basal_ganglia_gate_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `29`, target level `30`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B30 candidates passed; deterministic priority
  promoted the h48 basal-ganglia-gate candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B30 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B30 decisions,
  3/3 gate-state episodes, 3/3 gate-lock episodes, 3/3 gate-signal episodes,
  and 3/3 action-gate episodes. The B29 corridor diagnostic remained true.

## B31: Dopamine Prediction Error

B31 adds a dopamine-inspired prediction-error layer above B30. B30 already
turns salience competition into a go/no-go gate; B31 keeps that substrate and
adds tonic/phasic dopamine estimates, reward-prediction error, and a gate bias
that can stabilize corridor commitment or inhibit it when the no-go channel is
dominant. This is an incremental neuromodulatory step: same primitive public
action space, same internal semantic actions, and transfer from the accepted B30
brain.

Implemented B31 variants:

- `b31_dopamine_prediction_error_h48_bridge_policy`
- `b31_tonic_dopamine_gate_h48_bridge_policy`
- `b31_phasic_dopamine_gate_h48_bridge_policy`
- `b31_dopamine_prediction_error_h56_bridge_policy`
- `b31_genetic_dopamine_gate_h48_bridge_policy`

Core implementation details:

- transfer source: accepted B30 checkpoint
  `artifacts/b_series/evolution/b30_basal_ganglia_gate_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b31_dopamine_prediction_error_controller`.
- effective level: `B31-dopamine-prediction-error`.
- trace additions: `b31_controller_profile`, `b31_dopamine_state`,
  `b31_reward_prediction_error`, `b31_tonic_dopamine`,
  `b31_phasic_dopamine`, `b31_gate_bias`, `b31_dopamine_lock`, and
  `b31_decision`.

Validated B31 result:

- accepted variant: `b31_dopamine_prediction_error_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b31_dopamine_prediction_error_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `30`, target level `31`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B31 candidates passed; deterministic priority
  promoted the h48 dopamine-prediction-error candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B31 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B31 decisions,
  3/3 dopamine-state episodes, 3/3 dopamine-lock episodes, 3/3 dopamine-signal
  episodes, and 3/3 gate-bias episodes. The B30 corridor diagnostic remained
  true.

## B32: Actor-Critic Value

B32 adds an actor-critic-inspired value layer above B31. B31 supplies
prediction-error and tonic/phasic dopamine signals; B32 separates a critic value
estimate from actor advantage and uses their combined policy bias to stabilize
or abort corridor commitments. This is a modular value-control step, not a new
world API: the public action space remains primitive and the internal semantic
set remains unchanged.

Implemented B32 variants:

- `b32_actor_critic_value_h48_bridge_policy`
- `b32_advantage_value_gate_h48_bridge_policy`
- `b32_critic_stability_h48_bridge_policy`
- `b32_actor_critic_value_h56_bridge_policy`
- `b32_genetic_actor_critic_h48_bridge_policy`

Core implementation details:

- transfer source: accepted B31 checkpoint
  `artifacts/b_series/evolution/b31_dopamine_prediction_error_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b32_actor_critic_value_controller`.
- effective level: `B32-actor-critic-value`.
- trace additions: `b32_controller_profile`, `b32_critic_value`,
  `b32_actor_advantage`, `b32_value_error`, `b32_policy_bias`,
  `b32_value_lock`, and `b32_decision`.

Validated B32 result:

- accepted variant: `b32_actor_critic_value_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b32_actor_critic_value_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `31`, target level `32`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B32 candidates passed; deterministic priority
  promoted the h48 actor-critic-value candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B32 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B32 decisions,
  3/3 critic-value episodes, 3/3 actor-advantage episodes, 3/3 value-lock
  episodes, and 3/3 policy-bias episodes. The B31 corridor diagnostic remained
  true.

## B33: Temporal-Difference Error Decomposition

B33 adds a temporal-difference-inspired decomposition layer above B32. B32
separates actor advantage from critic value; B33 keeps that actor-critic
substrate and adds bootstrap value, reward trace, and TD error as distinct
signals. This makes the controller more biologically grounded in incremental
reward learning while preserving the same primitive public action space and the
same six internal semantic actions.

Implemented B33 variants:

- `b33_td_error_decomposition_h48_bridge_policy`
- `b33_bootstrapped_value_gate_h48_bridge_policy`
- `b33_reward_trace_critic_h48_bridge_policy`
- `b33_td_error_decomposition_h56_bridge_policy`
- `b33_genetic_td_value_h48_bridge_policy`

Core implementation details:

- transfer source: accepted B32 checkpoint
  `artifacts/b_series/evolution/b32_actor_critic_value_h48_bridge_policy/seed_7/best`.
- accepted controller source: `b33_td_error_decomposition_controller`.
- effective level: `B33-td-error-decomposition`.
- trace additions: `b33_controller_profile`, `b33_td_error`,
  `b33_bootstrap_value`, `b33_reward_trace`, `b33_actor_update`,
  `b33_td_lock`, and `b33_decision`.

Validated B33 result:

- accepted variant: `b33_td_error_decomposition_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b33_td_error_decomposition_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `32`, target level `33`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B33 candidates passed; deterministic priority
  promoted the h48 TD-error-decomposition candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B33 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B33 decisions,
  3/3 TD-error episodes, 3/3 bootstrap-value episodes, 3/3 reward-trace
  episodes, and 3/3 TD-lock episodes. The B32 corridor diagnostic remained
  true.

## B34: Eligibility-Credit Assignment

B34 starts from the accepted B33 checkpoint:
`artifacts/b_series/evolution/b33_td_error_decomposition_h48_bridge_policy/seed_7/best`.

The architectural increment is a short synaptic eligibility/credit-assignment
layer above the B33 temporal-difference decomposition. It preserves the public
primitive action space and the same six internal semantic actions, while adding
bioinspired delayed-credit state:

- `b34_eligibility_trace`: decayed local action eligibility from the actor
  update.
- `b34_synaptic_tag`: decayed reward-trace tag, approximating synaptic tagging.
- `b34_credit_assignment`: combined TD-error, eligibility, and tag signal.
- `b34_credit_lock`: transient commitment lock when assigned credit supports
  continuing through the corridor.

Implemented B34 candidates:

- `b34_eligibility_credit_h48_bridge_policy`
- `b34_delayed_credit_gate_h48_bridge_policy`
- `b34_synaptic_tagging_h48_bridge_policy`
- `b34_eligibility_credit_h56_bridge_policy`
- `b34_genetic_eligibility_h48_bridge_policy`

Acceptance keeps the same retention gates as B33 and adds corridor evidence that
the B34 trace is not just a B33 clone: explicit B34 decision, nonzero eligibility
trace, credit assignment, synaptic tag, and credit lock in at least 2/3 corridor
episodes with primitive-only traces and zero predator contact.

Validated B34 result:

- accepted variant: `b34_eligibility_credit_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b34_eligibility_credit_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `33`, target level `34`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B34 candidates passed; deterministic priority
  promoted the h48 eligibility-credit candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B34 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B34 decisions,
  3/3 eligibility-trace episodes, 3/3 credit-assignment episodes, 3/3
  synaptic-tag episodes, and 3/3 credit-lock episodes. The B33 corridor
  diagnostic remained true.

## B35: Forward-Model Value Prediction

B35 starts from the accepted B34 checkpoint:
`artifacts/b_series/evolution/b34_eligibility_credit_h48_bridge_policy/seed_7/best`.

The architectural increment is a compact forward-model/value-prediction layer
above B34 eligibility credit. It adds a prospective transition estimate without
changing the public action space or adding semantic actions:

- `b35_prediction_memory`: decayed prediction state over recent corridor
  progress and credit assignment.
- `b35_transition_error`: mismatch between current prospective value and the
  previous prediction memory.
- `b35_model_confidence`: confidence in the forward model under positive credit
  and bounded transition error.
- `b35_forward_value`: combined prospective value used to continue, abort, or
  preserve B34.
- `b35_model_lock`: transient commitment lock when the forward model supports
  continuing the corridor action.

Implemented B35 candidates:

- `b35_forward_model_value_h48_bridge_policy`
- `b35_transition_error_gate_h48_bridge_policy`
- `b35_model_confidence_h48_bridge_policy`
- `b35_forward_model_value_h56_bridge_policy`
- `b35_genetic_forward_model_h48_bridge_policy`

Validated B35 result:

- accepted variant: `b35_forward_model_value_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b35_forward_model_value_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `34`, target level `35`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B35 candidates passed; deterministic priority
  promoted the h48 forward-model-value candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B35 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B35 decisions,
  3/3 forward-value episodes, 3/3 transition-error episodes, 3/3
  model-confidence episodes, 3/3 prediction-memory episodes, and 3/3
  model-lock episodes. The B34 corridor diagnostic remained true.

## B36: Latent Belief-State Inference

B36 starts from the accepted B35 checkpoint:
`artifacts/b_series/evolution/b35_forward_model_value_h48_bridge_policy/seed_7/best`.

The architectural increment is a latent belief-state layer above the B35
forward model. It keeps the primitive world boundary and the same semantic
actions, but adds a compact inferred state over corridor context and prediction
quality:

- `b36_context_memory`: decayed latent context from local corridor geometry.
- `b36_latent_state`: inferred state combining forward value, prediction memory,
  and context.
- `b36_belief_error`: prediction/belief mismatch used as a corrective signal.
- `b36_state_confidence`: confidence in the inferred state.
- `b36_belief_lock`: transient commitment lock when the latent belief supports
  continuing the corridor action.

Implemented B36 candidates:

- `b36_latent_belief_state_h48_bridge_policy`
- `b36_belief_error_gate_h48_bridge_policy`
- `b36_context_inference_h48_bridge_policy`
- `b36_latent_belief_state_h56_bridge_policy`
- `b36_genetic_belief_state_h48_bridge_policy`

Validated B36 result:

- accepted variant: `b36_latent_belief_state_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b36_latent_belief_state_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `35`, target level `36`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B36 candidates passed; deterministic priority
  promoted the h48 latent-belief-state candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B36 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B36 decisions,
  3/3 latent-state episodes, 3/3 belief-error episodes, 3/3
  state-confidence episodes, 3/3 context-memory episodes, and 3/3 belief-lock
  episodes. The B35 corridor diagnostic remained true.

## B37: Explicit State Factorization

B37 starts from the accepted B36 checkpoint:
`artifacts/b_series/evolution/b36_latent_belief_state_h48_bridge_policy/seed_7/best`.

The architectural increment is an explicit state-factor gate above the B36
belief state. It keeps the primitive action boundary and the same internal
semantic actions, but splits the latent evidence into external and internal
state factors before committing or aborting in the corridor:

- `b37_external_state_factor`: decayed exteroceptive/corridor-context factor.
- `b37_internal_state_factor`: decayed interoceptive/belief-confidence factor.
- `b37_factor_alignment`: alignment between external and internal factors.
- `b37_factor_confidence`: confidence that the factorized state supports the
  action.
- `b37_factor_lock`: transient commitment lock when the factorized state
  supports continuing.

Implemented B37 candidates:

- `b37_state_factor_gate_h48_bridge_policy`
- `b37_intero_extero_factor_h48_bridge_policy`
- `b37_factor_confidence_h48_bridge_policy`
- `b37_state_factor_gate_h56_bridge_policy`
- `b37_genetic_state_factor_h48_bridge_policy`

Validated B37 result:

- accepted variant: `b37_state_factor_gate_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b37_state_factor_gate_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `36`, target level `37`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B37 candidates passed; deterministic priority
  promoted the h48 state-factor-gate candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B37 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B37 decisions,
  3/3 external-factor episodes, 3/3 internal-factor episodes, 3/3
  factor-alignment episodes, 3/3 factor-confidence episodes, and 3/3
  factor-lock episodes. The B36 corridor diagnostic remained true.

## B38: Factor Attention

B38 starts from the accepted B37 checkpoint:
`artifacts/b_series/evolution/b37_state_factor_gate_h48_bridge_policy/seed_7/best`.

The architectural increment is an attention layer over the explicit B37 state
factors. It keeps the primitive action boundary and the same internal semantic
actions, but adds attention-style weighting over external factor evidence,
internal/interoceptive evidence, and confidence before committing or aborting:

- `b38_external_attention`: attention accumulated from the external state
  factor.
- `b38_internal_attention`: attention accumulated from the internal state
  factor.
- `b38_attention_balance`: weighted balance between internal/external attention
  and factor alignment.
- `b38_attention_gain`: confidence-driven gain over the attended factor state.
- `b38_attention_lock`: transient commitment lock when attention supports
  continuing.

Implemented B38 candidates:

- `b38_factor_attention_h48_bridge_policy`
- `b38_interoceptive_attention_h48_bridge_policy`
- `b38_confidence_attention_h48_bridge_policy`
- `b38_factor_attention_h56_bridge_policy`
- `b38_genetic_factor_attention_h48_bridge_policy`

Validated B38 result:

- accepted variant: `b38_factor_attention_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b38_factor_attention_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `37`, target level `38`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B38 candidates passed; deterministic priority
  promoted the h48 factor-attention candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B38 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B38 decisions,
  3/3 external-attention episodes, 3/3 internal-attention episodes, 3/3
  attention-balance episodes, 3/3 attention-gain episodes, and 3/3
  attention-lock episodes. The B37 corridor diagnostic remained true.

## B39: Attention Binding

B39 starts from the accepted B38 checkpoint:
`artifacts/b_series/evolution/b38_factor_attention_h48_bridge_policy/seed_7/best`.

The architectural increment is a binding layer over B38 factor attention,
inspired by feature binding and thalamo-cortical integration. It keeps the
primitive action boundary and the same internal semantic actions, but binds
external attention, internal attention, attention balance, and gain into a
single context before committing or aborting:

- `b39_binding_strength`: accumulated strength of the bound attention state.
- `b39_cross_factor_coherence`: coherence between external and internal
  attention factors.
- `b39_bound_context`: decayed context created by binding attended factors.
- `b39_binding_gain`: confidence-style gain over the bound context.
- `b39_binding_lock`: transient commitment lock when binding supports
  continuing.

Implemented B39 candidates:

- `b39_attention_binding_h48_bridge_policy`
- `b39_cross_factor_binding_h48_bridge_policy`
- `b39_context_binding_attention_h48_bridge_policy`
- `b39_attention_binding_h56_bridge_policy`
- `b39_genetic_attention_binding_h48_bridge_policy`

Validated B39 result:

- accepted variant: `b39_attention_binding_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b39_attention_binding_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `38`, target level `39`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B39 candidates passed; deterministic priority
  promoted the h48 attention-binding candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B39 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B39 decisions,
  3/3 binding-strength episodes, 3/3 cross-factor-coherence episodes, 3/3
  bound-context episodes, 3/3 binding-gain episodes, and 3/3 binding-lock
  episodes. The B38 corridor diagnostic remained true.

## B40: Global Workspace

B40 starts from the accepted B39 checkpoint:
`artifacts/b_series/evolution/b39_attention_binding_h48_bridge_policy/seed_7/best`.

The architectural increment is a lightweight global workspace above B39
attention binding, inspired by global neuronal workspace and thalamo-cortical
broadcast. B40 keeps the primitive action boundary and the same internal
semantic actions, but adds a broadcast-style buffer that integrates binding
strength, cross-factor coherence, bound context, and binding gain before
committing or aborting in the corridor:

- `b40_workspace_activation`: decayed activation of the broadcast workspace.
- `b40_broadcast_gain`: gain available for globally broadcasting the bound
  context.
- `b40_context_availability`: retained context available to downstream action
  arbitration.
- `b40_workspace_stability`: stability of the workspace state across ticks.
- `b40_workspace_lock`: transient commitment lock when the workspace supports
  continuing.

Implemented B40 candidates:

- `b40_global_workspace_h48_bridge_policy`
- `b40_sensory_workspace_h48_bridge_policy`
- `b40_context_workspace_h48_bridge_policy`
- `b40_global_workspace_h56_bridge_policy`
- `b40_genetic_global_workspace_h48_bridge_policy`

Validated B40 result:

- accepted variant: `b40_global_workspace_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b40_global_workspace_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `39`, target level `40`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B40 candidates passed; deterministic priority
  promoted the h48 global-workspace candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B40 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B40 decisions,
  3/3 workspace-activation episodes, 3/3 broadcast-gain episodes, 3/3
  context-availability episodes, 3/3 workspace-stability episodes, and 3/3
  workspace-lock episodes. The B39 corridor diagnostic remained true.

## B41: Executive Workspace

B41 starts from the accepted B40 checkpoint:
`artifacts/b_series/evolution/b40_global_workspace_h48_bridge_policy/seed_7/best`.

The architectural increment is an executive-control layer above the global
workspace, inspired by prefrontal executive selection and inhibitory control.
B41 keeps the primitive action boundary and the same internal semantic actions,
but adds a small control state that selects, inhibits, and stabilizes workspace
commitments before corridor actions are bridged:

- `b41_executive_selection`: accumulated executive evidence for continuing.
- `b41_inhibitory_pressure`: inhibition pressure when workspace/context are
  unstable.
- `b41_goal_context`: retained goal context available to action arbitration.
- `b41_executive_stability`: stability of the executive state across ticks.
- `b41_executive_lock`: transient commitment lock when executive selection
  supports continuing.

Implemented B41 candidates:

- `b41_executive_workspace_h48_bridge_policy`
- `b41_inhibitory_control_h48_bridge_policy`
- `b41_goal_context_selector_h48_bridge_policy`
- `b41_executive_workspace_h56_bridge_policy`
- `b41_genetic_executive_workspace_h48_bridge_policy`

Validated B41 result:

- accepted variant: `b41_executive_workspace_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b41_executive_workspace_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `40`, target level `41`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B41 candidates passed; deterministic priority
  promoted the h48 executive-workspace candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B41 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B41 decisions,
  3/3 executive-selection episodes, 3/3 inhibitory-pressure episodes, 3/3
  goal-context episodes, 3/3 executive-stability episodes, and 3/3
  executive-lock episodes. The B40 corridor diagnostic remained true.

## B42: Error Monitor

B42 starts from the accepted B41 checkpoint:
`artifacts/b_series/evolution/b41_executive_workspace_h48_bridge_policy/seed_7/best`.

The architectural increment is a compact error/conflict monitor above the
executive workspace, inspired by anterior cingulate style performance
monitoring. B42 keeps the primitive action boundary and the same internal
semantic actions, but tracks error, conflict, and performance context before
allowing the executive commitment to continue through the corridor bridge:

- `b42_error_signal`: monitor signal from instability and inhibition.
- `b42_conflict_signal`: conflict between executive selection and goal context.
- `b42_performance_context`: retained performance context supporting
  commitment.
- `b42_monitor_stability`: stability of the monitor state across ticks.
- `b42_monitor_lock`: transient commitment lock when performance monitoring
  supports continuing.

Implemented B42 candidates:

- `b42_error_monitor_h48_bridge_policy`
- `b42_conflict_monitor_h48_bridge_policy`
- `b42_performance_monitor_h48_bridge_policy`
- `b42_error_monitor_h56_bridge_policy`
- `b42_genetic_error_monitor_h48_bridge_policy`

Validated B42 result:

- accepted variant: `b42_error_monitor_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b42_error_monitor_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `41`, target level `42`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B42 candidates passed; deterministic priority
  promoted the h48 error-monitor candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B42 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B42 decisions,
  3/3 error-signal episodes, 3/3 conflict-signal episodes, 3/3
  performance-context episodes, 3/3 monitor-stability episodes, and 3/3
  monitor-lock episodes. The B41 corridor diagnostic remained true.

## B43: Adaptive Precision

B43 starts from the accepted B42 checkpoint:
`artifacts/b_series/evolution/b42_error_monitor_h48_bridge_policy/seed_7/best`.

The architectural increment is an adaptive precision layer above the B42
error monitor, inspired by neuromodulatory gain control and arousal-sensitive
precision weighting. B43 keeps the primitive action boundary and the same
internal semantic actions, but modulates corridor commitments with an adaptive
threshold and a short precision lock:

- `b43_precision_signal`: retained precision evidence for continuing.
- `b43_adaptive_threshold`: context-sensitive commit threshold.
- `b43_arousal_context`: arousal signal derived from error and conflict.
- `b43_control_stability`: stability of the precision-modulated controller.
- `b43_precision_lock`: transient lock when precision supports commitment.

Implemented B43 candidates:

- `b43_adaptive_precision_h48_bridge_policy`
- `b43_arousal_precision_h48_bridge_policy`
- `b43_threshold_adaptation_h48_bridge_policy`
- `b43_adaptive_precision_h56_bridge_policy`
- `b43_genetic_adaptive_precision_h48_bridge_policy`

Validated B43 result:

- accepted variant: `b43_adaptive_precision_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b43_adaptive_precision_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `42`, target level `43`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B43 candidates passed; deterministic priority
  promoted the h48 adaptive-precision candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B43 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B43 decisions,
  3/3 precision-signal episodes, 3/3 adaptive-threshold episodes, 3/3
  arousal-context episodes, 3/3 control-stability episodes, and 3/3
  precision-lock episodes. The B42 corridor diagnostic remained true.

## B44: Thalamic Relay

B44 starts from the accepted B43 checkpoint:
`artifacts/b_series/evolution/b43_adaptive_precision_h48_bridge_policy/seed_7/best`.

The architectural increment is a thalamic-style relay/gating layer above the
adaptive precision controller. B44 keeps the primitive action boundary and the
same internal semantic actions, but separates sensory precision, contextual
relay, gate stability, and a short relay lock before corridor commitment:

- `b44_relay_gate`: accumulated relay evidence for continuing.
- `b44_sensory_precision`: sensory precision passed into the relay.
- `b44_context_relay`: contextual relay signal from stable precision state.
- `b44_gate_stability`: stability of the relay/gate state across ticks.
- `b44_relay_lock`: transient commitment lock when the relay opens.

Implemented B44 candidates:

- `b44_thalamic_relay_h48_bridge_policy`
- `b44_sensory_relay_h48_bridge_policy`
- `b44_context_relay_h48_bridge_policy`
- `b44_thalamic_relay_h56_bridge_policy`
- `b44_genetic_thalamic_relay_h48_bridge_policy`

Validated B44 result:

- accepted variant: `b44_thalamic_relay_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b44_thalamic_relay_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `43`, target level `44`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B44 candidates passed; deterministic priority
  promoted the h48 thalamic-relay candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B44 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B44 decisions,
  3/3 relay-gate episodes, 3/3 sensory-precision episodes, 3/3 context-relay
  episodes, 3/3 gate-stability episodes, and 3/3 relay-lock episodes. The B43
  corridor diagnostic remained true.

## B45: Reticular Inhibition

B45 starts from the accepted B44 checkpoint:
`artifacts/b_series/evolution/b44_thalamic_relay_h48_bridge_policy/seed_7/best`.

The architectural increment is a reticular-style inhibitory loop above the
thalamic relay. B45 keeps the primitive action boundary and the same internal
semantic actions, but adds sensory filtering, contextual suppression, loop
stability, and a short inhibition lock before preserving or aborting corridor
commitment:

- `b45_inhibitory_gate`: accumulated inhibitory-loop evidence for continuing.
- `b45_sensory_filter`: filtered sensory precision admitted by the loop.
- `b45_context_suppression`: contextual suppression signal from relay context.
- `b45_loop_stability`: stability of the reticular loop across ticks.
- `b45_inhibition_lock`: transient commitment lock when inhibition is stable.

Implemented B45 candidates:

- `b45_reticular_inhibition_h48_bridge_policy`
- `b45_sensory_inhibition_h48_bridge_policy`
- `b45_context_inhibition_h48_bridge_policy`
- `b45_reticular_inhibition_h56_bridge_policy`
- `b45_genetic_reticular_inhibition_h48_bridge_policy`

Validated B45 result:

- accepted variant: `b45_reticular_inhibition_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b45_reticular_inhibition_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `44`, target level `45`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B45 candidates passed; deterministic priority
  promoted the h48 reticular-inhibition candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B45 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B45 decisions,
  3/3 inhibitory-gate episodes, 3/3 sensory-filter episodes, 3/3
  context-suppression episodes, 3/3 loop-stability episodes, and 3/3
  inhibition-lock episodes. The B44 corridor diagnostic remained true.

## B46: Corticothalamic Feedback

B46 starts from the accepted B45 checkpoint:
`artifacts/b_series/evolution/b45_reticular_inhibition_h48_bridge_policy/seed_7/best`.

The architectural increment is a corticothalamic-style feedback loop above the
reticular inhibition layer. B46 keeps the primitive action boundary and the
same internal semantic actions, but adds top-down context, prediction match,
feedback gain, feedback stability, and a short feedback lock before preserving,
continuing, or aborting corridor commitment:

- `b46_feedback_gain`: accumulated feedback evidence for corridor commitment.
- `b46_topdown_context`: top-down context signal derived from the B45 loop.
- `b46_prediction_match`: sensory/context prediction agreement.
- `b46_feedback_stability`: stability of the feedback loop across ticks.
- `b46_feedback_lock`: transient commitment lock when feedback is stable.

Implemented B46 candidates:

- `b46_corticothalamic_feedback_h48_bridge_policy`
- `b46_feedback_gain_h48_bridge_policy`
- `b46_context_feedback_h48_bridge_policy`
- `b46_corticothalamic_feedback_h56_bridge_policy`
- `b46_genetic_corticothalamic_feedback_h48_bridge_policy`

Validated B46 result:

- accepted variant: `b46_corticothalamic_feedback_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b46_corticothalamic_feedback_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `45`, target level `46`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B46 candidates passed; deterministic priority
  promoted the h48 corticothalamic-feedback candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B46 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B46 decisions,
  3/3 feedback-gain episodes, 3/3 top-down-context episodes, 3/3
  prediction-match episodes, 3/3 feedback-stability episodes, and 3/3
  feedback-lock episodes. The B45 corridor diagnostic remained true.

## B47: Oscillatory Synchrony

B47 starts from the accepted B46 checkpoint:
`artifacts/b_series/evolution/b46_corticothalamic_feedback_h48_bridge_policy/seed_7/best`.

The architectural increment is an oscillatory-synchrony layer above the
corticothalamic feedback loop. B47 keeps the primitive action boundary and the
same internal semantic actions, but adds phase alignment, synchrony gain,
cross-loop coherence, and a short phase lock before preserving, continuing, or
aborting corridor commitment:

- `b47_phase_alignment`: phase-like alignment of feedback and prediction state.
- `b47_synchrony_gain`: accumulated synchrony evidence for commitment.
- `b47_cross_loop_coherence`: coherence between top-down and feedback loops.
- `b47_phase_lock`: transient commitment lock when oscillatory synchrony is stable.

Implemented B47 candidates:

- `b47_oscillatory_synchrony_h48_bridge_policy`
- `b47_phase_locking_h48_bridge_policy`
- `b47_coherence_gate_h48_bridge_policy`
- `b47_oscillatory_synchrony_h56_bridge_policy`
- `b47_genetic_oscillatory_synchrony_h48_bridge_policy`

Validated B47 result:

- accepted variant: `b47_oscillatory_synchrony_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b47_oscillatory_synchrony_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `46`, target level `47`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B47 candidates passed; deterministic priority
  promoted the h48 oscillatory-synchrony candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B47 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B47 decisions,
  3/3 phase-alignment episodes, 3/3 synchrony-gain episodes, 3/3 cross-loop
  coherence episodes, and 3/3 phase-lock episodes. The B46 corridor diagnostic
  remained true.

## B48: Cerebellar Timing

B48 starts from the accepted B47 checkpoint:
`artifacts/b_series/evolution/b47_oscillatory_synchrony_h48_bridge_policy/seed_7/best`.

The architectural increment is a cerebellar-style timing calibration layer
above oscillatory synchrony. B48 keeps the primitive action boundary and the
same internal semantic actions, but adds timing-error calibration, predictive
timing, corrective gain, and a short calibration lock before preserving,
continuing, or aborting corridor commitment:

- `b48_timing_error`: calibrated timing-error match against the phase target.
- `b48_predictive_timing`: predictive timing signal from synchrony and phase lock.
- `b48_corrective_gain`: corrective gain applied to stabilize commitment timing.
- `b48_calibration_lock`: transient commitment lock when timing calibration is stable.

Implemented B48 candidates:

- `b48_cerebellar_timing_h48_bridge_policy`
- `b48_timing_error_correction_h48_bridge_policy`
- `b48_predictive_timing_h48_bridge_policy`
- `b48_cerebellar_timing_h56_bridge_policy`
- `b48_genetic_cerebellar_timing_h48_bridge_policy`

Validated B48 result:

- accepted variant: `b48_cerebellar_timing_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b48_cerebellar_timing_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `47`, target level `48`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B48 candidates passed; deterministic priority
  promoted the h48 cerebellar-timing candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B48 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B48 decisions,
  3/3 timing-error episodes, 3/3 predictive-timing episodes, 3/3 corrective-gain
  episodes, and 3/3 calibration-lock episodes. The B47 corridor diagnostic
  remained true.

## B49: Striatal Action Gate

B49 starts from the accepted B48 checkpoint:
`artifacts/b_series/evolution/b48_cerebellar_timing_h48_bridge_policy/seed_7/best`.

The architectural increment is a basal-ganglia/striatal action-selection layer
above cerebellar timing. B49 keeps the primitive action boundary and the same
six internal semantic actions, but adds explicit `go/no-go` gating around the
timed corridor commitment produced by B48.

New B49 trace fields:

- `b49_controller_profile`: selected striatal-gate profile.
- `b49_go_signal`: direct-path style facilitation pressure.
- `b49_no_go_signal`: indirect-path style suppression pressure.
- `b49_action_gate_balance`: net selection balance after go/no-go competition.
- `b49_selection_lock`: transient action-selection commitment lock.
- `b49_decision`: explicit B49 corridor decision.

Implemented B49 candidates:

- `b49_striatal_action_gate_h48_bridge_policy`
- `b49_direct_path_facilitation_h48_bridge_policy`
- `b49_indirect_path_suppression_h48_bridge_policy`
- `b49_striatal_action_gate_h56_bridge_policy`
- `b49_genetic_striatal_gate_h48_bridge_policy`

Validated B49 result:

- accepted variant: `b49_striatal_action_gate_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b49_striatal_action_gate_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `48`, target level `49`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B49 candidates passed; deterministic priority
  promoted the h48 striatal-gate candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B49 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B49 decisions,
  3/3 go-signal episodes, 3/3 no-go-signal episodes, 3/3 action-gate-balance
  episodes, and 3/3 selection-lock episodes. The B48 corridor diagnostic
  remained true.

## B50: Habit Chunking

B50 starts from the accepted B49 checkpoint:
`artifacts/b_series/evolution/b49_striatal_action_gate_h48_bridge_policy/seed_7/best`.

The architectural increment is a cortico-striatal habit-chunking layer above
the B49 action gate. B50 keeps the primitive action boundary and the same six
internal semantic actions, while adding a compact habit/chunk module that
tracks whether a selected corridor action sequence is stable enough to preserve.

New B50 trace fields:

- `b50_controller_profile`: selected habit-chunking profile.
- `b50_habit_strength`: learned-habit pressure from repeated gate facilitation.
- `b50_chunk_value`: value assigned to the current action chunk.
- `b50_habit_stability`: stability of the selected chunk over time.
- `b50_chunk_lock`: transient lock preserving the selected action chunk.
- `b50_decision`: explicit B50 corridor decision.

Implemented B50 candidates:

- `b50_habit_chunking_h48_bridge_policy`
- `b50_action_chunk_value_h48_bridge_policy`
- `b50_habit_stability_h48_bridge_policy`
- `b50_habit_chunking_h56_bridge_policy`
- `b50_genetic_habit_chunking_h48_bridge_policy`

Validated B50 result:

- accepted variant: `b50_habit_chunking_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b50_habit_chunking_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `49`, target level `50`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B50 candidates passed; deterministic priority
  promoted the h48 habit-chunking candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B50 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B50 decisions,
  3/3 habit-strength episodes, 3/3 chunk-value episodes, 3/3 habit-stability
  episodes, and 3/3 chunk-lock episodes. The B49 corridor diagnostic remained
  true.

## B51: Dopaminergic Habit Modulation

B51 starts from the accepted B50 checkpoint:
`artifacts/b_series/evolution/b50_habit_chunking_h48_bridge_policy/seed_7/best`.

The architectural increment is a dopaminergic habit-modulation layer above the
B50 habit chunker. B51 keeps the primitive action boundary and the same six
internal semantic actions, while adding reward-prediction, dopamine-gain, and
habit-modulation signals that can strengthen or release a selected corridor
chunk without introducing new public actions.

New B51 trace fields:

- `b51_controller_profile`: selected dopaminergic habit profile.
- `b51_prediction_error`: reward-prediction error proxy for recent progress.
- `b51_dopamine_gain`: modulatory gain applied to the habit chunk.
- `b51_habit_modulation`: final modulation pressure over the B50 chunk.
- `b51_modulation_lock`: transient lock preserving modulation across steps.
- `b51_decision`: explicit B51 corridor decision.
- `b51_genetic_generation`: GA generation metadata when applicable.
- `b51_genetic_candidate`: GA candidate metadata when applicable.

Implemented B51 candidates:

- `b51_dopaminergic_habit_modulation_h48_bridge_policy`
- `b51_reward_prediction_gain_h48_bridge_policy`
- `b51_novelty_modulated_habit_h48_bridge_policy`
- `b51_dopaminergic_habit_modulation_h56_bridge_policy`
- `b51_genetic_dopamine_habit_h48_bridge_policy`

Validated B51 result:

- accepted variant: `b51_dopaminergic_habit_modulation_h48_bridge_policy`,
  saved at
  `artifacts/b_series/evolution/b51_dopaminergic_habit_modulation_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `50`, target level `51`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B51 candidates passed; deterministic priority
  promoted the h48 dopaminergic habit-modulation candidate and retained the
  other fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B51 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B51 decisions,
  3/3 prediction-error episodes, 3/3 dopamine-gain episodes, 3/3
  habit-modulation episodes, and 3/3 modulation-lock episodes. The B50
  corridor diagnostic remained true.

## B52: Cholinergic Precision Gate

B52 starts from the accepted B51 checkpoint:
`artifacts/b_series/evolution/b51_dopaminergic_habit_modulation_h48_bridge_policy/seed_7/best`.

The architectural increment is a cholinergic precision/attention layer above
the B51 dopaminergic habit modulator. B52 keeps the primitive action boundary
and the same six internal semantic actions, while adding an acetylcholine-like
uncertainty and precision signal that gates whether the B51 habit modulation is
sharp enough to continue, hold, or release a corridor commitment.

New B52 trace fields:

- `b52_controller_profile`: selected cholinergic precision profile.
- `b52_acetylcholine_level`: acetylcholine-like attention level.
- `b52_precision_gain`: precision gain applied over the B51 habit signal.
- `b52_uncertainty_signal`: uncertainty pressure used to modulate attention.
- `b52_attention_lock`: transient lock preserving precision-gated commitment.
- `b52_decision`: explicit B52 corridor decision.
- `b52_genetic_generation`: GA generation metadata when applicable.
- `b52_genetic_candidate`: GA candidate metadata when applicable.

Implemented B52 candidates:

- `b52_cholinergic_precision_gate_h48_bridge_policy`
- `b52_attention_gain_h48_bridge_policy`
- `b52_uncertainty_release_h48_bridge_policy`
- `b52_cholinergic_precision_gate_h56_bridge_policy`
- `b52_genetic_cholinergic_precision_h48_bridge_policy`

Validated B52 result:

- accepted variant: `b52_cholinergic_precision_gate_h48_bridge_policy`, saved
  at
  `artifacts/b_series/evolution/b52_cholinergic_precision_gate_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `51`, target level `52`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B52 candidates passed; deterministic priority
  promoted the h48 cholinergic precision-gate candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B52 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B52 decisions,
  3/3 acetylcholine-level episodes, 3/3 precision-gain episodes, 3/3
  uncertainty-signal episodes, and 3/3 attention-lock episodes. The B51
  corridor diagnostic remained true.

## B53: Noradrenergic Arousal Gain

B53 starts from the accepted B52 checkpoint:
`artifacts/b_series/evolution/b52_cholinergic_precision_gate_h48_bridge_policy/seed_7/best`.

The architectural increment is a noradrenergic arousal-gain layer above the B52
cholinergic precision gate. B53 keeps the primitive action boundary and the same
six internal semantic actions, while adding norepinephrine-like surprise and
arousal signals that modulate how strongly the precision-gated corridor
commitment is held under uncertainty.

New B53 trace fields:

- `b53_controller_profile`: selected noradrenergic arousal profile.
- `b53_norepinephrine_level`: norepinephrine-like arousal level.
- `b53_arousal_gain`: gain applied over the B52 precision signal.
- `b53_surprise_signal`: surprise pressure derived from uncertainty/threat.
- `b53_gain_lock`: transient lock preserving arousal-gated commitment.
- `b53_decision`: explicit B53 corridor decision.
- `b53_genetic_generation`: GA generation metadata when applicable.
- `b53_genetic_candidate`: GA candidate metadata when applicable.

Implemented B53 candidates:

- `b53_noradrenergic_arousal_gain_h48_bridge_policy`
- `b53_surprise_gain_h48_bridge_policy`
- `b53_stress_precision_h48_bridge_policy`
- `b53_noradrenergic_arousal_gain_h56_bridge_policy`
- `b53_genetic_arousal_precision_h48_bridge_policy`

Validated B53 result:

- accepted variant: `b53_noradrenergic_arousal_gain_h48_bridge_policy`, saved
  at
  `artifacts/b_series/evolution/b53_noradrenergic_arousal_gain_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `52`, target level `53`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B53 candidates passed; deterministic priority
  promoted the h48 noradrenergic arousal-gain candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B53 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B53 decisions,
  3/3 norepinephrine-level episodes, 3/3 arousal-gain episodes, 3/3
  surprise-signal episodes, and 3/3 gain-lock episodes. The B52 corridor
  diagnostic remained true.

## B54: Serotonergic Patience Gate

B54 starts from the accepted B53 checkpoint:

`artifacts/b_series/evolution/b53_noradrenergic_arousal_gain_h48_bridge_policy/seed_7/best`

The architectural increment is a serotonergic patience/inhibition layer above
the B53 noradrenergic arousal-gain controller. It keeps the same primitive
world action space and the same internal six semantic actions, but adds an
episodic patience signal, impulse-suppression signal, and transient patience
lock for corridor commitment or abort decisions. This is a conservative
bioinspired extension: B53 modulates arousal/gain, while B54 adds a serotonin
like persistence and impulse-control gate over that arousal.

Trace fields added by B54:

- `b54_controller_profile`: selected serotonergic patience profile.
- `b54_serotonin_level`: serotonin-like persistence level.
- `b54_patience_signal`: accumulated patience/commitment signal.
- `b54_impulse_suppression`: transient inhibition against impulsive switching.
- `b54_patience_lock`: transient lock preserving patience-gated commitment.
- `b54_decision`: explicit B54 corridor decision.
- `b54_genetic_generation`: GA generation metadata when applicable.
- `b54_genetic_candidate`: GA candidate metadata when applicable.

B54 candidates:

- `b54_serotonergic_patience_gate_h48_bridge_policy`
- `b54_impulse_suppression_h48_bridge_policy`
- `b54_patience_balance_h48_bridge_policy`
- `b54_serotonergic_patience_gate_h56_bridge_policy`
- `b54_genetic_serotonin_patience_h48_bridge_policy`

Validated B54 result:

- accepted variant: `b54_serotonergic_patience_gate_h48_bridge_policy`, saved
  at
  `artifacts/b_series/evolution/b54_serotonergic_patience_gate_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `53`, target level `54`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B54 candidates passed; deterministic priority
  promoted the h48 serotonergic patience-gate candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B54 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B54 decisions,
  3/3 serotonin-level episodes, 3/3 patience-signal episodes, 3/3
  impulse-suppression episodes, and 3/3 patience-lock episodes. The B53
  corridor diagnostic remained true.

## B55: Hypothalamic Drive Coupling

B55 starts from the accepted B54 checkpoint:

`artifacts/b_series/evolution/b54_serotonergic_patience_gate_h48_bridge_policy/seed_7/best`

The architectural increment is a hypothalamic drive-coupling layer above the
B54 serotonergic patience gate. It keeps the same primitive world action space
and the same internal six semantic actions, but adds an interoceptive drive
module that couples hunger urgency, satiety, recovery bias, threat gating, and
the B54 patience signal. This gives the B-series ladder a more explicit
bioinspired homeostatic nucleus above the neuromodulatory patience layer.

Trace fields added by B55:

- `b55_controller_profile`: selected hypothalamic drive profile.
- `b55_hypothalamic_drive`: hunger-driven foraging pressure.
- `b55_satiety_signal`: satiety/restraint signal opposing unnecessary foraging.
- `b55_recovery_bias`: sleep/health recovery pressure.
- `b55_drive_balance`: combined drive score after threat and recovery gating.
- `b55_drive_lock`: transient lock preserving drive-gated commitment.
- `b55_decision`: explicit B55 corridor decision.
- `b55_genetic_generation`: GA generation metadata when applicable.
- `b55_genetic_candidate`: GA candidate metadata when applicable.

B55 candidates:

- `b55_hypothalamic_drive_coupling_h48_bridge_policy`
- `b55_satiety_recovery_balance_h48_bridge_policy`
- `b55_sleep_hunger_arbiter_h48_bridge_policy`
- `b55_hypothalamic_drive_coupling_h56_bridge_policy`
- `b55_genetic_hypothalamic_drive_h48_bridge_policy`

Validated B55 result:

- accepted variant: `b55_hypothalamic_drive_coupling_h48_bridge_policy`, saved
  at
  `artifacts/b_series/evolution/b55_hypothalamic_drive_coupling_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `54`, target level `55`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B55 candidates passed; deterministic priority
  promoted the h48 hypothalamic drive-coupling candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B55 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B55 decisions,
  3/3 hypothalamic-drive episodes, 3/3 satiety-signal episodes, 3/3
  recovery-bias episodes, 3/3 drive-balance episodes, and 3/3 drive-lock
  episodes. The B54 corridor diagnostic remained true.

## B56: HPA Stress Axis

B56 starts from the accepted B55 checkpoint:

`artifacts/b_series/evolution/b55_hypothalamic_drive_coupling_h48_bridge_policy/seed_7/best`

The architectural increment is a slower HPA-axis-inspired stress layer above
the B55 hypothalamic drive coupler. It keeps the same primitive world action
space and the same internal six semantic actions, but adds cortisol-like
stress load, recovery signal, and endocrine balance signals over the
homeostatic drive. This gives the ladder a second-timescale modulation above
the faster neuromodulatory and hypothalamic gates.

Trace fields added by B56:

- `b56_controller_profile`: selected HPA stress profile.
- `b56_cortisol_level`: cortisol-like stress hormone level.
- `b56_stress_load`: accumulated threat/interoceptive stress load.
- `b56_recovery_signal`: recovery pressure opposing prolonged stress.
- `b56_endocrine_balance`: combined HPA score after drive/recovery/stress gating.
- `b56_stress_lock`: transient lock preserving HPA-gated commitment.
- `b56_decision`: explicit B56 corridor decision.
- `b56_genetic_generation`: GA generation metadata when applicable.
- `b56_genetic_candidate`: GA candidate metadata when applicable.

B56 candidates:

- `b56_hpa_stress_axis_h48_bridge_policy`
- `b56_cortisol_recovery_balance_h48_bridge_policy`
- `b56_stress_load_gate_h48_bridge_policy`
- `b56_hpa_stress_axis_h56_bridge_policy`
- `b56_genetic_hpa_stress_h48_bridge_policy`

Validated B56 result:

- accepted variant: `b56_hpa_stress_axis_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b56_hpa_stress_axis_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `55`, target level `56`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B56 candidates passed; deterministic priority
  promoted the h48 HPA stress-axis candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B56 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B56 decisions,
  3/3 cortisol-level episodes, 3/3 stress-load episodes, 3/3 recovery-signal
  episodes, 3/3 endocrine-balance episodes, and 3/3 stress-lock episodes.
  The B55 corridor diagnostic remained true.

## B57: Insular Interoceptive Awareness

B57 starts from the accepted B56 checkpoint:

`artifacts/b_series/evolution/b56_hpa_stress_axis_h48_bridge_policy/seed_7/best`

The architectural increment is an insula-inspired interoceptive awareness
layer above the B56 HPA stress axis. It keeps the same primitive world action
space and the same internal six semantic actions, but adds a short recurrent
body-state awareness module that binds visceral salience, body-state
confidence, HPA stress, and hypothalamic drive into a corridor commitment or
abort signal. This keeps B57 incremental while adding a distinct
interoceptive integration module.

Trace fields added by B57:

- `b57_controller_profile`: selected insular/interoceptive profile.
- `b57_interoceptive_awareness`: integrated body-state awareness signal.
- `b57_visceral_salience`: hunger/sleep/health salience over recent state.
- `b57_body_state_confidence`: recovery and low-threat confidence signal.
- `b57_awareness_balance`: combined awareness balance for commit/abort.
- `b57_awareness_lock`: transient lock preserving interoceptive commitment.
- `b57_decision`: explicit B57 corridor decision.
- `b57_genetic_generation`: GA generation metadata when applicable.
- `b57_genetic_candidate`: GA candidate metadata when applicable.

B57 candidates:

- `b57_insular_interoceptive_awareness_h48_bridge_policy`
- `b57_visceral_salience_gate_h48_bridge_policy`
- `b57_stress_drive_awareness_h48_bridge_policy`
- `b57_insular_interoceptive_awareness_h56_bridge_policy`
- `b57_genetic_interoceptive_awareness_h48_bridge_policy`

Validated B57 result:

- accepted variant: `b57_insular_interoceptive_awareness_h48_bridge_policy`,
  saved at
  `artifacts/b_series/evolution/b57_insular_interoceptive_awareness_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `56`, target level `57`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B57 candidates passed; deterministic priority
  promoted the h48 insular interoceptive-awareness candidate and retained the
  other fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B57 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B57 decisions,
  3/3 interoceptive-awareness episodes, 3/3 visceral-salience episodes, 3/3
  body-state-confidence episodes, 3/3 awareness-balance episodes, and 3/3
  awareness-lock episodes. The B56 corridor diagnostic remained true.

## B58: ACC Conflict Monitor

B58 starts from the accepted B57 checkpoint:

`artifacts/b_series/evolution/b57_insular_interoceptive_awareness_h48_bridge_policy/seed_7/best`

The architectural increment is an anterior-cingulate-inspired conflict monitor
above the B57 insular interoceptive layer. It keeps the primitive world action
space and the same six internal semantic actions, but adds conflict signal,
error likelihood, control allocation, and resolution balance over the
interoceptive corridor decision. This gives the B ladder a small control layer
that can detect body/risk conflict and stabilize a commit-or-abort choice.

Trace fields added by B58:

- `b58_controller_profile`: selected ACC conflict-monitor profile.
- `b58_conflict_signal`: accumulated conflict between route, threat, and body state.
- `b58_error_likelihood`: risk/error pressure from threat and stress.
- `b58_control_allocation`: control signal available to resolve conflict.
- `b58_resolution_balance`: combined commit/abort balance after conflict monitoring.
- `b58_conflict_lock`: transient lock preserving ACC-gated commitment.
- `b58_decision`: explicit B58 corridor decision.
- `b58_genetic_generation`: GA generation metadata when applicable.
- `b58_genetic_candidate`: GA candidate metadata when applicable.

B58 candidates:

- `b58_acc_conflict_monitor_h48_bridge_policy`
- `b58_error_salience_gate_h48_bridge_policy`
- `b58_conflict_resolution_balance_h48_bridge_policy`
- `b58_acc_conflict_monitor_h56_bridge_policy`
- `b58_genetic_acc_conflict_h48_bridge_policy`

Validated B58 result:

- accepted variant: `b58_acc_conflict_monitor_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b58_acc_conflict_monitor_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `57`, target level `58`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B58 candidates passed; deterministic priority
  promoted the h48 ACC conflict-monitor candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B58 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B58 decisions,
  3/3 conflict-signal episodes, 3/3 error-likelihood episodes, 3/3
  control-allocation episodes, 3/3 resolution-balance episodes, and 3/3
  conflict-lock episodes. The B57 corridor diagnostic remained true.

## B59: Prefrontal Goal Context

B59 starts from the accepted B58 checkpoint:

`artifacts/b_series/evolution/b58_acc_conflict_monitor_h48_bridge_policy/seed_7/best`

The architectural increment is a prefrontal goal-context and working-set layer
above the B58 ACC conflict monitor. It keeps the primitive world action space
and the same six internal semantic actions, but adds goal context, working-set
stability, task-set confidence, and executive balance over the ACC conflict
decision. This gives the B ladder a small executive-control module that can
stabilize the currently selected corridor goal after conflict monitoring.

Trace fields added by B59:

- `b59_controller_profile`: selected prefrontal goal-context profile.
- `b59_goal_context`: recurrent goal/context signal for corridor commitment.
- `b59_working_set_stability`: stability of the currently selected task set.
- `b59_task_set_confidence`: confidence that the task set is viable.
- `b59_executive_balance`: combined commit/abort balance after executive control.
- `b59_executive_lock`: transient lock preserving prefrontal commitment.
- `b59_decision`: explicit B59 corridor decision.
- `b59_genetic_generation`: GA generation metadata when applicable.
- `b59_genetic_candidate`: GA candidate metadata when applicable.

B59 candidates:

- `b59_prefrontal_goal_context_h48_bridge_policy`
- `b59_working_set_stability_h48_bridge_policy`
- `b59_executive_task_set_h48_bridge_policy`
- `b59_prefrontal_goal_context_h56_bridge_policy`
- `b59_genetic_prefrontal_control_h48_bridge_policy`

Validated B59 result:

- accepted variant: `b59_prefrontal_goal_context_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b59_prefrontal_goal_context_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `58`, target level `59`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B59 candidates passed; deterministic priority
  promoted the h48 prefrontal goal-context candidate and retained the other
  fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B59 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B59 decisions,
  3/3 goal-context episodes, 3/3 working-set-stability episodes, 3/3
  task-set-confidence episodes, 3/3 executive-balance episodes, and 3/3
  executive-lock episodes. The B58 corridor diagnostic remained true.

## B60: Orbitofrontal Outcome Value

B60 starts from the accepted B59 checkpoint:

`artifacts/b_series/evolution/b59_prefrontal_goal_context_h48_bridge_policy/seed_7/best`

The architectural increment is a small orbitofrontal outcome-valuation layer
above the B59 prefrontal goal-context controller. It keeps the primitive world
action space and the same six internal semantic actions, but adds outcome
value, reversal signal, goal-value confidence, and value balance. This gives
the B ladder a bioinspired valuation/reversal module that can preserve a viable
goal commitment or reverse it when the short-horizon outcome estimate becomes
unfavorable.

Trace fields added by B60:

- `b60_controller_profile`: selected orbitofrontal value profile.
- `b60_outcome_value`: recurrent estimate of the selected goal outcome.
- `b60_reversal_signal`: transient reversal pressure from threat and negative value.
- `b60_goal_value_confidence`: confidence in the goal's value estimate.
- `b60_value_balance`: combined commit/reversal balance after OFC modulation.
- `b60_value_lock`: transient lock preserving a positive outcome-value commitment.
- `b60_decision`: explicit B60 corridor decision.
- `b60_genetic_generation`: GA generation metadata when applicable.
- `b60_genetic_candidate`: GA candidate metadata when applicable.

B60 candidates:

- `b60_orbitofrontal_outcome_value_h48_bridge_policy`
- `b60_reversal_value_gate_h48_bridge_policy`
- `b60_goal_outcome_prediction_h48_bridge_policy`
- `b60_orbitofrontal_outcome_value_h56_bridge_policy`
- `b60_genetic_orbitofrontal_value_h48_bridge_policy`

Validated B60 result:

- first B60 smoke discarded all fixed/GA candidates because the corridor gate
  required a non-zero reversal signal in a safe corridor. The candidates had
  retention and value-lock progress, but the reversal channel correctly stayed
  at zero. The gate was corrected to require the reversal channel to be traced,
  not artificially active.
- accepted variant: `b60_orbitofrontal_outcome_value_h48_bridge_policy`, saved
  at
  `artifacts/b_series/evolution/b60_orbitofrontal_outcome_value_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `59`, target level `60`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B60 candidates passed after the gate correction;
  deterministic priority promoted the h48 orbitofrontal outcome-value candidate
  and retained the other fixed candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B60 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B60 decisions,
  3/3 outcome-value episodes, 3/3 reversal-channel episodes, 3/3
  goal-value-confidence episodes, 3/3 value-balance episodes, and 3/3
  value-lock episodes. The B59 corridor diagnostic remained true.

## B61: Amygdala Safety Value

B61 starts from the accepted B60 checkpoint:

`artifacts/b_series/evolution/b60_orbitofrontal_outcome_value_h48_bridge_policy/seed_7/best`

The architectural increment is a small amygdala-like safety/threat valuation
layer above the B60 orbitofrontal value controller. It keeps the primitive
world action space and the same six internal semantic actions, but adds
separate safety value, threat value, safety confidence, and affective balance.
This gives the B ladder a bioinspired affective safety module that can preserve
safe approach commitments or reverse them when threat value dominates.

Trace fields added by B61:

- `b61_controller_profile`: selected amygdala safety-value profile.
- `b61_safety_value`: recurrent safety valuation for the current goal.
- `b61_threat_value`: threat-value channel, traced even when safe corridors keep it low.
- `b61_safety_confidence`: confidence in the safety value estimate.
- `b61_affective_balance`: combined safety/threat balance after OFC modulation.
- `b61_safety_lock`: transient lock preserving a safe approach commitment.
- `b61_decision`: explicit B61 corridor decision.
- `b61_genetic_generation`: GA generation metadata when applicable.
- `b61_genetic_candidate`: GA candidate metadata when applicable.

B61 candidates:

- `b61_amygdala_safety_value_h48_bridge_policy`
- `b61_threat_value_tag_h48_bridge_policy`
- `b61_safety_prediction_gate_h48_bridge_policy`
- `b61_amygdala_safety_value_h56_bridge_policy`
- `b61_genetic_amygdala_safety_h48_bridge_policy`

Validated B61 result:

- accepted variant: `b61_amygdala_safety_value_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b61_amygdala_safety_value_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `60`, target level `61`, coverage `1.0`, low
  coverage disabled.
- fixed search: all four fixed B61 candidates passed; deterministic priority
  promoted the h48 amygdala safety-value candidate and retained the other fixed
  candidates under `discarded` for audit.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B61 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B61 decisions,
  3/3 safety-value episodes, 3/3 threat-channel episodes, 3/3
  safety-confidence episodes, 3/3 affective-balance episodes, and 3/3
  safety-lock episodes. The B60 corridor diagnostic remained true.

## B62: Defensive Mode Selector

B62 starts from the accepted B61 checkpoint:

`artifacts/b_series/evolution/b61_amygdala_safety_value_h48_bridge_policy/seed_7/best`

The incremental biological hypothesis is a periaqueductal-gray-like defensive
mode selector above the B61 amygdala safety-value layer. It preserves B61
retention and the primitive public action boundary while adding a modular
freeze/flee/safe-advance arbitration channel for corridor escape contexts.

Trace fields added by B62:

- `b62_controller_profile`: selected defensive-mode profile.
- `b62_defensive_mode`: active mode, such as `safe_advance`, `flee_to_shelter`
  or `freeze_hold`.
- `b62_freeze_pressure`: transient freeze pressure from threat and shelter
  proximity.
- `b62_flee_pressure`: transient flee pressure from threat and homeostatic
  vulnerability.
- `b62_shelter_bias`: shelter-oriented defensive bias.
- `b62_defense_balance`: combined defensive balance after B61 safety/threat
  modulation.
- `b62_defense_lock`: transient lock for defensive commitments.
- `b62_decision`: explicit B62 corridor decision.
- `b62_genetic_generation`: GA generation metadata when applicable.
- `b62_genetic_candidate`: GA candidate metadata when applicable.

B62 candidates:

- `b62_defensive_mode_selector_h48_bridge_policy`
- `b62_freeze_flee_balance_h48_bridge_policy`
- `b62_shelter_defense_gate_h48_bridge_policy`
- `b62_defensive_mode_selector_h56_bridge_policy`
- `b62_genetic_defensive_mode_h48_bridge_policy`

Validated B62 result:

- accepted variant: `b62_defensive_mode_selector_h48_bridge_policy`, saved at
  `artifacts/b_series/evolution/b62_defensive_mode_selector_h48_bridge_policy/seed_7/best`.
- transfer report: parent level `61`, target level `62`, coverage `1.0`, low
  coverage disabled.
- fixed search: the fixed candidates were evaluated in parallel; deterministic
  priority promoted the h48 defensive-mode selector and retained the fixed
  audit attempts under `discarded`.
- accepted retention aggregate: easy `0..4` passed; canonical `0..9` passed
  with 5 completed horizons, minimum 52 steps, 24 total predator contacts,
  9 food-cycle episodes, 9 sleep-cycle episodes, and 10 shelter-cycle episodes.
- retained probes: `food_deprivation`, `sleep_vs_exploration_conflict`, and
  `food_vs_predator_conflict` passed.
- B62 corridor evidence: `corridor_gauntlet` episodes `0..2` had 3/3 safe
  primitive traces with zero predator contact, 3/3 explicit B62 decisions,
  3/3 defensive-mode episodes, 3/3 defense-pressure episodes, 3/3
  shelter-bias episodes, 3/3 defense-balance episodes, and 3/3 lock-or-safe
  episodes. The B61 corridor diagnostic remained true.

## Relationship To `verify_goal_acceptance.py`

`artifacts/learning_discovery/2026-05-07/verify_goal_acceptance.py` is the current official artifact gate for the A-ladder continuous-survival goal. It is not a B0 legacy reproduction check.

The verifier currently checks these official ladder levels:

- `true_monolithic_policy`
- `monolithic_policy`
- `three_center_modular`
- `four_center_modular`
- `modular_full`

For each level, it expects enough artifact rows to show a real 10-day canonical continuous-survival result. It checks that the run is not a bootstrap shortcut, that it uses the correct target duration, that it has ecological signals such as food intake, shelter exits and returns, predator exposure, final health, and that benchmark documentation records a passing command.

That means:

- `B0 legacy` is outside this verifier by design.
- `B0 current bridge` can be run through `run_continuous_survival.py` as a diagnostic variant, but a `verify_goal_acceptance.py` failure does not invalidate B0.
- The verifier should become a promotion gate for B only after the project intentionally adds a B-series promotion row or replaces an official A-ladder candidate with a B-derived architecture.

Current status: `verify_goal_acceptance.py` still fails on the existing official artifacts because the current A0 frontier has the known `shelter_exits=1` blocker and the lower A levels do not have complete continuous-survival rows. That failure explains the official frontier status; it is not a statement about whether B0 legacy reproduced the old benchmark.

## Evaluation Sequence

Use this order:

1. Validate `B0 legacy` in `LegacyB0Simulation`.
   - Goal: reproduce old competence under the old abstraction.
   - Example signal: food and sleep events in a 120-step short evaluation after enough training.
2. Validate `B0 current bridge` with unit and smoke tests.
   - Goal: semantic action never reaches `world.step()`, bridge action is primitive, trace fields are present.
3. Run short current-world diagnostics.
   - Use `continuous_survival_easy_v1` first.
   - Then run `continuous_survival_canonical`.
   - Treat these as diagnostics, not promotion.
4. Train longer B0 current bridge only if the smoke shows real primitive behavior, not only semantic label movement.
5. Define B1 from B0.
   - Initialize from B0 checkpoint.
   - Change one hardening dimension at a time.
   - Record transfer coverage.
6. Consider promotion only when a B-derived variant beats the current frontier without hidden runtime override.
7. Run `verify_goal_acceptance.py` only as the final official gate after promotion criteria are met and artifacts are complete.

## Anti-Patterns

Do not interpret `B0 legacy` success as current ecological success.

Do not add semantic actions back into `SpiderWorld.step()`.

Do not hide routine food or shelter steering in a final safety layer.

Do not train B1, B2, or later B levels from scratch unless a failed or disabled transfer is explicitly recorded.

Do not use `verify_goal_acceptance.py` to reject the B strategy before B has been promoted into the official artifact gate.
