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
