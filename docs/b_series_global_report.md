# Global Report On B-Series Evolution

Updated on 2026-05-17.

This report summarizes the accepted evolution of the B series, from B0 through
B62, viewing the line as an incremental transfer program: each accepted level
starts from the `best` checkpoint of the previous level, preserves the primitive
public action space of `SpiderWorld.step()`, keeps the same six internal
semantic actions, and adds a small layer of capacity, modularity, or
bioinspiration.

The central rule of the B series was not "always outperform the previous
level". The consolidated criterion was: retain reasonable learning/survival,
keep the public world contracts, record the new layer in trace/metadata, and
accept only candidates that pass the gate defined for that phase. Candidates
that fail remain documented in `discarded` and do not become transfer sources.

## 1. Bioinspiration

B-series evolution began from a biologically simple observation: the old spider
learned when it could choose ecological intentions such as searching for food,
returning to shelter, staying, eating, or sleeping. B0 preserved that idea as
an internal representation, but removed the privilege at the environment
boundary: the current world still receives only primitive actions.

From B0 to B5, the bioinspiration is mainly ethological and homeostatic:

| Range | Dominant biological idea | Role in the spider |
| --- | --- | --- |
| B0 | Internal ecological intention | Separate "what to do" from "which primitive movement to execute". |
| B1 | Threat guard | Prevent food seeking from ignoring basic predator signals. |
| B2 | Temporal threat | Use short threat memory to avoid leaving shelter too early. |
| B3 | Contact memory and recurrent guard | Make recent pain/contact relevant after the immediate tick. |
| B4 | Multi-episode recovery | Rebalance sleep, health, and shelter exit across varied episodes. |
| B5 | Homeostatic arbiter | Separate hunger, sleep, recovery, and threat into interoceptive pressures. |

From B6 to B14, the biological focus moved toward spatial risk and affordances.
The spider started recording whether the corridor was viable, whether there was
body budget to cross it, whether it should abort/return, and whether the
decision came from a local map, waypoint, prospective replay, confidence,
predictive attention, local search, or uncertainty. This range did not
necessarily solve the whole corridor, but it created auditable decision signals:
continue, abort, return, search locally, or preserve the inherited policy.

From B15 to B22, the inspiration became more cognitive:

- B15/B16 introduced options and option ensembles, moving action closer to a
  hierarchical choice.
- B17 modulated those options through neuromodulatory signals.
- B18/B19 added eligibility traces and episodic memory.
- B20 formalized a working-memory gate.
- B21/B22 added hippocampal replay and prospective map replay.

From B23 to B39, the series went through a layer of monitoring, value, and
internal representation:

- B23/B24 monitor conflict and conflict precision.
- B25/B26 add metacognitive confidence and allostatic prediction.
- B27/B29 modulate gain, interoceptive attention, and salience competition.
- B30/B33 incorporate basal-ganglia gate, dopamine, actor-critic, and
  temporal-difference error.
- B34/B39 add temporal credit, predictive model, latent state, state
  factorization, factor attention, and attentional binding.

From B40 to B48, the bioinspiration moved closer to corticothalamic and
cerebellar circuits:

- B40 creates a lightweight global workspace.
- B41 adds executive workspace.
- B42/B43 monitor error and adjust adaptive precision.
- B44/B46 model thalamic relay, reticular inhibition, and corticothalamic
  feedback.
- B47 adds oscillatory synchronization.
- B48 introduces cerebellar timing.

From B49 to B62, the line became explicitly neuromodulatory and subcortical:

- B49/B50: striatal gate and habit chunking.
- B51/B54: dopaminergic modulation, cholinergic precision, noradrenergic
  arousal, and serotonergic patience.
- B55/B57: hypothalamic coupling, HPA axis, and insular interoceptive
  awareness.
- B58/B60: ACC monitor, prefrontal context, and orbitofrontal value.
- B61/B62: amygdala threat/safety value and PAG-like defensive selector for
  `safe_advance`, `flee_to_shelter`, and `freeze_hold` modes.

The global result is a bioinspired control stack: first homeostatic survival,
then spatial risk, then memory/planning, then metacognition/value, then
cortical broadcast/relay, and finally subcortical modulation of drive, stress,
safety, and defense.

## 2. Neuromodularity

The B series modularized behavior without modularizing the environment. The
external contract stayed constant: `SpiderWorld.step()` receives only one of
the nine primitive actions in `ACTIONS`; the six semantic actions of the B
series remain internal and are converted by the semantic-primitive bridge.

This choice created layered modularity:

1. **Internal semantic module**: chooses an auditable intention such as
   `MOVE_TO_FOOD`, `MOVE_TO_SHELTER`, `EXPLORE`, `STAY`, `EAT`, or `SLEEP`.
2. **Bridge module**: transforms the intention into primitive movement, using
   local signals for food, shelter, blockage, and transition.
3. **Bx controller module**: at each B level, adds a new set of signals, locks,
   balances, and decisions.
4. **Transfer module**: initializes Bx from the accepted brain of B(x-1),
   recording coverage, parent, target, source checkpoint, and loaded keys.
5. **Validation module**: accepts or discards candidates with gates focused on
   retention, probes, and trace evidence.

The pattern of each Bx module stabilized around trace fields:

- `bX_controller_profile`: which profile/controller is active.
- main module signals: for example value, conflict, attention, memory, drive,
  stress, safety, or defense.
- a composition signal: balance, confidence, stability, salience, or pressure.
- an episodic/transient lock: preserves commits without turning the action into
  a public macro action.
- `bX_decision`: explicit decision that makes it possible to reject clones of
  the previous level.

This form of modularity is important because it prevents the B series from
becoming a list of biological names without observable effect. Each module must
leave measurable evidence in the trace and must preserve the primitive contract.

The modular progression can be read in blocks:

| Block | Main modules | Modular gain |
| --- | --- | --- |
| B0-B5 | bridge, threat, contact, recovery, homeostasis | Minimal modularity between intention, bridge, and body drive. |
| B6-B14 | corridor risk, recurrence, affordance, map, waypoint, replay, confidence, attention | Separation between spatial risk, body budget, and local planning. |
| B15-B22 | options, ensembles, eligibility, episodic memory, working memory, hippocampal replay | Temporal and hierarchical decision modularity. |
| B23-B39 | conflict, precision, metacognition, value, latent state, factorization, binding | Representational and monitoring modularity. |
| B40-B48 | workspace, thalamic relay/inhibition/feedback, synchronization, timing | Global-integration and temporal-coordination modularity. |
| B49-B62 | striatum, habit, neuromodulators, hypothalamus, HPA, insula, ACC, PFC, OFC, amygdala, defense | Subcortical and affective-homeostatic modularity. |

From an engineering point of view, the neuromodularity of the B series is
incremental and auditable: the new module does not replace the previous one; it
wraps it, reads inherited signals, and decides whether to preserve, intensify,
abort, return, or block a decision.

## 3. Network Capacity

Network capacity increased conservatively. B0 established the bridge in the
current world. B1 introduced `b_hidden_dim=48` and partial transfer from B0,
with approximate coverage of `0.666896`. From that point on, the accepted line
almost always promoted the `h48` variant, because it preserved behavior with
coverage `1.0` in most transitions. `h56` variants were used as lightweight
capacity fallbacks, generally with expected coverage around `0.857`, but rarely
needed promotion.

This shows that B-series evolution was not mainly about "increasing the number
of neurons". The capacity gain came more from three sources:

- **Additional internal state**: each level added short memory, locks,
  accumulators, or decayed signals.
- **Functional decomposition**: hunger, sleep, threat, safety, confidence,
  conflict, value, attention, and defense gained separate channels.
- **Progressive environmental validation**: the same small network had to
  retain easy/canonical and then respond to probes such as `food_deprivation`,
  `sleep_vs_exploration_conflict`, `food_vs_predator_conflict`, and
  `corridor_gauntlet`.

Capacity growth can be summarized as follows:

| Stage | Type of capacity added | Example |
| --- | --- | --- |
| B0 | Interface capacity | Convert internal semantic intention into a valid primitive action. |
| B1-B5 | Homeostatic and temporal capacity | Threat guards, contact memory, sleep/hunger/recovery. |
| B6-B14 | Spatial and prospective capacity | Corridor risk, local map, waypoint, replay, and predictive attention. |
| B15-B22 | Hierarchical and episodic capacity | Options, ensembles, eligibility, working memory, and replay. |
| B23-B39 | Evaluative and representational capacity | Conflict, precision, value, latent state, factors, and binding. |
| B40-B48 | Integrative capacity | Workspace, thalamic relay, feedback, synchronization, and timing. |
| B49-B62 | Neuromodulatory and defensive capacity | Habit, dopamine, choline, noradrenaline, serotonin, HPA, amygdala, and PAG. |

The result pattern also suggests that the `h48` architecture is sufficient to
retain the accepted behaviors when the new level is a thin layer over the
previous one. When the change required parametric search, capacity was explored
through fixed profiles and genetic search, not by indiscriminately inflating the
network.

## 4. Neural-Network Techniques And Network Complexity

The central technique of the B series is incremental transfer learning. B1 and
later levels require `b_parent_level`, `b_transfer_source_checkpoint`,
`b_transfer_min_coverage`, and `b_transfer_allow_low_coverage=False`. Loading is
done by stable parameter names and compatible shape overlap: compatible
parameters are loaded; new or incompatible parameters remain normally
initialized; everything is recorded in metadata.

Techniques used throughout the series:

- **Transfer learning from accepted checkpoint**: Bx starts from the `best` of
  B(x-1), not from discarded candidates.
- **Partial weight loading**: allows increasing from `h48` to `h56` without
  requiring identical shapes, as long as the minimum coverage is respected.
- **Controllers with lightweight recurrent state**: locks, decayed memories,
  accumulated values, and episodic cooldowns.
- **Deterministic promotion gates**: fixed candidates can run in parallel, but
  promotion follows defined priority, not completion order.
- **Local genetic search**: used when interdependent thresholds were the
  bottleneck, as in B4/B5/B6 and parameterized fallbacks.
- **Ensembles and options**: B15-B17 introduce option-based decisions and
  ensemble modulation.
- **Temporal credit and RL value signals**: B18 and B31-B34 add eligibility
  traces, dopamine, actor-critic, TD error, and delayed credit.
- **Model-based/predictive components**: B10-B12, B21-B22, and B35-B36 add
  replay, prediction, transition model, and latent state.
- **Attention/workspace mechanisms**: B12-B14, B38-B41, and B43 use attention,
  binding, workspace, and adaptive precision.
- **Neuromodulatory gain control**: B27, B43, B51-B54, and B56 modulate
  thresholds, patience, arousal, precision, stress, and recovery.

Network complexity grew in layers, not in raw width. Each new phase added:

1. policy and checkpoint constants;
2. catalog config with parent/source/coverage;
3. runtime that wraps the previous level's decision;
4. new fields in `BrainStep` and trace;
5. gate that rejects a clone of the previous level without evidence of the new
   module;
6. runner that saves `attempt_report.json`, `metadata.json`, `best`, or
   `discarded`;
7. unit tests for catalog, transfer, runtime, trace, and gate.

At the end of B62, the accepted line contains a large stack of techniques, but
the operational contract remains narrow:

- public action space: still primitive;
- internal semantic actions: still the same six;
- source checkpoint: always the immediately previous accepted level;
- dominant accepted network: `h48`;
- typical recent coverage: `1.0`;
- recurring validation: compile, B-series tests, GUI tests, and `git diff --check`;
- recent accepted behavior: easy `0..4`, canonical `0..9`, homeostatic probes,
  food-predator conflict, and corridor evidence.

The current accepted phase is B62:

`artifacts/b_series/evolution/b62_defensive_mode_selector_h48_bridge_policy/seed_7/best`

It represents a neural architecture that is still small in width, but highly
modular in control: the final decision passes through semantic intention,
primitive bridge, homeostasis, risk, memory, planning, metacognition, workspace,
neuromodulation, affective value, and defensive mode before reaching a single
primitive action sent to the world.
