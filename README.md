# Neuro-Modular Spider Simulator With Explicit Predator And Standardized Interfaces

This project implements a simulated spider with a modular brain, explicit predator pressure, standardized neural interfaces, online learning, deterministic behavioral scenarios, and reproducible evaluation workflows.

The current version centers on three structural changes:

1. an explicit predator (`lizard`) inside the world
2. a primitive locomotion output space instead of high-level semantic actions
3. standardized input and output interfaces for each center or cortex to reduce excessive coupling between networks

That core has since been extended with:

- reward profiles (`classic`, `ecological`, `austere`)
- explicit shelter geometry with entrance, interior, and deep zones
- a richer lizard state machine (`PATROL`, `ORIENT`, `INVESTIGATE`, `CHASE`, `WAIT`, `RECOVER`)
- lizard working memory for investigate targets, ambush windows, chase streaks, and recovery
- explicit spider memory for food, predator, safe shelter, and escape route
- map templates and deterministic scenarios for behavioral evaluation
- behavioral scorecards, ablations, learning-evidence workflows, reward audits, and offline analysis

The system still uses small NumPy neural networks, online learning, and independent modules, but the current architecture is closer to a simplified organism under ecological constraints.

## What Changed

### 1. Explicit Predator: The Lizard

Probabilistic nighttime danger was replaced by a concrete predator in the environment.

The lizard:

- occupies a real grid cell
- has limited vision
- moves more slowly than the spider
- cannot enter the shelter
- causes pain and health loss on contact
- can be escaped if the spider keeps moving effectively

### 2. Primitive Motor Output

The motor system no longer chooses semantic actions such as `MOVE_TO_FOOD` or `MOVE_TO_SHELTER`.

The action space is now:

- `MOVE_UP`
- `MOVE_DOWN`
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `STAY`

That keeps the simulator closer to basic locomotion control instead of scripted behavior verbs.

### 3. Eating And Sleeping As Situated Behaviors

Because the motor output is now limited to locomotion, feeding and rest emerge from spatial context:

- when the spider reaches food, feeding happens automatically
- when the spider reaches shelter under fatigue or night pressure, recovery happens automatically
- staying still can still help feeding or rest, but it is no longer a rigid prerequisite

## Modular Architecture

The brain now contains five proposer modules plus `action_center` and `motor_cortex`:

1. `visual_cortex`
2. `sensory_cortex`
3. `hunger_center`
4. `sleep_center`
5. `alert_center`
6. `action_center`
7. `motor_cortex`

The first five networks propose locomotion in the same output space. `action_center` applies valence-based priority gating before final arbitration, and `motor_cortex` acts as the final locomotor executor or corrector.

The current architecture also includes:

- fixed, named interfaces per module
- module dropout during training
- local reflexes per module, defined from the module's own interface
- auxiliary per-module targets ("reflex targets") to reduce coadaptation
- explicit contribution metrics in `action_center` for dominance, agreement, and effective proposer count
- the `local_credit_only` ablation, which preserves current inference but removes global policy-gradient broadcast during modular training

Layer responsibilities:

- `spider_cortex_sim/world.py`: ecological dynamics, body state, and explicit observable memory
- `spider_cortex_sim/interfaces.py`: named contracts between world and brain
- `spider_cortex_sim/modules.py`: proposer networks only
- `spider_cortex_sim/agent.py`: local reflexes, auxiliary targets, valence gating, and final motor correction

The generated contract documentation lives in [docs/interfaces.md](docs/interfaces.md). The short topology note for the newer arbitration chain lives in [docs/action_center_design.md](docs/action_center_design.md).

## Environment

The 2D grid world contains:

- shelter or burrow (`H`)
- shelter spatial roles (`entrance`, `inside`, `deep`)
- walls or blocked cells (`#`)
- clutter terrain (`:`)
- food (`F`)
- spider (`A`)
- predator lizard (`L`)
- day/night cycle
- limited vision
- food and predator smell fields
- homeostatic pressures from hunger, fatigue, health, pain, and contact

If the spider and lizard occupy the same ASCII-rendered cell, the renderer shows `X`.

## Project Layout

```text
neuro_modular_sim/
├── README.md
├── docs/
│   ├── ablation_workflow.md
│   ├── action_center_design.md
│   └── interfaces.md
├── spider_cortex_sim/
│   ├── __main__.py
│   ├── agent.py
│   ├── bus.py
│   ├── cli.py
│   ├── gui.py
│   ├── interfaces.py
│   ├── modules.py
│   ├── nn.py
│   ├── simulation.py
│   └── world.py
└── tests/
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `numpy` is required
- `pygame-ce` is optional and only needed for the graphical interface (`--gui`)

## Quick Start

Run a standard training and evaluation session:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim --episodes 120 --eval-episodes 3 --max-steps 90
```

Save a summary and trace:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 \
  --eval-episodes 1 \
  --max-steps 90 \
  --summary spider_summary.json \
  --trace spider_trace.jsonl
```

Render the final evaluation episode in ASCII:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 \
  --eval-episodes 1 \
  --max-steps 90 \
  --render-eval
```

## Reward Profiles And Maps

Train with the ecological profile and an alternate map:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 \
  --eval-episodes 3 \
  --max-steps 90 \
  --reward-profile ecological \
  --map-template side_burrow
```

Available map templates:

- `central_burrow`
- `side_burrow`
- `corridor_escape`
- `two_shelters`
- `exposed_feeding_ground`
- `entrance_funnel`

Map notes:

- `corridor_escape` uses `NARROW` terrain, representing a constrained passage
- `two_shelters` introduces two deep shelters competing with a central transition zone
- `exposed_feeding_ground` concentrates food in open terrain with more exposed approach paths
- `entrance_funnel` compresses the shelter entrance through a bottleneck that favors blocking and ambush

Reward profiles:

- `classic`: more guided, best for quick training and baseline runs
- `ecological`: less direct shaping and more pressure from world dynamics
- `austere`: minimal-progress baseline for shaping audits and contrast

## Deterministic Scenarios

Run a single scenario:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --scenario night_rest
```

Run the full scenario suite:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --scenario-suite
```

Available scenarios:

- `night_rest`
- `predator_edge`
- `entrance_ambush`
- `open_field_foraging`
- `shelter_blockade`
- `recover_after_failed_chase`
- `corridor_gauntlet`
- `two_shelter_tradeoff`
- `exposed_day_foraging`
- `food_deprivation`
- `food_vs_predator_conflict`
- `sleep_vs_exploration_conflict`

Scenario-to-map specializations include:

- `entrance_ambush`, `shelter_blockade`, and `recover_after_failed_chase` use `entrance_funnel`
- `open_field_foraging` and `exposed_day_foraging` use `exposed_feeding_ground`
- `corridor_gauntlet` uses `corridor_escape`
- `two_shelter_tradeoff` uses `two_shelters`

## Behavioral Evaluation

Run the full behavioral suite with explicit scorecards:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --behavior-suite \
  --full-summary
```

Run one behavioral scenario and export flat CSV:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --behavior-scenario night_rest \
  --behavior-csv spider_behavior.csv \
  --full-summary
```

`summary["behavior_evaluation"]` includes:

- `suite`: aggregated scenario scorecards with `success_rate`, `checks`, `behavior_metrics`, `diagnostics`, and `failures`
- `summary`: overall suite success and detected regressions
- `comparisons`: optional profile/map/seed comparison matrices when behavioral comparison flags are used
- `learning_evidence`: optional comparison between a trained checkpoint and controls such as `random_init`, `reflex_only`, `freeze_half_budget`, and `trained_long_budget`

Weak-signal scenarios now publish scenario-owned interpretation metadata:

- `diagnostic_focus`
- `success_interpretation`
- `failure_interpretation`
- `budget_note`

The scenario diagnostics block also summarizes:

- `primary_outcome`
- `outcome_distribution`
- `partial_progress_rate`
- `died_without_contact_rate`

## Graphical Interface (Pygame)

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --gui \
  --episodes 120 \
  --eval-episodes 3 \
  --max-steps 90 \
  --reward-profile classic \
  --map-template central_burrow
```

The GUI displays:

- the predator lizard on the grid
- shelter roles and terrain types
- contact, sighting, and escape counters
- lizard position, mode, and current target
- explicit spider memory and sleep debt
- recent reward components
- nighttime shelter-role distribution and predator-state occupancy

Useful shortcuts:

- `V`: toggle visibility overlay
- `M`: toggle smell heatmap

## Saving And Loading The Brain

Train and save:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 120 --eval-episodes 3 --max-steps 90 \
  --save-brain spider_brain
```

Load and continue training:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 60 --eval-episodes 3 --max-steps 90 \
  --load-brain spider_brain \
  --save-brain spider_brain
```

Load only selected modules:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --episodes 60 --eval-episodes 3 --max-steps 90 \
  --load-brain spider_brain \
  --load-modules visual_cortex hunger_center
```

Compatibility notes:

- the current architecture uses an explicit interface signature
- older saves predating the current interface standardization are rejected with explicit incompatibility errors
- older checkpoints predating `sleep_phase`, `rest_streak`, `sleep_debt`, shelter-role signals, certainty/occlusion signals, explicit memory, or oriented perception are also incompatible with the current architecture
- the versioned interface registry and generated contract docs are in [docs/interfaces.md](docs/interfaces.md)

Because interface descriptions are part of the current fingerprinted metadata, this English translation pass also changes interface and architecture fingerprints. Older checkpoints may therefore fail compatibility checks even though behavior and identifiers were not intentionally refactored.

## Tests

Run the full suite:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The tests cover:

- standardized interface shapes and generated interface docs
- contextual feeding and rest
- auditable reward decomposition
- sleep progression `SETTLING -> RESTING -> DEEP_SLEEP`, sleep debt, and interruptions
- predator contact with real damage
- shelter geometry and occlusion
- explicit spider memory
- map templates and reachability
- deterministic scenario regressions
- memory-guided escape and foraging after loss of sight
- scenario runners and deterministic predator-response latency
- online parameter updates
- lightweight trainability checks across reward profiles, comparisons, and alternate maps

## Metrics And Tracing

The `summary` and `trace` include:

- per-step `reward_components`
- `reward_audit`, including component inventory, shaping categories, and leakage candidates
- nighttime shelter occupancy and nighttime stillness
- nighttime shelter-role distribution (`outside`, `entrance`, `inside`, `deep`)
- predator-response latency
- `reward_profile` and `map_template`
- `config.operational_profile`, including active thresholds and operational weights
- `config.budget`, including resolved profile, benchmark strength, seeds, and explicit overrides
- `checkpointing` when `--checkpoint-selection best` is used
- explicit certainty and occlusion fields per visual channel
- world-owned `heading` and decayed percept traces in trace metadata
- explicit `predator_motion_salience`
- normalized memory vectors in trace metadata
- predator occupancy by state (`PATROL`, `ORIENT`, `INVESTIGATE`, `CHASE`, `WAIT`, `RECOVER`)
- predator state transitions and dominant predator state per episode or scenario
- food and shelter distance deltas
- `event_log` stages per tick
- behavioral scorecards per scenario in `behavior_evaluation`
- diagnostic per-episode bands such as `progress_band` and `outcome_band`
- explicit `action_center` arbitration outputs such as `winning_valence`, `valence_scores`, `module_gates`, `suppressed_modules`, and `evidence`

When `--debug-trace` is combined with `--trace`, each tick also includes:

- serialized observations before and after transition
- reward components
- normalized memory vectors
- per-module logits before reflex, reflex delta, and post-reflex logits
- logits after valence gating, per-module `gate_weight`, and `debug.arbitration`
- `effective_reflex_scale`, `module_reflex_override`, `module_reflex_dominance`, and `final_reflex_override`
- full predator internal state

Use `--full-summary` to print the complete JSON summary to stdout.

## Budget Profiles

The CLI exposes explicit budget profiles:

- `smoke`: quick sanity or CI profile (`6` episodes, `1` evaluation run, `60` steps, seed `7`)
- `dev`: short reproducible local benchmark (`12` episodes, `2` evaluation runs, `90` steps, seeds `7/17/29`)
- `report`: stronger reporting workflow (`24` episodes, `4` evaluation runs, `120` steps, `2` repetitions per scenario, seeds `7/17/29/41/53`)

Canonical commands:

```bash
# smoke: sanity / CI
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --behavior-suite --full-summary

# dev: fast local benchmark
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite --full-summary

# report: stronger benchmark + automatic checkpoint selection
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile report \
  --checkpoint-selection best \
  --ablation-suite --full-summary
```

Without `--budget-profile`, the run still works in `custom` mode and records the effective values and overrides in `summary["config"]["budget"]`.

## Comparison Workflows

Compare reward profiles on the current map:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --compare-profiles --full-summary
```

Compare maps under the current reward profile:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --reward-profile ecological \
  --compare-maps --full-summary
```

Compare the behavioral suite across profiles:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --behavior-compare-profiles --full-summary
```

Compare the behavioral suite across maps and export CSV:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --reward-profile ecological \
  --behavior-compare-maps \
  --behavior-csv spider_behavior_compare.csv \
  --full-summary
```

The shaping audit uses `austere` as the minimal baseline and records deltas against it under `summary["behavior_evaluation"]["shaping_audit"]`.

## Offline Analysis

The project includes a separate runner that transforms `summary.json`, `trace.jsonl`, and `behavior_csv` into an offline analysis bundle:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim.offline_analysis \
  --summary spider_summary_compare.json \
  --trace spider_trace_debug.jsonl \
  --behavior-csv spider_behavior_compare.csv \
  --output-dir offline_analysis
```

Rules:

- `--output-dir` is required
- at least one of `--summary`, `--trace`, or `--behavior-csv` is required
- the report is always emitted, even with partial input
- missing blocks are reported in `report.md` and `report.json` instead of aborting execution

## Ablations And Learning Evidence

Compare the modular reference against the canonical ablation suite:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile dev \
  --ablation-suite \
  --behavior-csv spider_ablation_rows.csv \
  --full-summary
```

Run the `learning_evidence` suite under the smoke budget:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --behavior-scenario night_rest \
  --behavior-csv spider_learning_evidence_rows.csv \
  --full-summary
```

Run the canonical short-vs-long learning-evidence comparison:

```bash
PYTHONPATH=. python3 -m spider_cortex_sim \
  --budget-profile smoke \
  --learning-evidence \
  --learning-evidence-long-budget-profile report \
  --behavior-suite \
  --summary spider_learning_evidence_summary.json \
  --behavior-csv spider_learning_evidence_rows.csv \
  --full-summary
```

The detailed ablation workflow, variant definitions, and canonical check-in table live in [docs/ablation_workflow.md](docs/ablation_workflow.md).

## Modeling Notes

- The system is biologically inspired, not biologically faithful
- The "return vector to shelter" is a simplified form of proprioception or minimal spatial memory
- Explicit memory is world-owned: target, age, and TTL are maintained by the environment and only consumed by cortical modules through observation
- Local per-module reflexes act like innate behavior that online learning later refines
- There is no giant fallback center that integrates everything; each proposer receives only its own interface and emits only standardized locomotion proposals

## Natural Extensions

1. add recurrent memory per module
2. introduce multiple predators with different sensory niches
3. model a more strongly oriented field of view rather than a short omnidirectional one
4. separate locomotion into gait, speed, and body orientation
5. migrate the networks to PyTorch while preserving the same modular interface signature
