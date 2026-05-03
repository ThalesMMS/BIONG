# Behavioral gate diagnostics (night_rest + two_shelter_tradeoff)

This note documents how to run the deterministic behavioral fixtures and the diagnostic/reporting artifacts added for the `night_rest` and `two_shelter_tradeoff` gates.

All commands assume you run from the repository root.

## 0) Choose your Python

The repository expects Python 3.10+.

```bash
export PYTHON_BIN="${PYTHON_BIN:-python3.10}"
$PYTHON_BIN --version
```

If you run from a different directory, set `PYTHONPATH` to the repository root.

## 1) Run a single scenario (fixtures) and export CSV + JSON summary

The scenario fixtures are baked into the scenario setup code; you do not need to enable them separately.

### night_rest

```bash
$PYTHON_BIN -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --behavior-scenario night_rest \
  --behavior-seeds 0 1 2 3 4 \
  --scenario-episodes 10 \
  --behavior-csv /tmp/night_rest.csv \
  --summary /tmp/night_rest_summary.json
```

### two_shelter_tradeoff

```bash
$PYTHON_BIN -m spider_cortex_sim \
  --episodes 0 \
  --eval-episodes 0 \
  --behavior-scenario two_shelter_tradeoff \
  --behavior-seeds 0 1 2 3 4 \
  --scenario-episodes 10 \
  --behavior-csv /tmp/two_shelter_tradeoff.csv \
  --summary /tmp/two_shelter_tradeoff_summary.json
```

Notes:

- `--behavior-seeds ...` makes runs deterministic and repeatable.
- `--scenario-episodes` controls how many episodes are run per seed.
- The exported `--behavior-csv` is the flat per-episode table used by the gate report tooling.

## 2) Generate a concise report from the artifacts

The standardized report consumes the scenario `--summary` JSON and the exported behavior CSV.

```bash
$PYTHON_BIN -m spider_cortex_sim.scenarios.gate_report_cli \
  --gate night_rest \
  --summary /tmp/night_rest_summary.json \
  --behavior-csv /tmp/night_rest.csv \
  --output-dir /tmp/night_rest_gate_report
```

This writes:

- `/tmp/night_rest_gate_report/gate_report.md`
- `/tmp/night_rest_gate_report/gate_report.json`

Repeat for `two_shelter_tradeoff` by swapping the input paths and setting `--gate two_shelter_tradeoff`.

## 3) Where diagnostics live

Diagnostics are published into two places:

1) **Scenario summary JSON**: `summary["behavior_evaluation"]["suite"][<scenario>]["diagnostics"]`
2) **Behavior CSV columns**: scenario metrics are exported as `metric_<name>` columns.

In other words:

- machine-readable, structured diagnostics: the JSON summary
- per-episode numeric/string metrics: the CSV

## 4) Interpreting two_shelter_tradeoff failure attribution

`two_shelter_tradeoff` emits a per-episode label:

- CSV column: `metric_failure_attribution`

The label is intended as a *debugging hint* about which stage likely caused the failure:

- `perception`: the agent likely never perceived both candidate shelters (e.g., missing sensor evidence / occlusion / range)
- `memory`: the agent perceived candidates but did not encode/recall the relevant shelter attributes when deciding
- `arbitration`: candidates were available but the action selection compared the wrong options or ranked them incorrectly
- `reward`: utilities/terms appear inconsistent with the intended preference (e.g., scoring sign/scale issues)
- `motor`: a target was selected but pathing/target binding failed to execute the intended move sequence
- `success`: the episode satisfied the scenario success checks

Use it together with the other exported shelter metrics (distance deltas, shelter occupancy rates, survival flags) to decide whether the issue is upstream (sensing/memory) or downstream (action/motor).

## 5) Re-running deterministically (sanity check)

To verify determinism, run the same seed twice and compare output CSVs.

```bash
$PYTHON_BIN -m spider_cortex_sim --episodes 0 --eval-episodes 0 --behavior-scenario night_rest --behavior-seeds 0 --scenario-episodes 3 --behavior-csv /tmp/nr_a.csv --summary /tmp/nr_a.json
$PYTHON_BIN -m spider_cortex_sim --episodes 0 --eval-episodes 0 --behavior-scenario night_rest --behavior-seeds 0 --scenario-episodes 3 --behavior-csv /tmp/nr_b.csv --summary /tmp/nr_b.json

diff -u /tmp/nr_a.csv /tmp/nr_b.csv
```

No diff indicates deterministic behavior for that scenario/seed set.
