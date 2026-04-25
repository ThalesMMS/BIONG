# Release And Reproducibility

## Current Posture

This repository is an experimental research codebase. It is not presented as
production software, a stable SDK, or a mature release train.

In this repository, "publication-grade" means benchmark rigor and documented
reproducibility expectations for benchmark-of-record outputs. It does not mean
API stability, backwards-compatibility guarantees, or long-term support for
every interface.

## Benchmark-of-Record Artifacts

The benchmark-of-record is the benchmark package produced from the canonical
`paper` workflow, together with the run summary and exported behavior rows used
to build it.

The package-of-record should contain, at minimum:

- `benchmark_manifest.json`
- `resolved_config.json`
- `pip_freeze.txt`
- `seed_level_rows.csv`
- `aggregate_benchmark_tables.json`
- `credit_table.json`
- `capacity_sweep_tables.json`
- `claim_test_tables.json`
- `effect_size_tables.json`
- `reward_profile_ladder_tables.json`
- `ladder_comparison.json`
- `ladder_profile_comparison.json`
- `unified_ladder_report.json`
- `report.md` and `report.json`
- `plots/*.svg`
- `supporting_csvs/*.csv`
- `limitations.txt`

The manifest is the top-level provenance index. It records package metadata,
file hashes, resolved budget/checkpoint metadata, and the code/runtime
provenance needed to tie the package to a specific revision and environment.

## Dependency Posture

`requirements.txt` remains a minimal development baseline and is intentionally
not treated as a locked benchmark environment.

For benchmark-of-record runs, preserve the exact environment in one of these
ways:

Use a supported Python 3.10+ interpreter for benchmark packaging so the captured
environment reflects a replayable runtime.

1. Prefer the benchmark package itself, which records Python details plus a
   `pip freeze` snapshot in `pip_freeze.txt` and records environment metadata
   in `benchmark_manifest.json`.
2. If a standalone environment export is needed outside the package, run
   `python3.10 -m pip freeze > benchmark_requirements_lock.txt` from the same
   environment and archive it beside the benchmark package.

## Release Notes Convention

Use [CHANGELOG.md](./CHANGELOG.md) for changes that materially affect:

- benchmark reproducibility
- benchmark interpretation
- architecture or interface versioning relevant to reported results

Routine code churn does not need a changelog entry.

## Minimal Checklist

Use this checklist when producing a benchmark-of-record package or preparing a
future tag tied to benchmark results.

See the canonical release checklist at
[docs/release_checklist.md](./docs/release_checklist.md) for the full process.
That checklist covers pre-run verification, execution, post-run artifact
checks, archival guidance, and environment recreation from `pip_freeze.txt`.
