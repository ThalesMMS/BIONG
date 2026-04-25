# Release Checklist

This checklist is the operational path for producing benchmark-of-record
artifacts from the current research workflow. It is intentionally narrow: the
goal is to make the benchmark package auditable and replayable, not to imply a
mature product-release process.

## Pre-run checks

- Confirm the run is using the benchmark-of-record path:
  `--budget-profile paper --checkpoint-selection best`.
- If the run is for the no-reflex learning experiment-of-record instead of the
  architecture benchmark-of-record, confirm the command path in
  [ablation_workflow.md](./ablation_workflow.md) before proceeding.
- Check git state before the run. Prefer a clean working tree. If the tree is
  intentionally dirty, record the reason in your notes and make sure the dirty
  state is acceptable for the claim you intend to make.
- Confirm the expected architecture and interface versions before citing the
  result:
  `spider_brain/metadata.json` should match the intended architecture/interface
  state for the benchmark.
- Confirm the intended checkpoint source, output paths, and package directory
  names before starting the run.

## Execution steps

- Use the canonical benchmark-of-record commands from
  [ablation_workflow.md](./ablation_workflow.md).
- For architecture benchmark-of-record packages, run the `paper` ablation path
  with `--benchmark-package`.
- Save the JSON summary and exported behavior rows alongside the benchmark
  package.

## Post-run verification

- Confirm the benchmark package contains:
  `benchmark_manifest.json`, `resolved_config.json`, `pip_freeze.txt`,
  `seed_level_rows.csv`, `report.md`, `report.json`, plots, supporting CSVs,
  and `limitations.txt`.
- Review `limitations.txt` before sharing or citing the output.
- Open `benchmark_manifest.json` and verify the expected environment fields are
  present:
  `git_commit`, `git_tag`, `git_dirty`, `python_version`, and `platform`.
- Confirm `pip_freeze.txt` was captured and hashed as part of the package
  contents.
- Confirm the package reflects the intended claim target, budget profile, and
  checkpoint-selection metadata.

## Archival guidance

- Prefer committing the code state used for the benchmark before sharing the
  result.
- If the result will be referenced externally, tag the commit that matches the
  package's recorded `git_commit` when appropriate.
- Archive the benchmark package as a whole, including its `pip_freeze.txt`,
  rather than copying only selected plots or tables.
- Keep the summary JSON and behavior rows that produced the package with the
  archived bundle.

## Reproducibility / Environment Preservation

`pip_freeze.txt` is captured automatically in benchmark packages.

To recreate an environment from a benchmark package, use Python 3.10+ and a
versioned interpreter command:

`python3.10 --version`

If `python3.10` is unavailable, install Python 3.10+ or substitute your
available versioned interpreter, such as `python3.11`. The benchmark manifest
records the original Python version used for the package; match that version
when possible, or use a newer supported Python release.

1. Create a virtual environment:
   `python3.10 -m venv venv`
2. Activate it:
   `source venv/bin/activate`
3. Install the captured benchmark environment:
   `pip install -r pip_freeze.txt`

`requirements.txt` remains intentionally unpinned for development flexibility.
For benchmark-of-record reruns, use the captured `pip_freeze.txt` from the
package rather than treating `requirements.txt` as a lockfile.
