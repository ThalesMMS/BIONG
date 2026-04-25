# Contributing

This is an experimental research codebase. Keep contributions small,
reviewable, and tied to simulator behavior, benchmark reproducibility, or
documentation clarity.

## Local Setup

Use [README.md](README.md) as the source of truth for:

- Python 3.10+
- choosing `PYTHON_BIN`
- creating a virtual environment
- installing dependencies
- running the smoke workflow

Run commands from the repository root.

## Running Tests

Run the full test suite before opening a pull request:

```bash
$PYTHON_BIN -m unittest discover -s tests -v
```

For documentation-only changes, say whether tests were skipped.

## Proposing Changes

- Open an issue first for large behavior, architecture, or benchmark changes.
- Keep pull requests focused on one concern.
- Describe the research or reproducibility motivation, not only the code change.
- Do not assume public API stability. This repository prioritizes benchmark
  rigor and explicit interface metadata over long-term compatibility.
- Update docs and tests when behavior, CLI output, benchmark artifacts, or
  interfaces change.

## Interface Governance

The pull-request template is the checklist of record for interface-sensitive
changes. Review whether your change affects the interface registry, generated
interface docs, architecture signature, checkpoint compatibility, `summary`, or
`behavior_csv` outputs.

For interface or public-contract changes, confirm:

- [ ] interface versions are updated or intentionally unchanged
- [ ] `architecture_signature()` and the interface registry fingerprint are
      reviewed
- [ ] `docs/interfaces.md` is regenerated and checked
- [ ] save/load compatibility and checkpoint rejection behavior are verified
      when applicable
- [ ] tests cover schema, save/load, `summary`, and `behavior_csv` behavior when
      public contracts change

Explain what changed, why it changed, and how it was checked. Link related
issues when available.

Maintainer response is best-effort. No review or support timeline is guaranteed.
