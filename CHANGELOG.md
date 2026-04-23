# Changelog

This changelog tracks changes that can affect benchmark reproducibility,
benchmark interpretation, or the architecture/process claims attached to
published results. It is intentionally narrower than the full commit history.

## 2026-04-23

### Benchmark Changes

- Added benchmark-package environment capture for git commit, git tag, dirty
  working-tree state, Python version, and platform in the manifest.
- Added `pip_freeze.txt` to benchmark-of-record packages so the dependency
  snapshot is preserved as a hashed package artifact.

### Documentation

- Added explicit release posture, citation guidance, and benchmark-of-record
  checklist documentation.
- Defined a lightweight release-notes convention for future benchmark or
  process changes that affect reproducibility or interpretation.

### Architecture

- No architecture or interface contract change; this entry establishes a
  release and reproducibility baseline around the existing workflow.
