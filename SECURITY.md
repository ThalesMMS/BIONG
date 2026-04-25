# Security

This is an experimental research codebase. Security handling is intentionally
small, but sensitive findings still need a private reporting path.

## Reporting a Vulnerability

Do not open a public issue for exploitable security problems, private data
exposure, or sensitive benchmark-integrity findings until they have been
triaged.

Use GitHub private vulnerability reporting:

1. Open the repository on GitHub.
2. Go to the Security tab.
3. Choose "Report a vulnerability".

If private vulnerability reporting is unavailable, contact `ThalesMMS` directly
through GitHub and include enough detail to reproduce or evaluate the concern.

## Scope

Security-sensitive reports may include:

- dependency vulnerabilities that affect normal project use
- unsafe handling of local files, paths, or generated artifacts
- benchmark-integrity issues that could silently corrupt reproducibility claims
- data-handling problems involving traces, summaries, reports, or packaged
  artifacts

Regular bugs, modeling limitations, and research disagreements should use the
normal issue templates instead.

## Response

The maintainer will respond as capacity allows. No response SLA is promised.

Public disclosure should wait until the issue is understood and a fix or
mitigation path is available.
