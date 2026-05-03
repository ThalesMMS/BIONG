from __future__ import annotations

"""Lightweight verification of gate-status normalization/highlighting rules.

This is intentionally not a full pytest suite (the environment may not have
pytest available). It runs a few assertions over synthetic inputs so we can
confirm the normalization logic used by the index is consistent.
"""

from spider_cortex_sim.offline_analysis.reporting.writer import normalize_gate_status


def _run() -> None:
    cases = [
        (True, 'pass'),
        (False, 'fail'),
        ('pass', 'pass'),
        ('PASSED', 'pass'),
        ('ok', 'pass'),
        ('success', 'pass'),
        ('fail', 'fail'),
        ('FAILED', 'fail'),
        ('error', 'fail'),
        ('skipped', 'skipped'),
        ('skip', 'skipped'),
        ('not_run', 'skipped'),
        ('not_applicable', 'skipped'),
        ('borderline', 'borderline'),
        ('warning', 'borderline'),
        ('soft_fail', 'borderline'),
        ({'passed': True}, 'pass'),
        ({'passed': False}, 'fail'),
        ({'status': 'skipped'}, 'skipped'),
        ({'result': 'borderline'}, 'borderline'),
        (None, 'unknown'),
        ('???', 'unknown'),
        ({}, 'unknown'),
    ]

    for value, expected in cases:
        got = normalize_gate_status(value)
        assert got == expected, f"{value!r}: expected {expected!r}, got {got!r}"


if __name__ == '__main__':
    _run()
    print('OK: gate-status normalization cases passed')
