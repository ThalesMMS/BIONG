from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .writer import (
    assign_github_anchors,
    extract_toc_from_markdown_lines,
    normalize_gate_status,
)


def _table_rows(table: object) -> list[Mapping[str, object]]:
    if not isinstance(table, Mapping):
        return []
    rows = table.get("rows", [])
    return [item for item in rows if isinstance(item, Mapping)] if isinstance(rows, list) else []


def _gate_status(check: Mapping[str, object]) -> str:
    normalized = normalize_gate_status(check)
    if normalized != "unknown":
        return normalized

    pass_rate = check.get("pass_rate")
    if pass_rate is None:
        return "unknown"
    try:
        value = float(pass_rate)
    except (TypeError, ValueError):
        return "unknown"
    if value >= 0.999:
        return "pass"
    if value <= 0.0:
        return "fail"
    return "borderline"


def _markdown_table_cell(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", r"\|")


def collect_claims_and_evidence(report: Mapping[str, object]) -> list[dict[str, object]]:
    claim_tables = report.get("claim_test_tables", {})
    claim_results = {}
    if isinstance(claim_tables, Mapping):
        claim_results = claim_tables.get("claim_results", {})
    rows = _table_rows(claim_results)

    claim_rows: dict[str, list[Mapping[str, object]]] = {}
    for item in rows:
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        claim_rows.setdefault(claim, []).append(item)

    def _first_role(role: str, items: list[Mapping[str, object]]) -> Mapping[str, object]:
        for row in items:
            if str(row.get("role") or "") == role:
                return row
        return items[0] if items else {}

    claims: list[dict[str, object]] = []
    for claim, items in sorted(claim_rows.items()):
        representative = (
            _first_role("effect_size", items)
            or _first_role("delta", items)
            or _first_role("comparison", items)
            or _first_role("reference", items)
        )
        status = str(representative.get("status") or "").strip()
        passed = representative.get("passed")

        comparisons = sorted(
            {
                str(row.get("condition") or "").strip()
                for row in items
                if str(row.get("role") or "")
                in {"comparison", "delta", "effect_size"}
                and str(row.get("condition") or "").strip()
                and str(row.get("condition") or "").strip() not in {"reference"}
            }
        )

        claims.append(
            {
                "claim": claim,
                "status": status,
                "passed": bool(passed) if passed is not None else None,
                "comparisons": comparisons,
                "links": {
                    "report_md": "./report.md",
                    "claim_test_tables_json": "./claim_test_tables.json",
                    "aggregate_benchmark_tables_json": "./aggregate_benchmark_tables.json",
                    "effect_size_tables_json": "./effect_size_tables.json",
                    "seed_level_rows_csv": "./seed_level_rows.csv",
                    "traces_dir": "./traces/",
                },
            }
        )

    return claims


def collect_scenario_summaries(report: Mapping[str, object]) -> list[dict[str, object]]:
    """Extract per-scenario summary metadata for the top-level report index.

    Notes:
        Scenario checks may appear either as a top-level `report['scenario_checks']` table
        (older layout) or embedded per scenario in `report['scenario_success']['scenarios'][*]['checks']`.
    """

    scenario_success = report.get("scenario_success", {})
    scenarios = []
    if isinstance(scenario_success, Mapping):
        scenarios = scenario_success.get("scenarios", [])

    check_rows = report.get("scenario_checks", [])
    checks: list[Mapping[str, object]] = (
        [item for item in check_rows if isinstance(item, Mapping)]
        if isinstance(check_rows, list)
        else []
    )

    checks_by_scenario: dict[str, list[Mapping[str, object]]] = {}
    for item in checks:
        scenario = str(item.get("scenario") or "").strip()
        if not scenario:
            continue
        checks_by_scenario.setdefault(scenario, []).append(item)

    summaries: list[dict[str, object]] = []
    for item in scenarios if isinstance(scenarios, list) else []:
        if not isinstance(item, Mapping):
            continue
        scenario = str(item.get("scenario") or "").strip()
        if not scenario:
            continue

        embedded_checks: list[Mapping[str, object]] = []
        embedded_check_map = item.get("checks")
        if isinstance(embedded_check_map, Mapping):
            embedded_checks = [
                {"check_name": name, **(payload if isinstance(payload, Mapping) else {})}
                for name, payload in embedded_check_map.items()
                if isinstance(name, str)
            ]

        summary_checks = checks_by_scenario.get(scenario) or embedded_checks

        summaries.append(
            {
                "scenario": scenario,
                "success_rate": item.get("success_rate"),
                "check_count": int(item.get("check_count") or len(summary_checks) or 0),
                "gate_statuses": [
                    {
                        "check_name": str(check.get("check_name") or ""),
                        "pass_rate": check.get("pass_rate"),
                        "status": _gate_status(check),
                        "source": str(check.get("source") or ""),
                    }
                    for check in summary_checks
                ],
            }
        )
    return summaries


def _scenario_plots(output_path: Path, scenario: str) -> list[str]:
    scenario_slug = scenario.strip().lower().replace(" ", "_")
    candidates: list[Path] = []
    plots_dir = output_path / "plots"
    if plots_dir.exists():
        candidates.extend(sorted(plots_dir.glob("*.svg")))
    candidates.extend(sorted(output_path.glob("*.svg")))

    matched: list[str] = []
    for path in candidates:
        name = path.name.lower()
        if scenario_slug and scenario_slug in name:
            matched.append(f"./{path.relative_to(output_path).as_posix()}")
    return matched


def _has_path(output_path: Path, relative_path: str) -> bool:
    relative_path = relative_path.lstrip("./")
    return (output_path / relative_path).exists()


def write_index_markdown(*, output_dir: str | Path, report: Mapping[str, object]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    claims = collect_claims_and_evidence(report)
    scenarios = collect_scenario_summaries(report)

    report_md_path = output_path / "report.md"
    claim_anchor = ""
    scenario_anchor = ""
    limitations_anchor = ""
    if report_md_path.exists():
        report_lines = report_md_path.read_text(encoding="utf-8").splitlines()
        toc_headings = extract_toc_from_markdown_lines(report_lines)
        assigned = assign_github_anchors(toc_headings)
        for _, heading, anchor in assigned:
            if heading == "Claim Test Results with Uncertainty":
                claim_anchor = anchor
            if heading == "Scenario Success":
                scenario_anchor = anchor
            if heading == "Limitations":
                limitations_anchor = anchor

    def _status_badge(status: str) -> str:
        status = (status or "").strip().lower()
        if status in {"fail", "failed"}:
            return "**FAIL**"
        if status in {"skip", "skipped"}:
            return "**SKIP**"
        if status in {"borderline", "warn", "warning"}:
            return "**BORDERLINE**"
        if status in {"pass", "passed"}:
            return "PASS"
        return status or "UNKNOWN"

    def _maybe_link(label: str, href: str) -> str:
        return f"[{label}]({href})" if href and _has_path(output_path, href) else ""

    quick_links: list[str] = [
        "- [Report markdown](./report.md)" if _has_path(output_path, "report.md") else "- Report markdown (missing)",
        "- [Report JSON](./report.json)" if _has_path(output_path, "report.json") else "- Report JSON (missing)",
        "- [Aggregate benchmark tables](./aggregate_benchmark_tables.json)" if _has_path(output_path, "aggregate_benchmark_tables.json") else "- Aggregate benchmark tables (missing)",
        "- [Claim test tables](./claim_test_tables.json)" if _has_path(output_path, "claim_test_tables.json") else "- Claim test tables (missing)",
        "- [Limitations](./limitations.txt)" if _has_path(output_path, "limitations.txt") else "- Limitations (missing)",
    ]

    lines: list[str] = [
        "# Report index",
        "",
        "## Quick links",
        "",
        *quick_links,
        "",
        "## Scenarios",
        "",
        "Status legend: **FAIL**, **BORDERLINE**, **SKIP**, PASS",
        "",
    ]

    if scenario_anchor and _has_path(output_path, "report.md"):
        lines.append(
            f"_Scenario section in report: [Scenario Success](./report.md#{scenario_anchor})_\n"
        )

    if limitations_anchor and _has_path(output_path, "report.md"):
        lines.append(
            f"_Limitations section in report: [Limitations](./report.md#{limitations_anchor})_\n"
        )

    if scenarios:
        lines.append("| scenario | success_rate | checks | gates |")
        lines.append("|---|---:|---:|---|")
        for item in scenarios:
            statuses = item.get("gate_statuses")
            failed = 0
            borderline = 0
            skipped = 0
            unknown = 0
            if isinstance(statuses, list):
                failed = sum(1 for s in statuses if isinstance(s, Mapping) and s.get("status") == "fail")
                borderline = sum(
                    1 for s in statuses if isinstance(s, Mapping) and s.get("status") == "borderline"
                )
                skipped = sum(
                    1 for s in statuses if isinstance(s, Mapping) and str(s.get("status") or "") in {"skip", "skipped"}
                )
                unknown = sum(
                    1 for s in statuses if isinstance(s, Mapping) and str(s.get("status") or "") in {"unknown", ""}
                )

            gate_summary_parts: list[str] = []
            if failed:
                gate_summary_parts.append(f"{failed}×**FAIL**")
            if borderline:
                gate_summary_parts.append(f"{borderline}×**BORDERLINE**")
            if skipped:
                gate_summary_parts.append(f"{skipped}×**SKIP**")
            if unknown:
                gate_summary_parts.append(f"{unknown}×UNKNOWN")
            gate_summary = ", ".join(gate_summary_parts) if gate_summary_parts else "PASS"

            lines.append(
                f"| `{_markdown_table_cell(item['scenario'])}` | {item.get('success_rate', 'n/a')} | {item.get('check_count', 0)} | {gate_summary} |"
            )
        lines.append("")

        lines.append("## Scenario details")
        lines.append("")
        for item in scenarios:
            scenario = str(item.get("scenario") or "").strip()
            if not scenario:
                continue
            lines.append(f"### {_markdown_table_cell(scenario)}")
            lines.append("")
            lines.append(f"- success_rate: {item.get('success_rate', 'n/a')}")
            lines.append(f"- checks: {item.get('check_count', 0)}")
            lines.append("")

            statuses = item.get("gate_statuses")
            if isinstance(statuses, list) and statuses:
                lines.append("| gate | status | pass_rate |")
                lines.append("|---|---|---:|")
                for gate in statuses:
                    if not isinstance(gate, Mapping):
                        continue
                    gate_name = str(gate.get("check_name") or "").strip() or "<unknown>"
                    status = str(gate.get("status") or "unknown")
                    pass_rate = gate.get("pass_rate")
                    lines.append(
                        f"| `{_markdown_table_cell(gate_name)}` | {_status_badge(status)} | {pass_rate if pass_rate is not None else ''} |"
                    )
                lines.append("")

            plot_links = _scenario_plots(output_path, scenario)
            if plot_links:
                lines.append("Plots:")
                for link in plot_links:
                    lines.append(f"- [{Path(link).name}]({link})")
                lines.append("")

            lines.append("Artifacts:")
            lines.append(
                "- [Scenario checks table](./scenario_checks.csv)"
                if _has_path(output_path, "scenario_checks.csv")
                else "- Scenario checks table (missing)"
            )
            lines.append(
                "- [Seed-level rows](./seed_level_rows.csv)"
                if _has_path(output_path, "seed_level_rows.csv")
                else "- Seed-level rows (missing)"
            )
            lines.append(
                "- [Trace artifacts directory](./traces/)"
                if _has_path(output_path, "traces/")
                else "- Trace artifacts directory (missing)"
            )
            lines.append("")
    else:
        lines.extend(["_No per-scenario summaries were available._", ""])

    lines.extend(["## Claims", ""])
    if claim_anchor and _has_path(output_path, "report.md"):
        lines.append(
            f"_Claim section in report: [Claim Test Results](./report.md#{claim_anchor})_\n"
        )

    if claims:
        lines.append("| claim | status | passed | evidence |")
        lines.append("|---|---|---|---|")
        for item in claims:
            passed = item.get("passed")
            passed_str = "" if passed is None else ("pass" if passed else "fail")
            comparisons = item.get("comparisons") if isinstance(item.get("comparisons"), list) else []
            comparison_hint = ""
            if comparisons:
                comparison_hint = " (" + ", ".join((f"`{_markdown_table_cell(name)}`" for name in comparisons[:3])) + ("…" if len(comparisons) > 3 else "") + ")"

            links = item.get("links") if isinstance(item.get("links"), Mapping) else {}

            report_link = _maybe_link("report", str(links.get("report_md") or ""))
            claim_tables_link = _maybe_link(
                "claim tables", str(links.get("claim_test_tables_json") or "")
            )
            aggregate_link = _maybe_link(
                "aggregate tables",
                str(links.get("aggregate_benchmark_tables_json") or ""),
            )
            effect_size_link = _maybe_link(
                "effect sizes", str(links.get("effect_size_tables_json") or "")
            )
            seed_link = _maybe_link(
                "seed rows", str(links.get("seed_level_rows_csv") or "")
            )
            trace_link = _maybe_link("traces", str(links.get("traces_dir") or ""))

            evidence_links = "; ".join(
                [
                    link
                    for link in (
                        report_link,
                        claim_tables_link,
                        aggregate_link,
                        effect_size_link,
                        seed_link,
                        trace_link,
                    )
                    if link
                ]
            )
            lines.append(
                f"| `{_markdown_table_cell(item['claim'])}`{comparison_hint} | {item.get('status','')} | {passed_str} | {evidence_links} |"
            )
        lines.append("")
    else:
        lines.extend(["_No claim entries were available._", ""])

    index_path = output_path / "INDEX.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(index_path)
