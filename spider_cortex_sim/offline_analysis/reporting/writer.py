from __future__ import annotations
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from ..constants import LADDER_ACTIVE_RUNGS, LADDER_ADJACENT_COMPARISONS, LADDER_PROTOCOL_NAMES, LADDER_RUNG_DESCRIPTIONS, LADDER_RUNG_MAPPING, SHAPING_DEPENDENCE_WARNING_THRESHOLD
from ..extractors import _noise_robustness_marginal, _noise_robustness_rate, _normalize_float_map, _representation_interpretation, build_primary_benchmark, build_reflex_dependence_indicators, extract_ablations, extract_comparisons, extract_noise_robustness, extract_predator_type_specialization, extract_reflex_frequency, extract_representation_specialization, extract_scenario_success, extract_shaping_audit, extract_training_eval_series
from ..ingestion import normalize_behavior_rows
from ..plots import build_capacity_comparison_plot
from ..renderers import render_bar_chart, render_line_chart, render_matrix_heatmap, render_placeholder_svg
from ..tables import build_aggregate_benchmark_tables, build_claim_test_tables, build_diagnostics, build_effect_size_tables, build_reward_component_rows, build_scenario_checks_rows
from ..utils import _coerce_float, _coerce_optional_float, _format_optional_metric, _mapping_or_empty
from ..writers import _markdown_table, _markdown_value, _table_rows, _write_csv, _write_svg
from ..report_data import build_report_data

def _section_artifact_state(section: Mapping[str, object]) -> str:
    """
    Determine the availability state string for a section payload.
    
    Parameters:
        section (Mapping[str, object]): A section payload expected to contain keys like
            'available' (truthy when artifact is present) and optionally 'artifact_state'
            (explicit state string when not available).
    
    Returns:
        str: 'available' when section['available'] is truthy; otherwise the value of
        section['artifact_state'] converted to a string if present, or 'expected_but_missing'.
    """
    if bool(section.get('available')):
        return 'available'
    return str(section.get('artifact_state') or 'expected_but_missing')

def _unavailable_section_message(section: Mapping[str, object], *, expected_but_missing: str, not_in_artifact: str) -> str:
    """
    Return the appropriate unavailable-section message based on the section payload's artifact state.
    
    Parameters:
        section (Mapping[str, object]): Section metadata that may contain availability or artifact-state indicators.
        expected_but_missing (str): Message to use when the section is expected but missing.
        not_in_artifact (str): Message to use when the section is not included in this artifact.
    
    Returns:
        str: `not_in_artifact` if the section's artifact state is "not_in_this_artifact", otherwise `expected_but_missing`.
    """
    return not_in_artifact if _section_artifact_state(section) == 'not_in_this_artifact' else expected_but_missing

def _markdown_table_or_message(rows: Sequence[Sequence[object]], columns: Sequence[str], *, section: Mapping[str, object], expected_but_missing: str, not_in_artifact: str) -> str:
    """
    Emit a Markdown table for the provided rows and columns, or an availability message when no rows are present.
    
    Parameters:
        rows: Sequence of row tuples/lists to include in the table; if empty, an availability message is returned.
        columns: Column header names for the Markdown table.
        section: Payload metadata used to determine which unavailable message to return (e.g., artifact availability/state).
        expected_but_missing: Message text to use when the section is expected but the artifact is missing.
        not_in_artifact: Message text to use when the section is explicitly not included in this artifact.
    
    Returns:
        A Markdown-formatted table string when `rows` is non-empty, otherwise an availability message selected based on `section`.
    """
    if rows:
        return _markdown_table(rows, columns)
    return _unavailable_section_message(section, expected_but_missing=expected_but_missing, not_in_artifact=not_in_artifact)

def _report_input_lines(inputs: Mapping[str, object], combined_inputs: Mapping[str, object]) -> list[str]:
    lines = [
        f"- `summary`: {inputs.get('summary_path', '<missing>')}",
        f"- `trace`: {inputs.get('trace_path', '<missing>')}",
        f"- `behavior_csv`: {inputs.get('behavior_csv_path', '<missing>')}",
    ]
    if combined_inputs:
        lines.extend(
            [
                f"- `ablation_summary`: {combined_inputs.get('ablation_summary_path')}",
                f"- `ablation_behavior_csv`: {combined_inputs.get('ablation_behavior_csv_path')}",
                f"- `profile_summary`: {combined_inputs.get('profile_summary_path')}",
                f"- `profile_behavior_csv`: {combined_inputs.get('profile_behavior_csv_path')}",
                f"- `capacity_summary`: {combined_inputs.get('capacity_summary_path')}",
                f"- `capacity_behavior_csv`: {combined_inputs.get('capacity_behavior_csv_path')}",
                f"- `module_local_report`: {combined_inputs.get('module_local_report_path')}",
                f"- `distillation_summary`: {combined_inputs.get('distillation_summary_path')}",
            ]
        )
    return lines


def _availability_summary_schema() -> list[tuple[str, str, str]]:
    """Schema for the report overview summary.

    Returns a list of (section_key, label, notes) tuples.

    Notes:
        This is currently used only to define the rows that should appear in the
        overview block. Availability evaluation and linking is handled separately.
    """
    return [
        ('inputs', 'Inputs', 'Input artifact paths used for this report.'),
        ('diagnostics', 'Diagnostics', 'Basic run metadata extracted from the report payload.'),
        ('variant_metadata', 'Variant assumptions', 'Confound-control metadata and access notes.'),
        ('architecture_capacity', 'Architecture Capacity', 'Model parameter counts and capacity comparison tables.'),
        ('capacity_sweep', 'Capacity Sweep', 'Sweep results and interpretation table (if present).'),
        ('primary_benchmark', 'Primary Benchmark', 'Single-run benchmark-of-record metric row (if present).'),
        ('aggregate_benchmark_tables', 'Benchmark-of-Record Summary', 'Aggregate benchmark tables with uncertainty (if present).'),
        ('claim_test_tables', 'Claim Test Results with Uncertainty', 'Claim tests + uncertainty (if present).'),
        ('effect_size_tables', 'Effect Sizes Against Baselines', 'Effect sizes against baselines (if present).'),
        ('ladder_comparison', 'Architectural Ladder Comparison', 'Pairwise ladder comparisons and rung descriptions.'),
        ('ladder_profile_comparison', 'Cross-Profile Ladder Comparison', 'Cross-profile ladder summary + shaping gaps.'),
        ('unified_ladder_report', 'Unified Architectural Ladder Report', 'Unified ladder analysis with detailed subsections.'),
        ('shaping_program', 'Shaping Minimization Program', 'Reward shaping audit tables and interpretation.'),
        ('scenario_success', 'Scenario Success', 'Per-scenario success rates table.'),
        ('ablations', 'Ablations', 'Ablation benchmark summary and predator-type comparisons.'),
        ('credit_analysis', 'Credit Assignment Analysis', 'Credit assignment strategy analysis and interpretation.'),
        ('predator_type_specialization', 'Predator Type Specialization', 'Specialization summary and tables.'),
        ('representation_specialization', 'Representation Specialization', 'Representation divergence and per-module scores.'),
        ('reflex_frequency', 'Reflex Frequency', 'Reflex event summary table and plot.'),
        ('noise_robustness', 'Noise Robustness', 'Robustness matrix table and plot.'),
        ('module_local_sufficiency', 'Module-Local Sufficiency', 'Module-local sufficiency analysis (if present).'),
        ('distillation_analysis', 'Distillation Analysis', 'Distillation analysis summary and table (if present).'),
        ('limitations', 'Limitations', 'Freeform limitations list (if present).'),
        ('generated_files', 'Generated Files', 'List of artifact files written alongside the report.'),
    ]


def _availability_status_for_key(section_key: str, report: Mapping[str, object]) -> str:
    """Return a compact availability status for a known overview section key.

    This keeps logic local and intentionally shallow: it mirrors the existing
    conditional rendering in `write_report` (e.g., `available` flags and whether
    computed rows exist).
    """
    if section_key in {'inputs', 'generated_files'}:
        return 'available'
    if section_key == 'diagnostics':
        diagnostics = report.get('diagnostics', [])
        if isinstance(diagnostics, list) and diagnostics:
            return 'available'
        return 'expected_but_missing'
    if section_key == 'variant_metadata':
        variant_metadata = report.get('variant_metadata')
        if isinstance(variant_metadata, Mapping):
            return _section_artifact_state(variant_metadata)
        return 'not_in_this_artifact'

    payload_map = {
        'architecture_capacity': report.get('model_capacity'),
        'capacity_sweep': report.get('capacity_sweeps'),
        'primary_benchmark': report.get('primary_benchmark'),
        'aggregate_benchmark_tables': report.get('aggregate_benchmark_tables'),
        'claim_test_tables': report.get('claim_test_tables'),
        'effect_size_tables': report.get('effect_size_tables'),
        'ladder_comparison': report.get('ladder_comparison'),
        'ladder_profile_comparison': report.get('ladder_profile_comparison'),
        'unified_ladder_report': report.get('unified_ladder_report'),
        'shaping_program': report.get('shaping_program'),
        'scenario_success': report.get('scenario_success'),
        'ablations': report.get('ablations'),
        'credit_analysis': report.get('credit_analysis'),
        'predator_type_specialization': report.get('predator_type_specialization'),
        'representation_specialization': report.get('representation_specialization'),
        'reflex_frequency': report.get('reflex_frequency'),
        'noise_robustness': report.get('noise_robustness'),
        'module_local_sufficiency': report.get('module_local_sufficiency'),
        'distillation_analysis': report.get('distillation_analysis'),
        'limitations': report.get('limitations'),
    }
    payload = payload_map.get(section_key)
    if isinstance(payload, Mapping):
        return _section_artifact_state(payload)

    if section_key == 'limitations':
        limitations = report.get('limitations', [])
        if isinstance(limitations, list) and limitations:
            return 'available'
        return 'not_in_this_artifact'

    return 'not_in_this_artifact'


def _slugify_github_anchor(text: str) -> str:
    """Slugify heading text to match GitHub-flavored Markdown anchor IDs.

    GitHub's algorithm is not strictly documented, but for our headings we need:
    - lowercase
    - collapse whitespace to '-'
    - drop punctuation/symbols
    - keep hyphens
    """
    slug = text.strip().lower()
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _assign_github_anchors(headings: Sequence[tuple[int, str]]) -> list[tuple[int, str, str]]:
    """Assign GitHub-style unique anchors for headings in document order.

    GitHub disambiguates duplicate heading slugs by appending `-n` starting at 1
    (e.g., `heading`, `heading-1`, `heading-2`, ...).
    """
    anchor_counts: dict[str, int] = {}
    assigned: list[tuple[int, str, str]] = []
    for level, text in headings:
        base = _slugify_github_anchor(text)
        count = anchor_counts.get(base, 0)
        anchor_counts[base] = count + 1
        anchor = base if count == 0 else f"{base}-{count}"
        assigned.append((level, text, anchor))
    return assigned


def _heading_text(line: str) -> tuple[int, str] | None:
    stripped = line.lstrip()
    if not stripped.startswith('#'):
        return None
    hashes = len(stripped) - len(stripped.lstrip('#'))
    if hashes <= 0:
        return None
    if len(stripped) <= hashes or stripped[hashes] != ' ':
        return None
    return hashes, stripped[hashes + 1 :].strip()


def _extract_toc_from_markdown_lines(lines: Sequence[str]) -> list[tuple[int, str]]:
    headings: list[tuple[int, str]] = []
    for line in lines:
        parsed = _heading_text(line)
        if not parsed:
            continue
        level, text = parsed
        if level == 1:
            continue
        headings.append((level, text))
    return headings


def assign_github_anchors(headings: Sequence[tuple[int, str]]) -> list[tuple[int, str, str]]:
    return _assign_github_anchors(headings)


def extract_toc_from_markdown_lines(lines: Sequence[str]) -> list[tuple[int, str]]:
    return _extract_toc_from_markdown_lines(lines)


def _overview_section_anchor_map(headings: Sequence[tuple[int, str, str]]) -> dict[str, str]:
    """Return a mapping from overview section keys to anchors.

    This is intentionally a small, explicit mapping so it stays aligned with
    concrete headings emitted by `write_report`.
    """

    by_text = {text: anchor for _, text, anchor in headings}

    mapping = {
        'inputs': by_text.get('Inputs'),
        'diagnostics': by_text.get('Diagnostics'),
        'variant_metadata': by_text.get('Variant assumptions (confound-control metadata)'),
        'architecture_capacity': by_text.get('Architecture Capacity'),
        'capacity_sweep': by_text.get('Capacity Sweep'),
        'primary_benchmark': by_text.get('Primary Benchmark'),
        'aggregate_benchmark_tables': by_text.get('Benchmark-of-Record Summary'),
        'claim_test_tables': by_text.get('Claim Test Results with Uncertainty'),
        'effect_size_tables': by_text.get('Effect Sizes Against Baselines'),
        'ladder_comparison': by_text.get('Architectural Ladder Comparison'),
        'ladder_profile_comparison': by_text.get('Cross-Profile Ladder Comparison'),
        'unified_ladder_report': by_text.get('Unified Architectural Ladder Report'),
        'shaping_program': by_text.get('Shaping Minimization Program'),
        'scenario_success': by_text.get('Scenario Success'),
        'ablations': by_text.get('Ablations'),
        'credit_analysis': by_text.get('Credit Assignment Analysis'),
        'predator_type_specialization': by_text.get('Predator Type Specialization'),
        'representation_specialization': by_text.get('Representation Specialization'),
        'reflex_frequency': by_text.get('Reflex Frequency'),
        'noise_robustness': by_text.get('Noise Robustness'),
        'module_local_sufficiency': by_text.get('Module-Local Sufficiency'),
        'distillation_analysis': by_text.get('Distillation Analysis'),
        'limitations': by_text.get('Limitations'),
        'generated_files': by_text.get('Generated Files'),
    }

    return {key: anchor for key, anchor in mapping.items() if anchor}


def _render_toc_block(headings: Sequence[tuple[int, str, str]]) -> list[str]:
    toc_lines = ['## Table of contents', '']
    for level, text, anchor in headings:
        indent = '  ' * max(level - 2, 0)
        toc_lines.append(f"{indent}- [{text}](#{anchor})")
    toc_lines.append('')
    return toc_lines


def _overview_status_label(status: str) -> str:
    if status == 'available':
        return 'available'
    if status == 'expected_but_missing':
        return 'missing'
    if status == 'not_in_this_artifact':
        return 'n/a'
    return status


def _render_overview_summary_block(report: Mapping[str, object], *, section_anchors: Mapping[str, str] | None = None) -> list[str]:
    """Render an overview summary block (purely additive)."""
    section_anchors = dict(section_anchors or {})

    rows = []
    for section_key, label, notes in _availability_summary_schema():
        status = _availability_status_for_key(section_key, report)
        display_label = label
        anchor = section_anchors.get(section_key)
        if anchor:
            display_label = f"[{label}](#{anchor})"
        rows.append((display_label, _overview_status_label(status), notes))

    return [
        '## Overview',
        '',
        _markdown_table(rows, ('Section', 'Status', 'Notes')),
        '',
    ]

def write_report(output_dir: str | Path, report: Mapping[str, object]) -> dict[str, str]:
    """
    Write filesystem artifacts (SVG, CSV, JSON, and Markdown) that represent the provided analysis report.
    
    Parameters:
        output_dir (str | Path): Target directory where artifact files will be created; created if missing.
        report (Mapping[str, object]): Report data used to populate charts, tables, and markdown. Expected keys include (but are not limited to) "training_eval", "scenario_success", "shaping_program", "ablations", "ladder_comparison", "ladder_profile_comparison", "reflex_frequency", "scenario_checks", "reward_components", "diagnostics", "primary_benchmark", "limitations", and "inputs".
    
    Returns:
        dict[str, str]: Mapping of artifact identifiers to their filesystem paths. Keys include "report_md", "report_json", "training_eval_svg", "scenario_success_svg", "scenario_checks_csv", "reward_components_csv", "ablation_comparison_svg", and "reflex_frequency_svg".
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    training_eval = report.get('training_eval', {})
    scenario_success = report.get('scenario_success', {})
    primary_benchmark = report.get('primary_benchmark', {})
    shaping_program = report.get('shaping_program', {})
    noise_robustness = report.get('noise_robustness', {})
    ablations = report.get('ablations', {})
    ladder_comparison = _mapping_or_empty(report.get('ladder_comparison'))
    ladder_profile_comparison = _mapping_or_empty(report.get('ladder_profile_comparison'))
    unified_ladder_report = _mapping_or_empty(report.get('unified_ladder_report'))
    predator_type_specialization = report.get('predator_type_specialization', {})
    representation_specialization = report.get('representation_specialization', {})
    capacity_analysis = _mapping_or_empty(report.get('capacity_analysis'))
    model_capacity = _mapping_or_empty(report.get('model_capacity'))
    capacity_sweeps = _mapping_or_empty(report.get('capacity_sweeps'))
    reflex_frequency = report.get('reflex_frequency', {})
    credit_assignment = _mapping_or_empty(report.get('credit_assignment'))
    credit_analysis = _mapping_or_empty(report.get('credit_analysis'))
    if 'variant_metadata' in report:
        raw_variant_metadata = report.get('variant_metadata')
        variant_metadata = dict(_mapping_or_empty(raw_variant_metadata))
        raw_variant_metadata_map = _mapping_or_empty(raw_variant_metadata)
        if 'artifact_state' in raw_variant_metadata_map:
            variant_metadata['artifact_state'] = raw_variant_metadata_map.get('artifact_state')
    else:
        variant_metadata = {'artifact_state': 'not_in_this_artifact'}
    aggregate_benchmark_tables = _mapping_or_empty(report.get('aggregate_benchmark_tables'))
    claim_test_tables = _mapping_or_empty(report.get('claim_test_tables'))
    effect_size_tables = _mapping_or_empty(report.get('effect_size_tables'))
    module_local_sufficiency = _mapping_or_empty(report.get('module_local_sufficiency'))
    distillation_analysis = _mapping_or_empty(report.get('distillation_analysis'))
    if not module_local_sufficiency:
        module_local_sufficiency = {'available': False, 'artifact_state': 'not_in_this_artifact'}
    if not distillation_analysis:
        distillation_analysis = {'available': False, 'artifact_state': 'not_in_this_artifact'}
    training_series = []
    if isinstance(training_eval, Mapping):
        training_series.extend(training_eval.get('training_points', []) if isinstance(training_eval.get('training_points'), list) else [])
        evaluation_points = training_eval.get('evaluation_points', []) if isinstance(training_eval.get('evaluation_points'), list) else []
    else:
        evaluation_points = []
    training_svg_path = output_path / 'training_eval.svg'
    if training_series:
        _write_svg(training_svg_path, render_line_chart('Training reward trajectory', training_series))
    elif evaluation_points:
        _write_svg(training_svg_path, render_line_chart('Evaluation reward trajectory', evaluation_points))
    else:
        _write_svg(training_svg_path, render_placeholder_svg('Training / evaluation', 'No summary reward series was available.'))
    scenario_svg_path = output_path / 'scenario_success.svg'
    scenario_items = scenario_success.get('scenarios', []) if isinstance(scenario_success, Mapping) else []
    scenario_chart_items = [{'scenario': item.get('scenario'), 'success_rate': item.get('success_rate')} for item in scenario_items if isinstance(item, Mapping)]
    _write_svg(scenario_svg_path, render_bar_chart('Scenario success rate', scenario_chart_items, label_key='scenario', value_key='success_rate'))
    robustness_svg_path = output_path / 'robustness_matrix.svg'
    _nr = _mapping_or_empty(noise_robustness)
    robustness_matrix_spec = _mapping_or_empty(_nr.get('matrix_spec'))
    _train_conds_raw = robustness_matrix_spec.get('train_conditions')
    _eval_conds_raw = robustness_matrix_spec.get('eval_conditions')
    robustness_train_conditions = list(_train_conds_raw) if isinstance(_train_conds_raw, list) else []
    robustness_eval_conditions = list(_eval_conds_raw) if isinstance(_eval_conds_raw, list) else []
    robustness_matrix = _mapping_or_empty(_nr.get('matrix'))
    robustness_train_marginals = _mapping_or_empty(_nr.get('train_marginals'))
    robustness_eval_marginals = _mapping_or_empty(_nr.get('eval_marginals'))
    _write_svg(robustness_svg_path, render_matrix_heatmap('Noise robustness matrix', train_conditions=robustness_train_conditions, eval_conditions=robustness_eval_conditions, matrix=robustness_matrix, train_marginals=robustness_train_marginals, eval_marginals=robustness_eval_marginals))
    ablation_svg_path = ''
    if isinstance(ablations, Mapping) and ablations.get('available'):
        variants = []
        variants_payload = ablations.get('variants', {})
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get('summary', {})
                variants.append({'variant': variant_name, 'score': _coerce_float(summary_payload.get('scenario_success_rate'), 0.0)})
        if variants:
            ablation_svg_path = str(output_path / 'ablation_comparison.svg')
            _write_svg(Path(ablation_svg_path), render_bar_chart('Ablation scenario success rate', variants, label_key='variant', value_key='score'))
    capacity_comparison_svg_path = ''
    capacity_sweep_rows = capacity_sweeps.get('rows', []) if isinstance(capacity_sweeps.get('rows'), list) else []
    if capacity_sweep_rows:
        capacity_comparison_plot = build_capacity_comparison_plot(capacity_sweep_rows)
        capacity_comparison_svg = str(capacity_comparison_plot.get('svg') or '')
        if capacity_comparison_plot.get('available') and capacity_comparison_svg:
            capacity_comparison_svg_path = str(output_path / 'capacity_comparison.svg')
            _write_svg(Path(capacity_comparison_svg_path), capacity_comparison_svg)
    reflex_svg_path = ''
    if isinstance(reflex_frequency, Mapping) and reflex_frequency.get('available'):
        modules = reflex_frequency.get('modules', [])
        if isinstance(modules, list) and modules:
            reflex_svg_path = str(output_path / 'reflex_frequency.svg')
            _write_svg(Path(reflex_svg_path), render_bar_chart('Reflex events per module', modules, label_key='module', value_key='reflex_events'))
    representation_svg_path = output_path / 'representation_specialization.svg'
    representation_payload = _mapping_or_empty(representation_specialization)
    representation_chart_items = [{'module': str(module_name), 'score': _coerce_float(score)} for module_name, score in sorted(_normalize_float_map(representation_payload.get('proposer_divergence')).items())]
    if representation_chart_items:
        _write_svg(representation_svg_path, render_bar_chart('Representation specialization by module', representation_chart_items, label_key='module', value_key='score'))
    else:
        _write_svg(representation_svg_path, render_placeholder_svg('Representation specialization', 'No representation-specialization divergence scores were available.'))
    scenario_checks_path = output_path / 'scenario_checks.csv'
    _write_csv(scenario_checks_path, report.get('scenario_checks', []) if isinstance(report.get('scenario_checks'), list) else [], ('scenario', 'check_name', 'pass_rate', 'mean_value', 'expected', 'description', 'source'))
    reward_components_path = output_path / 'reward_components.csv'
    _write_csv(reward_components_path, report.get('reward_components', []) if isinstance(report.get('reward_components'), list) else [], ('source', 'scope', 'component', 'value'))
    report_json_path = output_path / 'report.json'
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    diagnostics = report.get('diagnostics', [])
    inputs = _mapping_or_empty(report.get('inputs'))
    combined_inputs = _mapping_or_empty(report.get('combined_inputs'))
    diagnostic_rows = [(str(item.get('label') or ''), str(item.get('value') or '')) for item in diagnostics if isinstance(item, Mapping)]
    primary_benchmark_rows = []
    if isinstance(primary_benchmark, Mapping) and primary_benchmark.get('available'):
        primary_benchmark_rows.append((str(primary_benchmark.get('label') or ''), f"{_coerce_float(primary_benchmark.get('scenario_success_rate')):.2f}", 'n/a' if primary_benchmark.get('eval_reflex_scale') is None else f"{_coerce_float(primary_benchmark.get('eval_reflex_scale')):.2f}", str(primary_benchmark.get('reference_variant') or ''), str(primary_benchmark.get('source') or '')))
    aggregate_primary_rows = []
    aggregate_primary_table = _mapping_or_empty(aggregate_benchmark_tables.get('primary_benchmark'))
    for item in _table_rows(aggregate_primary_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_primary_rows.append((str(item.get('metric') or ''), _markdown_value(item.get('value')), _markdown_value(item.get('ci_lower')), _markdown_value(item.get('ci_upper')), str(int(_coerce_float(item.get('n_seeds'), 0.0))), str(item.get('reference_variant') or ''), str(item.get('source') or '')))
    aggregate_scenario_rows = []
    aggregate_scenario_table = _mapping_or_empty(aggregate_benchmark_tables.get('per_scenario_success_rates'))
    for item in _table_rows(aggregate_scenario_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_scenario_rows.append((str(item.get('scenario') or ''), _markdown_value(item.get('value')), _markdown_value(item.get('ci_lower')), _markdown_value(item.get('ci_upper')), str(int(_coerce_float(item.get('n_seeds'), 0.0)))))
    aggregate_learning_rows = []
    aggregate_learning_table = _mapping_or_empty(aggregate_benchmark_tables.get('learning_evidence_deltas'))
    for item in _table_rows(aggregate_learning_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_learning_rows.append((str(item.get('comparison') or ''), str(item.get('metric') or ''), _markdown_value(item.get('value')), _markdown_value(item.get('ci_lower')), _markdown_value(item.get('ci_upper')), str(int(_coerce_float(item.get('n_seeds'), 0.0)))))
    model_capacity_rows = []
    for item in model_capacity.get('networks', []) if isinstance(model_capacity.get('networks'), list) else []:
        if not isinstance(item, Mapping):
            continue
        model_capacity_rows.append((str(item.get('network') or ''), str(int(_coerce_float(item.get('parameters'), 0.0))), _markdown_value(item.get('proportion'))))
    aggregate_architecture_rows = []
    aggregate_architecture_table = _mapping_or_empty(aggregate_benchmark_tables.get('architecture_capacity'))
    for item in _table_rows(aggregate_architecture_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_architecture_rows.append((str(item.get('variant') or ''), str(item.get('architecture') or ''), str(int(_coerce_float(item.get('total_trainable'), 0.0))), str(item.get('key_components') or ''), str(item.get('capacity_status') or ''), _markdown_value(item.get('total_ratio_vs_reference'))))
    capacity_reference_variant = ''
    architecture_capacity_rows = _table_rows(aggregate_architecture_table)
    if architecture_capacity_rows:
        capacity_reference_variant = str(architecture_capacity_rows[0].get('reference_variant') or '')
    capacity_sweep_curve_rows = []
    aggregate_capacity_sweep_table = _mapping_or_empty(aggregate_benchmark_tables.get('capacity_sweep_curves'))
    for item in _table_rows(aggregate_capacity_sweep_table):
        if not isinstance(item, Mapping):
            continue
        capacity_sweep_curve_rows.append((str(item.get('variant') or ''), str(item.get('capacity_profile') or ''), _markdown_value(item.get('scale_factor')), str(int(_coerce_float(item.get('total_trainable'), 0.0))), str(int(_coerce_float(item.get('approximate_compute_cost_total'), 0.0))), _markdown_value(item.get('scenario_success_rate')), _markdown_value(item.get('capability_probe_success_rate'))))
    capacity_sweep_interpretation_rows = [(str(item.get('variant') or ''), str(item.get('status') or ''), str(item.get('interpretation') or '')) for item in (capacity_sweeps.get('interpretations', []) if isinstance(capacity_sweeps.get('interpretations'), list) else []) if isinstance(item, Mapping)]
    if capacity_sweep_curve_rows:
        capacity_sweep_summary_block = _markdown_table(capacity_sweep_curve_rows, ('variant', 'capacity_profile', 'scale_factor', 'total_trainable', 'approx_compute', 'scenario_success_rate', 'capability_probe_success_rate'))
    elif capacity_sweep_interpretation_rows:
        capacity_sweep_summary_block = '_Raw capacity-sweep curve rows were not exported in this artifact; fallback interpretations are reported below._'
    else:
        capacity_sweep_summary_block = _unavailable_section_message(capacity_sweeps, expected_but_missing='_Capacity-sweep payload was expected in this artifact but missing._', not_in_artifact='_Capacity-sweep payload was not included in this artifact._')
    claim_uncertainty_rows = []
    claim_results_table = _mapping_or_empty(claim_test_tables.get('claim_results'))
    for item in _table_rows(claim_results_table):
        if not isinstance(item, Mapping):
            continue
        claim_uncertainty_rows.append((str(item.get('claim') or ''), str(item.get('role') or ''), str(item.get('condition') or ''), str(item.get('metric') or ''), _markdown_value(item.get('value')), _markdown_value(item.get('ci_lower')), _markdown_value(item.get('ci_upper')), str(int(_coerce_float(item.get('n_seeds'), 0.0))), _markdown_value(item.get('cohens_d')), str(item.get('effect_magnitude') or '')))
    effect_size_rows = []
    effect_size_table = _mapping_or_empty(effect_size_tables.get('effect_sizes'))
    for item in _table_rows(effect_size_table):
        if not isinstance(item, Mapping):
            continue
        effect_size_rows.append((str(item.get('domain') or ''), str(item.get('baseline') or ''), str(item.get('comparison') or ''), str(item.get('metric') or ''), _markdown_value(item.get('raw_delta')), _markdown_value(item.get('delta_ci_lower')), _markdown_value(item.get('delta_ci_upper')), _markdown_value(item.get('cohens_d')), str(item.get('magnitude_label') or ''), _markdown_value(item.get('ci_lower')), _markdown_value(item.get('ci_upper')), str(int(_coerce_float(item.get('n_seeds'), 0.0)))))
    ladder_description_rows = []
    ladder_comparison_rows = []
    ladder_summary_base = 'The ladder validation asks whether gains or regressions come from adding the decision-execution pipeline itself or from later sensory/proposer modularization.'
    ladder_summary_line = ladder_summary_base
    ladder_rungs_payload = _mapping_or_empty(_mapping_or_empty(ablations).get('ladder_rungs'))
    recognized_rungs = ladder_rungs_payload.get('recognized_rungs', [])
    recognized_rung_set = {str(item) for item in recognized_rungs} if isinstance(recognized_rungs, list) else set()
    for rung in LADDER_ACTIVE_RUNGS:
        if rung not in recognized_rung_set:
            continue
        ladder_description_rows.append((rung, LADDER_PROTOCOL_NAMES.get(rung, rung), LADDER_RUNG_DESCRIPTIONS.get(rung, '')))
    ladder_effect_size_lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in _table_rows(effect_size_table):
        if not isinstance(item, Mapping):
            continue
        if str(item.get('domain') or '') != 'ladder':
            continue
        ladder_effect_size_lookup[str(item.get('baseline') or ''), str(item.get('comparison') or '')] = item
    comparison_items = ladder_comparison.get('comparisons', [])
    produced_pair_labels: set[str] = set()
    if isinstance(comparison_items, list):
        for item in comparison_items:
            if not isinstance(item, Mapping):
                continue
            produced_pair_labels.add(f"{str(item.get('baseline_rung') or '')}->{str(item.get('comparison_rung') or '')}")
            metrics_payload = _mapping_or_empty(item.get('metrics'))
            deltas_payload = _mapping_or_empty(item.get('deltas'))
            baseline_variant = str(metrics_payload.get('baseline_variant') or '')
            comparison_variant = str(metrics_payload.get('comparison_variant') or '')
            effect_row = ladder_effect_size_lookup.get((baseline_variant, comparison_variant), {})
            effect_size = _coerce_optional_float(effect_row.get('cohens_d'))
            effect_magnitude = str(effect_row.get('magnitude_label') or '')
            scenario_delta = _coerce_float(deltas_payload.get('scenario_success_rate_delta'))
            interpretation = f"{('positive' if scenario_delta >= 0.0 else 'negative')} {effect_magnitude or 'unlabeled'} effect" if effect_size is not None else 'Effect size unavailable'
            ladder_comparison_rows.append((str(item.get('baseline_rung') or ''), str(item.get('comparison_rung') or ''), f'{scenario_delta:.2f}', f'{effect_size:.2f}' if effect_size is not None else 'n/a', interpretation))
    present_variants = {str(variant_name) for variant_name in _mapping_or_empty(ablations.get('variants')).keys()}
    required_variants = {variant_name for variant_name, rung in LADDER_RUNG_MAPPING.items() if rung in set(LADDER_ACTIVE_RUNGS)}
    missing_variants = sorted(required_variants - present_variants)
    expected_pair_labels = {f'{baseline_rung}->{comparison_rung}' for baseline_rung, comparison_rung in LADDER_ADJACENT_COMPARISONS}
    missing_pair_labels = sorted(expected_pair_labels - produced_pair_labels)
    ladder_limitations = [str(item) for item in ladder_comparison.get('limitations', []) if item]
    if missing_variants or missing_pair_labels or ladder_limitations:
        summary_parts = [ladder_summary_base]
        if missing_variants:
            summary_parts.append('Missing variants: ' + ', '.join((f'`{name}`' for name in missing_variants)) + '.')
        if missing_pair_labels:
            summary_parts.append('Missing adjacent ladder pairs: ' + ', '.join((f'`{label}`' for label in missing_pair_labels)) + '.')
        if ladder_limitations:
            summary_parts.append('Notes: ' + ' '.join(ladder_limitations))
        ladder_summary_line = ' '.join(summary_parts)
    ladder_profile_summary_rows = []
    ladder_profile_gap_rows = []
    ladder_profile_summary_line = _unavailable_section_message(ladder_profile_comparison, expected_but_missing='_Cross-profile architectural ladder comparison was expected in this artifact but missing._', not_in_artifact='_Cross-profile architectural ladder comparison was not included in this artifact._')
    ladder_profile_rows_payload = _mapping_or_empty(ladder_profile_comparison.get('classification_summary')).get('rows', [])
    if bool(ladder_profile_comparison.get('available')) and isinstance(ladder_profile_rows_payload, list) and ladder_profile_rows_payload:
        ladder_profile_summary_line = 'The cross-profile ladder re-runs A0-A4 under `classic`, `ecological`, and `austere` while preserving no-reflex `scenario_success_rate` as the benchmark surface. Classifications use the four-term taxonomy: `shaping_dependent`, `austere_survivor`, `fails_with_shaping`, and `mixed_or_borderline`.'
        for item in ladder_profile_rows_payload:
            if not isinstance(item, Mapping):
                continue
            ladder_profile_summary_rows.append((str(item.get('protocol_name') or ''), str(item.get('classification') or ''), _markdown_value(item.get('classic_success_rate')), _markdown_value(item.get('ecological_success_rate')), _markdown_value(item.get('austere_success_rate')), _markdown_value(item.get('austere_competence_gap')), str(item.get('reason') or '')))
            ladder_profile_gap_rows.append((str(item.get('protocol_name') or ''), _markdown_value(item.get('classic_minus_austere')), _markdown_value(item.get('classic_gap_ci_lower')), _markdown_value(item.get('classic_gap_ci_upper')), _markdown_value(item.get('classic_gap_effect_size')), str(int(_coerce_float(item.get('classic_gap_n_seeds'), 0.0))), _markdown_value(item.get('ecological_minus_austere')), _markdown_value(item.get('ecological_gap_ci_lower')), _markdown_value(item.get('ecological_gap_ci_upper')), _markdown_value(item.get('ecological_gap_effect_size')), str(int(_coerce_float(item.get('ecological_gap_n_seeds'), 0.0)))))
    ladder_report_summary_line = _unavailable_section_message(unified_ladder_report, expected_but_missing='_Unified architectural ladder report was expected in this artifact but missing._', not_in_artifact='_Unified architectural ladder report was not included in this artifact._')
    ladder_report_rows = []
    ladder_report_adjacent_rows = []
    ladder_report_capacity_rows = []
    ladder_report_credit_rows = []
    ladder_report_shaping_rows = []
    ladder_report_no_reflex_rows = []
    ladder_report_capability_rows = []
    ladder_report_conclusion_rows = []
    ladder_report_missing_rows = []
    ladder_report_limitation_lines = ['- None.']

    comparison_support_line = _unavailable_section_message(
        unified_ladder_report,
        expected_but_missing='_Comparison-support payload was expected in this artifact but missing._',
        not_in_artifact='_Comparison-support payload was not included in this artifact._',
    )
    comparison_support_status = 'unknown'
    comparison_support_diagnostic_lines = ['- None.']

    comparison_matrix_markdown_lines = [
        _unavailable_section_message(
            unified_ladder_report,
            expected_but_missing='_Comparison-support matrix payload was expected in this artifact but missing._',
            not_in_artifact='_Comparison-support matrix payload was not included in this artifact._',
        )
    ]

    variant_metadata_line = _unavailable_section_message(
        variant_metadata,
        expected_but_missing='_Variant metadata payload was expected in this artifact but missing._',
        not_in_artifact='_Variant metadata payload was not included in this artifact._',
    )
    variant_capacity_rows = []
    variant_interface_lines = ['- None.']
    variant_credit_lines = ['- None.']
    variant_metadata_has_payload = any(
        key != 'artifact_state' for key in variant_metadata
    )
    if variant_metadata_has_payload:
        schema_version = str(variant_metadata.get('schema_version') or '')
        capacity = _mapping_or_empty(variant_metadata.get('capacity'))
        interface_access = _mapping_or_empty(variant_metadata.get('interface_access'))
        credit_assignment_meta = _mapping_or_empty(variant_metadata.get('credit_assignment'))

        def _metadata_value(value: object) -> object:
            payload = _mapping_or_empty(value)
            if 'value' in payload:
                return payload.get('value')
            return value

        def _metadata_text(value: object) -> str:
            unwrapped = _metadata_value(value)
            if isinstance(unwrapped, list):
                return ', '.join(str(x) for x in unwrapped) if unwrapped else '(missing)'
            if isinstance(unwrapped, Mapping):
                return json.dumps(unwrapped, sort_keys=True)
            text = str(unwrapped)
            return text if text else '(missing)'

        variant_metadata_line = f"Schema version: `{schema_version or 'unknown'}`."
        total_trainable = _metadata_value(
            capacity.get('parameter_count_total', capacity.get('total_trainable'))
        )
        total_trainable_numeric = _coerce_optional_float(total_trainable)
        compute_cost = capacity.get('approximate_compute_cost_total')
        if compute_cost is None:
            approximate_compute_cost = _mapping_or_empty(capacity.get('approximate_compute_cost'))
            compute_cost = approximate_compute_cost.get('total') if approximate_compute_cost else capacity.get('approximate_compute_cost')
        variant_capacity_rows = [
            (
                schema_version or 'unknown',
                str(int(total_trainable_numeric)) if total_trainable_numeric is not None else 'unknown',
                _markdown_value(_metadata_value(compute_cost)),
                str(capacity.get('source') or ''),
                str(capacity.get('status') or ''),
            )
        ]
        observation_space = interface_access.get('observation_space', interface_access.get('observations'))
        action_space = interface_access.get('action_space', interface_access.get('actions'))
        module_inputs = interface_access.get('module_inputs', interface_access.get('internal_signals'))
        module_outputs = interface_access.get('module_outputs')
        privileged = interface_access.get('privileged')
        if isinstance(privileged, list) and privileged:
            privileged_text = ', '.join(str(x) for x in privileged)
        elif privileged in (None, '', []):
            privileged_text = '(none)'
        else:
            privileged_text = _metadata_text(privileged)
        variant_interface_lines = [
            f"- status: `{interface_access.get('status') or 'unknown'}`",
            f"- architecture_signature: `{interface_access.get('architecture_signature') or ''}`" if interface_access.get('architecture_signature') else '- architecture_signature: (missing)',
            f"- observation_space: {_metadata_text(observation_space)}",
            f"- action_space: {_metadata_text(action_space)}",
            f"- module_inputs: {_metadata_text(module_inputs)}",
            f"- module_outputs: {_metadata_text(module_outputs)}",
            f"- privileged: {privileged_text}",
        ]
        mechanism = credit_assignment_meta.get('mechanism')
        if not mechanism:
            legacy_mechanism = {}
            if credit_assignment_meta.get('update_rule'):
                legacy_mechanism['update_rule'] = credit_assignment_meta.get('update_rule')
            distillation = credit_assignment_meta.get('distillation_status') or credit_assignment_meta.get('distillation')
            if distillation:
                legacy_mechanism['distillation'] = distillation
            mechanism = legacy_mechanism
        notes = (
            variant_metadata.get('credit_assignment_notes')
            or credit_assignment_meta.get('credit_assignment_notes')
            or credit_assignment_meta.get('notes')
        )
        credit_notes = (
            '; '.join(str(x) for x in notes)
            if isinstance(notes, list) and notes
            else str(notes or '')
        )
        credit_items = [
            f"- status: `{credit_assignment_meta.get('status') or 'unknown'}`",
            f"- mechanism: {_metadata_text(mechanism)}" if mechanism else '- mechanism: (missing)',
        ]
        if credit_notes:
            credit_items.append(f"- notes: {credit_notes}")
        variant_credit_lines = credit_items

    if unified_ladder_report and bool(unified_ladder_report.get('available')):
        conclusion = str(unified_ladder_report.get('conclusion') or 'modularity inconclusive')
        rationale = str(unified_ladder_report.get('conclusion_rationale') or '')
        ladder_report_summary_line = f'Conclusion: `{conclusion}`. {rationale}'.strip()

        comparison_support = _mapping_or_empty(unified_ladder_report.get('comparison_support'))
        if comparison_support and bool(comparison_support.get('available')):
            comparison_support_status = str(comparison_support.get('status') or 'unknown')
            baseline_variant = str(comparison_support.get('baseline_variant') or '')
            comparison_variant = str(comparison_support.get('comparison_variant') or '')
            comparison_support_line = f"Comparison: `{baseline_variant}` → `{comparison_variant}`. Status: `{comparison_support_status}`."
            comparison_diagnostics_raw = comparison_support.get('diagnostics')
            if isinstance(comparison_diagnostics_raw, list) and comparison_diagnostics_raw:
                comparison_support_diagnostic_lines = [f"- {item!s}" for item in comparison_diagnostics_raw if item]
                if not comparison_support_diagnostic_lines:
                    comparison_support_diagnostic_lines = ['- No diagnostics reported.']
            else:
                comparison_support_diagnostic_lines = ['- No diagnostics reported.']

        comparison_matrix = _mapping_or_empty(
            unified_ladder_report.get('comparison_support_matrix')
        )
        if comparison_matrix and bool(comparison_matrix.get('available')):
            comparison_matrix_markdown_lines = [
                _markdown_table_or_message(
                    _table_rows(_mapping_or_empty(comparison_matrix.get('cells'))),
                    tuple(
                        str(item)
                        for item in (
                            _mapping_or_empty(comparison_matrix.get('cells')).get('columns')
                            or []
                        )
                    ),
                    section=comparison_matrix,
                    expected_but_missing='_Comparison-support matrix cells were expected in this artifact but missing._',
                    not_in_artifact='_Comparison-support matrix cells were not included in this artifact._',
                )
            ]
            matrix_limitations = comparison_matrix.get('limitations')
            if isinstance(matrix_limitations, list) and matrix_limitations:
                comparison_matrix_markdown_lines.extend(['', 'Limitations:', ''])
                comparison_matrix_markdown_lines.extend(
                    [f"- {item!s}" for item in matrix_limitations if item]
                    or ['- None.']
                )

        ladder_report_tables = _mapping_or_empty(unified_ladder_report.get('tables'))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('primary_rung_table'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_rows.append((str(item.get('rung') or ''), str(item.get('protocol_name') or ''), str(item.get('variant') or ''), str(int(_coerce_float(item.get('total_parameters'), 0.0))), _markdown_value(item.get('scenario_success_rate')), _markdown_value(item.get('ci_lower')), _markdown_value(item.get('ci_upper')), _markdown_value(item.get('effect_size_vs_a0')), str(item.get('effect_magnitude') or ''), str(int(_coerce_float(item.get('n_seeds'), 0.0)))))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('adjacent_comparison_table'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_adjacent_rows.append((str(item.get('baseline_rung') or ''), str(item.get('comparison_rung') or ''), _markdown_value(item.get('delta')), _markdown_value(item.get('ci_lower')), _markdown_value(item.get('ci_upper')), _markdown_value(item.get('cohens_d')), str(item.get('effect_magnitude') or ''), str(item.get('interpretation') or '')))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('capacity_summary_table'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_capacity_rows.append(('yes' if bool(item.get('capacity_matched')) else 'no', _markdown_value(item.get('ratio')), str(item.get('largest_variant') or ''), str(item.get('smallest_variant') or '')))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('credit_assignment_summary'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_credit_rows.append((str(item.get('rung') or ''), _markdown_value(item.get('local_only_delta_vs_broadcast')), _markdown_value(item.get('counterfactual_delta_vs_broadcast')), str(item.get('strategies_present') or '')))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('reward_shaping_sensitivity_summary'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_shaping_rows.append((str(item.get('rung') or ''), _markdown_value(item.get('classic_minus_austere')), _markdown_value(item.get('classic_effect_size')), _markdown_value(item.get('ecological_minus_austere')), _markdown_value(item.get('ecological_effect_size'))))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('no_reflex_competence_summary'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_no_reflex_rows.append((str(item.get('rung') or ''), 'yes' if bool(item.get('no_reflex_evaluated')) else 'no', _markdown_value(item.get('eval_reflex_scale')), _markdown_value(item.get('scenario_success_rate'))))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('capability_probe_boundaries'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_capability_rows.append((str(item.get('scenario') or ''), str(item.get('requirement_level') or ''), _markdown_value(item.get('first_competent_rung')), _markdown_value(item.get('highest_competent_rung')), str(item.get('rationale') or '')))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('conclusion_table'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_conclusion_rows.append((str(item.get('conclusion') or ''), str(item.get('rationale') or ''), _markdown_value(item.get('confidence_level')), str(item.get('confounding_factors') or '')))
        for item in _table_rows(_mapping_or_empty(ladder_report_tables.get('missing_experiments_table'))):
            if not isinstance(item, Mapping):
                continue
            ladder_report_missing_rows.append((str(item.get('experiment') or ''), str(item.get('description') or ''), str(item.get('priority') or '')))
        limitations = unified_ladder_report.get('limitations', [])
        if isinstance(limitations, list) and limitations:
            ladder_report_limitation_lines = [f'- {str(item)}' for item in limitations if item]
            if not ladder_report_limitation_lines:
                ladder_report_limitation_lines = ['- No limitations reported.']
    shaping_gap_rows = []
    shaping_component_rows = []
    shaping_profile_rows = []
    shaping_disposition_rows = []
    shaping_survival_rows = []
    shaping_summary_line = '_No shaping minimization audit was available._'
    shaping_removed_gap_line = '_No removed disposition weight gap was available._'
    shaping_warning_line = ''
    shaping_program = _mapping_or_empty(shaping_program)
    if shaping_program and bool(shaping_program.get('available')):
        dense_profile = str(shaping_program.get('dense_profile') or 'classic')
        minimal_profile = str(shaping_program.get('minimal_profile') or 'austere')
        gap_metrics = _mapping_or_empty(shaping_program.get('gap_metrics'))
        flags = _mapping_or_empty(shaping_program.get('interpretive_flags'))
        thresholds = _mapping_or_empty(shaping_program.get('thresholds'))
        if gap_metrics and bool(flags.get('gap_available')):
            scenario_delta = _coerce_float(gap_metrics.get('scenario_success_rate_delta'))
            episode_delta = _coerce_float(gap_metrics.get('episode_success_rate_delta'))
            reward_delta = _coerce_float(gap_metrics.get('mean_reward_delta'))
            shaping_gap_rows.append((f'{dense_profile} - {minimal_profile}', f'{scenario_delta:.2f}', f'{episode_delta:.2f}', f'{reward_delta:.2f}', str(shaping_program.get('interpretation') or '')))
        if bool(flags.get('shaping_dependent')):
            shaping_warning_line = f"> WARNING: High shaping dependence detected. Dense-profile scenario success exceeds the minimal-shaping warning threshold of {_coerce_float(thresholds.get('shaping_dependence'), SHAPING_DEPENDENCE_WARNING_THRESHOLD):.2f}."
        profile_weight_breakdown = _mapping_or_empty(shaping_program.get('profile_weight_breakdown'))
        for profile_name, weights in sorted(profile_weight_breakdown.items()):
            if not isinstance(weights, Mapping):
                continue
            for disposition, weight in sorted(weights.items()):
                shaping_profile_rows.append((str(profile_name), str(disposition), f'{_coerce_float(weight):.2f}'))
        disposition_summary = _mapping_or_empty(shaping_program.get('disposition_summary'))
        for disposition, payload in sorted(disposition_summary.items()):
            if not isinstance(payload, Mapping):
                continue
            components = payload.get('components', [])
            if isinstance(components, list):
                component_names = ', '.join((str(component) for component in components))
                default_count = len(components)
            else:
                component_names = ''
                default_count = 0
            shaping_disposition_rows.append((str(disposition), str(int(_coerce_float(payload.get('component_count'), default_count))), component_names))
        shaping_removed_gap_line = f"Removed disposition weight gap ({dense_profile} - {minimal_profile}): {_coerce_float(shaping_program.get('removed_weight_gap')):.2f}."
        components = shaping_program.get('component_classification', [])
        if isinstance(components, list):
            shaping_component_rows = [(str(item.get('component') or ''), str(item.get('category') or ''), str(item.get('risk') or ''), str(item.get('disposition') or ''), str(item.get('rationale') or '')) for item in components if isinstance(item, Mapping)]
        behavior_survival = _mapping_or_empty(shaping_program.get('behavior_survival'))
        survival_scenarios = behavior_survival.get('scenarios', [])
        if isinstance(survival_scenarios, list):
            shaping_survival_rows = [(str(item.get('scenario') or ''), f"{_coerce_float(item.get('austere_success_rate')):.2f}", 'yes' if bool(item.get('survives')) else 'no', str(int(_coerce_float(item.get('episodes'), 0.0)))) for item in survival_scenarios if isinstance(item, Mapping)]
        if not bool(behavior_survival.get('available')):
            shaping_summary_line = '_No shaping survival data available._'
        else:
            shaping_summary_line = f"{int(_coerce_float(behavior_survival.get('surviving_scenario_count'), 0.0))}/{int(_coerce_float(behavior_survival.get('scenario_count'), 0.0))} scenarios survive minimal shaping ({_coerce_float(behavior_survival.get('survival_rate')):.2f})."
    scenario_rows = [(str(item.get('scenario') or ''), f"{_coerce_float(item.get('success_rate')):.2f}", str(len(item.get('checks', {})) if isinstance(item.get('checks'), Mapping) else 0)) for item in scenario_items if isinstance(item, Mapping)]
    ablation_rows = []
    ablation_predator_type_rows = []
    if isinstance(ablations, Mapping):
        variants_payload = ablations.get('variants', {})
        architecture_capacity_by_variant = {str(item.get('variant') or ''): item for item in architecture_capacity_rows if isinstance(item, Mapping)}
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get('summary', {})
                config_payload = payload.get('config', {})
                capacity_payload = _mapping_or_empty(architecture_capacity_by_variant.get(str(variant_name)))
                if not isinstance(summary_payload, Mapping):
                    summary_payload = {}
                if not isinstance(config_payload, Mapping):
                    config_payload = {}
                ablation_rows.append((str(variant_name), str(config_payload.get('architecture') or ''), f"{_coerce_float(summary_payload.get('eval_reflex_scale'), _coerce_float(payload.get('eval_reflex_scale'), 0.0)):.2f}", f"{_coerce_float(summary_payload.get('scenario_success_rate')):.2f}", str(capacity_payload.get('capacity_status') or '')))
        predator_type_comparisons = ablations.get('predator_type_comparisons', {})
        if isinstance(predator_type_comparisons, Mapping):
            comparisons = predator_type_comparisons.get('comparisons', {})
            if isinstance(comparisons, Mapping):
                for variant_name, payload in sorted(comparisons.items()):
                    if not isinstance(payload, Mapping):
                        continue
                    visual_group = _mapping_or_empty(payload.get('visual_predator_scenarios'))
                    olfactory_group = _mapping_or_empty(payload.get('olfactory_predator_scenarios'))
                    ablation_predator_type_rows.append((str(variant_name), f"{_coerce_float(visual_group.get('mean_success_rate')):.2f}", f"{_coerce_float(olfactory_group.get('mean_success_rate')):.2f}", f"{_coerce_float(payload.get('visual_minus_olfactory_success_rate')):.2f}"))
    specialization_rows = []
    specialization_summary_line = '_No predator-type specialization metrics were available._'
    specialization = _mapping_or_empty(predator_type_specialization)
    if specialization and bool(specialization.get('available')):
        predator_types = _mapping_or_empty(specialization.get('predator_types'))
        for predator_type in ('visual', 'olfactory'):
            payload = _mapping_or_empty(predator_types.get(predator_type))
            specialization_rows.append((predator_type, f"{_coerce_float(payload.get('visual_cortex_activation')):.2f}", f"{_coerce_float(payload.get('sensory_cortex_activation')):.2f}", str(payload.get('dominant_module') or '')))
        specialization_summary_line = f"Specialization score: {_coerce_float(specialization.get('specialization_score')):.2f} ({str(specialization.get('interpretation') or 'unknown')}). Type-module correlation proxy: {_coerce_float(specialization.get('type_module_correlation')):.2f}."
    differential_activation = _mapping_or_empty(_mapping_or_empty(predator_type_specialization).get('differential_activation'))
    specialization_delta_rows = []
    if differential_activation:
        specialization_delta_rows = [('visual_cortex (visual - olfactory)', f"{_coerce_float(differential_activation.get('visual_cortex_visual_minus_olfactory')):.2f}"), ('sensory_cortex (olfactory - visual)', f"{_coerce_float(differential_activation.get('sensory_cortex_olfactory_minus_visual')):.2f}")]
    representation_divergence_rows = []
    representation_action_center_rows = []
    representation_summary_line = '_No representation-specialization metrics were available._'
    representation_chart_line = 'Chart: `representation_specialization.svg`.'
    representation_section = _mapping_or_empty(representation_specialization)
    if representation_section and bool(representation_section.get('available')):
        proposer_divergence = _normalize_float_map(representation_section.get('proposer_divergence'))
        for module_name, score in sorted(proposer_divergence.items()):
            representation_divergence_rows.append((str(module_name), f'{_coerce_float(score):.2f}', _representation_interpretation(_coerce_float(score))))
        gate_differential = _normalize_float_map(representation_section.get('action_center_gate_differential'))
        contribution_differential = _normalize_float_map(representation_section.get('action_center_contribution_differential'))
        differential_modules = sorted(set(gate_differential) | set(contribution_differential))
        for module_name in differential_modules:
            representation_action_center_rows.append((str(module_name), f'{_coerce_float(gate_differential.get(module_name)):.2f}', f'{_coerce_float(contribution_differential.get(module_name)):.2f}'))
        representation_summary_line = f"Representation specialization score: {_coerce_float(representation_section.get('representation_specialization_score')):.2f} ({(representation_section.get('interpretation') or 'unknown')!s}). Source: {representation_section.get('source') or 'unknown'!s}."
    credit_strategy_rows = []
    for item in _table_rows(_mapping_or_empty(credit_analysis.get('strategy_comparison_table'))):
        if not isinstance(item, Mapping):
            continue
        credit_strategy_rows.append((str(item.get('rung') or item.get('architecture_rung') or ''), str(item.get('credit_strategy') or ''), _markdown_value(item.get('scenario_success_rate')), _markdown_value(item.get('scenario_success_delta_vs_broadcast')), str(item.get('dominant_module') or '')))
    credit_architecture_rows = []
    for item in _table_rows(_mapping_or_empty(credit_analysis.get('architecture_strategy_matrix'))):
        if not isinstance(item, Mapping):
            continue
        credit_architecture_rows.append((str(item.get('architecture_rung') or ''), str(item.get('credit_strategy') or ''), str(item.get('variant') or ''), _markdown_value(item.get('scenario_success_rate')), _markdown_value(item.get('mean_effective_module_count'))))
    credit_module_rows = []
    for item in _table_rows(_mapping_or_empty(credit_assignment.get('module_credit'))):
        if not isinstance(item, Mapping):
            continue
        credit_module_rows.append((str(item.get('rung') or ''), str(item.get('credit_strategy') or ''), str(item.get('module') or ''), _markdown_value(item.get('mean_module_credit_weight')), _markdown_value(item.get('mean_counterfactual_credit_weight')), _markdown_value(item.get('mean_module_gradient_norm')), _markdown_value(item.get('dominant_module_rate'))))
    credit_scenario_rows = []
    for item in _table_rows(_mapping_or_empty(credit_assignment.get('scenario_success'))):
        if not isinstance(item, Mapping):
            continue
        credit_scenario_rows.append((str(item.get('rung') or ''), str(item.get('credit_strategy') or ''), str(item.get('scenario') or ''), _markdown_value(item.get('success_rate'))))
    credit_interpretation_lines = ['- ' + ' / '.join((part for part in (str(item.get('scope') or ''), str(item.get('strategy') or ''), str(item.get('pattern') or '')) if part)) + f": {str(item.get('interpretation') or '')}" for item in (credit_analysis.get('findings', []) if isinstance(credit_analysis.get('findings'), list) else []) if isinstance(item, Mapping) and (item.get('pattern') or item.get('interpretation'))]
    if not credit_interpretation_lines:
        credit_interpretation_lines = [_unavailable_section_message(credit_analysis, expected_but_missing='_Credit diagnostics were expected in this artifact but missing._', not_in_artifact='_Credit diagnostics were not included in this artifact._')]
    module_local_rows = []
    module_local_summary_line = ''
    if module_local_sufficiency and bool(module_local_sufficiency.get('available')):
        partial_coverage = [str(name) for name in module_local_sufficiency.get('partial_variant_coverage', []) if name]
        summary_parts = ['Paper gate: ' + ('pass' if bool(module_local_sufficiency.get('paper_gate_pass')) else 'blocked') + '.']
        if partial_coverage:
            summary_parts.append('Partial variant coverage: ' + ', '.join((f'`{name}`' for name in partial_coverage)) + '.')
        blocked_reasons = [str(reason) for reason in module_local_sufficiency.get('blocked_reasons', []) if reason]
        if blocked_reasons:
            summary_parts.append('Blocked reasons: ' + ' '.join(blocked_reasons))
        module_local_summary_line = ' '.join(summary_parts)
        for item in module_local_sufficiency.get('rows', []):
            if not isinstance(item, Mapping):
                continue
            module_local_rows.append((str(item.get('module') or ''), str(item.get('coverage_mode') or ''), str(item.get('seeds') or ''), _markdown_value(item.get('minimal_sufficient_level')), 'yes' if bool(item.get('canonical_v4_pass')) else 'no', 'yes' if bool(item.get('partial_variant_coverage')) else 'no'))
    distillation_rows = []
    distillation_summary_line = ''
    distillation_assessment = _mapping_or_empty(distillation_analysis.get('assessment'))
    if distillation_analysis and bool(distillation_analysis.get('available')):
        answer = str(distillation_assessment.get('answer') or 'unknown')
        rationale = str(distillation_assessment.get('rationale') or '')
        distillation_summary_line = f'Assessment: `{answer}`. {rationale}'.strip()
        for item in distillation_analysis.get('rows', []):
            if not isinstance(item, Mapping):
                continue
            distillation_rows.append((str(item.get('condition') or ''), str(int(_coerce_float(item.get('episodes'), 0.0))), _markdown_value(item.get('survival_rate')), _markdown_value(item.get('mean_reward')), _markdown_value(item.get('mean_food_distance_delta'))))
    robustness_rows = []
    if robustness_train_conditions and robustness_eval_conditions:
        for train_condition in robustness_train_conditions:
            row = [train_condition]
            for eval_condition in robustness_eval_conditions:
                row.append(_format_optional_metric(_noise_robustness_rate(robustness_matrix, train_condition=train_condition, eval_condition=eval_condition)))
            row.append(_format_optional_metric(_noise_robustness_marginal(robustness_train_marginals, train_condition)))
            robustness_rows.append(tuple(row))
        marginal_row = ['mean']
        for eval_condition in robustness_eval_conditions:
            marginal_row.append(_format_optional_metric(_noise_robustness_marginal(robustness_eval_marginals, eval_condition)))
        marginal_row.append(_format_optional_metric(_coerce_optional_float(noise_robustness.get('robustness_score'))))
        robustness_rows.append(tuple(marginal_row))
    markdown_body_lines = ['## Inputs', '', *_report_input_lines(inputs, combined_inputs), '', '## Diagnostics', '', _markdown_table(diagnostic_rows, ('metric', 'value')), '', '## Variant assumptions (confound-control metadata)', '', variant_metadata_line, '', _markdown_table(variant_capacity_rows, ('schema_version', 'total_trainable', 'approximate_compute_cost', 'capacity_source', 'capacity_status')), '', 'Interface access:', '', *variant_interface_lines, '', 'Credit assignment:', '', *variant_credit_lines, '', '## Architecture Capacity', '', f"Current run ({model_capacity.get('variant') or 'unknown'} / {model_capacity.get('architecture') or 'unknown'}): {int(_coerce_float(model_capacity.get('total_trainable'), 0.0))} trainable parameters." if model_capacity.get('available') else _unavailable_section_message(model_capacity, expected_but_missing='_Trainable-parameter payload was expected in this artifact but missing._', not_in_artifact='_Trainable-parameter payload was not included in this artifact._'), '', f"Capacity status: {capacity_analysis.get('status') or 'unavailable'} (ratio {_markdown_value(capacity_analysis.get('ratio'))})." if capacity_analysis.get('available') else 'Capacity status: unavailable.', '', _markdown_table(model_capacity_rows, ('network', 'parameters', 'proportion')), '', f'Capacity comparison against the reference variant `{capacity_reference_variant}`:' if capacity_reference_variant else 'Capacity comparison:', '', _markdown_table(aggregate_architecture_rows, ('variant', 'architecture', 'total_trainable', 'key_components', 'capacity_status', 'ratio_vs_reference')), '', '## Capacity Sweep', '', capacity_sweep_summary_block, '', _markdown_table(capacity_sweep_interpretation_rows, ('variant', 'status', 'interpretation')), '', '## Primary Benchmark', '', _markdown_table(primary_benchmark_rows, ('metric', 'scenario_success_rate', 'eval_reflex_scale', 'reference_variant', 'source')), '', '## Benchmark-of-Record Summary', '', _markdown_table_or_message(aggregate_primary_rows, ('primary_metric', 'value', 'ci_lower', 'ci_upper', 'n_seeds', 'reference_variant', 'source'), section=aggregate_benchmark_tables, expected_but_missing='_Benchmark-of-record summary rows were expected in this artifact but missing._', not_in_artifact='_Benchmark-of-record summary rows were not included in this artifact._'), '', 'Per-scenario success rates with CI[^ci-method]:', '', _markdown_table_or_message(aggregate_scenario_rows, ('scenario', 'success_rate', 'ci_lower', 'ci_upper', 'n_seeds'), section=aggregate_benchmark_tables, expected_but_missing='_Per-scenario benchmark rows were expected in this artifact but missing._', not_in_artifact='_Per-scenario benchmark rows were not included in this artifact._'), '', 'Learning-evidence deltas with CI[^ci-method]:', '', _markdown_table_or_message(aggregate_learning_rows, ('comparison', 'metric', 'delta', 'ci_lower', 'ci_upper', 'n_seeds'), section=aggregate_benchmark_tables, expected_but_missing='_Learning-evidence benchmark rows were expected in this artifact but missing._', not_in_artifact='_Learning-evidence benchmark rows were not included in this artifact._'), '', '## Claim Test Results with Uncertainty', '', _markdown_table_or_message(claim_uncertainty_rows, ('claim', 'role', 'condition', 'metric', 'value', 'ci_lower', 'ci_upper', 'n_seeds', 'cohens_d', 'magnitude'), section=claim_test_tables, expected_but_missing='_Claim-test uncertainty rows were expected in this artifact but missing._', not_in_artifact='_Claim-test uncertainty rows were not included in this artifact._'), '', '## Effect Sizes Against Baselines', '', "Cohen's d and magnitude labels are reported for the main baselines[^effect-size].", '', _markdown_table_or_message(effect_size_rows, ('domain', 'baseline', 'comparison', 'metric', 'raw_delta', 'delta_ci_lower', 'delta_ci_upper', 'cohens_d', 'magnitude', 'effect_ci_lower', 'effect_ci_upper', 'n_seeds'), section=effect_size_tables, expected_but_missing='_Effect-size rows were expected in this artifact but missing._', not_in_artifact='_Effect-size rows were not included in this artifact._'), '', '## Architectural Ladder Comparison', '', ladder_summary_line, '', _markdown_table(ladder_description_rows, ('rung', 'protocol_name', 'experimental_isolation_question')), '', 'Pairwise ladder comparisons:', '', _markdown_table(ladder_comparison_rows, ('baseline_rung', 'comparison_rung', 'scenario_success_rate_delta', 'effect_size', 'interpretation')), '', '## Cross-Profile Ladder Comparison', '', ladder_profile_summary_line, '', _markdown_table(ladder_profile_summary_rows, ('protocol_name', 'classification', 'classic_success_rate', 'ecological_success_rate', 'austere_success_rate', 'austere_competence_gap', 'interpretation')), '', 'Scenario-success shaping gaps:', '', _markdown_table(ladder_profile_gap_rows, ('protocol_name', 'classic_minus_austere', 'classic_ci_lower', 'classic_ci_upper', 'classic_effect_size', 'classic_n_seeds', 'ecological_minus_austere', 'ecological_ci_lower', 'ecological_ci_upper', 'ecological_effect_size', 'ecological_n_seeds')), '', '## Unified Architectural Ladder Report', '', ladder_report_summary_line, '', '### Comparison support status (confound-controlled)', '', comparison_support_line, '', 'Diagnostics:', '', *comparison_support_diagnostic_lines, '', '### Comparison status matrix (confound-controlled)', '', *comparison_matrix_markdown_lines, '', _markdown_table(ladder_report_rows, ('rung', 'protocol_name', 'variant', 'total_parameters', 'scenario_success_rate', 'ci_lower', 'ci_upper', 'effect_size_vs_a0', 'effect_magnitude', 'n_seeds')), '', 'Adjacent rung comparisons:', '', _markdown_table(ladder_report_adjacent_rows, ('baseline_rung', 'comparison_rung', 'delta', 'ci_lower', 'ci_upper', 'cohens_d', 'effect_magnitude', 'interpretation')), '', 'Capacity matching summary:', '', _markdown_table(ladder_report_capacity_rows, ('capacity_matched', 'ratio', 'largest_variant', 'smallest_variant')), '', 'Credit assignment comparison summary:', '', _markdown_table(ladder_report_credit_rows, ('rung', 'local_only_delta_vs_broadcast', 'counterfactual_delta_vs_broadcast', 'strategies_present')), '', 'Reward shaping sensitivity:', '', _markdown_table(ladder_report_shaping_rows, ('rung', 'classic_minus_austere', 'classic_effect_size', 'ecological_minus_austere', 'ecological_effect_size')), '', 'No-reflex competence:', '', _markdown_table(ladder_report_no_reflex_rows, ('rung', 'no_reflex_evaluated', 'eval_reflex_scale', 'scenario_success_rate')), '', 'Capability probe boundaries:', '', _markdown_table(ladder_report_capability_rows, ('scenario', 'requirement_level', 'first_competent_rung', 'highest_competent_rung', 'rationale')), '', '> Conclusion', '', _markdown_table(ladder_report_conclusion_rows, ('conclusion', 'rationale', 'confidence_level', 'confounding_factors')), '', 'Missing experiments before asserting modular emergence:', '', _markdown_table(ladder_report_missing_rows, ('experiment', 'description', 'priority')), '', 'Unified ladder limitations:', '', *ladder_report_limitation_lines, '', '## Shaping Minimization Program', '']
    toc_headings = _extract_toc_from_markdown_lines(markdown_body_lines)
    assigned_headings = _assign_github_anchors(toc_headings)
    overview_anchors = _overview_section_anchor_map(assigned_headings)
    markdown_lines = ['# Offline Analysis Report', '', *_render_overview_summary_block(report, section_anchors=overview_anchors), '', *_render_toc_block(assigned_headings), *markdown_body_lines]
    if shaping_warning_line:
        markdown_lines.extend([shaping_warning_line, ''])
    markdown_lines.extend(['### Dense vs Minimal Gap', '', _markdown_table(shaping_gap_rows, ('comparison', 'scenario_success_rate_delta', 'episode_success_rate_delta', 'mean_reward_delta', 'interpretation')), '', '### Profile-Level Summary', '', shaping_removed_gap_line, '', 'Profile disposition weight proxies:', '', _markdown_table(shaping_profile_rows, ('profile', 'disposition', 'total_weight_proxy')), '', 'Disposition component summary:', '', _markdown_table(shaping_disposition_rows, ('disposition', 'component_count', 'components')), '', '### Component Dispositions', '', _markdown_table(shaping_component_rows, ('Component', 'Category', 'Risk', 'Disposition', 'Rationale')), '', '### Behavior Survival', '', shaping_summary_line, '', _markdown_table(shaping_survival_rows, ('scenario', 'austere_success_rate', 'survives', 'episodes')), '', '## Scenario Success', '', _markdown_table(scenario_rows, ('scenario', 'success_rate', 'check_count')), '', '## Ablations', '', _markdown_table(ablation_rows, ('variant', 'architecture', 'eval_reflex_scale', 'scenario_success_rate', 'capacity_status')), '', 'Ablation predator-type comparisons:', '', _markdown_table(ablation_predator_type_rows, ('variant', 'visual_scenarios', 'olfactory_scenarios', 'visual_minus_olfactory')), '', '## Credit Assignment Analysis', '', _markdown_table(credit_strategy_rows, ('rung', 'strategy', 'scenario_success_rate', 'delta_vs_broadcast', 'dominant_module')), '', _markdown_table(credit_architecture_rows, ('architecture_rung', 'strategy', 'variant', 'scenario_success_rate', 'effective_module_count')), '', _markdown_table(credit_module_rows, ('rung', 'strategy', 'module', 'module_credit_weight', 'counterfactual_credit_weight', 'module_gradient_norm', 'dominant_module_rate')), '', _markdown_table(credit_scenario_rows, ('rung', 'strategy', 'scenario', 'success_rate')), '', *credit_interpretation_lines, '', '## Predator Type Specialization', '', specialization_summary_line, '', _markdown_table(specialization_rows, ('predator_type', 'visual_cortex_activation', 'sensory_cortex_activation', 'dominant_module')), '', _markdown_table(specialization_delta_rows, ('differential', 'value')), '', '## Representation Specialization', '', representation_summary_line, '', representation_chart_line, '', _markdown_table(representation_divergence_rows, ('module', 'js_divergence', 'interpretation')), '', _markdown_table(representation_action_center_rows, ('module', 'gate_differential', 'contribution_differential')), '', '## Noise Robustness Matrix', '', _markdown_table(robustness_rows, ('train \\ eval', *robustness_eval_conditions, 'mean')) if robustness_rows else '_No noise robustness matrix was available._', '', f"Overall robustness score: {_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('robustness_score')))}.", f"Diagonal score: {_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('diagonal_score')))}.", f"Off-diagonal score: {_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('off_diagonal_score')))}.", '', '## Module-Local Sufficiency', '', module_local_summary_line if module_local_summary_line else _unavailable_section_message(module_local_sufficiency, expected_but_missing='_Module-local sufficiency summary was expected in this artifact but missing._', not_in_artifact='_Module-local sufficiency summary was not included in this artifact._'), '', _markdown_table_or_message(module_local_rows, ('module', 'coverage_mode', 'seeds', 'minimal_sufficient_level', 'canonical_v4_pass', 'partial_variant_coverage'), section=module_local_sufficiency, expected_but_missing='_Module-local sufficiency rows were expected in this artifact but missing._', not_in_artifact='_Module-local sufficiency rows were not included in this artifact._'), '', '## Distillation Diagnostic', '', distillation_summary_line if distillation_summary_line else _unavailable_section_message(distillation_analysis, expected_but_missing='_Distillation diagnostic summary was expected in this artifact but missing._', not_in_artifact='_Distillation diagnostic summary was not included in this artifact._'), '', _markdown_table_or_message(distillation_rows, ('condition', 'episodes', 'survival_rate', 'mean_reward', 'mean_food_distance_delta'), section=distillation_analysis, expected_but_missing='_Distillation diagnostic rows were expected in this artifact but missing._', not_in_artifact='_Distillation diagnostic rows were not included in this artifact._'), '', '## Limitations', ''])
    limitations = report.get('limitations', [])
    if isinstance(limitations, list) and limitations:
        markdown_lines.extend((f'- {item}' for item in limitations))
    else:
        markdown_lines.append('- None.')
    markdown_lines.extend(['', '## Method Notes', '', '[^ci-method]: Confidence intervals use percentile bootstrap resampling over seed-level metric values at the reported confidence level; benchmark-of-record tables default to 95%.', "[^effect-size]: Cohen's d uses the pooled sample standard deviation. Magnitude labels follow common absolute-value thresholds: negligible < 0.2, small < 0.5, medium < 0.8, and large otherwise.", '', '## Generated Files', '', '- `training_eval.svg`', '- `scenario_success.svg`', '- `robustness_matrix.svg`', '- `representation_specialization.svg`', '- `scenario_checks.csv`', '- `reward_components.csv`', '- `report.json`'])
    if ablation_svg_path:
        markdown_lines.append('- `ablation_comparison.svg`')
    if capacity_comparison_svg_path:
        markdown_lines.append('- `capacity_comparison.svg`')
    if reflex_svg_path:
        markdown_lines.append('- `reflex_frequency.svg`')
    report_md_path = output_path / 'report.md'
    report_md_path.write_text('\n'.join(markdown_lines) + '\n', encoding='utf-8')
    return {'report_md': str(report_md_path), 'report_json': str(report_json_path), 'training_eval_svg': str(training_svg_path), 'scenario_success_svg': str(scenario_svg_path), 'robustness_matrix_svg': str(robustness_svg_path), 'representation_specialization_svg': str(representation_svg_path), 'scenario_checks_csv': str(scenario_checks_path), 'reward_components_csv': str(reward_components_path), 'ablation_comparison_svg': ablation_svg_path, 'capacity_comparison_svg': capacity_comparison_svg_path, 'capacity_sweep_svg': capacity_comparison_svg_path, 'reflex_frequency_svg': reflex_svg_path}

def normalize_gate_status(value: object) -> str:
    """Normalize a gate/check outcome to a small set of status labels.

    This helper exists so the index (and any other static views) can apply a
    consistent mapping for highlighting. It is intentionally conservative.

    Accepted inputs:
    - bool: True -> pass, False -> fail
    - str: common spellings like 'passed', 'failed', 'skipped', 'borderline'
    - mapping: looks for keys like 'passed'/'pass'/'status'

    Returns:
        One of: 'pass', 'fail', 'skipped', 'borderline', 'unknown'
    """
    if isinstance(value, bool):
        return 'pass' if value else 'fail'

    if isinstance(value, Mapping):
        for key in ('status', 'outcome', 'result'):
            if key in value:
                nested = value.get(key)
                if isinstance(nested, Mapping):
                    return 'unknown'
                return normalize_gate_status(nested)
        for key in ('passed', 'pass', 'ok', 'success'):
            if key in value and isinstance(value.get(key), bool):
                return 'pass' if bool(value.get(key)) else 'fail'

    if value is None:
        return 'unknown'

    status = str(value).strip().lower()
    if status in {'pass', 'passed', 'ok', 'success', 'true'}:
        return 'pass'
    if status in {'fail', 'failed', 'error', 'false'}:
        return 'fail'
    if status in {'skipped', 'skip', 'not_run', 'not_applicable', 'n/a'}:
        return 'skipped'
    if status in {'borderline', 'warning', 'soft_fail'}:
        return 'borderline'
    return 'unknown'


__all__ = [
    "assign_github_anchors",
    "build_report_data",
    "extract_toc_from_markdown_lines",
    "normalize_gate_status",
    "write_report",
]
