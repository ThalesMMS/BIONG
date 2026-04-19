"""CLI compatibility facade.

The monolithic ``spider_cortex_sim.cli`` module was split into parser,
command, and formatting submodules. Public callers should import
``build_parser`` and ``main`` from this package; legacy private helper aliases
remain here only for backwards compatibility.
"""

from __future__ import annotations

from . import commands as _commands
from .commands import (
    EXPERIMENT_OF_RECORD_CHECKPOINT_DOMINANCE_PENALTY,
    EXPERIMENT_OF_RECORD_CHECKPOINT_OVERRIDE_PENALTY,
    _benchmark_package_manifest_summary,
    _build_and_record_benchmark_package,
    _build_compare_training_kwargs,
)
from .formatting import (
    _short_claim_test_suite_summary,
    _short_robustness_matrix_summary,
)
from .parser import (
    build_parser,
    collect_requested_scenarios,
    parse_module_reflex_scales,
)
from ..ablations import BrainAblationConfig, canonical_ablation_variant_names
from ..benchmark_package import (
    BenchmarkManifest,
    assemble_benchmark_package,
    build_and_record_benchmark_package,
    summarize_benchmark_manifest,
)
from ..budget_profiles import (
    CHECKPOINT_METRIC_NAMES,
    CHECKPOINT_SELECTION_NAMES,
    canonical_budget_profile_names,
    resolve_budget,
    resolve_budget_profile,
)
from ..checkpointing import build_checkpointing_summary
from ..claim_evaluation import condense_claim_test_summary, run_claim_test_suite
from ..claim_tests import claim_test_names
from ..comparison import (
    compare_ablation_suite,
    compare_behavior_suite,
    compare_configurations,
    compare_learning_evidence,
    compare_noise_robustness,
    compare_training_regimes,
    condense_robustness_summary,
)
from ..maps import MAP_TEMPLATE_NAMES
from ..noise import canonical_noise_profile_names
from ..operational_profiles import canonical_operational_profile_names
from ..scenarios import SCENARIO_NAMES
from ..simulation import (
    CHECKPOINT_PENALTY_MODE_NAMES,
    CURRICULUM_PROFILE_NAMES,
    EXPERIMENT_OF_RECORD_REGIME,
    SpiderSimulation,
    default_behavior_evaluation,
    ensure_behavior_evaluation,
)
from ..training_regimes import canonical_training_regime_names
from ..world import REWARD_PROFILES

_collect_requested_scenarios = collect_requested_scenarios
_default_behavior_evaluation = default_behavior_evaluation
_default_checkpointing_summary = build_checkpointing_summary
_ensure_behavior_evaluation = ensure_behavior_evaluation
_parse_module_reflex_scales = parse_module_reflex_scales


def _sync_command_patch_points() -> None:
    _commands.SpiderSimulation = SpiderSimulation
    _commands.assemble_benchmark_package = assemble_benchmark_package
    _commands.run_claim_test_suite = run_claim_test_suite
    _commands.compare_ablation_suite = compare_ablation_suite
    _commands.compare_behavior_suite = compare_behavior_suite
    _commands.compare_configurations = compare_configurations
    _commands.compare_learning_evidence = compare_learning_evidence
    _commands.compare_noise_robustness = compare_noise_robustness
    _commands.compare_training_regimes = compare_training_regimes


def run_cli(args) -> None:
    _sync_command_patch_points()
    _commands.run_cli(args)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args._parser = parser
    run_cli(args)


__all__ = [
    "build_parser",
    "main",
    "run_cli",
]
