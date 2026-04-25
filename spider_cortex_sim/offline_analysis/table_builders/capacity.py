from __future__ import annotations

from collections.abc import Mapping

from .common import CAPACITY_SWEEP_INTERPRETATION_GUIDANCE, extract_capacity_sweeps
from ..writers import _table

def build_capacity_sweep_tables(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """
    Builds structured capacity-sweep tables and lookup matrices from an analysis summary.
    
    Parameters:
        summary (Mapping[str, object]): Analysis summary mapping from which capacity sweep data will be extracted.
    
    Returns:
        result (dict[str, object]): A dictionary with keys:
            - "available": `True` if capacity sweep rows were found, `False` otherwise.
            - "curves": A table-like structure describing capacity sweep rows (empty schema when unavailable).
            - "matrices": Nested dictionaries keyed first by metric, then variant, then capacity profile containing metric values.
            - "interpretations": List of interpretation entries extracted from the summary.
            - "metadata": Metadata including interpretation guidance.
            - "limitations": List of limitations extracted from the summary (stringified when no rows).
    """
    capacity_sweeps = extract_capacity_sweeps(summary)
    capacity_sweep_rows = list(capacity_sweeps.get("rows", []))
    if not capacity_sweep_rows:
        return {
            "available": False,
            "curves": _table(
                (
                    "variant",
                    "architecture",
                    "capacity_profile",
                    "capacity_profile_version",
                    "scale_factor",
                    "total_trainable",
                    "approximate_compute_cost_total",
                    "approximate_compute_cost_unit",
                    "scenario_success_rate",
                    "episode_success_rate",
                    "capability_probe_success_rate",
                    "source",
                ),
                (),
            ),
            "matrices": {},
            "interpretations": list(capacity_sweeps.get("interpretations", [])),
            "metadata": {
                "interpretation_guidance": CAPACITY_SWEEP_INTERPRETATION_GUIDANCE,
            },
            "limitations": [
                str(item)
                for item in capacity_sweeps.get("limitations", [])
                if item
            ],
        }

    matrices: dict[str, dict[str, dict[str, object]]] = {
        "scenario_success_rate": {},
        "episode_success_rate": {},
        "capability_probe_success_rate": {},
        "total_trainable": {},
        "approximate_compute_cost_total": {},
    }
    for row in capacity_sweep_rows:
        if not isinstance(row, Mapping):
            continue
        variant_name = str(row.get("variant") or "")
        profile_name = str(row.get("capacity_profile") or "")
        if not variant_name or not profile_name:
            continue
        matrices["scenario_success_rate"].setdefault(variant_name, {})[profile_name] = (
            row.get("scenario_success_rate")
        )
        matrices["episode_success_rate"].setdefault(variant_name, {})[profile_name] = (
            row.get("episode_success_rate")
        )
        matrices["capability_probe_success_rate"].setdefault(
            variant_name,
            {},
        )[profile_name] = row.get("capability_probe_success_rate")
        matrices["total_trainable"].setdefault(variant_name, {})[profile_name] = (
            row.get("total_trainable")
        )
        matrices["approximate_compute_cost_total"].setdefault(
            variant_name,
            {},
        )[profile_name] = row.get("approximate_compute_cost_total")

    return {
        "available": True,
        "curves": _table(
            (
                "variant",
                "architecture",
                "capacity_profile",
                "capacity_profile_version",
                "scale_factor",
                "total_trainable",
                "approximate_compute_cost_total",
                "approximate_compute_cost_unit",
                "scenario_success_rate",
                "episode_success_rate",
                "capability_probe_success_rate",
                "source",
            ),
            capacity_sweep_rows,
        ),
        "matrices": matrices,
        "interpretations": list(capacity_sweeps.get("interpretations", [])),
        "metadata": {
            "interpretation_guidance": CAPACITY_SWEEP_INTERPRETATION_GUIDANCE,
        },
        "limitations": list(capacity_sweeps.get("limitations", [])),
    }

__all__ = ["build_capacity_sweep_tables"]
