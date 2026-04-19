from __future__ import annotations

from collections.abc import Mapping, Sequence

from .benchmark import _fallback_group_summary

def extract_comparisons(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Produce reward-profile and map-template comparison summaries, preferring a precomputed comparisons payload in the provided summary and falling back to reconstruction from behavior CSV rows.
    
    When the summary contains a `behavior_evaluation.comparisons` mapping, that payload is returned (with `reward_profiles` and `map_templates` taken directly). Otherwise this function groups `behavior_rows` by `reward_profile` and `evaluation_map` to reconstruct comparable summaries and records a limitation note.
    
    Returns:
        dict: A mapping with the following keys:
            - available (bool): True when any comparison data is present.
            - source (str): `"summary.behavior_evaluation.comparisons"` when taken from the summary, otherwise `"behavior_csv"`.
            - reward_profiles (dict): Mapping of reward-profile identifiers to group summary objects (empty if none).
            - map_templates (dict): Mapping of map/template identifiers to group summary objects (empty if none).
            - limitations (list[str]): Human-readable notes describing reconstruction or missing-data limitations.
    """
    behavior_evaluation = summary.get("behavior_evaluation", {})
    comparisons = (
        behavior_evaluation.get("comparisons", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    if isinstance(comparisons, Mapping) and comparisons:
        reward_profiles = comparisons.get("reward_profiles", {})
        map_templates = comparisons.get("map_templates", {})
        return {
            "available": bool(reward_profiles or map_templates),
            "source": "summary.behavior_evaluation.comparisons",
            "reward_profiles": dict(reward_profiles) if isinstance(reward_profiles, Mapping) else {},
            "map_templates": dict(map_templates) if isinstance(map_templates, Mapping) else {},
            "limitations": [],
        }

    reward_profiles = _fallback_group_summary(behavior_rows, key_name="reward_profile")
    map_templates = _fallback_group_summary(behavior_rows, key_name="evaluation_map")
    if reward_profiles or map_templates:
        limitations.append(
            "Profile/map comparisons were reconstructed from behavior_csv and may not match the compact comparison payload exactly."
        )
    else:
        limitations.append("No profile or map comparison payload was available.")
    return {
        "available": bool(reward_profiles or map_templates),
        "source": "behavior_csv",
        "reward_profiles": reward_profiles,
        "map_templates": map_templates,
        "limitations": limitations,
    }
