from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from ...ablations import compare_predator_type_ablation_performance
from ..constants import (
    CANONICAL_NOISE_CONDITIONS,
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    DEFAULT_MODULE_NAMES,
    REFLEX_DOMINANCE_WARNING_THRESHOLD,
    REFLEX_OVERRIDE_WARNING_THRESHOLD,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
from ..uncertainty import _payload_has_zero_reflex_scale, _payload_uncertainty
from ..utils import (
    _coerce_bool,
    _coerce_float,
    _coerce_optional_float,
    _dominant_module_by_score,
    _mapping_or_empty,
    _mean,
    _safe_divide,
)

def _ordered_noise_conditions(names: Iterable[str]) -> list[str]:
    """
    Return a stable ordering of noise condition names prioritizing canonical conditions.
    
    Parameters:
        names (Iterable[str]): Iterable of noise condition names; empty or falsy names are ignored.
    
    Returns:
        list[str]: Unique names present in `names`, with canonical conditions (as listed in CANONICAL_NOISE_CONDITIONS)
        kept in their canonical order first, followed by any remaining names sorted alphabetically.
    """
    unique_names = {str(name) for name in names if str(name)}
    ordered = [
        name for name in CANONICAL_NOISE_CONDITIONS if name in unique_names
    ]
    extras = sorted(name for name in unique_names if name not in CANONICAL_NOISE_CONDITIONS)
    return [*ordered, *extras]

def _canonicalize_inferred_noise_conditions(names: Iterable[str]) -> list[str]:
    """
    Return an ordered list of noise condition names, expanding canonical subsets to the full canonical ordering.
    
    If `names` is empty, returns the full `CANONICAL_NOISE_CONDITIONS`. If every observed name is a member of `CANONICAL_NOISE_CONDITIONS`, returns the full canonical ordering. Otherwise returns the observed names ordered with canonical members first.
     
    Returns:
        list[str]: The final ordered list of noise condition names.
    """
    ordered = _ordered_noise_conditions(names)
    if not ordered:
        return list(CANONICAL_NOISE_CONDITIONS)
    if all(name in CANONICAL_NOISE_CONDITIONS for name in ordered):
        return list(CANONICAL_NOISE_CONDITIONS)
    return ordered

def _noise_robustness_cell_summary(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """
    Produce a normalized summary of a single robustness-matrix cell.
    
    Parameters:
        payload (Mapping[str, object]): Cell payload that may contain a `summary` mapping
            with numeric metrics and/or a `legacy_scenarios` mapping of per-scenario entries.
    
    Returns:
        dict[str, object]: A mapping with keys:
            - `scenario_success_rate`: float, coerced scenario-level success rate (0.0 if missing).
            - `episode_success_rate`: float, coerced episode-level success rate (0.0 if missing).
            - `mean_reward`: float, coerced mean reward for the cell; if `summary.mean_reward`
              is absent, the mean of `legacy_scenarios[*].mean_reward` is used (0.0 if none).
            - `scenario_count`: int, coerced scenario count (falls back to the number of
              `legacy_scenarios` entries when missing).
            - `episode_count`: int, coerced episode count (0 if missing).
    """
    summary = _mapping_or_empty(payload.get("summary"))
    legacy = _mapping_or_empty(payload.get("legacy_scenarios"))
    mean_reward_value = summary.get("mean_reward")
    mean_reward: float | None = (
        _coerce_float(mean_reward_value)
        if mean_reward_value is not None
        else None
    )
    if mean_reward is None:
        legacy_rewards = [
            _coerce_float(item.get("mean_reward"))
            for item in legacy.values()
            if isinstance(item, Mapping)
        ]
        mean_reward = _mean(legacy_rewards) if legacy_rewards else 0.0
    return {
        "scenario_success_rate": _coerce_float(summary.get("scenario_success_rate")),
        "episode_success_rate": _coerce_float(summary.get("episode_success_rate")),
        "mean_reward": _coerce_float(mean_reward),
        "scenario_count": int(_coerce_float(summary.get("scenario_count"), len(legacy))),
        "episode_count": int(_coerce_float(summary.get("episode_count"), 0.0)),
    }

def _normalize_noise_marginals(
    payload: object,
    *,
    conditions: Sequence[str],
    fallback: Mapping[str, object],
) -> dict[str, float]:
    """
    Builds a mapping of noise condition names to finite float marginals using values from `payload` with `fallback` as a secondary source.
    
    Parameters:
        payload (object): Mapping-like source of per-condition marginal values; non-mapping inputs are treated as empty.
        conditions (Sequence[str]): Ordered list of condition names to include in the result.
        fallback (Mapping[str, object]): Secondary mapping used when a condition is missing or not convertible in `payload`. Values from `fallback` are also coerced to finite floats.
    
    Returns:
        dict[str, float]: A dict mapping each condition name from `conditions` to a finite float. For each condition, the function uses the coerced value from `payload` when available; otherwise it uses the coerced value from `fallback`. If neither yields a finite number, the value defaults to 0.0.
    """
    raw = _mapping_or_empty(payload)
    return {
        condition: _coerce_float(raw.get(condition), _coerce_float(fallback.get(condition)))
        for condition in conditions
    }

def _noise_robustness_metrics(
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    train_conditions: Sequence[str],
    eval_conditions: Sequence[str],
) -> dict[str, object]:
    """
    Compute per-axis marginals and aggregate robustness scores from a train-by-eval robustness matrix.
    
    Missing or non-mapping cells are ignored when computing means; numeric values are coerced using the module's coercion helpers.
    
    Parameters:
        matrix (Mapping[str, Mapping[str, Mapping[str, object]]]): Nested mapping keyed by train condition then eval condition to a cell mapping containing numeric fields (notably `scenario_success_rate`).
        train_conditions (Sequence[str]): Ordered train-condition names to include as rows.
        eval_conditions (Sequence[str]): Ordered eval-condition names to include as columns.
    
    Returns:
        dict[str, object]: Summary with the following keys:
            - `train_marginals` (dict[str, float]): Mean `scenario_success_rate` per train condition across available eval columns.
            - `eval_marginals` (dict[str, float]): Mean `scenario_success_rate` per eval condition across available train rows.
            - `robustness_score` (float | None): Mean of all available cell `scenario_success_rate` values, or `None` if no cells contributed.
            - `diagonal_score` (float | None): Mean of cell scores where train == eval, or `None` if none present.
            - `off_diagonal_score` (float | None): Mean of cell scores where train != eval, or `None` if none present.
            - `available_cell_count` (int): Number of matrix cells that contributed to the computed scores.
    """
    train_marginals: dict[str, float] = {}
    eval_marginals: dict[str, float] = {}
    all_scores: list[float] = []
    diagonal_scores: list[float] = []
    off_diagonal_scores: list[float] = []

    for train_condition in train_conditions:
        train_scores = [
            _coerce_float(
                _mapping_or_empty(matrix.get(train_condition, {}))
                .get(eval_condition, {})
                .get("scenario_success_rate")
            )
            for eval_condition in eval_conditions
            if eval_condition in _mapping_or_empty(matrix.get(train_condition, {}))
        ]
        train_marginals[train_condition] = _mean(train_scores)

    for eval_condition in eval_conditions:
        eval_scores = [
            _coerce_float(
                _mapping_or_empty(matrix.get(train_condition, {}))
                .get(eval_condition, {})
                .get("scenario_success_rate")
            )
            for train_condition in train_conditions
            if eval_condition in _mapping_or_empty(matrix.get(train_condition, {}))
        ]
        eval_marginals[eval_condition] = _mean(eval_scores)

    for train_condition in train_conditions:
        for eval_condition in eval_conditions:
            cell = _mapping_or_empty(matrix.get(train_condition, {})).get(eval_condition)
            if not isinstance(cell, Mapping):
                continue
            score = _coerce_float(cell.get("scenario_success_rate"))
            all_scores.append(score)
            if train_condition == eval_condition:
                diagonal_scores.append(score)
            else:
                off_diagonal_scores.append(score)

    return {
        "train_marginals": train_marginals,
        "eval_marginals": eval_marginals,
        "robustness_score": _mean(all_scores),
        "diagonal_score": _mean(diagonal_scores),
        "off_diagonal_score": _mean(off_diagonal_scores),
        "available_cell_count": len(all_scores),
    }

def _noise_robustness_rate(
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    train_condition: str,
    eval_condition: str,
) -> float | None:
    """
    Retrieve the optional scenario success rate for a specific train/eval cell in the robustness matrix.
    
    Returns:
        float: The coerced `scenario_success_rate` for the specified cell, or `None` if the cell is absent, not a mapping, or does not contain a convertible `scenario_success_rate`.
    """
    cell = _mapping_or_empty(matrix.get(train_condition, {})).get(eval_condition)
    if not isinstance(cell, Mapping):
        return None
    return _coerce_optional_float(cell.get("scenario_success_rate"))

def _noise_robustness_marginal(
    marginals: Mapping[str, object],
    condition: str,
) -> float | None:
    """
    Retrieve the marginal value for a specific noise condition from a marginals mapping.
    
    Parameters:
        marginals (Mapping[str, object]): Mapping from condition names to raw marginal values (may be numeric or convertible).
        condition (str): The noise condition key to look up.
    
    Returns:
        The float value for the condition if present and convertible, otherwise `None`.
    """
    if condition not in marginals:
        return None
    return _coerce_optional_float(marginals.get(condition))

def _noise_robustness_mean(values: Iterable[float | None]) -> float | None:
    """
    Compute the mean of the input values while ignoring None entries.
    
    Parameters:
        values (Iterable[float | None]): Iterable of numeric values or `None` entries; `None` values are excluded from the calculation.
    
    Returns:
        float | None: The arithmetic mean of the non-`None` values, or `None` if no such values are present.
    """
    present = [value for value in values if value is not None]
    if not present:
        return None
    return _mean(present)

def _observed_noise_robustness_metrics(
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    train_conditions: Sequence[str],
    eval_conditions: Sequence[str],
) -> dict[str, object]:
    """
    Compute observed robustness aggregates from a nested train-by-eval matrix using only present cell rates.
    
    Parameters:
        matrix (Mapping[str, Mapping[str, Mapping[str, object]]]):
            Nested mapping keyed first by train condition then by eval condition to a cell payload that may contain a `scenario_success_rate`.
        train_conditions (Sequence[str]):
            Ordered list of train condition names to consider.
        eval_conditions (Sequence[str]):
            Ordered list of eval condition names to consider.
    
    Returns:
        dict[str, object]: A report containing:
            - "train_marginals" (dict[str, float]): Per-train-condition mean of present cell success rates across eval conditions (omits conditions with no present values).
            - "eval_marginals" (dict[str, float]): Per-eval-condition mean of present cell success rates across train conditions (omits conditions with no present values).
            - "robustness_score" (float|None): Mean of all present cell success rates, or `None` if none are present.
            - "diagonal_score" (float|None): Mean of present cell success rates where train==eval, or `None` if none are present.
            - "off_diagonal_score" (float|None): Mean of present cell success rates where train!=eval, or `None` if none are present.
            - "available_cell_count" (int): Number of cells that contributed a present success rate.
    """
    train_marginals: dict[str, float] = {}
    eval_marginals: dict[str, float] = {}
    all_scores: list[float] = []
    diagonal_scores: list[float] = []
    off_diagonal_scores: list[float] = []

    for train_condition in train_conditions:
        train_score = _noise_robustness_mean(
            _noise_robustness_rate(
                matrix,
                train_condition=train_condition,
                eval_condition=eval_condition,
            )
            for eval_condition in eval_conditions
        )
        if train_score is not None:
            train_marginals[train_condition] = train_score

    for eval_condition in eval_conditions:
        eval_score = _noise_robustness_mean(
            _noise_robustness_rate(
                matrix,
                train_condition=train_condition,
                eval_condition=eval_condition,
            )
            for train_condition in train_conditions
        )
        if eval_score is not None:
            eval_marginals[eval_condition] = eval_score

    for train_condition in train_conditions:
        for eval_condition in eval_conditions:
            score = _noise_robustness_rate(
                matrix,
                train_condition=train_condition,
                eval_condition=eval_condition,
            )
            if score is None:
                continue
            all_scores.append(score)
            if train_condition == eval_condition:
                diagonal_scores.append(score)
            else:
                off_diagonal_scores.append(score)

    return {
        "train_marginals": train_marginals,
        "eval_marginals": eval_marginals,
        "robustness_score": _noise_robustness_mean(all_scores),
        "diagonal_score": _noise_robustness_mean(diagonal_scores),
        "off_diagonal_score": _noise_robustness_mean(off_diagonal_scores),
        "available_cell_count": len(all_scores),
    }

def extract_noise_robustness(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Compute a train-by-eval noise robustness matrix and summary metrics from available summary payload or behavior CSV rows.
    
    Parameters:
        summary (Mapping[str, object]): Parsed analysis summary which may contain a precomputed
            `behavior_evaluation.robustness_matrix` payload.
        behavior_rows (Sequence[Mapping[str, object]]): Normalized behavior CSV rows; used to
            reconstruct the robustness matrix when the summary payload is absent.
    
    Returns:
        dict[str, object]: A structured robustness report containing:
            - available (bool): Whether any matrix cells are present.
            - source (str): Either "summary.behavior_evaluation.robustness_matrix", "behavior_csv", or "none".
            - matrix_spec (dict): Keys `train_conditions`, `eval_conditions`, and `cell_count`.
            - matrix (dict): Mapping train_condition -> eval_condition -> cell summary with keys
              `scenario_success_rate`, `episode_success_rate`, `mean_reward`, `scenario_count`, `episode_count`.
            - train_marginals / eval_marginals (dict): Marginal scores per condition.
            - robustness_score, diagonal_score, off_diagonal_score (float): Aggregate metrics.
            - metadata (dict): `complete` (bool), `available_cell_count`, and `expected_cell_count`.
            - limitations (list[str]): Human-readable notes when payloads or cells were missing or reconstructed.
    """
    behavior_evaluation = summary.get("behavior_evaluation", {})
    robustness_payload = (
        behavior_evaluation.get("robustness_matrix", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    expected_train_conditions: list[str] = []
    expected_eval_conditions: list[str] = []
    summary_available_cell_count = 0
    summary_was_incomplete = False
    summary_incomplete_limitation: str | None = None
    partial_summary_matrix: dict[str, dict[str, dict[str, object]]] = {}
    partial_summary_train_marginals: dict[str, float] = {}
    partial_summary_eval_marginals: dict[str, float] = {}

    if isinstance(robustness_payload, Mapping) and robustness_payload:
        raw_matrix = _mapping_or_empty(robustness_payload.get("matrix"))
        matrix_spec = _mapping_or_empty(
            robustness_payload.get("matrix_spec")
            or robustness_payload.get("robustness_matrix")
        )
        _tc_raw = matrix_spec.get("train_conditions")
        _ec_raw = matrix_spec.get("eval_conditions")
        train_conditions = list(_tc_raw) if isinstance(_tc_raw, list) else []
        eval_conditions = list(_ec_raw) if isinstance(_ec_raw, list) else []
        if not train_conditions:
            train_conditions = _canonicalize_inferred_noise_conditions(
                raw_matrix.keys()
            )
        if not eval_conditions:
            eval_names = {
                str(eval_condition)
                for row in raw_matrix.values()
                if isinstance(row, Mapping)
                for eval_condition in row.keys()
            }
            eval_conditions = _canonicalize_inferred_noise_conditions(eval_names)
        expected_train_conditions = list(train_conditions)
        expected_eval_conditions = list(eval_conditions)

        matrix: dict[str, dict[str, dict[str, object]]] = {}
        for train_condition in train_conditions:
            raw_row = _mapping_or_empty(raw_matrix.get(train_condition))
            matrix[train_condition] = {}
            for eval_condition in eval_conditions:
                raw_cell = _mapping_or_empty(raw_row.get(eval_condition))
                if raw_cell:
                    matrix[train_condition][eval_condition] = (
                        _noise_robustness_cell_summary(raw_cell)
                    )

        metrics = _noise_robustness_metrics(
            matrix,
            train_conditions=train_conditions,
            eval_conditions=eval_conditions,
        )
        train_marginals = _normalize_noise_marginals(
            robustness_payload.get("train_marginals"),
            conditions=train_conditions,
            fallback=metrics["train_marginals"],
        )
        eval_marginals = _normalize_noise_marginals(
            robustness_payload.get("eval_marginals"),
            conditions=eval_conditions,
            fallback=metrics["eval_marginals"],
        )
        expected_cell_count = len(train_conditions) * len(eval_conditions)
        if expected_cell_count > 0 and metrics["available_cell_count"] == expected_cell_count:
            return {
                "available": True,
                "source": "summary.behavior_evaluation.robustness_matrix",
                "matrix_spec": {
                    "train_conditions": train_conditions,
                    "eval_conditions": eval_conditions,
                    "cell_count": expected_cell_count,
                },
                "matrix": matrix,
                "train_marginals": train_marginals,
                "eval_marginals": eval_marginals,
                "robustness_score": _coerce_float(
                    robustness_payload.get("robustness_score"),
                    metrics["robustness_score"],
                ),
                "diagonal_score": _coerce_float(
                    robustness_payload.get("diagonal_score"),
                    metrics["diagonal_score"],
                ),
                "off_diagonal_score": _coerce_float(
                    robustness_payload.get("off_diagonal_score"),
                    metrics["off_diagonal_score"],
                ),
                "metadata": {
                    "complete": True,
                    "available_cell_count": metrics["available_cell_count"],
                    "expected_cell_count": expected_cell_count,
                },
                "limitations": limitations,
            }
        if expected_cell_count == 0:
            summary_incomplete_limitation = (
                "Noise robustness matrix was present in the summary but did not define any train/eval conditions."
            )
        else:
            summary_incomplete_limitation = (
                "Noise robustness matrix was present in the summary but some train/eval cells were missing."
            )
        observed_metrics = _observed_noise_robustness_metrics(
            matrix,
            train_conditions=train_conditions,
            eval_conditions=eval_conditions,
        )
        summary_available_cell_count = metrics["available_cell_count"]
        summary_was_incomplete = True
        partial_summary_matrix = matrix
        partial_summary_train_marginals = observed_metrics["train_marginals"]
        partial_summary_eval_marginals = observed_metrics["eval_marginals"]

    by_cell: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    missing_noise_metadata = 0
    for row in behavior_rows:
        train_condition = str(row.get("train_noise_profile") or "")
        eval_condition = str(row.get("eval_noise_profile") or "")
        if not train_condition or not eval_condition:
            if row.get("noise_profile") or row.get("train_noise_profile") or row.get("eval_noise_profile"):
                missing_noise_metadata += 1
            continue
        by_cell[(train_condition, eval_condition)].append(row)

    if not by_cell:
        if summary_incomplete_limitation is not None:
            limitations.append(summary_incomplete_limitation)
        if summary_was_incomplete:
            limitations.append(
                "Behavior CSV did not provide enough train/eval noise metadata to reconstruct the missing robustness cells."
            )
        train_conditions = list(expected_train_conditions)
        eval_conditions = list(expected_eval_conditions)
        expected_cell_count = len(train_conditions) * len(eval_conditions)
        available = (
            summary_available_cell_count > 0
            or bool(partial_summary_matrix)
            or bool(partial_summary_train_marginals)
            or bool(partial_summary_eval_marginals)
        )
        return {
            "available": available,
            "source": (
                "summary.behavior_evaluation.robustness_matrix"
                if summary_was_incomplete
                else "none"
            ),
            "matrix_spec": {
                "train_conditions": train_conditions,
                "eval_conditions": eval_conditions,
                "cell_count": expected_cell_count,
            },
            "matrix": partial_summary_matrix,
            "train_marginals": partial_summary_train_marginals,
            "eval_marginals": partial_summary_eval_marginals,
            "robustness_score": None,
            "diagonal_score": None,
            "off_diagonal_score": None,
            "metadata": {
                "complete": False,
                "available_cell_count": summary_available_cell_count,
                "expected_cell_count": expected_cell_count,
            },
            "limitations": limitations
            or [
                "No noise robustness matrix payload or train/eval noise rows were available."
            ],
        }

    derived_train_conditions = _ordered_noise_conditions(
        train_condition for train_condition, _ in by_cell
    )
    derived_eval_conditions = _ordered_noise_conditions(
        eval_condition for _, eval_condition in by_cell
    )
    train_conditions = [
        *expected_train_conditions,
        *[
            name
            for name in derived_train_conditions
            if name not in expected_train_conditions
        ],
    ] if expected_train_conditions else derived_train_conditions
    eval_conditions = [
        *expected_eval_conditions,
        *[
            name
            for name in derived_eval_conditions
            if name not in expected_eval_conditions
        ],
    ] if expected_eval_conditions else derived_eval_conditions
    matrix: dict[str, dict[str, dict[str, object]]] = {
        train_condition: {}
        for train_condition in train_conditions
    }
    for train_condition in train_conditions:
        partial_row = _mapping_or_empty(partial_summary_matrix.get(train_condition))
        for eval_condition in eval_conditions:
            partial_cell = _mapping_or_empty(partial_row.get(eval_condition))
            if partial_cell:
                matrix[train_condition][eval_condition] = dict(partial_cell)
    for (train_condition, eval_condition), items in sorted(by_cell.items()):
        success_values = [
            1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items
        ]
        scenario_names = {
            str(row.get("scenario") or "")
            for row in items
            if row.get("scenario")
        }
        matrix[train_condition][eval_condition] = {
            "scenario_success_rate": _mean(success_values),
            "episode_success_rate": _mean(success_values),
            "mean_reward": 0.0,
            "scenario_count": len(scenario_names),
            "episode_count": len(items),
        }

    metrics = _noise_robustness_metrics(
        matrix,
        train_conditions=train_conditions,
        eval_conditions=eval_conditions,
    )
    observed_metrics = _observed_noise_robustness_metrics(
        matrix,
        train_conditions=train_conditions,
        eval_conditions=eval_conditions,
    )
    expected_cell_count = len(train_conditions) * len(eval_conditions)
    if metrics["available_cell_count"] < expected_cell_count:
        if summary_incomplete_limitation is not None:
            limitations.append(summary_incomplete_limitation)
        limitations.append(
            "Noise robustness matrix was reconstructed from behavior_csv and some train/eval cells were missing."
        )
    else:
        limitations.append(
            (
                "Noise robustness matrix was reconstructed from behavior_csv because summary.behavior_evaluation.robustness_matrix was incomplete."
                if summary_was_incomplete
                else "Noise robustness matrix was reconstructed from behavior_csv because summary.behavior_evaluation.robustness_matrix was absent."
            )
        )
    if missing_noise_metadata:
        limitations.append(
            "Some behavior_csv rows were ignored because train_noise_profile or eval_noise_profile metadata was missing."
        )
    complete = bool(expected_cell_count) and metrics["available_cell_count"] == expected_cell_count
    available = metrics["available_cell_count"] > 0 or bool(matrix)
    return {
        "available": available,
        "source": "behavior_csv",
        "matrix_spec": {
            "train_conditions": train_conditions,
            "eval_conditions": eval_conditions,
            "cell_count": expected_cell_count,
        },
        "matrix": matrix,
        "train_marginals": (
            metrics["train_marginals"]
            if complete
            else observed_metrics["train_marginals"]
        ),
        "eval_marginals": (
            metrics["eval_marginals"]
            if complete
            else observed_metrics["eval_marginals"]
        ),
        "robustness_score": metrics["robustness_score"] if complete else None,
        "diagonal_score": metrics["diagonal_score"] if complete else None,
        "off_diagonal_score": metrics["off_diagonal_score"] if complete else None,
        "metadata": {
            "complete": complete,
            "available_cell_count": metrics["available_cell_count"],
            "expected_cell_count": expected_cell_count,
        },
        "limitations": limitations,
    }
