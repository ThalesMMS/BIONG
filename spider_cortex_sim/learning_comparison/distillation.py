from __future__ import annotations

from .common import *

def build_distillation_comparison_report(
    *,
    teacher: Mapping[str, object],
    modular_rl_from_scratch: Mapping[str, object] | None,
    modular_distilled: Mapping[str, object],
    modular_distilled_plus_rl_finetuning: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """
    Builds a comparison report assessing whether a modular architecture can express an effective policy.
    
    Parameters:
        teacher (Mapping[str, object]): Summary metrics from the teacher policy; required.
        modular_rl_from_scratch (Mapping[str, object] | None): Optional summary metrics for an RL-only modular baseline.
        modular_distilled (Mapping[str, object]): Summary metrics for the distilled modular student; required.
        modular_distilled_plus_rl_finetuning (Mapping[str, object] | None): Optional summary metrics for the distilled student after RL finetuning.
    
    Returns:
        Dict[str, object]: A report containing:
            - question: The posed evaluation question.
            - answer: One of "yes", "yes_after_rl_finetuning", "partially", "no", or "inconclusive".
            - rationale: Text explaining the chosen answer.
            - primary_metric: The metric used as primary evidence ("scenario_success_rate").
            - teacher: Normalized teacher summary (compact metric dictionary).
            - modular_rl_from_scratch: Normalized baseline summary or None.
            - modular_distilled: Normalized distilled student summary.
            - modular_distilled_plus_rl_finetuning: Normalized finetuned student summary or None.
            - best_student_condition: Either "modular_distilled" or "modular_distilled_plus_rl_finetuning".
            - quantitative_evidence: Dict of numeric comparisons including deltas for
              scenario_success_rate, episode_success_rate, mean_reward and gap-closed fractions when applicable.
    """
    def _summary(summary: Mapping[str, object] | None) -> Dict[str, float]:
        raw_payload = dict(summary or {})
        payload = (
            raw_payload
            if "summary" in raw_payload or "suite" in raw_payload
            else {"summary": raw_payload}
        )
        return condition_compact_summary(payload)

    def _metric_delta(
        candidate_summary: Dict[str, float],
        reference_summary: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute the difference for three primary metrics between a candidate and a reference summary.
        
        Both input summaries must contain the keys "scenario_success_rate", "episode_success_rate", and "mean_reward". Each returned delta is the candidate value minus the reference value, rounded to six decimal places.
        
        Parameters:
            candidate_summary (Dict[str, float]): Mapping with keys "scenario_success_rate", "episode_success_rate", and "mean_reward" for the candidate condition.
            reference_summary (Dict[str, float]): Mapping with the same keys for the reference condition.
        
        Returns:
            Dict[str, float]: A mapping with keys:
                - "scenario_success_rate_delta": candidate - reference for scenario success rate, rounded to 6 decimals.
                - "episode_success_rate_delta": candidate - reference for episode success rate, rounded to 6 decimals.
                - "mean_reward_delta": candidate - reference for mean reward, rounded to 6 decimals.
        """
        return {
            "scenario_success_rate_delta": round(
                candidate_summary["scenario_success_rate"]
                - reference_summary["scenario_success_rate"],
                6,
            ),
            "episode_success_rate_delta": round(
                candidate_summary["episode_success_rate"]
                - reference_summary["episode_success_rate"],
                6,
            ),
            "mean_reward_delta": round(
                candidate_summary["mean_reward"]
                - reference_summary["mean_reward"],
                6,
            ),
        }

    def _gap_closed_fraction(
        *,
        baseline_value: float,
        teacher_value: float,
        candidate_value: float,
    ) -> float | None:
        """
        Compute the fraction of the teacher–baseline gap that the candidate achieved for a scalar metric.
        
        Parameters:
            baseline_value (float): Metric value for the baseline condition.
            teacher_value (float): Metric value for the teacher/reference condition.
            candidate_value (float): Metric value for the candidate condition.
        
        Returns:
            float or None: The fraction (rounded to 6 decimals) of the gap from baseline to teacher closed by the candidate:
            (candidate_value - baseline_value) / (teacher_value - baseline_value). Returns `None` if the teacher does not exceed
            the baseline or the teacher–baseline gap is effectively zero.
        """
        teacher_gap = float(teacher_value) - float(baseline_value)
        if teacher_gap <= 0.0 or math.isclose(
            teacher_gap,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return None
        return round(
            (float(candidate_value) - float(baseline_value)) / teacher_gap,
            6,
        )

    teacher_summary = _summary(teacher)
    baseline_summary = (
        None
        if modular_rl_from_scratch is None
        else _summary(modular_rl_from_scratch)
    )
    distilled_summary = _summary(modular_distilled)
    finetuned_summary = (
        None
        if modular_distilled_plus_rl_finetuning is None
        else _summary(modular_distilled_plus_rl_finetuning)
    )

    baseline_success = (
        None
        if baseline_summary is None
        else baseline_summary["scenario_success_rate"]
    )
    distilled_success = distilled_summary["scenario_success_rate"]
    finetuned_success = (
        None
        if finetuned_summary is None
        else finetuned_summary["scenario_success_rate"]
    )
    teacher_success = teacher_summary["scenario_success_rate"]
    best_condition_name = "modular_distilled"
    best_summary = distilled_summary
    if (
        finetuned_summary is not None
        and finetuned_success is not None
        and finetuned_success > distilled_success
    ):
        best_condition_name = "modular_distilled_plus_rl_finetuning"
        best_summary = finetuned_summary

    if baseline_success is None:
        if teacher_success <= 0.0:
            answer = "inconclusive"
            rationale = (
                "The teacher did not establish a positive no-reflex reference, so expressivity remains unproven."
            )
        elif best_summary["scenario_success_rate"] > 0.0:
            answer = (
                "yes"
                if best_condition_name == "modular_distilled"
                else "yes_after_rl_finetuning"
            )
            rationale = (
                f"{best_condition_name} achieved scenario_success_rate "
                f"{best_summary['scenario_success_rate']:.3f} against a teacher reference of {teacher_success:.3f}."
            )
        else:
            answer = "no"
            rationale = (
                "The modular student did not achieve positive no-reflex success after distillation."
            )
    elif teacher_success <= baseline_success:
        answer = "inconclusive"
        rationale = (
            "The teacher did not outperform the modular RL baseline, so expressivity remains unproven."
        )
    elif best_summary["scenario_success_rate"] > baseline_success:
        answer = (
            "yes"
            if best_condition_name == "modular_distilled"
            else "yes_after_rl_finetuning"
        )
        rationale = (
            f"{best_condition_name} improved scenario_success_rate from "
            f"{baseline_success:.3f} to {best_summary['scenario_success_rate']:.3f} "
            f"against a teacher reference of {teacher_success:.3f}."
        )
    elif best_summary["scenario_success_rate"] >= baseline_success * 0.5:
        answer = "partially"
        rationale = (
            f"The modular student achieved non-zero scenario_success_rate "
            f"({best_summary['scenario_success_rate']:.3f}) but did not surpass the RL-only modular baseline ({baseline_success:.3f})."
        )
    else:
        answer = "no"
        rationale = (
            "The modular student did not achieve enough no-reflex success after distillation to support the expressivity claim."
        )

    quantitative_evidence = {
        "modular_distilled_vs_teacher": _metric_delta(
            distilled_summary,
            teacher_summary,
        ),
    }
    if baseline_summary is not None and baseline_success is not None:
        quantitative_evidence["teacher_vs_modular_rl_from_scratch"] = _metric_delta(
            teacher_summary,
            baseline_summary,
        )
        quantitative_evidence[
            "modular_distilled_vs_modular_rl_from_scratch"
        ] = _metric_delta(
            distilled_summary,
            baseline_summary,
        )
        quantitative_evidence["gap_closed_fraction"] = {
            "modular_distilled": _gap_closed_fraction(
                baseline_value=baseline_success,
                teacher_value=teacher_success,
                candidate_value=distilled_success,
            ),
            "modular_distilled_plus_rl_finetuning": (
                None
                if finetuned_success is None
                else _gap_closed_fraction(
                    baseline_value=baseline_success,
                    teacher_value=teacher_success,
                    candidate_value=finetuned_success,
                )
            ),
        }
    if finetuned_summary is not None:
        quantitative_evidence[
            "modular_distilled_plus_rl_finetuning_vs_teacher"
        ] = _metric_delta(finetuned_summary, teacher_summary)
        if baseline_summary is not None:
            quantitative_evidence[
                "modular_distilled_plus_rl_finetuning_vs_modular_rl_from_scratch"
            ] = _metric_delta(finetuned_summary, baseline_summary)

    return {
        "question": "Can modular architecture express an effective policy?",
        "answer": answer,
        "rationale": rationale,
        "primary_metric": "scenario_success_rate",
        "teacher": teacher_summary,
        "modular_rl_from_scratch": baseline_summary,
        "modular_distilled": distilled_summary,
        "modular_distilled_plus_rl_finetuning": finetuned_summary,
        "best_student_condition": best_condition_name,
        "quantitative_evidence": quantitative_evidence,
    }

__all__ = [name for name in globals() if not name.startswith("__")]
