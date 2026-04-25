from __future__ import annotations

from .common import *


class SimulationRuntimeBudgetMixin:
    def _set_runtime_budget(
        self,
        *,
        episodes: int,
        evaluation_episodes: int,
        scenario_episodes: int | None = None,
        behavior_seeds: Sequence[int] | None = None,
        ablation_seeds: Sequence[int] | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        """
        Update the simulation's runtime budget by resolving and storing episode counts, step limits, seed lists, and checkpoint interval into the instance budget summary.
        
        Parameters:
            episodes (int): Number of training episodes to resolve into the budget.
            evaluation_episodes (int): Number of evaluation episodes to resolve into the budget.
            scenario_episodes (int | None): Number of episodes to run per scenario; if omitted, preserves existing value or defaults to 1.
            behavior_seeds (Sequence[int] | None): Sequence of integer seeds to use for behavior evaluations; if provided, replaces the budget's behavior seed list.
            ablation_seeds (Sequence[int] | None): Sequence of integer seeds to use for ablation evaluations; if provided, replaces the budget's ablation seed list.
            checkpoint_interval (int | None): Interval (in completed episodes) at which to capture checkpoint candidates; if provided, updates the resolved checkpoint interval.
        """
        budget = deepcopy(self.budget_summary)
        resolved = budget.setdefault("resolved", {})
        resolved["episodes"] = int(episodes)
        resolved["eval_episodes"] = int(evaluation_episodes)
        resolved["max_steps"] = int(self.max_steps)
        if scenario_episodes is not None:
            resolved["scenario_episodes"] = int(scenario_episodes)
        elif "scenario_episodes" not in resolved:
            resolved["scenario_episodes"] = 1
        if behavior_seeds is not None:
            resolved["behavior_seeds"] = [int(seed) for seed in behavior_seeds]
        if ablation_seeds is not None:
            resolved["ablation_seeds"] = [int(seed) for seed in ablation_seeds]
        if checkpoint_interval is not None:
            resolved["checkpoint_interval"] = int(checkpoint_interval)
        self.budget_summary = budget
        self.budget_profile_name = str(budget.get("profile", self.budget_profile_name))
        self.benchmark_strength = str(
            budget.get("benchmark_strength", self.benchmark_strength)
        )

    def set_runtime_budget(
        self,
        *,
        episodes: int,
        evaluation_episodes: int,
        scenario_episodes: int | None = None,
        behavior_seeds: Sequence[int] | None = None,
        ablation_seeds: Sequence[int] | None = None,
        checkpoint_interval: int | None = None,
    ) -> None:
        """Package-internal API for entrypoints to update runtime budget state."""
        self._set_runtime_budget(
            episodes=episodes,
            evaluation_episodes=evaluation_episodes,
            scenario_episodes=scenario_episodes,
            behavior_seeds=behavior_seeds,
            ablation_seeds=ablation_seeds,
            checkpoint_interval=checkpoint_interval,
        )

    @contextmanager
    def _temporary_brain(self, brain: SpiderBrain) -> Iterator[None]:
        """
        Temporarily replaces the instance's active brain with the provided brain for the duration of a context.
        
        The original brain is restored after the context exits, even if the context body raises.
        
        Parameters:
            brain (SpiderBrain): The brain instance to set as the active brain within the context.
        """
        original_brain = self.brain
        self.brain = brain
        try:
            yield
        finally:
            self.brain = original_brain

    def _evaluate_active_brain_summary(
        self,
        *,
        evaluation_episodes: int,
        evaluation_reflex_scale: float | None,
        episode_start: int = 0,
    ) -> Dict[str, object]:
        """
        Evaluate the currently active brain for a short evaluation and return a labeled evaluation summary.
        
        Parameters:
            evaluation_episodes (int): Number of evaluation episodes to run.
            evaluation_reflex_scale (float | None): Reflex scale to use for the evaluation; if `None`, uses the active brain's current reflex scale.
            episode_start (int): Episode index to start evaluation from.
        
        Returns:
            Dict[str, object]: Aggregated evaluation summary dictionary including evaluation metrics and metadata; the result is labeled with `eval_reflex_scale` (the effective reflex scale used) and `competence_type`.
        """
        pre_eval_reflex_scale = float(self.brain.current_reflex_scale)
        _, evaluation_history, _ = self._train_histories(
            episodes=0,
            evaluation_episodes=evaluation_episodes,
            render_last_evaluation=False,
            capture_evaluation_trace=False,
            debug_trace=False,
            episode_start=episode_start,
            evaluation_reflex_scale=evaluation_reflex_scale,
            preserve_training_metadata=True,
            training_regime_spec=None,
        )
        effective_scale = self._effective_reflex_scale(
            evaluation_reflex_scale
            if evaluation_reflex_scale is not None
            else pre_eval_reflex_scale
        )
        competence_type = competence_label_from_eval_reflex_scale(effective_scale)
        return self._label_evaluation_summary(
            self._aggregate_group(evaluation_history),
            eval_reflex_scale=effective_scale,
            competence_type=competence_type,
        )

__all__ = [name for name in globals() if not name.startswith("__")]
