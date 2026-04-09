from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

from .ablations import PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES

SHELTER_ROLES: Sequence[str] = ("outside", "entrance", "inside", "deep")


@dataclass(frozen=True)
class BehaviorCheckSpec:
    name: str
    description: str
    expected: str


@dataclass(frozen=True)
class BehaviorCheckResult:
    name: str
    description: str
    expected: str
    passed: bool
    value: Any


@dataclass
class BehavioralEpisodeScore:
    episode: int
    seed: int
    scenario: str
    objective: str
    success: bool
    checks: Dict[str, BehaviorCheckResult]
    behavior_metrics: Dict[str, Any]
    failures: List[str]


@dataclass
class EpisodeStats:
    episode: int
    seed: int
    training: bool
    scenario: str | None
    total_reward: float
    steps: int
    food_eaten: int
    sleep_events: int
    shelter_entries: int
    alert_events: int
    predator_contacts: int
    predator_sightings: int
    predator_escapes: int
    night_ticks: int
    night_shelter_ticks: int
    night_still_ticks: int
    night_role_ticks: Dict[str, int]
    night_shelter_occupancy_rate: float
    night_stillness_rate: float
    night_role_distribution: Dict[str, float]
    predator_response_events: int
    mean_predator_response_latency: float
    mean_sleep_debt: float
    food_distance_delta: float
    shelter_distance_delta: float
    final_hunger: float
    final_fatigue: float
    final_sleep_debt: float
    final_health: float
    alive: bool
    reward_component_totals: Dict[str, float]
    predator_state_ticks: Dict[str, int]
    predator_mode_transitions: int
    dominant_predator_state: str
    reflex_usage_rate: float = 0.0
    final_reflex_override_rate: float = 0.0
    mean_reflex_dominance: float = 0.0
    module_reflex_usage_rates: Dict[str, float] = field(default_factory=dict)
    module_reflex_override_rates: Dict[str, float] = field(default_factory=dict)
    module_reflex_dominance: Dict[str, float] = field(default_factory=dict)
    module_contribution_share: Dict[str, float] = field(default_factory=dict)
    dominant_module: str = ""
    dominant_module_share: float = 0.0
    effective_module_count: float = 0.0
    module_agreement_rate: float = 0.0
    module_disagreement_rate: float = 0.0


@dataclass
class EpisodeMetricAccumulator:
    reward_component_names: Sequence[str]
    predator_states: Sequence[str]
    night_ticks: int = 0
    night_shelter_ticks: int = 0
    night_still_ticks: int = 0
    night_role_ticks: Dict[str, int] = field(default_factory=dict)
    predator_response_latencies: List[int] = field(default_factory=list)
    active_predator_response: Dict[str, int] | None = None
    reward_component_totals: Dict[str, float] = field(default_factory=dict)
    predator_state_ticks: Dict[str, int] = field(default_factory=dict)
    sleep_debt_samples: List[float] = field(default_factory=list)
    predator_mode_transitions: int = 0
    initial_food_dist: int | None = None
    final_food_dist: int | None = None
    initial_shelter_dist: int | None = None
    final_shelter_dist: int | None = None
    decision_steps: int = 0
    reflex_steps: int = 0
    final_reflex_override_steps: int = 0
    module_reflex_usage_steps: Dict[str, int] = field(default_factory=dict)
    module_reflex_override_steps: Dict[str, int] = field(default_factory=dict)
    module_reflex_dominance_sums: Dict[str, float] = field(default_factory=dict)
    module_contribution_share_sums: Dict[str, float] = field(default_factory=dict)
    dominant_module_counts: Dict[str, int] = field(default_factory=dict)
    dominant_module_share_sum: float = 0.0
    effective_module_count_sum: float = 0.0
    module_agreement_rate_sum: float = 0.0
    module_disagreement_rate_sum: float = 0.0

    def __post_init__(self) -> None:
        """
        Initialize missing accumulator maps with zeroed defaults.
        
        If a map is empty, populate it with keys taken from the corresponding configuration attributes and zero values:
        - `reward_component_totals` ← keys from `reward_component_names` with `0.0`
        - `predator_state_ticks` ← keys from `predator_states` with `0`
        - `night_role_ticks` ← keys from `SHELTER_ROLES` with `0`
        - `module_reflex_usage_steps` ← keys from `REFLEX_MODULE_NAMES` with `0`
        - `module_reflex_override_steps` ← keys from `REFLEX_MODULE_NAMES` with `0`
        - `module_reflex_dominance_sums` ← keys from `REFLEX_MODULE_NAMES` with `0.0`
        """
        if not self.reward_component_totals:
            self.reward_component_totals = {
                name: 0.0 for name in self.reward_component_names
            }
        if not self.predator_state_ticks:
            self.predator_state_ticks = {
                name: 0 for name in self.predator_states
            }
        if not self.night_role_ticks:
            self.night_role_ticks = {
                name: 0 for name in SHELTER_ROLES
            }
        for name in REFLEX_MODULE_NAMES:
            self.module_reflex_usage_steps.setdefault(name, 0)
            self.module_reflex_override_steps.setdefault(name, 0)
            self.module_reflex_dominance_sums.setdefault(name, 0.0)
        for name in PROPOSAL_SOURCE_NAMES:
            self.module_contribution_share_sums.setdefault(name, 0.0)
            self.dominant_module_counts.setdefault(name, 0)

    def record_decision(self, decision: object) -> None:
        """
        Update accumulator counters from a single decision, recording per-decision and per-module reflex usage, overrides, and dominance sums.
        
        Parameters:
            decision (object): Decision-like object with an iterable attribute `module_results` (each with `name`, `reflex_applied`, `module_reflex_override`, and `module_reflex_dominance` attributes) and an optional boolean attribute `final_reflex_override`. Only modules whose names exist in the accumulator's module maps are updated.
        
        """
        self.decision_steps += 1
        module_results = list(getattr(decision, "module_results", []))
        any_reflex = False
        for result in module_results:
            module_name = str(getattr(result, "name", ""))
            if module_name not in self.module_reflex_usage_steps:
                continue
            reflex_applied = bool(getattr(result, "reflex_applied", False))
            module_override = bool(getattr(result, "module_reflex_override", False))
            dominance = float(getattr(result, "module_reflex_dominance", 0.0))
            if reflex_applied:
                any_reflex = True
                self.module_reflex_usage_steps[module_name] += 1
            if module_override:
                self.module_reflex_override_steps[module_name] += 1
            self.module_reflex_dominance_sums[module_name] += dominance
        if any_reflex:
            self.reflex_steps += 1
        if bool(getattr(decision, "final_reflex_override", False)):
            self.final_reflex_override_steps += 1
        arbitration = getattr(decision, "arbitration_decision", None)
        if arbitration is None:
            return
        contribution_share = getattr(arbitration, "module_contribution_share", {})
        if isinstance(contribution_share, dict):
            for module_name in self.module_contribution_share_sums:
                self.module_contribution_share_sums[module_name] += float(
                    contribution_share.get(module_name, 0.0)
                )
        dominant_module = str(getattr(arbitration, "dominant_module", ""))
        if dominant_module in self.dominant_module_counts:
            self.dominant_module_counts[dominant_module] += 1
        self.dominant_module_share_sum += float(
            getattr(arbitration, "dominant_module_share", 0.0)
        )
        self.effective_module_count_sum += float(
            getattr(arbitration, "effective_module_count", 0.0)
        )
        self.module_agreement_rate_sum += float(
            getattr(arbitration, "module_agreement_rate", 0.0)
        )
        self.module_disagreement_rate_sum += float(
            getattr(arbitration, "module_disagreement_rate", 0.0)
        )

    def record_transition(
        self,
        *,
        step: int,
        observation_meta: Dict[str, object],
        next_meta: Dict[str, object],
        info: Dict[str, object],
        state: object,
        predator_state_before: str,
        predator_state: str,
    ) -> None:
        """
        Update internal accumulators with data from a single environment transition.
        
        This updates reward component totals, distance start/end samples, predator-mode transition counts and occupancy ticks, sleep-debt samples, night-related tick counters (including per-role counts, shelter occupancy, and stillness), and predator response event windowing/latency recording.
        
        Parameters:
            step (int): Current step index in the episode.
            observation_meta (Dict[str, object]): Metadata for the current observation. Expected keys:
                - "food_dist" (int-like)
                - "shelter_dist" (int-like)
                - "predator_visible" (truthy/falsey)
            next_meta (Dict[str, object]): Metadata for the next observation. Expected keys:
                - "food_dist" (int-like)
                - "shelter_dist" (int-like)
                - "predator_visible" (truthy/falsey)
                - "predator_dist" (int-like)
                - "night" (truthy/falsey)
                - "shelter_role" (role label)
                - "on_shelter" (truthy/falsey)
            info (Dict[str, object]): Transition info. Expected keys:
                - "reward_components" (mapping of component name to numeric value)
                - "predator_contact" (truthy/falsey)
            state (object): Agent state object providing at least `sleep_debt` and optional
                `last_move_dx`, `last_move_dy` attributes for stillness detection.
            predator_state_before (str): Predator mode label before this transition.
            predator_state (str): Predator mode label after this transition.
        """
        reward_components = info["reward_components"]
        for name, value in reward_components.items():
            self.reward_component_totals[name] += float(value)

        if self.initial_food_dist is None:
            self.initial_food_dist = int(observation_meta["food_dist"])
        if self.initial_shelter_dist is None:
            self.initial_shelter_dist = int(observation_meta["shelter_dist"])
        self.final_food_dist = int(next_meta["food_dist"])
        self.final_shelter_dist = int(next_meta["shelter_dist"])

        if predator_state_before != predator_state:
            self.predator_mode_transitions += 1

        if predator_state not in self.predator_state_ticks:
            self.predator_state_ticks[predator_state] = 0
        self.predator_state_ticks[predator_state] += 1
        self.sleep_debt_samples.append(float(state.sleep_debt))

        if bool(next_meta["night"]):
            self.night_ticks += 1
            role = str(next_meta["shelter_role"])
            if role not in self.night_role_ticks:
                self.night_role_ticks[role] = 0
            self.night_role_ticks[role] += 1
            if bool(next_meta["on_shelter"]):
                self.night_shelter_ticks += 1
            if getattr(state, "last_move_dx", 0) == 0 and getattr(state, "last_move_dy", 0) == 0:
                self.night_still_ticks += 1

        if self.active_predator_response is None and (
            (bool(next_meta["predator_visible"]) and not bool(observation_meta["predator_visible"]))
            or bool(info["predator_contact"])
        ):
            self.active_predator_response = {
                "start_step": step,
                "start_distance": int(next_meta["predator_dist"]),
            }

        if self.active_predator_response is not None and (
            str(next_meta["shelter_role"]) in {"inside", "deep"}
            or int(next_meta["predator_dist"]) >= self.active_predator_response["start_distance"] + 2
        ):
            self.predator_response_latencies.append(step - self.active_predator_response["start_step"])
            self.active_predator_response = None

    def snapshot(self) -> Dict[str, object]:
        """
        Compute derived episode metrics and aggregated counts as a snapshot dictionary.
        
        Returns:
            snapshot (Dict[str, object]): Mapping of derived episode statistics including:
                - night_ticks: total number of night timesteps recorded.
                - night_shelter_ticks: number of night timesteps spent on a shelter.
                - night_still_ticks: number of night timesteps with no movement.
                - night_role_ticks: per-role integer counts of night timesteps.
                - night_role_distribution: per-role fraction of night timesteps (sums to <= 1.0).
                - predator_state_ticks: integer counts of timesteps spent in each predator state.
                - predator_state_occupancy: per-state fraction of predator-state occupancy.
                - predator_mode_transitions: count of predator-mode changes observed.
                - dominant_predator_state: predator state with the highest observed occupancy (defaults to "PATROL" if none).
                - predator_response_events: number of recorded predator response events.
                - mean_predator_response_latency: mean latency (in timesteps) for completed predator response events, 0.0 if none.
                - food_distance_delta: initial minus final food distance (float; uses 0 when missing).
                - shelter_distance_delta: initial minus final shelter distance (float; uses 0 when missing).
                - mean_sleep_debt: mean of collected sleep debt samples, 0.0 if none.
                - reward_component_totals: mapping of configured reward component names to their accumulated totals (floats).
        """
        night_role_distribution = _normalize_counts(self.night_role_ticks, total=self.night_ticks)
        predator_total = sum(self.predator_state_ticks.values())
        predator_state_occupancy = _normalize_counts(
            self.predator_state_ticks,
            total=predator_total,
        )
        dominant_predator_state = (
            max(self.predator_state_ticks, key=self.predator_state_ticks.get)
            if self.predator_state_ticks
            else "PATROL"
        )
        decision_total = max(1, int(self.decision_steps))
        module_reflex_usage_rates = {
            name: float(self.module_reflex_usage_steps[name] / decision_total)
            for name in self.module_reflex_usage_steps
        }
        module_reflex_override_rates = {
            name: float(self.module_reflex_override_steps[name] / decision_total)
            for name in self.module_reflex_override_steps
        }
        module_reflex_dominance = {
            name: float(self.module_reflex_dominance_sums[name] / decision_total)
            for name in self.module_reflex_dominance_sums
        }
        module_contribution_share = {
            name: float(self.module_contribution_share_sums[name] / decision_total)
            for name in self.module_contribution_share_sums
        }
        dominant_module_distribution = {
            name: float(self.dominant_module_counts[name] / decision_total)
            for name in self.dominant_module_counts
        }
        dominant_module = (
            max(self.dominant_module_counts, key=self.dominant_module_counts.get)
            if self.dominant_module_counts
            else ""
        )
        return {
            "night_ticks": int(self.night_ticks),
            "night_shelter_ticks": int(self.night_shelter_ticks),
            "night_still_ticks": int(self.night_still_ticks),
            "night_role_ticks": {k: int(v) for k, v in self.night_role_ticks.items()},
            "night_role_distribution": night_role_distribution,
            "predator_state_ticks": {k: int(v) for k, v in self.predator_state_ticks.items()},
            "predator_state_occupancy": predator_state_occupancy,
            "predator_mode_transitions": int(self.predator_mode_transitions),
            "dominant_predator_state": dominant_predator_state,
            "predator_response_events": len(self.predator_response_latencies),
            "mean_predator_response_latency": (
                float(mean(self.predator_response_latencies))
                if self.predator_response_latencies
                else 0.0
            ),
            "food_distance_delta": float(
                (self.initial_food_dist or 0) - (self.final_food_dist or 0)
            ),
            "shelter_distance_delta": float(
                (self.initial_shelter_dist or 0) - (self.final_shelter_dist or 0)
            ),
            "mean_sleep_debt": (
                float(mean(self.sleep_debt_samples))
                if self.sleep_debt_samples
                else 0.0
            ),
            "reward_component_totals": {
                name: float(self.reward_component_totals[name])
                for name in self.reward_component_names
            },
            "reflex_usage_rate": float(self.reflex_steps / decision_total),
            "final_reflex_override_rate": float(
                self.final_reflex_override_steps / decision_total
            ),
            "module_reflex_usage_rates": module_reflex_usage_rates,
            "module_reflex_override_rates": module_reflex_override_rates,
            "module_reflex_dominance": module_reflex_dominance,
            "mean_reflex_dominance": (
                float(mean(module_reflex_dominance.values()))
                if module_reflex_dominance
                else 0.0
            ),
            "module_contribution_share": module_contribution_share,
            "dominant_module": dominant_module,
            "dominant_module_distribution": dominant_module_distribution,
            "dominant_module_share": float(
                self.dominant_module_share_sum / decision_total
            ),
            "effective_module_count": float(
                self.effective_module_count_sum / decision_total
            ),
            "module_agreement_rate": float(
                self.module_agreement_rate_sum / decision_total
            ),
            "module_disagreement_rate": float(
                self.module_disagreement_rate_sum / decision_total
            ),
        }

    def finalize(
        self,
        *,
        episode: int,
        seed: int,
        training: bool,
        scenario: str | None,
        total_reward: float,
        state: object,
    ) -> EpisodeStats:
        """
        Build an EpisodeStats summary combining accumulated metrics with episode context and the final environment state.
        
        Parameters:
            episode (int): Episode index.
            seed (int): Random seed used for the episode.
            training (bool): Whether the episode was collected in training mode.
            scenario (str | None): Optional scenario identifier.
            total_reward (float): Cumulative reward for the episode.
            state (object): Final environment state exposing numeric attributes used for summary:
                `steps_alive`, `food_eaten`, `sleep_events`, `shelter_entries`,
                `alert_events`, `predator_contacts`, `predator_sightings`, `predator_escapes`,
                `hunger`, `fatigue`, `sleep_debt`, and `health`.
        
        Returns:
            EpisodeStats: Consolidated episode summary containing:
              - episode metadata (episode, seed, training, scenario, total_reward)
              - event counters (steps, food/sleep/shelter/predator counts, alert events)
              - night summaries (ticks, per-role ticks and distribution, shelter occupancy and stillness rates)
              - predator statistics (state tick counts, mode transition count, dominant state, response event count and mean latency)
              - mean sleep debt and distance deltas (food and shelter)
              - final physiological metrics (hunger, fatigue, sleep debt, health) and survival flag
              - reward component totals
              - reflex metrics (overall reflex usage rate, final reflex override rate, mean reflex dominance)
              - per-module reflex rates and dominance maps.
        """
        snapshot = self.snapshot()
        night_ticks = max(0, int(snapshot["night_ticks"]))
        night_shelter_occupancy_rate = (
            self.night_shelter_ticks / night_ticks if night_ticks else 0.0
        )
        night_stillness_rate = (
            self.night_still_ticks / night_ticks if night_ticks else 0.0
        )
        return EpisodeStats(
            episode=episode,
            seed=seed,
            training=training,
            scenario=scenario,
            total_reward=float(total_reward),
            steps=int(state.steps_alive),
            food_eaten=int(state.food_eaten),
            sleep_events=int(state.sleep_events),
            shelter_entries=int(state.shelter_entries),
            alert_events=int(state.alert_events),
            predator_contacts=int(state.predator_contacts),
            predator_sightings=int(state.predator_sightings),
            predator_escapes=int(state.predator_escapes),
            night_ticks=night_ticks,
            night_shelter_ticks=int(self.night_shelter_ticks),
            night_still_ticks=int(self.night_still_ticks),
            night_role_ticks={role: int(self.night_role_ticks[role]) for role in SHELTER_ROLES},
            night_shelter_occupancy_rate=float(night_shelter_occupancy_rate),
            night_stillness_rate=float(night_stillness_rate),
            night_role_distribution={
                role: float(snapshot["night_role_distribution"][role])
                for role in SHELTER_ROLES
            },
            predator_response_events=len(self.predator_response_latencies),
            mean_predator_response_latency=float(snapshot["mean_predator_response_latency"]),
            mean_sleep_debt=float(snapshot["mean_sleep_debt"]),
            food_distance_delta=float(snapshot["food_distance_delta"]),
            shelter_distance_delta=float(snapshot["shelter_distance_delta"]),
            final_hunger=float(state.hunger),
            final_fatigue=float(state.fatigue),
            final_sleep_debt=float(state.sleep_debt),
            final_health=float(state.health),
            alive=bool(state.health > 0.0),
            reward_component_totals={
                name: float(self.reward_component_totals[name])
                for name in self.reward_component_names
            },
            predator_state_ticks={
                name: int(self.predator_state_ticks[name])
                for name in self.predator_states
            },
            predator_mode_transitions=int(snapshot["predator_mode_transitions"]),
            dominant_predator_state=str(snapshot["dominant_predator_state"]),
            reflex_usage_rate=float(snapshot["reflex_usage_rate"]),
            final_reflex_override_rate=float(snapshot["final_reflex_override_rate"]),
            mean_reflex_dominance=float(snapshot["mean_reflex_dominance"]),
            module_reflex_usage_rates={
                name: float(snapshot["module_reflex_usage_rates"][name])
                for name in REFLEX_MODULE_NAMES
            },
            module_reflex_override_rates={
                name: float(snapshot["module_reflex_override_rates"][name])
                for name in REFLEX_MODULE_NAMES
            },
            module_reflex_dominance={
                name: float(snapshot["module_reflex_dominance"][name])
                for name in REFLEX_MODULE_NAMES
            },
            module_contribution_share={
                name: float(snapshot["module_contribution_share"][name])
                for name in PROPOSAL_SOURCE_NAMES
            },
            dominant_module=str(snapshot["dominant_module"]),
            dominant_module_share=float(snapshot["dominant_module_share"]),
            effective_module_count=float(snapshot["effective_module_count"]),
            module_agreement_rate=float(snapshot["module_agreement_rate"]),
            module_disagreement_rate=float(snapshot["module_disagreement_rate"]),
        )


def _normalize_counts(counts: Dict[str, int], *, total: int) -> Dict[str, float]:
    if total <= 0:
        return {name: 0.0 for name in counts}
    return {
        name: float(value / total)
        for name, value in counts.items()
    }


def aggregate_episode_stats(history: List[EpisodeStats]) -> Dict[str, object]:
    """
    Compute suite-level aggregated statistics from a sequence of EpisodeStats.
    
    Parameters:
        history (List[EpisodeStats]): Per-episode summaries to aggregate.
    
    Returns:
        dict: Aggregate metrics including episode count and means for rewards, events, night metrics,
        predator statistics, distance and sleep deltas, survival rate, per-reward-component means,
        per-predator-state means and occupancy rates, dominant predator state, mean reflex rates
        (overall and per-module), and an "episodes_detail" list with asdict() for each EpisodeStats.
    """
    reward_component_names = (
        list(history[0].reward_component_totals.keys())
        if history
        else []
    )
    predator_states = (
        list(history[0].predator_state_ticks.keys())
        if history
        else []
    )
    reward_component_means = {
        name: mean(stats.reward_component_totals[name] for stats in history) if history else 0.0
        for name in reward_component_names
    }
    predator_state_means = {
        name: mean(stats.predator_state_ticks[name] for stats in history) if history else 0.0
        for name in predator_states
    }
    night_role_tick_means = {
        role: mean(stats.night_role_ticks[role] for stats in history) if history else 0.0
        for role in SHELTER_ROLES
    }
    night_role_distribution = {
        role: mean(stats.night_role_distribution[role] for stats in history) if history else 0.0
        for role in SHELTER_ROLES
    }
    steps_mean = mean(stats.steps for stats in history) if history else 1.0
    predator_state_rates = {
        name: (predator_state_means[name] / steps_mean if steps_mean else 0.0)
        for name in predator_state_means
    }
    module_reflex_usage_rate_means = {
        name: (
            mean(stats.module_reflex_usage_rates.get(name, 0.0) for stats in history)
            if history
            else 0.0
        )
        for name in REFLEX_MODULE_NAMES
    }
    module_reflex_override_rate_means = {
        name: (
            mean(stats.module_reflex_override_rates.get(name, 0.0) for stats in history)
            if history
            else 0.0
        )
        for name in REFLEX_MODULE_NAMES
    }
    module_reflex_dominance_means = {
        name: (
            mean(stats.module_reflex_dominance.get(name, 0.0) for stats in history)
            if history
            else 0.0
        )
        for name in REFLEX_MODULE_NAMES
    }
    module_contribution_share_means = {
        name: (
            mean(stats.module_contribution_share.get(name, 0.0) for stats in history)
            if history
            else 0.0
        )
        for name in PROPOSAL_SOURCE_NAMES
    }
    dominant_module_counter = Counter(
        stats.dominant_module
        for stats in history
        if stats.dominant_module in PROPOSAL_SOURCE_NAMES
    )
    dominant_module_distribution = {
        name: (
            float(dominant_module_counter.get(name, 0) / len(history))
            if history
            else 0.0
        )
        for name in PROPOSAL_SOURCE_NAMES
    }
    dominant_module = (
        max(dominant_module_counter, key=dominant_module_counter.get)
        if dominant_module_counter
        else ""
    )
    dominant_predator_state = (
        max(predator_state_means, key=predator_state_means.get)
        if predator_state_means
        else "PATROL"
    )
    return {
        "episodes": len(history),
        "mean_reward": mean(stats.total_reward for stats in history) if history else 0.0,
        "mean_food": mean(stats.food_eaten for stats in history) if history else 0.0,
        "mean_sleep": mean(stats.sleep_events for stats in history) if history else 0.0,
        "mean_predator_contacts": (
            mean(stats.predator_contacts for stats in history) if history else 0.0
        ),
        "mean_predator_escapes": (
            mean(stats.predator_escapes for stats in history) if history else 0.0
        ),
        "mean_night_shelter_occupancy_rate": (
            mean(stats.night_shelter_occupancy_rate for stats in history) if history else 0.0
        ),
        "mean_night_stillness_rate": (
            mean(stats.night_stillness_rate for stats in history) if history else 0.0
        ),
        "mean_night_role_ticks": night_role_tick_means,
        "mean_night_role_distribution": night_role_distribution,
        "mean_predator_response_events": (
            mean(stats.predator_response_events for stats in history) if history else 0.0
        ),
        "mean_predator_response_latency": (
            mean(stats.mean_predator_response_latency for stats in history) if history else 0.0
        ),
        "mean_sleep_debt": mean(stats.mean_sleep_debt for stats in history) if history else 0.0,
        "mean_food_distance_delta": (
            mean(stats.food_distance_delta for stats in history) if history else 0.0
        ),
        "mean_shelter_distance_delta": (
            mean(stats.shelter_distance_delta for stats in history) if history else 0.0
        ),
        "survival_rate": mean(1.0 if stats.alive else 0.0 for stats in history) if history else 0.0,
        "mean_reward_components": reward_component_means,
        "mean_predator_state_ticks": predator_state_means,
        "mean_predator_state_occupancy": predator_state_rates,
        "mean_predator_mode_transitions": (
            mean(stats.predator_mode_transitions for stats in history) if history else 0.0
        ),
        "dominant_predator_state": dominant_predator_state,
        "mean_reflex_usage_rate": (
            mean(stats.reflex_usage_rate for stats in history) if history else 0.0
        ),
        "mean_final_reflex_override_rate": (
            mean(stats.final_reflex_override_rate for stats in history) if history else 0.0
        ),
        "mean_reflex_dominance": (
            mean(stats.mean_reflex_dominance for stats in history) if history else 0.0
        ),
        "mean_module_reflex_usage_rate": module_reflex_usage_rate_means,
        "mean_module_reflex_override_rate": module_reflex_override_rate_means,
        "mean_module_reflex_dominance": module_reflex_dominance_means,
        "mean_module_contribution_share": module_contribution_share_means,
        "dominant_module": dominant_module,
        "dominant_module_distribution": dominant_module_distribution,
        "mean_dominant_module_share": (
            mean(stats.dominant_module_share for stats in history) if history else 0.0
        ),
        "mean_effective_module_count": (
            mean(stats.effective_module_count for stats in history) if history else 0.0
        ),
        "mean_module_agreement_rate": (
            mean(stats.module_agreement_rate for stats in history) if history else 0.0
        ),
        "mean_module_disagreement_rate": (
            mean(stats.module_disagreement_rate for stats in history) if history else 0.0
        ),
        "episodes_detail": [asdict(stats) for stats in history],
    }


def build_behavior_check(spec: BehaviorCheckSpec, *, passed: bool, value: Any) -> BehaviorCheckResult:
    """
    Create a BehaviorCheckResult from a BehaviorCheckSpec and an observed outcome.
    
    Parameters:
        spec (BehaviorCheckSpec): Specification of the check (name, description, expected outcome).
        passed (bool): Whether the observed outcome satisfies the spec; will be stored as `bool`.
        value (Any): Observed value associated with the check result.
    
    Returns:
        BehaviorCheckResult: Result populated with `name`, `description`, and `expected` from `spec`,
        and with `passed` coerced to `bool` and `value` set to the provided observation.
    """
    return BehaviorCheckResult(
        name=spec.name,
        description=spec.description,
        expected=spec.expected,
        passed=bool(passed),
        value=value,
    )


def build_behavior_score(
    *,
    stats: EpisodeStats,
    objective: str,
    checks: Sequence[BehaviorCheckResult],
    behavior_metrics: Mapping[str, Any],
) -> BehavioralEpisodeScore:
    """
    Builds a BehavioralEpisodeScore for an episode from episode stats, check results, and behavior metrics.
    
    Parameters:
        stats (EpisodeStats): Source of episode identifier, seed, and scenario (used for the score's episode/seed/scenario).
        objective (str): The objective name associated with this behavior score.
        checks (Sequence[BehaviorCheckResult]): Sequence of check results; they are indexed by `name` into the score's `checks` map.
        behavior_metrics (Mapping[str, Any]): Arbitrary per-episode behavior metrics copied into the score.
    
    Returns:
        BehavioralEpisodeScore: Score populated with:
            - `episode` and `seed` from `stats`
            - `scenario` from `stats.scenario` or `"default"` if falsy
            - `objective` as provided
            - `success` set to `true` if there are no failed checks, `false` otherwise
            - `checks` as a mapping of check name → BehaviorCheckResult
            - `behavior_metrics` as a plain dict copy of the provided mapping
            - `failures` as a list of names for checks that did not pass
    """
    check_map = {
        check.name: check
        for check in checks
    }
    failures = [
        name
        for name, check in check_map.items()
        if not check.passed
    ]
    return BehavioralEpisodeScore(
        episode=stats.episode,
        seed=stats.seed,
        scenario=stats.scenario or "default",
        objective=objective,
        success=not failures,
        checks=check_map,
        behavior_metrics=dict(behavior_metrics),
        failures=failures,
    )


def aggregate_behavior_scores(
    scores: Sequence[BehavioralEpisodeScore],
    *,
    scenario: str,
    description: str,
    objective: str,
    check_specs: Sequence[BehaviorCheckSpec],
    diagnostic_focus: str | None = None,
    success_interpretation: str | None = None,
    failure_interpretation: str | None = None,
    budget_note: str | None = None,
    legacy_metrics: Mapping[str, Any] | None = None,
) -> Dict[str, object]:
    """
    Aggregate per-episode behavioral scores into a scenario-level summary.
    
    Produces a dictionary summarizing the provided BehavioralEpisodeScore records for a single scenario, including per-check pass rates and mean values, aggregated behavior metrics (mean or most-common), episode-level success rate, diagnostic outcome summaries, a sorted list of observed failures, and a detailed list of episode records.
    
    Parameters:
        scores (Sequence[BehavioralEpisodeScore]): Episode-level behavioral scores to aggregate.
        scenario (str): Scenario identifier for which the aggregation is produced.
        description (str): Human-readable description of the scenario or aggregation target.
        objective (str): Objective name associated with the aggregated scores.
        check_specs (Sequence[BehaviorCheckSpec]): Specifications for checks to include; used to provide descriptions and expected values for each check.
        legacy_metrics (Mapping[str, Any] | None): Optional additional metrics to include unchanged under the `legacy_metrics` key.
    
    Returns:
        Dict[str, object]: A mapping containing:
            - "scenario": the provided scenario identifier.
            - "description": the provided description.
            - "objective": the provided objective.
            - "episodes": number of episodes aggregated.
            - "success_rate": mean of episode success indicators (float).
            - "checks": dict mapping check name to { "description", "expected", "pass_rate", "mean_value" }.
            - "behavior_metrics": dict of aggregated behavior metrics (numeric mean or most-common value).
            - "diagnostics": dict with a primary outcome label, normalized outcome distribution, and selected episode-level diagnostic rates.
            - "failures": sorted list of unique failure names observed across episodes.
            - "episodes_detail": list of episode records as plain dicts.
            - "legacy_metrics": a plain dict copy of `legacy_metrics` or empty dict.
    """
    score_list = list(scores)
    check_index = {
        spec.name: spec
        for spec in check_specs
    }
    aggregated_checks: Dict[str, object] = {}
    for name, spec in check_index.items():
        results = [
            score.checks[name]
            for score in score_list
            if name in score.checks
        ]
        values = [result.value for result in results]
        aggregated_checks[name] = {
            "description": spec.description,
            "expected": spec.expected,
            "pass_rate": (
                mean(1.0 if result.passed else 0.0 for result in results)
                if results
                else 0.0
            ),
            "mean_value": _mean_like(values),
        }

    metric_names = sorted(
        {
            metric_name
            for score in score_list
            for metric_name in score.behavior_metrics
        }
    )
    behavior_metrics = {
        name: _aggregate_values(
            [score.behavior_metrics[name] for score in score_list if name in score.behavior_metrics]
        )
        for name in metric_names
    }
    outcome_labels = [
        str(score.behavior_metrics["outcome_band"])
        for score in score_list
        if "outcome_band" in score.behavior_metrics
    ]
    outcome_counter = Counter(outcome_labels)
    outcome_total = max(1, len(outcome_labels))
    partial_progress_values = [
        score.behavior_metrics["partial_progress"]
        for score in score_list
        if "partial_progress" in score.behavior_metrics
    ]
    died_without_contact_values = [
        score.behavior_metrics["died_without_contact"]
        for score in score_list
        if "died_without_contact" in score.behavior_metrics
    ]
    diagnostics = {
        "primary_outcome": (
            outcome_counter.most_common(1)[0][0]
            if outcome_counter
            else "not_available"
        ),
        "outcome_distribution": {
            label: float(count / outcome_total)
            for label, count in sorted(outcome_counter.items())
        },
        "partial_progress_rate": (
            _aggregate_values(partial_progress_values)
            if partial_progress_values
            else None
        ),
        "died_without_contact_rate": (
            _aggregate_values(died_without_contact_values)
            if died_without_contact_values
            else None
        ),
    }
    failures = sorted(
        {
            failure
            for score in score_list
            for failure in score.failures
        }
    )
    return {
        "scenario": scenario,
        "description": description,
        "objective": objective,
        "diagnostic_focus": diagnostic_focus or "",
        "success_interpretation": success_interpretation or "",
        "failure_interpretation": failure_interpretation or "",
        "budget_note": budget_note or "",
        "episodes": len(score_list),
        "success_rate": (
            mean(1.0 if score.success else 0.0 for score in score_list)
            if score_list
            else 0.0
        ),
        "checks": aggregated_checks,
        "behavior_metrics": behavior_metrics,
        "diagnostics": diagnostics,
        "failures": failures,
        "episodes_detail": [asdict(score) for score in score_list],
        "legacy_metrics": dict(legacy_metrics or {}),
    }


def summarize_behavior_suite(suite: Mapping[str, Mapping[str, Any]]) -> Dict[str, object]:
    """
    Summarizes per-scenario behavior aggregation results into suite-level counts and rates.
    
    Parameters:
        suite (Mapping[str, Mapping[str, Any]]): Mapping from scenario name to its aggregated data. Each scenario mapping is expected to include numeric "episodes" and "success_rate" keys and may include a "failures" iterable.
    
    Returns:
        Dict[str, object]: Summary dictionary with keys:
            - "scenario_count": number of scenarios in the suite.
            - "episode_count": total number of episodes across all scenarios.
            - "scenario_success_rate": mean of per-scenario indicators where a scenario's `success_rate` is at least 1.0 (1.0 if fully successful, 0.0 otherwise).
            - "episode_success_rate": overall fraction of episodes considered successful (episodes-weighted success rate).
            - "regressions": list of regression entries (one per scenario that reported failures), each of the form {"scenario": <name>, "failures": [<failure names>]}.
    """
    scenario_items = list(suite.items())
    total_episodes = sum(int(data.get("episodes", 0)) for _, data in scenario_items)
    successful_episodes = sum(
        float(data.get("success_rate", 0.0)) * int(data.get("episodes", 0))
        for _, data in scenario_items
    )
    regressions = [
        {
            "scenario": name,
            "failures": list(data.get("failures", [])),
        }
        for name, data in scenario_items
        if data.get("failures")
    ]
    return {
        "scenario_count": len(scenario_items),
        "episode_count": total_episodes,
        "scenario_success_rate": (
            mean(1.0 if float(data.get("success_rate", 0.0)) >= 1.0 else 0.0 for _, data in scenario_items)
            if scenario_items
            else 0.0
        ),
        "episode_success_rate": (
            float(successful_episodes / total_episodes)
            if total_episodes
            else 0.0
        ),
        "regressions": regressions,
    }


def flatten_behavior_rows(
    scores: Sequence[BehavioralEpisodeScore],
    *,
    reward_profile: str,
    scenario_map: str,
    simulation_seed: int,
    scenario_description: str,
    scenario_objective: str,
    scenario_focus: str,
    evaluation_map: str | None = None,
) -> List[Dict[str, object]]:
    """
    Convert a sequence of BehavioralEpisodeScore objects into a list of flat dictionaries suitable for tabular export.
    
    Each returned row contains fixed columns:
    - reward_profile, scenario_map, evaluation_map, simulation_seed, episode_seed, scenario, scenario_description,
      scenario_objective, scenario_focus, episode, success, failure_count, failures
    and additional columns for each behavior metric (prefixed with `metric_{name}`) and for each check:
    - `check_{name}_passed`, `check_{name}_value`, `check_{name}_expected`.
    
    Parameters:
        scores (Sequence[BehavioralEpisodeScore]): Episode-level behavioral scores to flatten.
        reward_profile (str): Identifier for the reward configuration applied to all rows.
        scenario_map (str): Map template actually used by the scenario represented in the row.
        simulation_seed (int): Global simulation seed applied to all rows.
        evaluation_map (str | None): Outer sweep/default map context for the run when relevant.
    
    Returns:
        List[Dict[str, object]]: A list of row dictionaries, one per input score.
    """
    rows: List[Dict[str, object]] = []
    for score in scores:
        row: Dict[str, object] = {
            "reward_profile": reward_profile,
            "scenario_map": scenario_map,
            "evaluation_map": evaluation_map,
            "simulation_seed": simulation_seed,
            "episode_seed": score.seed,
            "scenario": score.scenario,
            "scenario_description": scenario_description,
            "scenario_objective": scenario_objective,
            "scenario_focus": scenario_focus,
            "episode": score.episode,
            "success": bool(score.success),
            "failure_count": len(score.failures),
            "failures": ",".join(score.failures),
        }
        for metric_name, value in sorted(score.behavior_metrics.items()):
            row[f"metric_{metric_name}"] = value
        for check_name, result in sorted(score.checks.items()):
            row[f"check_{check_name}_passed"] = bool(result.passed)
            row[f"check_{check_name}_value"] = result.value
            row[f"check_{check_name}_expected"] = result.expected
        rows.append(row)
    return rows


def _aggregate_values(values: Sequence[Any]) -> Any:
    """
    Aggregate a sequence of values into a single representative value.
    
    For an empty sequence returns 0.0. If all values are numeric-like (ints, floats, or bools),
    returns their arithmetic mean as a float. Otherwise returns the most common string
    representation of the values.
    
    Parameters:
        values (Sequence[Any]): Sequence of values to aggregate.
    
    Returns:
        Any: The aggregated value: `0.0` for empty input, a `float` mean when numeric-like,
        or the most common stringified value otherwise.
    """
    if not values:
        return 0.0
    mean_value = _mean_like(values)
    if mean_value is not None:
        return mean_value
    counter = Counter(str(value) for value in values)
    return counter.most_common(1)[0][0]


def _mean_like(values: Sequence[Any]) -> float | None:
    """
    Compute the mean of the input sequence when every element is numeric-like.
    
    Returns:
        float mean of the values when all elements are instances of int, float, or bool; `0.0` if `values` is empty; `None` if any element is not numeric-like.
    """
    if not values:
        return 0.0
    if all(isinstance(value, (int, float, bool)) for value in values):
        return float(mean(float(value) for value in values))
    return None
