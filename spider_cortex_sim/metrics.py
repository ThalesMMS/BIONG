from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, List, Sequence

from .ablations import PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES

SHELTER_ROLES: Sequence[str] = ("outside", "entrance", "inside", "deep")
PREDATOR_TYPE_NAMES: Sequence[str] = ("visual", "olfactory")
# Threats at or below this level are treated as resolved for predator-response timing.
PREDATOR_RESPONSE_END_THRESHOLD: float = 0.05
COMPETENCE_LABELS: Sequence[str] = ("self_sufficient", "scaffolded", "mixed")


def normalize_competence_label(label: str) -> str:
    """
    Validate and normalize a competence label to one of the allowed competence categories.
    
    Converts the input to a string and verifies it is one of COMPETENCE_LABELS; if not, raises ValueError listing the available labels.
    
    Parameters:
        label (str): The competence label to validate.
    
    Returns:
        str: The normalized competence label (one of COMPETENCE_LABELS).
    
    Raises:
        ValueError: If the normalized label is not one of the allowed COMPETENCE_LABELS.
    """
    normalized = str(label)
    if normalized not in COMPETENCE_LABELS:
        available = ", ".join(repr(item) for item in COMPETENCE_LABELS)
        raise ValueError(f"Invalid competence_label. Available labels: {available}.")
    return normalized


def competence_label_from_eval_reflex_scale(
    eval_reflex_scale: float | None,
) -> str:
    """
    Map an evaluation reflex scale to one of the competence labels.
    
    Parameters:
        eval_reflex_scale (float | None): Optional numeric scale indicating evaluation reflex strength.
    
    Returns:
        str: `"mixed"` if `eval_reflex_scale` is `None`, `"self_sufficient"` if the value equals `0.0`, otherwise `"scaffolded"`.
    """
    if eval_reflex_scale is None:
        return "mixed"
    return "self_sufficient" if float(eval_reflex_scale) == 0.0 else "scaffolded"


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
    predator_contacts_by_type: Dict[str, int] = field(default_factory=dict)
    predator_escapes_by_type: Dict[str, int] = field(default_factory=dict)
    predator_response_latency_by_type: Dict[str, float] = field(default_factory=dict)
    module_response_by_predator_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
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
    mean_module_credit_weights: Dict[str, float] = field(default_factory=dict)
    module_gradient_norm_means: Dict[str, float] = field(default_factory=dict)
    motor_slip_rate: float = 0.0
    mean_orientation_alignment: float = 0.0
    mean_terrain_difficulty: float = 0.0
    terrain_slip_rates: Dict[str, float] = field(default_factory=dict)


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
    predator_contacts_by_type: Dict[str, int] = field(default_factory=dict)
    predator_escapes_by_type: Dict[str, int] = field(default_factory=dict)
    predator_response_latencies_by_type: Dict[str, List[int]] = field(default_factory=dict)
    active_predator_responses_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
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
    module_response_by_predator_type_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    dominant_module_counts: Dict[str, int] = field(default_factory=dict)
    current_dominant_module: str = ""
    dominant_module_share_sum: float = 0.0
    effective_module_count_sum: float = 0.0
    module_agreement_rate_sum: float = 0.0
    module_disagreement_rate_sum: float = 0.0
    learning_steps: int = 0
    module_credit_weight_sums: Dict[str, float] = field(default_factory=dict)
    module_gradient_norm_sums: Dict[str, float] = field(default_factory=dict)
    motor_execution_steps: int = 0
    motor_slip_steps: int = 0
    orientation_alignment_samples: List[float] = field(default_factory=list)
    terrain_difficulty_samples: List[float] = field(default_factory=list)
    terrain_execution_counts: Dict[str, int] = field(default_factory=dict)
    terrain_slip_counts: Dict[str, int] = field(default_factory=dict)

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
        for predator_type in PREDATOR_TYPE_NAMES:
            self.predator_contacts_by_type.setdefault(predator_type, 0)
            self.predator_escapes_by_type.setdefault(predator_type, 0)
            self.predator_response_latencies_by_type.setdefault(predator_type, [])
            self.module_response_by_predator_type_counts.setdefault(
                predator_type,
                {name: 0 for name in PROPOSAL_SOURCE_NAMES},
            )
        for name in REFLEX_MODULE_NAMES:
            self.module_reflex_usage_steps.setdefault(name, 0)
            self.module_reflex_override_steps.setdefault(name, 0)
            self.module_reflex_dominance_sums.setdefault(name, 0.0)
        for name in PROPOSAL_SOURCE_NAMES:
            self.module_contribution_share_sums.setdefault(name, 0.0)
            self.dominant_module_counts.setdefault(name, 0)
            self.module_credit_weight_sums.setdefault(name, 0.0)
            self.module_gradient_norm_sums.setdefault(name, 0.0)
            for predator_type in PREDATOR_TYPE_NAMES:
                self.module_response_by_predator_type_counts[predator_type].setdefault(
                    name,
                    0,
                )

    def record_decision(self, decision: object) -> None:
        """
        Update accumulator counts based on a single decision, recording per-decision and per-module reflex, override, dominance, and arbitration statistics.
        
        Parameters:
            decision (object): Decision-like object with the following expected attributes:
                - module_results: iterable of result objects each exposing `name`, `reflex_applied`, `module_reflex_override`, and `module_reflex_dominance`.
                - final_reflex_override: optional boolean flag.
                - arbitration_decision: optional object that may provide `module_contribution_share` (mapping), `dominant_module`, `dominant_module_share`, `effective_module_count`, `module_agreement_rate`, and `module_disagreement_rate`.
            Only modules whose names appear in the accumulator's module maps are updated.
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
            self.current_dominant_module = ""
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
            self.current_dominant_module = dominant_module
        else:
            self.current_dominant_module = ""
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

    def record_learning(self, learn_stats: Mapping[str, object]) -> None:
        """
        Accumulate per-module learning-credit diagnostics from one learning step.

        Missing module entries count as zero for the step, so strategy-specific
        sparse maps still average cleanly over the whole episode.
        """
        self.learning_steps += 1
        for stats_key, target in [
            ("module_credit_weights", self.module_credit_weight_sums),
            ("module_gradient_norms", self.module_gradient_norm_sums),
        ]:
            values = learn_stats.get(stats_key, {})
            if isinstance(values, Mapping):
                for name, value in values.items():
                    module_name = str(name)
                    target.setdefault(module_name, 0.0)
                    target[module_name] += float(value)

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
        Update the accumulator with metrics observed during a single environment transition.
        
        This records reward-component totals; motor execution and slip metrics (orientation alignment, terrain difficulty, terrain slip counts); initial and final food/shelter distances; attribution of module responses to the current dominant module for a dominant predator type; predator mode transition counts and per-mode occupancy ticks; sleep-debt samples; night-related ticks (total night ticks, per-role ticks, shelter occupancy ticks, and stillness ticks); global and per-predator-type predator response event windowing and latencies; predator contacts and escapes by predator type; and related per-type module response counts.
        
        Parameters:
            step (int): Current step index in the episode.
            observation_meta (Dict[str, object]): Metadata for the current observation (e.g., "food_dist", "shelter_dist", "predator_visible").
            next_meta (Dict[str, object]): Metadata for the next observation (e.g., "food_dist", "shelter_dist", "predator_visible", "diagnostic", "night", "shelter_role", "on_shelter").
            info (Dict[str, object]): Transition info (e.g., "reward_components", "predator_contact", "motor_slip", "motor_noise_applied", "predator_escape").
            state (object): Agent state object containing at least `sleep_debt` and optional `last_move_dx`, `last_move_dy`.
            predator_state_before (str): Predator mode label before this transition.
            predator_state (str): Predator mode label after this transition.
        
        Raises:
            ValueError: If `next_meta["diagnostic"]` is present but not a mapping when a predator-response window needs diagnostic data.
        """
        reward_components = info["reward_components"]
        for name, value in reward_components.items():
            self.reward_component_totals[name] += float(value)

        motor_slip = info.get("motor_slip", {})
        has_motor_execution_info = (
            "motor_slip" in info
            or "motor_execution" in info
            or "motor_execution_components" in info
        )
        if has_motor_execution_info and isinstance(motor_slip, Mapping):
            self.motor_execution_steps += 1
            motor_slip_occurred = bool(
                motor_slip.get("occurred", info.get("motor_noise_applied", False))
            )
            if motor_slip_occurred:
                self.motor_slip_steps += 1
            components = info.get("motor_execution_components", {})
            if not isinstance(components, Mapping):
                components = motor_slip.get("components", {})
            if isinstance(components, Mapping):
                if "orientation_alignment" in components:
                    self.orientation_alignment_samples.append(
                        float(components["orientation_alignment"])
                    )
                if "terrain_difficulty" in components:
                    self.terrain_difficulty_samples.append(
                        float(components["terrain_difficulty"])
                    )
            terrain = str(motor_slip.get("terrain", "unknown"))
            self.terrain_execution_counts[terrain] = (
                self.terrain_execution_counts.get(terrain, 0) + 1
            )
            if motor_slip_occurred:
                self.terrain_slip_counts[terrain] = (
                    self.terrain_slip_counts.get(terrain, 0) + 1
                )

        if self.initial_food_dist is None:
            self.initial_food_dist = int(observation_meta["food_dist"])
        if self.initial_shelter_dist is None:
            self.initial_shelter_dist = int(observation_meta["shelter_dist"])
        self.final_food_dist = int(next_meta["food_dist"])
        self.final_shelter_dist = int(next_meta["shelter_dist"])
        dominant_predator_type = _dominant_predator_type(next_meta)
        if dominant_predator_type and self.current_dominant_module in PROPOSAL_SOURCE_NAMES:
            threat_before = _predator_type_threat(observation_meta, dominant_predator_type)
            threat_after = _predator_type_threat(next_meta, dominant_predator_type)
            if threat_before > 0.0 or threat_after > 0.0 or bool(info["predator_contact"]):
                self.module_response_by_predator_type_counts[dominant_predator_type][
                    self.current_dominant_module
                ] += 1

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
            next_diagnostic = next_meta["diagnostic"]
            if not isinstance(next_diagnostic, Mapping):
                raise ValueError("next_meta['diagnostic'] must be a mapping")
            predator_dist = int(next_diagnostic["diagnostic_predator_dist"])
            self.active_predator_response = {
                "start_step": step,
                "start_distance": predator_dist,
            }

        for predator_type in PREDATOR_TYPE_NAMES:
            threat_before = _predator_type_threat(observation_meta, predator_type)
            threat_after = _predator_type_threat(next_meta, predator_type)
            start_triggered = (
                bool(info["predator_contact"]) and dominant_predator_type == predator_type
            ) or (threat_after > 0.0 and threat_before <= 0.0)
            if predator_type not in self.active_predator_responses_by_type and start_triggered:
                self.active_predator_responses_by_type[predator_type] = {
                    "start_step": int(step),
                    "start_distance": int(
                        _diagnostic_predator_distance_for_type(
                            next_meta,
                            predator_type,
                            state=state,
                        )
                    ),
                    "start_threat": float(threat_after),
                }

        if self.active_predator_response is not None:
            next_diagnostic = next_meta["diagnostic"]
            if not isinstance(next_diagnostic, Mapping):
                raise ValueError("next_meta['diagnostic'] must be a mapping")
            predator_dist = int(next_diagnostic["diagnostic_predator_dist"])
            if (
                str(next_meta["shelter_role"]) in {"inside", "deep"}
                or predator_dist >= self.active_predator_response["start_distance"] + 2
            ):
                self.predator_response_latencies.append(step - self.active_predator_response["start_step"])
                self.active_predator_response = None

        preserved_escape_type = ""
        for predator_type, active_response in list(self.active_predator_responses_by_type.items()):
            threat_after = _predator_type_threat(next_meta, predator_type)
            predator_dist = _diagnostic_predator_distance_for_type(
                next_meta,
                predator_type,
                state=state,
            )
            if (
                str(next_meta["shelter_role"]) in {"inside", "deep"}
                or threat_after <= PREDATOR_RESPONSE_END_THRESHOLD
                or predator_dist >= int(active_response["start_distance"]) + 2
            ):
                if bool(info.get("predator_escape")) and not preserved_escape_type:
                    preserved_escape_type = dominant_predator_type or _first_active_predator_type(
                        self.active_predator_responses_by_type
                    )
                latency = int(step - int(active_response["start_step"]))
                self.predator_response_latencies_by_type[predator_type].append(latency)
                self.active_predator_responses_by_type.pop(predator_type, None)

        if bool(info["predator_contact"]):
            for predator_type in _contact_predator_types(next_meta, state=state):
                self.predator_contacts_by_type[predator_type] += 1

        if bool(info.get("predator_escape")):
            escape_type = preserved_escape_type or dominant_predator_type or _first_active_predator_type(
                self.active_predator_responses_by_type
            )
            if escape_type:
                self.predator_escapes_by_type[escape_type] += 1

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
        learning_total = max(1, int(self.learning_steps))
        mean_module_credit_weights = {
            name: float(self.module_credit_weight_sums[name] / learning_total)
            for name in self.module_credit_weight_sums
        }
        module_gradient_norm_means = {
            name: float(self.module_gradient_norm_sums[name] / learning_total)
            for name in self.module_gradient_norm_sums
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
        terrain_slip_rates = {
            terrain: float(
                self.terrain_slip_counts.get(terrain, 0) / count
                if count
                else 0.0
            )
            for terrain, count in sorted(self.terrain_execution_counts.items())
        }
        predator_response_latency_by_type = {
            predator_type: (
                float(mean(self.predator_response_latencies_by_type.get(predator_type, [])))
                if self.predator_response_latencies_by_type.get(predator_type)
                else 0.0
            )
            for predator_type in PREDATOR_TYPE_NAMES
        }
        module_response_by_predator_type = {
            predator_type: _normalize_distribution(
                self.module_response_by_predator_type_counts.get(predator_type, {})
            )
            for predator_type in PREDATOR_TYPE_NAMES
        }
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
            "predator_contacts_by_type": {
                predator_type: int(self.predator_contacts_by_type.get(predator_type, 0))
                for predator_type in PREDATOR_TYPE_NAMES
            },
            "predator_escapes_by_type": {
                predator_type: int(self.predator_escapes_by_type.get(predator_type, 0))
                for predator_type in PREDATOR_TYPE_NAMES
            },
            "predator_response_latency_by_type": predator_response_latency_by_type,
            "module_response_by_predator_type": module_response_by_predator_type,
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
            "mean_module_credit_weights": mean_module_credit_weights,
            "module_gradient_norm_means": module_gradient_norm_means,
            "motor_slip_rate": float(
                self.motor_slip_steps / self.motor_execution_steps
                if self.motor_execution_steps
                else 0.0
            ),
            "mean_orientation_alignment": (
                float(mean(self.orientation_alignment_samples))
                if self.orientation_alignment_samples
                else 0.0
            ),
            "mean_terrain_difficulty": (
                float(mean(self.terrain_difficulty_samples))
                if self.terrain_difficulty_samples
                else 0.0
            ),
            "terrain_slip_rates": terrain_slip_rates,
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
        Create an EpisodeStats containing consolidated metrics and final-state fields for an episode.
        
        Parameters:
            episode (int): Episode index.
            seed (int): Random seed used for the episode.
            training (bool): Whether the episode was collected in training mode.
            scenario (str | None): Optional scenario identifier.
            total_reward (float): Cumulative reward for the episode.
            state (object): Final environment state exposing numeric attributes used for summary:
                steps_alive, food_eaten, sleep_events, shelter_entries,
                alert_events, predator_contacts, predator_sightings, predator_escapes,
                hunger, fatigue, sleep_debt, and health.
        
        Returns:
            EpisodeStats: Consolidated episode summary including episode metadata (episode, seed, training, scenario, total_reward), event counters (steps, food/sleep/shelter/predator counts, alert events), night summaries (ticks, per-role ticks and distribution, shelter occupancy and stillness rates), predator statistics (state tick counts, mode transition count, dominant state, per-type contacts/escapes, response event counts and mean latencies, module-response-by-predator-type distributions), distance deltas, mean sleep debt, reward component totals, reflex and per-module metrics (usage, override, dominance, contribution shares, learning/gradient summaries), motor/terrain slip metrics, dominant module statistics, final physiological metrics (hunger, fatigue, sleep_debt, health) and an alive flag.
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
            predator_contacts_by_type={
                predator_type: int(
                    snapshot["predator_contacts_by_type"].get(predator_type, 0)
                )
                for predator_type in PREDATOR_TYPE_NAMES
            },
            predator_escapes_by_type={
                predator_type: int(
                    snapshot["predator_escapes_by_type"].get(predator_type, 0)
                )
                for predator_type in PREDATOR_TYPE_NAMES
            },
            predator_response_latency_by_type={
                predator_type: float(
                    snapshot["predator_response_latency_by_type"].get(predator_type, 0.0)
                )
                for predator_type in PREDATOR_TYPE_NAMES
            },
            module_response_by_predator_type={
                predator_type: {
                    name: float(value)
                    for name, value in (
                        snapshot["module_response_by_predator_type"]
                        .get(predator_type, {})
                        .items()
                    )
                }
                for predator_type in PREDATOR_TYPE_NAMES
            },
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
            mean_module_credit_weights={
                name: float(value)
                for name, value in snapshot["mean_module_credit_weights"].items()
            },
            module_gradient_norm_means={
                name: float(value)
                for name, value in snapshot["module_gradient_norm_means"].items()
            },
            motor_slip_rate=float(snapshot["motor_slip_rate"]),
            mean_orientation_alignment=float(snapshot["mean_orientation_alignment"]),
            mean_terrain_difficulty=float(snapshot["mean_terrain_difficulty"]),
            terrain_slip_rates={
                str(name): float(value)
                for name, value in snapshot["terrain_slip_rates"].items()
            },
            dominant_module=str(snapshot["dominant_module"]),
            dominant_module_share=float(snapshot["dominant_module_share"]),
            effective_module_count=float(snapshot["effective_module_count"]),
            module_agreement_rate=float(snapshot["module_agreement_rate"]),
            module_disagreement_rate=float(snapshot["module_disagreement_rate"]),
        )


def _normalize_counts(counts: Dict[str, int], *, total: int) -> Dict[str, float]:
    """
    Normalize integer counts into fractional proportions using the provided total.
    
    Parameters:
        counts (Dict[str, int]): Mapping of names to integer counts.
        total (int): Denominator used to normalize counts.
    
    Returns:
        Dict[str, float]: Mapping of the same names to their normalized fraction (count / total). If `total` is less than or equal to 0, returns all zeros.
    """
    if total <= 0:
        return {name: 0.0 for name in counts}
    return {
        name: float(value / total)
        for name, value in counts.items()
    }


def _normalize_distribution(values: Mapping[str, int]) -> Dict[str, float]:
    """
    Normalize an integer-count mapping into a distribution of floats that sum to 1.0 (or all zeros if total is zero).
    
    Parameters:
        values (Mapping[str, int]): Mapping from keys to integer counts to normalize.
    
    Returns:
        Dict[str, float]: Mapping with the same keys and values as floats equal to count / total.
            If the sum of counts is less than or equal to zero, returns all keys mapped to 0.0.
    """
    total = sum(int(value) for value in values.values())
    if total <= 0:
        return {str(name): 0.0 for name in values}
    return {
        str(name): float(int(value) / total)
        for name, value in values.items()
    }


def _predator_type_threat(meta: Mapping[str, object], predator_type: str) -> float:
    """
    Retrieve the predator threat value for a given predator type from metadata.
    
    Parameters:
        meta (Mapping[str, object]): Metadata mapping that may contain keys of the form "<predator_type>_predator_threat".
        predator_type (str): Predator type label (e.g., "visual" or "olfactory") used to form the metadata key.
    
    Returns:
        float: The threat value coerced to float; returns 0.0 if the key is missing or the value is falsy.
    """
    return float(meta.get(f"{predator_type}_predator_threat", 0.0) or 0.0)


def _dominant_predator_type(meta: Mapping[str, object]) -> str:
    """
    Determine the dominant predator type label from diagnostic metadata.
    
    Examines `meta["dominant_predator_type_label"]` (case-insensitive) and returns it if it matches a known predator type. If no valid label is present, compares numeric threat values for "visual" and "olfactory" (read via keys like `"visual_predator_threat"` / `"olfactory_predator_threat"`) and returns the type with the larger threat. Returns an empty string when both threats are zero or absent.
    
    Parameters:
        meta (Mapping[str, object]): Diagnostic metadata that may include
            - "dominant_predator_type_label": a preferred label (string)
            - "<type>_predator_threat": numeric threat values for "visual" and "olfactory"
    
    Returns:
        The dominant predator type: "visual" or "olfactory", or an empty string if none is dominant.
    """
    label = str(meta.get("dominant_predator_type_label") or "").strip().lower()
    if label in PREDATOR_TYPE_NAMES:
        return label
    visual_threat = _predator_type_threat(meta, "visual")
    olfactory_threat = _predator_type_threat(meta, "olfactory")
    if visual_threat <= 0.0 and olfactory_threat <= 0.0:
        return ""
    return "olfactory" if olfactory_threat > visual_threat else "visual"


def _diagnostic_predator_distance(meta: Mapping[str, object]) -> int:
    """
    Extract the diagnostic predator distance from a metadata mapping.
    
    Reads the "diagnostic" mapping from `meta` and returns the value of
    "diagnostic_predator_dist" coerced to an integer. If the diagnostic
    mapping or the distance value is missing or not a number, returns 0.
    
    Parameters:
        meta (Mapping[str, object]): Metadata that may contain a "diagnostic" mapping.
    
    Returns:
        int: The diagnostic predator distance, or 0 when unavailable or invalid.
    """
    diagnostic = meta.get("diagnostic", {})
    if not isinstance(diagnostic, Mapping):
        return 0
    return int(float(diagnostic.get("diagnostic_predator_dist", 0) or 0))


def _diagnostic_predator_distance_for_type(
    meta: Mapping[str, object],
    predator_type: str,
    *,
    state: object,
) -> int:
    """Return the nearest diagnostic predator distance for a specific predator type."""
    spider_x = getattr(state, "x", None)
    spider_y = getattr(state, "y", None)
    if spider_x is None or spider_y is None:
        return _diagnostic_predator_distance(meta)
    distances: list[int] = []
    predators = meta.get("predators", [])
    if isinstance(predators, list):
        for predator in predators:
            if not isinstance(predator, Mapping):
                continue
            profile = predator.get("profile", {})
            if not isinstance(profile, Mapping):
                continue
            detection_style = str(profile.get("detection_style") or "").strip().lower()
            if detection_style != predator_type:
                continue
            try:
                predator_x = int(predator["x"])
                predator_y = int(predator["y"])
            except (KeyError, TypeError, ValueError):
                continue
            distances.append(abs(int(spider_x) - predator_x) + abs(int(spider_y) - predator_y))
    if distances:
        return min(distances)
    return _diagnostic_predator_distance(meta)


def _first_active_predator_type(active: Mapping[str, object]) -> str:
    """
    Selects the first predator type key from the active mapping that matches the known predator type order.
    
    Parameters:
        active (Mapping[str, object]): Mapping whose keys may include predator type names.
    
    Returns:
        str: The first predator type (one of PREDATOR_TYPE_NAMES) found in `active`, or an empty string if none are present.
    """
    for predator_type in PREDATOR_TYPE_NAMES:
        if predator_type in active:
            return predator_type
    return ""


def _contact_predator_types(
    meta: Mapping[str, object],
    *,
    state: object,
) -> list[str]:
    """
    Determine which predator type(s) are in contact with the agent at its current grid coordinates.
    
    Parameters:
        meta (Mapping[str, object]): Metadata that may contain a "predators" list of mappings, and fields used by the dominant-predator heuristic.
        state (object): Agent state object expected to expose numeric attributes `x` and `y` for current grid coordinates.
    
    Returns:
        list[str]: A list of predator type labels (from PREDATOR_TYPE_NAMES) whose predators occupy the same integer grid cell as the agent.
                   If no predators are found at the agent's coordinates, returns a single-element list containing the dominant predator type when available;
                   returns an empty list if no dominant type can be determined.
    """
    spider_x = getattr(state, "x", None)
    spider_y = getattr(state, "y", None)
    if spider_x is None or spider_y is None:
        dominant_type = _dominant_predator_type(meta)
        return [dominant_type] if dominant_type else []
    contact_types: list[str] = []
    predators = meta.get("predators", [])
    if isinstance(predators, list):
        for predator in predators:
            if not isinstance(predator, Mapping):
                continue
            if int(predator.get("x", -1)) != int(spider_x) or int(predator.get("y", -1)) != int(spider_y):
                continue
            profile = predator.get("profile", {})
            if not isinstance(profile, Mapping):
                continue
            detection_style = str(profile.get("detection_style") or "").strip().lower()
            if detection_style in PREDATOR_TYPE_NAMES and detection_style not in contact_types:
                contact_types.append(detection_style)
    if contact_types:
        return contact_types
    dominant_type = _dominant_predator_type(meta)
    return [dominant_type] if dominant_type else []


def _mean_map(
    history: List[EpisodeStats],
    names: Sequence[str],
    getter: "Callable[[EpisodeStats, str], float]",
) -> Dict[str, float]:
    """
    Compute per-name means over a history of EpisodeStats using a caller-supplied getter.

    Parameters:
        history: Episodes to aggregate over.
        names: Keys to include in the result.
        getter: Function receiving (stats, name) and returning a float value for that name.

    Returns:
        Dict mapping each name to its mean across history, or 0.0 when history is empty.
    """
    if not history:
        return {name: 0.0 for name in names}
    return {name: mean(getter(stats, name) for stats in history) for name in names}


def aggregate_episode_stats(history: List[EpisodeStats]) -> Dict[str, object]:
    """
    Aggregate suite-level statistics from a sequence of EpisodeStats.
    
    Returns:
        dict: Mapping of aggregate metrics computed across the provided episodes. Keys include:
            - counts: "episodes"
            - overall means: "mean_reward", "mean_food", "mean_sleep", survival and event means
            - predator metrics: aggregate contacts, escapes, response event counts and latencies (including per-type)
            - night metrics: mean occupancy/stillness rates, per-role ticks and distributions
            - per-reward-component means ("mean_reward_components")
            - predator-state summaries: mean ticks, occupancy rates, mode transitions, and dominant state
            - reflex and module metrics: overall and per-module usage/override/dominance rates, contribution shares,
              mean module response distributions by predator type, dominant-module stats, credit-weight and gradient means
            - motor/terrain metrics: motor slip rate, orientation alignment, terrain difficulty, per-terrain slip rates
            - distance/sleep deltas and mean sleep debt
            - "episodes_detail": list of per-episode dicts produced by asdict(stats)
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
    reward_component_means = _mean_map(
        history, reward_component_names,
        lambda s, n: s.reward_component_totals[n],
    )
    predator_state_means = _mean_map(
        history, predator_states,
        lambda s, n: s.predator_state_ticks[n],
    )
    night_role_tick_means = _mean_map(
        history, SHELTER_ROLES,
        lambda s, n: s.night_role_ticks[n],
    )
    night_role_distribution = _mean_map(
        history, SHELTER_ROLES,
        lambda s, n: s.night_role_distribution[n],
    )
    steps_mean = mean(stats.steps for stats in history) if history else 1.0
    predator_state_rates = {
        name: (predator_state_means[name] / steps_mean if steps_mean else 0.0)
        for name in predator_state_means
    }
    module_reflex_usage_rate_means = _mean_map(
        history, REFLEX_MODULE_NAMES,
        lambda s, n: s.module_reflex_usage_rates.get(n, 0.0),
    )
    module_reflex_override_rate_means = _mean_map(
        history, REFLEX_MODULE_NAMES,
        lambda s, n: s.module_reflex_override_rates.get(n, 0.0),
    )
    module_reflex_dominance_means = _mean_map(
        history, REFLEX_MODULE_NAMES,
        lambda s, n: s.module_reflex_dominance.get(n, 0.0),
    )
    module_contribution_share_means = _mean_map(
        history, PROPOSAL_SOURCE_NAMES,
        lambda s, n: s.module_contribution_share.get(n, 0.0),
    )
    def _extra_module_names(attr: str) -> List[str]:
        return sorted(
            {name for stats in history for name in getattr(stats, attr) if name not in PROPOSAL_SOURCE_NAMES}
        )

    module_credit_names = list(PROPOSAL_SOURCE_NAMES) + (_extra_module_names("mean_module_credit_weights") if history else [])
    module_gradient_names = list(PROPOSAL_SOURCE_NAMES) + (_extra_module_names("module_gradient_norm_means") if history else [])
    module_credit_weight_means = _mean_map(
        history, module_credit_names,
        lambda s, n: s.mean_module_credit_weights.get(n, 0.0),
    )
    module_gradient_norm_means = _mean_map(
        history, module_gradient_names,
        lambda s, n: s.module_gradient_norm_means.get(n, 0.0),
    )
    terrain_slip_names = sorted(
        {
            terrain
            for stats in history
            for terrain in stats.terrain_slip_rates
        }
    )
    terrain_slip_rate_means = _mean_map(
        history, terrain_slip_names,
        lambda s, n: s.terrain_slip_rates.get(n, 0.0),
    )
    predator_type_names = sorted(
        {
            *PREDATOR_TYPE_NAMES,
            *(
                predator_type
                for stats in history
                for predator_type in (
                    set(stats.predator_contacts_by_type)
                    | set(stats.predator_escapes_by_type)
                    | set(stats.predator_response_latency_by_type)
                    | set(stats.module_response_by_predator_type)
                )
            ),
        }
    )
    mean_predator_contacts_by_type = _mean_map(
        history, predator_type_names,
        lambda s, n: s.predator_contacts_by_type.get(n, 0),
    )
    mean_predator_escapes_by_type = _mean_map(
        history, predator_type_names,
        lambda s, n: s.predator_escapes_by_type.get(n, 0),
    )
    mean_predator_response_latency_by_type = _mean_map(
        history, predator_type_names,
        lambda s, n: s.predator_response_latency_by_type.get(n, 0.0),
    )
    mean_module_response_by_predator_type = {
        pt: _mean_map(
            history,
            PROPOSAL_SOURCE_NAMES,
            lambda s, n, _pt=pt: s.module_response_by_predator_type.get(_pt, {}).get(n, 0.0),
        )
        for pt in predator_type_names
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
        "mean_predator_contacts_by_type": mean_predator_contacts_by_type,
        "mean_predator_escapes": (
            mean(stats.predator_escapes for stats in history) if history else 0.0
        ),
        "mean_predator_escapes_by_type": mean_predator_escapes_by_type,
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
        "mean_predator_response_latency_by_type": (
            mean_predator_response_latency_by_type
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
        "mean_module_response_by_predator_type": mean_module_response_by_predator_type,
        "mean_module_credit_weights": module_credit_weight_means,
        "module_gradient_norm_means": module_gradient_norm_means,
        "mean_motor_slip_rate": (
            mean(stats.motor_slip_rate for stats in history) if history else 0.0
        ),
        "mean_orientation_alignment": (
            mean(stats.mean_orientation_alignment for stats in history) if history else 0.0
        ),
        "mean_terrain_difficulty": (
            mean(stats.mean_terrain_difficulty for stats in history) if history else 0.0
        ),
        "mean_terrain_slip_rates": terrain_slip_rate_means,
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
    Aggregate episode-level behavioral scores into a scenario-level summary dictionary.
    
    Produces a mapping with per-check pass rates and mean values, aggregated behavior metrics (numeric mean or most-common), diagnostic summaries (primary outcome, outcome distribution, optional failure-mode diagnostics, partial progress and died-without-contact aggregates), the episode success rate, sorted unique failure names, per-episode detail records, and a plain-copy of any provided legacy metrics.
    
    Parameters:
        scores (Sequence[BehavioralEpisodeScore]): Episode-level behavioral scores to aggregate.
        scenario (str): Scenario identifier for the aggregation.
        description (str): Human-readable description of the scenario.
        objective (str): Objective name associated with the aggregated scores.
        check_specs (Sequence[BehaviorCheckSpec]): Specifications for checks to include; used to supply each check's description and expected value.
        diagnostic_focus (str | None): Optional diagnostic focus label to include in the output (empty string when None).
        success_interpretation (str | None): Optional textual interpretation of success included in the output (empty string when None).
        failure_interpretation (str | None): Optional textual interpretation of failures included in the output (empty string when None).
        budget_note (str | None): Optional budget/note string to include in the output (empty string when None).
        legacy_metrics (Mapping[str, Any] | None): Optional additional metrics preserved unchanged under the `legacy_metrics` key.
    
    Returns:
        Dict[str, object]: Aggregated summary containing at least the keys:
            - "scenario", "description", "objective": echoed input metadata.
            - "diagnostic_focus", "success_interpretation", "failure_interpretation", "budget_note": optional metadata strings.
            - "episodes" (int): number of episodes aggregated.
            - "success_rate" (float): mean of per-episode success indicators (`1.0` for success, `0.0` otherwise).
            - "checks" (dict): mapping check name -> { "description", "expected", "pass_rate", "mean_value" }.
            - "behavior_metrics" (dict): aggregated metrics (numeric mean or most-common value).
            - "diagnostics" (dict): includes "primary_outcome", "outcome_distribution", optional "primary_failure_mode" and "failure_mode_distribution", and aggregated diagnostic rates for "partial_progress" and "died_without_contact" when present.
            - "failures" (list[str]): sorted unique failure names observed across episodes.
            - "episodes_detail" (list[dict]): per-episode records converted to plain dicts.
            - "legacy_metrics" (dict): plain dict copy of `legacy_metrics` or an empty dict.
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
    failure_mode_labels = [
        str(score.behavior_metrics["failure_mode"])
        for score in score_list
        if "failure_mode" in score.behavior_metrics
    ]
    failure_mode_counter = Counter(failure_mode_labels)
    failure_mode_total = max(1, len(failure_mode_labels))
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
    if failure_mode_counter:
        diagnostics["primary_failure_mode"] = failure_mode_counter.most_common(1)[0][0]
        diagnostics["failure_mode_distribution"] = {
            label: float(count / failure_mode_total)
            for label, count in sorted(failure_mode_counter.items())
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


def summarize_behavior_suite(
    suite: Mapping[str, Mapping[str, Any]],
    *,
    competence_label: str = "mixed",
) -> Dict[str, object]:
    """
    Summarizes per-scenario behavior aggregation results into suite-level counts and rates.
    
    Parameters:
        suite (Mapping[str, Mapping[str, Any]]): Mapping from scenario name to its aggregated data. Each scenario mapping is expected to include numeric "episodes" and "success_rate" keys and may include a "failures" iterable.
        competence_label (str): Competence context for this evaluation; one of "self_sufficient", "scaffolded", or "mixed".
    
    Returns:
        Dict[str, object]: Summary dictionary with keys:
            - "scenario_count": number of scenarios in the suite.
            - "episode_count": total number of episodes across all scenarios.
            - "scenario_success_rate": mean of per-scenario indicators where a scenario's `success_rate` is at least 1.0 (1.0 if fully successful, 0.0 otherwise).
            - "episode_success_rate": overall fraction of episodes considered successful (episodes-weighted success rate).
            - "regressions": list of regression entries (one per scenario that reported failures), each of the form {"scenario": <name>, "failures": [<failure names>]}.
            - "competence_type": the validated competence label.
    """
    competence_type = normalize_competence_label(competence_label)
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
        "competence_type": competence_type,
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
    eval_reflex_scale: float | None = None,
    competence_label: str | None = None,
) -> List[Dict[str, object]]:
    """
    Flatten BehavioralEpisodeScore records into tabular row dictionaries.
    
    Each row contains fixed metadata columns (reward_profile, scenario_map, evaluation_map,
    competence_type, is_primary_benchmark, eval_reflex_scale, simulation_seed, episode_seed,
    scenario, scenario_description, scenario_objective, scenario_focus, episode, success,
    failure_count, failures), plus one `metric_{name}` entry per behavior metric and for each
    check three columns: `check_{name}_passed`, `check_{name}_value`, and `check_{name}_expected`.
    
    Parameters:
        scores (Sequence[BehavioralEpisodeScore]): Episode-level behavioral scores to flatten.
        reward_profile (str): Identifier for the reward configuration applied to all rows.
        scenario_map (str): Map template used by the scenario.
        simulation_seed (int): Global simulation seed applied to all rows.
        scenario_description (str): Human-readable scenario description to include in each row.
        scenario_objective (str): Scenario objective string to include in each row.
        scenario_focus (str): Scenario focus/category to include in each row.
        evaluation_map (str | None): Optional outer sweep or default map context for the run.
        eval_reflex_scale (float | None): Optional evaluation-time reflex scale used to derive competence when `competence_label` is not provided.
        competence_label (str | None): Optional explicit competence label override; when omitted, competence is derived from `eval_reflex_scale` and validated.
    
    Returns:
        List[Dict[str, object]]: A list of per-episode row dictionaries ready for tabular export. Each row includes the computed `competence_type` and `is_primary_benchmark` fields.
    """
    rows: List[Dict[str, object]] = []
    competence_type = normalize_competence_label(
        competence_label
        if competence_label is not None
        else competence_label_from_eval_reflex_scale(eval_reflex_scale)
    )
    for score in scores:
        row: Dict[str, object] = {
            "reward_profile": reward_profile,
            "scenario_map": scenario_map,
            "evaluation_map": evaluation_map,
            "competence_type": competence_type,
            "is_primary_benchmark": competence_type == "self_sufficient",
            "eval_reflex_scale": eval_reflex_scale,
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
