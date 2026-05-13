from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..ablations import resolve_ablation_configs
from ..b_series import B_CURRENT_BRIDGE_EFFECTIVE_LEVEL
from ..b_series_legacy import LEGACY_B0_ACTIONS, LegacyB0Simulation
from ..simulation import SpiderSimulation
from ..world import ACTIONS


EVOLUTION_SNAPSHOT_ROOT = Path("artifacts/gui_evolution_snapshots")


@dataclass(frozen=True)
class GUIModelSpec:
    id: str
    label: str
    family: str
    variant: str
    runtime_kind: str
    evolution_transfer_compatible: bool
    description: str
    ablation_variant: str | None = None


@dataclass(frozen=True)
class GUIRunConfig:
    width: int = 12
    height: int = 12
    food_count: int = 4
    day_length: int = 36
    night_length: int = 24
    max_steps: int = 120
    seed: int = 7
    gamma: float = 0.96
    module_lr: float = 0.010
    motor_lr: float = 0.012
    module_dropout: float = 0.05
    capacity_profile: Any = None
    reward_profile: str = "classic"
    map_template: str = "central_burrow"
    operational_profile: Any = "default_v1"
    noise_profile: Any = "none"

    @classmethod
    def from_simulation(cls, sim: SpiderSimulation) -> "GUIRunConfig":
        world = sim.world
        brain = sim.brain
        return cls(
            width=int(world.width),
            height=int(world.height),
            food_count=int(getattr(world, "food_count", len(world.food_positions))),
            day_length=int(world.day_length),
            night_length=int(world.night_length),
            max_steps=int(sim.max_steps),
            seed=int(sim.seed),
            gamma=float(brain.gamma),
            module_lr=float(brain.module_lr),
            motor_lr=float(brain.motor_lr),
            module_dropout=float(brain.module_dropout),
            capacity_profile=getattr(sim, "capacity_profile", None),
            reward_profile=str(world.reward_profile),
            map_template=str(world.map_template_name),
            operational_profile=getattr(sim, "operational_profile", "default_v1"),
            noise_profile=getattr(sim, "noise_profile", "none"),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "food_count": self.food_count,
            "day_length": self.day_length,
            "night_length": self.night_length,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "gamma": self.gamma,
            "module_lr": self.module_lr,
            "motor_lr": self.motor_lr,
            "module_dropout": self.module_dropout,
            "capacity_profile": _simple_metadata_value(self.capacity_profile),
            "reward_profile": self.reward_profile,
            "map_template": self.map_template,
            "operational_profile": _simple_metadata_value(self.operational_profile),
            "noise_profile": _simple_metadata_value(self.noise_profile),
        }


GUI_MODEL_SPECS: tuple[GUIModelSpec, ...] = (
    GUIModelSpec(
        id="modular_full",
        label="A4 Modular Full",
        family="A",
        variant="modular_full",
        runtime_kind="current_world",
        evolution_transfer_compatible=False,
        description="Current modular baseline in SpiderWorld.",
        ablation_variant="modular_full",
    ),
    GUIModelSpec(
        id="a0_true_monolithic",
        label="A0 True Monolithic",
        family="A",
        variant="true_monolithic_policy",
        runtime_kind="current_world",
        evolution_transfer_compatible=False,
        description="A0 primitive direct-control baseline.",
        ablation_variant="true_monolithic_policy",
    ),
    GUIModelSpec(
        id="a0_owned_option",
        label="A0 Owned Option",
        family="A",
        variant="true_monolithic_owned_option_controller_policy",
        runtime_kind="current_world",
        evolution_transfer_compatible=False,
        description="A0 diagnostic with owned internal option leaves.",
        ablation_variant="true_monolithic_owned_option_controller_policy",
    ),
    GUIModelSpec(
        id="b0_current_bridge",
        label="B0 Current Bridge",
        family="B",
        variant="b0_current_bridge_policy",
        runtime_kind="current_world",
        evolution_transfer_compatible=True,
        description="B0 semantic bridge running through current primitive actions.",
        ablation_variant="b0_current_bridge_policy",
    ),
    GUIModelSpec(
        id="b0_legacy_semantic",
        label="B0 Legacy Semantic",
        family="B",
        variant="b0_legacy_semantic_policy",
        runtime_kind="legacy_b0",
        evolution_transfer_compatible=False,
        description="Isolated reproduction harness with the six old semantic actions.",
        ablation_variant=None,
    ),
)

GUI_MODEL_SPECS_BY_ID = {spec.id: spec for spec in GUI_MODEL_SPECS}


def resolve_gui_model_spec(model_id: str) -> GUIModelSpec:
    try:
        return GUI_MODEL_SPECS_BY_ID[model_id]
    except KeyError as exc:
        available = ", ".join(sorted(GUI_MODEL_SPECS_BY_ID))
        raise KeyError(f"Unknown GUI model {model_id!r}. Available: {available}") from exc


def build_runtime_adapter(
    run_config: GUIRunConfig,
    model_id: str,
) -> "GUIRuntimeAdapter":
    spec = resolve_gui_model_spec(model_id)
    if spec.runtime_kind == "legacy_b0":
        sim = LegacyB0Simulation(
            width=run_config.width,
            height=run_config.height,
            food_count=run_config.food_count,
            day_length=run_config.day_length,
            night_length=run_config.night_length,
            max_steps=run_config.max_steps,
            seed=run_config.seed,
            gamma=run_config.gamma,
            module_lr=run_config.module_lr,
            motor_lr=run_config.motor_lr,
        )
        return LegacyB0RuntimeAdapter(sim=sim, spec=spec, run_config=run_config)

    if spec.ablation_variant is None:
        raise ValueError(f"Current-world GUI model {spec.id!r} needs an ablation variant.")
    brain_config = resolve_ablation_configs(
        [spec.ablation_variant],
        module_dropout=run_config.module_dropout,
        capacity_profile=run_config.capacity_profile,
    )[0]
    sim = SpiderSimulation(
        width=run_config.width,
        height=run_config.height,
        food_count=run_config.food_count,
        day_length=run_config.day_length,
        night_length=run_config.night_length,
        max_steps=run_config.max_steps,
        seed=run_config.seed,
        gamma=run_config.gamma,
        module_lr=run_config.module_lr,
        motor_lr=run_config.motor_lr,
        module_dropout=run_config.module_dropout,
        capacity_profile=run_config.capacity_profile,
        reward_profile=run_config.reward_profile,
        map_template=run_config.map_template,
        brain_config=brain_config,
        operational_profile=run_config.operational_profile,
        noise_profile=run_config.noise_profile,
    )
    return CurrentWorldRuntimeAdapter(sim=sim, spec=spec, run_config=run_config)


def adapter_for_existing_simulation(
    sim: SpiderSimulation,
    *,
    model_id: str = "modular_full",
) -> "CurrentWorldRuntimeAdapter":
    return CurrentWorldRuntimeAdapter(
        sim=sim,
        spec=resolve_gui_model_spec(model_id),
        run_config=GUIRunConfig.from_simulation(sim),
    )


class GUIRuntimeAdapter:
    sim: Any
    spec: GUIModelSpec
    run_config: GUIRunConfig
    actions: Sequence[str]
    supports_current_metrics: bool

    @property
    def world(self) -> Any:
        return self.sim.world

    @property
    def brain(self) -> Any:
        return self.sim.brain

    @property
    def bus(self) -> Any:
        return self.sim.bus

    @property
    def seed(self) -> int:
        return int(self.sim.seed)

    @property
    def max_steps(self) -> int:
        return int(self.sim.max_steps)

    def reset(self, *, seed: int) -> Mapping[str, Any]:
        return self.world.reset(seed=seed)

    def act(self, observation: Mapping[str, Any], *, sample: bool) -> Any:
        return self.brain.act(observation, self.bus, sample=sample)

    def step(self, action_idx: int) -> tuple[Mapping[str, Any], float, bool, Mapping[str, Any]]:
        return self.world.step(action_idx)

    def learn(
        self,
        decision: Any,
        reward: float,
        next_observation: Mapping[str, Any],
        terminal: bool,
    ) -> Mapping[str, float]:
        return self.brain.learn(decision, reward, next_observation, terminal)

    def audit_fields(self, decision: Any | None) -> dict[str, object]:
        raise NotImplementedError

    def save_evolution_snapshot(
        self,
        base_dir: str | Path,
        *,
        controller_state: Mapping[str, object],
    ) -> Path:
        raise NotImplementedError


@dataclass
class CurrentWorldRuntimeAdapter(GUIRuntimeAdapter):
    sim: SpiderSimulation
    spec: GUIModelSpec
    run_config: GUIRunConfig
    actions: Sequence[str] = ACTIONS
    supports_current_metrics: bool = True

    def audit_fields(self, decision: Any | None) -> dict[str, object]:
        config = self.brain.config
        fields: dict[str, object] = {
            "variant": config.name,
            "architecture": config.architecture,
            "architecture_description": config.to_summary()[
                "architecture_description"
            ],
            "runtime": self.spec.runtime_kind,
            "action_space": f"{len(ACTIONS)} primitive",
            "b_level": int(config.b_level) if config.is_b_series else None,
            "b_effective_level": (
                B_CURRENT_BRIDGE_EFFECTIVE_LEVEL
                if config.is_b_series
                and int(config.b_level) == 0
                and str(config.b_mode) == "current_bridge"
                else None
            ),
            "b_mode": str(config.b_mode) if config.is_b_series else None,
            "seed": int(self.seed),
            "map": str(self.world.map_template_name),
            "reward_profile": str(self.world.reward_profile),
            "evolution_transfer_compatible": bool(
                self.spec.evolution_transfer_compatible
            ),
        }
        if decision is not None:
            fields.update(
                {
                    "semantic_action": getattr(decision, "semantic_action", None),
                    "learned_semantic_action": getattr(
                        decision,
                        "learned_semantic_action",
                        None,
                    ),
                    "semantic_action_source": getattr(
                        decision,
                        "semantic_action_source",
                        None,
                    ),
                    "semantic_action_reason": getattr(
                        decision,
                        "semantic_action_reason",
                        None,
                    ),
                    "semantic_override_count": int(
                        getattr(decision, "semantic_override_count", 0)
                    ),
                    "bridge_primitive_action": getattr(
                        decision,
                        "bridge_primitive_action",
                        None,
                    ),
                    "bridge_reason": getattr(decision, "bridge_reason", None),
                    "external_override_count": int(
                        getattr(decision, "external_override_count", 0)
                    ),
                }
            )
        return fields

    def save_evolution_snapshot(
        self,
        base_dir: str | Path,
        *,
        controller_state: Mapping[str, object],
    ) -> Path:
        directory = _next_snapshot_dir(base_dir, self.spec.variant)
        self.brain.save(directory)
        _write_gui_snapshot_metadata(
            directory,
            spec=self.spec,
            run_config=self.run_config,
            controller_state=controller_state,
            extra={
                "brain_checkpoint": "SpiderBrain.save",
                "ablation_config": self.brain.config.to_summary(),
            },
        )
        return directory


@dataclass
class LegacyB0RuntimeAdapter(GUIRuntimeAdapter):
    sim: LegacyB0Simulation
    spec: GUIModelSpec
    run_config: GUIRunConfig
    actions: Sequence[str] = LEGACY_B0_ACTIONS
    supports_current_metrics: bool = False

    def audit_fields(self, decision: Any | None) -> dict[str, object]:
        semantic_action = None
        if decision is not None:
            semantic_action = LEGACY_B0_ACTIONS[int(decision.action_idx)]
        return {
            "variant": self.spec.variant,
            "architecture": "b_series",
            "architecture_description": "B0 isolated legacy semantic benchmark",
            "runtime": self.spec.runtime_kind,
            "action_space": f"{len(LEGACY_B0_ACTIONS)} semantic",
            "b_level": 0,
            "b_mode": "legacy_semantic",
            "seed": int(self.seed),
            "map": "legacy_3x3_shelter",
            "reward_profile": "legacy_dense_semantic",
            "semantic_action": semantic_action,
            "bridge_primitive_action": None,
            "bridge_reason": "legacy_world_executes_semantic_action",
            "external_override_count": 0,
            "evolution_transfer_compatible": False,
        }

    def save_evolution_snapshot(
        self,
        base_dir: str | Path,
        *,
        controller_state: Mapping[str, object],
    ) -> Path:
        directory = _next_snapshot_dir(base_dir, self.spec.variant)
        directory.mkdir(parents=True, exist_ok=True)
        weights: dict[str, np.ndarray] = {}
        metadata: dict[str, object] = {
            "schema": "legacy_b0_weights_v1",
            "modules": {},
            "motor_cortex": _network_metadata(self.brain.motor_cortex.state_dict()),
        }
        for name, network in self.brain.module_bank.modules.items():
            state = network.state_dict()
            metadata["modules"][name] = _network_metadata(state)
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    weights[f"module.{name}.{key}"] = value
        for key, value in self.brain.motor_cortex.state_dict().items():
            if isinstance(value, np.ndarray):
                weights[f"motor_cortex.{key}"] = value
        np.savez(directory / "legacy_b0_weights.npz", **weights)
        (directory / "legacy_b0_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        _write_gui_snapshot_metadata(
            directory,
            spec=self.spec,
            run_config=self.run_config,
            controller_state=controller_state,
            extra={
                "legacy_checkpoint": "legacy_b0_weights.npz",
                "legacy_metadata": "legacy_b0_metadata.json",
                "legacy_state": self.world.state_dict(),
            },
        )
        return directory


def _network_metadata(state: Mapping[str, object]) -> dict[str, object]:
    return {
        key: int(value) if isinstance(value, (int, np.integer)) else str(value)
        for key, value in state.items()
        if not isinstance(value, np.ndarray)
    }


def _simple_metadata_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, (str, int, float, bool)):
        return name
    return str(value)


def _next_snapshot_dir(base_dir: str | Path, variant: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(base_dir) / variant / timestamp


def _write_gui_snapshot_metadata(
    directory: Path,
    *,
    spec: GUIModelSpec,
    run_config: GUIRunConfig,
    controller_state: Mapping[str, object],
    extra: Mapping[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "gui_evolution_snapshot_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evolution": {
            "process_name": "Evolution",
            "source_model_id": spec.id,
            "source_variant": spec.variant,
            "source_family": spec.family,
            "transfer_compatible": bool(spec.evolution_transfer_compatible),
        },
        "model": asdict(spec),
        "run": run_config.to_metadata(),
        "controller": dict(controller_state),
        **dict(extra),
    }
    (directory / "gui_snapshot.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
