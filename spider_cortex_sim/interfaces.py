from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from typing import ClassVar, Dict, Mapping, Sequence, Tuple, TypeVar

import numpy as np

LOCOMOTION_ACTIONS: Sequence[str] = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "STAY",
)
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(LOCOMOTION_ACTIONS)}
ACTION_DELTAS: Dict[str, Tuple[int, int]] = {
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
    "STAY": (0, 0),
}


@dataclass(frozen=True)
class SignalSpec:
    name: str
    description: str
    minimum: float = -1.0
    maximum: float = 1.0

    def to_summary(self) -> dict[str, object]:
        """
        Return a JSON-serializable schema entry for this signal.
        """
        return {
            "name": self.name,
            "description": self.description,
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


def _validate_exact_keys(label: str, expected_names: Sequence[str], values: Mapping[str, float]) -> None:
    """
    Validate that the keys of `values` exactly match `expected_names`.
    
    Parameters:
    	label (str): Human-readable label inserted into the error message when keys do not match.
    	expected_names (Sequence[str]): Ordered sequence of required key names.
    	values (Mapping[str, float]): Mapping whose keys are validated against `expected_names`.
    
    Raises:
    	ValueError: If any expected names are missing or any extra keys are present; the error message lists missing and/or extra keys and includes `label`.
    """
    expected = tuple(expected_names)
    received_keys = tuple(values.keys())
    received = set(received_keys)
    expected_set = set(expected)

    missing = [name for name in expected if name not in received]
    extra = sorted(name for name in received_keys if name not in expected_set)
    if not missing and not extra:
        return

    details = []
    if missing:
        details.append(f"faltando {missing}")
    if extra:
        details.append(f"extras {extra}")
    raise ValueError(f"{label} recebeu sinais incompatíveis: {'; '.join(details)}.")


_ObservationViewT = TypeVar("_ObservationViewT", bound="ObservationView")


class ObservationView:
    observation_key: ClassVar[str]

    def as_mapping(self) -> dict[str, float]:
        """
        Return a mapping of this observation's dataclass field names to their float values.
        
        Returns:
            dict[str, float]: Mapping from each field name to its value converted to float.
        """
        return {
            field.name: float(getattr(self, field.name))
            for field in fields(self)
        }

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """
        Get the dataclass field names for the class in declaration order.
        
        Returns:
            tuple[str, ...]: Tuple of field name strings in declaration order.
        """
        return tuple(field.name for field in fields(cls))

    @classmethod
    def from_mapping(cls: type[_ObservationViewT], values: Mapping[str, float]) -> _ObservationViewT:
        """
        Construct an instance of the observation view from a mapping of signal names to numeric values.
        
        Validates that `values` contains exactly the class's field names, converts each value to `float`, and returns a new instance of `cls`.
        
        Parameters:
            values (Mapping[str, float]): Mapping from each field name to its numeric value.
        
        Returns:
            _ObservationViewT: An instance of `cls` populated with the provided values converted to floats.
        
        Raises:
            ValueError: If `values` is missing expected field names or contains unexpected keys.
        """
        names = cls.field_names()
        _validate_exact_keys(cls.__name__, names, values)
        return cls(**{name: float(values[name]) for name in names})


@dataclass(frozen=True)
class VisualObservation(ObservationView):
    observation_key: ClassVar[str] = "visual"

    food_visible: float
    food_certainty: float
    food_occluded: float
    food_dx: float
    food_dy: float
    shelter_visible: float
    shelter_certainty: float
    shelter_occluded: float
    shelter_dx: float
    shelter_dy: float
    predator_visible: float
    predator_certainty: float
    predator_occluded: float
    predator_dx: float
    predator_dy: float
    day: float
    night: float


@dataclass(frozen=True)
class SensoryObservation(ObservationView):
    observation_key: ClassVar[str] = "sensory"

    recent_pain: float
    recent_contact: float
    health: float
    hunger: float
    fatigue: float
    food_smell_strength: float
    food_smell_dx: float
    food_smell_dy: float
    predator_smell_strength: float
    predator_smell_dx: float
    predator_smell_dy: float
    light: float


@dataclass(frozen=True)
class HungerObservation(ObservationView):
    observation_key: ClassVar[str] = "hunger"

    hunger: float
    on_food: float
    food_visible: float
    food_certainty: float
    food_occluded: float
    food_dx: float
    food_dy: float
    food_smell_strength: float
    food_smell_dx: float
    food_smell_dy: float
    food_memory_dx: float
    food_memory_dy: float
    food_memory_age: float


@dataclass(frozen=True)
class SleepObservation(ObservationView):
    observation_key: ClassVar[str] = "sleep"

    fatigue: float
    hunger: float
    on_shelter: float
    night: float
    home_dx: float
    home_dy: float
    home_dist: float
    health: float
    recent_pain: float
    sleep_phase_level: float
    rest_streak_norm: float
    sleep_debt: float
    shelter_role_level: float
    shelter_memory_dx: float
    shelter_memory_dy: float
    shelter_memory_age: float


@dataclass(frozen=True)
class AlertObservation(ObservationView):
    observation_key: ClassVar[str] = "alert"

    predator_visible: float
    predator_certainty: float
    predator_occluded: float
    predator_dx: float
    predator_dy: float
    predator_dist: float
    predator_smell_strength: float
    home_dx: float
    home_dy: float
    recent_pain: float
    recent_contact: float
    on_shelter: float
    night: float
    predator_memory_dx: float
    predator_memory_dy: float
    predator_memory_age: float
    escape_memory_dx: float
    escape_memory_dy: float
    escape_memory_age: float


@dataclass(frozen=True)
class ActionContextObservation(ObservationView):
    observation_key: ClassVar[str] = "action_context"

    hunger: float
    fatigue: float
    health: float
    recent_pain: float
    recent_contact: float
    on_food: float
    on_shelter: float
    predator_visible: float
    predator_certainty: float
    predator_dist: float
    day: float
    night: float
    last_move_dx: float
    last_move_dy: float
    sleep_debt: float
    shelter_role_level: float


@dataclass(frozen=True)
class MotorContextObservation(ObservationView):
    observation_key: ClassVar[str] = "motor_context"

    on_food: float
    on_shelter: float
    predator_visible: float
    predator_certainty: float
    predator_dist: float
    day: float
    night: float
    last_move_dx: float
    last_move_dy: float
    shelter_role_level: float


@dataclass(frozen=True)
class ModuleInterface:
    """Contrato nomeado entre o mundo e um subsistema cortical."""

    name: str
    observation_key: str
    role: str
    inputs: tuple[SignalSpec, ...]
    outputs: tuple[str, ...] = tuple(LOCOMOTION_ACTIONS)
    version: int = 1
    description: str = ""
    save_compatibility: str = "exact_match_required"
    compatibility_notes: tuple[str, ...] = (
        "Mudanças em nome, observation_key, ordem dos sinais, versões ou outputs exigem incompatibilidade explícita.",
        "Checkpoints antigos são rejeitados; não há migração automática.",
    )

    @property
    def input_dim(self) -> int:
        return len(self.inputs)

    @property
    def output_dim(self) -> int:
        return len(self.outputs)

    @property
    def signal_names(self) -> tuple[str, ...]:
        """
        Get the ordered names of the module's input signals.
        
        Returns:
            signal_names (tuple[str, ...]): Tuple of input signal `name` values in the interface's defined order.
        """
        return tuple(signal.name for signal in self.inputs)

    def validate_signal_mapping(self, values: Mapping[str, float]) -> None:
        """
        Validate that the provided mapping contains exactly the interface's expected input signal names.
        
        Parameters:
            values (Mapping[str, float]): Mapping from input signal names to scalar values.
        
        Raises:
            ValueError: If any expected signal names are missing or any unexpected names are present.
        """
        _validate_exact_keys(f"Interface '{self.name}'", self.signal_names, values)

    def bind_values(self, values: Sequence[float]) -> dict[str, float]:
        """
        Bind an ordered sequence of input scalars to the interface's input signal names.
        
        Parameters:
            values (Sequence[float]): Sequence of numeric input values whose length must equal the interface's input_dim.
        
        Returns:
            dict[str, float]: Mapping from each input signal name to its corresponding scalar value.
        
        Raises:
            ValueError: If the provided sequence does not have the expected length (shape must equal (input_dim,)).
        """
        array = np.asarray(values, dtype=float)
        if array.shape != (self.input_dim,):
            raise ValueError(
                f"Interface '{self.name}' esperava shape {(self.input_dim,)}, recebeu {array.shape}."
            )
        return {
            signal.name: float(array[idx])
            for idx, signal in enumerate(self.inputs)
        }

    def vector_from_mapping(self, values: Mapping[str, float]) -> np.ndarray:
        """
        Build a numeric vector of input signals ordered by this interface's signal names.
        
        Parameters:
            values (Mapping[str, float]): Mapping from input signal name to its scalar value.
        
        Returns:
            np.ndarray: 1-D float array containing the signal values in the order of this interface's `signal_names`.
        """
        self.validate_signal_mapping(values)
        return np.array(
            [float(values[name]) for name in self.signal_names],
            dtype=float,
        )

    def to_summary(self) -> dict[str, object]:
        """
        Return a JSON-serializable schema entry for this interface.
        """
        return {
            "name": self.name,
            "observation_key": self.observation_key,
            "role": self.role,
            "version": int(self.version),
            "description": self.description,
            "signals": [signal.to_summary() for signal in self.inputs],
            "outputs": list(self.outputs),
            "compatibility": {
                "save_policy": self.save_compatibility,
                "exact_signal_order_required": True,
                "exact_output_order_required": True,
                "exact_observation_key_required": True,
                "notes": list(self.compatibility_notes),
            },
        }


MODULE_INTERFACES: tuple[ModuleInterface, ...] = (
    ModuleInterface(
        name="visual_cortex",
        observation_key="visual",
        role="proposal",
        description="Propositor visual orientado por comida, abrigo e predador dentro do campo visual local.",
        inputs=(
            SignalSpec("food_visible", "1 se há comida detectada com confiança suficiente no campo visual."),
            SignalSpec("food_certainty", "Confiança visual na comida percebida.", 0.0, 1.0),
            SignalSpec("food_occluded", "1 se a comida mais próxima está em alcance, mas ocluída.", 0.0, 1.0),
            SignalSpec("food_dx", "Deslocamento horizontal relativo da comida detectada."),
            SignalSpec("food_dy", "Deslocamento vertical relativo da comida detectada."),
            SignalSpec("shelter_visible", "1 se o abrigo está visível com confiança suficiente."),
            SignalSpec("shelter_certainty", "Confiança visual no abrigo percebido.", 0.0, 1.0),
            SignalSpec("shelter_occluded", "1 se o abrigo está em alcance visual, mas ocluído.", 0.0, 1.0),
            SignalSpec("shelter_dx", "Deslocamento horizontal relativo do abrigo detectado."),
            SignalSpec("shelter_dy", "Deslocamento vertical relativo do abrigo detectado."),
            SignalSpec("predator_visible", "1 se o predador está visível com confiança suficiente."),
            SignalSpec("predator_certainty", "Confiança visual no predador percebido.", 0.0, 1.0),
            SignalSpec("predator_occluded", "1 se o predador está em alcance visual, mas ocluído.", 0.0, 1.0),
            SignalSpec("predator_dx", "Deslocamento horizontal relativo do predador detectado."),
            SignalSpec("predator_dy", "Deslocamento vertical relativo do predador detectado."),
            SignalSpec("day", "1 durante o dia."),
            SignalSpec("night", "1 durante a noite."),
        ),
    ),
    ModuleInterface(
        name="sensory_cortex",
        observation_key="sensory",
        role="proposal",
        description="Propositor sensorial que integra dor, estado corporal, cheiro e luminosidade.",
        inputs=(
            SignalSpec("recent_pain", "Dor recente causada por ataque do predador.", 0.0, 1.0),
            SignalSpec("recent_contact", "Contato físico recente com o predador.", 0.0, 1.0),
            SignalSpec("health", "Saúde corporal.", 0.0, 1.0),
            SignalSpec("hunger", "Fome corporal.", 0.0, 1.0),
            SignalSpec("fatigue", "Fadiga corporal.", 0.0, 1.0),
            SignalSpec("food_smell_strength", "Intensidade do cheiro de alimento.", 0.0, 1.0),
            SignalSpec("food_smell_dx", "Gradiente horizontal do cheiro de alimento."),
            SignalSpec("food_smell_dy", "Gradiente vertical do cheiro de alimento."),
            SignalSpec("predator_smell_strength", "Intensidade do cheiro do predador.", 0.0, 1.0),
            SignalSpec("predator_smell_dx", "Gradiente horizontal do cheiro do predador."),
            SignalSpec("predator_smell_dy", "Gradiente vertical do cheiro do predador."),
            SignalSpec("light", "Nível simples de luminosidade.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="hunger_center",
        observation_key="hunger",
        role="proposal",
        description="Propositor homeostático voltado a forrageio e retorno à última comida percebida.",
        inputs=(
            SignalSpec("hunger", "Estado interno de fome.", 0.0, 1.0),
            SignalSpec("on_food", "1 se a aranha está sobre alimento.", 0.0, 1.0),
            SignalSpec("food_visible", "1 se há alimento visível.", 0.0, 1.0),
            SignalSpec("food_certainty", "Confiança visual no alimento percebido.", 0.0, 1.0),
            SignalSpec("food_occluded", "1 se há alimento próximo, mas ocluído.", 0.0, 1.0),
            SignalSpec("food_dx", "Direção horizontal do alimento detectado."),
            SignalSpec("food_dy", "Direção vertical do alimento detectado."),
            SignalSpec("food_smell_strength", "Intensidade do cheiro do alimento.", 0.0, 1.0),
            SignalSpec("food_smell_dx", "Gradiente horizontal do cheiro do alimento."),
            SignalSpec("food_smell_dy", "Gradiente vertical do cheiro do alimento."),
            SignalSpec("food_memory_dx", "Direção horizontal da última comida vista."),
            SignalSpec("food_memory_dy", "Direção vertical da última comida vista."),
            SignalSpec("food_memory_age", "Idade normalizada da memória de comida.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="sleep_center",
        observation_key="sleep",
        role="proposal",
        description="Propositor homeostático voltado a retorno ao abrigo, repouso e profundidade segura.",
        inputs=(
            SignalSpec("fatigue", "Estado interno de fadiga.", 0.0, 1.0),
            SignalSpec("hunger", "Estado interno de fome.", 0.0, 1.0),
            SignalSpec("on_shelter", "1 se a aranha está no abrigo.", 0.0, 1.0),
            SignalSpec("night", "1 durante a noite.", 0.0, 1.0),
            SignalSpec("home_dx", "Vetor interno horizontal até o abrigo."),
            SignalSpec("home_dy", "Vetor interno vertical até o abrigo."),
            SignalSpec("home_dist", "Distância normalizada até o abrigo.", 0.0, 1.0),
            SignalSpec("health", "Saúde corporal.", 0.0, 1.0),
            SignalSpec("recent_pain", "Dor recente.", 0.0, 1.0),
            SignalSpec("sleep_phase_level", "Nível atual da fase de sono.", 0.0, 1.0),
            SignalSpec("rest_streak_norm", "Continuidade recente do repouso.", 0.0, 1.0),
            SignalSpec("sleep_debt", "Dívida acumulada de sono.", 0.0, 1.0),
            SignalSpec("shelter_role_level", "Profundidade atual no abrigo.", 0.0, 1.0),
            SignalSpec("shelter_memory_dx", "Direção horizontal do abrigo seguro memorizado."),
            SignalSpec("shelter_memory_dy", "Direção vertical do abrigo seguro memorizado."),
            SignalSpec("shelter_memory_age", "Idade normalizada da memória do abrigo.", 0.0, 1.0),
        ),
    ),
    ModuleInterface(
        name="alert_center",
        observation_key="alert",
        role="proposal",
        description="Propositor defensivo voltado a ameaça, fuga e priorização do abrigo sob risco.",
        inputs=(
            SignalSpec("predator_visible", "1 se o predador está visível.", 0.0, 1.0),
            SignalSpec("predator_certainty", "Confiança visual no predador percebido.", 0.0, 1.0),
            SignalSpec("predator_occluded", "1 se o predador está em alcance, mas ocluído.", 0.0, 1.0),
            SignalSpec("predator_dx", "Direção horizontal do predador detectado."),
            SignalSpec("predator_dy", "Direção vertical do predador detectado."),
            SignalSpec("predator_dist", "Distância normalizada ao predador.", 0.0, 1.0),
            SignalSpec("predator_smell_strength", "Intensidade do cheiro do predador.", 0.0, 1.0),
            SignalSpec("home_dx", "Vetor interno horizontal até o abrigo."),
            SignalSpec("home_dy", "Vetor interno vertical até o abrigo."),
            SignalSpec("recent_pain", "Dor recente.", 0.0, 1.0),
            SignalSpec("recent_contact", "Contato físico recente.", 0.0, 1.0),
            SignalSpec("on_shelter", "1 se a aranha está no abrigo.", 0.0, 1.0),
            SignalSpec("night", "1 durante a noite.", 0.0, 1.0),
            SignalSpec("predator_memory_dx", "Direção horizontal da última posição vista do predador."),
            SignalSpec("predator_memory_dy", "Direção vertical da última posição vista do predador."),
            SignalSpec("predator_memory_age", "Idade normalizada da memória do predador.", 0.0, 1.0),
            SignalSpec("escape_memory_dx", "Direção horizontal da rota recente de fuga."),
            SignalSpec("escape_memory_dy", "Direção vertical da rota recente de fuga."),
            SignalSpec("escape_memory_age", "Idade normalizada da memória de fuga.", 0.0, 1.0),
        ),
    ),
)


ACTION_CONTEXT_INTERFACE = ModuleInterface(
    name="action_center_context",
    observation_key="action_context",
    role="context",
    description="Contexto bruto usado pelo action_center para arbitrar propostas locomotoras concorrentes.",
    inputs=(
        SignalSpec("hunger", "Fome corporal.", 0.0, 1.0),
        SignalSpec("fatigue", "Fadiga corporal.", 0.0, 1.0),
        SignalSpec("health", "Saúde corporal.", 0.0, 1.0),
        SignalSpec("recent_pain", "Dor recente.", 0.0, 1.0),
        SignalSpec("recent_contact", "Contato recente com o predador.", 0.0, 1.0),
        SignalSpec("on_food", "1 se está sobre comida.", 0.0, 1.0),
        SignalSpec("on_shelter", "1 se está sobre o abrigo.", 0.0, 1.0),
        SignalSpec("predator_visible", "1 se o predador está visível.", 0.0, 1.0),
        SignalSpec("predator_certainty", "Confiança visual no predador.", 0.0, 1.0),
        SignalSpec("predator_dist", "Distância normalizada ao predador.", 0.0, 1.0),
        SignalSpec("day", "1 durante o dia.", 0.0, 1.0),
        SignalSpec("night", "1 durante a noite.", 0.0, 1.0),
        SignalSpec("last_move_dx", "Último deslocamento horizontal realizado."),
        SignalSpec("last_move_dy", "Último deslocamento vertical realizado."),
        SignalSpec("sleep_debt", "Dívida acumulada de sono.", 0.0, 1.0),
        SignalSpec("shelter_role_level", "Profundidade atual no abrigo.", 0.0, 1.0),
    ),
)


MOTOR_CONTEXT_INTERFACE = ModuleInterface(
    name="motor_cortex_context",
    observation_key="motor_context",
    role="context",
    description="Contexto bruto usado pelo motor_cortex para corrigir e executar a intenção locomotora.",
    inputs=(
        SignalSpec("on_food", "1 se está sobre comida.", 0.0, 1.0),
        SignalSpec("on_shelter", "1 se está sobre o abrigo.", 0.0, 1.0),
        SignalSpec("predator_visible", "1 se o predador está visível.", 0.0, 1.0),
        SignalSpec("predator_certainty", "Confiança visual no predador.", 0.0, 1.0),
        SignalSpec("predator_dist", "Distância normalizada ao predador.", 0.0, 1.0),
        SignalSpec("day", "1 durante o dia.", 0.0, 1.0),
        SignalSpec("night", "1 durante a noite.", 0.0, 1.0),
        SignalSpec("last_move_dx", "Último deslocamento horizontal realizado."),
        SignalSpec("last_move_dy", "Último deslocamento vertical realizado."),
        SignalSpec("shelter_role_level", "Profundidade atual no abrigo.", 0.0, 1.0),
    ),
)


ALL_INTERFACES: tuple[ModuleInterface, ...] = MODULE_INTERFACES + (
    ACTION_CONTEXT_INTERFACE,
    MOTOR_CONTEXT_INTERFACE,
)

MODULE_INTERFACE_BY_NAME = {spec.name: spec for spec in MODULE_INTERFACES}
OBSERVATION_INTERFACE_BY_KEY: Dict[str, ModuleInterface] = {
    spec.observation_key: spec for spec in ALL_INTERFACES
}


def _build_observation_view_registry() -> Dict[str, type[ObservationView]]:
    registry: Dict[str, type[ObservationView]] = {}
    for view_cls in ObservationView.__subclasses__():
        key = view_cls.observation_key
        if not key:
            raise ValueError(f"ObservationView '{view_cls.__name__}' precisa declarar observation_key.")
        if key in registry:
            raise ValueError(f"ObservationView duplicada para observation_key '{key}'.")
        registry[key] = view_cls
    return registry


OBSERVATION_VIEW_BY_KEY: Dict[str, type[ObservationView]] = _build_observation_view_registry()
OBSERVATION_DIMS: Dict[str, int] = {
    spec.observation_key: spec.input_dim for spec in ALL_INTERFACES
}


INTERFACE_REGISTRY_SCHEMA_VERSION = 1


def _stable_json(payload: object) -> str:
    """
    Return a deterministic JSON encoding suitable for fingerprints.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fingerprint_payload(payload: object) -> str:
    """
    Compute a stable SHA-256 fingerprint for a JSON-serializable payload.
    """
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def interface_registry() -> dict[str, object]:
    """
    Return the machine-readable registry that governs all observation interfaces.
    """
    return {
        "schema_version": INTERFACE_REGISTRY_SCHEMA_VERSION,
        "actions": list(LOCOMOTION_ACTIONS),
        "proposal_interfaces": [spec.name for spec in MODULE_INTERFACES],
        "context_interfaces": [
            ACTION_CONTEXT_INTERFACE.name,
            MOTOR_CONTEXT_INTERFACE.name,
        ],
        "interfaces": {
            spec.name: spec.to_summary()
            for spec in ALL_INTERFACES
        },
        "observation_views": {
            key: {
                "field_names": list(view_cls.field_names()),
            }
            for key, view_cls in sorted(OBSERVATION_VIEW_BY_KEY.items())
        },
    }


def interface_registry_fingerprint() -> str:
    """
    Return a stable fingerprint for the current interface registry.
    """
    return _fingerprint_payload(interface_registry())


def render_interfaces_markdown() -> str:
    """
    Render the interface registry into the generated documentation page.
    """
    registry = interface_registry()
    lines: list[str] = [
        "# Interfaces Padronizadas",
        "",
        "<!-- Generated from spider_cortex_sim.interfaces.render_interfaces_markdown(); do not edit manually. -->",
        "",
        f"- Schema version: `{registry['schema_version']}`",
        f"- Registry fingerprint: `{interface_registry_fingerprint()}`",
        f"- Proposal interfaces: `{', '.join(registry['proposal_interfaces'])}`",
        f"- Context interfaces: `{', '.join(registry['context_interfaces'])}`",
        "",
        "## Compatibilidade",
        "",
        "- A política atual é `exact_match_required` para save/load.",
        "- Mudanças em nome, `observation_key`, ordem dos sinais, outputs ou versão exigem incompatibilidade explícita.",
        "- Não existe migração automática de checkpoints antigos.",
        "",
    ]
    for spec in ALL_INTERFACES:
        summary = spec.to_summary()
        compatibility = summary["compatibility"]
        lines.extend(
            [
                f"## `{spec.name}`",
                "",
                f"- Observation key: `{spec.observation_key}`",
                f"- Role: `{spec.role}`",
                f"- Version: `{spec.version}`",
                f"- Description: {spec.description}",
                f"- Outputs: `{', '.join(spec.outputs)}`",
                f"- Save policy: `{compatibility['save_policy']}`",
                "",
                "| # | Signal | Min | Max | Description |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for index, signal in enumerate(spec.inputs, start=1):
            lines.append(
                f"| {index} | `{signal.name}` | {signal.minimum:.1f} | {signal.maximum:.1f} | {signal.description} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def architecture_signature(
    *,
    proposal_backend: str = "modular",
    proposal_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """
    Return a JSON-serializable description of the agent architecture.
    
    Parameters:
        proposal_backend (str): Proposal stage topology. Supported values are `"modular"` and `"monolithic"`.
        proposal_order (Sequence[str] | None): Ordered proposal sources injected into the action-center input. When omitted,
            the modular layout defaults to all `MODULE_INTERFACES` names and the monolithic layout defaults to `["monolithic_policy"]`.

    The returned dictionary includes:
    - "actions": list of locomotion action names.
    - "modules": mapping from each module interface name to its observation key, ordered input signal names, and outputs.
    - "proposal_order": deterministic proposal slot order used when concatenating proposal logits into the action-center input.
    - "action_center": the action-center context specification with its observation key, raw context input names,
      inter-stage proposal inputs, locomotion outputs, proposal slot metadata, and `value_head` set to True.
    - "motor_cortex": the motor-cortex context specification with its observation key, raw context input names,
      inter-stage intent input metadata, and locomotion outputs with `value_head` set to False.
    
    Returns:
        dict[str, object]: Top-level architecture signature containing "actions", "modules", "proposal_order", "action_center", and "motor_cortex" as described above.
    """
    if proposal_backend not in {"modular", "monolithic"}:
        raise ValueError("proposal_backend deve ser 'modular' ou 'monolithic'.")
    if proposal_order is None:
        if proposal_backend == "modular":
            proposal_order = [spec.name for spec in MODULE_INTERFACES]
        else:
            proposal_order = ["monolithic_policy"]
    proposal_order = [str(name) for name in proposal_order]
    registry = interface_registry()
    proposal_interfaces = registry["interfaces"]
    payload = {
        "schema_version": INTERFACE_REGISTRY_SCHEMA_VERSION,
        "proposal_backend": proposal_backend,
        "registry_fingerprint": interface_registry_fingerprint(),
        "actions": list(LOCOMOTION_ACTIONS),
        "modules": {
            spec.name: proposal_interfaces[spec.name]
            for spec in MODULE_INTERFACES
        },
        "contexts": {
            ACTION_CONTEXT_INTERFACE.name: proposal_interfaces[ACTION_CONTEXT_INTERFACE.name],
            MOTOR_CONTEXT_INTERFACE.name: proposal_interfaces[MOTOR_CONTEXT_INTERFACE.name],
        },
        "interface_versions": {
            spec.name: int(spec.version)
            for spec in ALL_INTERFACES
        },
        "proposal_order": proposal_order,
        "action_center": {
            "context_interface": ACTION_CONTEXT_INTERFACE.name,
            "observation_key": ACTION_CONTEXT_INTERFACE.observation_key,
            "version": ACTION_CONTEXT_INTERFACE.version,
            "inputs": [signal.name for signal in ACTION_CONTEXT_INTERFACE.inputs],
            "inter_stage_inputs": [
                {
                    "name": "proposal_logits",
                    "source": name,
                    "encoding": "logits",
                    "size": len(LOCOMOTION_ACTIONS),
                    "signals": list(LOCOMOTION_ACTIONS),
                }
                for name in proposal_order
            ],
            "outputs": list(LOCOMOTION_ACTIONS),
            "proposal_slots": [
                {
                    "module": name,
                    "actions": list(LOCOMOTION_ACTIONS),
                }
                for name in proposal_order
            ],
            "value_head": True,
        },
        "motor_cortex": {
            "context_interface": MOTOR_CONTEXT_INTERFACE.name,
            "observation_key": MOTOR_CONTEXT_INTERFACE.observation_key,
            "version": MOTOR_CONTEXT_INTERFACE.version,
            "inputs": [signal.name for signal in MOTOR_CONTEXT_INTERFACE.inputs],
            "inter_stage_inputs": [
                {
                    "name": "intent",
                    "source": "action_center",
                    "encoding": "one_hot",
                    "size": len(LOCOMOTION_ACTIONS),
                    "signals": list(LOCOMOTION_ACTIONS),
                }
            ],
            "outputs": list(LOCOMOTION_ACTIONS),
            "value_head": False,
        },
    }
    payload["fingerprint"] = _fingerprint_payload(payload)
    return payload
