from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Dict, List, Mapping, Sequence, Union

import numpy as np

from .interfaces import (
    MODULE_INTERFACES,
    MODULE_INTERFACE_BY_NAME,
    ModuleInterface,
    VARIANT_MODULES,
    get_interface_variant,
    get_variant_levels,
)
from .nn import ProposalNetwork, RecurrentProposalNetwork, softmax

logger = logging.getLogger(__name__)


MODULE_HIDDEN_DIMS: Dict[str, int] = {
    "visual_cortex": 32,
    "sensory_cortex": 28,
    "hunger_center": 26,
    "sleep_center": 24,
    "alert_center": 28,
    "perception_center": 36,
    "homeostasis_center": 28,
    "threat_center": 28,
}

VARIANT_HIDDEN_DIMS: Dict[tuple[str, int], int] = {
    (module_name, level): (
        MODULE_HIDDEN_DIMS[module_name]
        if level == 4
        else max(
            1,
            ceil(
                MODULE_HIDDEN_DIMS[module_name]
                * (
                    get_interface_variant(module_name, level).input_dim
                    / MODULE_INTERFACE_BY_NAME[module_name].input_dim
                )
            ),
        )
    )
    for module_name in VARIANT_MODULES
    for level in get_variant_levels(module_name)
}


def get_variant_hidden_dim(module_name: str, level: int) -> int:
    """
    Return the hidden size assigned to a diagnostic interface variant.
    """
    try:
        return VARIANT_HIDDEN_DIMS[(module_name, level)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown variant hidden dim for module '{module_name}' at level {level}."
        ) from exc


@dataclass(frozen=True)
class ModuleSpec:
    interface: ModuleInterface
    hidden_dim: int

    @property
    def name(self) -> str:
        return self.interface.name

    @property
    def observation_key(self) -> str:
        return self.interface.observation_key

    @property
    def input_dim(self) -> int:
        return self.interface.input_dim


@dataclass
class ReflexDecision:
    action: str
    target_probs: np.ndarray
    reason: str
    triggers: Dict[str, float]
    auxiliary_weight: float
    logit_strength: float

    def to_payload(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "target_probs": self.target_probs.round(6).tolist(),
            "reason": self.reason,
            "triggers": {
                name: round(float(value), 6)
                for name, value in sorted(self.triggers.items())
            },
            "auxiliary_weight": round(float(self.auxiliary_weight), 6),
            "logit_strength": round(float(self.logit_strength), 6),
        }


@dataclass
class ModuleResult:
    interface: ModuleInterface | None
    name: str
    observation_key: str
    observation: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    active: bool
    reflex: ReflexDecision | None = None
    neural_logits: np.ndarray | None = None
    reflex_delta_logits: np.ndarray | None = None
    post_reflex_logits: np.ndarray | None = None
    reflex_applied: bool = False
    effective_reflex_scale: float = 0.0
    module_reflex_override: bool = False
    module_reflex_dominance: float = 0.0
    valence_role: str = "support"
    gate_weight: float = 1.0
    contribution_share: float = 0.0
    gated_logits: np.ndarray | None = None
    intent_before_gating: str | None = None
    intent_after_gating: str | None = None

    def named_observation(self) -> Dict[str, float]:
        """
        Return the module's observation values mapped to the interface's named observation keys.
        
        If the module has no associated interface, an empty dictionary is returned.
        
        Returns:
            Dict[str, float]: Mapping from observation field names to their float values; empty if no interface is set.
        """
        if self.interface is None:
            return {}
        return self.interface.bind_values(self.observation)


DEFAULT_MODULE_SPECS: List[ModuleSpec] = [
    ModuleSpec(interface=interface, hidden_dim=MODULE_HIDDEN_DIMS[interface.name])
    for interface in MODULE_INTERFACES
]


def _validated_hidden_dims(
    hidden_dims: Mapping[str, int] | None = None,
) -> dict[str, int]:
    if hidden_dims is None:
        return {}
    normalized_hidden_dims = dict(hidden_dims)
    unknown_names = sorted(
        str(name)
        for name in normalized_hidden_dims
        if str(name) not in MODULE_HIDDEN_DIMS
    )
    if unknown_names:
        raise ValueError(f"Unknown module hidden-dim overrides: {unknown_names}")
    validated: dict[str, int] = {}
    for raw_name, raw_hidden_dim in normalized_hidden_dims.items():
        name = str(raw_name)
        if isinstance(raw_hidden_dim, bool) or not isinstance(raw_hidden_dim, int):
            raise ValueError(
                f"hidden_dims[{name!r}] must be an int > 0; got {raw_hidden_dim!r}."
            )
        if raw_hidden_dim <= 0:
            raise ValueError(
                f"hidden_dims[{name!r}] must be an int > 0; got {raw_hidden_dim!r}."
            )
        validated[name] = int(raw_hidden_dim)
    return validated


def module_specs_from_hidden_dims(
    hidden_dims: Mapping[str, int] | None = None,
) -> List[ModuleSpec]:
    merged_hidden_dims = dict(MODULE_HIDDEN_DIMS)
    merged_hidden_dims.update(_validated_hidden_dims(hidden_dims))
    return [
        ModuleSpec(interface=interface, hidden_dim=merged_hidden_dims[interface.name])
        for interface in MODULE_INTERFACES
    ]

if TYPE_CHECKING:
    from typing import TypeAlias

    ModuleNetwork: TypeAlias = ProposalNetwork | RecurrentProposalNetwork
else:
    ModuleNetwork = Union[ProposalNetwork, RecurrentProposalNetwork]


class CorticalModuleBank:
    """Runs only the proposer networks; it does not decide reflexes or integrate global context."""

    def __init__(
        self,
        action_dim: int,
        rng: np.random.Generator,
        module_dropout: float = 0.05,
        disabled_modules: Sequence[str] = (),
        recurrent_modules: Sequence[str] = (),
        hidden_dims: Mapping[str, int] | None = None,
    ) -> None:
        """
        Create a CorticalModuleBank configured to produce proposals for each default module spec.
        
        Parameters:
            action_dim (int): Size of the action/output vector produced by each proposal network.
            rng (np.random.Generator): Random number generator used for network initialization and any stochastic behavior.
            module_dropout (float): Probability (0.0-1.0) that a module will be dropped during training; stored as a float.
            disabled_modules (Sequence[str]): Iterable of module names to mark as permanently disabled; converted to a set of strings internally.
            recurrent_modules (Sequence[str]): Iterable of module names whose proposal networks should maintain recurrent hidden state.
        """
        self.specs = module_specs_from_hidden_dims(hidden_dims)
        self.action_dim = action_dim
        self.rng = rng
        self.module_dropout = float(module_dropout)
        self.disabled_modules = {str(name) for name in disabled_modules}
        self.recurrent_modules = {str(name) for name in recurrent_modules}
        known_names = {spec.name for spec in self.specs}
        unknown = self.disabled_modules - known_names
        if unknown:
            raise ValueError(f"Invalid disabled modules: {sorted(unknown)}")
        unknown_recurrent = self.recurrent_modules - known_names
        if unknown_recurrent:
            raise ValueError(f"Invalid recurrent modules: {sorted(unknown_recurrent)}")
        self.enabled_specs = [
            spec for spec in self.specs if spec.name not in self.disabled_modules
        ]
        if not self.enabled_specs:
            raise ValueError(
                "Invalid disabled modules: all modules were disabled; at least one module must be enabled."
            )
        self.modules: Dict[str, ModuleNetwork] = {
            spec.name: (
                RecurrentProposalNetwork
                if spec.name in self.recurrent_modules
                else ProposalNetwork
            )(
                input_dim=spec.input_dim,
                hidden_dim=spec.hidden_dim,
                output_dim=action_dim,
                rng=rng,
                name=spec.name,
            )
            for spec in self.enabled_specs
        }
        self._active_names: List[str] = [spec.name for spec in self.enabled_specs]

    @property
    def has_recurrent_modules(self) -> bool:
        return bool(self.recurrent_modules)

    @staticmethod
    def _has_hidden_state_api(network: ModuleNetwork) -> bool:
        return (
            callable(getattr(network, "get_hidden_state", None))
            and callable(getattr(network, "set_hidden_state", None))
            and callable(getattr(network, "reset_hidden_state", None))
        )

    def reset_hidden_states(self) -> None:
        """Reset hidden state on all recurrent proposal networks."""
        for network in self.modules.values():
            if self._has_hidden_state_api(network):
                network.reset_hidden_state()

    def snapshot_hidden_states(self) -> Dict[str, np.ndarray]:
        """Return copies of runtime hidden states for recurrent modules."""
        snapshots: Dict[str, np.ndarray] = {}
        for name, network in self.modules.items():
            if self._has_hidden_state_api(network):
                snapshots[name] = np.asarray(network.get_hidden_state(), dtype=float).copy()
        return snapshots

    def restore_hidden_states(self, snapshots: Dict[str, np.ndarray]) -> None:
        """Restore recurrent runtime hidden states from a previous snapshot."""
        for name, hidden_state in snapshots.items():
            if name not in self.modules:
                raise KeyError(f"Module '{name}' does not exist in the module bank.")
            network = self.modules[name]
            if not self._has_hidden_state_api(network):
                raise ValueError(f"Module '{name}' does not have recurrent hidden state.")
            network.set_hidden_state(np.asarray(hidden_state, dtype=float))

    def forward(
        self,
        observation: Dict[str, np.ndarray],
        *,
        store_cache: bool = True,
        training: bool = False,
    ) -> List[ModuleResult]:
        """
        Compute per-module proposal outputs for the given observation.
        
        Sanitizes inputs (NaN→0.0, +inf→1.0, -inf→-1.0), runs each module's proposal network to produce logits (NaN→0.0, +inf→20.0, -inf→-20.0), clips logits to the range [-20.0, 20.0], and converts them to probabilities with softmax. Inactive modules produce a zeroed logit vector; each ModuleResult also includes copies for neural_logits and post_reflex_logits and an initialized reflex_delta_logits vector.
        
        Parameters:
            observation (Dict[str, np.ndarray]):
                Mapping from observation keys to numpy arrays; values are sanitized before being passed to module networks.
            store_cache (bool):
                If true, cache the list of active module names to self._active_names for later backward passes.
            training (bool):
                If true and module_dropout > 0, randomly drops modules with probability self.module_dropout while ensuring at least one eligible module remains active.
        
        Returns:
            List[ModuleResult]:
                A list of ModuleResult entries, one per module spec, containing the module's interface (or None), name, observation_key, a copy of the sanitized observation, effective logits, softmaxed probabilities, active flag, and initialized reflex-related fields.
        """
        outputs: List[ModuleResult] = []
        active_names: List[str] = []

        drop_mask = np.zeros(len(self.enabled_specs), dtype=bool)
        eligible_indices = list(range(len(self.enabled_specs)))
        if training and self.module_dropout > 0.0 and len(self.enabled_specs) > 1:
            drop_mask = self.rng.random(len(self.enabled_specs)) < self.module_dropout
            if eligible_indices and all(drop_mask[idx] for idx in eligible_indices):
                keep_idx = eligible_indices[int(self.rng.integers(0, len(eligible_indices)))]
                drop_mask[keep_idx] = False

        for idx, spec in enumerate(self.enabled_specs):
            obs = np.nan_to_num(
                np.asarray(observation[spec.observation_key], dtype=float),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            active = not bool(drop_mask[idx])
            if active:
                logits = self.modules[spec.name].forward(
                    obs,
                    store_cache=store_cache,
                )
                effective_logits = np.clip(
                    np.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0),
                    -20.0,
                    20.0,
                )
                active_names.append(spec.name)
            else:
                effective_logits = np.zeros(
                    self.action_dim,
                    dtype=float,
                )
            outputs.append(
                ModuleResult(
                    interface=spec.interface,
                    name=spec.name,
                    observation_key=spec.observation_key,
                    observation=obs.copy(),
                    logits=effective_logits,
                    probs=softmax(effective_logits),
                    active=active,
                    neural_logits=effective_logits.copy(),
                    reflex_delta_logits=np.zeros_like(effective_logits),
                    post_reflex_logits=effective_logits.copy(),
                )
            )

        if store_cache:
            self._active_names = active_names
        return outputs

    def backward(
        self,
        grad_logits: np.ndarray,
        lr: float,
        aux_grads: Dict[str, np.ndarray] | None = None,
    ) -> List[str]:
        updated_names: List[str] = []
        for name in self._active_names:
            total_grad = np.asarray(grad_logits, dtype=float).copy()
            if aux_grads is not None and name in aux_grads:
                total_grad += np.asarray(aux_grads[name], dtype=float)
            self.modules[name].backward(total_grad, lr=lr)
            updated_names.append(name)
        return updated_names

    def state_dict(self) -> Dict[str, object]:
        return {name: net.state_dict() for name, net in self.modules.items()}

    def load_state_dict(self, state: Dict[str, object], *, modules: List[str] | None = None) -> List[str]:
        """
        Load parameter state dictionaries into module networks.
        
        Parameters:
            state (Dict[str, object]): Mapping from module name to its saved state dictionary.
            modules (List[str] | None): Optional list of module names to load. If None, loads all names present in `state`.
                Entries whose names are not present in `self.modules` are skipped, so absent bank
                modules do not raise during loading.
        
        Returns:
            List[str]: Names of modules whose state was loaded successfully, in the order processed.
        
        Raises:
            KeyError: If a requested `name` is not present in `state` ("Module '<name>' not found in the state_dict.").
        """
        loaded: List[str] = []
        targets = modules if modules is not None else list(state.keys())
        for name in targets:
            if name not in state:
                raise KeyError(f"Module '{name}' not found in the state_dict.")
            if name not in self.modules:
                logger.debug(
                    "Skipping module %r while loading state_dict: not present in current module bank.",
                    name,
                )
                continue
            self.modules[name].load_state_dict(state[name])
            loaded.append(name)
        return loaded

    def parameter_norms(self) -> Dict[str, float]:
        return {name: net.parameter_norm() for name, net in self.modules.items()}

    def parameter_counts(self) -> Dict[str, int]:
        return {name: net.count_parameters() for name, net in self.modules.items()}
