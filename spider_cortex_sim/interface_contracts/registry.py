from __future__ import annotations

import hashlib
import json
from typing import Dict, Mapping, Sequence

from .actions import LOCOMOTION_ACTIONS
from .module_specs import (
    ACTION_CONTEXT_INTERFACE,
    ALL_INTERFACES,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from .observations import ObservationView
from .variants import VARIANT_INTERFACES, get_interface_variant


def _variant_interface_specs() -> tuple[object, ...]:
    return tuple(sorted(VARIANT_INTERFACES, key=lambda spec: spec.name))

def _build_observation_view_registry() -> Dict[str, type[ObservationView]]:
    """
    Collect all direct ObservationView subclasses and index them by their declared observation_key.
    
    Builds a dictionary that maps each subclass's non-empty `observation_key` to the subclass type.
    
    Returns:
        dict: Mapping from an observation key string to the corresponding `ObservationView` subclass.
    
    Raises:
        ValueError: If a subclass has an empty or falsey `observation_key`, or if multiple subclasses declare the same `observation_key`.
    """
    registry: Dict[str, type[ObservationView]] = {}
    for view_cls in ObservationView.__subclasses__():
        key = view_cls.observation_key
        if not key:
            raise ValueError(f"ObservationView '{view_cls.__name__}' must declare observation_key.")
        if key in registry:
            raise ValueError(f"Duplicate ObservationView for observation_key '{key}'.")
        registry[key] = view_cls
    return registry


OBSERVATION_VIEW_BY_KEY: Dict[str, type[ObservationView]] = _build_observation_view_registry()
OBSERVATION_DIMS: Dict[str, int] = {
    spec.observation_key: spec.input_dim for spec in ALL_INTERFACES
}


INTERFACE_REGISTRY_SCHEMA_VERSION = 1
ARBITRATION_EVIDENCE_INPUT_DIM = 24
ARBITRATION_HIDDEN_DIM = 32
ARBITRATION_VALENCE_DIM = 4
ARBITRATION_GATE_DIM = 9


def _stable_json(payload: object) -> str:
    """
    Produce a deterministic, compact JSON representation of `payload` suitable for stable hashing.
    
    The serialization sorts object keys, uses compact separators without extra whitespace, and preserves non-ASCII characters.
    
    Returns:
        json_str (str): JSON string representation of `payload`.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fingerprint_payload(payload: object) -> str:
    """
    Create a deterministic SHA-256 hex fingerprint for a JSON-serializable payload.
    
    Parameters:
        payload (object): A JSON-serializable object whose stable representation will be fingerprinted.
    
    Returns:
        str: Hexadecimal SHA-256 digest of the payload's stable JSON representation.
    """
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def interface_registry() -> dict[str, object]:
    """
    Produce a deterministic, machine-readable registry describing available interfaces and observation views.
    
    Returns:
        registry (dict): A dictionary with the following keys:
            - "schema_version" (int): Registry schema version.
            - "actions" (list[str]): Locomotion action names.
            - "proposal_interfaces" (list[str]): Names of module/proposal interfaces.
            - "context_interfaces" (list[str]): Names for the action and motor context interfaces.
            - "interfaces" (dict[str, object]): Mapping from interface name to its summary (as produced by each interface's `to_summary()`).
            - "observation_views" (dict[str, dict]): Mapping from observation key to a dict with:
                - "field_names" (list[str]): Ordered field names exposed by the corresponding ObservationView.
    """
    proposal_specs = (*MODULE_INTERFACES, *_variant_interface_specs())
    all_specs = (*ALL_INTERFACES, *_variant_interface_specs())
    return {
        "schema_version": INTERFACE_REGISTRY_SCHEMA_VERSION,
        "actions": list(LOCOMOTION_ACTIONS),
        "proposal_interfaces": [spec.name for spec in proposal_specs],
        "context_interfaces": [
            ACTION_CONTEXT_INTERFACE.name,
            MOTOR_CONTEXT_INTERFACE.name,
        ],
        "interfaces": {
            spec.name: spec.to_summary()
            for spec in all_specs
        },
        "observation_views": {
            key: {
                "field_names": list(view_cls.field_names()),
            }
            for key, view_cls in sorted(OBSERVATION_VIEW_BY_KEY.items())
        },
    }


def _validate_proposal_order(
    proposal_backend: str,
    proposal_order: Sequence[str],
) -> list[str]:
    resolved_order = [str(name) for name in proposal_order]
    if proposal_backend == "modular":
        allowed = {spec.name for spec in MODULE_INTERFACES}
    elif proposal_backend == "true_monolithic":
        allowed = {"true_monolithic_policy"}
    else:
        allowed = {"monolithic_policy"}
    unknown = sorted({name for name in resolved_order if name not in allowed})
    if unknown:
        raise ValueError(
            f"Unknown proposal_order entries for {proposal_backend!r}: {unknown}."
        )
    duplicates = sorted(
        {name for name in resolved_order if resolved_order.count(name) > 1}
    )
    if duplicates:
        raise ValueError(
            f"Duplicate proposal_order entries for {proposal_backend!r}: {duplicates}."
        )
    return resolved_order


def interface_registry_fingerprint(registry: Mapping[str, object] | None = None) -> str:
    """
    Compute a stable fingerprint of the current interface registry.
    
    The fingerprint is the SHA-256 hex digest of a deterministic JSON serialization of the provided registry, or the current interface_registry() payload when omitted.
    
    Returns:
        fingerprint (str): Hexadecimal SHA-256 digest representing the current interface registry.
    """
    return _fingerprint_payload(interface_registry() if registry is None else registry)


def architecture_signature(
    *,
    proposal_backend: str = "modular",
    proposal_order: Sequence[str] | None = None,
    module_variants: Mapping[str, int] | None = None,
    learned_arbitration: bool = True,
    arbitration_input_dim: int = ARBITRATION_EVIDENCE_INPUT_DIM,
    arbitration_hidden_dim: int | None = None,
    arbitration_regularization_weight: float = 0.1,
    capacity_profile_name: str = "current",
    module_hidden_dims: Mapping[str, int] | None = None,
    action_center_hidden_dim: int | None = None,
    motor_hidden_dim: int | None = None,
    integration_hidden_dim: int | None = None,
    monolithic_hidden_dim: int | None = None,
    capacity_profile: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Produce a deterministic, JSON-serializable description of the agent's observation/action architecture and its derived fingerprint.
    
    Parameters:
        proposal_backend (str): Proposal-stage topology; must be "modular", "monolithic", or "true_monolithic".
        proposal_order (Sequence[str] | None): Ordered proposal sources. If None, defaults by backend: for "modular" the declaration order of MODULE_INTERFACES, for "monolithic" ["monolithic_policy"], for "true_monolithic" ["true_monolithic_policy"].
        module_variants (Mapping[str, int] | None): Mapping from module name to integer diagnostic variant level; levels are normalized into interface name/version entries.
        learned_arbitration (bool): Whether arbitration uses a learned network (true) or a fixed-formula baseline (false).
        arbitration_input_dim (int): Dimensionality of the concatenated arbitration evidence input.
        arbitration_hidden_dim (int | None): Dimensionality of the arbitration network's shared hidden layer; if None, set to integration_hidden_dim and then to the default arbitration hidden dimension.
        arbitration_regularization_weight (float): Regularization weight applied to arbitration gating.
        capacity_profile_name (str): Identifier string for the capacity profile to embed in the payload.
        module_hidden_dims (Mapping[str, int] | None): Per-module hidden-dimension overrides to include in the capacity section.
        action_center_hidden_dim (int | None): Override hidden dimension for the action center; if None, set to integration_hidden_dim.
        motor_hidden_dim (int | None): Override hidden dimension for the motor cortex; if None, set to integration_hidden_dim.
        integration_hidden_dim (int | None): Default integration hidden dimension used when per-component overrides are absent.
        monolithic_hidden_dim (int | None): Hidden-dimension override for monolithic/true-monolithic direct policy.
        capacity_profile (Mapping[str, object] | None): Arbitrary capacity-profile metadata to embed under the "capacity" key.
    
    Returns:
        dict[str, object]: A payload describing the architecture, including:
            - "schema_version": registry schema version
            - "proposal_backend": chosen backend
            - "registry_fingerprint": fingerprint of the interface registry
            - "actions": list of locomotion actions
            - "modules": summaries for proposal modules
            - "contexts": summaries for action and motor context interfaces
            - "interface_versions": numeric interface versions
            - "module_variants": normalized variant entries when provided
            - "proposal_order": resolved ordered list of proposal sources
            - "capacity": optional capacity metadata and hidden-dimension settings
            - backend-specific sections: "arbitration_network" / "action_center" / "motor_cortex" or "direct_policy"
            - "fingerprint": SHA-256 fingerprint of the returned payload
    """
    if proposal_backend not in {"modular", "monolithic", "true_monolithic"}:
        raise ValueError(
            "proposal_backend must be 'modular', 'monolithic', or 'true_monolithic'."
        )
    if proposal_order is None:
        if proposal_backend == "modular":
            proposal_order = [spec.name for spec in MODULE_INTERFACES]
        elif proposal_backend == "true_monolithic":
            proposal_order = ["true_monolithic_policy"]
        else:
            proposal_order = ["monolithic_policy"]
    proposal_order = _validate_proposal_order(proposal_backend, proposal_order)
    normalized_module_variants: dict[str, object] = {}
    if module_variants is not None:
        for raw_name, raw_level in dict(module_variants).items():
            module_name = str(raw_name)
            try:
                level = int(raw_level)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"module_variants[{module_name!r}] must be an integer variant level."
                ) from exc
            interface = get_interface_variant(module_name, level)
            normalized_module_variants[module_name] = {
                "level": level,
                "interface": interface.name,
                "version": int(interface.version),
            }
    arbitration_module_roles = {
        "alert_center": "threat",
        "hunger_center": "hunger",
        "sleep_center": "sleep",
        "visual_cortex": "support",
        "sensory_cortex": "support",
        "perception_center": "support",
        "homeostasis_center": "hunger",
        "threat_center": "threat",
        "monolithic_policy": "integrated_policy",
        "true_monolithic_policy": "integrated_policy",
    }
    if proposal_backend in {"monolithic", "true_monolithic"}:
        filtered_module_roles = (
            {
                name: arbitration_module_roles[name]
                for name in proposal_order
                if name in arbitration_module_roles
            }
            if proposal_order
            else {}
        )
    else:
        filtered_module_roles = {
            name: arbitration_module_roles[name]
            for name in proposal_order
            if name in arbitration_module_roles
        }
    arbitration_input_dim = int(arbitration_input_dim)
    if integration_hidden_dim is not None:
        integration_hidden_dim = int(integration_hidden_dim)
    if action_center_hidden_dim is None:
        action_center_hidden_dim = integration_hidden_dim
    if motor_hidden_dim is None:
        motor_hidden_dim = integration_hidden_dim
    resolved_arbitration_hidden_dim = (
        int(arbitration_hidden_dim)
        if arbitration_hidden_dim is not None
        else (
            int(integration_hidden_dim)
            if integration_hidden_dim is not None
            else ARBITRATION_HIDDEN_DIM
        )
    )
    arbitration_parameter_shapes = {
        "W1": [resolved_arbitration_hidden_dim, arbitration_input_dim],
        "b1": [resolved_arbitration_hidden_dim],
        "W2_valence": [ARBITRATION_VALENCE_DIM, resolved_arbitration_hidden_dim],
        "b2_valence": [ARBITRATION_VALENCE_DIM],
        "W2_gate": [ARBITRATION_GATE_DIM, resolved_arbitration_hidden_dim],
        "b2_gate": [ARBITRATION_GATE_DIM],
        "W2_value": [1, resolved_arbitration_hidden_dim],
        "b2_value": [1],
    }
    arbitration_parameter_fingerprint = _fingerprint_payload(
        {
            "input_dim": arbitration_input_dim,
            "hidden_dim": resolved_arbitration_hidden_dim,
            "valence_dim": ARBITRATION_VALENCE_DIM,
            "gate_dim": ARBITRATION_GATE_DIM,
            "parameter_shapes": arbitration_parameter_shapes,
        }
    )
    registry = interface_registry()
    proposal_interfaces = registry["interfaces"]
    payload = {
        "schema_version": INTERFACE_REGISTRY_SCHEMA_VERSION,
        "proposal_backend": proposal_backend,
        "registry_fingerprint": interface_registry_fingerprint(registry),
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
            for spec in (*ALL_INTERFACES, *_variant_interface_specs())
        },
        "module_variants": normalized_module_variants,
        "proposal_order": proposal_order,
        "capacity_profile_name": str(capacity_profile_name),
    }
    if (
        capacity_profile is not None
        or module_hidden_dims is not None
        or action_center_hidden_dim is not None
        or arbitration_hidden_dim is not None
        or motor_hidden_dim is not None
        or integration_hidden_dim is not None
        or monolithic_hidden_dim is not None
    ):
        capacity_payload: dict[str, object] = {}
        if capacity_profile is not None:
            capacity_payload["profile"] = {
                str(key): value
                for key, value in dict(capacity_profile).items()
            }
        if module_hidden_dims is not None:
            capacity_payload["module_hidden_dims"] = {
                str(name): int(hidden_dim)
                for name, hidden_dim in dict(module_hidden_dims).items()
            }
        if action_center_hidden_dim is not None:
            capacity_payload["action_center_hidden_dim"] = int(action_center_hidden_dim)
        capacity_payload["arbitration_hidden_dim"] = int(
            resolved_arbitration_hidden_dim
        )
        if motor_hidden_dim is not None:
            capacity_payload["motor_hidden_dim"] = int(motor_hidden_dim)
        if integration_hidden_dim is not None:
            capacity_payload["integration_hidden_dim"] = int(integration_hidden_dim)
        if monolithic_hidden_dim is not None:
            capacity_payload["monolithic_hidden_dim"] = int(monolithic_hidden_dim)
        payload["capacity"] = capacity_payload
    if proposal_backend == "true_monolithic":
        payload["direct_policy"] = {
            "observation_sources": [spec.name for spec in MODULE_INTERFACES],
            "observation_keys": [spec.observation_key for spec in MODULE_INTERFACES],
            "outputs": list(LOCOMOTION_ACTIONS),
            "value_head": True,
        }
        if monolithic_hidden_dim is not None:
            payload["direct_policy"]["hidden_dim"] = int(monolithic_hidden_dim)
    else:
        payload["arbitration_network"] = {
            "learned": bool(learned_arbitration),
            "input_dim": arbitration_input_dim,
            "hidden_dim": resolved_arbitration_hidden_dim,
            "regularization_weight": float(arbitration_regularization_weight),
            "parameter_shapes": arbitration_parameter_shapes,
            "parameter_fingerprint": arbitration_parameter_fingerprint,
        }
        payload["action_center"] = {
            "context_interface": ACTION_CONTEXT_INTERFACE.name,
            "observation_key": ACTION_CONTEXT_INTERFACE.observation_key,
            "version": ACTION_CONTEXT_INTERFACE.version,
            "inputs": [signal.name for signal in ACTION_CONTEXT_INTERFACE.inputs],
            "arbitration": {
                "strategy": "priority_gating",
                "valences": ["threat", "hunger", "sleep", "exploration"],
                "module_roles": filtered_module_roles,
            },
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
        }
        if action_center_hidden_dim is not None:
            payload["action_center"]["hidden_dim"] = int(action_center_hidden_dim)
        payload["motor_cortex"] = {
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
        }
        if motor_hidden_dim is not None:
            payload["motor_cortex"]["hidden_dim"] = int(motor_hidden_dim)
    payload["fingerprint"] = _fingerprint_payload(payload)
    return payload

__all__ = [
    "ARBITRATION_EVIDENCE_INPUT_DIM",
    "ARBITRATION_GATE_DIM",
    "ARBITRATION_HIDDEN_DIM",
    "ARBITRATION_VALENCE_DIM",
    "INTERFACE_REGISTRY_SCHEMA_VERSION",
    "OBSERVATION_DIMS",
    "OBSERVATION_VIEW_BY_KEY",
    "architecture_signature",
    "interface_registry",
    "interface_registry_fingerprint",
]
