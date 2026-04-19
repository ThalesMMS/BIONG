from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Sequence

import numpy as np

from ..ablations import BrainAblationConfig, default_brain_config
from ..arbitration import (
    ARBITRATION_EVIDENCE_FIELDS as DEFAULT_ARBITRATION_EVIDENCE_FIELDS,
    ARBITRATION_GATE_MODULE_ORDER as DEFAULT_ARBITRATION_GATE_MODULE_ORDER,
    ARBITRATION_NETWORK_NAME as DEFAULT_ARBITRATION_NETWORK_NAME,
    MONOLITHIC_POLICY_NAME as DEFAULT_MONOLITHIC_POLICY_NAME,
    PRIORITY_GATING_WEIGHTS as DEFAULT_PRIORITY_GATING_WEIGHTS,
    VALENCE_EVIDENCE_WEIGHTS as DEFAULT_VALENCE_EVIDENCE_WEIGHTS,
    VALENCE_ORDER as DEFAULT_VALENCE_ORDER,
    ArbitrationDecision,
    ValenceScore,
    apply_priority_gating,
    arbitration_evidence_input_dim,
    arbitration_gate_weight_for,
    clamp_unit,
    compute_arbitration,
    fixed_formula_valence_scores_from_evidence,
    warm_start_arbitration_network,
)
from ..bus import MessageBus
from ..interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    architecture_signature,
    interface_registry,
)
from ..modules import MODULE_HIDDEN_DIMS, CorticalModuleBank, ModuleResult, ReflexDecision
from ..nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork, one_hot, softmax
from ..noise import _compute_execution_difficulty_core
from ..operational_profiles import OperationalProfile, runtime_operational_profile
from ..reflexes import (
    _apply_reflex_path as apply_reflex_path,
    _direction_action as direction_action,
    _module_reflex_decision as module_reflex_decision,
)
from ..world import ACTIONS

from .types import BrainStep


class BrainPersistenceMixin:
    def save(self, directory: str | Path) -> Path:
        """
        Persist the brain's trainable weights and configuration metadata to the specified directory.
        
        Saves per-module proposal weights (one .npz per module or a single monolithic policy .npz), the arbitration network .npz, action_center and motor_cortex .npz files, and a JSON metadata file ("metadata.json") containing architecture/version/fingerprint, interface registry, ablation and operational profile summaries, learning rates and related hyperparameters, action dimensionality, and per-module shape/type metadata.
        
        Parameters:
            directory (str | Path): Destination directory for saved files; created if it does not exist.
        
        Returns:
            Path: Path to the directory where files were written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        architecture = self._architecture_signature()
        interface_registry = self._interface_registry()
        architecture_fingerprint = str(architecture["fingerprint"])

        metadata: Dict[str, object] = {
            "architecture_version": self.ARCHITECTURE_VERSION,
            "architecture": architecture,
            "interface_registry": interface_registry,
            "architecture_fingerprint": architecture_fingerprint,
            "ablation_config": self.config.to_summary(),
            "operational_profile": self.operational_profile.to_summary(),
            "gamma": self.gamma,
            "module_lr": self.module_lr,
            "arbitration_lr": self.arbitration_lr,
            "arbitration_regularization_weight": self.arbitration_regularization_weight,
            "arbitration_valence_regularization_weight": self.arbitration_valence_regularization_weight,
            "motor_lr": self.motor_lr,
            "module_dropout": self.module_dropout,
            "action_dim": self.action_dim,
            "modules": {},
        }

        if self.module_bank is not None:
            bank_state = self.module_bank.state_dict()
            for name, sd in bank_state.items():
                arrays = {k: v for k, v in sd.items() if isinstance(v, np.ndarray)}
                np.savez(directory / f"{name}.npz", **arrays)
                metadata["modules"][name] = {
                    "type": "proposal",
                    "input_dim": sd["input_dim"],
                    "hidden_dim": sd["hidden_dim"],
                    "output_dim": sd["output_dim"],
                }
        elif self.monolithic_policy is not None:
            mono_sd = self.monolithic_policy.state_dict()
            mono_arrays = {k: v for k, v in mono_sd.items() if isinstance(v, np.ndarray)}
            np.savez(directory / f"{self.MONOLITHIC_POLICY_NAME}.npz", **mono_arrays)
            metadata["modules"][self.MONOLITHIC_POLICY_NAME] = {
                "type": "proposal",
                "input_dim": mono_sd["input_dim"],
                "hidden_dim": mono_sd["hidden_dim"],
                "output_dim": mono_sd["output_dim"],
            }

        arbitration_sd = self.arbitration_network.state_dict()
        arbitration_arrays = {
            k: (v if isinstance(v, np.ndarray) else np.asarray(v))
            for k, v in arbitration_sd.items()
        }
        np.savez(directory / f"{self.ARBITRATION_NETWORK_NAME}.npz", **arbitration_arrays)
        metadata["modules"][self.ARBITRATION_NETWORK_NAME] = {
            "type": "arbitration",
            "input_dim": arbitration_sd["input_dim"],
            "hidden_dim": arbitration_sd["hidden_dim"],
            "valence_dim": arbitration_sd["valence_dim"],
            "gate_dim": arbitration_sd["gate_dim"],
        }

        action_sd = self.action_center.state_dict()
        action_arrays = {k: v for k, v in action_sd.items() if isinstance(v, np.ndarray)}
        np.savez(directory / "action_center.npz", **action_arrays)
        metadata["modules"]["action_center"] = {
            "type": "action",
            "input_dim": action_sd["input_dim"],
            "hidden_dim": action_sd["hidden_dim"],
            "output_dim": action_sd["output_dim"],
        }

        motor_sd = self.motor_cortex.state_dict()
        motor_arrays = {k: v for k, v in motor_sd.items() if isinstance(v, np.ndarray)}
        np.savez(directory / "motor_cortex.npz", **motor_arrays)
        metadata["modules"]["motor_cortex"] = {
            "type": "motor",
            "input_dim": motor_sd["input_dim"],
            "hidden_dim": motor_sd["hidden_dim"],
            "output_dim": motor_sd["output_dim"],
        }

        (directory / self._METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    def load(
        self,
        directory: str | Path,
        *,
        modules: Sequence[str] | None = None,
    ) -> List[str]:
        """
        Load saved network weights into this brain and validate the save's metadata for compatibility.
        
        Parameters:
            directory (str | Path): Path to a directory containing metadata.json and per-module `.npz` weight files.
            modules (Sequence[str] | None): Optional subset of module names to load; when `None`, load all modules listed in the metadata.
        
        Returns:
            List[str]: Names of modules whose weights were loaded.
        
        Raises:
            FileNotFoundError: If metadata.json or any required `.npz` weight file is missing.
            ValueError: If the saved architecture signature, ablation configuration, or operational profile does not match this brain's configuration.
            KeyError: If a requested module is not present in the save, or if monolithic weights are required but missing.
        """
        directory = Path(directory)
        meta_path = directory / self._METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_architecture_version = metadata.get("architecture_version")
        if saved_architecture_version != self.ARCHITECTURE_VERSION:
            raise ValueError(
                "The saved brain uses an architecture version incompatible with the current "
                f"simulator version (save={saved_architecture_version}, current={self.ARCHITECTURE_VERSION})."
            )
        saved_architecture = metadata.get("architecture")
        expected_architecture = self._architecture_signature()
        saved_registry = metadata.get("interface_registry")
        expected_registry = self._interface_registry()
        if saved_registry != expected_registry:
            saved_registry_fingerprint = None
            if isinstance(saved_architecture, dict):
                saved_registry_fingerprint = saved_architecture.get("registry_fingerprint")
            raise ValueError(
                "The saved brain uses an interface registry incompatible with the current registry. "
                f"(save_registry_fingerprint={saved_registry_fingerprint}, "
                f"current={expected_architecture.get('registry_fingerprint')}). "
                "There is no automatic checkpoint migration for contract changes."
            )
        if saved_architecture != expected_architecture:
            raise ValueError(
                "The saved brain uses an architecture signature incompatible with the current topology "
                f"(save_fingerprint={metadata.get('architecture_fingerprint')}, "
                f"current={self._architecture_fingerprint()}). "
                "There is no automatic checkpoint migration for this change; train or load "
                "a brain compatible with the current simulator version."
            )
        saved_config = metadata.get("ablation_config")
        expected_config = self.config.to_summary()
        if saved_config != expected_config:
            raise ValueError(
                "The saved brain uses an ablation configuration different from the current configuration."
            )
        saved_operational_profile = metadata.get("operational_profile")
        expected_operational_profile = self.operational_profile.to_summary()
        if saved_operational_profile != expected_operational_profile:
            raise ValueError(
                "The saved brain uses an operational profile different from the current operational profile."
            )

        saved_modules = metadata.get("modules", {})
        if not isinstance(saved_modules, dict):
            raise ValueError("The saved brain metadata has an invalid modules section.")
        requested_module_set = (
            None
            if modules is None
            else {str(name) for name in modules}
        )
        needs_arbitration_weights = (
            requested_module_set is None
            or self.ARBITRATION_NETWORK_NAME in requested_module_set
        )
        arbitration_npz_path = directory / f"{self.ARBITRATION_NETWORK_NAME}.npz"
        if needs_arbitration_weights:
            if self.ARBITRATION_NETWORK_NAME not in saved_modules:
                raise FileNotFoundError(
                    "Checkpoint is missing arbitration_network metadata. "
                    "Older checkpoints must be migrated or retrained for this architecture."
                )
            if not arbitration_npz_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint is missing arbitration weights: {arbitration_npz_path}"
                )

        available = list(saved_modules.keys())
        targets = list(modules) if modules is not None else available

        loaded: List[str] = []
        bank_state: Dict[str, dict] = {}

        for name in targets:
            if name not in available:
                raise KeyError(
                    f"Module '{name}' not found in the save. "
                    f"Available: {available}"
                )
            npz_path = directory / f"{name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Weight file not found: {npz_path}")

            mod_meta = saved_modules[name]
            with np.load(npz_path) as npz:
                arrays = dict(npz)

            if mod_meta["type"] == "action":
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                self.action_center.load_state_dict(arrays)
            elif mod_meta["type"] == "arbitration":
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["valence_dim"] = mod_meta["valence_dim"]
                arrays["gate_dim"] = mod_meta["gate_dim"]
                self.arbitration_network.load_state_dict(arrays)
            elif mod_meta["type"] == "motor":
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                self.motor_cortex.load_state_dict(arrays)
            else:
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                bank_state[name] = arrays

            loaded.append(name)

        if bank_state and self.module_bank is not None:
            self.module_bank.load_state_dict(bank_state, modules=list(bank_state.keys()))
        elif bank_state and self.monolithic_policy is not None:
            monolithic_name = self.MONOLITHIC_POLICY_NAME
            if monolithic_name not in bank_state:
                raise KeyError("Incompatible save: monolithic weights are missing.")
            self.monolithic_policy.load_state_dict(bank_state[monolithic_name])

        return loaded
