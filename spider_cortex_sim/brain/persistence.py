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
from ..direct_policy_affordances import (
    DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
    DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
)
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
        
        Saves the active trainable weights for the configured architecture and a
        JSON metadata file ("metadata.json") containing architecture/version/
        fingerprint, interface registry, ablation and operational profile
        summaries, learning rates and related hyperparameters, action
        dimensionality, and per-module shape/type metadata.
        
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
        parameter_counts_by_network = self.count_parameters()
        total_trainable_parameters = int(sum(parameter_counts_by_network.values()))

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
            "parameter_counts": dict(parameter_counts_by_network),
            "total_parameters": total_trainable_parameters,
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
                    "parameter_count": int(parameter_counts_by_network.get(name, 0)),
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
                "parameter_count": int(
                    parameter_counts_by_network.get(self.MONOLITHIC_POLICY_NAME, 0)
                ),
            }
        elif self.true_monolithic_policy is not None:
            mono_sd = self.true_monolithic_policy.state_dict()
            mono_arrays = {k: v for k, v in mono_sd.items() if isinstance(v, np.ndarray)}
            np.savez(directory / f"{self.TRUE_MONOLITHIC_POLICY_NAME}.npz", **mono_arrays)
            module_metadata = {
                "type": "direct_policy",
                "input_dim": mono_sd["input_dim"],
                "hidden_dim": mono_sd["hidden_dim"],
                "hidden_sizes": list(mono_sd.get("hidden_sizes", [mono_sd["hidden_dim"]])),
                "recurrent": bool(mono_sd.get("recurrent", False)),
                "phase_output_dim": int(mono_sd.get("phase_output_dim", 0)),
                "event_attention": bool(mono_sd.get("event_attention", False)),
                "event_buffer_size": int(mono_sd.get("event_buffer_size", 0)),
                "option_head": bool(mono_sd.get("option_head", False)),
                "option_ttl": int(mono_sd.get("option_ttl", 0)),
                "affordance_head": bool(mono_sd.get("affordance_head", False)),
                "affordance_feedback": bool(mono_sd.get("affordance_feedback", False)),
                "geometry_head": bool(mono_sd.get("geometry_head", False)),
                "shelter_column_head": bool(
                    mono_sd.get("shelter_column_head", False)
                ),
                "shelter_position_head": bool(
                    mono_sd.get("shelter_position_head", False)
                ),
                "transition_prediction_head": bool(
                    mono_sd.get("transition_prediction_head", False)
                ),
                "transition_prediction_feedback": bool(
                    mono_sd.get("transition_prediction_feedback", False)
                ),
                "transition_prediction_feature_dim": int(
                    mono_sd.get(
                        "transition_prediction_feature_dim",
                        DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
                    )
                ),
                "transition_rollout_prediction_head": bool(
                    mono_sd.get("transition_rollout_prediction_head", False)
                ),
                "transition_rollout_prediction_feedback": bool(
                    mono_sd.get("transition_rollout_prediction_feedback", False)
                ),
                "transition_rollout_prediction_feature_dim": int(
                    mono_sd.get(
                        "transition_rollout_prediction_feature_dim",
                        DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
                    )
                ),
                "handoff_teacher": bool(
                    getattr(self.config, "direct_policy_handoff_teacher", False)
                ),
                "handoff_option_teacher": bool(
                    getattr(
                        self.config,
                        "direct_policy_handoff_option_teacher",
                        False,
                    )
                ),
                "post_rest_action_teacher": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_action_teacher",
                        False,
                    )
                ),
                "post_rest_release_sequence_teacher": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_release_sequence_teacher",
                        False,
                    )
                ),
                "post_rest_release_sequence_replay_boost": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_release_sequence_replay_boost",
                        False,
                    )
                ),
                "post_rest_release_sequence_distill": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_release_sequence_distill",
                        False,
                    )
                ),
                "post_rest_probe_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_distillation",
                        False,
                    )
                ),
                "post_rest_probe_sequence_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_sequence_distillation",
                        False,
                    )
                ),
                "post_rest_probe_family_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_family_distillation",
                        False,
                    )
                ),
                "post_rest_probe_handoff_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_handoff_distillation",
                        False,
                    )
                ),
                "post_rest_probe_trajectory_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_trajectory_distillation",
                        False,
                    )
                ),
                "post_rest_probe_cycle_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_cycle_distillation",
                        False,
                    )
                ),
                "post_rest_probe_trace_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_trace_distillation",
                        False,
                    )
                ),
                "post_rest_probe_rollout_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_rollout_distillation",
                        False,
                    )
                ),
                "post_rest_probe_frontier_teacher_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_frontier_teacher_distillation",
                        False,
                    )
                ),
                "post_rest_probe_replayable_teacher_distillation": bool(
                    getattr(
                        self.config,
                        "direct_policy_post_rest_probe_replayable_teacher_distillation",
                        False,
                    )
                ),
                "continuation_replay_passes": int(
                    getattr(
                        self.config,
                        "direct_policy_continuation_replay_passes",
                        0,
                    )
                ),
                "continuation_replay_lr_scale": float(
                    getattr(
                        self.config,
                        "direct_policy_continuation_replay_lr_scale",
                        0.0,
                    )
                ),
                "continuation_margin_weight": float(
                    getattr(
                        self.config,
                        "direct_policy_continuation_margin_weight",
                        0.0,
                    )
                ),
                "phase_option_feedback": bool(
                    getattr(
                        self.config,
                        "direct_policy_phase_option_feedback",
                        False,
                    )
                ),
                "option_transition_feedback": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_transition_feedback",
                        False,
                    )
                ),
                "option_termination_cooldown": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_termination_cooldown",
                        False,
                    )
                ),
                "option_action_head": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_head",
                        False,
                    )
                ),
                "option_decoder_state": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_decoder_state",
                        False,
                    )
                ),
                "option_recurrent_dynamics": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_recurrent_dynamics",
                        False,
                    )
                ),
                "option_sequence_head": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_sequence_head",
                        False,
                    )
                ),
                "option_decoder_recurrent_state": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_decoder_recurrent_state",
                        False,
                    )
                ),
                "option_action_transition_state": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_transition_state",
                        False,
                    )
                ),
                "option_action_controller_state": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_controller_state",
                        False,
                    )
                ),
                "option_action_token_decoder": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_token_decoder",
                        False,
                    )
                ),
                "option_action_recurrent_core": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_recurrent_core",
                        False,
                    )
                ),
                "option_action_separate_recurrent_head": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_separate_recurrent_head",
                        False,
                    )
                ),
                "option_action_separate_policy_path": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_separate_policy_path",
                        False,
                    )
                ),
                "option_action_separate_backbone": bool(
                    getattr(
                        self.config,
                        "direct_policy_option_action_separate_backbone",
                        False,
                    )
                ),
                "executive_physiology_option_gating": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_physiology_option_gating",
                        False,
                    )
                ),
                "executive_affordance_action_gating": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_affordance_action_gating",
                        False,
                    )
                ),
                "executive_option_action_masking": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_option_action_masking",
                        False,
                    )
                ),
                "executive_event_release_latching": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_event_release_latching",
                        False,
                    )
                ),
                "executive_event_release_action_commitment": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_event_release_action_commitment",
                        False,
                    )
                ),
                "executive_release_phase_state": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_release_phase_state",
                        False,
                    )
                ),
                "executive_release_progression": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_release_progression",
                        False,
                    )
                ),
                "executive_release_exit_contract": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_release_exit_contract",
                        False,
                    )
                ),
                "executive_release_substate_progression": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_release_substate_progression",
                        False,
                    )
                ),
                "executive_post_exit_continuation": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_continuation",
                        False,
                    )
                ),
                "executive_post_exit_food_guidance": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_food_guidance",
                        False,
                    )
                ),
                "executive_post_exit_food_commitment": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_food_commitment",
                        False,
                    )
                ),
                "executive_post_exit_food_progression": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_food_progression",
                        False,
                    )
                ),
                "executive_post_exit_food_heading_progression": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_food_heading_progression",
                        False,
                    )
                ),
                "executive_post_exit_smell_progression": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_smell_progression",
                        False,
                    )
                ),
                "executive_post_exit_corridor_progression": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_corridor_progression",
                        False,
                    )
                ),
                "executive_post_exit_corridor_affordance_progression": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_exit_corridor_affordance_progression",
                        False,
                    )
                ),
                "executive_post_food_return": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_food_return",
                        False,
                    )
                ),
                "executive_post_food_vector_return": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_food_vector_return",
                        False,
                    )
                ),
                "executive_post_food_path_return": bool(
                    getattr(
                        self.config,
                        "direct_policy_executive_post_food_path_return",
                        False,
                    )
                ),
                "output_dim": mono_sd["output_dim"],
                "parameter_count": int(
                    parameter_counts_by_network.get(self.TRUE_MONOLITHIC_POLICY_NAME, 0)
                ),
            }
            if bool(mono_sd.get("event_attention", False)):
                module_metadata["event_embedding_dim"] = int(
                    mono_sd["event_embedding_dim"]
                )
                module_metadata["event_context_dim"] = int(
                    mono_sd["event_context_dim"]
                )
                module_metadata["event_feature_dim"] = int(
                    mono_sd["event_feature_dim"]
                )
            if bool(mono_sd.get("option_head", False)):
                module_metadata["option_dim"] = int(mono_sd["option_dim"])
            if bool(mono_sd.get("affordance_head", False)):
                module_metadata["affordance_role_dim"] = int(
                    mono_sd["affordance_role_dim"]
                )
            if bool(mono_sd.get("affordance_feedback", False)):
                module_metadata["affordance_feature_dim"] = int(
                    mono_sd["affordance_feature_dim"]
                )
            if bool(mono_sd.get("geometry_head", False)):
                module_metadata["geometry_dim"] = int(mono_sd["geometry_dim"])
                module_metadata["geometry_feature_dim"] = int(
                    mono_sd["geometry_feature_dim"]
                )
            if bool(mono_sd.get("shelter_column_head", False)):
                module_metadata["shelter_column_dim"] = int(
                    mono_sd["shelter_column_dim"]
                )
                module_metadata["shelter_column_feature_dim"] = int(
                    mono_sd["shelter_column_feature_dim"]
                )
            if bool(mono_sd.get("shelter_position_head", False)):
                module_metadata["shelter_position_dim"] = int(
                    mono_sd["shelter_position_dim"]
                )
                module_metadata["shelter_position_feature_dim"] = int(
                    mono_sd["shelter_position_feature_dim"]
                )
            if bool(mono_sd.get("transition_prediction_head", False)):
                module_metadata["transition_prediction_feature_dim"] = int(
                    mono_sd["transition_prediction_feature_dim"]
                )
            if bool(mono_sd.get("transition_rollout_prediction_head", False)):
                module_metadata["transition_rollout_prediction_feature_dim"] = int(
                    mono_sd["transition_rollout_prediction_feature_dim"]
                )
            metadata["modules"][self.TRUE_MONOLITHIC_POLICY_NAME] = module_metadata

        if self.arbitration_network is not None:
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
                "parameter_count": int(
                    parameter_counts_by_network.get(self.ARBITRATION_NETWORK_NAME, 0)
                ),
            }

        if self.action_center is not None:
            action_sd = self.action_center.state_dict()
            action_arrays = {k: v for k, v in action_sd.items() if isinstance(v, np.ndarray)}
            np.savez(directory / "action_center.npz", **action_arrays)
            metadata["modules"]["action_center"] = {
                "type": "action",
                "input_dim": action_sd["input_dim"],
                "hidden_dim": action_sd["hidden_dim"],
                "output_dim": action_sd["output_dim"],
                "parameter_count": int(parameter_counts_by_network.get("action_center", 0)),
            }

        if self.motor_cortex is not None:
            motor_sd = self.motor_cortex.state_dict()
            motor_arrays = {k: v for k, v in motor_sd.items() if isinstance(v, np.ndarray)}
            np.savez(directory / "motor_cortex.npz", **motor_arrays)
            metadata["modules"]["motor_cortex"] = {
                "type": "motor",
                "input_dim": motor_sd["input_dim"],
                "hidden_dim": motor_sd["hidden_dim"],
                "output_dim": motor_sd["output_dim"],
                "parameter_count": int(parameter_counts_by_network.get("motor_cortex", 0)),
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
            self.arbitration_network is not None
            and (
            requested_module_set is None
            or self.ARBITRATION_NETWORK_NAME in requested_module_set
            )
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
                if self.action_center is None:
                    raise ValueError("Checkpoint contains action_center weights for a topology without action_center.")
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                self.action_center.load_state_dict(arrays)
            elif mod_meta["type"] == "arbitration":
                if self.arbitration_network is None:
                    raise ValueError("Checkpoint contains arbitration weights for a topology without arbitration.")
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["valence_dim"] = mod_meta["valence_dim"]
                arrays["gate_dim"] = mod_meta["gate_dim"]
                self.arbitration_network.load_state_dict(arrays)
            elif mod_meta["type"] == "motor":
                if self.motor_cortex is None:
                    raise ValueError("Checkpoint contains motor_cortex weights for a topology without motor_cortex.")
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                self.motor_cortex.load_state_dict(arrays)
            else:
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                hidden_sizes = mod_meta.get("hidden_sizes")
                if hidden_sizes is not None:
                    arrays["hidden_sizes"] = hidden_sizes
                if mod_meta["type"] == "direct_policy":
                    arrays["recurrent"] = bool(mod_meta.get("recurrent", False))
                    arrays["phase_output_dim"] = int(mod_meta.get("phase_output_dim", 0))
                    arrays["event_attention"] = bool(mod_meta.get("event_attention", False))
                    arrays["event_buffer_size"] = int(mod_meta.get("event_buffer_size", 0))
                    arrays["option_head"] = bool(mod_meta.get("option_head", False))
                    arrays["option_ttl"] = int(mod_meta.get("option_ttl", 0))
                    arrays["phase_option_feedback"] = bool(
                        mod_meta.get("phase_option_feedback", False)
                    )
                    arrays["option_transition_feedback"] = bool(
                        mod_meta.get("option_transition_feedback", False)
                    )
                    arrays["option_termination_cooldown"] = bool(
                        mod_meta.get("option_termination_cooldown", False)
                    )
                    arrays["option_action_head"] = bool(
                        mod_meta.get("option_action_head", False)
                    )
                    arrays["option_decoder_state"] = bool(
                        mod_meta.get("option_decoder_state", False)
                    )
                    arrays["option_recurrent_dynamics"] = bool(
                        mod_meta.get("option_recurrent_dynamics", False)
                    )
                    arrays["option_sequence_head"] = bool(
                        mod_meta.get("option_sequence_head", False)
                    )
                    arrays["option_decoder_recurrent_state"] = bool(
                        mod_meta.get("option_decoder_recurrent_state", False)
                    )
                    arrays["option_action_transition_state"] = bool(
                        mod_meta.get("option_action_transition_state", False)
                    )
                    arrays["option_action_controller_state"] = bool(
                        mod_meta.get("option_action_controller_state", False)
                    )
                    arrays["option_action_token_decoder"] = bool(
                        mod_meta.get("option_action_token_decoder", False)
                    )
                    arrays["option_action_recurrent_core"] = bool(
                        mod_meta.get("option_action_recurrent_core", False)
                    )
                    arrays["option_action_separate_recurrent_head"] = bool(
                        mod_meta.get("option_action_separate_recurrent_head", False)
                    )
                    arrays["option_action_separate_policy_path"] = bool(
                        mod_meta.get("option_action_separate_policy_path", False)
                    )
                    arrays["option_action_separate_backbone"] = bool(
                        mod_meta.get("option_action_separate_backbone", False)
                    )
                    arrays["affordance_head"] = bool(
                        mod_meta.get("affordance_head", False)
                    )
                    arrays["affordance_feedback"] = bool(
                        mod_meta.get("affordance_feedback", False)
                    )
                    arrays["geometry_head"] = bool(
                        mod_meta.get("geometry_head", False)
                    )
                    arrays["shelter_column_head"] = bool(
                        mod_meta.get("shelter_column_head", False)
                    )
                    arrays["shelter_position_head"] = bool(
                        mod_meta.get("shelter_position_head", False)
                    )
                    arrays["transition_prediction_head"] = bool(
                        mod_meta.get("transition_prediction_head", False)
                    )
                    arrays["transition_prediction_feedback"] = bool(
                        mod_meta.get("transition_prediction_feedback", False)
                    )
                    arrays["transition_prediction_feature_dim"] = int(
                        mod_meta.get(
                            "transition_prediction_feature_dim",
                            DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
                        )
                    )
                    arrays["transition_rollout_prediction_head"] = bool(
                        mod_meta.get(
                            "transition_rollout_prediction_head",
                            False,
                        )
                    )
                    arrays["transition_rollout_prediction_feedback"] = bool(
                        mod_meta.get(
                            "transition_rollout_prediction_feedback",
                            False,
                        )
                    )
                    arrays["transition_rollout_prediction_feature_dim"] = int(
                        mod_meta.get(
                            "transition_rollout_prediction_feature_dim",
                            DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
                        )
                    )
                    if arrays["event_attention"]:
                        arrays["event_embedding_dim"] = int(
                            mod_meta.get("event_embedding_dim", 0)
                        )
                        arrays["event_context_dim"] = int(
                            mod_meta.get("event_context_dim", 0)
                        )
                        arrays["event_feature_dim"] = int(
                            mod_meta.get("event_feature_dim", 0)
                        )
                    if arrays["option_head"]:
                        arrays["option_dim"] = int(mod_meta.get("option_dim", 0))
                    if arrays["affordance_head"]:
                        arrays["affordance_role_dim"] = int(
                            mod_meta.get("affordance_role_dim", 0)
                        )
                    if arrays["affordance_feedback"]:
                        arrays["affordance_feature_dim"] = int(
                            mod_meta.get("affordance_feature_dim", 0)
                        )
                    if arrays["geometry_head"]:
                        arrays["geometry_dim"] = int(mod_meta.get("geometry_dim", 0))
                        arrays["geometry_feature_dim"] = int(
                            mod_meta.get("geometry_feature_dim", 0)
                        )
                    if arrays["shelter_column_head"]:
                        arrays["shelter_column_dim"] = int(
                            mod_meta.get("shelter_column_dim", 0)
                        )
                        arrays["shelter_column_feature_dim"] = int(
                            mod_meta.get("shelter_column_feature_dim", 0)
                        )
                    if arrays["shelter_position_head"]:
                        arrays["shelter_position_dim"] = int(
                            mod_meta.get("shelter_position_dim", 0)
                        )
                        arrays["shelter_position_feature_dim"] = int(
                            mod_meta.get("shelter_position_feature_dim", 0)
                        )
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
        elif bank_state and self.true_monolithic_policy is not None:
            monolithic_name = self.TRUE_MONOLITHIC_POLICY_NAME
            if monolithic_name not in bank_state:
                raise KeyError("Incompatible save: true monolithic weights are missing.")
            direct_state = dict(bank_state[monolithic_name])
            if not hasattr(self.true_monolithic_policy, "hidden_sizes"):
                direct_state.pop("hidden_sizes", None)
            if not hasattr(self.true_monolithic_policy, "hidden_state"):
                direct_state.pop("recurrent", None)
            if not hasattr(self.true_monolithic_policy, "phase_output_dim"):
                direct_state.pop("phase_output_dim", None)
                direct_state.pop("phase_option_feedback", None)
                direct_state.pop("option_transition_feedback", None)
                direct_state.pop("option_termination_cooldown", None)
                direct_state.pop("option_action_head", None)
                direct_state.pop("option_decoder_state", None)
                direct_state.pop("option_recurrent_dynamics", None)
                direct_state.pop("option_sequence_head", None)
                direct_state.pop("option_decoder_recurrent_state", None)
                direct_state.pop("option_action_transition_state", None)
                direct_state.pop("option_action_controller_state", None)
                direct_state.pop("option_action_token_decoder", None)
                direct_state.pop("option_action_recurrent_core", None)
                direct_state.pop("option_action_separate_recurrent_head", None)
                direct_state.pop("option_action_separate_policy_path", None)
                direct_state.pop("option_action_separate_backbone", None)
            if not hasattr(self.true_monolithic_policy, "event_buffer_size"):
                direct_state.pop("event_attention", None)
                direct_state.pop("event_buffer_size", None)
            if not hasattr(self.true_monolithic_policy, "event_embedding_dim"):
                direct_state.pop("event_embedding_dim", None)
                direct_state.pop("event_context_dim", None)
                direct_state.pop("event_feature_dim", None)
            if not hasattr(self.true_monolithic_policy, "option_ttl"):
                direct_state.pop("option_head", None)
                direct_state.pop("option_ttl", None)
                direct_state.pop("option_dim", None)
                direct_state.pop("phase_option_feedback", None)
                direct_state.pop("option_transition_feedback", None)
                direct_state.pop("option_termination_cooldown", None)
                direct_state.pop("option_action_head", None)
                direct_state.pop("option_decoder_state", None)
                direct_state.pop("option_recurrent_dynamics", None)
                direct_state.pop("option_sequence_head", None)
                direct_state.pop("option_decoder_recurrent_state", None)
                direct_state.pop("option_action_transition_state", None)
                direct_state.pop("option_action_controller_state", None)
                direct_state.pop("option_action_token_decoder", None)
                direct_state.pop("option_action_recurrent_core", None)
                direct_state.pop("option_action_separate_recurrent_head", None)
                direct_state.pop("option_action_separate_policy_path", None)
                direct_state.pop("option_action_separate_backbone", None)
            if not hasattr(self.true_monolithic_policy, "affordance_role_dim"):
                direct_state.pop("affordance_head", None)
                direct_state.pop("affordance_role_dim", None)
                direct_state.pop("W2_affordance_blocked", None)
                direct_state.pop("b2_affordance_blocked", None)
                direct_state.pop("W2_affordance_role", None)
                direct_state.pop("b2_affordance_role", None)
            if not hasattr(self.true_monolithic_policy, "affordance_feature_dim"):
                direct_state.pop("affordance_feedback", None)
                direct_state.pop("affordance_feature_dim", None)
                direct_state.pop("W_affordance_feedback", None)
                direct_state.pop("b_affordance_feedback", None)
                direct_state.pop("W2_policy_feedback", None)
                direct_state.pop("b2_policy_feedback", None)
                direct_state.pop("W2_option_feedback", None)
                direct_state.pop("b2_option_feedback", None)
            if not hasattr(self.true_monolithic_policy, "geometry_dim"):
                direct_state.pop("geometry_head", None)
                direct_state.pop("geometry_dim", None)
                direct_state.pop("geometry_feature_dim", None)
                direct_state.pop("W2_geometry", None)
                direct_state.pop("b2_geometry", None)
            if not hasattr(self.true_monolithic_policy, "shelter_column_dim"):
                direct_state.pop("shelter_column_head", None)
                direct_state.pop("shelter_column_dim", None)
                direct_state.pop("shelter_column_feature_dim", None)
                direct_state.pop("W2_shelter_column", None)
                direct_state.pop("b2_shelter_column", None)
            if not hasattr(self.true_monolithic_policy, "shelter_position_dim"):
                direct_state.pop("shelter_position_head", None)
                direct_state.pop("shelter_position_dim", None)
                direct_state.pop("shelter_position_feature_dim", None)
                direct_state.pop("W2_shelter_position", None)
                direct_state.pop("b2_shelter_position", None)
            if not hasattr(self.true_monolithic_policy, "transition_prediction_feature_dim"):
                direct_state.pop("transition_prediction_head", None)
                direct_state.pop("transition_prediction_feedback", None)
                direct_state.pop("transition_prediction_feature_dim", None)
                direct_state.pop("W2_transition_prediction", None)
                direct_state.pop("b2_transition_prediction", None)
                direct_state.pop("W_transition_prediction_feedback", None)
                direct_state.pop("b_transition_prediction_feedback", None)
            if not hasattr(
                self.true_monolithic_policy,
                "transition_rollout_prediction_feature_dim",
            ):
                direct_state.pop("transition_rollout_prediction_head", None)
                direct_state.pop("transition_rollout_prediction_feedback", None)
                direct_state.pop("transition_rollout_prediction_feature_dim", None)
                direct_state.pop("W2_transition_rollout_prediction", None)
                direct_state.pop("b2_transition_rollout_prediction", None)
                direct_state.pop(
                    "W_transition_rollout_prediction_feedback",
                    None,
                )
                direct_state.pop(
                    "b_transition_rollout_prediction_feedback",
                    None,
                )
            if not getattr(self.true_monolithic_policy, "phase_option_feedback", False):
                direct_state.pop("W2_phase_option_feedback", None)
                direct_state.pop("b2_phase_option_feedback", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_transition_feedback",
                False,
            ):
                direct_state.pop("W2_option_transition_feedback", None)
                direct_state.pop("b2_option_transition_feedback", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_action_head",
                False,
            ):
                direct_state.pop("W2_option_action_head", None)
                direct_state.pop("b2_option_action_head", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_decoder_state",
                False,
            ):
                direct_state.pop("W_option_decoder_state", None)
                direct_state.pop("b_option_decoder_state", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_recurrent_dynamics",
                False,
            ):
                direct_state.pop("W_option_recurrent_dynamics", None)
                direct_state.pop("b_option_recurrent_dynamics", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_sequence_head",
                False,
            ):
                direct_state.pop("W2_option_sequence_head", None)
                direct_state.pop("b2_option_sequence_head", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_decoder_recurrent_state",
                False,
            ):
                direct_state.pop("W_option_decoder_recurrent_state", None)
                direct_state.pop("b_option_decoder_recurrent_state", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_action_transition_state",
                False,
            ):
                direct_state.pop("W_option_action_transition_state", None)
                direct_state.pop("b_option_action_transition_state", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_action_controller_state",
                False,
            ):
                direct_state.pop("W_option_action_controller_decoder", None)
                direct_state.pop("W_option_action_controller_prev", None)
                direct_state.pop("W_option_action_controller_action", None)
                direct_state.pop("b_option_action_controller", None)
                direct_state.pop("W2_option_action_controller_head", None)
                direct_state.pop("b2_option_action_controller_head", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_action_token_decoder",
                False,
            ):
                direct_state.pop("W_option_action_token_decoder", None)
                direct_state.pop("W_option_action_token_prev", None)
                direct_state.pop("W_option_action_token_action", None)
                direct_state.pop("b_option_action_token", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_action_recurrent_core",
                False,
            ):
                direct_state.pop("W_option_action_policy_decoder", None)
                direct_state.pop("W_option_action_policy_prev", None)
                direct_state.pop("W_option_action_policy_action", None)
                direct_state.pop("b_option_action_policy", None)
                direct_state.pop("W2_action_policy_core", None)
                direct_state.pop("b2_action_policy_core", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_action_separate_policy_path",
                False,
            ):
                direct_state.pop("W_action_policy_path_input", None)
                direct_state.pop("W_action_policy_path_prev", None)
                direct_state.pop("W_action_policy_path_action", None)
                direct_state.pop("b_action_policy_path", None)
                direct_state.pop("W2_action_policy_path", None)
                direct_state.pop("b2_action_policy_path", None)
            if not getattr(
                self.true_monolithic_policy,
                "option_action_separate_backbone",
                False,
            ):
                direct_state.pop("W_action_backbone_input", None)
                direct_state.pop("W_action_backbone_prev", None)
                direct_state.pop("W_action_backbone_action", None)
                direct_state.pop("b_action_backbone", None)
                direct_state.pop("W2_action_backbone", None)
                direct_state.pop("b2_action_backbone", None)
            self.true_monolithic_policy.load_state_dict(direct_state)

        return loaded
