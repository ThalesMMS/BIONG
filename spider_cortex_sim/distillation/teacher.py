from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from ..ablations import BrainAblationConfig
from ..interfaces import interface_registry
from ..operational_profiles import OperationalProfile

if TYPE_CHECKING:
    from ..agent import SpiderBrain


def load_teacher_checkpoint(
    checkpoint_path: str | Path,
) -> Tuple["SpiderBrain", Dict[str, object]]:
    from ..agent import SpiderBrain

    checkpoint_dir = Path(checkpoint_path)
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint metadata not found: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("architecture_version") != SpiderBrain.ARCHITECTURE_VERSION:
        raise ValueError(
            "Teacher checkpoint architecture version is incompatible with the "
            f"current simulator version (save={metadata.get('architecture_version')}, "
            f"current={SpiderBrain.ARCHITECTURE_VERSION})."
        )
    saved_registry = metadata.get("interface_registry")
    current_registry = interface_registry()
    if saved_registry != current_registry:
        raise ValueError(
            "Teacher checkpoint interface registry is incompatible with the current runtime."
        )

    config_summary = metadata.get("ablation_config", {})
    operational_summary = metadata.get("operational_profile", {})
    teacher_config = BrainAblationConfig.from_summary(config_summary)
    operational_profile = OperationalProfile.from_summary(operational_summary)
    teacher = SpiderBrain(
        seed=0,
        gamma=float(metadata.get("gamma", 0.96)),
        module_lr=float(metadata.get("module_lr", 0.010)),
        motor_lr=float(metadata.get("motor_lr", 0.012)),
        module_dropout=float(metadata.get("module_dropout", 0.0)),
        arbitration_lr=float(
            metadata.get("arbitration_lr", metadata.get("module_lr", 0.010))
        ),
        arbitration_regularization_weight=float(
            metadata.get("arbitration_regularization_weight", 0.0)
        ),
        arbitration_valence_regularization_weight=float(
            metadata.get("arbitration_valence_regularization_weight", 0.0)
        ),
        config=teacher_config,
        capacity_profile=teacher_config.capacity_profile_name,
        operational_profile=operational_profile,
    )
    teacher.load(checkpoint_dir)
    teacher.reset_hidden_states()

    teacher_metadata: Dict[str, object] = {
        "checkpoint_path": str(checkpoint_dir),
        "architecture_version": int(metadata.get("architecture_version", 0)),
        "architecture_fingerprint": str(
            metadata.get("architecture_fingerprint", "")
        ),
        "architecture": deepcopy(metadata.get("architecture", {})),
        "ablation_config": teacher_config.to_summary(),
        "operational_profile": operational_profile.to_summary(),
        "learning": {
            "gamma": float(metadata.get("gamma", 0.96)),
            "module_lr": float(metadata.get("module_lr", 0.010)),
            "arbitration_lr": float(
                metadata.get("arbitration_lr", metadata.get("module_lr", 0.010))
            ),
            "motor_lr": float(metadata.get("motor_lr", 0.012)),
        },
        "parameter_counts": deepcopy(metadata.get("parameter_counts", {})),
        "total_parameters": int(metadata.get("total_parameters", 0)),
        "training_regime": deepcopy(metadata.get("training_regime")),
    }
    return teacher, teacher_metadata
