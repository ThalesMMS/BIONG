import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.checkpointing import (
    CheckpointPenaltyMode,
    CheckpointSelectionConfig,
    checkpoint_candidate_composite_score,
    checkpoint_candidate_sort_key,
    checkpoint_preload_fingerprint,
    checkpoint_run_fingerprint,
    file_sha256,
    jsonify_observation,
    mean_reward_from_behavior_payload,
    persist_checkpoint_pair,
    resolve_checkpoint_load_dir,
)

def _write_checkpoint_artifact(path: Path, *, label: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(
        json.dumps({"label": label}),
        encoding="utf-8",
    )
    (path / "weights.npz").write_bytes(label.encode("utf-8"))
    return path
