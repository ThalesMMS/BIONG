import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.checkpointing import (
    CheckpointPenaltyMode,
    CheckpointSelectionConfig,
    build_checkpointing_summary,
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

from tests.fixtures.checkpointing import _write_checkpoint_artifact

__all__ = [name for name in globals() if not name.startswith("__")]
