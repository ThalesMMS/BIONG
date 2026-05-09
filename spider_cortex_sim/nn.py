from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .direct_policy_affordances import (
    AFFORDANCE_GEOMETRY_TARGET_NAMES,
    AFFORDANCE_SHELTER_COLUMN_NAMES,
    AFFORDANCE_SHELTER_POSITION_NAMES,
    AFFORDANCE_SHELTER_ROLE_NAMES,
    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
    DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
    DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
)
from .direct_policy_events import EVENT_TYPE_NAMES, EVENT_TYPE_TO_INDEX
from .direct_policy_options import OPTION_NAMES, OPTION_TO_INDEX
from .interfaces import ACTION_TO_INDEX, MODULE_INTERFACES
from .nn_utils import (
    Array,
    _clip_grad_logits,
    _coerce_state_array,
    _parameter_norm_of,
    _sigmoid,
    _state_scalar,
    _validate_state_dict,
    _weight_scale,
    one_hot,
    softmax,
)


def _module_signal_flat_index(module_name: str, signal_name: str) -> int:
    offset = 0
    for spec in MODULE_INTERFACES:
        if spec.name == module_name:
            try:
                signal_offset = spec.signal_names.index(signal_name)
            except ValueError as exc:
                raise KeyError(
                    f"Unknown signal {signal_name!r} for module {module_name!r}."
                ) from exc
            return offset + signal_offset
        offset += spec.input_dim
    raise KeyError(f"Unknown module {module_name!r}.")


_SLEEP_FATIGUE_IDX = _module_signal_flat_index("sleep_center", "fatigue")
_SLEEP_HUNGER_IDX = _module_signal_flat_index("sleep_center", "hunger")
_SLEEP_ON_SHELTER_IDX = _module_signal_flat_index("sleep_center", "on_shelter")
_SLEEP_NIGHT_IDX = _module_signal_flat_index("sleep_center", "night")
_SLEEP_PHASE_LEVEL_IDX = _module_signal_flat_index(
    "sleep_center", "sleep_phase_level"
)
_SLEEP_REST_STREAK_IDX = _module_signal_flat_index(
    "sleep_center", "rest_streak_norm"
)
_SLEEP_DEBT_IDX = _module_signal_flat_index("sleep_center", "sleep_debt")
_SLEEP_SHELTER_ROLE_LEVEL_IDX = _module_signal_flat_index(
    "sleep_center", "shelter_role_level"
)
_SLEEP_SHELTER_MEMORY_AGE_IDX = _module_signal_flat_index(
    "sleep_center", "shelter_memory_age"
)
_HUNGER_FOOD_VISIBLE_IDX = _module_signal_flat_index("hunger_center", "food_visible")
_HUNGER_FOOD_CERTAINTY_IDX = _module_signal_flat_index("hunger_center", "food_certainty")
_HUNGER_FOOD_DX_IDX = _module_signal_flat_index("hunger_center", "food_dx")
_HUNGER_FOOD_DY_IDX = _module_signal_flat_index("hunger_center", "food_dy")
_HUNGER_FOOD_SMELL_STRENGTH_IDX = _module_signal_flat_index("hunger_center", "food_smell_strength")
_HUNGER_FOOD_SMELL_DX_IDX = _module_signal_flat_index("hunger_center", "food_smell_dx")
_HUNGER_FOOD_SMELL_DY_IDX = _module_signal_flat_index("hunger_center", "food_smell_dy")
_HUNGER_FOOD_MEMORY_DX_IDX = _module_signal_flat_index("hunger_center", "food_memory_dx")
_HUNGER_FOOD_MEMORY_DY_IDX = _module_signal_flat_index("hunger_center", "food_memory_dy")
_HUNGER_FOOD_MEMORY_AGE_IDX = _module_signal_flat_index("hunger_center", "food_memory_age")
_ALERT_PREDATOR_VISIBLE_IDX = _module_signal_flat_index(
    "alert_center", "predator_visible"
)
_ALERT_PREDATOR_CERTAINTY_IDX = _module_signal_flat_index(
    "alert_center", "predator_certainty"
)
_ALERT_PREDATOR_SMELL_STRENGTH_IDX = _module_signal_flat_index(
    "alert_center", "predator_smell_strength"
)
_ALERT_PREDATOR_MOTION_SALIENCE_IDX = _module_signal_flat_index(
    "alert_center", "predator_motion_salience"
)
_ALERT_VISUAL_PREDATOR_THREAT_IDX = _module_signal_flat_index(
    "alert_center", "visual_predator_threat"
)
_ALERT_OLFACTORY_PREDATOR_THREAT_IDX = _module_signal_flat_index(
    "alert_center", "olfactory_predator_threat"
)
_ALERT_RECENT_PAIN_IDX = _module_signal_flat_index("alert_center", "recent_pain")
_ALERT_RECENT_CONTACT_IDX = _module_signal_flat_index(
    "alert_center", "recent_contact"
)
_ALERT_PREDATOR_TRACE_STRENGTH_IDX = _module_signal_flat_index(
    "alert_center", "predator_trace_strength"
)
_LOCAL_ACTION_TO_POLICY_INDEX = {
    action_name: int(ACTION_TO_INDEX[action_name])
    for action_name in DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES
}
_POLICY_ORIENTATION_ACTION_INDICES = tuple(
    int(ACTION_TO_INDEX[action_name])
    for action_name in ("ORIENT_UP", "ORIENT_DOWN", "ORIENT_LEFT", "ORIENT_RIGHT")
)
_DEEP_SHELTER_POSITION_INDICES = tuple(
    AFFORDANCE_SHELTER_POSITION_NAMES.index(name)
    for name in ("deep_left", "deep_center", "deep_right")
)
_INSIDE_SHELTER_POSITION_INDICES = tuple(
    AFFORDANCE_SHELTER_POSITION_NAMES.index(name)
    for name in ("inside_left", "inside_center", "inside_right")
)
_ENTRANCE_POSITION_INDICES = tuple(
    AFFORDANCE_SHELTER_POSITION_NAMES.index(name)
    for name in ("entrance_left", "entrance_center", "entrance_right")
)
_OUTSIDE_POSITION_INDEX = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
_GEOMETRY_DEEPEN_INDEX = AFFORDANCE_GEOMETRY_TARGET_NAMES.index("deepen_shelter")
_GEOMETRY_OUTSIDE_INDEX = AFFORDANCE_GEOMETRY_TARGET_NAMES.index("toward_outside")


@dataclass
class ProposalCache:
    x: Array
    h: Array


@dataclass
class RecurrentProposalCache:
    x: Array
    h_prev: Array
    h_new: Array


@dataclass
class EventAttentionCache:
    x: Array
    x_aug: Array
    h_prev: Array
    h_new: Array
    query_input: Array
    query: Array
    slot_raws: Array
    keys: Array
    values: Array
    attention_weights: Array
    valid_event_type_indices: Array


@dataclass
class OptionAttentionCache:
    x: Array
    x_aug: Array
    h_prev: Array
    h_new: Array
    query_input: Array
    query: Array
    slot_raws: Array
    keys: Array
    values: Array
    attention_weights: Array
    valid_event_type_indices: Array
    option_probs: Array
    selected_option_idx: int


@dataclass
class OptionAffordanceFeedbackCache(OptionAttentionCache):
    blocked_probs: Array
    role_probs: Array
    affordance_features: Array
    affordance_feedback: Array


@dataclass
class OptionAffordanceGeometryFeedbackCache(OptionAffordanceFeedbackCache):
    geometry_probs: Array


@dataclass
class OptionAffordanceTopologyFeedbackCache(OptionAffordanceGeometryFeedbackCache):
    shelter_column_probs: Array


@dataclass
class OptionAffordancePositionFeedbackCache(OptionAffordanceGeometryFeedbackCache):
    shelter_position_probs: Array
    transition_prediction_values: Array
    transition_prediction_feedback: Array
    transition_rollout_prediction_values: Array
    transition_rollout_prediction_feedback: Array
    combined_feedback: Array
    phase_probs: Array
    previous_option_vector: Array
    previous_option_idx: int
    previous_action_vector: Array
    selected_option_age_bucket: int
    previous_decoder_action_state: Array
    previous_action_backbone_state: Array
    action_backbone_state: Array
    action_backbone_pre: Array
    previous_action_policy_state: Array
    action_policy_state: Array
    action_policy_pre: Array
    previous_action_controller_state: Array
    action_controller_state: Array
    action_controller_pre: Array
    previous_action_token_state: Array
    action_token_state: Array
    action_token_pre: Array
    decoder_hidden: Array
    decoder_hidden_pre: Array


class ProposalNetwork:
    """Small single-hidden-layer MLP that produces action logits for a cortical subsystem."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator, name: str) -> None:
        """
        Initialize a single-hidden-layer proposal network with randomly initialized weights and zero biases.
        
        Parameters:
            input_dim (int): Dimensionality of the input vector.
            hidden_dim (int): Number of units in the hidden (tanh) layer.
            output_dim (int): Dimensionality of the output logits.
            rng (np.random.Generator): RNG used to draw initial weight values.
            name (str): Human-readable name for the network instance.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name
        self.W1 = rng.normal(0.0, _weight_scale(input_dim), size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2 = rng.normal(0.0, _weight_scale(hidden_dim), size=(output_dim, hidden_dim))
        self.b2 = np.zeros(output_dim, dtype=float)
        self.cache: Optional[ProposalCache] = None

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
        training: bool = False,
    ) -> Array:
        """
        Compute proposal logits from the input and optionally cache the input and hidden activation for later backpropagation.
        
        Parameters:
            x (Array): Input feature vector.
            store_cache (bool): If True, store input and hidden activation in the instance cache for use by backward.
            training (bool): Accepted for parity with modular proposal paths; this deterministic network has no training-only forward behavior.
        
        Returns:
            Array: Proposal logits (float array) with values clipped to the range [-20.0, 20.0].
        """
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        h = np.tanh(self.W1 @ x + self.b1)
        logits = np.clip(
            np.nan_to_num(self.W2 @ h + self.b2, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        if store_cache:
            self.cache = ProposalCache(x=x, h=h)
        return logits

    def backward(self, grad_logits: Array, lr: float, grad_clip: float = 5.0) -> None:
        """
        Apply gradients to the network parameters using the most recent cached input and hidden activation, updating parameters in-place with simple SGD.
        
        The provided `grad_logits` is sanitized (NaNs → 0, positive/negative infinities clamped to ±5) and, if its L2 norm exceeds `grad_clip`, scaled down to that threshold. Gradients for the output head (W2, b2) and the shared hidden layer (W1, b1) are computed using the cached `x` and `h` and the derivative of tanh, then applied as `param -= lr * grad`.
        
        Parameters:
            grad_logits (Array): Gradient of the loss with respect to the output logits.
            lr (float): Learning rate used to scale parameter updates.
            grad_clip (float): Maximum allowed L2 norm for `grad_logits`; gradients with larger norm are scaled down (default 5.0).
        
        Raises:
            RuntimeError: If called before a forward pass populated the cache.
        """
        if self.cache is None:
            raise RuntimeError(f"Network {self.name} backward called without cache.")
        grad_logits = _clip_grad_logits(grad_logits, grad_clip)

        x = self.cache.x
        h = self.cache.h
        grad_W2 = np.outer(grad_logits, h)
        grad_b2 = grad_logits

        dh = self.W2.T @ grad_logits
        dz1 = dh * (1.0 - h**2)
        grad_W1 = np.outer(dz1, x)
        grad_b1 = dz1

        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def state_dict(self) -> dict[str, object]:
        """
        Return a serializable state dictionary containing the network's configuration and parameter arrays.
        
        Returns:
            dict[str, object]: Mapping with keys:
                - "name": network name (str)
                - "input_dim", "hidden_dim", "output_dim": network dimensions (int)
                - "W1", "b1", "W2", "b2": copies of the weight and bias arrays (numpy.ndarray)
        """
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """
        Load and validate the network parameters from `state` and clear the forward cache.
        
        Expects `state` to contain keys "W1", "b1", "W2", and "b2" whose values are array-like objects with exact shapes
        `(hidden_dim, input_dim)`, `(hidden_dim,)`, `(output_dim, hidden_dim)`, and `(output_dim,)` respectively.
        Each array is converted to a float NumPy array before assignment. After loading, any cached forward activations are cleared.
        
        Parameters:
            state (dict[str, object]): Mapping of parameter names to array-like values.
        
        Raises:
            ValueError: If a required key is missing or an array has an unexpected shape.
        """
        _validate_state_dict(
            state,
            expected_keys={"name", "input_dim", "hidden_dim", "output_dim", "W1", "b1", "W2", "b2"},
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
            name=self.name,
        )
        self.W1 = _coerce_state_array(state, "W1", (self.hidden_dim, self.input_dim), name=self.name)
        self.b1 = _coerce_state_array(state, "b1", (self.hidden_dim,), name=self.name)
        self.W2 = _coerce_state_array(state, "W2", (self.output_dim, self.hidden_dim), name=self.name)
        self.b2 = _coerce_state_array(state, "b2", (self.output_dim,), name=self.name)
        self.cache = None

    def parameter_norm(self) -> float:
        """
        Compute the Euclidean (L2) norm of the network's parameters.

        Returns:
            The L2 norm computed as sqrt(sum of squares of W1, b1, W2, and b2) as a float.
        """
        return _parameter_norm_of(self.W1, self.b1, self.W2, self.b2)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the network."""
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)


class RecurrentProposalNetwork:
    """Simple Elman RNN proposer that keeps agent-owned hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator, name: str) -> None:
        """
        Initialize a recurrent proposal network with a single tanh hidden state.

        Parameters:
            input_dim (int): Dimensionality of the input vector.
            hidden_dim (int): Size of the recurrent hidden state.
            output_dim (int): Dimensionality of the output logits.
            rng (np.random.Generator): RNG used to draw initial weight values.
            name (str): Human-readable name for the network instance.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name
        self.W_xh = rng.normal(0.0, _weight_scale(input_dim), size=(hidden_dim, input_dim))
        self.W_hh = rng.normal(0.0, _weight_scale(hidden_dim), size=(hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim, dtype=float)
        self.W_out = rng.normal(0.0, _weight_scale(hidden_dim), size=(output_dim, hidden_dim))
        self.b_out = np.zeros(output_dim, dtype=float)
        self.hidden_state = np.zeros(hidden_dim, dtype=float)
        self.cache: Optional[RecurrentProposalCache] = None

    def reset_hidden_state(self) -> None:
        """Reset the recurrent hidden state and clear any cached activations."""
        self.hidden_state = np.zeros(self.hidden_dim, dtype=float)
        self.cache = None

    def get_hidden_state(self) -> Array:
        """Return a copy of the current recurrent hidden state."""
        return self.hidden_state.copy()

    def set_hidden_state(self, hidden_state: Array) -> None:
        """Replace the current recurrent hidden state with a validated copy."""
        hidden_state = np.asarray(hidden_state, dtype=float)
        if hidden_state.shape != (self.hidden_dim,):
            raise ValueError(
                f"{self.name}: hidden_state expected {(self.hidden_dim,)}, "
                f"received {hidden_state.shape}"
            )
        self.hidden_state = hidden_state.copy()

    def forward(self, x: Array, *, store_cache: bool = True) -> Array:
        """
        Compute proposal logits and advance the recurrent hidden state.

        The previous hidden state is treated as fixed context for a single-step
        update; backward updates recurrent parameters from the most recent step
        without backpropagating through earlier timesteps.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        h_new = np.tanh(self.W_xh @ x + self.W_hh @ h_prev + self.b_h)
        logits = np.clip(
            np.nan_to_num(self.W_out @ h_new + self.b_out, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        self.hidden_state = h_new.copy()
        if store_cache:
            self.cache = RecurrentProposalCache(x=x, h_prev=h_prev, h_new=h_new)
        return logits

    def backward(self, grad_logits: Array, lr: float, grad_clip: float = 5.0) -> None:
        """
        Apply a single-step recurrent update using the latest cached transition.

        Gradients are computed for W_out/b_out and for the current recurrent
        transition W_xh/W_hh/b_h. The cached h_prev is treated as constant,
        matching the simulator's existing single-step TD update pattern.
        """
        if self.cache is None:
            raise RuntimeError(f"Network {self.name} backward called without cache.")
        grad_logits = _clip_grad_logits(grad_logits, grad_clip)
        if grad_logits.shape != (self.output_dim,):
            raise ValueError(
                f"{self.name}: grad_logits expected shape {(self.output_dim,)}, "
                f"received {grad_logits.shape}"
            )

        x = self.cache.x
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        grad_W_out = np.outer(grad_logits, h_new)
        grad_b_out = grad_logits

        dh = self.W_out.T @ grad_logits
        dz = dh * (1.0 - h_new**2)
        grad_W_xh = np.outer(dz, x)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz

        self.W_out -= lr * grad_W_out
        self.b_out -= lr * grad_b_out
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h

    def state_dict(self) -> dict[str, object]:
        """
        Return a serializable parameter snapshot without live hidden state.

        Hidden state is episode-local runtime state, so checkpoints persist only
        the recurrent weights and biases.
        """
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "W_out": self.W_out.copy(),
            "b_out": self.b_out.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Load recurrent parameters from `state` and reset runtime hidden state."""
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "W_xh",
                "W_hh",
                "b_h",
                "W_out",
                "b_out",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
            name=self.name,
        )
        new_W_xh = _coerce_state_array(
            state,
            "W_xh",
            (self.hidden_dim, self.input_dim),
            name=self.name,
        )
        new_W_hh = _coerce_state_array(
            state,
            "W_hh",
            (self.hidden_dim, self.hidden_dim),
            name=self.name,
        )
        new_b_h = _coerce_state_array(
            state,
            "b_h",
            (self.hidden_dim,),
            name=self.name,
        )
        new_W_out = _coerce_state_array(
            state,
            "W_out",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        new_b_out = _coerce_state_array(
            state,
            "b_out",
            (self.output_dim,),
            name=self.name,
        )
        self.W_xh = new_W_xh
        self.W_hh = new_W_hh
        self.b_h = new_b_h
        self.W_out = new_W_out
        self.b_out = new_b_out
        self.reset_hidden_state()

    def parameter_norm(self) -> float:
        """Return the L2 norm of all recurrent proposer parameters."""
        return _parameter_norm_of(self.W_xh, self.W_hh, self.b_h, self.W_out, self.b_out)

    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the recurrent proposer."""
        return int(
            self.W_xh.size
            + self.W_hh.size
            + self.b_h.size
            + self.W_out.size
            + self.b_out.size
        )


@dataclass
class MotorCache:
    x: Array
    h: Array


@dataclass
class DeepMotorCache:
    x: Array
    hidden_states: tuple[Array, ...]


class MotorNetwork:
    """Motor network with a corrective policy head and a value critic head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator, name: str = "motor_cortex") -> None:
        """
        Initialize the MotorNetwork's parameters and empty runtime cache.
        
        Parameters:
            input_dim (int): Dimension of the input feature vector.
            hidden_dim (int): Size of the shared hidden layer.
            output_dim (int): Number of policy logits (action space size).
            rng (np.random.Generator): Random number generator used to sample initial weights.
            name (str): Human-readable name for the network instance (default "motor_cortex").
        
        Notes:
            - Shared and head weights are sampled from a normal distribution with standard
              deviation given by `_weight_scale(...)`; all biases are initialized to zero.
            - `self.cache` is initialized to `None`.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name
        self.W1 = rng.normal(0.0, _weight_scale(input_dim), size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2_policy = rng.normal(0.0, _weight_scale(hidden_dim), size=(output_dim, hidden_dim))
        self.b2_policy = np.zeros(output_dim, dtype=float)
        self.W2_value = rng.normal(0.0, _weight_scale(hidden_dim), size=(1, hidden_dim))
        self.b2_value = np.zeros(1, dtype=float)
        self.cache: Optional[MotorCache] = None

    def forward(self, x: Array, *, store_cache: bool = True) -> tuple[Array, float]:
        """
        Compute policy logits and a scalar value from input features using the network's shared trunk and two heads.
        
        Parameters:
            x (Array): Input feature vector; NaNs and infinities are sanitized before computation.
            store_cache (bool): If True, save inputs and hidden activation for later backpropagation.
        
        Returns:
            tuple[Array, float]: A pair where the first element is the clipped policy logits vector and the second is the scalar value estimate.
        """
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        h = np.tanh(self.W1 @ x + self.b1)
        policy_logits = np.clip(
            np.nan_to_num(self.W2_policy @ h + self.b2_policy, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        value = float(np.nan_to_num((self.W2_value @ h + self.b2_value)[0], nan=0.0, posinf=20.0, neginf=-20.0))
        if store_cache:
            self.cache = MotorCache(x=x, h=h)
        return policy_logits, value

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
    ) -> Array:
        """
        Apply gradients to the network's parameters using the cached forward activations and return the gradient with respect to the input.
        
        Parameters:
            grad_policy_logits (Array): Gradient of the loss with respect to the policy logits; will be sanitized and clipped by Euclidean norm to at most `grad_clip`.
            grad_value (float): Gradient of the loss with respect to the scalar value head; will be clipped to the range `[-grad_clip, grad_clip]`.
            lr (float): Learning rate applied to parameter updates.
            grad_clip (float): Threshold for gradient clipping (default 5.0).
        
        Returns:
            Array: Gradient of the loss with respect to the network input `x`.
        
        Raises:
            RuntimeError: If no forward cache is available when calling backward.
        """
        if self.cache is None:
            raise RuntimeError("Motor network backward called without cache.")
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))

        x = self.cache.x
        h = self.cache.h

        grad_W2_policy = np.outer(grad_policy_logits, h)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)

        dh = self.W2_policy.T @ grad_policy_logits + self.W2_value.T[:, 0] * grad_value
        dz1 = dh * (1.0 - h**2)
        grad_x = self.W1.T @ dz1
        grad_W1 = np.outer(dz1, x)
        grad_b1 = dz1

        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        return grad_x

    def value_only(self, x: Array) -> float:
        _, value = self.forward(x, store_cache=False)
        return value

    def state_dict(self) -> dict[str, object]:
        """
        Return a serializable snapshot of the network's configuration and parameters.
        
        Returns:
            A dict containing:
                - "name" (str): Network name.
                - "input_dim" (int): Input dimension.
                - "hidden_dim" (int): Hidden layer dimension.
                - "output_dim" (int): Output dimension.
                - "W1" (Array): Copy of the first-layer weight matrix.
                - "b1" (Array): Copy of the first-layer bias vector.
                - "W2_policy" (Array): Copy of the policy head weight matrix.
                - "b2_policy" (Array): Copy of the policy head bias vector.
                - "W2_value" (Array): Copy of the value head weight matrix.
                - "b2_value" (Array): Copy of the value head bias vector.
        """
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2_policy": self.W2_policy.copy(),
            "b2_policy": self.b2_policy.copy(),
            "W2_value": self.W2_value.copy(),
            "b2_value": self.b2_value.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """
        Load and validate network parameters from a state dictionary and reset the forward cache.
        
        Each parameter array is coerced and checked for the exact expected shape; on success the internal parameter arrays are replaced and the cached activations are cleared.
        
        Raises:
            ValueError: If a required key is missing or the associated array does not match the expected shape.
        """
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "W1",
                "b1",
                "W2_policy",
                "b2_policy",
                "W2_value",
                "b2_value",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
            name=self.name,
        )
        self.W1 = _coerce_state_array(state, "W1", (self.hidden_dim, self.input_dim), name=self.name)
        self.b1 = _coerce_state_array(state, "b1", (self.hidden_dim,), name=self.name)
        self.W2_policy = _coerce_state_array(
            state,
            "W2_policy",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_policy = _coerce_state_array(state, "b2_policy", (self.output_dim,), name=self.name)
        self.W2_value = _coerce_state_array(state, "W2_value", (1, self.hidden_dim), name=self.name)
        self.b2_value = _coerce_state_array(state, "b2_value", (1,), name=self.name)
        self.cache = None

    def parameter_norm(self) -> float:
        """
        Compute the Euclidean (L2) norm of all learnable parameters.
        
        Returns:
            float: L2 norm (square root of the sum of squares) of W1, b1, W2_policy, b2_policy, W2_value, and b2_value.
        """
        return _parameter_norm_of(
            self.W1, self.b1, self.W2_policy, self.b2_policy, self.W2_value, self.b2_value
        )

    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the motor network."""
        return int(
            self.W1.size
            + self.b1.size
            + self.W2_policy.size
            + self.b2_policy.size
            + self.W2_value.size
            + self.b2_value.size
        )


class TrueMonolithicNetwork(MotorNetwork):
    """Direct policy+value network for the true monolithic baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        name: str = "true_monolithic_policy",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            name=name,
        )


class RecurrentTrueMonolithicNetwork:
    """Direct policy+value recurrent controller for diagnostic true-monolithic runs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        name: str = "true_monolithic_policy",
        phase_output_dim: int = 0,
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.phase_output_dim = int(phase_output_dim)
        self.name = name
        self.W_xh = rng.normal(0.0, _weight_scale(input_dim), size=(hidden_dim, input_dim))
        self.W_hh = rng.normal(0.0, _weight_scale(hidden_dim), size=(hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim, dtype=float)
        self.W2_policy = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(output_dim, hidden_dim),
        )
        self.b2_policy = np.zeros(output_dim, dtype=float)
        self.W2_value = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(1, hidden_dim),
        )
        self.b2_value = np.zeros(1, dtype=float)
        self.W2_phase = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.phase_output_dim, hidden_dim),
            )
            if self.phase_output_dim > 0
            else np.zeros((0, hidden_dim), dtype=float)
        )
        self.b2_phase = (
            np.zeros(self.phase_output_dim, dtype=float)
            if self.phase_output_dim > 0
            else np.zeros(0, dtype=float)
        )
        self.hidden_state = np.zeros(hidden_dim, dtype=float)
        self.cache: Optional[RecurrentProposalCache] = None

    def reset_hidden_state(self) -> None:
        self.hidden_state = np.zeros(self.hidden_dim, dtype=float)
        self.cache = None

    def get_hidden_state(self) -> Array:
        return self.hidden_state.copy()

    def set_hidden_state(self, hidden_state: Array) -> None:
        hidden_state = np.asarray(hidden_state, dtype=float)
        if hidden_state.shape != (self.hidden_dim,):
            raise ValueError(
                f"{self.name}: hidden_state expected {(self.hidden_dim,)}, "
                f"received {hidden_state.shape}"
            )
        self.hidden_state = hidden_state.copy()

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array | None]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        h_new = np.tanh(self.W_xh @ x + self.W_hh @ h_prev + self.b_h)
        policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        value = float(
            np.nan_to_num(
                (self.W2_value @ h_new + self.b2_value)[0],
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        phase_logits: Array | None = None
        if self.phase_output_dim > 0:
            phase_logits = np.clip(
                np.nan_to_num(
                    self.W2_phase @ h_new + self.b2_phase,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ),
                -20.0,
                20.0,
            )
        self.hidden_state = h_new.copy()
        if store_cache:
            self.cache = RecurrentProposalCache(x=x, h_prev=h_prev, h_new=h_new)
        return policy_logits, value, phase_logits

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
        grad_phase_logits: Array | None = None,
    ) -> Array:
        if self.cache is None:
            raise RuntimeError("Recurrent true monolithic network backward called without cache.")
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        x = self.cache.x
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        grad_W2_policy = np.outer(grad_policy_logits, h_new)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        if grad_phase_logits is None or self.phase_output_dim <= 0:
            grad_phase_logits = np.zeros(self.phase_output_dim, dtype=float)
        else:
            grad_phase_logits = _clip_grad_logits(grad_phase_logits, grad_clip)
        grad_W2_phase = (
            np.outer(grad_phase_logits, h_new)
            if self.phase_output_dim > 0
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_b2_phase = np.asarray(grad_phase_logits, dtype=float)
        dh = (
            self.W2_policy.T @ grad_policy_logits
            + self.W2_value.T[:, 0] * grad_value
            + (
                self.W2_phase.T @ grad_phase_logits
                if self.phase_output_dim > 0
                else 0.0
            )
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs = self.W_xh.T @ dz
        grad_W_xh = np.outer(dz, x)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz
        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        if self.phase_output_dim > 0:
            self.W2_phase -= lr * grad_W2_phase
            self.b2_phase -= lr * grad_b2_phase
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        return grad_inputs

    def value_only(self, x: Array) -> float:
        _, value, _ = self.forward(x, store_cache=False)
        return value

    def state_dict(self) -> dict[str, object]:
        state = {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "recurrent": True,
            "phase_output_dim": self.phase_output_dim,
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "W2_policy": self.W2_policy.copy(),
            "b2_policy": self.b2_policy.copy(),
            "W2_value": self.W2_value.copy(),
            "b2_value": self.b2_value.copy(),
        }
        if self.phase_output_dim > 0:
            state["W2_phase"] = self.W2_phase.copy()
            state["b2_phase"] = self.b2_phase.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "recurrent",
                "phase_output_dim",
                "W_xh",
                "W_hh",
                "b_h",
                "W2_policy",
                "b2_policy",
                "W2_value",
                "b2_value",
                *(
                    {"W2_phase", "b2_phase"}
                    if self.phase_output_dim > 0
                    else set()
                ),
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "phase_output_dim": self.phase_output_dim,
            },
            name=self.name,
        )
        self.W_xh = _coerce_state_array(state, "W_xh", (self.hidden_dim, self.input_dim), name=self.name)
        self.W_hh = _coerce_state_array(state, "W_hh", (self.hidden_dim, self.hidden_dim), name=self.name)
        self.b_h = _coerce_state_array(state, "b_h", (self.hidden_dim,), name=self.name)
        self.W2_policy = _coerce_state_array(
            state,
            "W2_policy",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_policy = _coerce_state_array(state, "b2_policy", (self.output_dim,), name=self.name)
        self.W2_value = _coerce_state_array(state, "W2_value", (1, self.hidden_dim), name=self.name)
        self.b2_value = _coerce_state_array(state, "b2_value", (1,), name=self.name)
        if self.phase_output_dim > 0:
            self.W2_phase = _coerce_state_array(
                state,
                "W2_phase",
                (self.phase_output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_phase = _coerce_state_array(
                state,
                "b2_phase",
                (self.phase_output_dim,),
                name=self.name,
            )
        self.reset_hidden_state()

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_phase,
            self.b2_phase,
        )

    def count_parameters(self) -> int:
        return int(
            self.W_xh.size
            + self.W_hh.size
            + self.b_h.size
            + self.W2_policy.size
            + self.b2_policy.size
            + self.W2_value.size
            + self.b2_value.size
            + self.W2_phase.size
            + self.b2_phase.size
        )


class RecurrentEventAttentionTrueMonolithicNetwork:
    """Recurrent direct policy with a small learned event-attention memory."""

    event_feature_dim = 5
    event_embedding_dim = 8
    event_context_dim = 8

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        name: str = "true_monolithic_policy",
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.event_buffer_size = int(event_buffer_size)
        self.name = name
        recurrent_input_dim = self.input_dim + self.event_context_dim
        self.W_xh = rng.normal(
            0.0,
            _weight_scale(recurrent_input_dim),
            size=(hidden_dim, recurrent_input_dim),
        )
        self.W_hh = rng.normal(0.0, _weight_scale(hidden_dim), size=(hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim, dtype=float)
        self.W2_policy = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(output_dim, hidden_dim),
        )
        self.b2_policy = np.zeros(output_dim, dtype=float)
        self.W2_value = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(1, hidden_dim),
        )
        self.b2_value = np.zeros(1, dtype=float)
        event_raw_dim = self.event_embedding_dim + self.event_feature_dim + 1
        self.W_query = rng.normal(
            0.0,
            _weight_scale(self.input_dim + hidden_dim),
            size=(self.event_context_dim, self.input_dim + hidden_dim),
        )
        self.b_query = np.zeros(self.event_context_dim, dtype=float)
        self.W_key = rng.normal(
            0.0,
            _weight_scale(event_raw_dim),
            size=(self.event_context_dim, event_raw_dim),
        )
        self.b_key = np.zeros(self.event_context_dim, dtype=float)
        self.W_value = rng.normal(
            0.0,
            _weight_scale(event_raw_dim),
            size=(self.event_context_dim, event_raw_dim),
        )
        self.b_value = np.zeros(self.event_context_dim, dtype=float)
        self.event_type_embeddings = rng.normal(
            0.0,
            _weight_scale(self.event_embedding_dim),
            size=(len(EVENT_TYPE_NAMES), self.event_embedding_dim),
        )
        self.hidden_state = np.zeros(hidden_dim, dtype=float)
        self.event_type_buffer = np.full(self.event_buffer_size, -1, dtype=int)
        self.event_time_buffer = np.full(self.event_buffer_size, -1, dtype=int)
        self.event_feature_buffer = np.zeros(
            (self.event_buffer_size, self.event_feature_dim),
            dtype=float,
        )
        self.event_clock = 0
        self.cache: Optional[EventAttentionCache] = None
        self.last_attention_summary: dict[str, object] = {
            "event_attention_top_type": None,
            "event_attention_top_age": -1,
            "event_attention_entropy": 0.0,
        }

    def reset_hidden_state(self) -> None:
        self.hidden_state = np.zeros(self.hidden_dim, dtype=float)
        self.cache = None

    def reset_event_memory(self) -> None:
        self.event_type_buffer.fill(-1)
        self.event_time_buffer.fill(-1)
        self.event_feature_buffer.fill(0.0)
        self.event_clock = 0
        self.last_attention_summary = {
            "event_attention_top_type": None,
            "event_attention_top_age": -1,
            "event_attention_entropy": 0.0,
        }

    def get_hidden_state(self) -> Array:
        return self.hidden_state.copy()

    def set_hidden_state(self, hidden_state: Array) -> None:
        hidden_state = np.asarray(hidden_state, dtype=float)
        if hidden_state.shape != (self.hidden_dim,):
            raise ValueError(
                f"{self.name}: hidden_state expected {(self.hidden_dim,)}, "
                f"received {hidden_state.shape}"
            )
        self.hidden_state = hidden_state.copy()

    def set_event_clock(self, tick: int) -> None:
        self.event_clock = max(0, int(tick))

    def record_event(self, event_type: str, *, features: Array, tick: int | None = None) -> None:
        if tick is not None:
            self.set_event_clock(int(tick))
        feature_array = np.asarray(features, dtype=float)
        if feature_array.shape != (self.event_feature_dim,):
            raise ValueError(
                f"{self.name}: event features expected {(self.event_feature_dim,)}, "
                f"received {feature_array.shape}"
            )
        if event_type not in EVENT_TYPE_TO_INDEX:
            raise ValueError(f"{self.name}: unknown event type {event_type!r}")
        if self.event_buffer_size <= 0:
            return
        self.event_type_buffer[1:] = self.event_type_buffer[:-1]
        self.event_time_buffer[1:] = self.event_time_buffer[:-1]
        self.event_feature_buffer[1:] = self.event_feature_buffer[:-1]
        self.event_type_buffer[0] = int(EVENT_TYPE_TO_INDEX[event_type])
        self.event_time_buffer[0] = int(self.event_clock)
        self.event_feature_buffer[0] = np.nan_to_num(
            feature_array,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    def _attention_context(
        self,
        x: Array,
        h_prev: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        valid_mask = self.event_type_buffer >= 0
        valid_indices = np.nonzero(valid_mask)[0]
        query_input = np.concatenate([x, h_prev], axis=0)
        query = np.tanh(self.W_query @ query_input + self.b_query)
        if valid_indices.size <= 0:
            self.last_attention_summary = {
                "event_attention_top_type": None,
                "event_attention_top_age": -1,
                "event_attention_entropy": 0.0,
            }
            empty_slots = np.zeros(
                (0, self.event_embedding_dim + self.event_feature_dim + 1),
                dtype=float,
            )
            empty_hidden = np.zeros((0, self.event_context_dim), dtype=float)
            empty_types = np.zeros(0, dtype=int)
            empty_weights = np.zeros(0, dtype=float)
            return (
                np.zeros(self.event_context_dim, dtype=float),
                query_input,
                query,
                empty_slots,
                empty_hidden,
                empty_hidden.copy(),
                empty_weights,
                empty_types,
            )
        event_type_indices = self.event_type_buffer[valid_indices].astype(int, copy=True)
        ages = np.maximum(
            0,
            self.event_clock - self.event_time_buffer[valid_indices],
        ).astype(float, copy=False)
        age_norm = np.clip(ages / 32.0, 0.0, 1.0).reshape(-1, 1)
        slot_raws = np.concatenate(
            [
                self.event_type_embeddings[event_type_indices],
                age_norm,
                self.event_feature_buffer[valid_indices],
            ],
            axis=1,
        )
        keys = np.tanh(slot_raws @ self.W_key.T + self.b_key)
        values = np.tanh(slot_raws @ self.W_value.T + self.b_value)
        scale = float(np.sqrt(max(1, self.event_context_dim)))
        scores = (keys @ query) / scale
        attention_weights = softmax(scores)
        context = np.sum(values * attention_weights.reshape(-1, 1), axis=0)
        top_index = int(np.argmax(attention_weights))
        entropy = max(
            0.0,
            float(-np.sum(attention_weights * np.log(attention_weights + 1e-8))),
        )
        self.last_attention_summary = {
            "event_attention_top_type": EVENT_TYPE_NAMES[event_type_indices[top_index]],
            "event_attention_top_age": int(ages[top_index]),
            "event_attention_entropy": entropy,
        }
        return (
            context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        )

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        (
            event_context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        ) = self._attention_context(x, h_prev)
        x_aug = np.concatenate([x, event_context], axis=0)
        h_new = np.tanh(self.W_xh @ x_aug + self.W_hh @ h_prev + self.b_h)
        policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        value = float(
            np.nan_to_num(
                (self.W2_value @ h_new + self.b2_value)[0],
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        self.hidden_state = h_new.copy()
        if store_cache:
            self.cache = EventAttentionCache(
                x=x,
                x_aug=x_aug,
                h_prev=h_prev,
                h_new=h_new,
                query_input=query_input,
                query=query,
                slot_raws=slot_raws,
                keys=keys,
                values=values,
                attention_weights=attention_weights,
                valid_event_type_indices=event_type_indices,
            )
        return policy_logits, value

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
    ) -> Array:
        if self.cache is None:
            raise RuntimeError(
                "Recurrent event-attention true monolithic network backward called without cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        x = self.cache.x
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        grad_W2_policy = np.outer(grad_policy_logits, h_new)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        dh = self.W2_policy.T @ grad_policy_logits + self.W2_value.T[:, 0] * grad_value
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[self.input_dim :].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz

        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)

        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            event_embedding_width = self.event_embedding_dim
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value

                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    :event_embedding_width
                ]

            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query

        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        return grad_x

    def value_only(self, x: Array) -> float:
        _, value = self.forward(x, store_cache=False)
        return value

    def state_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "recurrent": True,
            "event_attention": True,
            "event_buffer_size": self.event_buffer_size,
            "event_embedding_dim": self.event_embedding_dim,
            "event_context_dim": self.event_context_dim,
            "event_feature_dim": self.event_feature_dim,
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "W2_policy": self.W2_policy.copy(),
            "b2_policy": self.b2_policy.copy(),
            "W2_value": self.W2_value.copy(),
            "b2_value": self.b2_value.copy(),
            "W_query": self.W_query.copy(),
            "b_query": self.b_query.copy(),
            "W_key": self.W_key.copy(),
            "b_key": self.b_key.copy(),
            "W_value": self.W_value.copy(),
            "b_value": self.b_value.copy(),
            "event_type_embeddings": self.event_type_embeddings.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "recurrent",
                "event_attention",
                "event_buffer_size",
                "event_embedding_dim",
                "event_context_dim",
                "event_feature_dim",
                "W_xh",
                "W_hh",
                "b_h",
                "W2_policy",
                "b2_policy",
                "W2_value",
                "b2_value",
                "W_query",
                "b_query",
                "W_key",
                "b_key",
                "W_value",
                "b_value",
                "event_type_embeddings",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
            },
            name=self.name,
        )
        recurrent_input_dim = self.input_dim + self.event_context_dim
        event_raw_dim = self.event_embedding_dim + self.event_feature_dim + 1
        self.W_xh = _coerce_state_array(
            state,
            "W_xh",
            (self.hidden_dim, recurrent_input_dim),
            name=self.name,
        )
        self.W_hh = _coerce_state_array(
            state,
            "W_hh",
            (self.hidden_dim, self.hidden_dim),
            name=self.name,
        )
        self.b_h = _coerce_state_array(state, "b_h", (self.hidden_dim,), name=self.name)
        self.W2_policy = _coerce_state_array(
            state,
            "W2_policy",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_policy = _coerce_state_array(
            state,
            "b2_policy",
            (self.output_dim,),
            name=self.name,
        )
        self.W2_value = _coerce_state_array(
            state,
            "W2_value",
            (1, self.hidden_dim),
            name=self.name,
        )
        self.b2_value = _coerce_state_array(state, "b2_value", (1,), name=self.name)
        self.W_query = _coerce_state_array(
            state,
            "W_query",
            (self.event_context_dim, self.input_dim + self.hidden_dim),
            name=self.name,
        )
        self.b_query = _coerce_state_array(
            state,
            "b_query",
            (self.event_context_dim,),
            name=self.name,
        )
        self.W_key = _coerce_state_array(
            state,
            "W_key",
            (self.event_context_dim, event_raw_dim),
            name=self.name,
        )
        self.b_key = _coerce_state_array(
            state,
            "b_key",
            (self.event_context_dim,),
            name=self.name,
        )
        self.W_value = _coerce_state_array(
            state,
            "W_value",
            (self.event_context_dim, event_raw_dim),
            name=self.name,
        )
        self.b_value = _coerce_state_array(
            state,
            "b_value",
            (self.event_context_dim,),
            name=self.name,
        )
        self.event_type_embeddings = _coerce_state_array(
            state,
            "event_type_embeddings",
            (len(EVENT_TYPE_NAMES), self.event_embedding_dim),
            name=self.name,
        )
        self.reset_hidden_state()
        self.reset_event_memory()

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            self.W_xh.size
            + self.W_hh.size
            + self.b_h.size
            + self.W2_policy.size
            + self.b2_policy.size
            + self.W2_value.size
            + self.b2_value.size
            + self.W_query.size
            + self.b_query.size
            + self.W_key.size
            + self.b_key.size
            + self.W_value.size
            + self.b_value.size
            + self.event_type_embeddings.size
        )


class RecurrentOptionTrueMonolithicNetwork:
    """Recurrent direct policy with event attention and short-lived option commitment."""

    event_feature_dim = 5
    event_embedding_dim = 8
    event_context_dim = 8

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        name: str = "true_monolithic_policy",
    ) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.event_buffer_size = int(event_buffer_size)
        self.option_dim = len(OPTION_NAMES)
        self.option_ttl = max(1, int(option_ttl))
        self.name = name
        recurrent_input_dim = (
            self.input_dim + self.event_context_dim + self.option_dim
        )
        self.W_xh = rng.normal(
            0.0,
            _weight_scale(recurrent_input_dim),
            size=(hidden_dim, recurrent_input_dim),
        )
        self.W_hh = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(hidden_dim, hidden_dim),
        )
        self.b_h = np.zeros(hidden_dim, dtype=float)
        self.W2_policy = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(output_dim, hidden_dim),
        )
        self.b2_policy = np.zeros(output_dim, dtype=float)
        self.W2_value = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(1, hidden_dim),
        )
        self.b2_value = np.zeros(1, dtype=float)
        self.W2_option = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.option_dim, hidden_dim),
        )
        self.b2_option = np.zeros(self.option_dim, dtype=float)
        self.option_action_bias = rng.normal(
            0.0,
            _weight_scale(self.option_dim),
            size=(self.option_dim, self.output_dim),
        )
        event_raw_dim = self.event_embedding_dim + self.event_feature_dim + 1
        self.W_query = rng.normal(
            0.0,
            _weight_scale(self.input_dim + hidden_dim),
            size=(self.event_context_dim, self.input_dim + hidden_dim),
        )
        self.b_query = np.zeros(self.event_context_dim, dtype=float)
        self.W_key = rng.normal(
            0.0,
            _weight_scale(event_raw_dim),
            size=(self.event_context_dim, event_raw_dim),
        )
        self.b_key = np.zeros(self.event_context_dim, dtype=float)
        self.W_value = rng.normal(
            0.0,
            _weight_scale(event_raw_dim),
            size=(self.event_context_dim, event_raw_dim),
        )
        self.b_value = np.zeros(self.event_context_dim, dtype=float)
        self.event_type_embeddings = rng.normal(
            0.0,
            _weight_scale(self.event_embedding_dim),
            size=(len(EVENT_TYPE_NAMES), self.event_embedding_dim),
        )
        self.hidden_state = np.zeros(hidden_dim, dtype=float)
        self.event_type_buffer = np.full(self.event_buffer_size, -1, dtype=int)
        self.event_time_buffer = np.full(self.event_buffer_size, -1, dtype=int)
        self.event_feature_buffer = np.zeros(
            (self.event_buffer_size, self.event_feature_dim),
            dtype=float,
        )
        self.event_clock = 0
        self.current_option_idx = -1
        self.current_option_age = 0
        self.current_option_steps_remaining = 0
        self.cache: Optional[OptionAttentionCache] = None
        self.last_attention_summary: dict[str, object] = {
            "event_attention_top_type": None,
            "event_attention_top_age": -1,
            "event_attention_entropy": 0.0,
        }
        self.last_option_summary: dict[str, object] = {
            "selected_option": None,
            "option_age": -1,
            "option_termination_reason": "none",
            "option_logits": [],
        }

    def reset_hidden_state(self) -> None:
        self.hidden_state = np.zeros(self.hidden_dim, dtype=float)
        self.cache = None
        self.current_option_idx = -1
        self.current_option_age = 0
        self.current_option_steps_remaining = 0
        self.last_option_summary = {
            "selected_option": None,
            "option_age": -1,
            "option_termination_reason": "none",
            "option_logits": [],
        }

    def reset_event_memory(self) -> None:
        self.event_type_buffer.fill(-1)
        self.event_time_buffer.fill(-1)
        self.event_feature_buffer.fill(0.0)
        self.event_clock = 0
        self.last_attention_summary = {
            "event_attention_top_type": None,
            "event_attention_top_age": -1,
            "event_attention_entropy": 0.0,
        }

    def get_hidden_state(self) -> Array:
        return self.hidden_state.copy()

    def set_hidden_state(self, hidden_state: Array) -> None:
        hidden_state = np.asarray(hidden_state, dtype=float)
        if hidden_state.shape != (self.hidden_dim,):
            raise ValueError(
                f"{self.name}: hidden_state expected {(self.hidden_dim,)}, "
                f"received {hidden_state.shape}"
            )
        self.hidden_state = hidden_state.copy()

    def get_runtime_state(self) -> dict[str, object]:
        return {
            "hidden_state": self.hidden_state.copy(),
            "current_option_idx": int(self.current_option_idx),
            "current_option_age": int(self.current_option_age),
            "current_option_steps_remaining": int(self.current_option_steps_remaining),
        }

    def set_runtime_state(self, runtime_state: dict[str, object]) -> None:
        self.set_hidden_state(np.asarray(runtime_state["hidden_state"], dtype=float))
        self.current_option_idx = int(runtime_state["current_option_idx"])
        self.current_option_age = int(runtime_state["current_option_age"])
        self.current_option_steps_remaining = int(
            runtime_state["current_option_steps_remaining"]
        )

    def set_event_clock(self, tick: int) -> None:
        self.event_clock = max(0, int(tick))

    def record_event(
        self,
        event_type: str,
        *,
        features: Array,
        tick: int | None = None,
    ) -> None:
        if tick is not None:
            self.set_event_clock(int(tick))
        feature_array = np.asarray(features, dtype=float)
        if feature_array.shape != (self.event_feature_dim,):
            raise ValueError(
                f"{self.name}: event features expected {(self.event_feature_dim,)}, "
                f"received {feature_array.shape}"
            )
        if event_type not in EVENT_TYPE_TO_INDEX:
            raise ValueError(f"{self.name}: unknown event type {event_type!r}")
        if self.event_buffer_size <= 0:
            return
        self.event_type_buffer[1:] = self.event_type_buffer[:-1]
        self.event_time_buffer[1:] = self.event_time_buffer[:-1]
        self.event_feature_buffer[1:] = self.event_feature_buffer[:-1]
        self.event_type_buffer[0] = int(EVENT_TYPE_TO_INDEX[event_type])
        self.event_time_buffer[0] = int(self.event_clock)
        self.event_feature_buffer[0] = np.nan_to_num(
            feature_array,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

    def _attention_context(
        self,
        x: Array,
        h_prev: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        valid_mask = self.event_type_buffer >= 0
        valid_indices = np.nonzero(valid_mask)[0]
        query_input = np.concatenate([x, h_prev], axis=0)
        query = np.tanh(self.W_query @ query_input + self.b_query)
        if valid_indices.size <= 0:
            self.last_attention_summary = {
                "event_attention_top_type": None,
                "event_attention_top_age": -1,
                "event_attention_entropy": 0.0,
            }
            empty_slots = np.zeros(
                (0, self.event_embedding_dim + self.event_feature_dim + 1),
                dtype=float,
            )
            empty_hidden = np.zeros((0, self.event_context_dim), dtype=float)
            empty_types = np.zeros(0, dtype=int)
            empty_weights = np.zeros(0, dtype=float)
            return (
                np.zeros(self.event_context_dim, dtype=float),
                query_input,
                query,
                empty_slots,
                empty_hidden,
                empty_hidden.copy(),
                empty_weights,
                empty_types,
            )
        event_type_indices = self.event_type_buffer[valid_indices].astype(int, copy=True)
        ages = np.maximum(
            0,
            self.event_clock - self.event_time_buffer[valid_indices],
        ).astype(float, copy=False)
        age_norm = np.clip(ages / 32.0, 0.0, 1.0).reshape(-1, 1)
        slot_raws = np.concatenate(
            [
                self.event_type_embeddings[event_type_indices],
                age_norm,
                self.event_feature_buffer[valid_indices],
            ],
            axis=1,
        )
        keys = np.tanh(slot_raws @ self.W_key.T + self.b_key)
        values = np.tanh(slot_raws @ self.W_value.T + self.b_value)
        scale = float(np.sqrt(max(1, self.event_context_dim)))
        scores = (keys @ query) / scale
        attention_weights = softmax(scores)
        context = np.sum(values * attention_weights.reshape(-1, 1), axis=0)
        top_index = int(np.argmax(attention_weights))
        entropy = max(
            0.0,
            float(-np.sum(attention_weights * np.log(attention_weights + 1e-8))),
        )
        self.last_attention_summary = {
            "event_attention_top_type": EVENT_TYPE_NAMES[event_type_indices[top_index]],
            "event_attention_top_age": int(ages[top_index]),
            "event_attention_entropy": entropy,
        }
        return (
            context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        )

    def _current_option_vector(self) -> Array:
        if self.current_option_idx < 0:
            return np.zeros(self.option_dim, dtype=float)
        return one_hot(self.current_option_idx, self.option_dim)

    def _latest_event(self) -> tuple[str | None, int]:
        if self.event_buffer_size <= 0 or int(self.event_type_buffer[0]) < 0:
            return None, -1
        event_idx = int(self.event_type_buffer[0])
        age = max(0, int(self.event_clock - self.event_time_buffer[0]))
        return EVENT_TYPE_NAMES[event_idx], age

    def _termination_reason(self) -> str | None:
        if self.current_option_idx < 0:
            return None
        latest_event, latest_age = self._latest_event()
        if latest_age == 0 and latest_event == "FOOD_EATEN":
            return "food_reached"
        if latest_age == 0 and latest_event == "SHELTER_EXIT":
            return "shelter_exited"
        if latest_age == 0 and latest_event == "DEEP_SLEEP_REACHED":
            return "deep_shelter_reached"
        if latest_age == 0 and latest_event == "ACUTE_PREDATOR_THREAT":
            return "acute_predator_threat"
        if latest_age == 0 and latest_event == "RECOVERY_COMPLETED":
            return "recovery_completed"
        if latest_age <= 1 and latest_event == "BLOCKED_MOVE":
            return "no_progress"
        if self.current_option_steps_remaining <= 0:
            return "ttl_expired"
        return None

    def _apply_executive_post_exit_continuation(
        self,
        x: Array,
        termination_reason: str | None,
    ) -> str | None:
        if not self.executive_post_exit_continuation or self.current_option_idx < 0:
            return termination_reason
        signals = self._executive_state_signals(x)
        sheltered = signals["on_shelter"] > 0.5
        acute_threat = signals["acute_threat"]
        hunger = signals["hunger"]
        current_option = OPTION_NAMES[int(self.current_option_idx)]
        if sheltered or acute_threat >= 0.2:
            self.executive_post_exit_steps_remaining = 0
            self.executive_post_exit_corridor_steps_remaining = 0
            return termination_reason
        if (
            termination_reason == "shelter_exited"
            and current_option == "POST_REST_REACTIVATE"
            and hunger >= 0.12
        ):
            seeded_steps = (
                7
                if self.executive_post_exit_corridor_progression
                else (
                    4
                    if self.executive_post_exit_food_commitment
                    else (3 if self.executive_post_exit_smell_progression else 2)
                )
            )
            self.executive_post_exit_steps_remaining = max(
                self.executive_post_exit_steps_remaining,
                seeded_steps,
            )
            if self.executive_post_exit_corridor_progression:
                self.executive_post_exit_corridor_steps_remaining = max(
                    self.executive_post_exit_corridor_steps_remaining,
                    seeded_steps,
                )
            return None
        if (
            self.executive_post_exit_steps_remaining > 0
            and current_option == "POST_REST_REACTIVATE"
            and termination_reason in {"ttl_expired", "no_progress"}
        ):
            if self.executive_post_exit_corridor_progression:
                self.executive_post_exit_steps_remaining = max(
                    self.executive_post_exit_steps_remaining,
                    3,
                )
            elif self.executive_post_exit_food_commitment:
                self.executive_post_exit_steps_remaining = max(
                    self.executive_post_exit_steps_remaining,
                    2,
                )
            return None
        return termination_reason

    def _prime_executive_post_food_return(
        self,
        x: Array,
        termination_reason: str | None,
    ) -> None:
        if not self.executive_post_food_return:
            return
        signals = self._executive_state_signals(x)
        if signals["acute_threat"] >= 0.2:
            self.executive_post_food_return_steps_remaining = 0
            return
        current_option = (
            OPTION_NAMES[int(self.current_option_idx)]
            if self.current_option_idx >= 0
            else None
        )
        if (
            termination_reason == "food_reached"
            and current_option in {"POST_REST_REACTIVATE", "FORAGE", "RETURN_TO_SHELTER"}
        ):
            self.executive_post_food_return_steps_remaining = max(
                self.executive_post_food_return_steps_remaining,
                8,
            )
            if self.executive_post_food_path_return and self.executive_post_food_path_history:
                inverse_actions = {
                    ACTION_TO_INDEX["MOVE_UP"]: ACTION_TO_INDEX["MOVE_DOWN"],
                    ACTION_TO_INDEX["MOVE_DOWN"]: ACTION_TO_INDEX["MOVE_UP"],
                    ACTION_TO_INDEX["MOVE_LEFT"]: ACTION_TO_INDEX["MOVE_RIGHT"],
                    ACTION_TO_INDEX["MOVE_RIGHT"]: ACTION_TO_INDEX["MOVE_LEFT"],
                }
                self.executive_post_food_return_queue = [
                    inverse_actions[action_idx]
                    for action_idx in reversed(self.executive_post_food_path_history)
                    if action_idx in inverse_actions
                ]

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        (
            event_context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        ) = self._attention_context(x, h_prev)
        option_input = self._current_option_vector()
        x_aug = np.concatenate([x, event_context, option_input], axis=0)
        h_new = np.tanh(self.W_xh @ x_aug + self.W_hh @ h_prev + self.b_h)
        base_policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_logits = np.clip(
            np.nan_to_num(
                self.W2_option @ h_new + self.b2_option,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_probs = softmax(option_logits)
        termination_reason = self._termination_reason()
        started_new_option = self.current_option_idx < 0 or termination_reason is not None
        if started_new_option:
            selected_option_idx = int(np.argmax(option_probs))
            self.current_option_idx = selected_option_idx
            self.current_option_age = 0
            self.current_option_steps_remaining = self.option_ttl
        else:
            selected_option_idx = int(self.current_option_idx)
            self.current_option_age += 1
        self.current_option_steps_remaining = max(
            0,
            int(self.current_option_steps_remaining) - 1,
        )
        option_bias = self.option_action_bias[selected_option_idx]
        policy_logits = np.clip(
            np.nan_to_num(
                base_policy_logits + option_bias,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        value = float(
            np.nan_to_num(
                (self.W2_value @ h_new + self.b2_value)[0],
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        self.hidden_state = h_new.copy()
        self.last_option_summary = {
            "selected_option": OPTION_NAMES[selected_option_idx],
            "option_age": int(self.current_option_age),
            "option_termination_reason": (
                "initial_selection"
                if termination_reason is None and option_input.sum() <= 0.0
                else ("active" if termination_reason is None else termination_reason)
            ),
            "option_logits": option_logits.round(6).tolist(),
        }
        if store_cache:
            self.cache = OptionAttentionCache(
                x=x,
                x_aug=x_aug,
                h_prev=h_prev,
                h_new=h_new,
                query_input=query_input,
                query=query,
                slot_raws=slot_raws,
                keys=keys,
                values=values,
                attention_weights=attention_weights,
                valid_event_type_indices=event_type_indices,
                option_probs=option_probs,
                selected_option_idx=selected_option_idx,
            )
        return policy_logits, value, option_logits

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
    ) -> Array:
        if self.cache is None:
            raise RuntimeError(
                "Recurrent option true monolithic network backward called without cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        grad_W2_policy = np.outer(grad_policy_logits, h_new)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_option_action_bias = np.zeros_like(self.option_action_bias)
        grad_option_action_bias[self.cache.selected_option_idx] += grad_policy_logits
        option_target = one_hot(self.cache.selected_option_idx, self.option_dim)
        option_advantage = -grad_value
        grad_option_logits = 0.2 * option_advantage * (
            self.cache.option_probs - option_target
        )
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        dh = (
            self.W2_policy.T @ grad_policy_logits
            + self.W2_value.T[:, 0] * grad_value
            + self.W2_option.T @ grad_option_logits
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[
            self.input_dim : self.input_dim + self.event_context_dim
        ].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz

        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)
        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value

                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    : self.event_embedding_dim
                ]

            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query

        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
        self.option_action_bias -= lr * grad_option_action_bias
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        return grad_x

    def value_only(self, x: Array) -> float:
        runtime_state = self.get_runtime_state()
        try:
            _, value, _ = self.forward(x, store_cache=False)
            return value
        finally:
            self.set_runtime_state(runtime_state)

    def state_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "recurrent": True,
            "event_attention": True,
            "event_buffer_size": self.event_buffer_size,
            "event_embedding_dim": self.event_embedding_dim,
            "event_context_dim": self.event_context_dim,
            "event_feature_dim": self.event_feature_dim,
            "option_head": True,
            "option_ttl": self.option_ttl,
            "option_dim": self.option_dim,
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "b_h": self.b_h.copy(),
            "W2_policy": self.W2_policy.copy(),
            "b2_policy": self.b2_policy.copy(),
            "W2_value": self.W2_value.copy(),
            "b2_value": self.b2_value.copy(),
            "W2_option": self.W2_option.copy(),
            "b2_option": self.b2_option.copy(),
            "option_action_bias": self.option_action_bias.copy(),
            "W_query": self.W_query.copy(),
            "b_query": self.b_query.copy(),
            "W_key": self.W_key.copy(),
            "b_key": self.b_key.copy(),
            "W_value": self.W_value.copy(),
            "b_value": self.b_value.copy(),
            "event_type_embeddings": self.event_type_embeddings.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "recurrent",
                "event_attention",
                "event_buffer_size",
                "event_embedding_dim",
                "event_context_dim",
                "event_feature_dim",
                "option_head",
                "option_ttl",
                "option_dim",
                "W_xh",
                "W_hh",
                "b_h",
                "W2_policy",
                "b2_policy",
                "W2_value",
                "b2_value",
                "W2_option",
                "b2_option",
                "option_action_bias",
                "W_query",
                "b_query",
                "W_key",
                "b_key",
                "W_value",
                "b_value",
                "event_type_embeddings",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
                "option_head": True,
                "option_ttl": self.option_ttl,
                "option_dim": self.option_dim,
            },
            name=self.name,
        )
        recurrent_input_dim = self.input_dim + self.event_context_dim + self.option_dim
        event_raw_dim = self.event_embedding_dim + self.event_feature_dim + 1
        self.W_xh = _coerce_state_array(
            state,
            "W_xh",
            (self.hidden_dim, recurrent_input_dim),
            name=self.name,
        )
        self.W_hh = _coerce_state_array(
            state,
            "W_hh",
            (self.hidden_dim, self.hidden_dim),
            name=self.name,
        )
        self.b_h = _coerce_state_array(state, "b_h", (self.hidden_dim,), name=self.name)
        self.W2_policy = _coerce_state_array(
            state,
            "W2_policy",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_policy = _coerce_state_array(state, "b2_policy", (self.output_dim,), name=self.name)
        self.W2_value = _coerce_state_array(
            state,
            "W2_value",
            (1, self.hidden_dim),
            name=self.name,
        )
        self.b2_value = _coerce_state_array(state, "b2_value", (1,), name=self.name)
        self.W2_option = _coerce_state_array(
            state,
            "W2_option",
            (self.option_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_option = _coerce_state_array(
            state,
            "b2_option",
            (self.option_dim,),
            name=self.name,
        )
        self.option_action_bias = _coerce_state_array(
            state,
            "option_action_bias",
            (self.option_dim, self.output_dim),
            name=self.name,
        )
        self.W_query = _coerce_state_array(
            state,
            "W_query",
            (self.event_context_dim, self.input_dim + self.hidden_dim),
            name=self.name,
        )
        self.b_query = _coerce_state_array(
            state,
            "b_query",
            (self.event_context_dim,),
            name=self.name,
        )
        self.W_key = _coerce_state_array(
            state,
            "W_key",
            (self.event_context_dim, event_raw_dim),
            name=self.name,
        )
        self.b_key = _coerce_state_array(
            state,
            "b_key",
            (self.event_context_dim,),
            name=self.name,
        )
        self.W_value = _coerce_state_array(
            state,
            "W_value",
            (self.event_context_dim, event_raw_dim),
            name=self.name,
        )
        self.b_value = _coerce_state_array(
            state,
            "b_value",
            (self.event_context_dim,),
            name=self.name,
        )
        self.event_type_embeddings = _coerce_state_array(
            state,
            "event_type_embeddings",
            (len(EVENT_TYPE_NAMES), self.event_embedding_dim),
            name=self.name,
        )
        self.reset_hidden_state()
        self.reset_event_memory()

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_option,
            self.b2_option,
            self.option_action_bias,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            self.W_xh.size
            + self.W_hh.size
            + self.b_h.size
            + self.W2_policy.size
            + self.b2_policy.size
            + self.W2_value.size
            + self.b2_value.size
            + self.W2_option.size
            + self.b2_option.size
            + self.option_action_bias.size
            + self.W_query.size
            + self.b_query.size
            + self.W_key.size
            + self.b_key.size
            + self.W_value.size
            + self.b_value.size
            + self.event_type_embeddings.size
        )


class RecurrentOptionAffordanceTrueMonolithicNetwork(
    RecurrentOptionTrueMonolithicNetwork
):
    """Option controller with auxiliary local-affordance heads."""

    affordance_role_dim = len(AFFORDANCE_SHELTER_ROLE_NAMES)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        name: str = "true_monolithic_policy",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            event_buffer_size=event_buffer_size,
            option_ttl=option_ttl,
            name=name,
        )
        self.W2_affordance_blocked = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.output_dim, self.hidden_dim),
        )
        self.b2_affordance_blocked = np.zeros(self.output_dim, dtype=float)
        self.W2_affordance_role = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.output_dim * self.affordance_role_dim, self.hidden_dim),
        )
        self.b2_affordance_role = np.zeros(
            self.output_dim * self.affordance_role_dim,
            dtype=float,
        )
        self.last_affordance_summary: dict[str, object] = {
            "blocked_logits": [],
            "role_logits": [],
        }

    def reset_hidden_state(self) -> None:
        super().reset_hidden_state()
        self.last_affordance_summary = {
            "blocked_logits": [],
            "role_logits": [],
        }

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array]:
        policy_logits, value, option_logits = super().forward(x, store_cache=store_cache)
        if self.cache is None:
            h_new = self.hidden_state
        else:
            h_new = self.cache.h_new
        blocked_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_blocked @ h_new + self.b2_affordance_blocked,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        role_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_role @ h_new + self.b2_affordance_role,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        self.last_affordance_summary = {
            "blocked_logits": blocked_logits.round(6).tolist(),
            "role_logits": role_logits.round(6).tolist(),
        }
        return policy_logits, value, option_logits

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
        grad_affordance_blocked_logits: Array | None = None,
        grad_affordance_role_logits: Array | None = None,
    ) -> Array:
        if self.cache is None:
            raise RuntimeError(
                "Recurrent option affordance true monolithic network backward called without cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        if grad_affordance_blocked_logits is None:
            grad_affordance_blocked_logits = np.zeros(self.output_dim, dtype=float)
        else:
            grad_affordance_blocked_logits = _clip_grad_logits(
                grad_affordance_blocked_logits,
                grad_clip,
            )
        affordance_role_output_dim = self.output_dim * self.affordance_role_dim
        if grad_affordance_role_logits is None:
            grad_affordance_role_logits = np.zeros(
                affordance_role_output_dim,
                dtype=float,
            )
        else:
            grad_affordance_role_logits = _clip_grad_logits(
                np.asarray(grad_affordance_role_logits, dtype=float),
                grad_clip,
            )
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        grad_W2_policy = np.outer(grad_policy_logits, h_new)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_option_action_bias = np.zeros_like(self.option_action_bias)
        grad_option_action_bias[self.cache.selected_option_idx] += grad_policy_logits
        option_target = one_hot(self.cache.selected_option_idx, self.option_dim)
        option_advantage = -grad_value
        grad_option_logits = 0.2 * option_advantage * (
            self.cache.option_probs - option_target
        )
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        grad_W2_affordance_blocked = np.outer(
            grad_affordance_blocked_logits,
            h_new,
        )
        grad_b2_affordance_blocked = grad_affordance_blocked_logits
        grad_W2_affordance_role = np.outer(
            grad_affordance_role_logits,
            h_new,
        )
        grad_b2_affordance_role = grad_affordance_role_logits
        dh = (
            self.W2_policy.T @ grad_policy_logits
            + self.W2_value.T[:, 0] * grad_value
            + self.W2_option.T @ grad_option_logits
            + self.W2_affordance_blocked.T @ grad_affordance_blocked_logits
            + self.W2_affordance_role.T @ grad_affordance_role_logits
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[
            self.input_dim : self.input_dim + self.event_context_dim
        ].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz

        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)
        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value

                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    : self.event_embedding_dim
                ]

            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query

        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
        self.option_action_bias -= lr * grad_option_action_bias
        self.W2_affordance_blocked -= lr * grad_W2_affordance_blocked
        self.b2_affordance_blocked -= lr * grad_b2_affordance_blocked
        self.W2_affordance_role -= lr * grad_W2_affordance_role
        self.b2_affordance_role -= lr * grad_b2_affordance_role
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        return grad_x

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["affordance_head"] = True
        state["affordance_role_dim"] = self.affordance_role_dim
        state["W2_affordance_blocked"] = self.W2_affordance_blocked.copy()
        state["b2_affordance_blocked"] = self.b2_affordance_blocked.copy()
        state["W2_affordance_role"] = self.W2_affordance_role.copy()
        state["b2_affordance_role"] = self.b2_affordance_role.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "recurrent",
                "event_attention",
                "event_buffer_size",
                "event_embedding_dim",
                "event_context_dim",
                "event_feature_dim",
                "option_head",
                "option_ttl",
                "option_dim",
                "affordance_head",
                "affordance_role_dim",
                "W_xh",
                "W_hh",
                "b_h",
                "W2_policy",
                "b2_policy",
                "W2_value",
                "b2_value",
                "W2_option",
                "b2_option",
                "option_action_bias",
                "W2_affordance_blocked",
                "b2_affordance_blocked",
                "W2_affordance_role",
                "b2_affordance_role",
                "W_query",
                "b_query",
                "W_key",
                "b_key",
                "W_value",
                "b_value",
                "event_type_embeddings",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
                "option_head": True,
                "option_ttl": self.option_ttl,
                "option_dim": self.option_dim,
                "affordance_head": True,
                "affordance_role_dim": self.affordance_role_dim,
            },
            name=self.name,
        )
        super().load_state_dict({
            key: value
            for key, value in state.items()
            if key
            not in {
                "affordance_head",
                "affordance_role_dim",
                "W2_affordance_blocked",
                "b2_affordance_blocked",
                "W2_affordance_role",
                "b2_affordance_role",
            }
        })
        self.W2_affordance_blocked = _coerce_state_array(
            state,
            "W2_affordance_blocked",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_affordance_blocked = _coerce_state_array(
            state,
            "b2_affordance_blocked",
            (self.output_dim,),
            name=self.name,
        )
        self.W2_affordance_role = _coerce_state_array(
            state,
            "W2_affordance_role",
            (self.output_dim * self.affordance_role_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_affordance_role = _coerce_state_array(
            state,
            "b2_affordance_role",
            (self.output_dim * self.affordance_role_dim,),
            name=self.name,
        )
        self.last_affordance_summary = {
            "blocked_logits": [],
            "role_logits": [],
        }

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_option,
            self.b2_option,
            self.option_action_bias,
            self.W2_affordance_blocked,
            self.b2_affordance_blocked,
            self.W2_affordance_role,
            self.b2_affordance_role,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            super().count_parameters()
            + self.W2_affordance_blocked.size
            + self.b2_affordance_blocked.size
            + self.W2_affordance_role.size
            + self.b2_affordance_role.size
        )


class RecurrentOptionAffordanceFeedbackTrueMonolithicNetwork(
    RecurrentOptionAffordanceTrueMonolithicNetwork
):
    """Option controller that reuses predicted affordances as internal feedback."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        name: str = "true_monolithic_policy",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            event_buffer_size=event_buffer_size,
            option_ttl=option_ttl,
            name=name,
        )
        self.affordance_feature_dim = int(
            self.output_dim + self.output_dim * self.affordance_role_dim
        )
        self.W_affordance_feedback = rng.normal(
            0.0,
            _weight_scale(self.affordance_feature_dim),
            size=(self.hidden_dim, self.affordance_feature_dim),
        )
        self.b_affordance_feedback = np.zeros(self.hidden_dim, dtype=float)
        self.W2_policy_feedback = rng.normal(
            0.0,
            _weight_scale(self.hidden_dim),
            size=(self.output_dim, self.hidden_dim),
        )
        self.b2_policy_feedback = np.zeros(self.output_dim, dtype=float)
        self.W2_option_feedback = rng.normal(
            0.0,
            _weight_scale(self.hidden_dim),
            size=(self.option_dim, self.hidden_dim),
        )
        self.b2_option_feedback = np.zeros(self.option_dim, dtype=float)

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        (
            event_context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        ) = self._attention_context(x, h_prev)
        option_input = self._current_option_vector()
        x_aug = np.concatenate([x, event_context, option_input], axis=0)
        h_new = np.tanh(self.W_xh @ x_aug + self.W_hh @ h_prev + self.b_h)
        blocked_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_blocked @ h_new + self.b2_affordance_blocked,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        role_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_role @ h_new + self.b2_affordance_role,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        blocked_probs = _sigmoid(blocked_logits)
        role_logits_matrix = role_logits.reshape(self.output_dim, self.affordance_role_dim)
        role_probs = np.vstack(
            [softmax(role_logits_matrix[action_idx]) for action_idx in range(self.output_dim)]
        )
        affordance_features = np.concatenate(
            [blocked_probs, role_probs.reshape(-1)],
            axis=0,
        )
        affordance_feedback = np.tanh(
            self.W_affordance_feedback @ affordance_features
            + self.b_affordance_feedback
        )
        base_policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_logits = np.clip(
            np.nan_to_num(
                self.W2_option @ h_new
                + self.b2_option
                + self.W2_option_feedback @ affordance_feedback
                + self.b2_option_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_probs = softmax(option_logits)
        termination_reason = self._termination_reason()
        if self.current_option_idx < 0 or termination_reason is not None:
            selected_option_idx = int(np.argmax(option_probs))
            self.current_option_idx = selected_option_idx
            self.current_option_age = 0
            self.current_option_steps_remaining = self.option_ttl
        else:
            selected_option_idx = int(self.current_option_idx)
            self.current_option_age += 1
        self.current_option_steps_remaining = max(
            0,
            int(self.current_option_steps_remaining) - 1,
        )
        option_bias = self.option_action_bias[selected_option_idx]
        policy_logits = np.clip(
            np.nan_to_num(
                base_policy_logits
                + self.W2_policy_feedback @ affordance_feedback
                + self.b2_policy_feedback
                + option_bias,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        value = float(
            np.nan_to_num(
                (self.W2_value @ h_new + self.b2_value)[0],
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        self.hidden_state = h_new.copy()
        self.last_option_summary = {
            "selected_option": OPTION_NAMES[selected_option_idx],
            "option_age": int(self.current_option_age),
            "option_termination_reason": (
                "initial_selection"
                if termination_reason is None and option_input.sum() <= 0.0
                else ("active" if termination_reason is None else termination_reason)
            ),
            "option_logits": option_logits.round(6).tolist(),
        }
        self.last_affordance_summary = {
            "blocked_logits": blocked_logits.round(6).tolist(),
            "role_logits": role_logits.round(6).tolist(),
        }
        if store_cache:
            self.cache = OptionAffordanceFeedbackCache(
                x=x,
                x_aug=x_aug,
                h_prev=h_prev,
                h_new=h_new,
                query_input=query_input,
                query=query,
                slot_raws=slot_raws,
                keys=keys,
                values=values,
                attention_weights=attention_weights,
                valid_event_type_indices=event_type_indices,
                option_probs=option_probs,
                selected_option_idx=selected_option_idx,
                blocked_probs=blocked_probs,
                role_probs=role_probs,
                affordance_features=affordance_features,
                affordance_feedback=affordance_feedback,
            )
        return policy_logits, value, option_logits

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
        grad_affordance_blocked_logits: Array | None = None,
        grad_affordance_role_logits: Array | None = None,
    ) -> Array:
        if not isinstance(self.cache, OptionAffordanceFeedbackCache):
            raise RuntimeError(
                "Recurrent option affordance feedback network backward called without feedback cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        if grad_affordance_blocked_logits is None:
            grad_affordance_blocked_logits = np.zeros(self.output_dim, dtype=float)
        else:
            grad_affordance_blocked_logits = _clip_grad_logits(
                grad_affordance_blocked_logits,
                grad_clip,
            )
        affordance_role_output_dim = self.output_dim * self.affordance_role_dim
        if grad_affordance_role_logits is None:
            grad_affordance_role_logits = np.zeros(
                affordance_role_output_dim,
                dtype=float,
            )
        else:
            grad_affordance_role_logits = _clip_grad_logits(
                np.asarray(grad_affordance_role_logits, dtype=float),
                grad_clip,
            )
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        affordance_feedback = self.cache.affordance_feedback
        grad_W2_policy = np.outer(grad_policy_logits, h_new)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_option_action_bias = np.zeros_like(self.option_action_bias)
        grad_option_action_bias[self.cache.selected_option_idx] += grad_policy_logits
        option_target = one_hot(self.cache.selected_option_idx, self.option_dim)
        option_advantage = -grad_value
        grad_option_logits = 0.2 * option_advantage * (
            self.cache.option_probs - option_target
        )
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        grad_W2_policy_feedback = np.outer(grad_policy_logits, affordance_feedback)
        grad_b2_policy_feedback = grad_policy_logits
        grad_W2_option_feedback = np.outer(grad_option_logits, affordance_feedback)
        grad_b2_option_feedback = grad_option_logits
        grad_feedback = (
            self.W2_policy_feedback.T @ grad_policy_logits
            + self.W2_option_feedback.T @ grad_option_logits
        )
        dz_feedback = grad_feedback * (1.0 - affordance_feedback**2)
        grad_W_affordance_feedback = np.outer(
            dz_feedback,
            self.cache.affordance_features,
        )
        grad_b_affordance_feedback = dz_feedback
        grad_affordance_features = self.W_affordance_feedback.T @ dz_feedback
        grad_blocked_probs = grad_affordance_features[: self.output_dim]
        grad_role_probs = grad_affordance_features[self.output_dim :].reshape(
            self.output_dim,
            self.affordance_role_dim,
        )
        grad_affordance_blocked_logits = np.asarray(
            grad_affordance_blocked_logits,
            dtype=float,
        ) + grad_blocked_probs * (
            self.cache.blocked_probs * (1.0 - self.cache.blocked_probs)
        )
        grad_affordance_role_matrix = np.asarray(
            grad_affordance_role_logits,
            dtype=float,
        ).reshape(self.output_dim, self.affordance_role_dim)
        feedback_role_grad_matrix = np.zeros_like(grad_affordance_role_matrix)
        for action_idx in range(self.output_dim):
            role_probs = self.cache.role_probs[action_idx]
            role_prob_grad = grad_role_probs[action_idx]
            feedback_role_grad_matrix[action_idx] = role_probs * (
                role_prob_grad - float(np.dot(role_prob_grad, role_probs))
            )
        grad_affordance_role_logits = (
            grad_affordance_role_matrix + feedback_role_grad_matrix
        ).reshape(-1)
        grad_W2_affordance_blocked = np.outer(
            grad_affordance_blocked_logits,
            h_new,
        )
        grad_b2_affordance_blocked = grad_affordance_blocked_logits
        grad_W2_affordance_role = np.outer(
            grad_affordance_role_logits,
            h_new,
        )
        grad_b2_affordance_role = grad_affordance_role_logits
        dh = (
            self.W2_policy.T @ grad_policy_logits
            + self.W2_value.T[:, 0] * grad_value
            + self.W2_option.T @ grad_option_logits
            + self.W2_affordance_blocked.T @ grad_affordance_blocked_logits
            + self.W2_affordance_role.T @ grad_affordance_role_logits
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[
            self.input_dim : self.input_dim + self.event_context_dim
        ].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz
        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)
        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value
                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    : self.event_embedding_dim
                ]
            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query
        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
        self.option_action_bias -= lr * grad_option_action_bias
        self.W2_policy_feedback -= lr * grad_W2_policy_feedback
        self.b2_policy_feedback -= lr * grad_b2_policy_feedback
        self.W2_option_feedback -= lr * grad_W2_option_feedback
        self.b2_option_feedback -= lr * grad_b2_option_feedback
        self.W_affordance_feedback -= lr * grad_W_affordance_feedback
        self.b_affordance_feedback -= lr * grad_b_affordance_feedback
        self.W2_affordance_blocked -= lr * grad_W2_affordance_blocked
        self.b2_affordance_blocked -= lr * grad_b2_affordance_blocked
        self.W2_affordance_role -= lr * grad_W2_affordance_role
        self.b2_affordance_role -= lr * grad_b2_affordance_role
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        return grad_x

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["affordance_feedback"] = True
        state["affordance_feature_dim"] = self.affordance_feature_dim
        state["W_affordance_feedback"] = self.W_affordance_feedback.copy()
        state["b_affordance_feedback"] = self.b_affordance_feedback.copy()
        state["W2_policy_feedback"] = self.W2_policy_feedback.copy()
        state["b2_policy_feedback"] = self.b2_policy_feedback.copy()
        state["W2_option_feedback"] = self.W2_option_feedback.copy()
        state["b2_option_feedback"] = self.b2_option_feedback.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "output_dim",
                "recurrent",
                "event_attention",
                "event_buffer_size",
                "event_embedding_dim",
                "event_context_dim",
                "event_feature_dim",
                "option_head",
                "option_ttl",
                "option_dim",
                "affordance_head",
                "affordance_role_dim",
                "affordance_feedback",
                "affordance_feature_dim",
                "W_xh",
                "W_hh",
                "b_h",
                "W2_policy",
                "b2_policy",
                "W2_value",
                "b2_value",
                "W2_option",
                "b2_option",
                "option_action_bias",
                "W2_affordance_blocked",
                "b2_affordance_blocked",
                "W2_affordance_role",
                "b2_affordance_role",
                "W_affordance_feedback",
                "b_affordance_feedback",
                "W2_policy_feedback",
                "b2_policy_feedback",
                "W2_option_feedback",
                "b2_option_feedback",
                "W_query",
                "b_query",
                "W_key",
                "b_key",
                "W_value",
                "b_value",
                "event_type_embeddings",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
                "option_head": True,
                "option_ttl": self.option_ttl,
                "option_dim": self.option_dim,
                "affordance_head": True,
                "affordance_role_dim": self.affordance_role_dim,
                "affordance_feedback": True,
                "affordance_feature_dim": self.affordance_feature_dim,
            },
            name=self.name,
        )
        super().load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "affordance_feedback",
                    "affordance_feature_dim",
                    "W_affordance_feedback",
                    "b_affordance_feedback",
                    "W2_policy_feedback",
                    "b2_policy_feedback",
                    "W2_option_feedback",
                    "b2_option_feedback",
                }
            }
        )
        self.W_affordance_feedback = _coerce_state_array(
            state,
            "W_affordance_feedback",
            (self.hidden_dim, self.affordance_feature_dim),
            name=self.name,
        )
        self.b_affordance_feedback = _coerce_state_array(
            state,
            "b_affordance_feedback",
            (self.hidden_dim,),
            name=self.name,
        )
        self.W2_policy_feedback = _coerce_state_array(
            state,
            "W2_policy_feedback",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_policy_feedback = _coerce_state_array(
            state,
            "b2_policy_feedback",
            (self.output_dim,),
            name=self.name,
        )
        self.W2_option_feedback = _coerce_state_array(
            state,
            "W2_option_feedback",
            (self.option_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_option_feedback = _coerce_state_array(
            state,
            "b2_option_feedback",
            (self.option_dim,),
            name=self.name,
        )

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_option,
            self.b2_option,
            self.option_action_bias,
            self.W2_affordance_blocked,
            self.b2_affordance_blocked,
            self.W2_affordance_role,
            self.b2_affordance_role,
            self.W_affordance_feedback,
            self.b_affordance_feedback,
            self.W2_policy_feedback,
            self.b2_policy_feedback,
            self.W2_option_feedback,
            self.b2_option_feedback,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            super().count_parameters()
            + self.W_affordance_feedback.size
            + self.b_affordance_feedback.size
            + self.W2_policy_feedback.size
            + self.b2_policy_feedback.size
            + self.W2_option_feedback.size
            + self.b2_option_feedback.size
        )


class RecurrentOptionAffordanceGeometryFeedbackTrueMonolithicNetwork(
    RecurrentOptionAffordanceFeedbackTrueMonolithicNetwork
):
    """Affordance-feedback controller with extra shelter-geometry heads."""

    geometry_dim = len(AFFORDANCE_GEOMETRY_TARGET_NAMES)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        name: str = "true_monolithic_policy",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            event_buffer_size=event_buffer_size,
            option_ttl=option_ttl,
            name=name,
        )
        self.geometry_feature_dim = int(self.output_dim * self.geometry_dim)
        self.affordance_feature_dim = int(
            self.output_dim
            + self.output_dim * self.affordance_role_dim
            + self.geometry_feature_dim
        )
        self.W2_geometry = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.geometry_feature_dim, self.hidden_dim),
        )
        self.b2_geometry = np.zeros(self.geometry_feature_dim, dtype=float)
        self.W_affordance_feedback = rng.normal(
            0.0,
            _weight_scale(self.affordance_feature_dim),
            size=(self.hidden_dim, self.affordance_feature_dim),
        )
        self.b_affordance_feedback = np.zeros(self.hidden_dim, dtype=float)
        self.last_affordance_summary["geometry_logits"] = []

    def reset_hidden_state(self) -> None:
        super().reset_hidden_state()
        self.last_affordance_summary["geometry_logits"] = []

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        (
            event_context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        ) = self._attention_context(x, h_prev)
        option_input = self._current_option_vector()
        x_aug = np.concatenate([x, event_context, option_input], axis=0)
        h_new = np.tanh(self.W_xh @ x_aug + self.W_hh @ h_prev + self.b_h)
        blocked_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_blocked @ h_new + self.b2_affordance_blocked,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        role_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_role @ h_new + self.b2_affordance_role,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        geometry_logits = np.clip(
            np.nan_to_num(
                self.W2_geometry @ h_new + self.b2_geometry,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        blocked_probs = _sigmoid(blocked_logits)
        role_logits_matrix = role_logits.reshape(self.output_dim, self.affordance_role_dim)
        role_probs = np.vstack(
            [softmax(role_logits_matrix[action_idx]) for action_idx in range(self.output_dim)]
        )
        geometry_probs = _sigmoid(geometry_logits)
        affordance_features = np.concatenate(
            [blocked_probs, role_probs.reshape(-1), geometry_probs],
            axis=0,
        )
        affordance_feedback = np.tanh(
            self.W_affordance_feedback @ affordance_features
            + self.b_affordance_feedback
        )
        base_policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_logits = np.clip(
            np.nan_to_num(
                self.W2_option @ h_new
                + self.b2_option
                + self.W2_option_feedback @ affordance_feedback
                + self.b2_option_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_probs = softmax(option_logits)
        termination_reason = self._termination_reason()
        if self.current_option_idx < 0 or termination_reason is not None:
            selected_option_idx = int(np.argmax(option_probs))
            self.current_option_idx = selected_option_idx
            self.current_option_age = 0
            self.current_option_steps_remaining = self.option_ttl
        else:
            selected_option_idx = int(self.current_option_idx)
            self.current_option_age += 1
        self.current_option_steps_remaining = max(
            0,
            int(self.current_option_steps_remaining) - 1,
        )
        option_bias = self.option_action_bias[selected_option_idx]
        policy_logits = np.clip(
            np.nan_to_num(
                base_policy_logits
                + self.W2_policy_feedback @ affordance_feedback
                + self.b2_policy_feedback
                + option_bias,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        value = float(
            np.nan_to_num(
                (self.W2_value @ h_new + self.b2_value)[0],
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        self.hidden_state = h_new.copy()
        self.last_option_summary = {
            "selected_option": OPTION_NAMES[selected_option_idx],
            "option_age": int(self.current_option_age),
            "option_termination_reason": (
                "initial_selection"
                if termination_reason is None and option_input.sum() <= 0.0
                else ("active" if termination_reason is None else termination_reason)
            ),
            "option_logits": option_logits.round(6).tolist(),
        }
        self.last_affordance_summary = {
            "blocked_logits": blocked_logits.round(6).tolist(),
            "role_logits": role_logits.round(6).tolist(),
            "geometry_logits": geometry_logits.round(6).tolist(),
        }
        if store_cache:
            self.cache = OptionAffordanceGeometryFeedbackCache(
                x=x,
                x_aug=x_aug,
                h_prev=h_prev,
                h_new=h_new,
                query_input=query_input,
                query=query,
                slot_raws=slot_raws,
                keys=keys,
                values=values,
                attention_weights=attention_weights,
                valid_event_type_indices=event_type_indices,
                option_probs=option_probs,
                selected_option_idx=selected_option_idx,
                blocked_probs=blocked_probs,
                role_probs=role_probs,
                affordance_features=affordance_features,
                affordance_feedback=affordance_feedback,
                geometry_probs=geometry_probs,
            )
        return policy_logits, value, option_logits

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
        grad_affordance_blocked_logits: Array | None = None,
        grad_affordance_role_logits: Array | None = None,
        grad_geometry_logits: Array | None = None,
    ) -> Array:
        if not isinstance(self.cache, OptionAffordanceGeometryFeedbackCache):
            raise RuntimeError(
                "Recurrent option affordance geometry feedback network backward called without geometry cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        if grad_affordance_blocked_logits is None:
            grad_affordance_blocked_logits = np.zeros(self.output_dim, dtype=float)
        else:
            grad_affordance_blocked_logits = _clip_grad_logits(
                grad_affordance_blocked_logits,
                grad_clip,
            )
        affordance_role_output_dim = self.output_dim * self.affordance_role_dim
        if grad_affordance_role_logits is None:
            grad_affordance_role_logits = np.zeros(
                affordance_role_output_dim,
                dtype=float,
            )
        else:
            grad_affordance_role_logits = _clip_grad_logits(
                np.asarray(grad_affordance_role_logits, dtype=float),
                grad_clip,
            )
        if grad_geometry_logits is None:
            grad_geometry_logits = np.zeros(self.geometry_feature_dim, dtype=float)
        else:
            grad_geometry_logits = _clip_grad_logits(
                np.asarray(grad_geometry_logits, dtype=float),
                grad_clip,
            )
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        affordance_feedback = self.cache.affordance_feedback
        grad_W2_policy = np.outer(grad_policy_logits, h_new)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_option_action_bias = np.zeros_like(self.option_action_bias)
        grad_option_action_bias[self.cache.selected_option_idx] += grad_policy_logits
        option_target = one_hot(self.cache.selected_option_idx, self.option_dim)
        option_advantage = -grad_value
        grad_option_logits = 0.2 * option_advantage * (
            self.cache.option_probs - option_target
        )
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        grad_W2_policy_feedback = np.outer(grad_policy_logits, affordance_feedback)
        grad_b2_policy_feedback = grad_policy_logits
        grad_W2_option_feedback = np.outer(grad_option_logits, affordance_feedback)
        grad_b2_option_feedback = grad_option_logits
        grad_feedback = (
            self.W2_policy_feedback.T @ grad_policy_logits
            + self.W2_option_feedback.T @ grad_option_logits
        )
        dz_feedback = grad_feedback * (1.0 - affordance_feedback**2)
        grad_W_affordance_feedback = np.outer(
            dz_feedback,
            self.cache.affordance_features,
        )
        grad_b_affordance_feedback = dz_feedback
        grad_affordance_features = self.W_affordance_feedback.T @ dz_feedback
        grad_blocked_probs = grad_affordance_features[: self.output_dim]
        role_prob_end = self.output_dim + affordance_role_output_dim
        grad_role_probs = grad_affordance_features[
            self.output_dim : role_prob_end
        ].reshape(self.output_dim, self.affordance_role_dim)
        grad_geometry_probs = grad_affordance_features[role_prob_end:]
        grad_affordance_blocked_logits = np.asarray(
            grad_affordance_blocked_logits,
            dtype=float,
        ) + grad_blocked_probs * (
            self.cache.blocked_probs * (1.0 - self.cache.blocked_probs)
        )
        grad_affordance_role_matrix = np.asarray(
            grad_affordance_role_logits,
            dtype=float,
        ).reshape(self.output_dim, self.affordance_role_dim)
        feedback_role_grad_matrix = np.zeros_like(grad_affordance_role_matrix)
        for action_idx in range(self.output_dim):
            role_probs = self.cache.role_probs[action_idx]
            role_prob_grad = grad_role_probs[action_idx]
            feedback_role_grad_matrix[action_idx] = role_probs * (
                role_prob_grad - float(np.dot(role_prob_grad, role_probs))
            )
        grad_affordance_role_logits = (
            grad_affordance_role_matrix + feedback_role_grad_matrix
        ).reshape(-1)
        grad_geometry_logits = np.asarray(grad_geometry_logits, dtype=float) + (
            grad_geometry_probs * self.cache.geometry_probs * (1.0 - self.cache.geometry_probs)
        )
        grad_W2_affordance_blocked = np.outer(
            grad_affordance_blocked_logits,
            h_new,
        )
        grad_b2_affordance_blocked = grad_affordance_blocked_logits
        grad_W2_affordance_role = np.outer(
            grad_affordance_role_logits,
            h_new,
        )
        grad_b2_affordance_role = grad_affordance_role_logits
        grad_W2_geometry = np.outer(grad_geometry_logits, h_new)
        grad_b2_geometry = grad_geometry_logits
        dh = (
            self.W2_policy.T @ grad_policy_logits
            + self.W2_value.T[:, 0] * grad_value
            + self.W2_option.T @ grad_option_logits
            + self.W2_affordance_blocked.T @ grad_affordance_blocked_logits
            + self.W2_affordance_role.T @ grad_affordance_role_logits
            + self.W2_geometry.T @ grad_geometry_logits
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[
            self.input_dim : self.input_dim + self.event_context_dim
        ].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz
        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)
        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value
                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    : self.event_embedding_dim
                ]
            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query
        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
        self.option_action_bias -= lr * grad_option_action_bias
        self.W2_policy_feedback -= lr * grad_W2_policy_feedback
        self.b2_policy_feedback -= lr * grad_b2_policy_feedback
        self.W2_option_feedback -= lr * grad_W2_option_feedback
        self.b2_option_feedback -= lr * grad_b2_option_feedback
        self.W_affordance_feedback -= lr * grad_W_affordance_feedback
        self.b_affordance_feedback -= lr * grad_b_affordance_feedback
        self.W2_affordance_blocked -= lr * grad_W2_affordance_blocked
        self.b2_affordance_blocked -= lr * grad_b2_affordance_blocked
        self.W2_affordance_role -= lr * grad_W2_affordance_role
        self.b2_affordance_role -= lr * grad_b2_affordance_role
        self.W2_geometry -= lr * grad_W2_geometry
        self.b2_geometry -= lr * grad_b2_geometry
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        return grad_x

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["geometry_head"] = True
        state["geometry_dim"] = self.geometry_dim
        state["geometry_feature_dim"] = self.geometry_feature_dim
        state["W2_geometry"] = self.W2_geometry.copy()
        state["b2_geometry"] = self.b2_geometry.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys=set(super().state_dict().keys()) | {
                "geometry_head",
                "geometry_dim",
                "geometry_feature_dim",
                "W2_geometry",
                "b2_geometry",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
                "option_head": True,
                "option_ttl": self.option_ttl,
                "option_dim": self.option_dim,
                "affordance_head": True,
                "affordance_role_dim": self.affordance_role_dim,
                "affordance_feedback": True,
                "geometry_head": True,
                "geometry_dim": self.geometry_dim,
                "geometry_feature_dim": self.geometry_feature_dim,
            },
            name=self.name,
        )
        super().load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "geometry_head",
                    "geometry_dim",
                    "geometry_feature_dim",
                    "W2_geometry",
                    "b2_geometry",
                }
            }
        )
        self.W2_geometry = _coerce_state_array(
            state,
            "W2_geometry",
            (self.geometry_feature_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_geometry = _coerce_state_array(
            state,
            "b2_geometry",
            (self.geometry_feature_dim,),
            name=self.name,
        )
        self.last_affordance_summary["geometry_logits"] = []

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_option,
            self.b2_option,
            self.option_action_bias,
            self.W2_affordance_blocked,
            self.b2_affordance_blocked,
            self.W2_affordance_role,
            self.b2_affordance_role,
            self.W2_geometry,
            self.b2_geometry,
            self.W_affordance_feedback,
            self.b_affordance_feedback,
            self.W2_policy_feedback,
            self.b2_policy_feedback,
            self.W2_option_feedback,
            self.b2_option_feedback,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            super().count_parameters()
            + self.W2_geometry.size
            + self.b2_geometry.size
        )


class RecurrentOptionAffordanceTopologyFeedbackTrueMonolithicNetwork(
    RecurrentOptionAffordanceGeometryFeedbackTrueMonolithicNetwork
):
    """Geometry-feedback controller with explicit shelter-column targets."""

    shelter_column_dim = len(AFFORDANCE_SHELTER_COLUMN_NAMES)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        name: str = "true_monolithic_policy",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            event_buffer_size=event_buffer_size,
            option_ttl=option_ttl,
            name=name,
        )
        self.shelter_column_feature_dim = int(
            self.output_dim * self.shelter_column_dim
        )
        self.affordance_feature_dim = int(
            self.output_dim
            + self.output_dim * self.affordance_role_dim
            + self.geometry_feature_dim
            + self.shelter_column_feature_dim
        )
        self.W2_shelter_column = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.shelter_column_feature_dim, self.hidden_dim),
        )
        self.b2_shelter_column = np.zeros(self.shelter_column_feature_dim, dtype=float)
        self.W_affordance_feedback = rng.normal(
            0.0,
            _weight_scale(self.affordance_feature_dim),
            size=(self.hidden_dim, self.affordance_feature_dim),
        )
        self.b_affordance_feedback = np.zeros(self.hidden_dim, dtype=float)
        self.last_affordance_summary["shelter_column_logits"] = []

    def reset_hidden_state(self) -> None:
        super().reset_hidden_state()
        self.last_affordance_summary["shelter_column_logits"] = []

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        (
            event_context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        ) = self._attention_context(x, h_prev)
        option_input = self._current_option_vector()
        x_aug = np.concatenate([x, event_context, option_input], axis=0)
        h_new = np.tanh(self.W_xh @ x_aug + self.W_hh @ h_prev + self.b_h)
        blocked_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_blocked @ h_new + self.b2_affordance_blocked,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        role_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_role @ h_new + self.b2_affordance_role,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        geometry_logits = np.clip(
            np.nan_to_num(
                self.W2_geometry @ h_new + self.b2_geometry,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        shelter_column_logits = np.clip(
            np.nan_to_num(
                self.W2_shelter_column @ h_new + self.b2_shelter_column,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        blocked_probs = _sigmoid(blocked_logits)
        role_logits_matrix = role_logits.reshape(self.output_dim, self.affordance_role_dim)
        role_probs = np.vstack(
            [softmax(role_logits_matrix[action_idx]) for action_idx in range(self.output_dim)]
        )
        geometry_probs = _sigmoid(geometry_logits)
        shelter_column_logits_matrix = shelter_column_logits.reshape(
            self.output_dim,
            self.shelter_column_dim,
        )
        shelter_column_probs = np.vstack(
            [
                softmax(shelter_column_logits_matrix[action_idx])
                for action_idx in range(self.output_dim)
            ]
        )
        affordance_features = np.concatenate(
            [
                blocked_probs,
                role_probs.reshape(-1),
                geometry_probs,
                shelter_column_probs.reshape(-1),
            ],
            axis=0,
        )
        affordance_feedback = np.tanh(
            self.W_affordance_feedback @ affordance_features
            + self.b_affordance_feedback
        )
        base_policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_logits = np.clip(
            np.nan_to_num(
                self.W2_option @ h_new
                + self.b2_option
                + self.W2_option_feedback @ affordance_feedback
                + self.b2_option_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_probs = softmax(option_logits)
        termination_reason = self._termination_reason()
        if self.current_option_idx < 0 or termination_reason is not None:
            selected_option_idx = int(np.argmax(option_probs))
            self.current_option_idx = selected_option_idx
            self.current_option_age = 0
            self.current_option_steps_remaining = self.option_ttl
        else:
            selected_option_idx = int(self.current_option_idx)
            self.current_option_age += 1
        self.current_option_steps_remaining = max(
            0,
            int(self.current_option_steps_remaining) - 1,
        )
        option_vector = one_hot(selected_option_idx, self.option_dim)
        policy_logits = np.clip(
            np.nan_to_num(
                base_policy_logits
                + self.option_action_bias[selected_option_idx]
                + self.W2_policy_feedback @ affordance_feedback
                + self.b2_policy_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        value = float(
            np.clip(
                np.nan_to_num(
                    self.W2_value @ h_new + self.b2_value,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                )[0],
                -1e6,
                1e6,
            )
        )
        self.hidden_state = h_new.copy()
        self.current_option_vector = option_vector.copy()
        self.last_option_summary = {
            "selected_option": OPTION_NAMES[selected_option_idx],
            "option_age": int(self.current_option_age),
            "option_termination_reason": (
                "initial_selection"
                if termination_reason is None and option_input.sum() <= 0.0
                else ("active" if termination_reason is None else termination_reason)
            ),
            "option_logits": option_logits.round(6).tolist(),
        }
        self.last_affordance_summary = {
            "blocked_logits": blocked_logits.round(6).tolist(),
            "role_logits": role_logits.round(6).tolist(),
            "geometry_logits": geometry_logits.round(6).tolist(),
            "shelter_column_logits": shelter_column_logits.round(6).tolist(),
        }
        if store_cache:
            self.cache = OptionAffordanceTopologyFeedbackCache(
                x=x,
                x_aug=x_aug,
                h_prev=h_prev,
                h_new=h_new,
                query_input=query_input,
                query=query,
                slot_raws=slot_raws,
                keys=keys,
                values=values,
                attention_weights=attention_weights,
                valid_event_type_indices=event_type_indices,
                option_probs=option_probs,
                selected_option_idx=selected_option_idx,
                blocked_probs=blocked_probs,
                role_probs=role_probs,
                affordance_features=affordance_features,
                affordance_feedback=affordance_feedback,
                geometry_probs=geometry_probs,
                shelter_column_probs=shelter_column_probs,
            )
        return policy_logits, value, option_logits

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
        grad_affordance_blocked_logits: Array | None = None,
        grad_affordance_role_logits: Array | None = None,
        grad_geometry_logits: Array | None = None,
        grad_shelter_column_logits: Array | None = None,
    ) -> Array:
        if not isinstance(self.cache, OptionAffordanceTopologyFeedbackCache):
            raise RuntimeError(
                "Recurrent option affordance topology feedback network backward called without topology cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        if grad_affordance_blocked_logits is None:
            grad_affordance_blocked_logits = np.zeros(self.output_dim, dtype=float)
        else:
            grad_affordance_blocked_logits = _clip_grad_logits(
                grad_affordance_blocked_logits,
                grad_clip,
            )
        affordance_role_output_dim = self.output_dim * self.affordance_role_dim
        if grad_affordance_role_logits is None:
            grad_affordance_role_logits = np.zeros(
                affordance_role_output_dim,
                dtype=float,
            )
        else:
            grad_affordance_role_logits = _clip_grad_logits(
                np.asarray(grad_affordance_role_logits, dtype=float),
                grad_clip,
            )
        if grad_geometry_logits is None:
            grad_geometry_logits = np.zeros(self.geometry_feature_dim, dtype=float)
        else:
            grad_geometry_logits = _clip_grad_logits(
                np.asarray(grad_geometry_logits, dtype=float),
                grad_clip,
            )
        shelter_column_output_dim = self.output_dim * self.shelter_column_dim
        if grad_shelter_column_logits is None:
            grad_shelter_column_logits = np.zeros(
                shelter_column_output_dim,
                dtype=float,
            )
        else:
            grad_shelter_column_logits = _clip_grad_logits(
                np.asarray(grad_shelter_column_logits, dtype=float),
                grad_clip,
            )
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        affordance_feedback = self.cache.affordance_feedback
        grad_W2_policy = np.outer(grad_policy_logits, h_new)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_option_action_bias = np.zeros_like(self.option_action_bias)
        grad_option_action_bias[self.cache.selected_option_idx] += grad_policy_logits
        option_target = one_hot(self.cache.selected_option_idx, self.option_dim)
        option_advantage = -grad_value
        grad_option_logits = 0.2 * option_advantage * (
            self.cache.option_probs - option_target
        )
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        grad_W2_policy_feedback = np.outer(grad_policy_logits, affordance_feedback)
        grad_b2_policy_feedback = grad_policy_logits
        grad_W2_option_feedback = np.outer(grad_option_logits, affordance_feedback)
        grad_b2_option_feedback = grad_option_logits
        grad_feedback = (
            self.W2_policy_feedback.T @ grad_policy_logits
            + self.W2_option_feedback.T @ grad_option_logits
        )
        dz_feedback = grad_feedback * (1.0 - affordance_feedback**2)
        grad_W_affordance_feedback = np.outer(
            dz_feedback,
            self.cache.affordance_features,
        )
        grad_b_affordance_feedback = dz_feedback
        grad_affordance_features = self.W_affordance_feedback.T @ dz_feedback
        grad_blocked_probs = grad_affordance_features[: self.output_dim]
        role_prob_end = self.output_dim + affordance_role_output_dim
        geometry_prob_end = role_prob_end + self.geometry_feature_dim
        grad_role_probs = grad_affordance_features[
            self.output_dim : role_prob_end
        ].reshape(self.output_dim, self.affordance_role_dim)
        grad_geometry_probs = grad_affordance_features[role_prob_end:geometry_prob_end]
        grad_shelter_column_probs = grad_affordance_features[
            geometry_prob_end:
        ].reshape(self.output_dim, self.shelter_column_dim)
        grad_affordance_blocked_logits = np.asarray(
            grad_affordance_blocked_logits,
            dtype=float,
        ) + grad_blocked_probs * (
            self.cache.blocked_probs * (1.0 - self.cache.blocked_probs)
        )
        grad_affordance_role_matrix = np.asarray(
            grad_affordance_role_logits,
            dtype=float,
        ).reshape(self.output_dim, self.affordance_role_dim)
        feedback_role_grad_matrix = np.zeros_like(grad_affordance_role_matrix)
        for action_idx in range(self.output_dim):
            role_probs = self.cache.role_probs[action_idx]
            role_prob_grad = grad_role_probs[action_idx]
            feedback_role_grad_matrix[action_idx] = role_probs * (
                role_prob_grad - float(np.dot(role_prob_grad, role_probs))
            )
        grad_affordance_role_logits = (
            grad_affordance_role_matrix + feedback_role_grad_matrix
        ).reshape(-1)
        grad_geometry_logits = np.asarray(grad_geometry_logits, dtype=float) + (
            grad_geometry_probs * self.cache.geometry_probs * (1.0 - self.cache.geometry_probs)
        )
        grad_shelter_column_matrix = np.asarray(
            grad_shelter_column_logits,
            dtype=float,
        ).reshape(self.output_dim, self.shelter_column_dim)
        feedback_column_grad_matrix = np.zeros_like(grad_shelter_column_matrix)
        for action_idx in range(self.output_dim):
            column_probs = self.cache.shelter_column_probs[action_idx]
            column_prob_grad = grad_shelter_column_probs[action_idx]
            feedback_column_grad_matrix[action_idx] = column_probs * (
                column_prob_grad
                - float(np.dot(column_prob_grad, column_probs))
            )
        grad_shelter_column_logits = (
            grad_shelter_column_matrix + feedback_column_grad_matrix
        ).reshape(-1)
        grad_W2_affordance_blocked = np.outer(
            grad_affordance_blocked_logits,
            h_new,
        )
        grad_b2_affordance_blocked = grad_affordance_blocked_logits
        grad_W2_affordance_role = np.outer(
            grad_affordance_role_logits,
            h_new,
        )
        grad_b2_affordance_role = grad_affordance_role_logits
        grad_W2_geometry = np.outer(grad_geometry_logits, h_new)
        grad_b2_geometry = grad_geometry_logits
        grad_W2_shelter_column = np.outer(grad_shelter_column_logits, h_new)
        grad_b2_shelter_column = grad_shelter_column_logits
        dh = (
            self.W2_policy.T @ grad_policy_logits
            + self.W2_value.T[:, 0] * grad_value
            + self.W2_option.T @ grad_option_logits
            + self.W2_affordance_blocked.T @ grad_affordance_blocked_logits
            + self.W2_affordance_role.T @ grad_affordance_role_logits
            + self.W2_geometry.T @ grad_geometry_logits
            + self.W2_shelter_column.T @ grad_shelter_column_logits
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[
            self.input_dim : self.input_dim + self.event_context_dim
        ].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz
        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)
        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value
                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    : self.event_embedding_dim
                ]
            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query
        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
        self.option_action_bias -= lr * grad_option_action_bias
        self.W2_policy_feedback -= lr * grad_W2_policy_feedback
        self.b2_policy_feedback -= lr * grad_b2_policy_feedback
        self.W2_option_feedback -= lr * grad_W2_option_feedback
        self.b2_option_feedback -= lr * grad_b2_option_feedback
        self.W_affordance_feedback -= lr * grad_W_affordance_feedback
        self.b_affordance_feedback -= lr * grad_b_affordance_feedback
        self.W2_affordance_blocked -= lr * grad_W2_affordance_blocked
        self.b2_affordance_blocked -= lr * grad_b2_affordance_blocked
        self.W2_affordance_role -= lr * grad_W2_affordance_role
        self.b2_affordance_role -= lr * grad_b2_affordance_role
        self.W2_geometry -= lr * grad_W2_geometry
        self.b2_geometry -= lr * grad_b2_geometry
        self.W2_shelter_column -= lr * grad_W2_shelter_column
        self.b2_shelter_column -= lr * grad_b2_shelter_column
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        return grad_x

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["shelter_column_head"] = True
        state["shelter_column_dim"] = self.shelter_column_dim
        state["shelter_column_feature_dim"] = self.shelter_column_feature_dim
        state["W2_shelter_column"] = self.W2_shelter_column.copy()
        state["b2_shelter_column"] = self.b2_shelter_column.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys=set(super().state_dict().keys()) | {
                "shelter_column_head",
                "shelter_column_dim",
                "shelter_column_feature_dim",
                "W2_shelter_column",
                "b2_shelter_column",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
                "option_head": True,
                "option_ttl": self.option_ttl,
                "option_dim": self.option_dim,
                "affordance_head": True,
                "affordance_role_dim": self.affordance_role_dim,
                "affordance_feedback": True,
                "geometry_head": True,
                "geometry_dim": self.geometry_dim,
                "geometry_feature_dim": self.geometry_feature_dim,
                "shelter_column_head": True,
                "shelter_column_dim": self.shelter_column_dim,
                "shelter_column_feature_dim": self.shelter_column_feature_dim,
            },
            name=self.name,
        )
        super().load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "shelter_column_head",
                    "shelter_column_dim",
                    "shelter_column_feature_dim",
                    "W2_shelter_column",
                    "b2_shelter_column",
                }
            }
        )
        self.W2_shelter_column = _coerce_state_array(
            state,
            "W2_shelter_column",
            (self.shelter_column_feature_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_shelter_column = _coerce_state_array(
            state,
            "b2_shelter_column",
            (self.shelter_column_feature_dim,),
            name=self.name,
        )
        self.last_affordance_summary["shelter_column_logits"] = []

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_option,
            self.b2_option,
            self.option_action_bias,
            self.W2_affordance_blocked,
            self.b2_affordance_blocked,
            self.W2_affordance_role,
            self.b2_affordance_role,
            self.W2_geometry,
            self.b2_geometry,
            self.W2_shelter_column,
            self.b2_shelter_column,
            self.W_affordance_feedback,
            self.b_affordance_feedback,
            self.W2_policy_feedback,
            self.b2_policy_feedback,
            self.W2_option_feedback,
            self.b2_option_feedback,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            super().count_parameters()
            + self.W2_shelter_column.size
            + self.b2_shelter_column.size
        )


class RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
    RecurrentOptionAffordanceGeometryFeedbackTrueMonolithicNetwork
):
    """Geometry-feedback controller with joint shelter-position targets."""

    shelter_position_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        phase_output_dim: int = 0,
        phase_option_feedback: bool = False,
        option_transition_feedback: bool = False,
        option_termination_cooldown: bool = False,
        option_action_head: bool = False,
        option_decoder_state: bool = False,
        option_recurrent_dynamics: bool = False,
        option_sequence_head: bool = False,
        option_decoder_recurrent_state: bool = False,
        option_action_transition_state: bool = False,
        option_action_controller_state: bool = False,
        option_action_token_decoder: bool = False,
        option_action_recurrent_core: bool = False,
        option_action_separate_recurrent_head: bool = False,
        option_action_separate_policy_path: bool = False,
        option_action_separate_backbone: bool = False,
        executive_physiology_option_gating: bool = False,
        executive_affordance_action_gating: bool = False,
        executive_option_action_masking: bool = False,
        executive_event_release_latching: bool = False,
        executive_event_release_action_commitment: bool = False,
        executive_release_phase_state: bool = False,
        executive_release_progression: bool = False,
        executive_release_exit_contract: bool = False,
        executive_release_substate_progression: bool = False,
        executive_post_exit_continuation: bool = False,
        executive_post_exit_food_guidance: bool = False,
        executive_post_exit_food_commitment: bool = False,
        executive_post_exit_food_progression: bool = False,
        executive_post_exit_food_heading_progression: bool = False,
        executive_post_exit_smell_progression: bool = False,
        executive_post_exit_corridor_progression: bool = False,
        executive_post_exit_corridor_affordance_progression: bool = False,
        executive_post_food_return: bool = False,
        executive_post_food_vector_return: bool = False,
        executive_post_food_path_return: bool = False,
        transition_prediction_head: bool = False,
        transition_prediction_feedback: bool = False,
        transition_rollout_prediction_head: bool = False,
        transition_rollout_prediction_feedback: bool = False,
        name: str = "true_monolithic_policy",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            event_buffer_size=event_buffer_size,
            option_ttl=option_ttl,
            name=name,
        )
        self.phase_output_dim = int(phase_output_dim)
        self.phase_option_feedback = bool(
            phase_option_feedback and self.phase_output_dim > 0
        )
        self.option_transition_feedback = bool(option_transition_feedback)
        self.option_termination_cooldown = bool(option_termination_cooldown)
        self.option_action_head = bool(option_action_head)
        self.option_decoder_state = bool(option_decoder_state)
        self.option_recurrent_dynamics = bool(option_recurrent_dynamics)
        self.option_sequence_head = bool(option_sequence_head)
        self.option_decoder_recurrent_state = bool(option_decoder_recurrent_state)
        self.option_action_transition_state = bool(option_action_transition_state)
        self.option_action_controller_state = bool(option_action_controller_state)
        self.option_action_token_decoder = bool(option_action_token_decoder)
        self.option_action_recurrent_core = bool(option_action_recurrent_core)
        self.option_action_separate_recurrent_head = bool(
            option_action_separate_recurrent_head
        )
        self.option_action_separate_policy_path = bool(
            option_action_separate_policy_path
        )
        self.option_action_separate_backbone = bool(
            option_action_separate_backbone
        )
        self.executive_physiology_option_gating = bool(
            executive_physiology_option_gating and self.option_dim > 0
        )
        self.executive_affordance_action_gating = bool(
            executive_affordance_action_gating
            and self.option_dim > 0
            and self.output_dim > 0
        )
        self.executive_option_action_masking = bool(
            executive_option_action_masking and self.executive_affordance_action_gating
        )
        self.executive_event_release_latching = bool(
            executive_event_release_latching
            and self.executive_physiology_option_gating
            and self.executive_affordance_action_gating
        )
        self.executive_event_release_action_commitment = bool(
            executive_event_release_action_commitment
            and self.executive_event_release_latching
        )
        self.executive_release_phase_state = bool(
            executive_release_phase_state
            and self.executive_event_release_action_commitment
        )
        self.executive_release_progression = bool(
            executive_release_progression and self.executive_release_phase_state
        )
        self.executive_release_exit_contract = bool(
            executive_release_exit_contract and self.executive_release_phase_state
        )
        self.executive_release_substate_progression = bool(
            executive_release_substate_progression
            and self.executive_release_exit_contract
        )
        self.executive_post_exit_continuation = bool(
            executive_post_exit_continuation
            and self.executive_release_substate_progression
        )
        self.executive_post_exit_food_guidance = bool(
            executive_post_exit_food_guidance
            and self.executive_post_exit_continuation
        )
        self.executive_post_exit_food_commitment = bool(
            executive_post_exit_food_commitment
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_food_progression = bool(
            executive_post_exit_food_progression
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_food_heading_progression = bool(
            executive_post_exit_food_heading_progression
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_smell_progression = bool(
            executive_post_exit_smell_progression
            and self.executive_post_exit_food_guidance
        )
        self.executive_post_exit_corridor_progression = bool(
            executive_post_exit_corridor_progression
            and self.executive_post_exit_continuation
        )
        self.executive_post_exit_corridor_affordance_progression = bool(
            executive_post_exit_corridor_affordance_progression
            and self.executive_post_exit_corridor_progression
        )
        self.executive_post_food_return = bool(
            executive_post_food_return and self.executive_post_exit_continuation
        )
        self.executive_post_food_vector_return = bool(
            executive_post_food_vector_return and self.executive_post_food_return
        )
        self.executive_post_food_path_return = bool(
            executive_post_food_path_return and self.executive_post_food_return
        )
        self.transition_prediction_head = bool(transition_prediction_head)
        self.transition_prediction_feedback = bool(
            transition_prediction_feedback and self.transition_prediction_head
        )
        self.transition_rollout_prediction_head = bool(
            transition_rollout_prediction_head
        )
        self.transition_rollout_prediction_feedback = bool(
            transition_rollout_prediction_feedback
            and self.transition_rollout_prediction_head
        )
        self.option_cooldowns = np.zeros(self.option_dim, dtype=int)
        self.decoder_action_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_backbone_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_policy_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_controller_state = np.zeros(self.hidden_dim, dtype=float)
        self.action_token_state = np.zeros(self.hidden_dim, dtype=float)
        self.previous_action_idx = -1
        self.executive_release_steps_remaining = 0
        self.executive_post_exit_steps_remaining = 0
        self.executive_runtime_meta: dict[str, object] = {}
        self.executive_post_food_path_history: list[int] = []
        self.executive_post_food_return_queue: list[int] = []
        self.shelter_position_feature_dim = int(
            self.output_dim * self.shelter_position_dim
        )
        self.affordance_feature_dim = int(
            self.output_dim
            + self.output_dim * self.affordance_role_dim
            + self.geometry_feature_dim
            + self.shelter_position_feature_dim
        )
        self.transition_prediction_feature_dim = int(
            DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM
        )
        self.transition_rollout_prediction_feature_dim = int(
            DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM
        )
        self.W2_shelter_position = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.shelter_position_feature_dim, self.hidden_dim),
        )
        self.b2_shelter_position = np.zeros(
            self.shelter_position_feature_dim,
            dtype=float,
        )
        self.W2_transition_prediction = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.transition_prediction_feature_dim, self.hidden_dim),
            )
            if self.transition_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_transition_prediction = (
            np.zeros(self.transition_prediction_feature_dim, dtype=float)
            if self.transition_prediction_head
            else np.zeros(0, dtype=float)
        )
        self.W_transition_prediction_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.transition_prediction_feature_dim),
                size=(self.hidden_dim, self.transition_prediction_feature_dim),
            )
            if self.transition_prediction_feedback
            else np.zeros((0, self.transition_prediction_feature_dim), dtype=float)
        )
        self.b_transition_prediction_feedback = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.transition_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_transition_rollout_prediction = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(
                    self.transition_rollout_prediction_feature_dim,
                    self.hidden_dim,
                ),
            )
            if self.transition_rollout_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_transition_rollout_prediction = (
            np.zeros(
                self.transition_rollout_prediction_feature_dim,
                dtype=float,
            )
            if self.transition_rollout_prediction_head
            else np.zeros(0, dtype=float)
        )
        self.W_transition_rollout_prediction_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.transition_rollout_prediction_feature_dim),
                size=(self.hidden_dim, self.transition_rollout_prediction_feature_dim),
            )
            if self.transition_rollout_prediction_feedback
            else np.zeros(
                (0, self.transition_rollout_prediction_feature_dim),
                dtype=float,
            )
        )
        self.b_transition_rollout_prediction_feedback = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.transition_rollout_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_phase = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.phase_output_dim, self.hidden_dim),
            )
            if self.phase_output_dim > 0
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_phase = (
            np.zeros(self.phase_output_dim, dtype=float)
            if self.phase_output_dim > 0
            else np.zeros(0, dtype=float)
        )
        self.W2_phase_option_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.phase_output_dim),
                size=(self.option_dim, self.phase_output_dim),
            )
            if self.phase_option_feedback
            else np.zeros((self.option_dim, 0), dtype=float)
        )
        self.b2_phase_option_feedback = (
            np.zeros(self.option_dim, dtype=float)
            if self.phase_option_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_option_transition_feedback = (
            rng.normal(
                0.0,
                _weight_scale(self.option_dim),
                size=(self.option_dim, self.option_dim),
            )
            if self.option_transition_feedback
            else np.zeros((self.option_dim, 0), dtype=float)
        )
        self.b2_option_transition_feedback = (
            np.zeros(self.option_dim, dtype=float)
            if self.option_transition_feedback
            else np.zeros(0, dtype=float)
        )
        self.W2_option_action_head = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.output_dim, self.hidden_dim),
            )
            if self.option_action_head
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        self.b2_option_action_head = (
            np.zeros((self.option_dim, self.output_dim), dtype=float)
            if self.option_action_head
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.W_option_decoder_state = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_decoder_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.b_option_decoder_state = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_decoder_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_recurrent_dynamics = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.b_option_recurrent_dynamics = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W2_option_sequence_head = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(
                    self.option_dim,
                    self.option_ttl,
                    self.output_dim,
                    self.hidden_dim,
                ),
            )
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim, self.hidden_dim), dtype=float)
        )
        self.b2_option_sequence_head = (
            np.zeros((self.option_dim, self.option_ttl, self.output_dim), dtype=float)
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim), dtype=float)
        )
        self.W_option_decoder_recurrent_state = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.b_option_decoder_recurrent_state = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_action_transition_state = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_transition_state = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_action_controller_decoder = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_controller_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_controller_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_controller = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W2_option_action_controller_head = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.output_dim, self.hidden_dim),
            )
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        self.b2_option_action_controller_head = (
            np.zeros((self.option_dim, self.output_dim), dtype=float)
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.W_option_action_token_decoder = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_token_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_token_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_token = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_option_action_policy_decoder = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_policy_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.option_dim, self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        self.W_option_action_policy_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.option_dim, self.hidden_dim, self.output_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        self.b_option_action_policy = (
            np.zeros((self.option_dim, self.hidden_dim), dtype=float)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W2_action_policy_core = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.output_dim, self.hidden_dim),
            )
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_action_policy_core = (
            np.zeros(self.output_dim, dtype=float)
            if self.option_action_recurrent_core
            else np.zeros(0, dtype=float)
        )
        self.W_action_policy_path_input = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_action_policy_path_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_action_policy_path_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.hidden_dim, self.output_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.b_action_policy_path = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.option_action_separate_policy_path
            else np.zeros(0, dtype=float)
        )
        self.W2_action_policy_path = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.output_dim, self.hidden_dim),
            )
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_action_policy_path = (
            np.zeros(self.output_dim, dtype=float)
            if self.option_action_separate_policy_path
            else np.zeros(0, dtype=float)
        )
        self.W_action_backbone_input = (
            rng.normal(
                0.0,
                _weight_scale(self.input_dim),
                size=(self.hidden_dim, self.input_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.input_dim), dtype=float)
        )
        self.W_action_backbone_prev = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.hidden_dim, self.hidden_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.W_action_backbone_action = (
            rng.normal(
                0.0,
                _weight_scale(self.output_dim),
                size=(self.hidden_dim, self.output_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.output_dim), dtype=float)
        )
        self.b_action_backbone = (
            np.zeros(self.hidden_dim, dtype=float)
            if self.option_action_separate_backbone
            else np.zeros(0, dtype=float)
        )
        self.W2_action_backbone = (
            rng.normal(
                0.0,
                _weight_scale(hidden_dim),
                size=(self.output_dim, self.hidden_dim),
            )
            if self.option_action_separate_backbone
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        self.b2_action_backbone = (
            np.zeros(self.output_dim, dtype=float)
            if self.option_action_separate_backbone
            else np.zeros(0, dtype=float)
        )
        self.W_affordance_feedback = rng.normal(
            0.0,
            _weight_scale(self.affordance_feature_dim),
            size=(self.hidden_dim, self.affordance_feature_dim),
        )
        self.b_affordance_feedback = np.zeros(self.hidden_dim, dtype=float)
        self.last_affordance_summary["shelter_position_logits"] = []

    def reset_hidden_state(self) -> None:
        super().reset_hidden_state()
        self.option_cooldowns.fill(0)
        self.decoder_action_state.fill(0.0)
        self.action_backbone_state.fill(0.0)
        self.action_policy_state.fill(0.0)
        self.action_controller_state.fill(0.0)
        self.action_token_state.fill(0.0)
        self.previous_action_idx = -1
        self.executive_release_steps_remaining = 0
        self.executive_post_exit_steps_remaining = 0
        self.executive_post_exit_corridor_steps_remaining = 0
        self.executive_post_food_return_steps_remaining = 0
        self.executive_runtime_meta = {}
        self.executive_post_food_path_history = []
        self.executive_post_food_return_queue = []
        self.last_affordance_summary["shelter_position_logits"] = []

    def set_runtime_observation_meta(self, meta: dict[str, object] | None) -> None:
        self.executive_runtime_meta = dict(meta or {})

    def get_runtime_state(self) -> dict[str, object]:
        runtime_state = super().get_runtime_state()
        runtime_state["option_cooldowns"] = self.option_cooldowns.copy()
        runtime_state["decoder_action_state"] = self.decoder_action_state.copy()
        runtime_state["action_backbone_state"] = self.action_backbone_state.copy()
        runtime_state["action_policy_state"] = self.action_policy_state.copy()
        runtime_state["action_controller_state"] = self.action_controller_state.copy()
        runtime_state["action_token_state"] = self.action_token_state.copy()
        runtime_state["previous_action_idx"] = int(self.previous_action_idx)
        runtime_state["executive_release_steps_remaining"] = int(
            self.executive_release_steps_remaining
        )
        runtime_state["executive_post_exit_steps_remaining"] = int(
            self.executive_post_exit_steps_remaining
        )
        runtime_state["executive_post_exit_corridor_steps_remaining"] = int(
            self.executive_post_exit_corridor_steps_remaining
        )
        runtime_state["executive_post_food_return_steps_remaining"] = int(
            self.executive_post_food_return_steps_remaining
        )
        runtime_state["executive_post_food_path_history"] = list(
            self.executive_post_food_path_history
        )
        runtime_state["executive_post_food_return_queue"] = list(
            self.executive_post_food_return_queue
        )
        return runtime_state

    def set_runtime_state(self, runtime_state: dict[str, object]) -> None:
        super().set_runtime_state(runtime_state)
        cooldowns = runtime_state.get("option_cooldowns")
        decoder_action_state = runtime_state.get("decoder_action_state")
        action_backbone_state = runtime_state.get("action_backbone_state")
        action_policy_state = runtime_state.get("action_policy_state")
        action_controller_state = runtime_state.get("action_controller_state")
        action_token_state = runtime_state.get("action_token_state")
        previous_action_idx = runtime_state.get("previous_action_idx", -1)
        executive_release_steps_remaining = runtime_state.get(
            "executive_release_steps_remaining",
            0,
        )
        executive_post_exit_steps_remaining = runtime_state.get(
            "executive_post_exit_steps_remaining",
            0,
        )
        executive_post_exit_corridor_steps_remaining = runtime_state.get(
            "executive_post_exit_corridor_steps_remaining",
            0,
        )
        executive_post_food_return_steps_remaining = runtime_state.get(
            "executive_post_food_return_steps_remaining",
            0,
        )
        executive_post_food_path_history = runtime_state.get(
            "executive_post_food_path_history",
            [],
        )
        executive_post_food_return_queue = runtime_state.get(
            "executive_post_food_return_queue",
            [],
        )
        if cooldowns is None:
            self.option_cooldowns.fill(0)
        else:
            cooldown_array = np.asarray(cooldowns, dtype=int)
            if cooldown_array.shape != (self.option_dim,):
                raise ValueError(
                    f"{self.name}: option_cooldowns expected {(self.option_dim,)}, received {cooldown_array.shape}"
                )
            self.option_cooldowns = np.maximum(0, cooldown_array)
        self.executive_release_steps_remaining = max(
            0,
            int(executive_release_steps_remaining),
        )
        self.executive_post_exit_steps_remaining = max(
            0,
            int(executive_post_exit_steps_remaining),
        )
        self.executive_post_exit_corridor_steps_remaining = max(
            0,
            int(executive_post_exit_corridor_steps_remaining),
        )
        self.executive_post_food_return_steps_remaining = max(
            0,
            int(executive_post_food_return_steps_remaining),
        )
        self.executive_post_food_path_history = [
            int(action_idx)
            for action_idx in list(executive_post_food_path_history)
            if int(action_idx) in _LOCAL_ACTION_TO_POLICY_INDEX.values()
        ]
        self.executive_post_food_return_queue = [
            int(action_idx)
            for action_idx in list(executive_post_food_return_queue)
            if int(action_idx) in _LOCAL_ACTION_TO_POLICY_INDEX.values()
        ]
        if decoder_action_state is None:
            self.decoder_action_state.fill(0.0)
            return
        decoder_action_state_array = np.asarray(decoder_action_state, dtype=float)
        if decoder_action_state_array.shape != (self.hidden_dim,):
            raise ValueError(
                f"{self.name}: decoder_action_state expected {(self.hidden_dim,)}, received {decoder_action_state_array.shape}"
            )
        self.decoder_action_state = decoder_action_state_array.copy()
        if action_backbone_state is None:
            self.action_backbone_state.fill(0.0)
        else:
            action_backbone_state_array = np.asarray(
                action_backbone_state,
                dtype=float,
            )
            if action_backbone_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_backbone_state expected {(self.hidden_dim,)}, received {action_backbone_state_array.shape}"
                )
            self.action_backbone_state = action_backbone_state_array.copy()
        if action_policy_state is None:
            self.action_policy_state.fill(0.0)
        else:
            action_policy_state_array = np.asarray(action_policy_state, dtype=float)
            if action_policy_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_policy_state expected {(self.hidden_dim,)}, received {action_policy_state_array.shape}"
                )
            self.action_policy_state = action_policy_state_array.copy()
        if action_controller_state is None:
            self.action_controller_state.fill(0.0)
        else:
            action_controller_state_array = np.asarray(
                action_controller_state,
                dtype=float,
            )
            if action_controller_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_controller_state expected {(self.hidden_dim,)}, received {action_controller_state_array.shape}"
                )
            self.action_controller_state = action_controller_state_array.copy()
        if action_token_state is None:
            self.action_token_state.fill(0.0)
        else:
            action_token_state_array = np.asarray(
                action_token_state,
                dtype=float,
            )
            if action_token_state_array.shape != (self.hidden_dim,):
                raise ValueError(
                    f"{self.name}: action_token_state expected {(self.hidden_dim,)}, received {action_token_state_array.shape}"
                )
            self.action_token_state = action_token_state_array.copy()
        self.previous_action_idx = int(previous_action_idx)

    def record_executed_action(self, action_idx: int) -> None:
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self.output_dim:
            raise ValueError(
                f"{self.name}: action_idx expected in [0, {self.output_dim}), received {action_idx}"
            )
        locomotion_indices = set(_LOCAL_ACTION_TO_POLICY_INDEX.values())
        if (
            self.executive_post_food_path_return
            and action_idx in locomotion_indices
            and self.executive_post_food_return_steps_remaining <= 0
            and self.executive_post_exit_steps_remaining > 0
            and len(self.executive_post_food_path_history) < 16
        ):
            self.executive_post_food_path_history.append(action_idx)
        if (
            self.executive_post_food_path_return
            and self.executive_post_food_return_queue
            and action_idx == int(self.executive_post_food_return_queue[0])
        ):
            self.executive_post_food_return_queue.pop(0)
        self.previous_action_idx = action_idx

    def _should_cooldown_terminated_option(self, termination_reason: str | None) -> bool:
        if not self.option_termination_cooldown or self.current_option_idx < 0:
            return False
        current_option = OPTION_NAMES[int(self.current_option_idx)]
        return (
            (current_option == "REST" and termination_reason == "recovery_completed")
            or (
                current_option == "RETURN_TO_SHELTER"
                and termination_reason == "shelter_exited"
            )
            or (current_option == "FORAGE" and termination_reason == "food_reached")
            or (
                current_option == "DEEPEN_IN_SHELTER"
                and termination_reason == "deep_shelter_reached"
            )
        )

    @staticmethod
    def _bounded_input_signal(x: Array, index: int) -> float:
        if index < 0 or index >= x.shape[0]:
            return 0.0
        return float(np.clip(x[index], -1.0, 1.0))

    def _executive_state_signals(self, x: Array) -> dict[str, float]:
        predator_visible = self._bounded_input_signal(
            x, _ALERT_PREDATOR_VISIBLE_IDX
        )
        predator_certainty = self._bounded_input_signal(
            x, _ALERT_PREDATOR_CERTAINTY_IDX
        )
        predator_smell_strength = self._bounded_input_signal(
            x, _ALERT_PREDATOR_SMELL_STRENGTH_IDX
        )
        predator_motion_salience = self._bounded_input_signal(
            x, _ALERT_PREDATOR_MOTION_SALIENCE_IDX
        )
        visual_predator_threat = self._bounded_input_signal(
            x, _ALERT_VISUAL_PREDATOR_THREAT_IDX
        )
        olfactory_predator_threat = self._bounded_input_signal(
            x, _ALERT_OLFACTORY_PREDATOR_THREAT_IDX
        )
        recent_pain = self._bounded_input_signal(x, _ALERT_RECENT_PAIN_IDX)
        recent_contact = self._bounded_input_signal(x, _ALERT_RECENT_CONTACT_IDX)
        predator_trace_strength = self._bounded_input_signal(
            x, _ALERT_PREDATOR_TRACE_STRENGTH_IDX
        )
        acute_threat = max(
            predator_visible * predator_certainty,
            predator_smell_strength,
            predator_motion_salience,
            visual_predator_threat,
            olfactory_predator_threat,
            recent_pain,
            recent_contact,
            predator_trace_strength,
        )
        night = self._bounded_input_signal(x, _SLEEP_NIGHT_IDX)
        fatigue = self._bounded_input_signal(x, _SLEEP_FATIGUE_IDX)
        sleep_debt = self._bounded_input_signal(x, _SLEEP_DEBT_IDX)
        return {
            "on_shelter": self._bounded_input_signal(x, _SLEEP_ON_SHELTER_IDX),
            "hunger": self._bounded_input_signal(x, _SLEEP_HUNGER_IDX),
            "fatigue": fatigue,
            "night": night,
            "sleep_phase_level": self._bounded_input_signal(
                x, _SLEEP_PHASE_LEVEL_IDX
            ),
            "rest_streak": self._bounded_input_signal(
                x, _SLEEP_REST_STREAK_IDX
            ),
            "sleep_debt": sleep_debt,
            "shelter_role_level": self._bounded_input_signal(
                x, _SLEEP_SHELTER_ROLE_LEVEL_IDX
            ),
            "shelter_memory_age": self._bounded_input_signal(
                x, _SLEEP_SHELTER_MEMORY_AGE_IDX
            ),
            "acute_threat": acute_threat,
            "rest_pressure": max(night, fatigue, sleep_debt),
            "food_visible": self._bounded_input_signal(x, _HUNGER_FOOD_VISIBLE_IDX),
            "food_certainty": self._bounded_input_signal(x, _HUNGER_FOOD_CERTAINTY_IDX),
            "food_dx": self._bounded_input_signal(x, _HUNGER_FOOD_DX_IDX),
            "food_dy": self._bounded_input_signal(x, _HUNGER_FOOD_DY_IDX),
            "food_smell_strength": self._bounded_input_signal(
                x, _HUNGER_FOOD_SMELL_STRENGTH_IDX
            ),
            "food_smell_dx": self._bounded_input_signal(x, _HUNGER_FOOD_SMELL_DX_IDX),
            "food_smell_dy": self._bounded_input_signal(x, _HUNGER_FOOD_SMELL_DY_IDX),
            "food_memory_dx": self._bounded_input_signal(x, _HUNGER_FOOD_MEMORY_DX_IDX),
            "food_memory_dy": self._bounded_input_signal(x, _HUNGER_FOOD_MEMORY_DY_IDX),
            "food_memory_age": self._bounded_input_signal(
                x, _HUNGER_FOOD_MEMORY_AGE_IDX
            ),
        }

    def _apply_executive_physiology_option_gating(
        self,
        x: Array,
        selection_option_logits: Array,
    ) -> Array:
        if not self.executive_physiology_option_gating:
            return selection_option_logits
        signals = self._executive_state_signals(x)
        on_shelter = signals["on_shelter"]
        hunger = signals["hunger"]
        sleep_phase_level = signals["sleep_phase_level"]
        rest_streak = signals["rest_streak"]
        shelter_role_level = signals["shelter_role_level"]
        shelter_memory_age = signals["shelter_memory_age"]
        acute_threat = signals["acute_threat"]
        rest_pressure = signals["rest_pressure"]
        sheltered = on_shelter > 0.5
        fresh_shelter_memory = shelter_memory_age < 0.95
        gated_logits = selection_option_logits.copy()

        if self.executive_post_food_return and self.executive_post_food_return_steps_remaining > 0:
            if sheltered:
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 8.0
                gated_logits[OPTION_TO_INDEX["REST"]] += 6.0
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 8.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 8.0
            else:
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 10.0
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 8.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 6.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 4.0
            return gated_logits

        if sheltered:
            if acute_threat >= 0.2:
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 4.0
                gated_logits[OPTION_TO_INDEX["REST"]] += 2.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 6.0
            elif rest_pressure >= 0.18 and hunger <= 0.22:
                gated_logits[OPTION_TO_INDEX["REST"]] += 6.0
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 3.0
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 6.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 4.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 8.0
            elif (
                hunger >= 0.16
                and rest_pressure <= 0.12
                and sleep_phase_level <= 0.2
                and rest_streak >= 0.1
            ):
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] += 6.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] += 2.0
                gated_logits[OPTION_TO_INDEX["REST"]] -= 4.0
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 3.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 8.0
            else:
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 2.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 6.0
            return gated_logits

        if acute_threat >= 0.2:
            gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 6.0
            gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 6.0
            gated_logits[OPTION_TO_INDEX["REST"]] -= 8.0
            gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 8.0
            if fresh_shelter_memory:
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 6.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 2.0
            else:
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] += 4.0
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 2.0
            return gated_logits

        if rest_pressure >= 0.18 and fresh_shelter_memory:
            gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 5.0
            gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 3.0
            gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 2.0
            return gated_logits

        if hunger >= 0.16:
            gated_logits[OPTION_TO_INDEX["FORAGE"]] += 4.0
            gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] += 3.0
            gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 4.0
            gated_logits[OPTION_TO_INDEX["REST"]] -= 6.0
            gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 6.0
            if fresh_shelter_memory:
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] -= 1.0

        if shelter_role_level < 0.25:
            gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 4.0
        return gated_logits

    def _prime_executive_release_latch(
        self,
        x: Array,
        termination_reason: str | None,
    ) -> None:
        if not self.executive_event_release_latching:
            return
        signals = self._executive_state_signals(x)
        sheltered = signals["on_shelter"] > 0.5
        hunger = signals["hunger"]
        acute_threat = signals["acute_threat"]
        role_level = signals["shelter_role_level"]
        if (
            not sheltered
            or acute_threat >= 0.2
            or hunger < 0.12
            or termination_reason in {"shelter_exited", "acute_predator_threat"}
        ):
            self.executive_release_steps_remaining = 0
            return
        if termination_reason == "recovery_completed":
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )
            return
        if (
            termination_reason in {"deep_shelter_reached", "no_progress"}
            and role_level >= 0.95
        ):
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )

    def _prime_executive_release_phase_state(
        self,
        x: Array,
        termination_reason: str | None,
    ) -> None:
        if not self.executive_release_phase_state:
            return
        signals = self._executive_state_signals(x)
        sheltered = signals["on_shelter"] > 0.5
        hunger = signals["hunger"]
        acute_threat = signals["acute_threat"]
        rest_pressure = signals["rest_pressure"]
        role_level = signals["shelter_role_level"]
        if not sheltered or acute_threat >= 0.2:
            self.executive_release_steps_remaining = 0
            return
        if self.executive_release_steps_remaining > 0:
            return
        if (
            self.current_option_idx < 0
            and hunger >= 0.08
            and rest_pressure <= 0.32
        ):
            self.executive_release_steps_remaining = 3
            return
        if (
            termination_reason in {"recovery_completed", "deep_shelter_reached"}
            and hunger >= 0.12
        ):
            self.executive_release_steps_remaining = 3
            return
        if (
            termination_reason == "no_progress"
            and role_level >= 0.95
            and hunger >= 0.12
        ):
            self.executive_release_steps_remaining = 3

    def _apply_executive_event_release_latch(
        self,
        x: Array,
        selection_option_logits: Array,
    ) -> Array:
        if (
            not self.executive_event_release_latching
            or self.executive_release_steps_remaining <= 0
        ):
            return selection_option_logits
        signals = self._executive_state_signals(x)
        if signals["on_shelter"] <= 0.5 or signals["acute_threat"] >= 0.2:
            self.executive_release_steps_remaining = 0
            return selection_option_logits
        gated_logits = selection_option_logits.copy()
        gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] += 10.0
        gated_logits[OPTION_TO_INDEX["FORAGE"]] += 3.0
        gated_logits[OPTION_TO_INDEX["REST"]] -= 8.0
        gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 8.0
        gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] -= 4.0
        gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 8.0
        return gated_logits

    def _apply_executive_release_substate_progression(
        self,
        x: Array,
        selected_option_idx: int,
    ) -> None:
        if (
            not self.executive_release_substate_progression
            or self.executive_release_steps_remaining <= 0
            or OPTION_NAMES[selected_option_idx] != "POST_REST_REACTIVATE"
        ):
            return
        signals = self._executive_state_signals(x)
        if signals["on_shelter"] <= 0.5 or signals["acute_threat"] >= 0.2:
            self.executive_release_steps_remaining = 0
            return
        if signals["hunger"] < 0.12:
            return
        role_level = signals["shelter_role_level"]
        if role_level >= 0.95:
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                3,
            )
        elif role_level >= 0.55:
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )
        elif role_level > 0.05:
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )

    def _apply_executive_affordance_action_gating(
        self,
        x: Array,
        selected_option_idx: int,
        policy_logits: Array,
        blocked_probs: Array,
        geometry_probs: Array,
        shelter_position_probs: Array,
    ) -> Array:
        if not self.executive_affordance_action_gating:
            return policy_logits
        signals = self._executive_state_signals(x)
        sheltered = signals["on_shelter"] > 0.5
        hunger = signals["hunger"]
        acute_threat = signals["acute_threat"]
        rest_pressure = signals["rest_pressure"]
        role_level = signals["shelter_role_level"]
        option_name = OPTION_NAMES[selected_option_idx]
        gated_logits = policy_logits.copy()
        orientation_indices = _POLICY_ORIENTATION_ACTION_INDICES
        locomotion_indices = tuple(_LOCAL_ACTION_TO_POLICY_INDEX.values())
        geometry_matrix = geometry_probs.reshape(
            self.output_dim,
            len(AFFORDANCE_GEOMETRY_TARGET_NAMES),
        )

        def action_score(action_name: str) -> float:
            action_idx = _LOCAL_ACTION_TO_POLICY_INDEX[action_name]
            blocked_penalty = 8.0 * float(blocked_probs[action_idx])
            pos_probs = shelter_position_probs[action_idx]
            deep_prob = float(sum(pos_probs[idx] for idx in _DEEP_SHELTER_POSITION_INDICES))
            inside_prob = float(
                sum(pos_probs[idx] for idx in _INSIDE_SHELTER_POSITION_INDICES)
            )
            entrance_prob = float(
                sum(pos_probs[idx] for idx in _ENTRANCE_POSITION_INDICES)
            )
            outside_prob = float(pos_probs[_OUTSIDE_POSITION_INDEX])
            deepen_prob = float(
                geometry_matrix[action_idx, _GEOMETRY_DEEPEN_INDEX]
            )
            outside_geom = float(
                geometry_matrix[action_idx, _GEOMETRY_OUTSIDE_INDEX]
            )
            if option_name == "POST_REST_REACTIVATE":
                return outside_prob * 8.0 + entrance_prob * 3.0 + outside_geom * 4.0 - blocked_penalty
            if option_name == "FORAGE":
                return outside_prob * 6.0 + entrance_prob * 2.0 + outside_geom * 3.0 - blocked_penalty
            if option_name == "RETURN_TO_SHELTER":
                return deep_prob * 5.0 + inside_prob * 3.0 + entrance_prob * 2.0 + deepen_prob * 3.0 - blocked_penalty
            if option_name == "DEEPEN_IN_SHELTER":
                return deep_prob * 8.0 + inside_prob * 2.0 + deepen_prob * 5.0 - blocked_penalty
            if option_name == "REST":
                return 5.0 if action_name == "STAY" else -blocked_penalty
            return -blocked_penalty

        def promote_local_action(action_name: str, bonus: float) -> None:
            action_idx = _LOCAL_ACTION_TO_POLICY_INDEX[action_name]
            gated_logits[action_idx] += bonus

        def guided_food_action() -> str | None:
            if not self.executive_post_exit_food_guidance:
                return None
            dx = 0.0
            dy = 0.0
            weight = 0.0
            if signals["food_visible"] > 0.1 and signals["food_certainty"] > 0.1:
                dx = signals["food_dx"]
                dy = signals["food_dy"]
                weight = signals["food_certainty"]
            elif signals["food_smell_strength"] > 0.1:
                dx = signals["food_smell_dx"]
                dy = signals["food_smell_dy"]
                weight = signals["food_smell_strength"]
            elif signals["food_memory_age"] < 0.75:
                dx = signals["food_memory_dx"]
                dy = signals["food_memory_dy"]
                weight = 1.0 - max(0.0, signals["food_memory_age"])
            if weight <= 0.05:
                return None
            if abs(dx) >= abs(dy):
                if dx > 0.05:
                    return "MOVE_RIGHT"
                if dx < -0.05:
                    return "MOVE_LEFT"
            if dy > 0.05:
                return "MOVE_DOWN"
            if dy < -0.05:
                return "MOVE_UP"
            return None

        def guided_food_action_with_source() -> tuple[str | None, str | None]:
            if not self.executive_post_exit_food_guidance:
                return None, None
            if signals["food_visible"] > 0.1 and signals["food_certainty"] > 0.1:
                dx = signals["food_dx"]
                dy = signals["food_dy"]
                source = "visible"
            elif signals["food_smell_strength"] > 0.1:
                dx = signals["food_smell_dx"]
                dy = signals["food_smell_dy"]
                source = "smell"
            elif signals["food_memory_age"] < 0.75:
                dx = signals["food_memory_dx"]
                dy = signals["food_memory_dy"]
                source = "memory"
            else:
                return None, None
            action = guided_food_action()
            return action, source

        def has_live_food_guidance() -> bool:
            return guided_food_action() is not None

        def guided_shelter_return_action() -> str | None:
            if not self.executive_post_food_vector_return:
                return None
            meta = self.executive_runtime_meta if isinstance(self.executive_runtime_meta, dict) else {}
            memory_vectors = meta.get("memory_vectors", {})
            shelter_vector = (
                memory_vectors.get("shelter", {})
                if isinstance(memory_vectors, dict)
                else {}
            )
            dx = float(shelter_vector.get("dx", 0.0) or 0.0)
            dy = float(shelter_vector.get("dy", 0.0) or 0.0)
            if abs(dx) >= abs(dy):
                if dx > 0.05:
                    action_name = "MOVE_RIGHT"
                elif dx < -0.05:
                    action_name = "MOVE_LEFT"
                else:
                    action_name = None
            else:
                if dy > 0.05:
                    action_name = "MOVE_DOWN"
                elif dy < -0.05:
                    action_name = "MOVE_UP"
                else:
                    action_name = None
            if action_name is None:
                return None
            if float(blocked_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]]) > 0.5:
                return None
            return action_name

        def guided_shelter_path_return_action() -> str | None:
            if not self.executive_post_food_path_return:
                return None
            while self.executive_post_food_return_queue:
                action_idx = int(self.executive_post_food_return_queue[0])
                if action_idx not in _LOCAL_ACTION_TO_POLICY_INDEX.values():
                    self.executive_post_food_return_queue.pop(0)
                    continue
                blocked = float(blocked_probs[action_idx]) > 0.5
                if blocked:
                    return None
                action_name = next(
                    name
                    for name, idx in _LOCAL_ACTION_TO_POLICY_INDEX.items()
                    if idx == action_idx
                )
                return action_name
            return None

        if (
            self.executive_release_exit_contract
            and self.executive_release_steps_remaining > 0
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
        ):
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            if role_level <= 0.4:
                promote_local_action("MOVE_LEFT", 16.0)
                promote_local_action("MOVE_UP", 10.0)
            elif role_level >= 0.95:
                promote_local_action("MOVE_UP", 14.0)
                promote_local_action("MOVE_LEFT", 5.0)
            else:
                promote_local_action("MOVE_UP", 12.0)
                promote_local_action("MOVE_LEFT", 8.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 10.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_RIGHT"]] -= 10.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 8.0
            return gated_logits

        if (
            self.executive_release_progression
            and self.executive_release_steps_remaining > 0
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
        ):
            pos_scores = {
                name: float(
                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]][idx]
                )
                for idx, name in enumerate(AFFORDANCE_SHELTER_POSITION_NAMES)
            }
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            entrance_prob = max(
                pos_scores.get("entrance_left", 0.0),
                pos_scores.get("entrance_center", 0.0),
                pos_scores.get("entrance_right", 0.0),
            )
            if entrance_prob >= 0.3 or role_level <= 0.4:
                promote_local_action("MOVE_RIGHT", 14.0)
                promote_local_action("MOVE_UP", 4.0)
            elif role_level >= 0.95:
                promote_local_action("MOVE_UP", 14.0)
                promote_local_action("MOVE_RIGHT", 3.0)
            else:
                promote_local_action("MOVE_UP", 12.0)
                promote_local_action("MOVE_RIGHT", 6.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 10.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 8.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_LEFT"]] -= 8.0
            return gated_logits

        if (
            self.executive_event_release_action_commitment
            and self.executive_release_steps_remaining > 0
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
        ):
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=lambda action_name: (
                    action_score(action_name)
                    + (1.5 if action_name == "MOVE_UP" else 0.0)
                ),
            )
            promote_local_action(best_action, 12.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if (
            self.executive_post_exit_continuation
            and self.executive_post_exit_steps_remaining > 0
            and not sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
            and option_name == "POST_REST_REACTIVATE"
        ):
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] = -20.0
            best_action, guidance_source = guided_food_action_with_source()
            if (
                self.executive_post_exit_corridor_progression
                and self.executive_post_exit_corridor_steps_remaining > 0
                and best_action is None
            ):
                if self.executive_post_exit_corridor_affordance_progression:
                    move_down_idx = _LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]
                    move_right_idx = _LOCAL_ACTION_TO_POLICY_INDEX["MOVE_RIGHT"]
                    move_down_blocked = float(blocked_probs[move_down_idx]) > 0.5
                    move_right_blocked = float(blocked_probs[move_right_idx]) > 0.5
                    move_down_pos = shelter_position_probs[move_down_idx]
                    move_down_sheltered = float(
                        sum(move_down_pos[idx] for idx in _DEEP_SHELTER_POSITION_INDICES)
                        + sum(move_down_pos[idx] for idx in _INSIDE_SHELTER_POSITION_INDICES)
                        + sum(move_down_pos[idx] for idx in _ENTRANCE_POSITION_INDICES)
                    ) > 0.5
                    best_action = (
                        "MOVE_RIGHT"
                        if (not move_right_blocked and (move_down_blocked or move_down_sheltered))
                        else "MOVE_DOWN"
                    )
                else:
                    best_action = (
                        "MOVE_RIGHT"
                        if self.executive_post_exit_corridor_steps_remaining > 3
                        else "MOVE_DOWN"
                    )
            elif best_action is None:
                best_action = max(
                    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                    key=action_score,
                )
            elif (
                self.executive_post_exit_food_heading_progression
                and guidance_source == "memory"
                and self.previous_action_idx == ACTION_TO_INDEX["MOVE_RIGHT"]
                and best_action == "MOVE_RIGHT"
            ):
                best_action = "MOVE_UP"
            elif (
                self.executive_post_exit_smell_progression
                and guidance_source == "smell"
                and self.previous_action_idx == ACTION_TO_INDEX["MOVE_RIGHT"]
                and best_action == "MOVE_RIGHT"
            ):
                best_action = "MOVE_UP"
            elif (
                self.executive_post_exit_food_progression
                and guidance_source == "memory"
                and self.previous_action_idx >= 0
            ):
                previous_action = DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES[
                    list(_LOCAL_ACTION_TO_POLICY_INDEX.values()).index(self.previous_action_idx)
                ] if self.previous_action_idx in _LOCAL_ACTION_TO_POLICY_INDEX.values() else None
                if previous_action == "MOVE_LEFT" and best_action == "MOVE_RIGHT":
                    best_action = "MOVE_UP"
                elif previous_action == "MOVE_RIGHT" and best_action == "MOVE_LEFT":
                    best_action = "MOVE_UP"
            promote_local_action(best_action, 10.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 6.0
            if self.executive_post_exit_food_commitment and has_live_food_guidance():
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 6.0
            if self.executive_post_exit_food_progression and guidance_source == "memory":
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 4.0
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if (
            self.executive_post_food_return
            and self.executive_post_food_return_steps_remaining > 0
            and acute_threat < 0.2
            and option_name in {"RETURN_TO_SHELTER", "DEEPEN_IN_SHELTER", "REST"}
        ):
            if sheltered and signals["shelter_role_level"] >= 0.95:
                if self.executive_option_action_masking:
                    gated_logits[:] = -20.0
                promote_local_action("STAY", 12.0)
                for idx in orientation_indices:
                    gated_logits[idx] -= 8.0
                return gated_logits
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
            if not sheltered:
                best_action = guided_shelter_path_return_action()
                if best_action is None:
                    best_action = guided_shelter_return_action()
                if best_action is None:
                    best_action = max(
                        DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                        key=lambda action_name: (
                            float(
                                sum(
                                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]][idx]
                                    for idx in _DEEP_SHELTER_POSITION_INDICES
                                )
                            ) * 5.0
                            + float(
                                sum(
                                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]][idx]
                                    for idx in _INSIDE_SHELTER_POSITION_INDICES
                                )
                            ) * 3.0
                            + float(
                                sum(
                                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]][idx]
                                    for idx in _ENTRANCE_POSITION_INDICES
                                )
                            ) * 2.0
                            - 8.0 * float(blocked_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]])
                        ),
                    )
            else:
                best_action = max(
                    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                    key=action_score,
                )
            promote_local_action(best_action, 10.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 4.0
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if option_name == "REST" and sheltered:
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            promote_local_action("STAY", 10.0)
            for idx in locomotion_indices:
                if idx != _LOCAL_ACTION_TO_POLICY_INDEX["STAY"]:
                    gated_logits[idx] -= 4.0
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if (
            option_name == "POST_REST_REACTIVATE"
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.14
            and rest_pressure <= 0.18
        ):
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=action_score,
            )
            promote_local_action(best_action, 8.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 5.0
            for idx in orientation_indices:
                gated_logits[idx] -= 6.0
            return gated_logits

        if option_name in {"RETURN_TO_SHELTER", "DEEPEN_IN_SHELTER"} and (
            not sheltered or role_level < 0.95
        ):
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=action_score,
            )
            promote_local_action(best_action, 7.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 5.0
            return gated_logits

        if option_name == "FORAGE" and sheltered and acute_threat < 0.2:
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=action_score,
            )
            promote_local_action(best_action, 5.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 2.0
            for idx in orientation_indices:
                gated_logits[idx] -= 4.0
        return gated_logits

    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array] | tuple[Array, float, Array, Array]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        (
            event_context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        ) = self._attention_context(x, h_prev)
        previous_option_vector = self._current_option_vector()
        previous_option_idx = (
            int(np.argmax(previous_option_vector))
            if previous_option_vector.sum() > 0.0
            else -1
        )
        x_aug = np.concatenate([x, event_context, previous_option_vector], axis=0)
        recurrent_pre = self.W_xh @ x_aug + self.W_hh @ h_prev + self.b_h
        if self.option_recurrent_dynamics and previous_option_idx >= 0:
            recurrent_pre = (
                recurrent_pre
                + self.W_option_recurrent_dynamics[previous_option_idx] @ h_prev
                + self.b_option_recurrent_dynamics[previous_option_idx]
            )
        h_new = np.tanh(recurrent_pre)
        blocked_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_blocked @ h_new + self.b2_affordance_blocked,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        role_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_role @ h_new + self.b2_affordance_role,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        geometry_logits = np.clip(
            np.nan_to_num(
                self.W2_geometry @ h_new + self.b2_geometry,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        shelter_position_logits = np.clip(
            np.nan_to_num(
                self.W2_shelter_position @ h_new + self.b2_shelter_position,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        blocked_probs = _sigmoid(blocked_logits)
        role_logits_matrix = role_logits.reshape(self.output_dim, self.affordance_role_dim)
        role_probs = np.vstack(
            [softmax(role_logits_matrix[action_idx]) for action_idx in range(self.output_dim)]
        )
        geometry_probs = _sigmoid(geometry_logits)
        shelter_position_logits_matrix = shelter_position_logits.reshape(
            self.output_dim,
            self.shelter_position_dim,
        )
        shelter_position_probs = np.vstack(
            [
                softmax(shelter_position_logits_matrix[action_idx])
                for action_idx in range(self.output_dim)
            ]
        )
        transition_prediction_logits = np.tanh(
            np.nan_to_num(
                self.W2_transition_prediction @ h_new + self.b2_transition_prediction,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        transition_rollout_prediction_logits = np.tanh(
            np.nan_to_num(
                self.W2_transition_rollout_prediction @ h_new
                + self.b2_transition_rollout_prediction,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        affordance_features = np.concatenate(
            [
                blocked_probs,
                role_probs.reshape(-1),
                geometry_probs,
                shelter_position_probs.reshape(-1),
            ],
            axis=0,
        )
        affordance_feedback = np.tanh(
            self.W_affordance_feedback @ affordance_features
            + self.b_affordance_feedback
        )
        transition_prediction_feedback = np.zeros(self.hidden_dim, dtype=float)
        if self.transition_prediction_feedback:
            transition_prediction_feedback = np.tanh(
                self.W_transition_prediction_feedback @ transition_prediction_logits
                + self.b_transition_prediction_feedback
            )
        transition_rollout_prediction_feedback = np.zeros(
            self.hidden_dim,
            dtype=float,
        )
        if self.transition_rollout_prediction_feedback:
            transition_rollout_prediction_feedback = np.tanh(
                self.W_transition_rollout_prediction_feedback
                @ transition_rollout_prediction_logits
                + self.b_transition_rollout_prediction_feedback
            )
        combined_feedback = (
            affordance_feedback
            + transition_prediction_feedback
            + transition_rollout_prediction_feedback
        )
        base_policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_logits = np.clip(
            np.nan_to_num(
                self.W2_option @ h_new
                + self.b2_option
                + self.W2_option_feedback @ combined_feedback
                + self.b2_option_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        phase_logits: Array | None = None
        phase_probs = np.zeros(self.phase_output_dim, dtype=float)
        if self.phase_output_dim > 0:
            phase_logits = np.clip(
                np.nan_to_num(
                    self.W2_phase @ h_new + self.b2_phase,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ),
                -20.0,
                20.0,
            )
            phase_probs = softmax(phase_logits)
        if self.phase_option_feedback:
            option_logits = np.clip(
                np.nan_to_num(
                    option_logits
                    + self.W2_phase_option_feedback @ phase_probs
                    + self.b2_phase_option_feedback,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ),
                -20.0,
                20.0,
            )
        if self.option_transition_feedback:
            option_logits = np.clip(
                np.nan_to_num(
                    option_logits
                    + self.W2_option_transition_feedback @ previous_option_vector
                    + self.b2_option_transition_feedback,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ),
                -20.0,
                20.0,
            )
        termination_reason = self._termination_reason()
        termination_reason = self._apply_executive_post_exit_continuation(
            x,
            termination_reason,
        )
        self._prime_executive_post_food_return(x, termination_reason)
        if self._should_cooldown_terminated_option(termination_reason):
            self.option_cooldowns[int(self.current_option_idx)] = self.option_ttl
        self._prime_executive_release_phase_state(x, termination_reason)
        self._prime_executive_release_latch(x, termination_reason)
        selection_option_logits = option_logits.copy()
        if self.option_termination_cooldown:
            cooldown_mask = self.option_cooldowns > 0
            if np.any(cooldown_mask):
                selection_option_logits[cooldown_mask] = -20.0
        selection_option_logits = self._apply_executive_physiology_option_gating(
            x,
            selection_option_logits,
        )
        selection_option_logits = self._apply_executive_event_release_latch(
            x,
            selection_option_logits,
        )
        option_probs = softmax(selection_option_logits)
        started_new_option = self.current_option_idx < 0 or termination_reason is not None
        if started_new_option:
            selected_option_idx = int(np.argmax(option_probs))
            self.current_option_idx = selected_option_idx
            self.current_option_age = 0
            self.current_option_steps_remaining = self.option_ttl
        else:
            selected_option_idx = int(self.current_option_idx)
            self.current_option_age += 1
        self._apply_executive_release_substate_progression(x, selected_option_idx)
        self.current_option_steps_remaining = max(
            0,
            int(self.current_option_steps_remaining) - 1,
        )
        if self.executive_event_release_latching:
            if (
                OPTION_NAMES[selected_option_idx] == "POST_REST_REACTIVATE"
                and self.executive_release_steps_remaining > 0
            ):
                self.executive_release_steps_remaining = max(
                    0,
                    int(self.executive_release_steps_remaining) - 1,
                )
            elif termination_reason == "shelter_exited":
                self.executive_release_steps_remaining = 0
        if self.executive_post_exit_continuation:
            signals = self._executive_state_signals(x)
            if (
                OPTION_NAMES[selected_option_idx] == "POST_REST_REACTIVATE"
                and self.executive_post_exit_steps_remaining > 0
                and signals["on_shelter"] <= 0.5
                and signals["acute_threat"] < 0.2
            ):
                self.executive_post_exit_steps_remaining = max(
                    0,
                    int(self.executive_post_exit_steps_remaining) - 1,
                )
                self.executive_post_exit_corridor_steps_remaining = max(
                    0,
                    int(self.executive_post_exit_corridor_steps_remaining) - 1,
                )
            elif signals["on_shelter"] > 0.5 or termination_reason == "food_reached":
                self.executive_post_exit_steps_remaining = 0
                self.executive_post_exit_corridor_steps_remaining = 0
        if self.executive_post_food_return:
            signals = self._executive_state_signals(x)
            if (
                self.executive_post_food_return_steps_remaining > 0
                or self.executive_post_food_return_queue
            ) and (
                signals["acute_threat"] >= 0.2
                or termination_reason == "recovery_completed"
            ):
                self.executive_post_food_return_steps_remaining = 0
                self.executive_post_food_return_queue = []
                self.executive_post_food_path_history = []
            elif self.executive_post_food_return_steps_remaining > 0:
                if signals["on_shelter"] > 0.5 and signals["shelter_role_level"] >= 0.95:
                    self.executive_post_food_return_steps_remaining = max(
                        self.executive_post_food_return_steps_remaining,
                        2,
                    )
                else:
                    self.executive_post_food_return_steps_remaining = max(
                        0,
                        int(self.executive_post_food_return_steps_remaining) - 1,
                    )
                if self.executive_post_food_return_steps_remaining <= 0:
                    self.executive_post_food_return_queue = []
                    self.executive_post_food_path_history = []
        if self.option_termination_cooldown:
            self.option_cooldowns = np.maximum(0, self.option_cooldowns - 1)
        option_vector = one_hot(selected_option_idx, self.option_dim)
        selected_option_age_bucket = min(
            max(int(self.current_option_age), 0),
            max(self.option_ttl - 1, 0),
        )
        previous_action_vector = (
            np.zeros(self.output_dim, dtype=float)
            if started_new_option or self.previous_action_idx < 0
            else one_hot(int(self.previous_action_idx), self.output_dim)
        )
        previous_decoder_action_state = (
            np.zeros(self.hidden_dim, dtype=float)
            if started_new_option
            else self.decoder_action_state.copy()
        )
        previous_action_backbone_state = self.action_backbone_state.copy()
        previous_action_policy_state = self.action_policy_state.copy()
        previous_action_controller_state = self.action_controller_state.copy()
        previous_action_token_state = self.action_token_state.copy()
        decoder_hidden_pre = h_new.copy()
        if self.option_decoder_state:
            decoder_hidden_pre = (
                h_new
                + self.W_option_decoder_state[selected_option_idx] @ h_new
                + self.b_option_decoder_state[selected_option_idx]
            )
        if self.option_decoder_recurrent_state:
            decoder_hidden_pre = (
                decoder_hidden_pre
                + self.W_option_decoder_recurrent_state[selected_option_idx]
                @ previous_decoder_action_state
                + self.b_option_decoder_recurrent_state[selected_option_idx]
            )
        if self.option_action_transition_state:
            decoder_hidden_pre = (
                decoder_hidden_pre
                + self.W_option_action_transition_state[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_transition_state[selected_option_idx]
            )
        decoder_hidden = np.tanh(decoder_hidden_pre)
        action_backbone_pre = np.zeros(self.hidden_dim, dtype=float)
        action_backbone_state = np.zeros(self.hidden_dim, dtype=float)
        if self.option_action_separate_backbone:
            action_backbone_pre = (
                self.W_action_backbone_input @ x
                + self.W_action_backbone_prev @ previous_action_backbone_state
                + self.W_action_backbone_action @ previous_action_vector
                + self.b_action_backbone
            )
            action_backbone_state = np.tanh(action_backbone_pre)
        action_policy_source = (
            h_new
            if (
                self.option_action_separate_recurrent_head
                or self.option_action_separate_policy_path
            )
            else decoder_hidden
        )
        action_policy_pre = action_policy_source.copy()
        if self.option_action_separate_policy_path:
            action_policy_pre = (
                self.W_action_policy_path_input @ action_policy_source
                + self.W_action_policy_path_prev @ previous_action_policy_state
                + self.W_action_policy_path_action @ previous_action_vector
                + self.b_action_policy_path
            )
        elif self.option_action_recurrent_core:
            action_policy_pre = (
                self.W_option_action_policy_decoder[selected_option_idx]
                @ action_policy_source
                + self.W_option_action_policy_prev[selected_option_idx]
                @ previous_action_policy_state
                + self.W_option_action_policy_action[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_policy[selected_option_idx]
            )
        action_policy_state = np.tanh(action_policy_pre)
        action_token_pre = decoder_hidden.copy()
        if self.option_action_token_decoder:
            action_token_pre = (
                self.W_option_action_token_decoder[selected_option_idx]
                @ decoder_hidden
                + self.W_option_action_token_prev[selected_option_idx]
                @ previous_action_token_state
                + self.W_option_action_token_action[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_token[selected_option_idx]
            )
        action_token_state = np.tanh(action_token_pre)
        policy_core = (
            action_token_state
            if self.option_action_token_decoder
            else (
                action_backbone_state
                if self.option_action_separate_backbone
                else (
                action_policy_state
                if self.option_action_recurrent_core
                else decoder_hidden
                )
            )
        )
        action_controller_pre = decoder_hidden.copy()
        if self.option_action_controller_state:
            action_controller_pre = (
                self.W_option_action_controller_decoder[selected_option_idx]
                @ decoder_hidden
                + self.W_option_action_controller_prev[selected_option_idx]
                @ previous_action_controller_state
                + self.W_option_action_controller_action[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_controller[selected_option_idx]
            )
        action_controller_state = np.tanh(action_controller_pre)
        policy_logits = np.clip(
            np.nan_to_num(
                (
                    self.W2_action_backbone @ action_backbone_state
                    + self.b2_action_backbone
                    if self.option_action_separate_backbone
                    else
                    (
                    self.W2_action_policy_path @ action_policy_state
                    + self.b2_action_policy_path
                    if self.option_action_separate_policy_path
                    else
                    (
                    self.W2_action_policy_core @ action_policy_state
                    + self.b2_action_policy_core
                    if self.option_action_recurrent_core
                    else self.W2_policy @ policy_core + self.b2_policy
                    )
                    )
                )
                + (
                    0.0
                    if self.option_action_separate_policy_path
                    or self.option_action_separate_backbone
                    else self.option_action_bias[selected_option_idx]
                )
                + (
                    self.W2_option_action_head[selected_option_idx] @ policy_core
                    + self.b2_option_action_head[selected_option_idx]
                    if self.option_action_head
                    else 0.0
                )
                + (
                    self.W2_option_sequence_head[
                        selected_option_idx,
                        selected_option_age_bucket,
                    ]
                    @ policy_core
                    + self.b2_option_sequence_head[
                        selected_option_idx,
                        selected_option_age_bucket,
                    ]
                    if self.option_sequence_head
                    else 0.0
                )
                + (
                    self.W2_option_action_controller_head[selected_option_idx]
                    @ action_controller_state
                    + self.b2_option_action_controller_head[selected_option_idx]
                    if self.option_action_controller_state
                    else 0.0
                )
                + self.W2_policy_feedback @ combined_feedback
                + self.b2_policy_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        policy_logits = self._apply_executive_affordance_action_gating(
            x,
            selected_option_idx,
            policy_logits,
            blocked_probs,
            geometry_probs,
            shelter_position_probs,
        )
        value = float(
            np.clip(
                np.nan_to_num(
                    self.W2_value @ h_new + self.b2_value,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                )[0],
                -1e6,
                1e6,
            )
        )
        self.hidden_state = h_new.copy()
        self.current_option_vector = option_vector.copy()
        self.decoder_action_state = decoder_hidden.copy()
        self.action_backbone_state = action_backbone_state.copy()
        self.action_policy_state = action_policy_state.copy()
        self.action_controller_state = action_controller_state.copy()
        self.action_token_state = action_token_state.copy()
        self.last_option_summary = {
                "selected_option": OPTION_NAMES[selected_option_idx],
                "option_age": int(self.current_option_age),
                "option_termination_reason": (
                    "initial_selection"
                if termination_reason is None and previous_option_vector.sum() <= 0.0
                else ("active" if termination_reason is None else termination_reason)
            ),
            "option_logits": selection_option_logits.round(6).tolist(),
            "option_cooldowns": self.option_cooldowns.astype(int).tolist(),
            "executive_release_steps_remaining": int(
                self.executive_release_steps_remaining
            ),
        }
        self.last_affordance_summary = {
            "blocked_logits": blocked_logits.round(6).tolist(),
            "role_logits": role_logits.round(6).tolist(),
            "geometry_logits": geometry_logits.round(6).tolist(),
            "shelter_position_logits": shelter_position_logits.round(6).tolist(),
            "transition_prediction_logits": transition_prediction_logits.round(6).tolist(),
            "transition_rollout_prediction_logits": transition_rollout_prediction_logits.round(6).tolist(),
        }
        if store_cache:
            self.cache = OptionAffordancePositionFeedbackCache(
                x=x,
                x_aug=x_aug,
                h_prev=h_prev,
                h_new=h_new,
                query_input=query_input,
                query=query,
                slot_raws=slot_raws,
                keys=keys,
                values=values,
                attention_weights=attention_weights,
                valid_event_type_indices=event_type_indices,
                option_probs=option_probs,
                selected_option_idx=selected_option_idx,
                blocked_probs=blocked_probs,
                role_probs=role_probs,
                affordance_features=affordance_features,
                affordance_feedback=affordance_feedback,
                geometry_probs=geometry_probs,
                shelter_position_probs=shelter_position_probs,
                transition_prediction_values=transition_prediction_logits,
                transition_prediction_feedback=transition_prediction_feedback,
                transition_rollout_prediction_values=transition_rollout_prediction_logits,
                transition_rollout_prediction_feedback=transition_rollout_prediction_feedback,
                combined_feedback=combined_feedback,
                phase_probs=phase_probs,
                previous_option_vector=previous_option_vector,
                previous_option_idx=previous_option_idx,
                previous_action_vector=previous_action_vector,
                selected_option_age_bucket=selected_option_age_bucket,
                previous_decoder_action_state=previous_decoder_action_state,
                previous_action_backbone_state=previous_action_backbone_state,
                action_backbone_state=action_backbone_state,
                action_backbone_pre=action_backbone_pre,
                previous_action_policy_state=previous_action_policy_state,
                action_policy_state=action_policy_state,
                action_policy_pre=action_policy_pre,
                previous_action_controller_state=previous_action_controller_state,
                action_controller_state=action_controller_state,
                action_controller_pre=action_controller_pre,
                previous_action_token_state=previous_action_token_state,
                action_token_state=action_token_state,
                action_token_pre=action_token_pre,
                decoder_hidden=decoder_hidden,
                decoder_hidden_pre=decoder_hidden_pre,
            )
        if phase_logits is not None:
            return policy_logits, value, option_logits, phase_logits
        return policy_logits, value, option_logits

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
        grad_option_logits: Array | None = None,
        grad_phase_logits: Array | None = None,
        grad_affordance_blocked_logits: Array | None = None,
        grad_affordance_role_logits: Array | None = None,
        grad_geometry_logits: Array | None = None,
        grad_shelter_position_logits: Array | None = None,
        grad_transition_prediction_logits: Array | None = None,
        grad_transition_rollout_prediction_logits: Array | None = None,
    ) -> Array:
        if not isinstance(self.cache, OptionAffordancePositionFeedbackCache):
            raise RuntimeError(
                "Recurrent option affordance position feedback network backward called without position cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        if grad_option_logits is None:
            grad_option_logits = np.zeros(self.option_dim, dtype=float)
        else:
            grad_option_logits = _clip_grad_logits(
                np.asarray(grad_option_logits, dtype=float),
                grad_clip,
            )
        if grad_phase_logits is None or self.phase_output_dim <= 0:
            grad_phase_logits = np.zeros(self.phase_output_dim, dtype=float)
        else:
            grad_phase_logits = _clip_grad_logits(
                np.asarray(grad_phase_logits, dtype=float),
                grad_clip,
            )
        if grad_affordance_blocked_logits is None:
            grad_affordance_blocked_logits = np.zeros(self.output_dim, dtype=float)
        else:
            grad_affordance_blocked_logits = _clip_grad_logits(
                grad_affordance_blocked_logits,
                grad_clip,
            )
        affordance_role_output_dim = self.output_dim * self.affordance_role_dim
        if grad_affordance_role_logits is None:
            grad_affordance_role_logits = np.zeros(
                affordance_role_output_dim,
                dtype=float,
            )
        else:
            grad_affordance_role_logits = _clip_grad_logits(
                np.asarray(grad_affordance_role_logits, dtype=float),
                grad_clip,
            )
        if grad_geometry_logits is None:
            grad_geometry_logits = np.zeros(self.geometry_feature_dim, dtype=float)
        else:
            grad_geometry_logits = _clip_grad_logits(
                np.asarray(grad_geometry_logits, dtype=float),
                grad_clip,
            )
        shelter_position_output_dim = self.output_dim * self.shelter_position_dim
        if grad_shelter_position_logits is None:
            grad_shelter_position_logits = np.zeros(
                shelter_position_output_dim,
                dtype=float,
            )
        else:
            grad_shelter_position_logits = _clip_grad_logits(
                np.asarray(grad_shelter_position_logits, dtype=float),
                grad_clip,
            )
        if grad_transition_prediction_logits is None or not self.transition_prediction_head:
            grad_transition_prediction_logits = np.zeros(0, dtype=float)
        else:
            grad_transition_prediction_logits = _clip_grad_logits(
                np.asarray(grad_transition_prediction_logits, dtype=float),
                grad_clip,
            )
        if (
            grad_transition_rollout_prediction_logits is None
            or not self.transition_rollout_prediction_head
        ):
            grad_transition_rollout_prediction_logits = np.zeros(0, dtype=float)
        else:
            grad_transition_rollout_prediction_logits = _clip_grad_logits(
                np.asarray(
                    grad_transition_rollout_prediction_logits,
                    dtype=float,
                ),
                grad_clip,
            )
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        decoder_hidden = self.cache.decoder_hidden
        decoder_hidden_pre = self.cache.decoder_hidden_pre
        policy_core = (
            self.cache.action_token_state
            if self.option_action_token_decoder
            else (
                self.cache.action_backbone_state
                if self.option_action_separate_backbone
                else (
                self.cache.action_policy_state
                if self.option_action_recurrent_core
                else decoder_hidden
                )
            )
        )
        affordance_feedback = self.cache.affordance_feedback
        transition_prediction_feedback = self.cache.transition_prediction_feedback
        transition_rollout_prediction_feedback = (
            self.cache.transition_rollout_prediction_feedback
        )
        combined_feedback = self.cache.combined_feedback
        grad_W2_policy = (
            np.outer(grad_policy_logits, policy_core)
            if not self.option_action_recurrent_core
            and not self.option_action_separate_policy_path
            and not self.option_action_separate_backbone
            else np.zeros_like(self.W2_policy)
        )
        grad_b2_policy = (
            grad_policy_logits
            if not self.option_action_recurrent_core
            and not self.option_action_separate_policy_path
            and not self.option_action_separate_backbone
            else np.zeros_like(self.b2_policy)
        )
        grad_W2_action_backbone = (
            np.outer(grad_policy_logits, self.cache.action_backbone_state)
            if self.option_action_separate_backbone
            else np.zeros_like(self.W2_action_backbone)
        )
        grad_b2_action_backbone = (
            grad_policy_logits.copy()
            if self.option_action_separate_backbone
            else np.zeros_like(self.b2_action_backbone)
        )
        grad_W2_action_policy_core = (
            np.outer(grad_policy_logits, self.cache.action_policy_state)
            if self.option_action_recurrent_core
            else np.zeros_like(self.W2_action_policy_core)
        )
        grad_b2_action_policy_core = (
            grad_policy_logits.copy()
            if self.option_action_recurrent_core
            else np.zeros_like(self.b2_action_policy_core)
        )
        grad_W2_action_policy_path = (
            np.outer(grad_policy_logits, self.cache.action_policy_state)
            if self.option_action_separate_policy_path
            else np.zeros_like(self.W2_action_policy_path)
        )
        grad_b2_action_policy_path = (
            grad_policy_logits.copy()
            if self.option_action_separate_policy_path
            else np.zeros_like(self.b2_action_policy_path)
        )
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_option_action_bias = np.zeros_like(self.option_action_bias)
        if (
            not self.option_action_separate_policy_path
            and not self.option_action_separate_backbone
        ):
            grad_option_action_bias[self.cache.selected_option_idx] += grad_policy_logits
        grad_W2_option_action_head = (
            np.zeros_like(self.W2_option_action_head)
            if self.option_action_head
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        grad_b2_option_action_head = (
            np.zeros_like(self.b2_option_action_head)
            if self.option_action_head
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_W2_option_sequence_head = (
            np.zeros_like(self.W2_option_sequence_head)
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim, self.hidden_dim), dtype=float)
        )
        grad_b2_option_sequence_head = (
            np.zeros_like(self.b2_option_sequence_head)
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim), dtype=float)
        )
        if self.option_action_head:
            grad_W2_option_action_head[self.cache.selected_option_idx] = np.outer(
                grad_policy_logits,
                policy_core,
            )
            grad_b2_option_action_head[self.cache.selected_option_idx] = (
                grad_policy_logits
            )
        if self.option_sequence_head:
            grad_W2_option_sequence_head[
                self.cache.selected_option_idx,
                self.cache.selected_option_age_bucket,
            ] = np.outer(
                grad_policy_logits,
                policy_core,
            )
            grad_b2_option_sequence_head[
                self.cache.selected_option_idx,
                self.cache.selected_option_age_bucket,
            ] = grad_policy_logits
        option_target = one_hot(self.cache.selected_option_idx, self.option_dim)
        option_advantage = -grad_value
        grad_option_logits = grad_option_logits + 0.2 * option_advantage * (
            self.cache.option_probs - option_target
        )
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        grad_W2_phase = (
            np.outer(grad_phase_logits, h_new)
            if self.phase_output_dim > 0
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_b2_phase = np.asarray(grad_phase_logits, dtype=float)
        grad_W2_policy_feedback = np.outer(grad_policy_logits, combined_feedback)
        grad_b2_policy_feedback = grad_policy_logits
        grad_W2_option_feedback = np.outer(grad_option_logits, combined_feedback)
        grad_b2_option_feedback = grad_option_logits
        grad_W2_phase_option_feedback = (
            np.outer(grad_option_logits, self.cache.phase_probs)
            if self.phase_option_feedback
            else np.zeros_like(self.W2_phase_option_feedback)
        )
        grad_b2_phase_option_feedback = (
            grad_option_logits.copy()
            if self.phase_option_feedback
            else np.zeros_like(self.b2_phase_option_feedback)
        )
        grad_W2_option_transition_feedback = (
            np.outer(grad_option_logits, self.cache.previous_option_vector)
            if self.option_transition_feedback
            else np.zeros_like(self.W2_option_transition_feedback)
        )
        grad_b2_option_transition_feedback = (
            grad_option_logits.copy()
            if self.option_transition_feedback
            else np.zeros_like(self.b2_option_transition_feedback)
        )
        grad_feedback = (
            self.W2_policy_feedback.T @ grad_policy_logits
            + self.W2_option_feedback.T @ grad_option_logits
        )
        dz_feedback = grad_feedback * (1.0 - affordance_feedback**2)
        grad_W_affordance_feedback = np.outer(
            dz_feedback,
            self.cache.affordance_features,
        )
        grad_b_affordance_feedback = dz_feedback
        grad_affordance_features = self.W_affordance_feedback.T @ dz_feedback
        grad_W_transition_prediction_feedback = (
            np.zeros_like(self.W_transition_prediction_feedback)
            if self.transition_prediction_feedback
            else np.zeros((0, self.transition_prediction_feature_dim), dtype=float)
        )
        grad_b_transition_prediction_feedback = (
            np.zeros_like(self.b_transition_prediction_feedback)
            if self.transition_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        grad_W_transition_rollout_prediction_feedback = (
            np.zeros_like(self.W_transition_rollout_prediction_feedback)
            if self.transition_rollout_prediction_feedback
            else np.zeros(
                (0, self.transition_rollout_prediction_feature_dim),
                dtype=float,
            )
        )
        grad_b_transition_rollout_prediction_feedback = (
            np.zeros_like(self.b_transition_rollout_prediction_feedback)
            if self.transition_rollout_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        grad_transition_prediction_from_feedback = np.zeros(0, dtype=float)
        grad_transition_rollout_prediction_from_feedback = np.zeros(0, dtype=float)
        if self.transition_prediction_feedback:
            grad_transition_prediction_from_feedback = np.zeros(
                self.transition_prediction_feature_dim,
                dtype=float,
            )
            dz_transition_feedback = grad_feedback * (
                1.0 - transition_prediction_feedback**2
            )
            grad_W_transition_prediction_feedback = np.outer(
                dz_transition_feedback,
                self.cache.transition_prediction_values,
            )
            grad_b_transition_prediction_feedback = dz_transition_feedback
            grad_transition_prediction_from_feedback = (
                self.W_transition_prediction_feedback.T @ dz_transition_feedback
            )
        if self.transition_rollout_prediction_feedback:
            grad_transition_rollout_prediction_from_feedback = np.zeros(
                self.transition_rollout_prediction_feature_dim,
                dtype=float,
            )
            dz_transition_rollout_feedback = grad_feedback * (
                1.0 - transition_rollout_prediction_feedback**2
            )
            grad_W_transition_rollout_prediction_feedback = np.outer(
                dz_transition_rollout_feedback,
                self.cache.transition_rollout_prediction_values,
            )
            grad_b_transition_rollout_prediction_feedback = (
                dz_transition_rollout_feedback
            )
            grad_transition_rollout_prediction_from_feedback = (
                self.W_transition_rollout_prediction_feedback.T
                @ dz_transition_rollout_feedback
            )
        grad_blocked_probs = grad_affordance_features[: self.output_dim]
        role_prob_end = self.output_dim + affordance_role_output_dim
        geometry_prob_end = role_prob_end + self.geometry_feature_dim
        grad_role_probs = grad_affordance_features[
            self.output_dim : role_prob_end
        ].reshape(self.output_dim, self.affordance_role_dim)
        grad_geometry_probs = grad_affordance_features[role_prob_end:geometry_prob_end]
        grad_shelter_position_probs = grad_affordance_features[
            geometry_prob_end:
        ].reshape(self.output_dim, self.shelter_position_dim)
        grad_phase_probs = (
            self.W2_phase_option_feedback.T @ grad_option_logits
            if self.phase_option_feedback
            else np.zeros(self.phase_output_dim, dtype=float)
        )
        if self.phase_output_dim > 0 and grad_phase_probs.size > 0:
            phase_probs = self.cache.phase_probs
            grad_phase_logits = grad_phase_logits + phase_probs * (
                grad_phase_probs - float(np.dot(grad_phase_probs, phase_probs))
            )
            grad_W2_phase = np.outer(grad_phase_logits, h_new)
            grad_b2_phase = np.asarray(grad_phase_logits, dtype=float)
        grad_affordance_blocked_logits = np.asarray(
            grad_affordance_blocked_logits,
            dtype=float,
        ) + grad_blocked_probs * (
            self.cache.blocked_probs * (1.0 - self.cache.blocked_probs)
        )
        grad_affordance_role_matrix = np.asarray(
            grad_affordance_role_logits,
            dtype=float,
        ).reshape(self.output_dim, self.affordance_role_dim)
        feedback_role_grad_matrix = np.zeros_like(grad_affordance_role_matrix)
        for action_idx in range(self.output_dim):
            role_probs = self.cache.role_probs[action_idx]
            role_prob_grad = grad_role_probs[action_idx]
            feedback_role_grad_matrix[action_idx] = role_probs * (
                role_prob_grad - float(np.dot(role_prob_grad, role_probs))
            )
        grad_affordance_role_logits = (
            grad_affordance_role_matrix + feedback_role_grad_matrix
        ).reshape(-1)
        grad_geometry_logits = np.asarray(grad_geometry_logits, dtype=float) + (
            grad_geometry_probs * self.cache.geometry_probs * (1.0 - self.cache.geometry_probs)
        )
        grad_shelter_position_matrix = np.asarray(
            grad_shelter_position_logits,
            dtype=float,
        ).reshape(self.output_dim, self.shelter_position_dim)
        feedback_position_grad_matrix = np.zeros_like(grad_shelter_position_matrix)
        for action_idx in range(self.output_dim):
            position_probs = self.cache.shelter_position_probs[action_idx]
            position_prob_grad = grad_shelter_position_probs[action_idx]
            feedback_position_grad_matrix[action_idx] = position_probs * (
                position_prob_grad
                - float(np.dot(position_prob_grad, position_probs))
            )
        grad_shelter_position_logits = (
            grad_shelter_position_matrix + feedback_position_grad_matrix
        ).reshape(-1)
        grad_W2_affordance_blocked = np.outer(
            grad_affordance_blocked_logits,
            h_new,
        )
        grad_b2_affordance_blocked = grad_affordance_blocked_logits
        grad_W2_affordance_role = np.outer(
            grad_affordance_role_logits,
            h_new,
        )
        grad_b2_affordance_role = grad_affordance_role_logits
        grad_W2_geometry = np.outer(grad_geometry_logits, h_new)
        grad_b2_geometry = grad_geometry_logits
        grad_W2_shelter_position = np.outer(grad_shelter_position_logits, h_new)
        grad_b2_shelter_position = grad_shelter_position_logits
        if self.transition_prediction_head:
            grad_transition_prediction_pre = (
                np.asarray(grad_transition_prediction_logits, dtype=float)
                + grad_transition_prediction_from_feedback
            ) * (1.0 - self.cache.transition_prediction_values**2)
        else:
            grad_transition_prediction_pre = np.zeros(0, dtype=float)
        if self.transition_rollout_prediction_head:
            grad_transition_rollout_prediction_pre = (
                np.asarray(
                    grad_transition_rollout_prediction_logits,
                    dtype=float,
                )
                + grad_transition_rollout_prediction_from_feedback
            ) * (1.0 - self.cache.transition_rollout_prediction_values**2)
        else:
            grad_transition_rollout_prediction_pre = np.zeros(0, dtype=float)
        grad_W2_transition_prediction = (
            np.outer(grad_transition_prediction_pre, h_new)
            if self.transition_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_b2_transition_prediction = (
            grad_transition_prediction_pre
            if self.transition_prediction_head
            else np.zeros(0, dtype=float)
        )
        grad_W2_transition_rollout_prediction = (
            np.outer(grad_transition_rollout_prediction_pre, h_new)
            if self.transition_rollout_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_b2_transition_rollout_prediction = (
            grad_transition_rollout_prediction_pre
            if self.transition_rollout_prediction_head
            else np.zeros(0, dtype=float)
        )
        grad_policy_core = (
            self.W2_action_backbone.T @ grad_policy_logits
            if self.option_action_separate_backbone
            else
            (
            self.W2_action_policy_path.T @ grad_policy_logits
            if self.option_action_separate_policy_path
            else
            (
            self.W2_action_policy_core.T @ grad_policy_logits
            if self.option_action_recurrent_core
            else self.W2_policy.T @ grad_policy_logits
            )
            )
        )
        if self.option_action_head:
            grad_policy_core = (
                grad_policy_core
                + self.W2_option_action_head[self.cache.selected_option_idx].T
                @ grad_policy_logits
            )
        if self.option_sequence_head:
            grad_policy_core = (
                grad_policy_core
                + self.W2_option_sequence_head[
                    self.cache.selected_option_idx,
                    self.cache.selected_option_age_bucket,
                ].T
                @ grad_policy_logits
            )
        grad_W_option_action_policy_decoder = (
            np.zeros_like(self.W_option_action_policy_decoder)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_policy_prev = (
            np.zeros_like(self.W_option_action_policy_prev)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_policy_action = (
            np.zeros_like(self.W_option_action_policy_action)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_policy = (
            np.zeros_like(self.b_option_action_policy)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_backbone_input = (
            np.zeros_like(self.W_action_backbone_input)
            if self.option_action_separate_backbone
            else np.zeros((0, self.input_dim), dtype=float)
        )
        grad_W_action_backbone_prev = (
            np.zeros_like(self.W_action_backbone_prev)
            if self.option_action_separate_backbone
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_backbone_action = (
            np.zeros_like(self.W_action_backbone_action)
            if self.option_action_separate_backbone
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_b_action_backbone = (
            np.zeros_like(self.b_action_backbone)
            if self.option_action_separate_backbone
            else np.zeros(0, dtype=float)
        )
        grad_x_from_action_backbone = np.zeros(self.input_dim, dtype=float)
        grad_W_action_policy_path_input = (
            np.zeros_like(self.W_action_policy_path_input)
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_policy_path_prev = (
            np.zeros_like(self.W_action_policy_path_prev)
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_policy_path_action = (
            np.zeros_like(self.W_action_policy_path_action)
            if self.option_action_separate_policy_path
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_b_action_policy_path = (
            np.zeros_like(self.b_action_policy_path)
            if self.option_action_separate_policy_path
            else np.zeros(0, dtype=float)
        )
        grad_h_from_action_policy = np.zeros(self.hidden_dim, dtype=float)
        grad_W_option_action_token_decoder = (
            np.zeros_like(self.W_option_action_token_decoder)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_token_prev = (
            np.zeros_like(self.W_option_action_token_prev)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_token_action = (
            np.zeros_like(self.W_option_action_token_action)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_token = (
            np.zeros_like(self.b_option_action_token)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_action_token_decoder:
            grad_action_token_pre = grad_policy_core * (
                1.0 - self.cache.action_token_state**2
            )
            grad_W_option_action_token_decoder[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_token_pre,
                decoder_hidden,
            )
            grad_W_option_action_token_prev[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_token_pre,
                self.cache.previous_action_token_state,
            )
            grad_W_option_action_token_action[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_token_pre,
                self.cache.previous_action_vector,
            )
            grad_b_option_action_token[
                self.cache.selected_option_idx
            ] = grad_action_token_pre
            grad_decoder_hidden = (
                self.W_option_action_token_decoder[
                    self.cache.selected_option_idx
                ].T
                @ grad_action_token_pre
            )
        elif self.option_action_separate_backbone:
            grad_action_backbone_pre = grad_policy_core * (
                1.0 - self.cache.action_backbone_state**2
            )
            grad_W_action_backbone_input = np.outer(
                grad_action_backbone_pre,
                self.cache.x,
            )
            grad_W_action_backbone_prev = np.outer(
                grad_action_backbone_pre,
                self.cache.previous_action_backbone_state,
            )
            grad_W_action_backbone_action = np.outer(
                grad_action_backbone_pre,
                self.cache.previous_action_vector,
            )
            grad_b_action_backbone = grad_action_backbone_pre
            grad_x_from_action_backbone = (
                self.W_action_backbone_input.T @ grad_action_backbone_pre
            )
            grad_decoder_hidden = np.zeros(self.hidden_dim, dtype=float)
        elif self.option_action_separate_policy_path:
            grad_action_policy_pre = grad_policy_core * (
                1.0 - self.cache.action_policy_state**2
            )
            grad_W_action_policy_path_input = np.outer(
                grad_action_policy_pre,
                h_new,
            )
            grad_W_action_policy_path_prev = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_policy_state,
            )
            grad_W_action_policy_path_action = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_vector,
            )
            grad_b_action_policy_path = grad_action_policy_pre
            grad_h_from_action_policy = (
                self.W_action_policy_path_input.T @ grad_action_policy_pre
            )
            grad_decoder_hidden = np.zeros(self.hidden_dim, dtype=float)
        elif self.option_action_recurrent_core:
            grad_action_policy_pre = grad_policy_core * (
                1.0 - self.cache.action_policy_state**2
            )
            action_policy_source = (
                h_new if self.option_action_separate_recurrent_head else decoder_hidden
            )
            grad_W_option_action_policy_decoder[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_policy_pre,
                action_policy_source,
            )
            grad_W_option_action_policy_prev[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_policy_state,
            )
            grad_W_option_action_policy_action[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_vector,
            )
            grad_b_option_action_policy[
                self.cache.selected_option_idx
            ] = grad_action_policy_pre
            grad_action_policy_source = (
                self.W_option_action_policy_decoder[
                    self.cache.selected_option_idx
                ].T
                @ grad_action_policy_pre
            )
            if self.option_action_separate_recurrent_head:
                grad_h_from_action_policy = grad_action_policy_source
                grad_decoder_hidden = np.zeros(self.hidden_dim, dtype=float)
            else:
                grad_decoder_hidden = grad_action_policy_source
        else:
            grad_decoder_hidden = grad_policy_core
        grad_action_controller_state = (
            self.W2_option_action_controller_head[self.cache.selected_option_idx].T
            @ grad_policy_logits
            if self.option_action_controller_state
            else np.zeros(self.hidden_dim, dtype=float)
        )
        grad_decoder_hidden_pre = grad_decoder_hidden * (1.0 - decoder_hidden**2)
        grad_W2_option_action_controller_head = (
            np.zeros_like(self.W2_option_action_controller_head)
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        grad_b2_option_action_controller_head = (
            np.zeros_like(self.b2_option_action_controller_head)
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_action_controller_pre = np.zeros(self.hidden_dim, dtype=float)
        grad_W_option_action_controller_decoder = (
            np.zeros_like(self.W_option_action_controller_decoder)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_controller_prev = (
            np.zeros_like(self.W_option_action_controller_prev)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_controller_action = (
            np.zeros_like(self.W_option_action_controller_action)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_controller = (
            np.zeros_like(self.b_option_action_controller)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_action_controller_state:
            grad_W2_option_action_controller_head[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_policy_logits,
                self.cache.action_controller_state,
            )
            grad_b2_option_action_controller_head[
                self.cache.selected_option_idx
            ] = grad_policy_logits
            grad_action_controller_pre = (
                grad_action_controller_state
                * (1.0 - self.cache.action_controller_state**2)
            )
            grad_W_option_action_controller_decoder[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_controller_pre,
                self.cache.decoder_hidden,
            )
            grad_W_option_action_controller_prev[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_controller_pre,
                self.cache.previous_action_controller_state,
            )
            grad_W_option_action_controller_action[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_controller_pre,
                self.cache.previous_action_vector,
            )
            grad_b_option_action_controller[
                self.cache.selected_option_idx
            ] = grad_action_controller_pre
            grad_decoder_hidden_pre = (
                grad_decoder_hidden_pre
                + self.W_option_action_controller_decoder[
                    self.cache.selected_option_idx
                ].T
                @ grad_action_controller_pre
            )
        grad_W_option_decoder_recurrent_state = (
            np.zeros_like(self.W_option_decoder_recurrent_state)
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_b_option_decoder_recurrent_state = (
            np.zeros_like(self.b_option_decoder_recurrent_state)
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_decoder_recurrent_state:
            grad_W_option_decoder_recurrent_state[self.cache.selected_option_idx] = (
                np.outer(
                    grad_decoder_hidden_pre,
                    self.cache.previous_decoder_action_state,
                )
            )
            grad_b_option_decoder_recurrent_state[self.cache.selected_option_idx] = (
                grad_decoder_hidden_pre
            )
        grad_W_option_action_transition_state = (
            np.zeros_like(self.W_option_action_transition_state)
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_transition_state = (
            np.zeros_like(self.b_option_action_transition_state)
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_action_transition_state:
            grad_W_option_action_transition_state[self.cache.selected_option_idx] = (
                np.outer(
                    grad_decoder_hidden_pre,
                    self.cache.previous_action_vector,
                )
            )
            grad_b_option_action_transition_state[self.cache.selected_option_idx] = (
                grad_decoder_hidden_pre
            )
        if self.option_decoder_state:
            grad_W_option_decoder_state = np.zeros_like(self.W_option_decoder_state)
            grad_b_option_decoder_state = np.zeros_like(self.b_option_decoder_state)
            grad_W_option_decoder_state[self.cache.selected_option_idx] = np.outer(
                grad_decoder_hidden_pre,
                h_new,
            )
            grad_b_option_decoder_state[self.cache.selected_option_idx] = (
                grad_decoder_hidden_pre
            )
            dh = (
                grad_decoder_hidden_pre
                + self.W_option_decoder_state[self.cache.selected_option_idx].T
                @ grad_decoder_hidden_pre
            )
        else:
            grad_W_option_decoder_state = np.zeros(
                (0, self.hidden_dim, self.hidden_dim),
                dtype=float,
            )
            grad_b_option_decoder_state = np.zeros((0, self.hidden_dim), dtype=float)
            dh = grad_decoder_hidden
        dh = (
            dh
            + self.W2_value.T[:, 0] * grad_value
            + self.W2_option.T @ grad_option_logits
            + grad_h_from_action_policy
            + (
                self.W2_phase.T @ grad_phase_logits
                if self.phase_output_dim > 0
                else 0.0
            )
            + self.W2_affordance_blocked.T @ grad_affordance_blocked_logits
            + self.W2_affordance_role.T @ grad_affordance_role_logits
            + self.W2_geometry.T @ grad_geometry_logits
            + self.W2_shelter_position.T @ grad_shelter_position_logits
            + (
                self.W2_transition_prediction.T @ grad_transition_prediction_pre
                if self.transition_prediction_head
                else 0.0
            )
            + (
                self.W2_transition_rollout_prediction.T
                @ grad_transition_rollout_prediction_pre
                if self.transition_rollout_prediction_head
                else 0.0
            )
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[
            self.input_dim : self.input_dim + self.event_context_dim
        ].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz
        grad_W_option_recurrent_dynamics = (
            np.zeros_like(self.W_option_recurrent_dynamics)
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_b_option_recurrent_dynamics = (
            np.zeros_like(self.b_option_recurrent_dynamics)
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_recurrent_dynamics and self.cache.previous_option_idx >= 0:
            grad_W_option_recurrent_dynamics[self.cache.previous_option_idx] = np.outer(
                dz,
                h_prev,
            )
            grad_b_option_recurrent_dynamics[self.cache.previous_option_idx] = dz
            grad_b_h = grad_b_h
        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)
        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value
                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    : self.event_embedding_dim
                ]
            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query
        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        if self.option_action_recurrent_core:
            self.W2_action_policy_core -= lr * grad_W2_action_policy_core
            self.b2_action_policy_core -= lr * grad_b2_action_policy_core
            self.W_option_action_policy_decoder -= (
                lr * grad_W_option_action_policy_decoder
            )
            self.W_option_action_policy_prev -= lr * grad_W_option_action_policy_prev
            self.W_option_action_policy_action -= (
                lr * grad_W_option_action_policy_action
            )
            self.b_option_action_policy -= lr * grad_b_option_action_policy
        if self.option_action_separate_backbone:
            self.W2_action_backbone -= lr * grad_W2_action_backbone
            self.b2_action_backbone -= lr * grad_b2_action_backbone
            self.W_action_backbone_input -= lr * grad_W_action_backbone_input
            self.W_action_backbone_prev -= lr * grad_W_action_backbone_prev
            self.W_action_backbone_action -= lr * grad_W_action_backbone_action
            self.b_action_backbone -= lr * grad_b_action_backbone
        if self.option_action_separate_policy_path:
            self.W2_action_policy_path -= lr * grad_W2_action_policy_path
            self.b2_action_policy_path -= lr * grad_b2_action_policy_path
            self.W_action_policy_path_input -= lr * grad_W_action_policy_path_input
            self.W_action_policy_path_prev -= lr * grad_W_action_policy_path_prev
            self.W_action_policy_path_action -= lr * grad_W_action_policy_path_action
            self.b_action_policy_path -= lr * grad_b_action_policy_path
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
        if self.phase_output_dim > 0:
            self.W2_phase -= lr * grad_W2_phase
            self.b2_phase -= lr * grad_b2_phase
        self.option_action_bias -= lr * grad_option_action_bias
        self.W2_policy_feedback -= lr * grad_W2_policy_feedback
        self.b2_policy_feedback -= lr * grad_b2_policy_feedback
        self.W2_option_feedback -= lr * grad_W2_option_feedback
        self.b2_option_feedback -= lr * grad_b2_option_feedback
        if self.phase_option_feedback:
            self.W2_phase_option_feedback -= lr * grad_W2_phase_option_feedback
            self.b2_phase_option_feedback -= lr * grad_b2_phase_option_feedback
        if self.option_transition_feedback:
            self.W2_option_transition_feedback -= (
                lr * grad_W2_option_transition_feedback
            )
            self.b2_option_transition_feedback -= (
                lr * grad_b2_option_transition_feedback
            )
        if self.option_action_head:
            self.W2_option_action_head -= lr * grad_W2_option_action_head
            self.b2_option_action_head -= lr * grad_b2_option_action_head
        if self.option_sequence_head:
            self.W2_option_sequence_head -= lr * grad_W2_option_sequence_head
            self.b2_option_sequence_head -= lr * grad_b2_option_sequence_head
        if self.option_action_controller_state:
            self.W2_option_action_controller_head -= (
                lr * grad_W2_option_action_controller_head
            )
            self.b2_option_action_controller_head -= (
                lr * grad_b2_option_action_controller_head
            )
            self.W_option_action_controller_decoder -= (
                lr * grad_W_option_action_controller_decoder
            )
            self.W_option_action_controller_prev -= (
                lr * grad_W_option_action_controller_prev
            )
            self.W_option_action_controller_action -= (
                lr * grad_W_option_action_controller_action
            )
            self.b_option_action_controller -= lr * grad_b_option_action_controller
        if self.option_action_token_decoder:
            self.W_option_action_token_decoder -= lr * grad_W_option_action_token_decoder
            self.W_option_action_token_prev -= lr * grad_W_option_action_token_prev
            self.W_option_action_token_action -= lr * grad_W_option_action_token_action
            self.b_option_action_token -= lr * grad_b_option_action_token
        if self.option_decoder_recurrent_state:
            self.W_option_decoder_recurrent_state -= (
                lr * grad_W_option_decoder_recurrent_state
            )
            self.b_option_decoder_recurrent_state -= (
                lr * grad_b_option_decoder_recurrent_state
            )
        if self.option_action_transition_state:
            self.W_option_action_transition_state -= (
                lr * grad_W_option_action_transition_state
            )
            self.b_option_action_transition_state -= (
                lr * grad_b_option_action_transition_state
            )
        if self.option_decoder_state:
            self.W_option_decoder_state -= lr * grad_W_option_decoder_state
            self.b_option_decoder_state -= lr * grad_b_option_decoder_state
        if self.option_recurrent_dynamics:
            self.W_option_recurrent_dynamics -= lr * grad_W_option_recurrent_dynamics
            self.b_option_recurrent_dynamics -= lr * grad_b_option_recurrent_dynamics
        self.W_affordance_feedback -= lr * grad_W_affordance_feedback
        self.b_affordance_feedback -= lr * grad_b_affordance_feedback
        self.W2_affordance_blocked -= lr * grad_W2_affordance_blocked
        self.b2_affordance_blocked -= lr * grad_b2_affordance_blocked
        self.W2_affordance_role -= lr * grad_W2_affordance_role
        self.b2_affordance_role -= lr * grad_b2_affordance_role
        self.W2_geometry -= lr * grad_W2_geometry
        self.b2_geometry -= lr * grad_b2_geometry
        self.W2_shelter_position -= lr * grad_W2_shelter_position
        self.b2_shelter_position -= lr * grad_b2_shelter_position
        if self.transition_prediction_head:
            self.W2_transition_prediction -= lr * grad_W2_transition_prediction
            self.b2_transition_prediction -= lr * grad_b2_transition_prediction
        if self.transition_prediction_feedback:
            self.W_transition_prediction_feedback -= (
                lr * grad_W_transition_prediction_feedback
            )
            self.b_transition_prediction_feedback -= (
                lr * grad_b_transition_prediction_feedback
            )
        if self.transition_rollout_prediction_head:
            self.W2_transition_rollout_prediction -= (
                lr * grad_W2_transition_rollout_prediction
            )
            self.b2_transition_rollout_prediction -= (
                lr * grad_b2_transition_rollout_prediction
            )
        if self.transition_rollout_prediction_feedback:
            self.W_transition_rollout_prediction_feedback -= (
                lr * grad_W_transition_rollout_prediction_feedback
            )
            self.b_transition_rollout_prediction_feedback -= (
                lr * grad_b_transition_rollout_prediction_feedback
            )
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        grad_x = grad_x + grad_x_from_action_backbone
        return grad_x

    def state_dict(self) -> dict[str, object]:
        state = super().state_dict()
        state["phase_output_dim"] = self.phase_output_dim
        state["phase_option_feedback"] = self.phase_option_feedback
        state["option_transition_feedback"] = self.option_transition_feedback
        state["option_termination_cooldown"] = self.option_termination_cooldown
        state["option_action_head"] = self.option_action_head
        state["option_decoder_state"] = self.option_decoder_state
        state["option_recurrent_dynamics"] = self.option_recurrent_dynamics
        state["option_sequence_head"] = self.option_sequence_head
        state["option_decoder_recurrent_state"] = self.option_decoder_recurrent_state
        state["option_action_transition_state"] = self.option_action_transition_state
        state["option_action_controller_state"] = self.option_action_controller_state
        state["option_action_token_decoder"] = self.option_action_token_decoder
        state["option_action_recurrent_core"] = self.option_action_recurrent_core
        state["option_action_separate_recurrent_head"] = (
            self.option_action_separate_recurrent_head
        )
        state["option_action_separate_policy_path"] = (
            self.option_action_separate_policy_path
        )
        state["option_action_separate_backbone"] = (
            self.option_action_separate_backbone
        )
        state["transition_prediction_head"] = self.transition_prediction_head
        state["transition_prediction_feedback"] = (
            self.transition_prediction_feedback
        )
        state["transition_rollout_prediction_head"] = (
            self.transition_rollout_prediction_head
        )
        state["transition_rollout_prediction_feedback"] = (
            self.transition_rollout_prediction_feedback
        )
        state["shelter_position_head"] = True
        state["shelter_position_dim"] = self.shelter_position_dim
        state["shelter_position_feature_dim"] = self.shelter_position_feature_dim
        state["transition_prediction_feature_dim"] = (
            self.transition_prediction_feature_dim
        )
        state["transition_rollout_prediction_feature_dim"] = (
            self.transition_rollout_prediction_feature_dim
        )
        state["W2_shelter_position"] = self.W2_shelter_position.copy()
        state["b2_shelter_position"] = self.b2_shelter_position.copy()
        if self.transition_prediction_head:
            state["W2_transition_prediction"] = (
                self.W2_transition_prediction.copy()
            )
            state["b2_transition_prediction"] = (
                self.b2_transition_prediction.copy()
            )
        if self.transition_prediction_feedback:
            state["W_transition_prediction_feedback"] = (
                self.W_transition_prediction_feedback.copy()
            )
            state["b_transition_prediction_feedback"] = (
                self.b_transition_prediction_feedback.copy()
            )
        if self.transition_rollout_prediction_head:
            state["W2_transition_rollout_prediction"] = (
                self.W2_transition_rollout_prediction.copy()
            )
            state["b2_transition_rollout_prediction"] = (
                self.b2_transition_rollout_prediction.copy()
            )
        if self.transition_rollout_prediction_feedback:
            state["W_transition_rollout_prediction_feedback"] = (
                self.W_transition_rollout_prediction_feedback.copy()
            )
            state["b_transition_rollout_prediction_feedback"] = (
                self.b_transition_rollout_prediction_feedback.copy()
            )
        if self.phase_output_dim > 0:
            state["W2_phase"] = self.W2_phase.copy()
            state["b2_phase"] = self.b2_phase.copy()
        if self.phase_option_feedback:
            state["W2_phase_option_feedback"] = self.W2_phase_option_feedback.copy()
            state["b2_phase_option_feedback"] = self.b2_phase_option_feedback.copy()
        if self.option_transition_feedback:
            state["W2_option_transition_feedback"] = (
                self.W2_option_transition_feedback.copy()
            )
            state["b2_option_transition_feedback"] = (
                self.b2_option_transition_feedback.copy()
            )
        if self.option_action_head:
            state["W2_option_action_head"] = self.W2_option_action_head.copy()
            state["b2_option_action_head"] = self.b2_option_action_head.copy()
        if self.option_decoder_state:
            state["W_option_decoder_state"] = self.W_option_decoder_state.copy()
            state["b_option_decoder_state"] = self.b_option_decoder_state.copy()
        if self.option_recurrent_dynamics:
            state["W_option_recurrent_dynamics"] = (
                self.W_option_recurrent_dynamics.copy()
            )
            state["b_option_recurrent_dynamics"] = (
                self.b_option_recurrent_dynamics.copy()
            )
        if self.option_sequence_head:
            state["W2_option_sequence_head"] = self.W2_option_sequence_head.copy()
            state["b2_option_sequence_head"] = self.b2_option_sequence_head.copy()
        if self.option_decoder_recurrent_state:
            state["W_option_decoder_recurrent_state"] = (
                self.W_option_decoder_recurrent_state.copy()
            )
            state["b_option_decoder_recurrent_state"] = (
                self.b_option_decoder_recurrent_state.copy()
            )
        if self.option_action_transition_state:
            state["W_option_action_transition_state"] = (
                self.W_option_action_transition_state.copy()
            )
            state["b_option_action_transition_state"] = (
                self.b_option_action_transition_state.copy()
            )
        if self.option_action_controller_state:
            state["W_option_action_controller_decoder"] = (
                self.W_option_action_controller_decoder.copy()
            )
            state["W_option_action_controller_prev"] = (
                self.W_option_action_controller_prev.copy()
            )
            state["W_option_action_controller_action"] = (
                self.W_option_action_controller_action.copy()
            )
            state["b_option_action_controller"] = (
                self.b_option_action_controller.copy()
            )
            state["W2_option_action_controller_head"] = (
                self.W2_option_action_controller_head.copy()
            )
            state["b2_option_action_controller_head"] = (
                self.b2_option_action_controller_head.copy()
            )
        if self.option_action_token_decoder:
            state["W_option_action_token_decoder"] = (
                self.W_option_action_token_decoder.copy()
            )
            state["W_option_action_token_prev"] = (
                self.W_option_action_token_prev.copy()
            )
            state["W_option_action_token_action"] = (
                self.W_option_action_token_action.copy()
            )
            state["b_option_action_token"] = self.b_option_action_token.copy()
        if self.option_action_recurrent_core:
            state["W_option_action_policy_decoder"] = (
                self.W_option_action_policy_decoder.copy()
            )
            state["W_option_action_policy_prev"] = (
                self.W_option_action_policy_prev.copy()
            )
            state["W_option_action_policy_action"] = (
                self.W_option_action_policy_action.copy()
            )
            state["b_option_action_policy"] = self.b_option_action_policy.copy()
            state["W2_action_policy_core"] = self.W2_action_policy_core.copy()
            state["b2_action_policy_core"] = self.b2_action_policy_core.copy()
        if self.option_action_separate_policy_path:
            state["W_action_policy_path_input"] = (
                self.W_action_policy_path_input.copy()
            )
            state["W_action_policy_path_prev"] = (
                self.W_action_policy_path_prev.copy()
            )
            state["W_action_policy_path_action"] = (
                self.W_action_policy_path_action.copy()
            )
            state["b_action_policy_path"] = self.b_action_policy_path.copy()
            state["W2_action_policy_path"] = self.W2_action_policy_path.copy()
            state["b2_action_policy_path"] = self.b2_action_policy_path.copy()
        if self.option_action_separate_backbone:
            state["W_action_backbone_input"] = self.W_action_backbone_input.copy()
            state["W_action_backbone_prev"] = self.W_action_backbone_prev.copy()
            state["W_action_backbone_action"] = self.W_action_backbone_action.copy()
            state["b_action_backbone"] = self.b_action_backbone.copy()
            state["W2_action_backbone"] = self.W2_action_backbone.copy()
            state["b2_action_backbone"] = self.b2_action_backbone.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        _validate_state_dict(
            state,
            expected_keys=(
                set(super().state_dict().keys())
                | {
                    "phase_output_dim",
                    "phase_option_feedback",
                    "option_transition_feedback",
                    "option_termination_cooldown",
                    "option_action_head",
                    "option_decoder_state",
                    "option_recurrent_dynamics",
                    "option_sequence_head",
                    "option_decoder_recurrent_state",
                    "option_action_transition_state",
                    "option_action_controller_state",
                    "option_action_token_decoder",
                    "option_action_recurrent_core",
                    "option_action_separate_recurrent_head",
                    "option_action_separate_policy_path",
                    "option_action_separate_backbone",
                    "transition_prediction_head",
                    "transition_prediction_feedback",
                    "transition_rollout_prediction_head",
                    "transition_rollout_prediction_feedback",
                    "shelter_position_head",
                    "shelter_position_dim",
                    "shelter_position_feature_dim",
                    "transition_prediction_feature_dim",
                    "transition_rollout_prediction_feature_dim",
                    "W2_shelter_position",
                    "b2_shelter_position",
                }
                | (
                    {
                        "W2_transition_prediction",
                        "b2_transition_prediction",
                    }
                    if self.transition_prediction_head
                    else set()
                )
                | (
                    {
                        "W_transition_prediction_feedback",
                        "b_transition_prediction_feedback",
                    }
                    if self.transition_prediction_feedback
                    else set()
                )
                | (
                    {
                        "W2_transition_rollout_prediction",
                        "b2_transition_rollout_prediction",
                    }
                    if self.transition_rollout_prediction_head
                    else set()
                )
                | (
                    {
                        "W_transition_rollout_prediction_feedback",
                        "b_transition_rollout_prediction_feedback",
                    }
                    if self.transition_rollout_prediction_feedback
                    else set()
                )
                | (
                    {"W2_phase", "b2_phase"}
                    if self.phase_output_dim > 0
                    else set()
                )
                | (
                    {"W2_phase_option_feedback", "b2_phase_option_feedback"}
                    if self.phase_option_feedback
                    else set()
                )
                | (
                    {
                        "W2_option_transition_feedback",
                        "b2_option_transition_feedback",
                    }
                    if self.option_transition_feedback
                    else set()
                )
                | (
                    {
                        "W2_option_action_head",
                        "b2_option_action_head",
                    }
                    if self.option_action_head
                    else set()
                )
                | (
                    {
                        "W_option_decoder_state",
                        "b_option_decoder_state",
                    }
                    if self.option_decoder_state
                    else set()
                )
                | (
                    {
                        "W_option_recurrent_dynamics",
                        "b_option_recurrent_dynamics",
                    }
                    if self.option_recurrent_dynamics
                    else set()
                )
                | (
                    {
                        "W2_option_sequence_head",
                        "b2_option_sequence_head",
                    }
                    if self.option_sequence_head
                    else set()
                )
                | (
                    {
                        "W_option_decoder_recurrent_state",
                        "b_option_decoder_recurrent_state",
                    }
                    if self.option_decoder_recurrent_state
                    else set()
                )
                | (
                    {
                        "W_option_action_transition_state",
                        "b_option_action_transition_state",
                    }
                    if self.option_action_transition_state
                    else set()
                )
                | (
                    {
                        "W_option_action_controller_decoder",
                        "W_option_action_controller_prev",
                        "W_option_action_controller_action",
                        "b_option_action_controller",
                        "W2_option_action_controller_head",
                        "b2_option_action_controller_head",
                    }
                    if self.option_action_controller_state
                    else set()
                )
                | (
                    {
                        "W_option_action_token_decoder",
                        "W_option_action_token_prev",
                        "W_option_action_token_action",
                        "b_option_action_token",
                    }
                    if self.option_action_token_decoder
                    else set()
                )
                | (
                    {
                        "W_option_action_policy_decoder",
                        "W_option_action_policy_prev",
                        "W_option_action_policy_action",
                        "b_option_action_policy",
                        "W2_action_policy_core",
                        "b2_action_policy_core",
                    }
                    if self.option_action_recurrent_core
                    else set()
                )
                | (
                    {
                        "W_action_policy_path_input",
                        "W_action_policy_path_prev",
                        "W_action_policy_path_action",
                        "b_action_policy_path",
                        "W2_action_policy_path",
                        "b2_action_policy_path",
                    }
                    if self.option_action_separate_policy_path
                    else set()
                )
                | (
                    {
                        "W_action_backbone_input",
                        "W_action_backbone_prev",
                        "W_action_backbone_action",
                        "b_action_backbone",
                        "W2_action_backbone",
                        "b2_action_backbone",
                    }
                    if self.option_action_separate_backbone
                    else set()
                )
            ),
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "recurrent": True,
                "phase_output_dim": self.phase_output_dim,
                "event_attention": True,
                "event_buffer_size": self.event_buffer_size,
                "event_embedding_dim": self.event_embedding_dim,
                "event_context_dim": self.event_context_dim,
                "event_feature_dim": self.event_feature_dim,
                "option_head": True,
                "option_ttl": self.option_ttl,
                "option_dim": self.option_dim,
                "phase_option_feedback": self.phase_option_feedback,
                "option_transition_feedback": self.option_transition_feedback,
                "option_termination_cooldown": self.option_termination_cooldown,
                "option_action_head": self.option_action_head,
                "option_decoder_state": self.option_decoder_state,
                "option_recurrent_dynamics": self.option_recurrent_dynamics,
                "option_sequence_head": self.option_sequence_head,
                "option_decoder_recurrent_state": (
                    self.option_decoder_recurrent_state
                ),
                "option_action_transition_state": (
                    self.option_action_transition_state
                ),
                "option_action_controller_state": (
                    self.option_action_controller_state
                ),
                "option_action_token_decoder": (
                    self.option_action_token_decoder
                ),
                "option_action_recurrent_core": (
                    self.option_action_recurrent_core
                ),
                "option_action_separate_recurrent_head": (
                    self.option_action_separate_recurrent_head
                ),
                "option_action_separate_policy_path": (
                    self.option_action_separate_policy_path
                ),
                "option_action_separate_backbone": (
                    self.option_action_separate_backbone
                ),
                "transition_prediction_head": self.transition_prediction_head,
                "transition_prediction_feedback": (
                    self.transition_prediction_feedback
                ),
                "transition_rollout_prediction_head": (
                    self.transition_rollout_prediction_head
                ),
                "transition_rollout_prediction_feedback": (
                    self.transition_rollout_prediction_feedback
                ),
                "affordance_head": True,
                "affordance_role_dim": self.affordance_role_dim,
                "affordance_feedback": True,
                "geometry_head": True,
                "geometry_dim": self.geometry_dim,
                "geometry_feature_dim": self.geometry_feature_dim,
                "shelter_position_head": True,
                "shelter_position_dim": self.shelter_position_dim,
                "shelter_position_feature_dim": self.shelter_position_feature_dim,
                "transition_prediction_feature_dim": (
                    self.transition_prediction_feature_dim
                ),
                "transition_rollout_prediction_feature_dim": (
                    self.transition_rollout_prediction_feature_dim
                ),
            },
            name=self.name,
        )
        super().load_state_dict(
            {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "phase_output_dim",
                    "phase_option_feedback",
                    "option_transition_feedback",
                    "option_termination_cooldown",
                    "option_action_head",
                    "option_decoder_state",
                    "option_recurrent_dynamics",
                    "option_sequence_head",
                    "option_decoder_recurrent_state",
                    "option_action_transition_state",
                    "option_action_controller_state",
                    "option_action_token_decoder",
                    "option_action_recurrent_core",
                    "option_action_separate_recurrent_head",
                    "option_action_separate_policy_path",
                    "option_action_separate_backbone",
                    "transition_prediction_head",
                    "transition_prediction_feedback",
                    "transition_rollout_prediction_head",
                    "transition_rollout_prediction_feedback",
                    "shelter_position_head",
                    "shelter_position_dim",
                    "shelter_position_feature_dim",
                    "transition_prediction_feature_dim",
                    "transition_rollout_prediction_feature_dim",
                    "W2_shelter_position",
                    "b2_shelter_position",
                    "W2_transition_prediction",
                    "b2_transition_prediction",
                    "W_transition_prediction_feedback",
                    "b_transition_prediction_feedback",
                    "W2_transition_rollout_prediction",
                    "b2_transition_rollout_prediction",
                    "W_transition_rollout_prediction_feedback",
                    "b_transition_rollout_prediction_feedback",
                    "W2_phase",
                    "b2_phase",
                    "W2_phase_option_feedback",
                    "b2_phase_option_feedback",
                    "W2_option_transition_feedback",
                    "b2_option_transition_feedback",
                    "W2_option_action_head",
                    "b2_option_action_head",
                    "W_option_decoder_state",
                    "b_option_decoder_state",
                    "W_option_recurrent_dynamics",
                    "b_option_recurrent_dynamics",
                    "W2_option_sequence_head",
                    "b2_option_sequence_head",
                    "W_option_decoder_recurrent_state",
                    "b_option_decoder_recurrent_state",
                    "W_option_action_transition_state",
                    "b_option_action_transition_state",
                    "W_option_action_controller_decoder",
                    "W_option_action_controller_prev",
                    "W_option_action_controller_action",
                    "b_option_action_controller",
                    "W2_option_action_controller_head",
                    "b2_option_action_controller_head",
                    "W_option_action_token_decoder",
                    "W_option_action_token_prev",
                    "W_option_action_token_action",
                    "b_option_action_token",
                    "W_option_action_policy_decoder",
                    "W_option_action_policy_prev",
                    "W_option_action_policy_action",
                    "b_option_action_policy",
                    "W2_action_policy_core",
                    "b2_action_policy_core",
                    "W_action_policy_path_input",
                    "W_action_policy_path_prev",
                    "W_action_policy_path_action",
                    "b_action_policy_path",
                    "W2_action_policy_path",
                    "b2_action_policy_path",
                    "W_action_backbone_input",
                    "W_action_backbone_prev",
                    "W_action_backbone_action",
                    "b_action_backbone",
                    "W2_action_backbone",
                    "b2_action_backbone",
                }
            }
        )
        self.W2_shelter_position = _coerce_state_array(
            state,
            "W2_shelter_position",
            (self.shelter_position_feature_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_shelter_position = _coerce_state_array(
            state,
            "b2_shelter_position",
            (self.shelter_position_feature_dim,),
            name=self.name,
        )
        if self.transition_prediction_head:
            self.W2_transition_prediction = _coerce_state_array(
                state,
                "W2_transition_prediction",
                (self.transition_prediction_feature_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_transition_prediction = _coerce_state_array(
                state,
                "b2_transition_prediction",
                (self.transition_prediction_feature_dim,),
                name=self.name,
            )
        if self.transition_prediction_feedback:
            self.W_transition_prediction_feedback = _coerce_state_array(
                state,
                "W_transition_prediction_feedback",
                (self.hidden_dim, self.transition_prediction_feature_dim),
                name=self.name,
            )
            self.b_transition_prediction_feedback = _coerce_state_array(
                state,
                "b_transition_prediction_feedback",
                (self.hidden_dim,),
                name=self.name,
            )
        if self.transition_rollout_prediction_head:
            self.W2_transition_rollout_prediction = _coerce_state_array(
                state,
                "W2_transition_rollout_prediction",
                (
                    self.transition_rollout_prediction_feature_dim,
                    self.hidden_dim,
                ),
                name=self.name,
            )
            self.b2_transition_rollout_prediction = _coerce_state_array(
                state,
                "b2_transition_rollout_prediction",
                (self.transition_rollout_prediction_feature_dim,),
                name=self.name,
            )
        if self.transition_rollout_prediction_feedback:
            self.W_transition_rollout_prediction_feedback = _coerce_state_array(
                state,
                "W_transition_rollout_prediction_feedback",
                (
                    self.hidden_dim,
                    self.transition_rollout_prediction_feature_dim,
                ),
                name=self.name,
            )
            self.b_transition_rollout_prediction_feedback = _coerce_state_array(
                state,
                "b_transition_rollout_prediction_feedback",
                (self.hidden_dim,),
                name=self.name,
            )
        if self.phase_output_dim > 0:
            self.W2_phase = _coerce_state_array(
                state,
                "W2_phase",
                (self.phase_output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_phase = _coerce_state_array(
                state,
                "b2_phase",
                (self.phase_output_dim,),
                name=self.name,
            )
        if self.phase_option_feedback:
            self.W2_phase_option_feedback = _coerce_state_array(
                state,
                "W2_phase_option_feedback",
                (self.option_dim, self.phase_output_dim),
                name=self.name,
            )
            self.b2_phase_option_feedback = _coerce_state_array(
                state,
                "b2_phase_option_feedback",
                (self.option_dim,),
                name=self.name,
            )
        if self.option_transition_feedback:
            self.W2_option_transition_feedback = _coerce_state_array(
                state,
                "W2_option_transition_feedback",
                (self.option_dim, self.option_dim),
                name=self.name,
            )
            self.b2_option_transition_feedback = _coerce_state_array(
                state,
                "b2_option_transition_feedback",
                (self.option_dim,),
                name=self.name,
            )
        if self.option_action_head:
            self.W2_option_action_head = _coerce_state_array(
                state,
                "W2_option_action_head",
                (self.option_dim, self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_option_action_head = _coerce_state_array(
                state,
                "b2_option_action_head",
                (self.option_dim, self.output_dim),
                name=self.name,
            )
        if self.option_decoder_state:
            self.W_option_decoder_state = _coerce_state_array(
                state,
                "W_option_decoder_state",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.b_option_decoder_state = _coerce_state_array(
                state,
                "b_option_decoder_state",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_recurrent_dynamics:
            self.W_option_recurrent_dynamics = _coerce_state_array(
                state,
                "W_option_recurrent_dynamics",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.b_option_recurrent_dynamics = _coerce_state_array(
                state,
                "b_option_recurrent_dynamics",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_sequence_head:
            self.W2_option_sequence_head = _coerce_state_array(
                state,
                "W2_option_sequence_head",
                (self.option_dim, self.option_ttl, self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_option_sequence_head = _coerce_state_array(
                state,
                "b2_option_sequence_head",
                (self.option_dim, self.option_ttl, self.output_dim),
                name=self.name,
            )
        if self.option_decoder_recurrent_state:
            self.W_option_decoder_recurrent_state = _coerce_state_array(
                state,
                "W_option_decoder_recurrent_state",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.b_option_decoder_recurrent_state = _coerce_state_array(
                state,
                "b_option_decoder_recurrent_state",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_action_transition_state:
            self.W_option_action_transition_state = _coerce_state_array(
                state,
                "W_option_action_transition_state",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_transition_state = _coerce_state_array(
                state,
                "b_option_action_transition_state",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_action_controller_state:
            self.W_option_action_controller_decoder = _coerce_state_array(
                state,
                "W_option_action_controller_decoder",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_controller_prev = _coerce_state_array(
                state,
                "W_option_action_controller_prev",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_controller_action = _coerce_state_array(
                state,
                "W_option_action_controller_action",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_controller = _coerce_state_array(
                state,
                "b_option_action_controller",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
            self.W2_option_action_controller_head = _coerce_state_array(
                state,
                "W2_option_action_controller_head",
                (self.option_dim, self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_option_action_controller_head = _coerce_state_array(
                state,
                "b2_option_action_controller_head",
                (self.option_dim, self.output_dim),
                name=self.name,
            )
        if self.option_action_token_decoder:
            self.W_option_action_token_decoder = _coerce_state_array(
                state,
                "W_option_action_token_decoder",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_token_prev = _coerce_state_array(
                state,
                "W_option_action_token_prev",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_token_action = _coerce_state_array(
                state,
                "W_option_action_token_action",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_token = _coerce_state_array(
                state,
                "b_option_action_token",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
        if self.option_action_recurrent_core:
            self.W_option_action_policy_decoder = _coerce_state_array(
                state,
                "W_option_action_policy_decoder",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_policy_prev = _coerce_state_array(
                state,
                "W_option_action_policy_prev",
                (self.option_dim, self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_option_action_policy_action = _coerce_state_array(
                state,
                "W_option_action_policy_action",
                (self.option_dim, self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_option_action_policy = _coerce_state_array(
                state,
                "b_option_action_policy",
                (self.option_dim, self.hidden_dim),
                name=self.name,
            )
            self.W2_action_policy_core = _coerce_state_array(
                state,
                "W2_action_policy_core",
                (self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_action_policy_core = _coerce_state_array(
                state,
                "b2_action_policy_core",
                (self.output_dim,),
                name=self.name,
            )
        if self.option_action_separate_policy_path:
            self.W_action_policy_path_input = _coerce_state_array(
                state,
                "W_action_policy_path_input",
                (self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_action_policy_path_prev = _coerce_state_array(
                state,
                "W_action_policy_path_prev",
                (self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_action_policy_path_action = _coerce_state_array(
                state,
                "W_action_policy_path_action",
                (self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_action_policy_path = _coerce_state_array(
                state,
                "b_action_policy_path",
                (self.hidden_dim,),
                name=self.name,
            )
            self.W2_action_policy_path = _coerce_state_array(
                state,
                "W2_action_policy_path",
                (self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_action_policy_path = _coerce_state_array(
                state,
                "b2_action_policy_path",
                (self.output_dim,),
                name=self.name,
            )
        if self.option_action_separate_backbone:
            self.W_action_backbone_input = _coerce_state_array(
                state,
                "W_action_backbone_input",
                (self.hidden_dim, self.input_dim),
                name=self.name,
            )
            self.W_action_backbone_prev = _coerce_state_array(
                state,
                "W_action_backbone_prev",
                (self.hidden_dim, self.hidden_dim),
                name=self.name,
            )
            self.W_action_backbone_action = _coerce_state_array(
                state,
                "W_action_backbone_action",
                (self.hidden_dim, self.output_dim),
                name=self.name,
            )
            self.b_action_backbone = _coerce_state_array(
                state,
                "b_action_backbone",
                (self.hidden_dim,),
                name=self.name,
            )
            self.W2_action_backbone = _coerce_state_array(
                state,
                "W2_action_backbone",
                (self.output_dim, self.hidden_dim),
                name=self.name,
            )
            self.b2_action_backbone = _coerce_state_array(
                state,
                "b2_action_backbone",
                (self.output_dim,),
                name=self.name,
            )
        self.last_affordance_summary["shelter_position_logits"] = []

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            self.W_xh,
            self.W_hh,
            self.b_h,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
            self.W2_option,
            self.b2_option,
            self.W2_phase,
            self.b2_phase,
            self.option_action_bias,
            self.W2_affordance_blocked,
            self.b2_affordance_blocked,
            self.W2_affordance_role,
            self.b2_affordance_role,
            self.W2_geometry,
            self.b2_geometry,
            self.W2_shelter_position,
            self.b2_shelter_position,
            self.W2_transition_prediction,
            self.b2_transition_prediction,
            self.W_transition_prediction_feedback,
            self.b_transition_prediction_feedback,
            self.W_affordance_feedback,
            self.b_affordance_feedback,
            self.W2_policy_feedback,
            self.b2_policy_feedback,
            self.W2_option_feedback,
            self.b2_option_feedback,
            self.W2_phase_option_feedback,
            self.b2_phase_option_feedback,
            self.W2_option_transition_feedback,
            self.b2_option_transition_feedback,
            self.W2_option_action_head,
            self.b2_option_action_head,
            self.W_option_decoder_state,
            self.b_option_decoder_state,
            self.W_option_recurrent_dynamics,
            self.b_option_recurrent_dynamics,
            self.W2_option_sequence_head,
            self.b2_option_sequence_head,
            self.W_option_decoder_recurrent_state,
            self.b_option_decoder_recurrent_state,
            self.W_option_action_transition_state,
            self.b_option_action_transition_state,
            self.W_option_action_controller_decoder,
            self.W_option_action_controller_prev,
            self.W_option_action_controller_action,
            self.b_option_action_controller,
            self.W2_option_action_controller_head,
            self.b2_option_action_controller_head,
            self.W_option_action_token_decoder,
            self.W_option_action_token_prev,
            self.W_option_action_token_action,
            self.b_option_action_token,
            self.W_option_action_policy_decoder,
            self.W_option_action_policy_prev,
            self.W_option_action_policy_action,
            self.b_option_action_policy,
            self.W2_action_policy_core,
            self.b2_action_policy_core,
            self.W_action_backbone_input,
            self.W_action_backbone_prev,
            self.W_action_backbone_action,
            self.b_action_backbone,
            self.W2_action_backbone,
            self.b2_action_backbone,
            self.W_action_policy_path_input,
            self.W_action_policy_path_prev,
            self.W_action_policy_path_action,
            self.b_action_policy_path,
            self.W2_action_policy_path,
            self.b2_action_policy_path,
            self.W_query,
            self.b_query,
            self.W_key,
            self.b_key,
            self.W_value,
            self.b_value,
            self.event_type_embeddings,
        )

    def count_parameters(self) -> int:
        return int(
            super().count_parameters()
            + self.W2_shelter_position.size
            + self.b2_shelter_position.size
            + self.W2_transition_prediction.size
            + self.b2_transition_prediction.size
            + self.W_transition_prediction_feedback.size
            + self.b_transition_prediction_feedback.size
            + self.W2_phase.size
            + self.b2_phase.size
            + self.W2_phase_option_feedback.size
            + self.b2_phase_option_feedback.size
            + self.W2_option_transition_feedback.size
            + self.b2_option_transition_feedback.size
            + self.W2_option_action_head.size
            + self.b2_option_action_head.size
            + self.W_option_decoder_state.size
            + self.b_option_decoder_state.size
            + self.W_option_recurrent_dynamics.size
            + self.b_option_recurrent_dynamics.size
            + self.W2_option_sequence_head.size
            + self.b2_option_sequence_head.size
            + self.W_option_decoder_recurrent_state.size
            + self.b_option_decoder_recurrent_state.size
            + self.W_option_action_transition_state.size
            + self.b_option_action_transition_state.size
            + self.W_option_action_controller_decoder.size
            + self.W_option_action_controller_prev.size
            + self.W_option_action_controller_action.size
            + self.b_option_action_controller.size
            + self.W2_option_action_controller_head.size
            + self.b2_option_action_controller_head.size
            + self.W_option_action_token_decoder.size
            + self.W_option_action_token_prev.size
            + self.W_option_action_token_action.size
            + self.b_option_action_token.size
            + self.W_option_action_policy_decoder.size
            + self.W_option_action_policy_prev.size
            + self.W_option_action_policy_action.size
            + self.b_option_action_policy.size
            + self.W2_action_policy_core.size
            + self.b2_action_policy_core.size
            + self.W_action_backbone_input.size
            + self.W_action_backbone_prev.size
            + self.W_action_backbone_action.size
            + self.b_action_backbone.size
            + self.W2_action_backbone.size
            + self.b2_action_backbone.size
            + self.W_action_policy_path_input.size
            + self.W_action_policy_path_prev.size
            + self.W_action_policy_path_action.size
            + self.b_action_policy_path.size
            + self.W2_action_policy_path.size
            + self.b2_action_policy_path.size
        )


class DeepTrueMonolithicNetwork:
    """Direct policy+value MLP with configurable hidden sizes for diagnostic control."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        output_dim: int,
        rng: np.random.Generator,
        name: str = "true_monolithic_policy",
    ) -> None:
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain only positive integers.")
        self.hidden_dim = int(self.hidden_sizes[-1])
        self.output_dim = int(output_dim)
        self.name = name
        self.hidden_weights = []
        self.hidden_biases = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_sizes:
            self.hidden_weights.append(
                rng.normal(0.0, _weight_scale(prev_dim), size=(hidden_dim, prev_dim))
            )
            self.hidden_biases.append(np.zeros(hidden_dim, dtype=float))
            prev_dim = hidden_dim
        self.W2_policy = rng.normal(
            0.0,
            _weight_scale(self.hidden_dim),
            size=(self.output_dim, self.hidden_dim),
        )
        self.b2_policy = np.zeros(self.output_dim, dtype=float)
        self.W2_value = rng.normal(
            0.0,
            _weight_scale(self.hidden_dim),
            size=(1, self.hidden_dim),
        )
        self.b2_value = np.zeros(1, dtype=float)
        self.cache: Optional[DeepMotorCache] = None

    def forward(self, x: Array, *, store_cache: bool = True) -> tuple[Array, float]:
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        hidden_states: list[Array] = []
        current = x
        for weight, bias in zip(self.hidden_weights, self.hidden_biases):
            current = np.tanh(weight @ current + bias)
            hidden_states.append(current)
        policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ current + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        value = float(
            np.nan_to_num(
                (self.W2_value @ current + self.b2_value)[0],
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        if store_cache:
            self.cache = DeepMotorCache(x=x, hidden_states=tuple(hidden_states))
        return policy_logits, value

    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
    ) -> Array:
        if self.cache is None:
            raise RuntimeError("Deep true monolithic network backward called without cache.")
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        x = self.cache.x
        hidden_states = list(self.cache.hidden_states)
        final_hidden = hidden_states[-1]
        grad_W2_policy = np.outer(grad_policy_logits, final_hidden)
        grad_b2_policy = grad_policy_logits
        grad_W2_value = grad_value * final_hidden.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_hidden = (
            self.W2_policy.T @ grad_policy_logits
            + self.W2_value.T[:, 0] * grad_value
        )
        grad_inputs: Array = np.zeros(self.input_dim, dtype=float)
        for layer_idx in range(len(hidden_states) - 1, -1, -1):
            hidden = hidden_states[layer_idx]
            prev_activation = x if layer_idx == 0 else hidden_states[layer_idx - 1]
            dz = grad_hidden * (1.0 - hidden**2)
            grad_inputs = self.hidden_weights[layer_idx].T @ dz
            grad_weight = np.outer(dz, prev_activation)
            grad_bias = dz
            self.hidden_weights[layer_idx] -= lr * grad_weight
            self.hidden_biases[layer_idx] -= lr * grad_bias
            grad_hidden = grad_inputs
        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        return grad_inputs

    def value_only(self, x: Array) -> float:
        _, value = self.forward(x, store_cache=False)
        return value

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "output_dim": self.output_dim,
            "W2_policy": self.W2_policy.copy(),
            "b2_policy": self.b2_policy.copy(),
            "W2_value": self.W2_value.copy(),
            "b2_value": self.b2_value.copy(),
        }
        for idx, (weight, bias) in enumerate(zip(self.hidden_weights, self.hidden_biases), start=1):
            state[f"W{idx}"] = weight.copy()
            state[f"b{idx}"] = bias.copy()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected_keys = {
            "name",
            "input_dim",
            "hidden_dim",
            "hidden_sizes",
            "output_dim",
            "W2_policy",
            "b2_policy",
            "W2_value",
            "b2_value",
        }
        for idx in range(1, len(self.hidden_sizes) + 1):
            expected_keys.add(f"W{idx}")
            expected_keys.add(f"b{idx}")
        _validate_state_dict(
            state,
            expected_keys=expected_keys,
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "hidden_sizes": list(self.hidden_sizes),
                "output_dim": self.output_dim,
            },
            name=self.name,
        )
        prev_dim = self.input_dim
        new_hidden_weights: list[Array] = []
        new_hidden_biases: list[Array] = []
        for idx, hidden_dim in enumerate(self.hidden_sizes, start=1):
            new_hidden_weights.append(
                _coerce_state_array(
                    state,
                    f"W{idx}",
                    (hidden_dim, prev_dim),
                    name=self.name,
                )
            )
            new_hidden_biases.append(
                _coerce_state_array(
                    state,
                    f"b{idx}",
                    (hidden_dim,),
                    name=self.name,
                )
            )
            prev_dim = hidden_dim
        self.hidden_weights = new_hidden_weights
        self.hidden_biases = new_hidden_biases
        self.W2_policy = _coerce_state_array(
            state,
            "W2_policy",
            (self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_policy = _coerce_state_array(
            state,
            "b2_policy",
            (self.output_dim,),
            name=self.name,
        )
        self.W2_value = _coerce_state_array(
            state,
            "W2_value",
            (1, self.hidden_dim),
            name=self.name,
        )
        self.b2_value = _coerce_state_array(
            state,
            "b2_value",
            (1,),
            name=self.name,
        )
        self.cache = None

    def parameter_norm(self) -> float:
        return _parameter_norm_of(
            *self.hidden_weights,
            *self.hidden_biases,
            self.W2_policy,
            self.b2_policy,
            self.W2_value,
            self.b2_value,
        )

    def count_parameters(self) -> int:
        hidden_total = sum(weight.size + bias.size for weight, bias in zip(self.hidden_weights, self.hidden_biases))
        return int(
            hidden_total
            + self.W2_policy.size
            + self.b2_policy.size
            + self.W2_value.size
            + self.b2_value.size
        )


@dataclass
class ArbitrationCache:
    x: Array
    h: Array


class ArbitrationNetwork:
    """
    Learned arbitration network for valence selection, module gate adjustment, and value estimation.

    The input is the concatenated evidence vector used by arbitration: six threat
    signals, six hunger signals, six sleep signals, and six exploration signals.
    The gate head returns multiplicative adjustments constrained to configured
    bounds, defaulting to [0.5, 1.5].
    This lets learning reduce or amplify a module's influence, while preventing
    complete module silencing and uncapped amplification.
    """

    EVIDENCE_SIGNAL_NAMES = (
        "threat.predator_visible",
        "threat.predator_certainty",
        "threat.predator_motion_salience",
        "threat.recent_contact",
        "threat.recent_pain",
        "threat.predator_smell_strength",
        "hunger.hunger",
        "hunger.on_food",
        "hunger.food_visible",
        "hunger.food_certainty",
        "hunger.food_smell_strength",
        "hunger.food_memory_freshness",
        "sleep.fatigue",
        "sleep.sleep_debt",
        "sleep.night",
        "sleep.on_shelter",
        "sleep.shelter_role_level",
        "sleep.shelter_path_confidence",
        "exploration.safety_margin",
        "exploration.residual_drive",
        "exploration.day",
        "exploration.off_shelter",
        "exploration.visual_openness",
        "exploration.food_smell_directionality",
    )
    INPUT_DIM = len(EVIDENCE_SIGNAL_NAMES)
    VALENCE_DIM = 4
    GATE_DIM = 9
    GATE_ADJUSTMENT_MIN = 0.5
    GATE_ADJUSTMENT_MAX = 1.5

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
        name: str = "arbitration",
        gate_adjustment_min: float = GATE_ADJUSTMENT_MIN,
        gate_adjustment_max: float = GATE_ADJUSTMENT_MAX,
    ) -> None:
        """
        Create an arbitration network with a shared hidden trunk and three output heads (valence logits, gate adjustments, and scalar value), and initialize learnable parameters and cache.
        
        Parameters:
            input_dim (int): Expected dimensionality of the concatenated arbitration evidence vector; must equal ArbitrationNetwork.INPUT_DIM.
            hidden_dim (int): Size of the shared hidden layer.
            rng (np.random.Generator): Random number generator used to initialize weight arrays.
            name (str): Human-readable name used in state metadata and diagnostics.
            gate_adjustment_min (float): Lower bound for learned multiplicative gate adjustments.
            gate_adjustment_max (float): Upper bound for learned multiplicative gate adjustments.
        
        Raises:
            ValueError: If `input_dim` does not equal ArbitrationNetwork.INPUT_DIM,
                or if gate adjustment bounds are non-finite or unordered.
        """
        if input_dim != self.INPUT_DIM:
            raise ValueError(
                f"{name}: input_dim must be {self.INPUT_DIM} for the concatenated arbitration evidence vector; "
                f"received {input_dim}"
            )
        gate_adjustment_min = float(gate_adjustment_min)
        gate_adjustment_max = float(gate_adjustment_max)
        if not np.isfinite(gate_adjustment_min) or not np.isfinite(gate_adjustment_max):
            raise ValueError(f"{name}: gate adjustment bounds must be finite.")
        if gate_adjustment_min >= gate_adjustment_max:
            raise ValueError(f"{name}: gate adjustment bounds must be ordered as min < max.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.valence_dim = self.VALENCE_DIM
        self.gate_dim = self.GATE_DIM
        self.name = name
        self.gate_adjustment_min = gate_adjustment_min
        self.gate_adjustment_max = gate_adjustment_max
        self.W1 = rng.normal(0.0, _weight_scale(input_dim), size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2_valence = rng.normal(0.0, _weight_scale(hidden_dim), size=(self.valence_dim, hidden_dim))
        self.b2_valence = np.zeros(self.valence_dim, dtype=float)
        self.W2_gate = rng.normal(0.0, _weight_scale(hidden_dim), size=(self.gate_dim, hidden_dim))
        self.b2_gate = np.zeros(self.gate_dim, dtype=float)
        self.W2_value = rng.normal(0.0, _weight_scale(hidden_dim), size=(1, hidden_dim))
        self.b2_value = np.zeros(1, dtype=float)
        self.cache: Optional[ArbitrationCache] = None

    def forward(self, evidence_vector: Array, *, store_cache: bool = True) -> tuple[Array, Array, float]:
        """
        Compute arbitration outputs from an evidence vector: valence logits, bounded gate adjustments, and a scalar value estimate.
        
        Parameters:
            evidence_vector (Array): Input evidence of shape (input_dim,). Values are sanitized (NaNs → 0, +inf → 1, -inf → -1); a ValueError is raised if the shaped input does not equal (input_dim,).
            store_cache (bool): If True, cache the sanitized input and hidden activations on the instance for use by a subsequent backward pass.
        
        Returns:
            tuple:
                valence_logits (Array): Logits for valence classification with shape (VALENCE_DIM,), clipped to [-20, 20].
                gate_adjustments (Array): Bounded gate adjustment factors with shape (GATE_DIM,); each element equals gate_adjustment_min + (gate_adjustment_max - gate_adjustment_min) * sigmoid(raw_gate).
                value (float): Scalar value estimate, clipped to [-20, 20].
        """
        x = np.nan_to_num(np.asarray(evidence_vector, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        if x.shape != (self.input_dim,):
            raise ValueError(f"{self.name}: evidence_vector expected shape {(self.input_dim,)}, received {x.shape}")
        h = np.tanh(self.W1 @ x + self.b1)
        valence_logits = np.clip(
            np.nan_to_num(self.W2_valence @ h + self.b2_valence, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        raw_gate = np.clip(
            np.nan_to_num(self.W2_gate @ h + self.b2_gate, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        gate_range = self.gate_adjustment_max - self.gate_adjustment_min
        gate_adjustments = self.gate_adjustment_min + gate_range * _sigmoid(raw_gate)
        value = float(np.nan_to_num((self.W2_value @ h + self.b2_value)[0], nan=0.0, posinf=20.0, neginf=-20.0))
        if store_cache:
            self.cache = ArbitrationCache(x=x, h=h)
        return valence_logits, gate_adjustments, value

    def backward(
        self,
        grad_valence_logits: Array,
        grad_gate_adjustments: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
    ) -> Array:
        """
        Apply SGD updates to the arbitration network's parameters using provided output gradients and return the gradient with respect to the evidence input.
        
        Parameters:
            grad_valence_logits (Array): Gradient of the loss with respect to the valence logits output.
            grad_gate_adjustments (Array): Gradient of the loss with respect to the constrained gate-adjustment outputs; this is backpropagated through the sigmoid-and-affine mapping to obtain gradients for the raw gate head.
            grad_value (float): Gradient of the loss with respect to the scalar value output; this is clipped elementwise to the range [-grad_clip, grad_clip].
            lr (float): Learning rate for the SGD parameter updates.
            grad_clip (float): Maximum L2-norm for clipping applied to vector gradients (valence and gate); scalar gradients are clipped elementwise to [-grad_clip, grad_clip].
        
        Returns:
            grad_x (Array): Gradient of the loss with respect to the evidence input (same shape as the network input).
        
        Raises:
            RuntimeError: If no forward-pass cache is available when calling backward.
        """
        if self.cache is None:
            raise RuntimeError("Arbitration network backward called without cache.")
        grad_valence_logits = _clip_grad_logits(grad_valence_logits, grad_clip)
        grad_gate_adjustments = _clip_grad_logits(grad_gate_adjustments, grad_clip)
        grad_value = float(
            np.clip(
                np.nan_to_num(
                    np.asarray(grad_value, dtype=float),
                    nan=0.0,
                    posinf=grad_clip,
                    neginf=-grad_clip,
                ),
                -grad_clip,
                grad_clip,
            )
        )

        x = self.cache.x
        h = self.cache.h
        raw_gate = np.clip(
            np.nan_to_num(self.W2_gate @ h + self.b2_gate, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        gate_sigmoid = _sigmoid(raw_gate)
        gate_range = self.gate_adjustment_max - self.gate_adjustment_min
        grad_gate_raw = grad_gate_adjustments * gate_range * gate_sigmoid * (1.0 - gate_sigmoid)

        grad_W2_valence = np.outer(grad_valence_logits, h)
        grad_b2_valence = grad_valence_logits
        grad_W2_gate = np.outer(grad_gate_raw, h)
        grad_b2_gate = grad_gate_raw
        grad_W2_value = grad_value * h.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)

        dh = (
            self.W2_valence.T @ grad_valence_logits
            + self.W2_gate.T @ grad_gate_raw
            + self.W2_value.T[:, 0] * grad_value
        )
        dz1 = dh * (1.0 - h**2)
        grad_x = self.W1.T @ dz1
        grad_W1 = np.outer(dz1, x)
        grad_b1 = dz1

        self.W2_valence -= lr * grad_W2_valence
        self.b2_valence -= lr * grad_b2_valence
        self.W2_gate -= lr * grad_W2_gate
        self.b2_gate -= lr * grad_b2_gate
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        return grad_x

    def state_dict(self) -> dict[str, object]:
        """
        Create a serializable snapshot of the network's metadata and parameter arrays.
        
        Returns:
            state (dict[str, object]): Dictionary containing network metadata (`name`, `input_dim`, `hidden_dim`, `valence_dim`, `gate_dim`, `gate_adjustment_min`, `gate_adjustment_max`) and copies of the parameter arrays `W1`, `b1`, `W2_valence`, `b2_valence`, `W2_gate`, `b2_gate`, `W2_value`, and `b2_value`.
        """
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "valence_dim": self.valence_dim,
            "gate_dim": self.gate_dim,
            "gate_adjustment_min": self.gate_adjustment_min,
            "gate_adjustment_max": self.gate_adjustment_max,
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2_valence": self.W2_valence.copy(),
            "b2_valence": self.b2_valence.copy(),
            "W2_gate": self.W2_gate.copy(),
            "b2_gate": self.b2_gate.copy(),
            "W2_value": self.W2_value.copy(),
            "b2_value": self.b2_value.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """
        Load network parameters from `state` into this ArbitrationNetwork, validating keys and metadata,
        coercing arrays to exact expected shapes, and clearing the internal cache.
        
        Parameters:
            state (dict[str, object]): Serializable state produced by `state_dict()` containing
                the keys "name", "input_dim", "hidden_dim", "valence_dim", "gate_dim",
                "gate_adjustment_min", "gate_adjustment_max",
                "W1", "b1", "W2_valence", "b2_valence", "W2_gate", "b2_gate", "W2_value", and "b2_value".
        
        Raises:
            ValueError: If the state is missing or contains unexpected keys, if metadata values
                (name/dimensions) do not match this instance, or if any parameter has an incorrect shape.
        """
        _validate_state_dict(
            state,
            expected_keys={
                "name",
                "input_dim",
                "hidden_dim",
                "valence_dim",
                "gate_dim",
                "gate_adjustment_min",
                "gate_adjustment_max",
                "W1",
                "b1",
                "W2_valence",
                "b2_valence",
                "W2_gate",
                "b2_gate",
                "W2_value",
                "b2_value",
            },
            expected_metadata={
                "name": self.name,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "valence_dim": self.valence_dim,
                "gate_dim": self.gate_dim,
                "gate_adjustment_min": self.gate_adjustment_min,
                "gate_adjustment_max": self.gate_adjustment_max,
            },
            name=self.name,
        )
        self.W1 = _coerce_state_array(state, "W1", (self.hidden_dim, self.input_dim), name=self.name)
        self.b1 = _coerce_state_array(state, "b1", (self.hidden_dim,), name=self.name)
        self.W2_valence = _coerce_state_array(
            state,
            "W2_valence",
            (self.valence_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_valence = _coerce_state_array(state, "b2_valence", (self.valence_dim,), name=self.name)
        self.W2_gate = _coerce_state_array(
            state,
            "W2_gate",
            (self.gate_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_gate = _coerce_state_array(state, "b2_gate", (self.gate_dim,), name=self.name)
        self.W2_value = _coerce_state_array(state, "W2_value", (1, self.hidden_dim), name=self.name)
        self.b2_value = _coerce_state_array(state, "b2_value", (1,), name=self.name)
        self.cache = None

    def parameter_norm(self) -> float:
        """
        Compute the L2 (Euclidean) norm of all learnable arbitration parameters.
        
        Returns:
            norm (float): L2 norm of the concatenated parameters W1, b1, W2_valence, b2_valence, W2_gate, b2_gate, W2_value, and b2_value.
        """
        return _parameter_norm_of(
            self.W1, self.b1,
            self.W2_valence, self.b2_valence,
            self.W2_gate, self.b2_gate,
            self.W2_value, self.b2_value,
        )

    def count_parameters(self) -> int:
        """Return the number of trainable parameters in the arbitration network."""
        return int(
            self.W1.size
            + self.b1.size
            + self.W2_valence.size
            + self.b2_valence.size
            + self.W2_gate.size
            + self.b2_gate.size
            + self.W2_value.size
            + self.b2_value.size
        )
