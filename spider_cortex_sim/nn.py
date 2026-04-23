from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

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


@dataclass
class ProposalCache:
    x: Array
    h: Array


@dataclass
class RecurrentProposalCache:
    x: Array
    h_prev: Array
    h_new: Array


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
