from __future__ import annotations

from ._nn_shared import *

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


__all__ = [name for name in globals() if not name.startswith("__")]
