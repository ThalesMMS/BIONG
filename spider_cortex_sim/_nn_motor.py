from __future__ import annotations

from ._nn_shared import *
from ._nn_proposal import *

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


__all__ = [name for name in globals() if not name.startswith("__")]
