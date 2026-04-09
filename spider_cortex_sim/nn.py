from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


Array = np.ndarray


def softmax(logits: Array) -> Array:
    logits = np.clip(
        np.nan_to_num(np.asarray(logits, dtype=float), nan=0.0, posinf=20.0, neginf=-20.0),
        -20.0,
        20.0,
    )
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(logits, 1.0 / len(logits), dtype=float)
    return exp / total


def one_hot(index: int, size: int) -> Array:
    vec = np.zeros(size, dtype=float)
    vec[index] = 1.0
    return vec


@dataclass
class ProposalCache:
    x: Array
    h: Array


class ProposalNetwork:
    """Small MLP that produces action logits for a cortical subsystem."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator, name: str) -> None:
        """
        Initialize the ProposalNetwork's parameters and cache.
        
        Parameters:
            input_dim (int): Dimensionality of the input vector.
            hidden_dim (int): Number of hidden units in the single hidden layer.
            output_dim (int): Number of output logits.
            rng (np.random.Generator): Random number generator used to sample initial weights.
            name (str): Identifier for this network instance.
        
        Details:
            - Initializes weight matrices and bias vectors with shapes:
                W1: (hidden_dim, input_dim), b1: (hidden_dim,)
                W2: (output_dim, hidden_dim), b2: (output_dim,)
            - Weights are drawn from a normal distribution with standard deviations
              scale1 = 0.35 / sqrt(max(1, input_dim)) and scale2 = 0.35 / sqrt(max(1, hidden_dim)).
            - The parameter cache is initialized to None.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name
        scale1 = 0.35 / np.sqrt(max(1, input_dim))
        scale2 = 0.35 / np.sqrt(max(1, hidden_dim))
        self.W1 = rng.normal(0.0, scale1, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2 = rng.normal(0.0, scale2, size=(output_dim, hidden_dim))
        self.b2 = np.zeros(output_dim, dtype=float)
        self.cache: Optional[ProposalCache] = None

    def forward(self, x: Array, *, store_cache: bool = True) -> Array:
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        z1 = self.W1 @ x + self.b1
        h = np.tanh(z1)
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
        Perform a gradient step on network parameters using backpropagated logits.
        
        Sanitizes and optionally clips `grad_logits`, computes parameter gradients using the cached
        forward pass, and updates weights and biases with a simple SGD step.
        
        Parameters:
            grad_logits: Gradient of the loss w.r.t. the output logits. NaNs are converted to 0, +inf to 5, and -inf to -5; the vector is then scaled so its Euclidean norm does not exceed `grad_clip`.
            lr: Learning rate used to scale the parameter updates.
            grad_clip: Maximum allowed Euclidean norm for `grad_logits`; values above this are scaled down.
        
        Raises:
            RuntimeError: If called when no forward-pass cache is available.
        """
        if self.cache is None:
            raise RuntimeError(f"Network {self.name} backward called without cache.")
        grad_logits = np.nan_to_num(np.asarray(grad_logits, dtype=float), nan=0.0, posinf=5.0, neginf=-5.0)
        norm = float(np.linalg.norm(grad_logits))
        if norm > grad_clip:
            grad_logits = grad_logits * (grad_clip / (norm + 1e-8))
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
        W1 = np.asarray(state["W1"], dtype=float)
        b1 = np.asarray(state["b1"], dtype=float)
        W2 = np.asarray(state["W2"], dtype=float)
        b2 = np.asarray(state["b2"], dtype=float)
        if W1.shape != (self.hidden_dim, self.input_dim):
            raise ValueError(
                f"{self.name}: W1 expected {(self.hidden_dim, self.input_dim)}, received {W1.shape}"
            )
        if W2.shape != (self.output_dim, self.hidden_dim):
            raise ValueError(
                f"{self.name}: W2 expected {(self.output_dim, self.hidden_dim)}, received {W2.shape}"
            )
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.cache = None

    def parameter_norm(self) -> float:
        total = (
            np.sum(self.W1**2)
            + np.sum(self.b1**2)
            + np.sum(self.W2**2)
            + np.sum(self.b2**2)
        )
        return float(np.sqrt(total))


@dataclass
class MotorCache:
    x: Array
    h: Array


class MotorNetwork:
    """Motor network with a corrective policy head and a value critic head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator, name: str = "motor_cortex") -> None:
        """
        Initialize the MotorNetwork's parameters and metadata.
        
        Parameters:
            input_dim (int): Dimensionality of the input vector.
            hidden_dim (int): Number of hidden units in the shared layer.
            output_dim (int): Number of policy outputs (action logits).
            rng (np.random.Generator): Random generator used to sample initial weights from normal distributions.
            name (str): Optional network name; defaults to "motor_cortex".
        
        Details:
            - Allocates and initializes weight and bias arrays:
                - W1: (hidden_dim, input_dim), b1: (hidden_dim,)
                - W2_policy: (output_dim, hidden_dim), b2_policy: (output_dim,)
                - W2_value: (1, hidden_dim), b2_value: (1,)
            - Weight scales use 0.35 / sqrt(max(1, dimension)) per layer.
            - Sets self.cache to None.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name
        scale1 = 0.35 / np.sqrt(max(1, input_dim))
        scale2 = 0.35 / np.sqrt(max(1, hidden_dim))
        self.W1 = rng.normal(0.0, scale1, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim, dtype=float)
        self.W2_policy = rng.normal(0.0, scale2, size=(output_dim, hidden_dim))
        self.b2_policy = np.zeros(output_dim, dtype=float)
        self.W2_value = rng.normal(0.0, scale2, size=(1, hidden_dim))
        self.b2_value = np.zeros(1, dtype=float)
        self.cache: Optional[MotorCache] = None

    def forward(self, x: Array, *, store_cache: bool = True) -> tuple[Array, float]:
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        z1 = self.W1 @ x + self.b1
        h = np.tanh(z1)
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
        Apply gradients to the motor network's parameters and return the gradient with respect to the input.
        
        Sanitizes and clips the provided policy-logits and value gradients, computes gradients for the policy head, value head, and shared layers, performs an SGD parameter update using the given learning rate, and returns the gradient of the loss with respect to the input vector.
        
        Parameters:
            grad_policy_logits (Array): Gradient of the loss with respect to the policy logits.
            grad_value (float): Gradient of the loss with respect to the scalar value output.
            lr (float): Learning rate used for the parameter update.
            grad_clip (float): Maximum allowed norm (or absolute value for the scalar) for gradients; defaults to 5.0.
        
        Returns:
            Array: Gradient of the loss with respect to the input vector `x`.
        
        Raises:
            RuntimeError: If called when no forward-pass cache is available.
        """
        if self.cache is None:
            raise RuntimeError("Motor network backward called without cache.")
        grad_policy_logits = np.nan_to_num(
            np.asarray(grad_policy_logits, dtype=float),
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        )
        gp_norm = float(np.linalg.norm(grad_policy_logits))
        if gp_norm > grad_clip:
            grad_policy_logits = grad_policy_logits * (grad_clip / (gp_norm + 1e-8))
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
        W1 = np.asarray(state["W1"], dtype=float)
        b1 = np.asarray(state["b1"], dtype=float)
        W2_policy = np.asarray(state["W2_policy"], dtype=float)
        b2_policy = np.asarray(state["b2_policy"], dtype=float)
        W2_value = np.asarray(state["W2_value"], dtype=float)
        b2_value = np.asarray(state["b2_value"], dtype=float)
        if W1.shape != (self.hidden_dim, self.input_dim):
            raise ValueError(
                f"{self.name}: W1 expected {(self.hidden_dim, self.input_dim)}, received {W1.shape}"
            )
        if W2_policy.shape != (self.output_dim, self.hidden_dim):
            raise ValueError(
                f"{self.name}: W2_policy expected {(self.output_dim, self.hidden_dim)}, received {W2_policy.shape}"
            )
        self.W1 = W1
        self.b1 = b1
        self.W2_policy = W2_policy
        self.b2_policy = b2_policy
        self.W2_value = W2_value
        self.b2_value = b2_value
        self.cache = None

    def parameter_norm(self) -> float:
        total = (
            np.sum(self.W1**2)
            + np.sum(self.b1**2)
            + np.sum(self.W2_policy**2)
            + np.sum(self.b2_policy**2)
            + np.sum(self.W2_value**2)
            + np.sum(self.b2_value**2)
        )
        return float(np.sqrt(total))