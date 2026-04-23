from __future__ import annotations

import numpy as np


Array = np.ndarray


def softmax(logits: Array, temperature: float = 1.0) -> Array:
    """
    Convert a vector of logits into a numerically stable probability distribution using the softmax function.
    
    The input is sanitized (NaNs replaced with 0.0, infinities clamped, values clipped), then shifted by the maximum to avoid overflow before exponentiation. If the normalization factor is non-positive or not finite, a uniform distribution is returned as a safe fallback.
    
    Parameters:
        logits (Array): 1-D array-like of unnormalized log-probabilities.
        temperature (float): Positive softmax temperature; probabilities are computed as `softmax(logits / temperature)`.
    
    Returns:
        Array: Float array of the same shape containing probabilities that sum to 1.0; if normalization fails, a uniform distribution across elements.
    """
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite positive scalar")
    logits = np.clip(
        np.nan_to_num(np.asarray(logits, dtype=float), nan=0.0, posinf=20.0, neginf=-20.0),
        -20.0,
        20.0,
    )
    logits = logits / temperature
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(logits, 1.0 / len(logits), dtype=float)
    return exp / total


def cross_entropy_loss(
    student_logits: Array,
    teacher_probs: Array,
    *,
    temperature: float = 1.0,
) -> float:
    """
    Compute soft-target cross-entropy between student logits and teacher probabilities.

    The student distribution is computed as `softmax(student_logits / temperature)`.
    Teacher probabilities are sanitized, renormalized, and clamped away from zero
    in the log term to avoid numerical issues.
    """
    student_probs = softmax(student_logits, temperature=temperature)
    teacher_probs = np.asarray(teacher_probs, dtype=float)
    if teacher_probs.ndim != 1:
        raise ValueError("teacher_probs must be a 1-D vector")
    if student_probs.shape != teacher_probs.shape:
        raise ValueError(
            "student_logits and teacher_probs must describe the same distribution shape"
        )
    teacher_probs = np.nan_to_num(
        teacher_probs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    teacher_probs = np.clip(teacher_probs, 0.0, None)
    total = float(np.sum(teacher_probs))
    if total <= 0.0 or not np.isfinite(total):
        teacher_probs = np.full_like(student_probs, 1.0 / len(student_probs), dtype=float)
    else:
        teacher_probs = teacher_probs / total
    return float(-np.sum(teacher_probs * np.log(np.clip(student_probs, 1e-8, 1.0))))


def kl_divergence(
    student_logits: Array,
    teacher_probs: Array,
    *,
    temperature: float = 1.0,
) -> float:
    """
    Compute KL(teacher || student) from teacher probabilities and student logits.

    Both distributions are sanitized and the student distribution is computed as
    `softmax(student_logits / temperature)`.
    """
    student_probs = softmax(student_logits, temperature=temperature)
    teacher_probs = np.asarray(teacher_probs, dtype=float)
    if teacher_probs.ndim != 1:
        raise ValueError("teacher_probs must be a 1-D vector")
    if student_probs.shape != teacher_probs.shape:
        raise ValueError(
            "student_logits and teacher_probs must describe the same distribution shape"
        )
    teacher_probs = np.nan_to_num(
        teacher_probs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    teacher_probs = np.clip(teacher_probs, 0.0, None)
    total = float(np.sum(teacher_probs))
    if total <= 0.0 or not np.isfinite(total):
        teacher_probs = np.full_like(student_probs, 1.0 / len(student_probs), dtype=float)
    else:
        teacher_probs = teacher_probs / total
    teacher_safe = np.clip(teacher_probs, 1e-8, 1.0)
    student_safe = np.clip(student_probs, 1e-8, 1.0)
    return float(np.sum(teacher_safe * (np.log(teacher_safe) - np.log(student_safe))))


def one_hot(index: int, size: int) -> Array:
    """
    Create a one-hot float vector of length `size` with 1.0 at `index` and 0.0 elsewhere.
    
    Parameters:
        index (int): Position to set to 1.0.
        size (int): Length of the output vector.
    
    Returns:
        Array: Float NumPy array of shape (size,) containing the one-hot encoding.
    
    Raises:
        IndexError: If `index` is outside the valid range for the output vector.
    """
    vec = np.zeros(size, dtype=float)
    vec[index] = 1.0
    return vec


def _weight_scale(dim: int) -> float:
    """
    Compute the standard deviation to use for weight initialization for a layer of the given dimension.

    Parameters:
        dim (int): Layer dimension used to scale the initialization; values less than 1 are treated as 1.

    Returns:
        float: Standard deviation computed as 0.35 / sqrt(max(1, dim)).
    """
    return 0.35 / np.sqrt(max(1, dim))


def _coerce_state_array(state: dict[str, object], key: str, shape: tuple[int, ...], *, name: str) -> Array:
    """
    Validate and convert an entry from a saved state dict into a float NumPy array with an exact shape.

    Parameters:
        state (dict[str, object]): Mapping containing serialized arrays.
        key (str): Key in `state` whose value will be converted.
        shape (tuple[int, ...]): Expected shape of the resulting array.
        name (str): Human-readable name used in the error message on mismatch.

    Returns:
        Array: A copied NumPy array of dtype `float` with the specified `shape`.

    Raises:
        ValueError: If `state[key]` is missing or does not have the expected `shape`.
    """
    if key not in state:
        raise ValueError(f"{name}: missing required state key {key!r}")
    array = np.asarray(state[key], dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name}: {key} expected {shape}, received {array.shape}")
    return np.array(array, dtype=float, copy=True)


def _state_scalar(value: object) -> object:
    """
    Coerce NumPy scalar/array values to native Python types for serialization.
    
    If `value` is a 0-dimensional NumPy array, returns its Python scalar via `item()`.
    If `value` is a multi-dimensional NumPy array, returns a Python list via `tolist()`.
    Otherwise returns `value` unchanged.
    
    Parameters:
        value (object): The value to coerce; commonly a NumPy array or a native Python object.
    
    Returns:
        object: A Python scalar for 0-d NumPy arrays, a Python list for other NumPy arrays, or the original `value`.
    """
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _validate_state_dict(
    state: dict[str, object],
    *,
    expected_keys: set[str],
    expected_metadata: dict[str, object],
    name: str,
) -> None:
    """
    Validate that a state dictionary contains exactly the expected keys and that specified metadata entries match expected values.

    Parameters:
        state (dict[str, object]): Serialized state dictionary to check.
        expected_keys (set[str]): Exact set of keys that `state` must contain.
        expected_metadata (dict[str, object]): Mapping of metadata keys to their expected scalar values; each entry is compared against the corresponding value in `state` after coercion via `_state_scalar`.
        name (str): Contextual name used in raised error messages.

    Raises:
        ValueError: If `state` is missing or contains unexpected keys, or if any metadata entry does not equal its expected value.
    """
    actual_keys = set(state.keys())
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        parts: list[str] = []
        if missing:
            parts.append(f"missing keys {missing}")
        if unexpected:
            parts.append(f"unexpected keys {unexpected}")
        raise ValueError(f"{name}: state_dict key mismatch ({'; '.join(parts)})")

    for key, expected in expected_metadata.items():
        actual = _state_scalar(state[key])
        if actual != expected:
            raise ValueError(f"{name}: metadata {key!r} expected {expected!r}, received {actual!r}")


def _parameter_norm_of(*arrays: np.ndarray) -> float:
    """
    Compute the L2 (Euclidean) norm of all elements in the provided arrays.
    
    Returns:
        norm (float): Square root of the sum of squares of every element across the given arrays. Returns 0.0 if no arrays are provided.
    """
    return float(np.sqrt(sum(np.sum(a**2) for a in arrays)))


def _clip_grad_logits(grad_logits: Array, grad_clip: float) -> Array:
    """
    Clamp and rescale a 1-D gradient vector so its L2 norm does not exceed the specified clip value.
    
    Parameters:
        grad_logits (np.ndarray): 1-D array of gradient logits; NaNs and infinities are sanitized before clipping.
        grad_clip (float): Maximum allowed L2 norm for the returned vector; must be a finite, non-negative scalar.
    
    Returns:
        np.ndarray: A float array with the same shape as `grad_logits` containing the sanitized and possibly rescaled gradients. If `grad_clip` is 0.0, an all-zero array is returned.
    
    Raises:
        ValueError: If `grad_logits` is not 1-D, or if `grad_clip` is not a finite non-negative scalar.
    """
    grad_logits = np.asarray(grad_logits, dtype=float)
    if grad_logits.ndim != 1:
        raise ValueError("grad_logits must be a 1-D vector")
    grad_clip_array = np.asarray(grad_clip, dtype=float)
    if grad_clip_array.shape != ():
        raise ValueError("grad_clip must be a finite non-negative scalar")
    grad_clip_value = float(grad_clip_array)
    if not np.isfinite(grad_clip_value) or grad_clip_value < 0.0:
        raise ValueError("grad_clip must be a finite non-negative scalar")
    if grad_clip_value == 0.0:
        return np.zeros_like(grad_logits, dtype=float)
    grad_logits = np.nan_to_num(grad_logits, nan=0.0, posinf=5.0, neginf=-5.0)
    norm = float(np.linalg.norm(grad_logits))
    if norm > grad_clip_value:
        grad_logits = grad_logits * (grad_clip_value / (norm + 1e-8))
    return grad_logits


def _sigmoid(x: Array) -> Array:
    """
    Compute the numerically stabilized sigmoid of the input.
    
    Parameters:
        x (array-like): Input values. Converted to a float NumPy array; NaNs are replaced with 0.0 and infinities are clamped to [-60.0, 60.0] before applying the logistic function.
    
    Returns:
        ndarray: Array with the same shape as `x` containing values in the interval (0.0, 1.0).
    """
    x = np.clip(np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=60.0, neginf=-60.0), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))
