from __future__ import annotations

from ._nn_shared import *
from ._nn_motor import *

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


__all__ = [name for name in globals() if not name.startswith("__")]
