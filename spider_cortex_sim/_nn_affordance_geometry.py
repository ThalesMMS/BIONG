from __future__ import annotations

from ._nn_shared import *
from ._nn_option_controller import *

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


__all__ = [name for name in globals() if not name.startswith("__")]
