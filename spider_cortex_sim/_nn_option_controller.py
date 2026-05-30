from __future__ import annotations

from ._nn_shared import *
from ._nn_recurrent_monolithic import *

class OwnedOptionControllerTrueMonolithicNetwork(RecurrentOptionTrueMonolithicNetwork):
    """Direct policy where the active option owns the final primitive action logits."""

    owned_option_controller = True

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        *,
        event_buffer_size: int = 8,
        option_ttl: int = 4,
        leaf_context_bias_scale: float = 3.0,
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
        self.leaf_context_bias_scale = float(leaf_context_bias_scale)
        self.W2_option_leaf = rng.normal(
            0.0,
            _weight_scale(hidden_dim),
            size=(self.option_dim, output_dim, hidden_dim),
        )
        self.b2_option_leaf = np.zeros((self.option_dim, output_dim), dtype=float)
        self.runtime_observation_meta: dict[str, object] = {}
        self.external_override_count = 0
        self.safety_mask_count = 0
        self._initialize_owned_leaf_priors()

    def _initialize_owned_leaf_priors(self) -> None:
        self.b2_option_leaf[OPTION_TO_INDEX["REST"], ACTION_TO_INDEX["STAY"]] += 0.6
        self.b2_option_leaf[
            OPTION_TO_INDEX["POST_REST_REACTIVATE"],
            ACTION_TO_INDEX["MOVE_UP"],
        ] += 0.35
        self.b2_option_leaf[
            OPTION_TO_INDEX["DEEPEN_IN_SHELTER"],
            ACTION_TO_INDEX["MOVE_DOWN"],
        ] += 0.35

    def reset_hidden_state(self) -> None:
        super().reset_hidden_state()
        self.last_option_summary.update(
            {
                "option_leaf_logits": [],
                "option_owned_action": None,
                "safety_mask_applied": False,
                "safety_masked_actions": [],
                "external_override_count": int(self.external_override_count),
            }
        )

    def set_runtime_observation_meta(self, meta: Mapping[str, object] | None) -> None:
        self.runtime_observation_meta = dict(meta or {})

    def record_external_override(self, reason: str | None = None) -> None:
        self.external_override_count += 1

    def _owned_signals(self, x: Array) -> dict[str, float]:
        x = np.asarray(x, dtype=float)
        acute_threat = max(
            float(x[_ALERT_VISUAL_PREDATOR_THREAT_IDX]),
            float(x[_ALERT_OLFACTORY_PREDATOR_THREAT_IDX]),
            float(x[_ALERT_RECENT_CONTACT_IDX]),
            float(x[_ALERT_RECENT_PAIN_IDX]),
        )
        food_signal = max(
            float(x[_HUNGER_FOOD_VISIBLE_IDX]) * float(x[_HUNGER_FOOD_CERTAINTY_IDX]),
            float(x[_HUNGER_FOOD_SMELL_STRENGTH_IDX]),
            max(0.0, 1.0 - float(x[_HUNGER_FOOD_MEMORY_AGE_IDX])),
        )
        shelter_memory = max(0.0, 1.0 - float(x[_SLEEP_SHELTER_MEMORY_AGE_IDX]))
        fatigue = float(x[_SLEEP_FATIGUE_IDX])
        sleep_debt = float(x[_SLEEP_DEBT_IDX])
        night = float(x[_SLEEP_NIGHT_IDX])
        return {
            "hunger": float(x[_SLEEP_HUNGER_IDX]),
            "fatigue": fatigue,
            "sleep_debt": sleep_debt,
            "rest_pressure": max(fatigue, sleep_debt, night * 0.7),
            "on_food": float(x[_HUNGER_ON_FOOD_IDX]),
            "on_shelter": float(x[_SLEEP_ON_SHELTER_IDX]),
            "shelter_role_level": float(x[_SLEEP_SHELTER_ROLE_LEVEL_IDX]),
            "food_signal": food_signal,
            "shelter_memory": shelter_memory,
            "acute_threat": acute_threat,
        }

    def _owned_signal_termination(
        self,
        x: Array,
        reason: str | None,
    ) -> str | None:
        if reason is not None or self.current_option_idx < 0:
            return reason
        signals = self._owned_signals(x)
        option_name = OPTION_NAMES[int(self.current_option_idx)]
        if option_name == "FORAGE" and signals["on_food"] > 0.5:
            return "food_reached"
        if option_name == "RETURN_TO_SHELTER" and signals["on_shelter"] > 0.5:
            return "shelter_reached"
        if (
            option_name == "DEEPEN_IN_SHELTER"
            and signals["shelter_role_level"] >= 0.85
        ):
            return "deep_shelter_reached"
        if (
            option_name == "REST"
            and signals["on_shelter"] > 0.5
            and signals["rest_pressure"] < 0.25
            and signals["hunger"] > 0.35
        ):
            return "recovery_completed"
        if option_name == "POST_REST_REACTIVATE" and signals["on_shelter"] <= 0.0:
            return "shelter_exited"
        return None

    def _executive_option_logits(self, x: Array, learned_logits: Array) -> Array:
        logits = np.asarray(learned_logits, dtype=float).copy()
        signals = self._owned_signals(x)
        hunger = signals["hunger"]
        rest_pressure = signals["rest_pressure"]
        on_shelter = signals["on_shelter"] > 0.5
        acute_threat = signals["acute_threat"]
        if acute_threat >= 0.35:
            logits[OPTION_TO_INDEX["ESCAPE"]] += 6.0 * acute_threat
            logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 2.5 * acute_threat
        elif on_shelter:
            logits[OPTION_TO_INDEX["REST"]] += 3.0 * rest_pressure
            logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 1.8 * max(
                0.0,
                0.85 - signals["shelter_role_level"],
            )
            if hunger > 0.35 and rest_pressure < 0.65:
                logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] += 3.0 * hunger
        else:
            return_drive = max(rest_pressure, signals["shelter_memory"] * 0.5)
            if return_drive > 0.35:
                logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 3.0 * return_drive
            forage_drive = max(hunger, signals["food_signal"])
            if forage_drive > 0.15:
                logits[OPTION_TO_INDEX["FORAGE"]] += 2.5 * forage_drive
        return np.clip(np.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)

    @staticmethod
    @staticmethod
    def _add_direction_bias(bias: Array, dx: float, dy: float, strength: float) -> None:
        strength = float(max(0.0, strength))
        if strength <= 0.0:
            return
        dx = float(dx)
        dy = float(dy)
        if abs(dx) >= abs(dy) and abs(dx) > 1e-6:
            move = "MOVE_RIGHT" if dx > 0.0 else "MOVE_LEFT"
            orient = "ORIENT_RIGHT" if dx > 0.0 else "ORIENT_LEFT"
        elif abs(dy) > 1e-6:
            move = "MOVE_DOWN" if dy > 0.0 else "MOVE_UP"
            orient = "ORIENT_DOWN" if dy > 0.0 else "ORIENT_UP"
        else:
            return
        bias[ACTION_TO_INDEX[move]] += strength
        bias[ACTION_TO_INDEX[orient]] += 0.25 * strength

    def _option_leaf_context_bias(self, option_name: str, x: Array) -> Array:
        bias = np.zeros(self.output_dim, dtype=float)
        scale = float(self.leaf_context_bias_scale)
        if option_name == "FORAGE":
            if float(x[_HUNGER_FOOD_VISIBLE_IDX]) > 0.0:
                self._add_direction_bias(
                    bias,
                    float(x[_HUNGER_FOOD_DX_IDX]),
                    float(x[_HUNGER_FOOD_DY_IDX]),
                    scale * max(0.2, float(x[_HUNGER_FOOD_CERTAINTY_IDX])),
                )
            elif float(x[_HUNGER_FOOD_SMELL_STRENGTH_IDX]) > 0.0:
                self._add_direction_bias(
                    bias,
                    float(x[_HUNGER_FOOD_SMELL_DX_IDX]),
                    float(x[_HUNGER_FOOD_SMELL_DY_IDX]),
                    scale * float(x[_HUNGER_FOOD_SMELL_STRENGTH_IDX]),
                )
            else:
                memory_strength = max(0.0, 1.0 - float(x[_HUNGER_FOOD_MEMORY_AGE_IDX]))
                self._add_direction_bias(
                    bias,
                    float(x[_HUNGER_FOOD_MEMORY_DX_IDX]),
                    float(x[_HUNGER_FOOD_MEMORY_DY_IDX]),
                    scale * memory_strength,
                )
            transitions = self.runtime_observation_meta.get("local_transition_consequences")
            if isinstance(transitions, Mapping):
                for action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"):
                    payload = transitions.get(action_name)
                    if isinstance(payload, Mapping):
                        bias[ACTION_TO_INDEX[action_name]] += (
                            0.5 * scale * float(payload.get("food_dist_delta", 0.0))
                        )
        elif option_name == "RETURN_TO_SHELTER":
            self._add_direction_bias(
                bias,
                float(x[_SLEEP_SHELTER_MEMORY_DX_IDX]),
                float(x[_SLEEP_SHELTER_MEMORY_DY_IDX]),
                scale * max(0.2, 1.0 - float(x[_SLEEP_SHELTER_MEMORY_AGE_IDX])),
            )
            transitions = self.runtime_observation_meta.get("local_transition_consequences")
            if isinstance(transitions, Mapping):
                for action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"):
                    payload = transitions.get(action_name)
                    if isinstance(payload, Mapping):
                        bias[ACTION_TO_INDEX[action_name]] += (
                            0.8 * scale * float(payload.get("shelter_dist_delta", 0.0))
                        )
        elif option_name == "DEEPEN_IN_SHELTER":
            geodesics = self.runtime_observation_meta.get("local_geodesic_consequences")
            if isinstance(geodesics, Mapping):
                for action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"):
                    payload = geodesics.get(action_name)
                    if isinstance(payload, Mapping):
                        bias[ACTION_TO_INDEX[action_name]] += (
                            scale * float(payload.get("deep_geodesic_delta", 0.0))
                        )
                        if bool(payload.get("next_on_deep_target", False)):
                            bias[ACTION_TO_INDEX[action_name]] += scale
            else:
                bias[ACTION_TO_INDEX["MOVE_DOWN"]] += scale
        elif option_name == "REST":
            bias[ACTION_TO_INDEX["STAY"]] += scale * (
                1.0 + max(float(x[_SLEEP_FATIGUE_IDX]), float(x[_SLEEP_DEBT_IDX]))
            )
        elif option_name == "POST_REST_REACTIVATE":
            geodesics = self.runtime_observation_meta.get("local_geodesic_consequences")
            if isinstance(geodesics, Mapping):
                for action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"):
                    payload = geodesics.get(action_name)
                    if isinstance(payload, Mapping):
                        bias[ACTION_TO_INDEX[action_name]] += (
                            scale * float(payload.get("exit_geodesic_delta", 0.0))
                        )
                        if bool(payload.get("next_on_exit_target", False)):
                            bias[ACTION_TO_INDEX[action_name]] += scale
            else:
                bias[ACTION_TO_INDEX["MOVE_UP"]] += scale
                bias[ACTION_TO_INDEX["MOVE_LEFT"]] += 0.25 * scale
        elif option_name == "ESCAPE":
            self._add_direction_bias(
                bias,
                -float(x[_ALERT_PREDATOR_DX_IDX]),
                -float(x[_ALERT_PREDATOR_DY_IDX]),
                scale
                * max(
                    float(x[_ALERT_PREDATOR_VISIBLE_IDX])
                    * float(x[_ALERT_PREDATOR_CERTAINTY_IDX]),
                    float(x[_ALERT_PREDATOR_TRACE_STRENGTH_IDX]),
                    float(x[_ALERT_VISUAL_PREDATOR_THREAT_IDX]),
                    float(x[_ALERT_OLFACTORY_PREDATOR_THREAT_IDX]),
                ),
            )
            transitions = self.runtime_observation_meta.get("local_transition_consequences")
            if isinstance(transitions, Mapping):
                for action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"):
                    payload = transitions.get(action_name)
                    if isinstance(payload, Mapping):
                        bias[ACTION_TO_INDEX[action_name]] += (
                            scale * float(payload.get("predator_dist_delta", 0.0))
                        )
        return bias

    def _apply_safety_mask(self, logits: Array) -> tuple[Array, bool, list[str]]:
        masked = np.asarray(logits, dtype=float).copy()
        local_affordances = self.runtime_observation_meta.get("local_affordances")
        if not isinstance(local_affordances, Mapping):
            return masked, False, []
        masked_actions: list[str] = []
        for action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
            payload = local_affordances.get(action_name)
            if isinstance(payload, Mapping) and bool(payload.get("blocked", False)):
                masked[ACTION_TO_INDEX[action_name]] = -20.0
                masked_actions.append(action_name)
        if masked_actions:
            self.safety_mask_count += 1
        return masked, bool(masked_actions), masked_actions

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
        learned_option_logits = np.clip(
            np.nan_to_num(
                self.W2_option @ h_new + self.b2_option,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_logits = self._executive_option_logits(x, learned_option_logits)
        option_probs = softmax(option_logits)
        termination_reason = self._owned_signal_termination(
            x,
            super()._termination_reason(),
        )
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
        all_leaf_logits = np.einsum("ohd,d->oh", self.W2_option_leaf, h_new) + self.b2_option_leaf
        option_name = OPTION_NAMES[selected_option_idx]
        leaf_logits = np.clip(
            np.nan_to_num(
                all_leaf_logits[selected_option_idx]
                + self._option_leaf_context_bias(option_name, x),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        policy_logits, safety_mask_applied, safety_masked_actions = self._apply_safety_mask(
            leaf_logits
        )
        policy_logits = np.clip(
            np.nan_to_num(policy_logits, nan=0.0, posinf=20.0, neginf=-20.0),
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
            "selected_option": option_name,
            "option_age": int(self.current_option_age),
            "option_termination_reason": (
                "initial_selection"
                if termination_reason is None and option_input.sum() <= 0.0
                else ("active" if termination_reason is None else termination_reason)
            ),
            "option_logits": option_logits.round(6).tolist(),
            "option_leaf_logits": leaf_logits.round(6).tolist(),
            "option_owned_action": next(
                name
                for name, idx in ACTION_TO_INDEX.items()
                if idx == int(np.argmax(policy_logits))
            ),
            "safety_mask_applied": bool(safety_mask_applied),
            "safety_masked_actions": list(safety_masked_actions),
            "external_override_count": int(self.external_override_count),
            "safety_mask_count": int(self.safety_mask_count),
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
        grad_option_logits: Array | None = None,
        **_: object,
    ) -> Array:
        if self.cache is None:
            raise RuntimeError(
                "Owned option controller backward called without cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        selected_option_idx = int(self.cache.selected_option_idx)
        grad_W2_option_leaf = np.zeros_like(self.W2_option_leaf)
        grad_b2_option_leaf = np.zeros_like(self.b2_option_leaf)
        grad_W2_option_leaf[selected_option_idx] += np.outer(
            grad_policy_logits,
            h_new,
        )
        grad_b2_option_leaf[selected_option_idx] += grad_policy_logits
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        if grad_option_logits is None:
            option_target = one_hot(selected_option_idx, self.option_dim)
            option_advantage = -grad_value
            grad_option_logits = 0.2 * option_advantage * (
                self.cache.option_probs - option_target
            )
        else:
            grad_option_logits = _clip_grad_logits(grad_option_logits, grad_clip)
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        dh = (
            self.W2_option_leaf[selected_option_idx].T @ grad_policy_logits
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

        self.W2_option_leaf -= lr * grad_W2_option_leaf
        self.b2_option_leaf -= lr * grad_b2_option_leaf
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
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
        state.update(
            {
                "owned_option_controller": True,
                "leaf_context_bias_scale": float(self.leaf_context_bias_scale),
                "W2_option_leaf": self.W2_option_leaf.copy(),
                "b2_option_leaf": self.b2_option_leaf.copy(),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        base_state_keys = set(super().state_dict().keys())
        expected_keys = base_state_keys | {
            "owned_option_controller",
            "leaf_context_bias_scale",
            "W2_option_leaf",
            "b2_option_leaf",
        }
        _validate_state_dict(
            state,
            expected_keys=expected_keys,
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
                "owned_option_controller": True,
                "leaf_context_bias_scale": self.leaf_context_bias_scale,
            },
            name=self.name,
        )
        super().load_state_dict({key: state[key] for key in base_state_keys})
        self.W2_option_leaf = _coerce_state_array(
            state,
            "W2_option_leaf",
            (self.option_dim, self.output_dim, self.hidden_dim),
            name=self.name,
        )
        self.b2_option_leaf = _coerce_state_array(
            state,
            "b2_option_leaf",
            (self.option_dim, self.output_dim),
            name=self.name,
        )
        self.external_override_count = 0
        self.safety_mask_count = 0

    def parameter_norm(self) -> float:
        parent_norm = super().parameter_norm()
        return float(
            np.sqrt(
                parent_norm**2
                + np.sum(self.W2_option_leaf**2)
                + np.sum(self.b2_option_leaf**2)
            )
        )

    def count_parameters(self) -> int:
        return int(
            super().count_parameters()
            + self.W2_option_leaf.size
            + self.b2_option_leaf.size
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


__all__ = [name for name in globals() if not name.startswith("__")]
