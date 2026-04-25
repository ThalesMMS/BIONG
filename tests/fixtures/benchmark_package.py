import csv
from pathlib import Path


class BenchmarkPackageFixtures:
    def _uncertainty(
        self,
        *,
        mean: float = 0.75,
        seeds: list[float] | None = None,
    ) -> dict[str, object]:
        """
        Constructs a synthetic uncertainty object for a metric.
        
        Parameters:
            mean (float): Central estimate for the metric. Defaults to 0.75.
            seeds (list[float] | None): Seed values used to derive confidence-interval endpoints and counts.
                If None, defaults to [0.5, 1.0].
        
        Returns:
            dict[str, object]: A dictionary containing:
                - "mean": the provided mean
                - "ci_lower": minimum of `seeds`
                - "ci_upper": maximum of `seeds`
                - "std_error": standard error (fixed at 0.1)
                - "n_seeds": number of seed values
                - "confidence_level": confidence level (0.95)
                - "seed_values": a list copy of the seed values
        """
        if seeds is None:
            seeds = [0.5, 1.0]
        if not seeds:
            raise ValueError("_uncertainty requires at least one seed value.")
        return {
            "mean": mean,
            "ci_lower": min(seeds),
            "ci_upper": max(seeds),
            "std_error": 0.1,
            "n_seeds": len(seeds),
            "confidence_level": 0.95,
            "seed_values": list(seeds),
        }

    def _summary(self) -> dict[str, object]:
        """
        Builds a synthetic, nested benchmark payload used by tests.
        
        Returns:
            summary (dict[str, object]): A dictionary representing a complete synthetic benchmark package. Top-level keys include:
                - config: baseline configuration (brain, architecture, budget).
                - checkpointing: selection and metric used for checkpoint choice.
                - parameter_counts: per-network counts, total_trainable, and proportions.
                - behavior_evaluation: evaluation summary, suite entries (including per-scenario uncertainty and seed-level rows), ablations (reference_variant and per-variant payloads with their own summaries, seed-level rows, uncertainties, config, and parameter_counts), capacity_sweeps (profiles with capacity_profile and variants), and claim_tests (package-level claim results with uncertainties, deltas, and effect sizes).
            The returned payload contains multiple synthetic seed-level and scenario-seed-level rows for different ablation conditions and uses the fixture's _uncertainty helper to populate confidence-interval fields.
        """
        seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.5,
                "condition": "modular_full",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 1.0,
                "condition": "modular_full",
            },
        ]
        scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in seed_level
        ]
        monolithic_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.25,
                "condition": "monolithic_policy",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.5,
                "condition": "monolithic_policy",
            },
        ]
        monolithic_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in monolithic_seed_level
        ]
        true_monolithic_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.2,
                "condition": "true_monolithic_policy",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.3,
                "condition": "true_monolithic_policy",
            },
        ]
        true_monolithic_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in true_monolithic_seed_level
        ]
        three_center_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.55,
                "condition": "three_center_modular",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.65,
                "condition": "three_center_modular",
            },
        ]
        three_center_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in three_center_seed_level
        ]
        four_center_seed_level = [
            {
                "metric_name": "scenario_success_rate",
                "seed": 1,
                "value": 0.65,
                "condition": "four_center_modular",
            },
            {
                "metric_name": "scenario_success_rate",
                "seed": 2,
                "value": 0.75,
                "condition": "four_center_modular",
            },
        ]
        four_center_scenario_seed_level = [
            {
                **row,
                "scenario": "night_rest",
            }
            for row in four_center_seed_level
        ]
        def make_variant_payload(
            *,
            architecture: str,
            scenario_success_rate: float,
            seed_rows: list[dict[str, object]],
            scenario_seed_rows: list[dict[str, object]],
            parameter_counts: dict[str, object],
            uncertainty_seeds: list[float] | None = None,
        ) -> dict[str, object]:
            """
            Create a variant payload dict representing a behavior-evaluation variant for the "night_rest" scenario.
            
            Parameters:
                architecture (str): Architecture label to place in `config.architecture`.
                scenario_success_rate (float): Success rate to set for both scenario and episode summaries.
                seed_rows (list[dict]): Seed-level metric rows for the variant (global/experiment-level).
                scenario_seed_rows (list[dict]): Seed-level metric rows annotated with the scenario (scenario-specific).
                parameter_counts (dict): Parameter count summary to include under `parameter_counts`.
                uncertainty_seeds (list[float] | None): Optional seed values to compute the uncertainty; when omitted a default uncertainty is used.
            
            Returns:
                dict[str, object]: A payload containing `summary`, `suite` (with `night_rest` including `uncertainty` and `seed_level`), top-level `seed_level`, `uncertainty` for the primary metric, `config`, and `parameter_counts`.
            """
            uncertainty = (
                self._uncertainty(mean=scenario_success_rate, seeds=uncertainty_seeds)
                if uncertainty_seeds is not None
                else self._uncertainty(mean=scenario_success_rate)
            )
            return {
                "summary": {
                    "scenario_success_rate": scenario_success_rate,
                    "episode_success_rate": scenario_success_rate,
                },
                "suite": {
                    "night_rest": {
                        "success_rate": scenario_success_rate,
                        "uncertainty": {"success_rate": uncertainty},
                        "seed_level": scenario_seed_rows,
                    }
                },
                "seed_level": seed_rows,
                "uncertainty": {"scenario_success_rate": uncertainty},
                "config": {"architecture": architecture},
                "parameter_counts": parameter_counts,
            }

        return {
            "config": {
                "brain": {
                    "name": "modular_full",
                    "architecture": "modular",
                },
                "budget": {
                    "profile": "paper",
                    "benchmark_strength": "paper",
                    "resolved": {"episodes": 10},
                }
            },
            "checkpointing": {
                "selection": "best",
                "metric": "scenario_success_rate",
            },
            "parameter_counts": {
                "architecture": "modular",
                "by_network": {
                    "visual_cortex": 120,
                    "sensory_cortex": 120,
                    "alert_center": 120,
                    "hunger_center": 120,
                    "sleep_center": 120,
                    "arbitration_network": 64,
                    "action_center": 96,
                    "motor_cortex": 80,
                },
                "total_trainable": 840,
                "proportions": {
                    "visual_cortex": 120 / 840,
                    "sensory_cortex": 120 / 840,
                    "alert_center": 120 / 840,
                    "hunger_center": 120 / 840,
                    "sleep_center": 120 / 840,
                    "arbitration_network": 64 / 840,
                    "action_center": 96 / 840,
                    "motor_cortex": 80 / 840,
                },
            },
            "behavior_evaluation": {
                "summary": {
                    "scenario_success_rate": 0.75,
                    "episode_success_rate": 0.75,
                },
                "suite": {
                    "night_rest": {
                        "success_rate": 0.75,
                        "uncertainty": {"success_rate": self._uncertainty()},
                        "seed_level": scenario_seed_level,
                    }
                },
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": make_variant_payload(
                            architecture="modular",
                            scenario_success_rate=0.75,
                            seed_rows=seed_level,
                            scenario_seed_rows=scenario_seed_level,
                            parameter_counts={
                                "architecture": "modular",
                                "by_network": {
                                    "visual_cortex": 120,
                                    "sensory_cortex": 120,
                                    "alert_center": 120,
                                    "hunger_center": 120,
                                    "sleep_center": 120,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 840,
                                "proportions": {
                                    "visual_cortex": 120 / 840,
                                    "sensory_cortex": 120 / 840,
                                    "alert_center": 120 / 840,
                                    "hunger_center": 120 / 840,
                                    "sleep_center": 120 / 840,
                                    "arbitration_network": 64 / 840,
                                    "action_center": 96 / 840,
                                    "motor_cortex": 80 / 840,
                                },
                            },
                        ),
                        "monolithic_policy": make_variant_payload(
                            architecture="monolithic",
                            scenario_success_rate=0.375,
                            seed_rows=monolithic_seed_level,
                            scenario_seed_rows=monolithic_scenario_seed_level,
                            uncertainty_seeds=[0.25, 0.5],
                            parameter_counts={
                                "architecture": "monolithic",
                                "by_network": {
                                    "monolithic_policy": 1692,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 1932,
                                "proportions": {
                                    "monolithic_policy": 1692 / 1932,
                                    "arbitration_network": 64 / 1932,
                                    "action_center": 96 / 1932,
                                    "motor_cortex": 80 / 1932,
                                },
                            },
                        ),
                        "true_monolithic_policy": make_variant_payload(
                            architecture="true_monolithic",
                            scenario_success_rate=0.25,
                            seed_rows=true_monolithic_seed_level,
                            scenario_seed_rows=true_monolithic_scenario_seed_level,
                            uncertainty_seeds=[0.2, 0.3],
                            parameter_counts={
                                "architecture": "true_monolithic",
                                "by_network": {"true_monolithic_policy": 1500},
                                "total_trainable": 1500,
                                "proportions": {"true_monolithic_policy": 1.0},
                            },
                        ),
                        "three_center_modular": make_variant_payload(
                            architecture="modular",
                            scenario_success_rate=0.6,
                            seed_rows=three_center_seed_level,
                            scenario_seed_rows=three_center_scenario_seed_level,
                            uncertainty_seeds=[0.55, 0.65],
                            parameter_counts={
                                "architecture": "modular",
                                "by_network": {
                                    "perception_center": 180,
                                    "homeostasis_center": 150,
                                    "threat_center": 150,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 720,
                                "proportions": {
                                    "perception_center": 180 / 720,
                                    "homeostasis_center": 150 / 720,
                                    "threat_center": 150 / 720,
                                    "arbitration_network": 64 / 720,
                                    "action_center": 96 / 720,
                                    "motor_cortex": 80 / 720,
                                },
                            },
                        ),
                        "four_center_modular": make_variant_payload(
                            architecture="modular",
                            scenario_success_rate=0.7,
                            seed_rows=four_center_seed_level,
                            scenario_seed_rows=four_center_scenario_seed_level,
                            uncertainty_seeds=[0.65, 0.75],
                            parameter_counts={
                                "architecture": "modular",
                                "by_network": {
                                    "visual_cortex": 120,
                                    "sensory_cortex": 120,
                                    "homeostasis_center": 150,
                                    "threat_center": 150,
                                    "arbitration_network": 64,
                                    "action_center": 96,
                                    "motor_cortex": 80,
                                },
                                "total_trainable": 780,
                                "proportions": {
                                    "visual_cortex": 120 / 780,
                                    "sensory_cortex": 120 / 780,
                                    "homeostasis_center": 150 / 780,
                                    "threat_center": 150 / 780,
                                    "arbitration_network": 64 / 780,
                                    "action_center": 96 / 780,
                                    "motor_cortex": 80 / 780,
                                },
                            },
                        ),
                    },
                },
                "capacity_sweeps": {
                    "profiles": {
                        "current": {
                            "capacity_profile": {
                                "profile": "current",
                                "version": "v1",
                                "module_hidden_dims": {
                                    "visual_cortex": 32,
                                    "sensory_cortex": 28,
                                    "hunger_center": 26,
                                    "sleep_center": 24,
                                    "alert_center": 28,
                                    "perception_center": 36,
                                    "homeostasis_center": 28,
                                    "threat_center": 28,
                                },
                                "integration_hidden_dim": 32,
                                "scale_factor": 1.0,
                            },
                            "variants": {
                                "modular_full": {
                                    "summary": {
                                        "scenario_success_rate": 0.75,
                                        "episode_success_rate": 0.75,
                                    },
                                    "suite": {},
                                    "config": {"architecture": "modular"},
                                    "parameter_counts": {
                                        "total": 840,
                                        "per_network": {
                                            "visual_cortex": 120,
                                            "sensory_cortex": 120,
                                        },
                                    },
                                    "approximate_compute_cost": {
                                        "total": 2000,
                                        "unit": "approx_forward_macs",
                                        "per_network": {
                                            "visual_cortex": 400,
                                            "sensory_cortex": 400,
                                        },
                                    },
                                }
                            },
                        },
                        "large": {
                            "capacity_profile": {
                                "profile": "large",
                                "version": "v1",
                                "module_hidden_dims": {
                                    "visual_cortex": 64,
                                    "sensory_cortex": 56,
                                    "hunger_center": 52,
                                    "sleep_center": 48,
                                    "alert_center": 56,
                                    "perception_center": 72,
                                    "homeostasis_center": 56,
                                    "threat_center": 56,
                                },
                                "integration_hidden_dim": 64,
                                "scale_factor": 2.0,
                            },
                            "variants": {
                                "modular_full": {
                                    "summary": {
                                        "scenario_success_rate": 0.8,
                                        "episode_success_rate": 0.8,
                                    },
                                    "suite": {},
                                    "config": {"architecture": "modular"},
                                    "parameter_counts": {
                                        "total": 1680,
                                        "per_network": {
                                            "visual_cortex": 240,
                                            "sensory_cortex": 240,
                                        },
                                    },
                                    "approximate_compute_cost": {
                                        "total": 4000,
                                        "unit": "approx_forward_macs",
                                        "per_network": {
                                            "visual_cortex": 800,
                                            "sensory_cortex": 800,
                                        },
                                    },
                                }
                            },
                        },
                    }
                },
                "claim_tests": {
                    "claims": {
                        "package_claim": {
                            "status": "passed",
                            "passed": True,
                            "reference_value": 0.5,
                            "comparison_values": {"modular_full": 0.75},
                            "delta": {"modular_full": 0.25},
                            "effect_size": {"modular_full": 0.25},
                            "reference_uncertainty": self._uncertainty(mean=0.5),
                            "comparison_uncertainty": {
                                "modular_full": self._uncertainty(mean=0.75)
                            },
                            "delta_uncertainty": {
                                "modular_full": self._uncertainty(mean=0.25)
                            },
                            "effect_size_uncertainty": {
                                "modular_full": self._uncertainty(mean=0.25)
                            },
                            "cohens_d": {"modular_full": 1.0},
                            "effect_magnitude": {"modular_full": "large"},
                            "primary_metric": "scenario_success_rate",
                        }
                    }
                },
            },
        }

    def _write_behavior_csv(self, path: Path) -> None:
        """
        Write a minimal behavior CSV file used by tests.
        
        The CSV contains the header fields "simulation_seed", "scenario", "success", "ablation_variant", and "eval_reflex_scale" and two data rows matching the fixture manifest seed count.
        
        Parameters:
            path (Path): Filesystem path where the CSV will be created/overwritten.
        """
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=(
                    "simulation_seed",
                    "scenario",
                    "success",
                    "ablation_variant",
                    "eval_reflex_scale",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "simulation_seed": 1,
                    "scenario": "night_rest",
                    "success": "true",
                    "ablation_variant": "modular_full",
                    "eval_reflex_scale": 0.0,
                }
            )
            writer.writerow(
                {
                    "simulation_seed": 2,
                    "scenario": "night_rest",
                    "success": "true",
                    "ablation_variant": "modular_full",
                    "eval_reflex_scale": 0.0,
                }
            )

    def _manifest_kwargs(self) -> dict[str, object]:
        """
        Builds a synthetic manifest kwargs dictionary for a benchmark package used in tests.
        
        Returns:
            dict: Manifest keyword arguments containing:
                - package_version: package version string.
                - created_at: ISO 8601 creation timestamp.
                - command_metadata: mapping with `argv` list for the command invocation.
                - budget_profile: mapping with `profile` name (e.g., "paper").
                - checkpoint_selection: mapping with `selection` strategy (e.g., "best").
                - environment: mapping with environment metadata (`git_commit`, `git_tag`, `git_dirty`, `python_version`, `platform`).
                - contents: list of file descriptors each with `path`, `bytes`, and `sha256`.
                - seed_count: number of seeds included in the package.
                - confidence_level: numeric confidence level used in analyses.
                - limitations: list of limitation notes (e.g., ["Synthetic fixture."]).
        """
        return {
            "package_version": "1.0",
            "created_at": "2026-04-15T00:00:00+00:00",
            "command_metadata": {"argv": []},
            "budget_profile": {"profile": "paper"},
            "checkpoint_selection": {"selection": "best"},
            "environment": {
                "git_commit": "abc123",
                "git_tag": None,
                "git_dirty": False,
                "python_version": "3.11.0",
                "platform": "linux",
            },
            "contents": [{"path": "report.json", "bytes": 2, "sha256": "abc"}],
            "seed_count": 2,
            "confidence_level": 0.95,
            "limitations": ["Synthetic fixture."],
        }

    def _rows_with_numeric_ci(self, payload: object) -> list[dict[str, object]]:
        """
        Collects all dictionaries within a nested payload that contain numeric `ci_lower` and `ci_upper` fields.
        
        Recursively traverses dictionaries and lists in `payload` and returns every dict object where both `ci_lower` and `ci_upper` are instances of int or float. Non-dict/list values are ignored.
        
        Parameters:
            payload (object): Nested structure of dicts and lists to search.
        
        Returns:
            list[dict[str, object]]: A list of dicts from the payload that contain numeric `ci_lower` and `ci_upper`.
        """
        rows: list[dict[str, object]] = []

        def visit(value: object) -> None:
            """
            Recursively traverses a nested structure and appends any dicts that contain numeric `ci_lower` and `ci_upper` keys to the outer-scope `rows` list.
            
            Parameters:
                value (object): A possibly nested combination of dicts and lists to visit. Dictionaries with numeric `ci_lower` and `ci_upper` are collected by appending them to the surrounding `rows` list as a side effect.
            """
            if isinstance(value, dict):
                ci_lower = value.get("ci_lower")
                ci_upper = value.get("ci_upper")
                if isinstance(ci_lower, (int, float)) and isinstance(
                    ci_upper,
                    (int, float),
                ):
                    rows.append(value)
                for nested_value in value.values():
                    visit(nested_value)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payload)
        return rows

    def _conclusion_input(
        self,
        *,
        ratio: float = 1.0,
        any_interface_insufficient: bool = False,
        delta: float | None = 0.3,
        ci_lower: float | None = 0.1,
        ci_upper: float | None = 0.5,
        cohens_d: float | None = 0.6,
    ) -> dict[str, object]:
        """
        Builds a synthetic conclusion input containing capacity-matching, interface sufficiency, and overall comparison statistics.
        
        Parameters:
            ratio (float): Capacity ratio used to determine whether capacity is matched; values <= 1.2 mark capacity as matched.
            any_interface_insufficient (bool): Whether any evaluated interface is considered insufficient.
            delta (float | None): Observed difference for the overall comparison; when None the overall comparison is marked unavailable.
            ci_lower (float | None): Lower bound of the confidence interval for `delta`.
            ci_upper (float | None): Upper bound of the confidence interval for `delta`.
            cohens_d (float | None): Cohen's d effect size for the overall comparison.
        
        Returns:
            dict[str, object]: A dict with keys:
              - "capacity_matched_comparison": {"available": True, "capacity_matched": bool, "ratio": float}
              - "interface_sufficiency_results": {"available": True, "any_interface_insufficient": bool}
              - "overall_comparison": {"available": bool, "delta": float | None, "delta_ci_lower": float | None, "delta_ci_upper": float | None, "cohens_d": float | None}
        """
        return {
            "capacity_matched_comparison": {
                "available": True,
                "capacity_matched": ratio <= 1.2,
                "ratio": ratio,
            },
            "interface_sufficiency_results": {
                "available": True,
                "any_interface_insufficient": any_interface_insufficient,
            },
            "overall_comparison": {
                "available": delta is not None,
                "delta": delta,
                "delta_ci_lower": ci_lower,
                "delta_ci_upper": ci_upper,
                "cohens_d": cohens_d,
            },
        }
