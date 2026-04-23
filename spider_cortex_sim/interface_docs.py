from __future__ import annotations

from typing import Sequence

from .interfaces import (
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    ALL_INTERFACES,
    LOCOMOTION_ACTIONS,
    VARIANT_MODULES,
    ORIENT_HEADINGS,
    get_interface_variant,
    get_variant_levels,
    interface_registry,
    interface_registry_fingerprint,
)
from .noise import NONE_NOISE_PROFILE
from .operational_profiles import DEFAULT_OPERATIONAL_PROFILE


def escape_markdown_cell(text: object) -> str:
    return (
        str(text)
        .replace("|", "&#124;")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def _variant_added_signals(current_signals: tuple[str, ...], previous_signals: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(signal for signal in current_signals if signal not in previous_signals)


def _render_candidate_signal_table(
    lines: list[str],
    signals: Sequence[tuple[str, str, str, str]],
) -> None:
    lines.extend(
        [
            "| # | Signal | Source | Range | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for index, (name, source, value_range, description) in enumerate(signals, start=1):
        lines.append(
            "| "
            f"{index} | `{escape_markdown_cell(name)}` | {escape_markdown_cell(source)} | "
            f"`{escape_markdown_cell(value_range)}` | {escape_markdown_cell(description)} |"
        )
    lines.append("")


def _render_a5_candidate_interfaces(lines: list[str], outputs_text: str) -> None:
    lines.extend(
        [
            "## A5 Candidate Interfaces",
            "",
            "- `A5` candidate modules are gated by the prerequisite checklist, module admission criteria, and operating rules in [architectural_ladder.md](./architectural_ladder.md).",
            "- These candidate entries are protocol-defined proposal surfaces for `A5_future_bio_refined`; they are not part of `interface_registry()`, do not affect save/load compatibility, and do not change the current registry fingerprint.",
            "- Placeholder signals documented here are allowed at the protocol level even when the runtime does not yet expose them. They exist so each future module proposal has a concrete interface hypothesis before implementation starts.",
            "- `homeostasis_center` remains the coarse A2 control surface. Any A5 homeostatic refinement is an alternative experiment path, not an automatic extension of the current proposer stack.",
            "",
            "### `olfactory_center`",
            "",
            "- Status: `proposal_only`",
            "- Candidate input count: `9`",
            "- Outputs: "
            f"`{outputs_text}`",
            "- Connection: feeds `action_center` with valence role "
            '`"foraging"` or `"threat"` based on the dominant smell-driven signal. '
            'If the current runtime arbitration vocabulary is kept unchanged, `"foraging"` would map to the existing `hunger` lane.',
            "- Experimental hypothesis: a dedicated smell-specialized proposer will improve odor-led food approach and odor-led threat avoidance in mixed-scent conditions, especially when direct vision is weak or absent.",
            "- Ablation test: `drop_olfactory_center` must selectively reduce performance on smell-dominant tasks such as `olfactory_visual_dissociation` without producing the same failure pattern as `drop_sensory_cortex` or `drop_alert_center`.",
            "- Coupling risk: `HIGH` because the candidate overlaps both the current smell integrator (`sensory_cortex`) and smell-aware threat handling (`alert_center`).",
            "",
        ]
    )
    _render_candidate_signal_table(
        lines,
        (
            (
                "food_smell_strength",
                "`sensory_cortex`",
                "0.0..1.0",
                "Strength of the local food odor field.",
            ),
            (
                "food_smell_dx",
                "`sensory_cortex`",
                "-1.0..1.0",
                "Horizontal food odor gradient.",
            ),
            (
                "food_smell_dy",
                "`sensory_cortex`",
                "-1.0..1.0",
                "Vertical food odor gradient.",
            ),
            (
                "predator_smell_strength",
                "`sensory_cortex` + `alert_center`",
                "0.0..1.0",
                "Strength of the local predator odor field.",
            ),
            (
                "predator_smell_dx",
                "`sensory_cortex`",
                "-1.0..1.0",
                "Horizontal predator odor gradient.",
            ),
            (
                "predator_smell_dy",
                "`sensory_cortex`",
                "-1.0..1.0",
                "Vertical predator odor gradient.",
            ),
            (
                "olfactory_predator_threat",
                "`alert_center`",
                "0.0..1.0",
                "Aggregated threat estimate from olfactory-oriented predators.",
            ),
            (
                "dominant_predator_olfactory",
                "`alert_center`",
                "0.0..1.0",
                "Indicator that olfactory hunters are the dominant current predator type.",
            ),
            (
                "smell_ambiguity",
                "derived from `sensory_cortex` + `alert_center`",
                "0.0..1.0",
                "Conflict score for mixed food and predator gradients or unstable odor localization.",
            ),
        ),
    )
    lines.extend(
        [
            "### `nociceptive_center`",
            "",
            "- Status: `proposal_only`",
            "- Candidate input count: `8`",
            "- Outputs: "
            f"`{outputs_text}`",
            '- Connection: feeds `action_center` with valence role `"threat"` as a body-state emergency proposer.',
            "- Experimental hypothesis: a pain-specialized proposer will shorten escape latency after damage or contact and reduce repeated contact events after the first hit, especially when the predator is no longer directly visible.",
            "- Ablation test: `drop_nociceptive_center` must degrade `pain_escape_latency` or post-contact survival more than smell-led detection tasks, while remaining distinguishable from the broader `drop_alert_center` failure pattern.",
            "- Coupling risk: `MEDIUM` because the candidate overlaps current threat signals but proposes a narrower post-contact recovery role than `alert_center`.",
            "",
        ]
    )
    _render_candidate_signal_table(
        lines,
        (
            (
                "recent_pain",
                "`sensory_cortex` + `alert_center`",
                "0.0..1.0",
                "Immediate recent pain caused by a predator hit.",
            ),
            (
                "recent_contact",
                "`sensory_cortex` + `alert_center`",
                "0.0..1.0",
                "Immediate recent physical contact with the predator.",
            ),
            (
                "health",
                "`sensory_cortex` + `alert_center`",
                "0.0..1.0",
                "Current body health available for damage-sensitive retreat control.",
            ),
            (
                "damage_rate",
                "derived from `health` delta",
                "0.0..1.0",
                "Short-horizon rate of health loss after contact.",
            ),
            (
                "pain_memory_dx",
                "derived from `alert_center` memory traces",
                "-1.0..1.0",
                "Horizontal direction of the last harmful contact or pain source.",
            ),
            (
                "pain_memory_dy",
                "derived from `alert_center` memory traces",
                "-1.0..1.0",
                "Vertical direction of the last harmful contact or pain source.",
            ),
            (
                "pain_memory_age",
                "derived from `alert_center` memory traces",
                "0.0..1.0",
                "Age of the last harmful contact memory.",
            ),
            (
                "contact_intensity",
                "derived from `recent_contact` + `recent_pain`",
                "0.0..1.0",
                "Combined severity estimate for recent bodily impact.",
            ),
        ),
    )
    lines.extend(
        [
            "### `arousal_center`",
            "",
            "- Status: `proposal_only`",
            "- Candidate input count: `10`",
            "- Outputs: "
            f"`{outputs_text}`",
            "- Connection: feeds `action_center` with valence role "
            '`"sleep"` when arousal is low and `"exploration"` when arousal is high, acting as a state-transition proposer rather than a direct threat or feeding specialist.',
            "- Experimental hypothesis: a dedicated arousal proposer will improve sleep-wake transitions, scan initiation, and exploratory reactivation under changing circadian and threat pressure without taking over the core sheltering job of `sleep_center`.",
            "- Ablation test: `drop_arousal_center` must selectively hurt `arousal_state_transition`, timely `ORIENT_*` use, or post-rest reactivation without collapsing into the same deficit profile as `drop_sleep_center`.",
            "- Coupling risk: `HIGH` because the candidate overlaps `sleep_center`, contextual arbitration, and activity-state features that may already be expressible without another proposer.",
            "",
        ]
    )
    _render_candidate_signal_table(
        lines,
        (
            (
                "fatigue",
                "`sensory_cortex` + `sleep_center` + `action_context`",
                "0.0..1.0",
                "Immediate fatigue pressure acting against wakeful exploration.",
            ),
            (
                "sleep_debt",
                "`sleep_center` + `action_context`",
                "0.0..1.0",
                "Accumulated need for restorative sleep.",
            ),
            (
                "night",
                "`sleep_center` + `action_context`",
                "0.0..1.0",
                "Nighttime state indicator.",
            ),
            (
                "day_night_transition",
                "derived from `day`/`night` history",
                "0.0..1.0",
                "Recent circadian transition marker for waking or winding down.",
            ),
            (
                "arousal_level",
                "derived from `sleep_center` + `action_context`",
                "0.0..1.0",
                "Latent wakefulness estimate combining rest and vigilance state.",
            ),
            (
                "threat_induced_arousal",
                "derived from `sensory_cortex` + `action_context`",
                "0.0..1.0",
                "Transient activation caused by distant threat evidence.",
            ),
            (
                "activity_level",
                "derived from `action_context.last_move_*` history",
                "0.0..1.0",
                "Short-horizon locomotor activity or quiescence estimate.",
            ),
            (
                "metabolic_state",
                "derived from `sensory_cortex` body-state signals",
                "0.0..1.0",
                "Compact scalar summarizing available internal energy and recovery pressure.",
            ),
            (
                "circadian_phase",
                "derived from day/night cycle phase",
                "0.0..1.0",
                "Normalized position within the circadian cycle.",
            ),
            (
                "rest_urgency",
                "derived from `sleep_center`",
                "0.0..1.0",
                "Immediate pressure to stay in shelter or resume rest.",
            ),
        ),
    )
    lines.extend(
        [
            "### `homeostasis_center` A5 Refinement Pathway",
            "",
            "- Current status: `homeostasis_center` is the coarse `A2_three_center` homeostatic control with `28` canonical signals and remains the rollback target when finer splits regress.",
            "- Refinement admission rule: consider an A5 homeostatic refinement only if the current `A4_current_full_modular` split still shows reproducible interference that the existing `hunger_center` and `sleep_center` separation does not resolve.",
            "- Refinement topology rule: treat `metabolic_regulation_center` and `thermoregulation_center` as alternative sixth-module experiments. Do not admit both at once without new evidence, so the first A5 comparison stays `modular_full` versus a hypothetical 6-module proposer stack.",
            "",
            "#### `metabolic_regulation_center`",
            "",
            "- Status: `proposal_only`, blocked unless `A4` homeostatic splits prove insufficient",
            "- Candidate input count: `14`",
            "- Outputs: "
            f"`{outputs_text}`",
            '- Connection: would feed `action_center` with valence role `"hunger"` as a narrower metabolic-drive proposer.',
            "",
        ]
    )
    _render_candidate_signal_table(
        lines,
        (
            ("hunger", "`homeostasis_center` subset", "0.0..1.0", "Primary metabolic deficit signal."),
            ("health", "`homeostasis_center` subset", "0.0..1.0", "Body viability constraint on foraging persistence."),
            ("on_food", "`homeostasis_center` subset", "0.0..1.0", "Immediate feeding context."),
            ("food_visible", "`homeostasis_center` subset", "0.0..1.0", "Direct food visibility gate."),
            ("food_certainty", "`homeostasis_center` subset", "0.0..1.0", "Confidence in the visible food cue."),
            ("food_smell_strength", "`homeostasis_center` subset", "0.0..1.0", "Food odor strength."),
            ("food_smell_dx", "`homeostasis_center` subset", "-1.0..1.0", "Horizontal food odor gradient."),
            ("food_smell_dy", "`homeostasis_center` subset", "-1.0..1.0", "Vertical food odor gradient."),
            ("food_trace_dx", "`homeostasis_center` subset", "-1.0..1.0", "Horizontal direction of the short food trace."),
            ("food_trace_dy", "`homeostasis_center` subset", "-1.0..1.0", "Vertical direction of the short food trace."),
            ("food_trace_strength", "`homeostasis_center` subset", "0.0..1.0", "Strength of the short food trace."),
            ("food_memory_dx", "`homeostasis_center` subset", "-1.0..1.0", "Horizontal direction of remembered food."),
            ("food_memory_dy", "`homeostasis_center` subset", "-1.0..1.0", "Vertical direction of remembered food."),
            ("food_memory_age", "`homeostasis_center` subset", "0.0..1.0", "Age of remembered food."),
        ),
    )
    lines.extend(
        [
            "#### `thermoregulation_center`",
            "",
            "- Status: `proposal_only`, blocked unless `A4` homeostatic splits prove insufficient",
            "- Candidate input count: `14`",
            "- Outputs: "
            f"`{outputs_text}`",
            '- Connection: would feed `action_center` with valence role `"sleep"` as a narrower shelter and recovery proposer.',
            "",
        ]
    )
    _render_candidate_signal_table(
        lines,
        (
            ("fatigue", "`homeostasis_center` subset", "0.0..1.0", "Immediate fatigue pressure."),
            ("on_shelter", "`homeostasis_center` subset", "0.0..1.0", "Current shelter occupancy."),
            ("day", "`homeostasis_center` subset", "0.0..1.0", "Daytime exposure proxy."),
            ("night", "`homeostasis_center` subset", "0.0..1.0", "Nighttime sheltering proxy."),
            ("sleep_phase_level", "`homeostasis_center` subset", "0.0..1.0", "Current sleep depth or rest phase."),
            ("rest_streak_norm", "`homeostasis_center` subset", "0.0..1.0", "Recent rest continuity."),
            ("sleep_debt", "`homeostasis_center` subset", "0.0..1.0", "Accumulated unmet rest demand."),
            ("shelter_role_level", "`homeostasis_center` subset", "0.0..1.0", "Current depth inside the shelter."),
            ("shelter_trace_dx", "`homeostasis_center` subset", "-1.0..1.0", "Horizontal direction of the short shelter trace."),
            ("shelter_trace_dy", "`homeostasis_center` subset", "-1.0..1.0", "Vertical direction of the short shelter trace."),
            ("shelter_trace_strength", "`homeostasis_center` subset", "0.0..1.0", "Strength of the short shelter trace."),
            ("shelter_memory_dx", "`homeostasis_center` subset", "-1.0..1.0", "Horizontal direction of remembered shelter."),
            ("shelter_memory_dy", "`homeostasis_center` subset", "-1.0..1.0", "Vertical direction of remembered shelter."),
            ("shelter_memory_age", "`homeostasis_center` subset", "0.0..1.0", "Age of remembered shelter."),
        ),
    )
    lines.extend(
        [
            "- Refinement experimental hypothesis: only pursue an A5 homeostatic refinement when the current split still shows a reproducible interference pattern, and the candidate sixth-module configuration improves that pattern with `Cohen's d >= 0.3` while preserving inherited gates.",
            "- Ablation design: compare `modular_full` against a hypothetical 6-module A5 configuration that introduces exactly one homeostatic refinement module at a time, then require the added module's removal or rollback comparison to isolate the intended interference reduction under the same `Cohen's d >= 0.3` admission threshold.",
            "- Coupling risk: `HIGH` because the current A4 architecture already decomposes coarse homeostasis into `hunger_center` and `sleep_center`, so additional splitting is likely to duplicate existing behavior unless the interference pattern is cleanly separable.",
            "",
        ]
    )


def render_interfaces_markdown() -> str:
    """
    Render the interface registry as a Markdown document.

    The document includes schema version and registry fingerprint, lists of proposal and context interfaces, compatibility and perception/action notes (including explicit memory signals), and one section per interface containing metadata and a table of its input signals (index, min, max, description).

    Returns:
        markdown (str): The rendered Markdown string; ends with a single trailing newline.
    """
    registry = interface_registry()
    perception_defaults = DEFAULT_OPERATIONAL_PROFILE.perception
    delay_noise_defaults = NONE_NOISE_PROFILE.delay
    movement_action_names = tuple(ACTION_DELTAS)
    orient_action_names = tuple(ORIENT_HEADINGS)
    movement_action_count = len(movement_action_names)
    total_action_count = len(LOCOMOTION_ACTIONS)
    movement_action_indices = ", ".join(
        f"`{name}`={ACTION_TO_INDEX[name]}" for name in movement_action_names
    )
    orient_action_index_items = [
        f"`{name}`={ACTION_TO_INDEX[name]}" for name in orient_action_names
    ]
    if not orient_action_index_items:
        orient_action_indices = "none"
    elif len(orient_action_index_items) == 1:
        orient_action_indices = orient_action_index_items[0]
    else:
        orient_action_indices = ", ".join(
            (*orient_action_index_items[:-1], f"and {orient_action_index_items[-1]}")
        )
    lines: list[str] = [
        "# Standardized Interfaces",
        "",
        "<!-- Generated from spider_cortex_sim.interface_docs.render_interfaces_markdown(); do not edit manually. -->",
        "",
        f"- Schema version: `{registry['schema_version']}`",
        f"- Registry fingerprint: `{interface_registry_fingerprint()}`",
        f"- Proposal interfaces: `{', '.join(registry['proposal_interfaces'])}`",
        f"- Context interfaces: `{', '.join(registry['context_interfaces'])}`",
        "",
        "## Compatibility",
        "",
        "- The current policy is `exact_match_required` for save/load.",
        "- Changes to the name, `observation_key`, signal order, outputs, or version require explicit incompatibility handling.",
        "- There is no automatic migration for older checkpoints.",
        "",
        "## Perception And Active Sensing",
        "",
        "### Breaking Change: Action Space",
        "",
        f"- The locomotion action space expanded from {movement_action_count} to {total_action_count} actions.",
        f"- Existing indices are unchanged: {movement_action_indices}.",
        f"- New active-sensing actions are {orient_action_indices}.",
        "",
        "### Foveal And Peripheral Vision",
        "",
        "- Visual targets are classified relative to the spider heading as `foveal`, `peripheral`, or `outside`.",
        f"- `fov_half_angle` defaults to `{perception_defaults['fov_half_angle']}` degrees and defines the foveal cone.",
        f"- The `peripheral_half_angle` defaults to `{perception_defaults['peripheral_half_angle']}` degrees and defines the outer visual cone.",
        f"- A default `peripheral_certainty_penalty` of `{perception_defaults['peripheral_certainty_penalty']}` reduces peripheral visual certainty.",
        "- Targets beyond `peripheral_half_angle` are outside the visual field (`visible=0`).",
        "- Heading zones are not exposed as observation fields; agents receive the resulting `visible`, `certainty`, occlusion, and direction signals.",
        "- Smell and olfactory gradients remain omnidirectional and are not gated by heading zones.",
        "",
        "### Scan Recency",
        "",
        "- `foveal_scan_age` reports how recently the current heading was actively scanned.",
        f"- `max_scan_age` defaults to `{perception_defaults['max_scan_age']}` ticks for normalizing scan age into `[0, 1]`.",
        "- A value of `0.0` means the current foveal heading was scanned this tick; `1.0` means stale or never scanned.",
        "- `PerceptTrace.heading_dx` and `PerceptTrace.heading_dy` record the body or gaze heading active when a food, shelter, or predator trace was refreshed.",
        "- Trace headings surface as `food_trace_heading_*`, `shelter_trace_heading_*`, and `predator_trace_heading_*` signals.",
        "- Scan recency is self-knowledge derived from the spider's own heading changes, not hidden world-state access.",
        "",
        "### ORIENT Actions",
        "",
        "- `ORIENT_*` actions update `heading_dx` and `heading_dy` from `ORIENT_HEADINGS`; they are not movement translations in `ACTION_DELTAS`.",
        "- After the heading update, the world refreshes the current tick's perception buffer immediately, before the normal end-of-step tick commit.",
        "- The refresh itself does not advance the tick or run hunger, fatigue, predator, memory, or perceptual-delay stages.",
        "- `ORIENT_*` actions still consume the step: after the refresh, the normal commit advances the tick and applies hunger, fatigue, predator, and perceptual-buffer progression.",
        "- `ORIENT_*` actions do not cause displacement and do not trigger motor slip.",
        "",
        "### Percept Traces",
        "",
        "- Trace signals carry target direction, decayed strength, and the heading active when the trace was refreshed.",
        "- `*_trace_heading_dx` and `*_trace_heading_dy` are scan context for the trace; they are not target directions.",
        "",
        "### Perceptual Delay",
        "",
        f"- `perceptual_delay_ticks` defaults to `{perception_defaults['perceptual_delay_ticks']}` and returns observations from the previous tick when buffered.",
        f"- `perceptual_delay_noise` defaults to `{perception_defaults['perceptual_delay_noise']}` and scales delay-noise effects.",
        f"- `NoiseConfig.delay.certainty_decay_per_tick` defaults to `{delay_noise_defaults['certainty_decay_per_tick']}` and decays certainty as observations age.",
        f"- `NoiseConfig.delay.direction_jitter_per_tick` defaults to `{delay_noise_defaults['direction_jitter_per_tick']}` and adds age-proportional jitter to delayed direction signals.",
        "- Delay noise uses a dedicated RNG stream to preserve reproducibility.",
        "- Trace signals such as `food_trace_strength` reflect the delayed percept state, not the current world state.",
        "",
        "### Perception Categories",
        "",
        "- Direct sight means visual certainty is above the debug threshold and the current scan is fresh.",
        "- Trace means a decayed percept trace is active after direct sight has gone stale or disappeared.",
        "- Delayed means perceptual delay is active and the current payload was not refreshed by an active scan.",
        "- Uncertain means no direct, trace, or delayed classification applies.",
        "- These categories are debug and audit labels from `_perception_category()`; they are not observation signals.",
        "",
        "### Explicit Memory Signals",
        "",
        "- `food_memory_dx`/`food_memory_dy`/`food_memory_age`, `shelter_memory_*`, `predator_memory_*`, and `escape_memory_*` are derived from perception-grounded memory slots.",
        "- The environment pipeline maintains mechanical timing only: refresh cycles age slots, enforce TTLs, and clear expired targets.",
        "- Data sources are limited to local visual perception, contact events, and movement history; the pipeline cannot inject information the spider has not perceived.",
        "- `spider_cortex_sim.memory.MEMORY_LEAKAGE_AUDIT` is the authoritative classification source for explicit memory ownership and leakage risk.",
        "",
        "## Interface Variants",
        "",
        "- Interface variants are diagnostic sufficiency probes for local tasks and organism-level ablations; they are not production save/load interfaces.",
        "- Use the variant ladder to ask how much signal a module needs before escalating to broader architectural diagnosis.",
        "",
    ]
    for module_name in VARIANT_MODULES:
        levels = get_variant_levels(module_name)
        progression = " -> ".join(
            f"`{get_interface_variant(module_name, level).name}`"
            for level in levels
        )
        lines.extend(
            [
                f"### `{module_name}`",
                "",
                f"- Progression: {progression}",
                "",
                "| Level | Interface | Signal Count | Added At This Level |",
                "| --- | --- | ---: | --- |",
            ]
        )
        previous_signals: tuple[str, ...] = tuple()
        for level in levels:
            interface = get_interface_variant(module_name, level)
            added_signals = _variant_added_signals(interface.signal_names, previous_signals)
            added_text = ", ".join(f"`{escape_markdown_cell(signal)}`" for signal in added_signals)
            if level == 4:
                added_text += " (canonical interface)"
            lines.append(
                f"| v{level} | `{interface.name}` | {interface.input_dim} | {added_text} |"
            )
            previous_signals = interface.signal_names
        lines.append("")

    _render_a5_candidate_interfaces(lines, ", ".join(LOCOMOTION_ACTIONS))

    for spec in ALL_INTERFACES:
        summary = spec.to_summary()
        compatibility = summary["compatibility"]
        lines.extend(
            [
                f"## `{spec.name}`",
                "",
                f"- Observation key: `{spec.observation_key}`",
                f"- Role: `{spec.role}`",
                f"- Version: `{spec.version}`",
                f"- Description: {spec.description}",
                f"- Outputs: `{', '.join(spec.outputs)}`",
                f"- Save policy: `{compatibility['save_policy']}`",
                "",
                "| # | Signal | Min | Max | Description |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for index, signal in enumerate(spec.inputs, start=1):
            signal_name = escape_markdown_cell(signal.name)
            signal_description = escape_markdown_cell(signal.description)
            lines.append(
                f"| {index} | `{signal_name}` | {signal.minimum:.1f} | {signal.maximum:.1f} | {signal_description} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
