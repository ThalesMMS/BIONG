from __future__ import annotations

from .interfaces import (
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    ALL_INTERFACES,
    LOCOMOTION_ACTIONS,
    ORIENT_HEADINGS,
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
    ]
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
