# Module Layout Refactor Notes - 2026-06-04

This refactor improves the recent structural split without changing algorithms, config values, CLI behavior, checkpoint formats, metric names, artifact paths, or public APIs.

## Renamed Runtime Modules

The former `spider_cortex_sim/brain/runtime_partN.py` implementation files were renamed by responsibility. Each old path remains as a compatibility facade that imports the same private mixin class name.

| Old implementation path | New implementation path | Responsibility |
| --- | --- | --- |
| `brain/runtime_part1.py` | `brain/runtime_b0_b6_semantics.py` | B0-B6 semantic action helpers |
| `brain/runtime_part2.py` | `brain/runtime_b6_b14_semantics.py` | B6-B14 semantic action helpers |
| `brain/runtime_part3.py` | `brain/runtime_b14_b23_semantics.py` | B14-B23 semantic action helpers |
| `brain/runtime_part4.py` | `brain/runtime_b23_b31_semantics.py` | B23-B31 semantic action helpers |
| `brain/runtime_part5.py` | `brain/runtime_b32_b40_semantics.py` | B32-B40 semantic action helpers |
| `brain/runtime_part6.py` | `brain/runtime_b40_b48_semantics.py` | B40-B48 semantic action helpers |
| `brain/runtime_part7.py` | `brain/runtime_b48_b56_semantics.py` | B48-B56 semantic action helpers |
| `brain/runtime_part8.py` | `brain/runtime_b56_b62_compute.py` | B56-B62 semantics and compute-cost helpers |
| `brain/runtime_part9.py` | `brain/runtime_control_modes.py` | action modes, reflex scaling, and B-series selection |
| `brain/runtime_part10.py` | `brain/runtime_action_loop.py` | primary `act()` loop |
| `brain/runtime_part11.py` | `brain/runtime_introspection.py` | value estimates, signatures, norms, and parameter counts |

## Moved NN Internals

The flat private `_nn_*` implementation modules moved into `spider_cortex_sim/nn_internal/`:

- `_nn_shared.py` -> `nn_internal/shared.py`
- `_nn_proposal.py` -> `nn_internal/proposal.py`
- `_nn_motor.py` -> `nn_internal/motor.py`
- `_nn_recurrent_monolithic.py` -> `nn_internal/recurrent_monolithic.py`
- `_nn_option_controller.py` -> `nn_internal/option_controller.py`
- `_nn_affordance_geometry.py` -> `nn_internal/affordance_geometry.py`
- `_nn_affordance_position_core.py` -> `nn_internal/affordance_position.py`
- `_nn_affordance_position_gating.py` -> `nn_internal/affordance_position_gating.py`
- `_nn_affordance_position_forward.py` -> `nn_internal/affordance_position_forward.py`
- `_nn_affordance_position_backward.py` -> `nn_internal/affordance_position_backward.py`
- `_nn_deep_arbitration.py` -> `nn_internal/deep_arbitration.py`

`spider_cortex_sim/nn.py` now re-exports from the internal modules without `import *`. Legacy `_nn_*` paths remain dynamic compatibility facades.

## Moved B-Series Evolution Internals

The flat private `_b_series_evolution_*` implementation modules moved into `spider_cortex_sim/b_series_evolution_internal/` with the long private prefix removed:

- `_b_series_evolution_shared.py` -> `b_series_evolution_internal/shared.py`
- `_b_series_evolution_constants.py` -> `b_series_evolution_internal/constants.py`
- `_b_series_evolution_checkpoint_paths.py` -> `b_series_evolution_internal/checkpoint_paths.py`
- `_b_series_evolution_config_builders.py` -> `b_series_evolution_internal/config_builders.py`
- `_b_series_evolution_cli.py` -> `b_series_evolution_internal/cli.py`
- `_b_series_evolution_gates_*.py` -> `b_series_evolution_internal/gates_*.py`
- `_b_series_evolution_sequences_*.py` -> `b_series_evolution_internal/sequences_*.py`
- `_b_series_evolution_sequence_b62.py` -> `b_series_evolution_internal/sequence_b62.py`
- `_b_series_evolution_requires_sequences_b1_b5.py` -> `b_series_evolution_internal/requires_sequences_b1_b5.py`

`spider_cortex_sim/b_series_evolution.py` remains the public facade and still sets exported objects' `__module__` to `spider_cortex_sim.b_series_evolution`. Legacy `_b_series_evolution_*` paths remain dynamic compatibility facades.

## Compatibility Facades

Compatibility facades intentionally remain in these places:

- `spider_cortex_sim/brain/runtime_part1.py` through `runtime_part11.py`
- `spider_cortex_sim/_nn_*.py`
- `spider_cortex_sim/_b_series_evolution_*.py`
- `spider_cortex_sim/nn.py`
- `spider_cortex_sim/b_series_evolution.py`

These facades preserve old import paths while routing implementation imports to responsibility-based modules and internal subpackages.

## Import-Star Cleanup

Removed wildcard imports from `spider_cortex_sim/nn.py` and replaced them with explicit internal module aggregation.

Remaining wildcard imports are concentrated in compatibility facades or tightly coupled internal decomposition modules where replacing them would require large explicit dependency lists. Those can be reduced later one cluster at a time, but doing so in this refactor would add risk without changing behavior.

## Verification

Commands run after the refactor:

- `python3 -m compileall spider_cortex_sim tests` - passed
- package import sweep over tracked `spider_cortex_sim/**/*.py` modules - passed, `imported=332`
- `python3 -m spider_cortex_sim --help` - passed
- tracked line-count gate - passed, `tracked_python_files=577`, `max_lines=1953 spider_cortex_sim/b_series.py`, `over_2000=0`
- `python3 -m unittest discover tests` - passed, `6271` tests, `4` skipped

## Remaining Risks

- Private implementation classes and functions imported from their new internal modules may report the new internal module in `__module__`. Public B-series facade exports still use the canonical `spider_cortex_sim.b_series_evolution` module name.
- Some B-series internal module names still use B-range batches, such as `sequences_b38_b42.py`, because the file boundaries already match the generated B-series workflow ranges. Splitting those further would be a separate behavior-preserving cleanup.
- Remaining internal `import *` usage is deliberate for this pass. Replacing it safely should be done per subsystem with focused import-surface checks.
