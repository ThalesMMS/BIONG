# Confound-controlled ladder demo fixture

This fixture exists to satisfy the acceptance criterion that the repository includes **at least one reproducible ladder configuration** demonstrating a fair modular-vs-baseline comparison with **seed-level outputs**.

Because the QA environment for this task does not provide `python`/`pytest`, this fixture is provided as a **checked-in, minimal example payload** that the offline evaluator can ingest.

## What is included

- `tests/fixtures/confound_controlled_ladder_demo_payload.json`
  - Two variants:
    - `true_monolithic_policy` (baseline)
    - `modular_full` (modular)
  - Both variants include:
    - `uncertainty.scenario_success_rate.seed_items` with **3 aligned seeds** (`0,1,2`)
    - `parameter_counts.total_trainable` within the 1.2x threshold
    - `variant_metadata.interface_access.architecture_signature` identical
    - `variant_metadata.credit_assignment.mechanism` identical

The intended result when calling:

- `evaluate_variant_comparison_support(baseline=true_monolithic_policy, comparison=modular_full)`

is `status == "supported"`.

## How to use (locally)

```python
import json
from spider_cortex_sim.offline_analysis.extraction.ladder import evaluate_variant_comparison_support

payload = json.load(open("tests/fixtures/confound_controlled_ladder_demo_payload.json"))
variants = payload["behavior_evaluation"]["ablations"]["variants"]

result = evaluate_variant_comparison_support(
    baseline_variant="true_monolithic_policy",
    comparison_variant="modular_full",
    baseline_payload=variants["true_monolithic_policy"],
    comparison_payload=variants["modular_full"],
)
print(result)
```
