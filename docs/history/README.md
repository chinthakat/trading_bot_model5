# Development log

These files are a chronological record of work done on this project. They were
written as status notes at the time of the change, not as reference
documentation, so they describe how things stood on the day they were written
and may not match the current code.

They are kept because they explain *why* parts of the codebase look the way
they do. For documentation that is meant to be current, see [`../`](../) and
the [main README](../../README.md).

| File | What it records |
| --- | --- |
| [`simple-reward-system-integration.md`](simple-reward-system-integration.md) | The switch from the earlier `EnhancedRewardCalculator` to `SimpleRewardSystem` as the default reward path in `src/environment.py` and `src/train_memory_efficient.py`. Was `src/INTEGRATION_SUMMARY.md`. |
| [`custom-1-dataset-generation.md`](custom-1-dataset-generation.md) | The run that produced the `CUSTOM_1` synthetic dataset, and the checks made on it. Was `DATA_GEN/CUSTOM_1_SUMMARY.md`. The dataset itself is not in the repository; regenerate it with `DATA_GEN/btc_data_generator.py --market-type CUSTOM_1`. |

The reference description of the `CUSTOM_1` market phases moved to
[`../custom-1-dataset.md`](../custom-1-dataset.md).
