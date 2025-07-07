# scripts/

One-off scripts that were used to probe the environment and the model by hand.
They previously sat in the repository root.

**These are not automated tests.** Despite the `test_` prefix on most of them,
none of them import `pytest` and none of them contain a single `assert`. They
build a small synthetic DataFrame, step the environment or the model, and print
what happened for a human to read. There is no pass/fail exit code. Do not
point `pytest` at this directory.

Run one from the repository root:

```bash
python scripts/test_exploration_model.py
```

Most of these scripts add `../src` to `sys.path` themselves, so no installation
step is needed beyond the dependencies in `requirements.txt`. Two are
exceptions: `test_visualization_logging.py` adds `../GRAPH_GEN` instead, because
that is where the module it exercises lives, and `analyze_csv.py` does not touch
`sys.path` at all - it imports nothing from `src/` and only needs pandas.

| Script | What it prints |
| --- | --- |
| `test_exploration_model.py` | Action counts from `ExplorationTradingModel`, to see whether HOLD/BUY/SELL/CLOSE_ALL are all being tried. |
| `test_close_all_exploration.py` | The same, in scenarios constructed so that CLOSE_ALL is the sensible action. |
| `test_truly_discrete.py` | The same again, with the environment wrapped in a discrete action space. |
| `test_discrete_mapping.py` | How continuous policy output maps onto the discrete action indices. |
| `test_simplified_action_space.py` | The shape and bounds of the environment's action space. |
| `test_enhanced_rewards.py` | The reward breakdown the environment returns for a scripted sequence of trades. |
| `test_enhanced_logging.py` | The per-step trading log line, including market timestamp and reward breakdown. |
| `test_visualization_logging.py` | Whether `GRAPH_GEN/live_trade_visualizer_enhanced.py` writes its log file. |
| `debug_action_distribution.py` | The raw policy distribution and network outputs for a batch of observations. |
| `simple_action_debug.py` | A short dump of the configured action space, with no model involved. |
| `analyze_csv.py` | Summary statistics for a trade-analysis CSV. Pass the CSV path as an argument - the built-in default points at a file that is not in the repository. |
| `test_individual_trade_closing.py` | Nothing - the file is empty. It was committed as a placeholder and never written. |
