# week04-gpt teaching scaffold

This folder is a classroom-oriented pseudocode scaffold for Week 04.

## Why this layout
- Keep entrypoint thin: main.py only orchestrates
- Keep knobs centralized: config.py stores all experiment params/hyperparams
- Keep algorithms isolated: solvers.py focuses on model logic
- Keep data concerns isolated: data_generator.py
- Keep pipeline helpers modular: utils/ handles timing, metrics, benchmark, display

## Suggested teaching flow
1. Start from config.py and modify one hyperparameter.
2. Run main.py and observe all solver outcomes.
3. Edit one solver implementation in solvers.py.
4. Compare behavior without touching pipeline orchestration.

## Folder map
- __init__.py: module boundary and exported config objects
- config.py: scenario and hyperparameter dataclasses
- data_generator.py: synthetic regression data
- solvers.py: analytical and GD pseudocode
- utils/timing.py: decorator-based runtime measurement
- utils/metrics.py: mse and other metrics
- utils/benchmark.py: scenario benchmark workflow
- utils/display.py: table/header output helpers
- main.py: thin orchestration layer
