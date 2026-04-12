"""week04-gpt teaching template.

This package demonstrates a clean classroom-friendly engineering layout:
- central config for experiment parameters and solver hyperparameters
- isolated data generation, solver interface, and benchmark pipeline
- thin main entrypoint for orchestration only
"""

from .config import ExperimentConfig, SolverConfig, ScenarioConfig
from .utils.timing import timed

__all__ = [
    "ExperimentConfig",
    "SolverConfig",
    "ScenarioConfig",
    "timed",
]
