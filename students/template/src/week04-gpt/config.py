"""Configuration layer for week04-gpt.

Teaching goal:
- all tunable values live here, not scattered across main.py
- students can compare scenarios by editing one place
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    """Data shape for one benchmark scenario."""

    name: str
    n_samples: int
    n_features: int
    noise_std: float = 0.1


@dataclass(frozen=True)
class SolverConfig:
    """Hyperparameters for custom and API solvers."""

    gd_learning_rate: float = 0.01
    gd_max_iter: int = 1000
    gd_tol: float = 1e-6

    sgd_max_iter: int = 1000
    sgd_tol: float = 1e-3
    random_seed: int = 2026


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment config.

    Students can alter scenarios and reuse the same pipeline.
    """

    low_dim: ScenarioConfig
    high_dim: ScenarioConfig
    solver: SolverConfig


def get_default_config() -> ExperimentConfig:
    """Factory for classroom defaults."""
    return ExperimentConfig(
        low_dim=ScenarioConfig(name="low_dim", n_samples=10_000, n_features=10),
        high_dim=ScenarioConfig(name="high_dim", n_samples=10_000, n_features=2_000),
        solver=SolverConfig(),
    )
