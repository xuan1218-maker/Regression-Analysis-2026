"""Data generation pseudocode for linear regression benchmarks."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def generate_regression_data(
    n_samples: int,
    n_features: int,
    noise_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, y, beta_true.

    Pseudocode notes:
    - draw X from standard normal
    - prepend intercept column for stable shape handling
    - generate beta_true and y = X @ beta_true + noise
    """
    rng = np.random.default_rng(seed)

    X_raw = rng.standard_normal((n_samples, n_features))
    X = np.column_stack([np.ones(n_samples), X_raw])

    beta_true = rng.standard_normal(n_features + 1)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = X @ beta_true + noise

    return X, y, beta_true
