"""Benchmark orchestration extracted from main.py."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression, SGDRegressor
import statsmodels.api as sm

from config import ScenarioConfig, SolverConfig
from data_generator import generate_regression_data
from solvers import AnalyticalSolver, GradientDescentSolver
from .metrics import mse
from .timing import timed


def run_scenario(cfg: ScenarioConfig, solver_cfg: SolverConfig) -> list[tuple[str, float, float]]:
    """Run one scenario and return rows: (solver_name, seconds, mse)."""

    X, y, _ = generate_regression_data(
        n_samples=cfg.n_samples,
        n_features=cfg.n_features,
        noise_std=cfg.noise_std,
        seed=solver_cfg.random_seed,
    )

    rows: list[tuple[str, float, float]] = []

    @timed
    def fit_custom_analytical() -> AnalyticalSolver:
        return AnalyticalSolver().fit(X, y)

    @timed
    def fit_custom_gd() -> GradientDescentSolver:
        return GradientDescentSolver(
            learning_rate=solver_cfg.gd_learning_rate,
            max_iter=solver_cfg.gd_max_iter,
            tol=solver_cfg.gd_tol,
        ).fit(X, y)

    @timed
    def fit_statsmodels_ols():
        return sm.OLS(y, X).fit()

    @timed
    def fit_sklearn_lr() -> LinearRegression:
        return LinearRegression(fit_intercept=False).fit(X, y)

    @timed
    def fit_sklearn_sgd() -> SGDRegressor:
        return SGDRegressor(
            fit_intercept=False,
            max_iter=solver_cfg.sgd_max_iter,
            tol=solver_cfg.sgd_tol,
            random_state=solver_cfg.random_seed,
        ).fit(X, y)

    model, sec = fit_custom_analytical()
    rows.append(("AnalyticalSolver", sec, mse(y, model.predict(X))))

    model, sec = fit_custom_gd()
    rows.append(("GradientDescentSolver", sec, mse(y, model.predict(X))))

    model, sec = fit_statsmodels_ols()
    rows.append(("statsmodels.OLS", sec, mse(y, model.predict(X))))

    model, sec = fit_sklearn_lr()
    rows.append(("sklearn.LinearRegression", sec, mse(y, model.predict(X))))

    model, sec = fit_sklearn_sgd()
    rows.append(("sklearn.SGDRegressor", sec, mse(y, model.predict(X))))

    return rows
