from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
SEED = 20260523


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(errors**2)))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.abs(errors)))


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.2 * x) + 0.15 * x


def make_noisy_sample(
    n: int = 120,
    noise_std: float = 0.35,
    x_low: float = -3.0,
    x_high: float = 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED if seed is None else seed)
    x = np.sort(rng.uniform(x_low, x_high, n))
    y = true_function(x) + rng.normal(0, noise_std, n)
    return x.reshape(-1, 1), y


def polynomial_model(degree: int) -> Pipeline:
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linreg", LinearRegression()),
        ]
    )


def fit_degree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, Pipeline]:
    model = polynomial_model(degree)
    model.fit(x_train, y_train)
    return model.predict(x_eval), model


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def stage_candidate_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    x_grid: np.ndarray,
    y_true_grid: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    degrees = [1, 4, 15]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, degree in zip(axes, degrees):
        y_grid_pred, model = fit_degree(x_train, y_train, x_grid, degree)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        train_rmse = calculate_rmse(y_train, train_pred)
        test_rmse = calculate_rmse(y_test, test_pred)

        records.append(
            {
                "degree": degree,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
            }
        )

        ax.scatter(x_train[:, 0], y_train, s=18, alpha=0.6, label="train")
        ax.scatter(x_test[:, 0], y_test, s=18, alpha=0.6, label="test")
        ax.plot(
            x_grid[:, 0],
            y_true_grid,
            color="black",
            linewidth=2,
            linestyle="--",
            label="truth",
        )
        ax.plot(
            x_grid[:, 0],
            y_grid_pred,
            color="#d62728",
            linewidth=2.5,
            label=f"degree={degree}",
        )
        ax.set_title(
            f"degree={degree}\ntrain RMSE={train_rmse:.3f}, test RMSE={test_rmse:.3f}"
        )
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
    axes[-1].legend(loc="upper left", fontsize=10)
    fig.suptitle("Candidate models: which one would you ship?", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "candidate_models.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(records)


def stage_error_curves(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, int, int]:
    records: list[dict[str, float | int]] = []
    for degree in range(1, 19):
        model = polynomial_model(degree)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        records.append(
            {
                "degree": degree,
                "train_rmse": calculate_rmse(y_train, train_pred),
                "test_rmse": calculate_rmse(y_test, test_pred),
            }
        )

    error_df = pd.DataFrame(records)
    error_df["generalization_gap"] = error_df["test_rmse"] - error_df["train_rmse"]
    best_degree = int(error_df.loc[error_df["test_rmse"].idxmin(), "degree"])
    largest_gap_degree = int(
        error_df.loc[error_df["generalization_gap"].idxmax(), "degree"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        error_df["degree"],
        error_df["train_rmse"],
        marker="o",
        linewidth=2.2,
        label="train RMSE",
    )
    ax.plot(
        error_df["degree"],
        error_df["test_rmse"],
        marker="o",
        linewidth=2.2,
        label="test RMSE",
    )
    ax.axvline(
        best_degree,
        color="gray",
        linestyle="--",
        alpha=0.75,
        label=f"best degree={best_degree}",
    )
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE")
    ax.set_title("Training vs test error across model complexity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return error_df, best_degree, largest_gap_degree


def stage_variance_demo() -> pd.DataFrame:
    x_eval = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_eval_true = true_function(x_eval.ravel())

    degree_predictions: dict[int, np.ndarray] = {}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, degree in zip(axes, [2, 15]):
        collected_predictions = []
        for sample_idx in range(14):
            x_sample, y_sample = make_noisy_sample(
                n=35, noise_std=0.35, seed=1000 + sample_idx
            )
            y_pred, _ = fit_degree(x_sample, y_sample, x_eval, degree)
            collected_predictions.append(y_pred)
            ax.plot(x_eval[:, 0], y_pred, alpha=0.30, linewidth=1.4)

        stacked_predictions = np.vstack(collected_predictions)
        degree_predictions[degree] = stacked_predictions
        ax.plot(
            x_eval[:, 0],
            y_eval_true,
            color="black",
            linewidth=3,
            linestyle="--",
            label="truth",
        )
        ax.set_title(f"Repeated fits with degree={degree}")
        ax.set_xlabel("x")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("predicted y")
    fig.suptitle("Variance demo: how much do the curves wobble?", y=1.03)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "variance_demo.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for degree, predictions in degree_predictions.items():
        pointwise_std = predictions.std(axis=0)
        rows.append(
            {
                "degree": degree,
                "mean_prediction_std": float(pointwise_std.mean()),
                "max_prediction_std": float(pointwise_std.max()),
            }
        )
    return pd.DataFrame(rows)


def stage_loss_comparison() -> pd.DataFrame:
    y_true = np.array([100, 102, 98, 101, 99, 103, 100, 97], dtype=float)
    y_pred_clean = np.array([101, 101, 99, 100, 100, 102, 99, 98], dtype=float)
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[-1] = 80

    metrics_df = pd.DataFrame(
        {
            "scenario": ["clean prediction", "one large outlier"],
            "RMSE": [
                calculate_rmse(y_true, y_pred_clean),
                calculate_rmse(y_true, y_pred_outlier),
            ],
            "MAE": [
                calculate_mae(y_true, y_pred_clean),
                calculate_mae(y_true, y_pred_outlier),
            ],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].scatter(range(len(y_true)), y_true, s=85, label="true")
    axes[0].scatter(range(len(y_true)), y_pred_outlier, s=85, label="pred")
    axes[0].set_title("One outlier changes one prediction")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("value")
    axes[0].legend()

    width = 0.35
    x = np.arange(len(metrics_df))
    axes[1].bar(x - width / 2, metrics_df["RMSE"], width=width, label="RMSE")
    axes[1].bar(x + width / 2, metrics_df["MAE"], width=width, label="MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_df["scenario"], rotation=10)
    axes[1].set_title("Which metric gets hit harder?")
    axes[1].set_ylabel("metric value")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "loss_outlier_comparison.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    return metrics_df


def format_table(df: pd.DataFrame, decimals: int = 3) -> str:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(decimals)

    headers = list(rounded.columns)
    separator = ["---"] * len(headers)
    rows = [headers, separator]

    for row in rounded.itertuples(index=False, name=None):
        rows.append([str(value) for value in row])

    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_summary(
    candidate_df: pd.DataFrame,
    error_df: pd.DataFrame,
    best_degree: int,
    largest_gap_degree: int,
    variance_df: pd.DataFrame,
    loss_df: pd.DataFrame,
) -> None:
    best_candidate = int(candidate_df.loc[candidate_df["test_rmse"].idxmin(), "degree"])
    summary = f"""# Week 12 Example Summary

## Three key conclusions
1. Lower training error does not guarantee better generalization.
2. High-complexity models can become unstable across different training samples.
3. RMSE reacts more strongly than MAE when one prediction error becomes very large.

## Candidate model comparison
{format_table(candidate_df)}

A reasonable first guess for deployment among the three candidates is `degree={best_candidate}` because it has the best test-side behavior among the shown models.

## Full complexity sweep
Best test RMSE occurs at `degree={best_degree}`.
Largest generalization gap occurs at `degree={largest_gap_degree}`.

{format_table(error_df.loc[:, ["degree", "train_rmse", "test_rmse", "generalization_gap"]].head(10))}

## Variance demo summary
{format_table(variance_df)}

The higher-degree model has larger prediction dispersion, which is the visible signature of high variance.

## RMSE vs MAE under an outlier
{format_table(loss_df)}

RMSE is hit harder because it squares large errors, while MAE grows linearly.

## Natural transition to next week
If high model complexity creates unstable models, then regularization becomes a natural next step: it intentionally constrains the model to trade a little bias for lower variance.
"""
    (RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    print("[Stage 1] Generating synthetic data and candidate model plots...")
    x, y = make_noisy_sample(n=120, noise_std=0.35, seed=7)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=42
    )
    x_grid = np.linspace(-3.2, 3.2, 500).reshape(-1, 1)
    y_true_grid = true_function(x_grid.ravel())

    candidate_df = stage_candidate_models(
        x_train, x_test, y_train, y_test, x_grid, y_true_grid
    )

    print("[Stage 2] Sweeping model complexity...")
    error_df, best_degree, largest_gap_degree = stage_error_curves(
        x_train, x_test, y_train, y_test
    )

    print("[Stage 3] Repeating sampling to visualize variance...")
    variance_df = stage_variance_demo()

    print("[Stage 4] Comparing RMSE and MAE under an outlier...")
    loss_df = stage_loss_comparison()

    print("[Stage 5] Writing markdown summary...")
    write_summary(
        candidate_df, error_df, best_degree, largest_gap_degree, variance_df, loss_df
    )
    print(f"Done. Figures saved to: {FIGURES_DIR}")
    print(f"Summary saved to: {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
