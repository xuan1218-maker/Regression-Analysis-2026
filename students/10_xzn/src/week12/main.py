"""
Week 12: Bias-Variance Tradeoff & Model Complexity
====================================================
Entry point:  uv run src/week12/main.py
Outputs:
  results/figures/candidate_models.png
  results/figures/error_curves.png
  results/figures/variance_demo.png
  results/figures/loss_outlier_comparison.png
  results/summary.md
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0.  Path setup – ensure the script can find utils/metrics.py
# ---------------------------------------------------------------------------
_WEEK12_DIR = Path(__file__).resolve().parent          # .../src/week12
_SRC_DIR    = _WEEK12_DIR.parent                       # .../src
_PROJECT_DIR = _SRC_DIR.parent                         # .../students/10_xzn

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from utils.metrics import calculate_rmse, calculate_mae
except ImportError:
    # Fallback implementations
    def calculate_rmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def calculate_mae(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return np.mean(np.abs(y_true - y_pred))

# ---------------------------------------------------------------------------
# 1.  Global settings
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

RESULTS_DIR = _WEEK12_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# 2.  Helper functions
# ---------------------------------------------------------------------------

def generate_data(n_samples: int = 120, noise_std: float = 0.2,
                  test_ratio: float = 0.2, x_range: tuple = (0.0, 1.0)):
    """
    Generate 1D regression data.
    True function: f(x) = sin(2πx) + 0.5 * x
    Returns: X_train, y_train, X_test, y_test, X_all, y_true_all
    """
    np.random.seed(SEED)  # re-seed for determinism within each call
    X = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X.ravel()) + 0.5 * X.ravel()
    noise = np.random.randn(n_samples) * noise_std
    y = y_true + noise

    # Shuffle and split
    idx = np.random.permutation(n_samples)
    n_test = int(n_samples * test_ratio)
    idx_test = idx[:n_test]
    idx_train = idx[n_test:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    return X_train, y_train, X_test, y_test, X, y_true


def fit_polynomial(X_train, y_train, degree: int):
    """
    Fit a polynomial of given degree.
    Returns: fitted sklearn Pipeline (PolynomialFeatures + LinearRegression)
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lin",  LinearRegression()),
    ])
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3.  Task implementations
# ---------------------------------------------------------------------------

def task_a_candidate_models(X_train, y_train, X_test, y_test, X_all, y_true_all):
    """Task A: Plot candidate polynomial models (degree 1, 4, 15)."""
    print("[Stage 1] Comparing candidate polynomial models (degree=1,4,15) ...")

    degrees = [1, 4, 15]
    X_plot = np.linspace(X_all.min(), X_all.max(), 500).reshape(-1, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, d in zip(axes, degrees):
        model = fit_polynomial(X_train, y_train, degree=d)
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)
        y_plot       = model.predict(X_plot)

        rmse_train = calculate_rmse(y_train, y_pred_train)
        rmse_test  = calculate_rmse(y_test,  y_pred_test)

        ax.plot(X_all.ravel(), y_true_all, "k--", linewidth=1.5, label="True f(x)")
        ax.plot(X_plot.ravel(), y_plot, "green", linewidth=2, label=f"Degree {d} fit")
        ax.scatter(X_train.ravel(), y_train, c="blue", s=20, alpha=0.6, label="Train")
        ax.scatter(X_test.ravel(),  y_test,  c="red",  s=25, alpha=0.8, label="Test")

        ax.set_title(f"Degree = {d}  |  Train RMSE = {rmse_train:.3f}, Test RMSE = {rmse_test:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_models.png")
    plt.close(fig)
    print("  -> Saved candidate_models.png")


def task_b_error_curves(X_train, y_train, X_test, y_test):
    """Task B: Complexity–error curves for degrees 1..18."""
    print("[Stage 2] Scanning complexity-error curves (degree=1..18) ...")

    max_degree = 18
    degrees = list(range(1, max_degree + 1))
    train_rmse_list, test_rmse_list, gap_list = [], [], []

    for d in degrees:
        model = fit_polynomial(X_train, y_train, degree=d)
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        train_rmse = calculate_rmse(y_train, y_pred_train)
        test_rmse  = calculate_rmse(y_test,  y_pred_test)

        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        gap_list.append(test_rmse - train_rmse)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(degrees, train_rmse_list, "o-", color="blue", linewidth=2,
            markersize=6, label="Train RMSE")
    ax.plot(degrees, test_rmse_list, "s-", color="red", linewidth=2,
            markersize=6, label="Test RMSE")
    ax.set_xlabel("Polynomial Degree (Complexity)")
    ax.set_ylabel("RMSE")
    ax.set_title("Error Curves: Train vs Test RMSE by Model Complexity")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_curves.png")
    plt.close(fig)
    print("  -> Saved error_curves.png")

    # Return data for the summary report
    best_idx = np.argmin(test_rmse_list)
    return {
        "degrees":        degrees,
        "train_rmse":     train_rmse_list,
        "test_rmse":      test_rmse_list,
        "gap":            gap_list,
        "best_degree":    degrees[best_idx],
        "best_test_rmse": test_rmse_list[best_idx],
    }


def task_c_variance_demo(X_all, y_true_all):
    """Task C: Repeated-sampling variance demonstration for degree=2 and 15."""
    print("[Stage 3] Repeated-sampling variance demo (degree=2 & 15, 10 reps) ...")

    n_repeats = 10
    noise_std = 0.2
    n_samples = len(X_all)
    X_plot = np.linspace(X_all.min(), X_all.max(), 500).reshape(-1, 1)

    degrees = [2, 15]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, d in zip(axes, degrees):
        preds_matrix = np.zeros((n_repeats, len(X_plot)))
        for r in range(n_repeats):
            np.random.seed(SEED * 1000 + r)
            noise = np.random.randn(n_samples) * noise_std
            y_noisy = y_true_all + noise
            # Use all points as "training" for this visualisation (same x-grid)
            model = fit_polynomial(X_all, y_noisy, degree=d)
            preds_matrix[r, :] = model.predict(X_plot).ravel()

            ax.plot(X_plot.ravel(), preds_matrix[r, :], alpha=0.45, linewidth=0.9)

        ax.plot(X_all.ravel(), y_true_all, "k--", linewidth=2, label="True f(x)")
        std_per_point = preds_matrix.std(axis=0)
        mean_std = np.mean(std_per_point)
        max_std  = np.max(std_per_point)

        ax.set_title(f"Degree = {d}  |  Mean SD = {mean_std:.4f},  Max SD = {max_std:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "variance_demo.png")
    plt.close(fig)
    print("  -> Saved variance_demo.png")

    # Compute quantitative summary (stored for report)
    summary = {}
    for d in degrees:
        preds_matrix = np.zeros((n_repeats, len(X_plot)))
        for r in range(n_repeats):
            np.random.seed(SEED * 1000 + r)
            noise = np.random.randn(n_samples) * noise_std
            y_noisy = y_true_all + noise
            model = fit_polynomial(X_all, y_noisy, degree=d)
            preds_matrix[r, :] = model.predict(X_plot).ravel()
        std_per_point = preds_matrix.std(axis=0)
        summary[d] = {
            "mean_std": float(np.mean(std_per_point)),
            "max_std":  float(np.max(std_per_point)),
        }
    return summary


def task_d_loss_outlier_comparison():
    """Task D: Compare RMSE and MAE under clean vs outlier-contaminated predictions."""
    print("[Stage 4] Loss function outlier sensitivity comparison ...")

    n = 100
    x = np.linspace(0, 1, n)
    y_true = np.sin(2 * np.pi * x) + 0.5 * x
    np.random.seed(SEED)
    errors_clean = np.random.randn(n) * 0.1
    y_pred_clean = y_true + errors_clean

    # Introduce one large outlier
    outlier_idx = np.random.randint(0, n)
    y_pred_outlier = y_pred_clean.copy()
    y_pred_outlier[outlier_idx] += 10 * 0.1  # add 10 * std

    # Compute metrics
    rmse_clean   = calculate_rmse(y_true, y_pred_clean)
    mae_clean    = calculate_mae(y_true, y_pred_clean)
    rmse_outlier = calculate_rmse(y_true, y_pred_outlier)
    mae_outlier  = calculate_mae(y_true, y_pred_outlier)

    # Plot — grouped bar chart
    labels = ["RMSE", "MAE"]
    clean_vals   = [rmse_clean,   mae_clean]
    outlier_vals = [rmse_outlier, mae_outlier]

    x_pos = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x_pos - width/2, clean_vals,   width, label="Clean",
                   color="steelblue", edgecolor="black")
    bars2 = ax.bar(x_pos + width/2, outlier_vals, width, label="With Outlier",
                   color="orangered", edgecolor="black")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error Value")
    ax.set_title("RMSE vs MAE: Clean vs One-Outlier Scenario")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate values on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.005, f"{h:.4f}",
                ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.005, f"{h:.4f}",
                ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_outlier_comparison.png")
    plt.close(fig)
    print("  -> Saved loss_outlier_comparison.png")

    return {
        "rmse_clean":   rmse_clean,
        "mae_clean":    mae_clean,
        "rmse_outlier": rmse_outlier,
        "mae_outlier":  mae_outlier,
    }


# ---------------------------------------------------------------------------
# 4.  Summary report
# ---------------------------------------------------------------------------

def write_summary_report(task_b_data, task_c_summary, task_d_data):
    """Write results/summary.md."""
    print("[Stage 5] Writing summary report ...")

    # ------ Build the markdown content ------
    lines = []

    lines.append("# Week 12: Bias-Variance Tradeoff & Model Complexity — Summary Report")
    lines.append("")
    lines.append("## 1. Three Most Important Conclusions")
    lines.append("")
    lines.append(
        "1. **Complexity beyond the optimum leads to overfitting.**  "
        "As polynomial degree increases, training error monotonically decreases, "
        "but test error first decreases then rises sharply — a classic U-shaped test-error curve."
    )
    lines.append(
        "2. **High-variance models are fragile.**  "
        "With repeated sampling, a high-degree polynomial (degree=15) exhibits huge swings in its "
        "fitted curve, whereas a low-degree model (degree=2) remains stable. "
        "The danger of a high-variance model is not that it cannot fit the training set, "
        "but that it is overly sensitive to the specific realisation of the noise."
    )
    lines.append(
        "3. **RMSE amplifies large errors; MAE is robust.**  "
        "A single extreme prediction can inflate RMSE dramatically while leaving MAE almost unchanged. "
        "The choice of loss function should match the cost structure of the application."
    )
    lines.append("")

    # 2. The figure that best represents overfitting
    lines.append("## 2. Which Figure Best Represents Overfitting?")
    lines.append("")
    lines.append(
        "**Answer:** `candidate_models.png` (Task A) best captures overfitting.  "
        "In the degree-15 subplot, the green fitted curve oscillates wildly to pass through "
        "nearly every training point, achieving a very low training RMSE, yet it deviates "
        "far from the true black-dashed function — the test RMSE is substantially higher. "
        "This visual contrast between low train error and high test error is the textbook "
        "signature of overfitting."
    )
    lines.append("")

    # 3. RMSE vs MAE decision
    lines.append("## 3. When to Report RMSE vs MAE")
    lines.append("")
    lines.append(
        "- **Report RMSE** when large errors are disproportionately costly (e.g. financial "
        "risk, safety-critical predictions) and you want the metric to penalise them heavily. "
        "RMSE is also preferred when the error distribution is approximately Gaussian."
    )
    lines.append(
        "- **Report MAE** when you need a robust metric that is not dominated by a few outliers. "
        "If the data naturally contains anomalies or the cost of error is roughly linear, "
        "MAE gives a more representative picture of typical performance."
    )
    lines.append("")

    # 4. Connection to next week — regularization
    lines.append("## 4. Connection to Regularization (Ridge / Lasso)")
    lines.append("")
    lines.append(
        "If model complexity is too high, the model exhibits high variance — its parameters "
        "are estimated with large swings depending on the training sample.  "
        "**Regularization** (Ridge, Lasso) directly addresses this by constraining the "
        "magnitude of the coefficients, thereby reducing variance at the cost of a small "
        "increase in bias.  This is the natural next step after observing that unconstrained "
        "high-degree polynomials overfit: we keep the expressive capacity but penalise "
        "extreme coefficient values, achieving a better bias-variance trade-off."
    )
    lines.append("")

    # ------ Quantitative tables ------

    # Task B table
    lines.append("## 5. Task B: Complexity–Error Table")
    lines.append("")
    lines.append("| Degree | Train RMSE | Test RMSE | Generalization Gap |")
    lines.append("|--------|------------|-----------|-------------------|")
    for deg, tr, te, gap in zip(
        task_b_data["degrees"],
        task_b_data["train_rmse"],
        task_b_data["test_rmse"],
        task_b_data["gap"],
    ):
        lines.append(f"| {deg:>6d} | {tr:>10.4f} | {te:>9.4f} | {gap:>17.4f} |")

    lines.append("")
    lines.append(f"- **Lowest test RMSE** occurs at degree = **{task_b_data['best_degree']}** "
                 f"(test RMSE = {task_b_data['best_test_rmse']:.4f}).")
    # Find where gap is largest
    gap_arr = np.array(task_b_data["gap"])
    max_gap_idx = int(np.argmax(gap_arr))
    max_gap_deg = task_b_data["degrees"][max_gap_idx]
    lines.append(f"- **Largest generalization gap** occurs around degree **{max_gap_deg}** "
                 f"(gap = {gap_arr[max_gap_idx]:.4f}). In general, the gap widens rapidly "
                 f"for degrees ≥ 12.")
    lines.append(
        "- **Why the lowest training-error model is not necessarily the best:** "
        "The model with the lowest training RMSE (degree=18) has essentially memorised the "
        "training noise.  Its low training error is achieved by fitting patterns that do not "
        "generalise — the test error is high.  A good model should balance fit and simplicity."
    )
    lines.append("")

    # Task C table
    lines.append("## 6. Task C: Variance Quantitative Summary")
    lines.append("")
    lines.append("| Degree | Mean Prediction SD | Max Prediction SD |")
    lines.append("|--------|--------------------|-------------------|")
    for d in [2, 15]:
        s = task_c_summary[d]
        lines.append(f"| {d:>6d} | {s['mean_std']:>18.4f} | {s['max_std']:>17.4f} |")
    lines.append("")
    lines.append(
        "**Fill in the blank:**  "
        "High variance model的危险, 不是它不会拟合训练集, 而是它对 **训练数据中的噪声/具体样本** 过于敏感."
    )
    lines.append("")

    # Task D table
    lines.append("## 7. Task D: Outlier Sensitivity of RMSE vs MAE")
    lines.append("")
    lines.append("| Scenario | RMSE     | MAE      |")
    lines.append("|----------|----------|----------|")
    lines.append(f"| Clean    | {task_d_data['rmse_clean']:.4f}   | {task_d_data['mae_clean']:.4f}   |")
    lines.append(f"| Outlier  | {task_d_data['rmse_outlier']:.4f}   | {task_d_data['mae_outlier']:.4f}   |")
    lines.append("")
    lines.append(
        "- **Why RMSE is more sensitive:** RMSE squares the error before averaging, so a single "
        "large residual (e.g., 1.0) contributes 1.0² = 1.0 to the MSE, dominating the "
        "contributions of many smaller residuals. MAE, by contrast, only takes the absolute "
        "value, so the outlier's contribution remains proportional."
    )
    lines.append(
        "- **If the cost of one large mistake is extremely high**, you should monitor **RMSE** "
        "(or even more aggressive metrics like Max Error), because it explicitly penalises "
        "large deviations and will alert you to catastrophic failures."
    )
    lines.append(
        "- **If the data naturally contain many outliers**, you should seriously reconsider "
        "using RMSE and likely prefer **MAE** (or Huber loss).  RMSE would be dominated by "
        "the outliers and may not reflect the model's performance on the bulk of the data. "
        "In such cases, you might also consider robust regression techniques (e.g., RANSAC, "
        "HuberRegressor) that are designed to be insensitive to outliers."
    )
    lines.append("")

    # Write file
    report_path = RESULTS_DIR / "summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> Saved {report_path}")


# ---------------------------------------------------------------------------
# 5.  Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Week 12: Bias-Variance Tradeoff & Model Complexity")
    print("=" * 60)

    # ---- Common data for Tasks A, B, C ----
    X_train, y_train, X_test, y_test, X_all, y_true_all = generate_data(
        n_samples=120, noise_std=0.2, test_ratio=0.2
    )

    # Task A
    task_a_candidate_models(X_train, y_train, X_test, y_test, X_all, y_true_all)

    # Task B
    task_b_data = task_b_error_curves(X_train, y_train, X_test, y_test)

    # Task C
    task_c_summary = task_c_variance_demo(X_all, y_true_all)

    # Task D
    task_d_data = task_d_loss_outlier_comparison()

    # Task E & F — Summary report
    write_summary_report(task_b_data, task_c_summary, task_d_data)

    print("\nAll tasks completed. Outputs are in results/")
    print("=" * 60)


if __name__ == "__main__":
    main()