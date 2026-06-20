"""Week 15 homework: Logistic Regression (GLM core).

Run from students/01_waz with:

    python3 src/week15/main.py

Covers:
  Task A — Why OLS fails for binary classification
  Task B — Sigmoid: mapping linear predictor to probability
  Task C — Bernoulli MLE & log loss vs MSE
  Task D — Log-odds and coefficient interpretation
  Task E — Classification metrics: confusion matrix, threshold tradeoff
  Task F — Regularized logistic regression (L1 vs L2)
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure src/ is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.metrics import (                                              # noqa: E402
    accuracy, binary_log_loss, confusion_counts, f1_score,
    precision, recall, threshold_metrics,
)
from utils.models import CustomLogisticRegression                       # noqa: E402
from utils.transformers import CustomStandardScaler                      # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
WEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK_DIR / "data"
RESULTS_DIR = WEEK_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset_outputs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_classification_data(
    n_samples: int = 500,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic binary classification dataset.

    True DGP:
        logit(p) = -1.0 + 2.5 * x1 - 1.8 * x2 + 0.8 * x3 + 0.0 * x_noise
        y ~ Bernoulli(p)

    Features:
        x1, x2, x3: informative features
        x_noise: pure noise (true coefficient = 0)
    """
    rng = np.random.default_rng(random_state)

    x1 = rng.normal(0, 1, n_samples)
    x2 = rng.normal(0, 1, n_samples)
    x3 = rng.normal(0, 1, n_samples)
    x_noise = rng.normal(0, 1, n_samples)

    logit = -1.0 + 2.5 * x1 - 1.8 * x2 + 0.8 * x3 + 0.0 * x_noise
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3, "x_noise": x_noise, "y": y,
    })


# ---------------------------------------------------------------------------
# Task A: Why OLS fails for binary classification
# ---------------------------------------------------------------------------

def run_ols_vs_logistic_on_classification() -> pd.DataFrame:
    """Show that OLS on 0/1 labels predicts values outside [0,1]."""
    df = make_classification_data(n_samples=200, random_state=42)
    X = df[["x1"]].to_numpy()
    y = df["y"].to_numpy()

    # OLS
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X, y)
    y_ols = ols.predict(X)

    # Logistic
    clf = CustomLogisticRegression(learning_rate=0.1, max_iter=2000, random_state=42)
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)

    # Count predictions outside [0,1]
    n_below = int(np.sum(y_ols < 0))
    n_above = int(np.sum(y_ols > 1))
    n_outside = n_below + n_above

    print(f"  OLS predictions < 0: {n_below}, > 1: {n_above}, outside [0,1]: {n_outside}/{len(y_ols)}")

    # Plot
    x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 200).reshape(-1, 1)
    ols_line = ols.predict(x_line)
    logit_line = clf.predict_proba(x_line)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(X.ravel(), y, s=15, alpha=0.5, label="Data (0/1 labels)")
    ax.plot(x_line, ols_line, color="red", linewidth=2, label="OLS fit")
    ax.plot(x_line, logit_line, color="blue", linewidth=2, label="Logistic prob.")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("Predicted value / Probability")
    ax.set_title("OLS vs Logistic Regression on Binary Classification")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure("A1_ols_vs_logistic.png")

    result = pd.DataFrame([{
        "model": "OLS", "min_pred": float(y_ols.min()), "max_pred": float(y_ols.max()),
        "n_below_0": n_below, "n_above_1": n_above,
    }, {
        "model": "Logistic", "min_pred": float(y_prob.min()), "max_pred": float(y_prob.max()),
        "n_below_0": int(np.sum(y_prob < 0)), "n_above_1": int(np.sum(y_prob > 1)),
    }])
    return result


# ---------------------------------------------------------------------------
# Task B: Sigmoid — mapping linear predictor to probability
# ---------------------------------------------------------------------------

def run_sigmoid_demo() -> None:
    """Plot sigmoid curve and show how linear predictor maps to probability."""
    eta = np.linspace(-6, 6, 300)
    sigmoid = 1.0 / (1.0 + np.exp(-eta))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eta, sigmoid, linewidth=2.5, color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7)
    ax.scatter([0], [0.5], color="red", s=80, zorder=5)
    ax.annotate("η=0, p=0.5", (0, 0.5), textcoords="offset points",
                xytext=(10, -15), fontsize=10, color="red")

    # Annotate extreme regions
    ax.annotate("p → 1 as η → +∞", (4, sigmoid[250]), textcoords="offset points",
                xytext=(10, 0), fontsize=9, color="darkgreen")
    ax.annotate("p → 0 as η → −∞", (-4, sigmoid[50]), textcoords="offset points",
                xytext=(-80, 0), fontsize=9, color="darkgreen")

    ax.set_xlabel("Linear predictor η = Xβ")
    ax.set_ylabel("Predicted probability p = σ(η)")
    ax.set_title("Sigmoid: Mapping Linear Predictor to Probability")
    ax.grid(alpha=0.3)
    save_figure("B1_sigmoid_curve.png")


# ---------------------------------------------------------------------------
# Task C: Bernoulli MLE — log loss vs MSE
# ---------------------------------------------------------------------------

def run_loss_comparison() -> None:
    """Compare squared error and negative log-likelihood for binary outcomes."""
    p_pred = np.linspace(0.01, 0.99, 200)

    # For y=1: MSE = (1-p)^2, log loss = -log(p)
    mse_y1 = (1 - p_pred) ** 2
    logloss_y1 = -np.log(p_pred)

    # For y=0: MSE = (0-p)^2 = p^2, log loss = -log(1-p)
    mse_y0 = p_pred ** 2
    logloss_y0 = -np.log(1 - p_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(p_pred, mse_y1, linewidth=2, label="MSE = (1-p)^2")
    ax.plot(p_pred, logloss_y1, linewidth=2, label="Log Loss = -log(p)")
    ax.set_xlabel("Predicted probability p")
    ax.set_ylabel("Loss")
    ax.set_title("y = 1: MSE vs Log Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(p_pred, mse_y0, linewidth=2, label="MSE = p^2")
    ax.plot(p_pred, logloss_y0, linewidth=2, label="Log Loss = -log(1-p)")
    ax.set_xlabel("Predicted probability p")
    ax.set_ylabel("Loss")
    ax.set_title("y = 0: MSE vs Log Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("Why Log Loss Penalizes Confident Mistakes More Severely")
    save_figure("C1_loss_comparison.png")


# ---------------------------------------------------------------------------
# Task D: Log-odds and coefficient interpretation
# ---------------------------------------------------------------------------

def run_logodds_interpretation() -> None:
    """Demonstrate log-odds, odds, and probability correspondence."""
    p_vals = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    odds = p_vals / (1 - p_vals)
    logodds = np.log(odds)

    # Table
    table_df = pd.DataFrame({
        "Probability p": [f"{v:.2f}" for v in p_vals],
        "Odds = p/(1-p)": [f"{v:.3f}" for v in odds],
        "Log-odds = log(p/(1-p))": [f"{v:.3f}" for v in logodds],
    })
    print("\n[Task D] Probability → Odds → Log-odds table:")
    print(table_df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_vals, logodds, marker="o", linewidth=2, color="steelblue")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Probability p")
    ax.set_ylabel("Log-odds = log(p/(1-p))")
    ax.set_title("Log-odds vs Probability")
    ax.grid(alpha=0.3)

    # Annotate regions
    ax.annotate("p > 0.5 → log-odds > 0", (0.75, 1.1), fontsize=9, color="darkgreen")
    ax.annotate("p < 0.5 → log-odds < 0", (0.25, -1.1), fontsize=9, color="darkred")
    ax.annotate("p = 0.5 → log-odds = 0", (0.5, 0), textcoords="offset points",
                xytext=(10, -15), fontsize=9, color="gray")
    save_figure("D1_logodds_vs_probability.png")

    return table_df


# ---------------------------------------------------------------------------
# Task E: Classification metrics & threshold tradeoff
# ---------------------------------------------------------------------------

def run_metrics_and_threshold() -> None:
    """Fit logistic regression, show confusion matrix and threshold curve."""
    df = make_classification_data(n_samples=500, random_state=42)
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )

    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = CustomLogisticRegression(learning_rate=0.2, max_iter=3000, random_state=42)
    clf.fit(X_train_s, y_train)

    y_prob = clf.predict_proba(X_test_s)
    y_pred_hard = (y_prob >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_counts(y_test, y_pred_hard)
    print(f"\n[Task E] Confusion Matrix (threshold=0.5): {cm}")
    print(f"  Accuracy : {accuracy(y_test, y_pred_hard):.4f}")
    print(f"  Precision: {precision(y_test, y_pred_hard):.4f}")
    print(f"  Recall   : {recall(y_test, y_pred_hard):.4f}")
    print(f"  F1       : {f1_score(y_test, y_pred_hard):.4f}")
    print(f"  Log Loss : {binary_log_loss(y_test, y_prob):.4f}")

    # Threshold sweep
    thresholds = np.linspace(0.05, 0.95, 50)
    t_metrics = [threshold_metrics(y_test, y_prob, t) for t in thresholds]
    t_df = pd.DataFrame(t_metrics)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t_df["threshold"], t_df["accuracy"], linewidth=2, label="Accuracy")
    ax.plot(t_df["threshold"], t_df["precision"], linewidth=2, label="Precision")
    ax.plot(t_df["threshold"], t_df["recall"], linewidth=2, label="Recall")
    ax.plot(t_df["threshold"], t_df["f1"], linewidth=2, label="F1")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default threshold")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metrics vs Classification Threshold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(alpha=0.3)
    save_figure("E1_threshold_metrics.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, color="steelblue", label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure("E2_roc_curve.png")


# ---------------------------------------------------------------------------
# Task F: Regularized logistic regression (L1 vs L2)
# ---------------------------------------------------------------------------

def run_regularized_logistic() -> pd.DataFrame:
    """Compare L1 and L2 regularized logistic regression."""
    df = make_classification_data(n_samples=500, random_state=42)
    X = df.drop(columns=["y"]).to_numpy()
    y = df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "None (no reg)": SklearnLogisticRegression(penalty=None, solver="lbfgs",
                                                    max_iter=5000, random_state=42),
        "L2 (C=0.1)": SklearnLogisticRegression(penalty="l2", C=0.1, solver="lbfgs",
                                                  max_iter=5000, random_state=42),
        "L2 (C=1.0)": SklearnLogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                                  max_iter=5000, random_state=42),
        "L1 (C=0.1)": SklearnLogisticRegression(penalty="l1", C=0.1, solver="saga",
                                                  max_iter=5000, random_state=42),
        "L1 (C=1.0)": SklearnLogisticRegression(penalty="l1", C=1.0, solver="saga",
                                                  max_iter=5000, random_state=42),
    }

    rows = []
    coefs = {}
    feature_names = ["x1", "x2", "x3", "x_noise"]

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        n_nonzero = int(np.sum(np.abs(model.coef_.ravel()) > 1e-6))

        rows.append({
            "model": name,
            "test_accuracy": accuracy(y_test, y_pred),
            "test_log_loss": binary_log_loss(y_test, y_prob),
            "n_nonzero_coefs": n_nonzero,
            "coef_sum_abs": float(np.sum(np.abs(model.coef_))),
        })
        coefs[name] = model.coef_.ravel()

    result = pd.DataFrame(rows)
    print("\n[Task F] Regularized logistic regression results:")
    print(result.to_string(index=False))

    # Coefficient plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: coefficients
    ax = axes[0]
    x_pos = np.arange(len(feature_names))
    width = 0.15
    for i, (name, coef) in enumerate(coefs.items()):
        ax.bar(x_pos + (i - 2) * width, coef, width=width, label=name[:20])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Coefficient")
    ax.set_title("Coefficients across regularization settings")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Right: accuracy vs complexity
    ax = axes[1]
    for _, row in result.iterrows():
        label = row["model"].split(" ")[0]
        ax.scatter(row["n_nonzero_coefs"], row["test_accuracy"], s=80,
                   label=row["model"])
    ax.set_xlabel("Number of non-zero coefficients")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Model Complexity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Regularized Logistic Regression: L1 vs L2")
    save_figure("F1_regularized_logistic.png")

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    ols_result: pd.DataFrame,
    logodds_table: pd.DataFrame,
    reg_result: pd.DataFrame,
) -> None:
    report = f"""# Week 15 Report: Logistic Regression (GLM)

## Task A: Why OLS Fails for Binary Classification

| Model | Min prediction | Max prediction | Below 0 | Above 1 |
|---|---|---|---|---|
| OLS | {ols_result['min_pred'].iloc[0]:.3f} | {ols_result['max_pred'].iloc[0]:.3f} | {ols_result['n_below_0'].iloc[0]} | {ols_result['n_above_1'].iloc[0]} |
| Logistic | {ols_result['min_pred'].iloc[1]:.3f} | {ols_result['max_pred'].iloc[1]:.3f} | {ols_result['n_below_0'].iloc[1]} | {ols_result['n_above_1'].iloc[1]} |

OLS predicts values outside [0,1], which have no natural probability interpretation.
Logistic regression's sigmoid output always stays in (0,1).

## Task B: Sigmoid Function

  sigma(eta) = 1 / (1 + exp(-eta))

- Input eta (linear predictor) is unbounded
- Output is always in (0,1) -> can be interpreted as a probability
- eta = 0 -> p = 0.5 (equal odds for both classes)

## Task C: Bernoulli MLE & Log Loss

For binary response Y in {{0,1}} with Y ~ Bernoulli(p),
the likelihood is p^y * (1-p)^(1-y).

Taking negative log gives the log loss: -[y * log(p) + (1-y) * log(1-p)].

Log loss penalizes **confident mistakes** much more severely than MSE:
- If y=1 but model predicts p=0.01 -> log loss ~ 4.6 (very high)
- The same mistake under MSE would be (1-0.01)^2 ~ 0.98 (moderate)

## Task D: Log-odds and Coefficient Interpretation

The logistic regression model is linear in **log-odds**:

  log(p / (1-p)) = X * beta

Therefore:
- beta_j > 0 -> odds increase with x_j (positive association)
- beta_j < 0 -> odds decrease with x_j (negative association)
- exp(beta_j) = odds ratio: multiplicative change in odds per unit increase in x_j

## Task E: Classification Metrics

- **Accuracy**: fraction of correct predictions
- **Precision**: TP / (TP + FP) -- how many predicted positives are real
- **Recall**: TP / (TP + FN) -- how many actual positives are found
- **F1**: harmonic mean of precision and recall
- **Log Loss**: measures probability quality, not just hard classification

The threshold controls the tradeoff: raising it favors precision, lowering it favors recall.

## Task F: Regularized Logistic Regression

| Model | Test Acc. | Log Loss | Non-zero coefs |
|---|---|---|---|
{chr(10).join(f"| {r['model']} | {r['test_accuracy']:.4f} | {r['test_log_loss']:.4f} | {r['n_nonzero_coefs']} |" for _, r in reg_result.iterrows())}

- **L1** drives coefficients to exactly zero -> variable selection
- **L2** shrinks all coefficients toward zero -> stabilization
- The noise feature x_noise has true coefficient = 0; L1 at C=0.1 should be most effective at zeroing it out
"""
    with open(RESULTS_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Week 15: Logistic Regression (GLM)")
    print("=" * 60)

    reset_outputs()

    # Task A
    print("\n[Task A] OLS vs Logistic on binary classification ...")
    ols_result = run_ols_vs_logistic_on_classification()
    print(ols_result.to_string(index=False))

    # Task B
    print("\n[Task B] Sigmoid demo ...")
    run_sigmoid_demo()

    # Task C
    print("\n[Task C] Log loss vs MSE ...")
    run_loss_comparison()

    # Task D
    print("\n[Task D] Log-odds interpretation ...")
    logodds_table = run_logodds_interpretation()

    # Task E
    print("\n[Task E] Metrics and threshold analysis ...")
    run_metrics_and_threshold()

    # Task F
    print("\n[Task F] Regularized logistic regression (L1 vs L2) ...")
    reg_result = run_regularized_logistic()

    # Report
    print("\n[Report] Generating ...")
    write_report(ols_result, logodds_table, reg_result)

    print("\n" + "=" * 60)
    print("Week 15 complete!")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Report : {RESULTS_DIR / 'report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
