import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calculate_mae, calculate_mape, calculate_rmse
from utils.models import GradientDescentOLS
from utils.transformers import CustomStandardScaler


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = REPO_ROOT / "homework" / "week09" / "data" / "dirty_marketing.csv"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    target_col = "Sales"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X_df = pd.get_dummies(
        df.drop(columns=[target_col]),
        columns=["Region"],
        drop_first=True,
        dtype=float,
    )
    y = df[target_col].to_numpy(dtype=float)
    return X_df.to_numpy(dtype=float), y, X_df.columns.tolist()


def impute_with_means(X, means):
    X_filled = X.copy()
    nan_mask = np.isnan(X_filled)
    if np.any(nan_mask):
        row_idx, col_idx = np.where(nan_mask)
        X_filled[row_idx, col_idx] = means[col_idx]
    return X_filled


def evaluate_predictions(y_true, y_pred):
    return {
        "rmse": calculate_rmse(y_true, y_pred),
        "mae": calculate_mae(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
    }


def bad_cross_validation(X, y, n_splits=5):
    print("\n" + "=" * 60)
    print("TASK 3: Bad Cross-Validation (WITH Data Leakage)")
    print("=" * 60)

    scaler = CustomStandardScaler().fit(X)
    X_global_imputed = impute_with_means(X, scaler.mean_)
    X_scaled = scaler.transform(X_global_imputed)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_scaled), start=1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GradientDescentOLS(learning_rate=0.01, max_iter=2000, tol=1e-8)
        model.fit(X_train, y_train)
        metrics = evaluate_predictions(y_val, model.predict(X_val))
        fold_metrics.append(metrics)
        print(
            f"Fold {fold_num}: "
            f"RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"MAPE={metrics['mape']:.2f}%"
        )

    mean_metrics = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in ["rmse", "mae", "mape"]
    }

    print(f"\nAverage RMSE: {mean_metrics['rmse']:.4f}")
    print(f"Average MAE:  {mean_metrics['mae']:.4f}")
    print(f"Average MAPE: {mean_metrics['mape']:.2f}%")
    return {"method": "bad_cv", **mean_metrics}


def good_cross_validation(X, y, n_splits=5):
    print("\n" + "=" * 60)
    print("TASK 4: Good Cross-Validation (NO Data Leakage)")
    print("=" * 60)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = CustomStandardScaler().fit(X_train_raw)
        X_train = impute_with_means(X_train_raw, scaler.mean_)
        X_val = impute_with_means(X_val_raw, scaler.mean_)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = GradientDescentOLS(learning_rate=0.01, max_iter=2000, tol=1e-8)
        model.fit(X_train_scaled, y_train)
        metrics = evaluate_predictions(y_val, model.predict(X_val_scaled))
        fold_metrics.append(metrics)
        print(
            f"Fold {fold_num}: "
            f"RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"MAPE={metrics['mape']:.2f}%"
        )

    mean_metrics = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in ["rmse", "mae", "mape"]
    }

    print(f"\nAverage RMSE: {mean_metrics['rmse']:.4f}")
    print(f"Average MAE:  {mean_metrics['mae']:.4f}")
    print(f"Average MAPE: {mean_metrics['mape']:.2f}%")
    return {"method": "good_cv", **mean_metrics}


def cleanup_results_dir():
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_comparison_report(results_dir, bad_results, good_results, feature_names, sample_count):
    report = f"""# Week 10 Milestone 2: Evaluation Comparison

## Dataset

- Source: `homework/week09/data/dirty_marketing.csv`
- Samples: {sample_count}
- Features after one-hot encoding: {", ".join(feature_names)}

## Metrics Comparison

| Metric | Bad CV (Leaky) | Good CV (Leakage-Free) | Bad - Good |
|--------|-----------------|------------------------|------------|
| RMSE | {bad_results["rmse"]:.4f} | {good_results["rmse"]:.4f} | {bad_results["rmse"] - good_results["rmse"]:.4f} |
| MAE | {bad_results["mae"]:.4f} | {good_results["mae"]:.4f} | {bad_results["mae"] - good_results["mae"]:.4f} |
| MAPE | {bad_results["mape"]:.2f}% | {good_results["mape"]:.2f}% | {bad_results["mape"] - good_results["mape"]:.2f}% |

## Business Interpretation

- Good CV is the result we should trust for deployment because each validation fold is transformed only with parameters learned from its matching training fold.
- Bad CV can look better because it lets validation information leak into preprocessing, so the evaluation becomes unrealistically optimistic.
- Under the leakage-free pipeline, the model's average absolute error is about {good_results["mae"]:.2f} sales units, and the average percentage error is about {good_results["mape"]:.2f}%.
"""

    report_path = results_dir / "evaluation_comparison.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def save_comparison_plot(results_dir, bad_results, good_results):
    metrics = ["RMSE", "MAE", "MAPE"]
    bad_values = [bad_results["rmse"], bad_results["mae"], bad_results["mape"]]
    good_values = [good_results["rmse"], good_results["mae"], good_results["mape"]]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, bad_values, width, label="Bad CV", color="#d95f02")
    ax.bar(x + width / 2, good_values, width, label="Good CV", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Error")
    ax.set_title("Leakage vs Leakage-Free Validation")
    ax.legend()
    fig.tight_layout()

    plot_path = results_dir / "leakage_analysis.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main():
    print("\n" + "=" * 60)
    print("Week 10 Milestone 2: Data Leakage & Clean Pipelines")
    print("=" * 60)

    results_dir = cleanup_results_dir()
    print(f"\nResults directory: {results_dir}")

    print("\nLoading data...")
    X, y, feature_names = load_data()
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Missing values in X: {np.isnan(X).sum()}")

    bad_results = bad_cross_validation(X, y)
    good_results = good_cross_validation(X, y)

    report_path = save_comparison_report(
        results_dir=results_dir,
        bad_results=bad_results,
        good_results=good_results,
        feature_names=feature_names,
        sample_count=len(y),
    )
    plot_path = save_comparison_plot(results_dir, bad_results, good_results)

    print(f"\nReport saved to: {report_path}")
    print(f"Plot saved to:   {plot_path}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nBad CV  - RMSE: {bad_results['rmse']:.4f}")
    print(f"Good CV - RMSE: {good_results['rmse']:.4f}")
    print(f"RMSE gap: {bad_results['rmse'] - good_results['rmse']:.4f}")


if __name__ == "__main__":
    main()
