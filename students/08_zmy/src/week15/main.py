"""
Week 15: Logistic Regression and Binary Classification
使用 scikit-learn 1.8+ 新 API（penalty='elasticnet' + l1_ratio）实现逻辑回归，无弃用警告。
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# 导入现有工具中的标准化类
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.transformers import CustomStandardScaler

# ======================== 辅助函数 ========================
def binary_classification_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

def threshold_scan(y_true, y_pred_prob, thresholds):
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        results.append({'threshold': thresh, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})
    return pd.DataFrame(results)

def plot_loss_curves(save_path):
    p = np.linspace(0.001, 0.999, 200)
    log_loss_y1 = -np.log(p)
    log_loss_y0 = -np.log(1-p)
    mse_y1 = (1 - p)**2
    mse_y0 = p**2

    plt.figure(figsize=(8, 5))
    plt.plot(p, log_loss_y1, 'r-', label='Log Loss (y=1)')
    plt.plot(p, log_loss_y0, 'r--', label='Log Loss (y=0)')
    plt.plot(p, mse_y1, 'b-', label='MSE (y=1)')
    plt.plot(p, mse_y0, 'b--', label='MSE (y=0)')
    plt.xlabel('Predicted probability $p$')
    plt.ylabel('Loss value')
    plt.title('Comparison of Log Loss and MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_threshold_curves(df_metrics, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics['threshold'], df_metrics['accuracy'], 'o-', label='Accuracy')
    plt.plot(df_metrics['threshold'], df_metrics['precision'], 's-', label='Precision')
    plt.plot(df_metrics['threshold'], df_metrics['recall'], '^-', label='Recall')
    plt.plot(df_metrics['threshold'], df_metrics['f1'], 'd-', label='F1')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_coefficients(coef_l1, coef_l2, feature_names, save_path):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(feature_names))
    width = 0.35
    plt.bar(x - width/2, coef_l1, width, label='L1 (Lasso)')
    plt.bar(x + width/2, coef_l2, width, label='L2 (Ridge)')
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Coefficient value')
    plt.title('L1 vs L2 Logistic Regression Coefficients')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def standard_scale(X_train, X_test):
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled

# ======================== 固定路径 ========================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# ======================== Task A ========================
def generate_synthetic_data(n_samples=500, n_features=4, effective_features=[0,1], beta_vals=[2.0, -1.5]):
    X = np.random.randn(n_samples, n_features)
    beta = np.zeros(n_features)
    for idx, val in zip(effective_features, beta_vals):
        beta[idx] = val
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p)
    return X, y, beta

def run_task_a():
    print("=== Task A: 生成数据和基础模型比较 ===")
    X, y, true_beta = generate_synthetic_data(n_samples=500, n_features=4)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    df['y'] = y
    df.to_csv(os.path.join(DATA_DIR, 'synthetic_binary.csv'), index=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    _, X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)

    # 线性回归
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)

    # 逻辑回归：使用极大 C 实现近乎无正则化
    logit_model = LogisticRegression(penalty='elasticnet', l1_ratio=0, C=1e10, solver='saga', max_iter=1000)
    logit_model.fit(X_train_scaled, y_train)
    logit_prob = logit_model.predict_proba(X_test_scaled)[:, 1]

    # 画图
    plt.figure(figsize=(8,5))
    plt.scatter(X_test_scaled[:, 0], y_test, alpha=0.5, label='True label (0/1)', color='gray')
    sorted_idx = np.argsort(X_test_scaled[:, 0])
    plt.plot(X_test_scaled[sorted_idx, 0], lr_pred[sorted_idx], 'b-', label='Linear Regression output')
    plt.plot(X_test_scaled[sorted_idx, 0], logit_prob[sorted_idx], 'r-', label='Logistic Regression probability')
    plt.xlabel('Feature x0 (standardized)')
    plt.ylabel('Model output / Probability')
    plt.title('Comparison: Linear Regression vs Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, 'fig_a4.png'), dpi=150)
    plt.close()

    report_a = f"""# Synthetic Data Report (Task A)

## A1. Data Generation Process (DGP)

- Sample size: {X.shape[0]}
- Number of features: {X.shape[1]}
- Effective features: x0 (positive effect, beta=2.0), x1 (negative effect, beta=-1.5)
- Noise features: x2, x3 (beta=0)
- DGP: 
  1. Linear predictor η = Xβ
  2. Probability p = 1/(1+exp(-η))
  3. y ~ Bernoulli(p)

## A3. Model Comparison

**LinearRegression output issues**: 
- Predictions range from negative to positive, not bounded in [0,1].
- Cannot be interpreted as probability.
- Hard thresholding at 0.5 ignores uncertainty.

**LogisticRegression output**: 
- Naturally bounded between 0 and 1.
- Directly interpretable as P(y=1|X).

## A4. Figure Explanation (fig_a4.png)

- X-axis: Standardized feature x0
- Y-axis: Model output (Linear Regression) or probability (Logistic Regression)
- Gray dots: True binary labels
- Blue line: Linear Regression predictions (unbounded)
- Red line: Logistic Regression probabilities (bounded in [0,1])

**Key observation**: Linear Regression produces values outside [0,1] and does not fit the binary pattern; Logistic Regression outputs smooth S-curve probabilities.

## A5. Core Questions

1. **Most unnatural aspect of Linear Regression**: Output is not constrained to [0,1]; assumes constant variance.
2. **Why Logistic Regression output is interpretable as probability**: Uses sigmoid to map any real number to (0,1) and is trained via maximum likelihood for Bernoulli data.
3. **Key distinction**: Not about "ability to classify" but **probabilistic meaning**. Logistic Regression gives calibrated probabilities; Linear Regression gives arbitrary scores.
"""
    with open(os.path.join(RESULTS_DIR, 'synthetic_report.md'), 'w') as f:
        f.write(report_a)
    print("Task A completed.")

# ======================== Task B ========================
def run_task_b():
    print("=== Task B: Loss functions and Bernoulli likelihood ===")
    plot_loss_curves(os.path.join(RESULTS_DIR, 'fig_b2.png'))
    content_b = """# Threshold & Loss Function Report (Task B & C)

## Task B: Bernoulli Likelihood and Log Loss

### B1. Three Essential Formulas

1. **Bernoulli distribution**:
   \\[
   Y \\sim \\text{Bernoulli}(p)
   \\]
   *Explanation*: Y takes value 1 with probability p and 0 with probability 1-p. Natural for binary outcomes.

2. **Single-sample likelihood**:
   \\[
   L(p;y) = p^{y}(1-p)^{1-y}
   \\]
   *Explanation*: When y=1, likelihood = p; when y=0, likelihood = 1-p.

3. **Negative log-likelihood (Log Loss)**:
   \\[
   -\\ln L(p;y) = -[y\\ln p + (1-y)\\ln(1-p)]
   \\]
   *Explanation*: Minimizing this is equivalent to maximizing likelihood.

### B2. Loss Comparison Figure (fig_b2.png)

- X-axis: Predicted probability p
- Y-axis: Loss value
- Red solid: Log Loss (y=1); Red dashed: Log Loss (y=0)
- Blue solid: MSE (y=1); Blue dashed: MSE (y=0)

**Key insight**: When confidently wrong (e.g., predict p≈0 but true y=1), Log Loss → ∞, while MSE remains bounded. Log Loss heavily penalizes confident mistakes.

### B3. Discussion

1. **Why heavily penalize confident mistakes?** In critical decisions (medical diagnosis, fraud detection), confident errors can be catastrophic.
2. **Log loss from Bernoulli likelihood**: It derives directly from the negative logarithm of the Bernoulli likelihood.
3. **Why Log Loss over MSE?** MSE assumes Gaussian errors and symmetric loss; Log Loss respects [0,1] domain and Bernoulli nature.
"""
    return content_b

# ======================== Task C ========================
def run_task_c(prev_content):
    print("=== Task C: Classification metrics and threshold analysis ===")
    df = pd.read_csv(os.path.join(DATA_DIR, 'synthetic_binary.csv'))
    X = df.drop('y', axis=1).values
    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    _, X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)
    
    logit = LogisticRegression(penalty='elasticnet', l1_ratio=0, C=1e10, solver='saga', max_iter=1000)
    logit.fit(X_train_scaled, y_train)
    proba = logit.predict_proba(X_test_scaled)[:, 1]

    metrics_dict = binary_classification_metrics(y_test, proba, threshold=0.5)

    thresholds = np.arange(0.1, 1.0, 0.1)
    df_scan = threshold_scan(y_test, proba, thresholds)
    plot_threshold_curves(df_scan, os.path.join(RESULTS_DIR, 'fig_c3.png'))

    report_c = prev_content + f"""
## Task C: Confusion Matrix and Threshold Trade-offs

### C1. Basic Metrics (threshold = 0.5)

| Metric | Value |
|--------|-------|
| TP     | {metrics_dict['TP']} |
| TN     | {metrics_dict['TN']} |
| FP     | {metrics_dict['FP']} |
| FN     | {metrics_dict['FN']} |
| Accuracy | {metrics_dict['accuracy']:.4f} |
| Precision | {metrics_dict['precision']:.4f} |
| Recall | {metrics_dict['recall']:.4f} |
| F1     | {metrics_dict['f1']:.4f} |

### C2 & C3. Threshold Scan

Thresholds from 0.1 to 0.9, step 0.1.

**Figure (fig_c3.png)**:
- X-axis: Classification threshold
- Y-axis: Metric value
- Lines: Accuracy, Precision, Recall, F1

**Observations**:
- As threshold increases, recall decreases (fewer positives predicted).
- Precision often increases (higher confidence in predicted positives).
- Accuracy may peak at an intermediate threshold.
- F1 balances precision and recall.

**Trade-off**: Lower threshold → higher recall but lower precision; higher threshold → opposite.

### C4. Business Scenario: Credit Default Prediction

**Scenario**: Predict whether a borrower will default.

**Most important metric**: Recall (or F1). Missing a defaulter (FN) causes direct financial loss; false alarms (FP) cause customer dissatisfaction but lower cost. Accuracy can be misleading if default rate is low.

**Recommended threshold**: Choose threshold where recall is high enough (e.g., 80%) while maintaining acceptable precision. Use cost-benefit analysis: cost of FN (loss amount) vs cost of FP (lost business).
"""
    with open(os.path.join(RESULTS_DIR, 'threshold_report.md'), 'w') as f:
        f.write(report_c)
    print("Task C completed.")

# ======================== Task D ========================
def run_task_d():
    print("=== Task D: Regularized Logistic Regression (L1 vs L2) ===")
    np.random.seed(123)
    n_samples = 300
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    # 引入共线性
    X[:, 5:10] = X[:, 0:5] + np.random.randn(n_samples, 5) * 0.5
    true_beta = np.zeros(n_features)
    true_beta[:5] = [1.5, -1.0, 2.0, -0.8, 1.2]
    eta = X @ true_beta
    p = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, p)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {'C': np.logspace(-3, 2, 20)}

    # L1: 使用 elasticnet + l1_ratio=1 (纯 L1)
    l1_model = LogisticRegression(penalty='elasticnet', l1_ratio=1, solver='saga', max_iter=2000)
    # L2: 使用 elasticnet + l1_ratio=0 (纯 L2)
    l2_model = LogisticRegression(penalty='elasticnet', l1_ratio=0, solver='saga', max_iter=2000)

    grid_l1 = GridSearchCV(l1_model, param_grid, cv=5, scoring='roc_auc')
    grid_l2 = GridSearchCV(l2_model, param_grid, cv=5, scoring='roc_auc')
    grid_l1.fit(X_train_scaled, y_train)
    grid_l2.fit(X_train_scaled, y_train)

    best_l1 = grid_l1.best_estimator_
    best_l2 = grid_l2.best_estimator_

    proba_l1 = best_l1.predict_proba(X_test_scaled)[:, 1]
    proba_l2 = best_l2.predict_proba(X_test_scaled)[:, 1]

    metrics_l1 = {
        'accuracy': (best_l1.predict(X_test_scaled) == y_test).mean(),
        'recall': recall_score(y_test, best_l1.predict(X_test_scaled)),
        'roc_auc': roc_auc_score(y_test, proba_l1),
        'log_loss': log_loss(y_test, proba_l1),
        'n_nonzero': np.sum(best_l1.coef_[0] != 0)
    }
    metrics_l2 = {
        'accuracy': (best_l2.predict(X_test_scaled) == y_test).mean(),
        'recall': recall_score(y_test, best_l2.predict(X_test_scaled)),
        'roc_auc': roc_auc_score(y_test, proba_l2),
        'log_loss': log_loss(y_test, proba_l2),
        'n_nonzero': np.sum(best_l2.coef_[0] != 0)
    }

    feature_names = [f'f{i}' for i in range(n_features)]
    plot_coefficients(best_l1.coef_[0], best_l2.coef_[0], feature_names, os.path.join(RESULTS_DIR, 'fig_d3.png'))

    report_d = f"""# Regularization Report (Task D)

## D1. High-dimensional data with collinearity
- Samples: {n_samples}, Features: {n_features}
- Collinearity: features 5-9 are correlated with features 0-4 (noise added)
- True relevant features: first 5
- Target generated via Bernoulli(sigmoid(Xβ))

## D2. L1 vs L2 Comparison (best C chosen by cross-validation)

| Model | Accuracy | Recall | ROC-AUC | Log Loss | Non-zero coeffs |
|-------|----------|--------|---------|----------|-----------------|
| L1 (Lasso) | {metrics_l1['accuracy']:.4f} | {metrics_l1['recall']:.4f} | {metrics_l1['roc_auc']:.4f} | {metrics_l1['log_loss']:.4f} | {metrics_l1['n_nonzero']} |
| L2 (Ridge) | {metrics_l2['accuracy']:.4f} | {metrics_l2['recall']:.4f} | {metrics_l2['roc_auc']:.4f} | {metrics_l2['log_loss']:.4f} | {metrics_l2['n_nonzero']} |

## D3. Coefficient plot (fig_d3.png)
- X-axis: Features
- Y-axis: Coefficient value
- Blue bars: L1 coefficients
- Orange bars: L2 coefficients

**Observation**: L1 produces many zero coefficients (sparse), while L2 shrinks coefficients but rarely to zero.

## D4. Core questions

1. **Prediction performance**: Usually similar when signal is strong; here they are close.
2. **Which is sparser?** L1 is much sparser ({metrics_l1['n_nonzero']} vs {metrics_l2['n_nonzero']} non-zero).
3. **Which gives a shorter variable list?** L1. It performs feature selection, easier to explain.
4. **Stability over variable selection**: If business cares more about stability, L2 is better because it does not perform hard selection and is more robust.
"""
    with open(os.path.join(RESULTS_DIR, 'regularization_report.md'), 'w') as f:
        f.write(report_d)
    print("Task D completed.")

# ======================== Task F ========================
def run_task_f():
    print("=== Task F: Summary ===")
    summary = """# Summary: Logistic Regression and Binary Classification

## 1. Why Logistic Regression is not just "linear regression + sigmoid"

While mathematically it can be viewed that way, the crucial difference lies in the **objective function**. Linear regression minimizes squared error, which is inappropriate for binary data. Logistic regression maximizes Bernoulli likelihood (minimizes log loss), which respects the probabilistic nature of the output.

## 2. Relationship between sigmoid, Bernoulli likelihood, and log loss

- **Sigmoid**: Maps linear predictor η = Xβ to probability p = σ(η), ensuring p ∈ (0,1).
- **Bernoulli likelihood**: Describes the data-generating process for binary outcomes.
- **Log loss**: Negative log of Bernoulli likelihood; minimizing it = maximizing likelihood.

## 3. Why accuracy alone is insufficient

Accuracy treats all misclassifications equally and ignores class imbalance. A model that always predicts the majority class can have high accuracy but zero recall for the minority class. Precision, recall, F1, and ROC-AUC give a more nuanced view.

## 4. When to use L1 vs L2 logistic regression

- **L1 (Lasso)**: When feature selection is desired, sparse interpretable model.
- **L2 (Ridge)**: When prediction stability and handling multicollinearity are important.

## 5. Why Logistic Regression remains a strong baseline

- **Probabilistic outputs**: Well-calibrated probabilities for decision-making.
- **Interpretability**: Coefficients indicate direction and magnitude of feature influence.
- **Stability**: With L2 regularization, handles collinearity well.
- **Efficiency**: Fast to train and deploy.
- **Performance**: Often competitive with complex models as a baseline.
"""
    with open(os.path.join(RESULTS_DIR, 'summary.md'), 'w') as f:
        f.write(summary)
    print("Task F completed.")

# ======================== Main ========================
def main():
    print("Starting Week 15 Assignment...")
    run_task_a()
    content_b = run_task_b()
    run_task_c(content_b)
    run_task_d()
    run_task_f()
    print("\nAll tasks completed. Check the 'results/' folder for reports and figures.")

if __name__ == '__main__':
    main()