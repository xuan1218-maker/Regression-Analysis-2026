"""
Week 15: Logistic Regression and Binary Classification
======================================================
Tasks A–F: synthetic data, log loss, threshold trade-offs, regularization, Titanic, summary.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, log_loss
)

# ── paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# set Chinese-capable font for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
#  Task A: Synthetic binary classification
# =====================================================================

def task_a():
    print("\n" + "=" * 60)
    print("Task A: Synthetic Binary Classification")
    print("=" * 60)

    # --- A1 & A2: Generate data and save ---
    np.random.seed(42)
    n = 500
    # 4 features: X1, X2 are informative; X3, X4 are noise
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    X3 = np.random.randn(n)  # noise
    X4 = np.random.randn(n)  # noise

    # True coefficients: X1 positive effect, X2 negative effect
    beta0, beta1, beta2 = -0.5, 1.5, -1.0
    eta = beta0 + beta1 * X1 + beta2 * X2
    p_true = 1.0 / (1.0 + np.exp(-eta))
    y = np.random.binomial(1, p_true)

    # Save synthetic data
    df_syn = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4,
        'p_true': p_true, 'y': y
    })
    syn_path = os.path.join(DATA_DIR, "synthetic_binary.csv")
    df_syn.to_csv(syn_path, index=False)
    print(f"  Saved synthetic data to {syn_path}")
    print(f"  Samples: {n}, Features: 4, Positive rate: {y.mean():.3f}")

    # --- A3: Train LinearRegression and LogisticRegression ---
    feature_cols = ['X1', 'X2', 'X3', 'X4']
    X = df_syn[feature_cols].values
    y_arr = df_syn['y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=0.3, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    logr = LogisticRegression(max_iter=1000)
    logr.fit(X_train, y_train)
    logr_prob = logr.predict_proba(X_test)[:, 1]

    # LinearRegression clipped as "probability"
    lr_prob_clipped = np.clip(lr_pred, 0, 1)
    lr_acc = accuracy_score(y_test, (lr_pred >= 0.5).astype(int))
    logr_acc = accuracy_score(y_test, (logr_prob >= 0.5).astype(int))
    lr_logloss = log_loss(y_test, lr_prob_clipped)
    logr_logloss = log_loss(y_test, logr_prob)

    print(f"\n  LinearRegression  - Accuracy: {lr_acc:.4f}, Log Loss: {lr_logloss:.4f}")
    print(f"  LogisticRegression - Accuracy: {logr_acc:.4f}, Log Loss: {logr_logloss:.4f}")
    print(f"  LinearRegression output range: [{lr_pred.min():.3f}, {lr_pred.max():.3f}]")

    # --- A4: Core comparison plot ---
    # Use X1 as the main feature, hold X2 at its mean
    x1_grid = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 300)
    x2_mean = X_test[:, 1].mean()
    x3_mean = X_test[:, 2].mean()
    x4_mean = X_test[:, 3].mean()

    grid = np.column_stack([
        x1_grid,
        np.full_like(x1_grid, x2_mean),
        np.full_like(x1_grid, x3_mean),
        np.full_like(x1_grid, x4_mean),
    ])

    lr_curve = lr.predict(grid)
    logr_curve = logr.predict_proba(grid)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    # scatter: true labels
    ax.scatter(X_test[:, 0], y_test, alpha=0.3, s=15, c='gray', label='True labels (0/1)')
    ax.plot(x1_grid, lr_curve, 'b-', linewidth=2, label='LinearRegression')
    ax.plot(x1_grid, logr_curve, 'r-', linewidth=2, label='LogisticRegression (prob)')
    ax.set_xlabel('X1 (informative feature)')
    ax.set_ylabel('Model output / Probability')
    ax.set_title('Task A: LinearRegression vs LogisticRegression')
    ax.legend()
    ax.set_ylim(-0.5, 1.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "task_a_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved task_a_comparison.png")

    # --- A5: Write synthetic_report.md ---
    report = f"""# Task A: Synthetic Binary Classification Report

## A2. Data Generation Process (DGP)

- **样本量**: {n}
- **特征数**: 4 (X1, X2, X3, X4)
- **正类比例**: {y.mean():.3f}
- **DGP**: 先构造线性预测值 $\\eta = {beta0} + {beta1} \\cdot X_1 + {beta2} \\cdot X_2$，再通过 sigmoid 转换为概率 $p = 1/(1+e^{{-\\eta}})$，最后从 Bernoulli(p) 采样得到 y。
- **X1** 对正类概率有正向影响（系数 {beta1}）；
- **X2** 对正类概率有负向影响（系数 {beta2}）；
- **X3、X4** 是纯噪声特征，不影响类别概率。

## A3. 模型对比

| 模型 | Accuracy | Log Loss |
|------|----------|----------|
| LinearRegression (threshold=0.5) | {lr_acc:.4f} | {lr_logloss:.4f} |
| LogisticRegression | {logr_acc:.4f} | {logr_logloss:.4f} |

LinearRegression 的输出范围为 [{lr_pred.min():.3f}, {lr_pred.max():.3f}]，
出现了超出 [0, 1] 的值，这些值无法被解释为概率。

## A4. 核心对比图

**图 task_a_comparison.png**：
- 横轴：X1（一个有信息量的特征），其他特征固定在均值
- 纵轴：模型输出
- 灰色散点：真实标签 (0/1)
- 蓝色曲线：LinearRegression 的预测值（线性，可能超出 [0,1]）
- 红色曲线：LogisticRegression 的预测概率（S 形，始终在 [0,1] 内）
- **结论**：LinearRegression 的输出是无界的直线，不能合理解释为概率；LogisticRegression 通过 sigmoid 将输出压缩到 (0,1)，天然适合概率解释。

## A5. 核心问题

### Q1. LinearRegression 在这个任务里最不自然的地方是什么？
LinearRegression 的输出是无界的连续值，可能小于 0 或大于 1。在二分类问题中，我们需要的是"属于正类的概率"，而概率必须在 [0,1] 区间内。LinearRegression 无法保证这一点，因此将其输出硬解释为概率在数学上是不合理的。

### Q2. 为什么逻辑回归的输出更容易解释成概率？
逻辑回归通过 sigmoid 函数 $\\sigma(\\eta) = 1/(1+e^{{-\\eta}})$ 将线性组合映射到 (0,1) 区间，输出天然满足概率的基本性质：有界性。同时，逻辑回归的训练目标（最大化 Bernoulli 似然）保证了输出在统计意义上是对 $P(Y=1|X)$ 的一致估计。

### Q3. 关键区别是"能不能分类"还是"输出是否有概率意义"？
关键区别是**输出是否有概率意义**。LinearRegression 也能做分类（通过设定阈值），但它的输出没有概率解释。逻辑回归的输出不仅能在 (0,1) 内，而且经过 MLE 训练后，输出值与真实的条件概率 $P(Y=1|X)$ 有明确的对应关系。这使得我们可以对输出做更有意义的决策——比如调整阈值、校准概率等。
"""
    report_path = os.path.join(RESULTS_DIR, "synthetic_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")

    return df_syn, X_test, y_test, logr, logr_prob


# =====================================================================
#  Task B: Bernoulli likelihood and log loss
# =====================================================================

def task_b():
    print("\n" + "=" * 60)
    print("Task B: Bernoulli Likelihood and Log Loss")
    print("=" * 60)

    # --- B2: Loss curves plot ---
    p = np.linspace(0.001, 0.999, 500)

    # Squared error loss
    se_y1 = (1 - p) ** 2
    se_y0 = p ** 2

    # Log loss (negative log-likelihood)
    log_y1 = -np.log(p)
    log_y0 = -np.log(1 - p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # y = 1
    axes[0].plot(p, se_y1, 'b-', linewidth=2, label='Squared Error')
    axes[0].plot(p, log_y1, 'r-', linewidth=2, label='Log Loss')
    axes[0].set_xlabel('Predicted probability p')
    axes[0].set_ylabel('Loss value')
    axes[0].set_title('Loss when y = 1')
    axes[0].legend()
    axes[0].set_ylim(0, 8)

    # y = 0
    axes[1].plot(p, se_y0, 'b-', linewidth=2, label='Squared Error')
    axes[1].plot(p, log_y0, 'r-', linewidth=2, label='Log Loss')
    axes[1].set_xlabel('Predicted probability p')
    axes[1].set_ylabel('Loss value')
    axes[1].set_title('Loss when y = 0')
    axes[1].legend()
    axes[1].set_ylim(0, 8)

    fig.suptitle('Task B: Loss Comparison — Squared Error vs Log Loss', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "task_b_loss_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved task_b_loss_curves.png")

    # --- Write threshold_report.md (includes B1 and B3) ---
    report = """# Task B: Bernoulli Likelihood and Log Loss Report

## B1. 三个核心公式

### 1. Bernoulli 分布

$$Y \\sim Bernoulli(p)$$

Bernoulli 分布描述一次二元试验的结果：Y 只能取 0 或 1，取 1 的概率为 p，取 0 的概率为 1-p。这是二分类问题中最基本的概率模型，也是逻辑回归的建模起点。

### 2. 单样本 likelihood

$$L(p; y) = p^y (1-p)^{1-y}$$

这个公式用一个统一的表达式同时覆盖了 y=1 和 y=0 两种情况：当 y=1 时，likelihood = p；当 y=0 时，likelihood = 1-p。Likelihood 衡量的是"在给定参数 p 的情况下，观测到当前样本 y 的可能性"。MLE 的目标就是找到使 likelihood 最大的参数 p。

### 3. 单样本负对数似然（log loss）

$$-\\log L(p; y) = -y \\log p - (1-y) \\log (1-p)$$

对 likelihood 取负对数有两个好处：(1) 把乘法变成加法，便于数值计算；(2) 对数函数是单调递增的，最大化 likelihood 等价于最小化负对数似然。这就是逻辑回归的损失函数——log loss，也叫 cross-entropy loss。

## B2. 损失曲线图

**图 task_b_loss_curves.png**：
- 左图：当真实标签 y=1 时，横轴为预测概率 p，纵轴为损失值
- 右图：当真实标签 y=0 时，横轴为预测概率 p，纵轴为损失值
- 蓝色线：Squared Error $(y-p)^2$
- 红色线：Log Loss $-y\\log p - (1-y)\\log(1-p)$

### 观察与结论

1. **当模型"错得很自信"时**（y=1 但 p→0，或 y=0 但 p→1）：
   - Squared Error 的损失趋近于 1（有上界）
   - Log Loss 的损失趋近于 +∞（无上界）
   - **Log Loss 对"自信的错误"惩罚更重**

2. **当模型预测正确时**（y=1 且 p→1，或 y=0 且 p→0）：
   - 两种 loss 都趋近于 0

3. **这张图支持的结论**：Log Loss 比 Squared Error 更适合分类问题，因为它对"自信但错误"的预测施加了更强的惩罚，这与直觉一致——如果一个模型非常自信地预测错了，应该受到严厉的惩罚。

## B3. 统计建模对应

### Q1. 为什么二分类里"错得很自信"需要被重罚？
在二分类中，模型输出的是概率。如果一个模型对错误的预测非常自信（比如真实标签是 1，但模型给出 p=0.01），这意味着模型对数据的理解有严重偏差。从决策角度看，这种错误可能导致业务方做出灾难性的决策（比如拒绝了一个实际会还款的贷款申请人）。因此，损失函数应该对这类错误施加重罚，以促使模型避免"自信地犯错"。

### Q2. 为什么说 log loss 不是凭空指定的，而是来自 Bernoulli likelihood？
逻辑回归假设 $Y|X \\sim Bernoulli(\\sigma(X\\beta))$。给定训练数据，我们希望找到参数 $\\beta$ 使得观测数据的 likelihood 最大。对 likelihood 取负对数，就得到了 log loss。因此，log loss 不是一个随意选择的损失函数，而是从 Bernoulli 概率模型出发，通过最大似然估计（MLE）自然推导出来的。

### Q3. 如果我们已经把输出解释成概率，那么为什么 log loss 比 MSE 更自然？
如果我们把输出解释成概率，那么模型的目标就是估计条件概率 $P(Y=1|X)$。从概率建模的角度，最自然的训练目标是最大化数据的 likelihood（或等价地最小化负对数似然）。MSE 虽然也能用，但它没有对应的概率解释——它假设的是高斯噪声，而不是 Bernoulli 分布。Log Loss 直接对应 Bernoulli likelihood，因此在概率解释框架下更自然、更一致。
"""
    report_path = os.path.join(RESULTS_DIR, "threshold_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")


# =====================================================================
#  Task C: Classification metrics and threshold trade-offs
# =====================================================================

def task_c(X_test, y_test, logr_prob):
    print("\n" + "=" * 60)
    print("Task C: Classification Metrics and Threshold Trade-offs")
    print("=" * 60)

    # --- C1: Confusion matrix at default threshold 0.5 ---
    y_pred_05 = (logr_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_05)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred_05)
    prec = precision_score(y_test, y_pred_05)
    rec = recall_score(y_test, y_pred_05)
    f1 = f1_score(y_test, y_pred_05)

    print(f"\n  Confusion Matrix (threshold=0.5):")
    print(f"    TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"    Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    # --- C2 & C3: Threshold scan ---
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    for t in thresholds:
        y_pred_t = (logr_prob >= t).astype(int)
        results.append({
            'threshold': t,
            'accuracy': accuracy_score(y_test, y_pred_t),
            'precision': precision_score(y_test, y_pred_t, zero_division=0),
            'recall': recall_score(y_test, y_pred_t),
            'f1': f1_score(y_test, y_pred_t),
        })

    df_thresh = pd.DataFrame(results)
    print("\n  Threshold Scan:")
    print(df_thresh.to_string(index=False, float_format='%.4f'))

    # Plot threshold curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_thresh['threshold'], df_thresh['accuracy'], 'g-o', linewidth=2, markersize=5, label='Accuracy')
    ax.plot(df_thresh['threshold'], df_thresh['precision'], 'b-s', linewidth=2, markersize=5, label='Precision')
    ax.plot(df_thresh['threshold'], df_thresh['recall'], 'r-^', linewidth=2, markersize=5, label='Recall')
    ax.plot(df_thresh['threshold'], df_thresh['f1'], 'k-D', linewidth=2, markersize=5, label='F1')
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title('Task C: Metrics vs Classification Threshold')
    ax.legend()
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "task_c_threshold_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved task_c_threshold_curves.png")

    # --- ROC-AUC ---
    roc_auc = roc_auc_score(y_test, logr_prob)
    print(f"\n  ROC-AUC: {roc_auc:.4f}")

    # --- Append to threshold_report.md ---
    with open(os.path.join(RESULTS_DIR, "threshold_report.md"), 'a') as f:
        f.write(f"""

---

# Task C: Classification Metrics and Threshold Trade-offs

## C1. 混淆矩阵与基础指标 (threshold=0.5)

| 指标 | 值 |
|------|-----|
| TP | {tp} |
| TN | {tn} |
| FP | {fp} |
| FN | {fn} |
| Accuracy | {acc:.4f} |
| Precision | {prec:.4f} |
| Recall | {rec:.4f} |
| F1 | {f1:.4f} |
| ROC-AUC | {roc_auc:.4f} |

## C2. Threshold 扫描结果

{df_thresh.to_markdown(index=False, floatfmt='.4f')}

## C3. Threshold 曲线图

**图 task_c_threshold_curves.png**：
- 横轴：classification threshold (0.1 到 0.9)
- 纵轴：metric value (0 到 1)
- 绿色圆点线：Accuracy
- 蓝色方块线：Precision
- 红色三角线：Recall
- 黑色菱形线：F1

### 观察到的 Trade-off

1. **Precision 与 Recall 之间存在明显的此消彼长关系**：
   - 当阈值升高时，Precision 上升（预测为正类的样本更少、更精准），但 Recall 下降（漏掉更多真正的正类）
   - 当阈值降低时，Recall 上升（捕获更多正类），但 Precision 下降（误报增多）

2. **Accuracy 相对稳定**，在多数阈值下变化不大，这说明 accuracy 对阈值不敏感，不能反映模型在不同决策策略下的表现差异。

3. **F1 是 Precision 和 Recall 的调和平均**，通常在中间阈值处达到最高，代表了一种平衡。

## C4. 业务场景解释：疾病初筛

在**疾病初筛**场景中：

1. **最在意的指标是 Recall（召回率）**。
2. **原因**：初筛的目标是尽可能不漏掉任何真正的患者。如果一个患者被漏诊（FN），后果可能是延误治疗甚至危及生命。而误报（FP）虽然会带来额外的检查成本，但远比漏诊的后果轻。因此，我们应该优先保证高 Recall，即使 Precision 会有所下降。
3. **推荐阈值**：我会建议使用较低的阈值（如 0.3），这样可以最大化 Recall。向业务方解释时，我会说："我们宁可多做一些不必要的检查，也不能漏掉真正的患者。低阈值意味着更高的敏感度，虽然会有一些误报，但能最大程度保障患者安全。"
""")
    print("  Appended Task C to threshold_report.md")


# =====================================================================
#  Task D: Regularization — L1 vs L2
# =====================================================================

def task_d():
    print("\n" + "=" * 60)
    print("Task D: Regularization — L1 vs L2 Logistic Regression")
    print("=" * 60)

    # --- D1: Generate high-dimensional data with collinearity ---
    np.random.seed(123)
    n = 600
    p_informative = 5
    p_correlated = 5
    p_noise = 10
    p_total = p_informative + p_correlated + p_noise

    # Informative features
    X_info = np.random.randn(n, p_informative)
    beta_info = np.array([1.5, -1.0, 0.8, -0.5, 0.3])

    # Correlated features (linear combos of informative + noise)
    X_corr = X_info @ np.random.randn(p_informative, p_correlated) + 0.3 * np.random.randn(n, p_correlated)

    # Noise features
    X_noise = np.random.randn(n, p_noise)

    X_all = np.column_stack([X_info, X_corr, X_noise])
    eta = X_info @ beta_info - 0.5
    p_true = 1.0 / (1.0 + np.exp(-eta))
    y = np.random.binomial(1, p_true)

    feature_names = (
        [f'info_{i}' for i in range(p_informative)] +
        [f'corr_{i}' for i in range(p_correlated)] +
        [f'noise_{i}' for i in range(p_noise)]
    )

    print(f"  Generated data: n={n}, p={p_total} (informative={p_informative}, correlated={p_correlated}, noise={p_noise})")
    print(f"  Positive rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.3, random_state=42
    )

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- D2: GridSearchCV for L1 and L2 ---
    C_values = [0.01, 0.1, 1, 10, 100]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # L2 (l1_ratio=0)
    grid_l2 = GridSearchCV(
        LogisticRegression(l1_ratio=0, solver='saga', max_iter=5000),
        param_grid={'C': C_values}, cv=cv, scoring='roc_auc'
    )
    grid_l2.fit(X_train_s, y_train)
    best_l2 = grid_l2.best_estimator_
    print(f"\n  L2 best C: {grid_l2.best_params_['C']}")

    # L1 (l1_ratio=1)
    grid_l1 = GridSearchCV(
        LogisticRegression(l1_ratio=1, solver='saga', max_iter=5000),
        param_grid={'C': C_values}, cv=cv, scoring='roc_auc'
    )
    grid_l1.fit(X_train_s, y_train)
    best_l1 = grid_l1.best_estimator_
    print(f"  L1 best C: {grid_l1.best_params_['C']}")

    # --- Evaluate on test set ---
    def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
        prob_te = model.predict_proba(X_te)[:, 1]
        pred_te = model.predict(X_te)
        return {
            'model': name,
            'accuracy': accuracy_score(y_te, pred_te),
            'recall': recall_score(y_te, pred_te),
            'roc_auc': roc_auc_score(y_te, prob_te),
            'log_loss': log_loss(y_te, prob_te),
            'n_nonzero': np.sum(np.abs(model.coef_[0]) > 1e-6),
        }

    res_l1 = evaluate_model(best_l1, X_train_s, y_train, X_test_s, y_test, 'L1')
    res_l2 = evaluate_model(best_l2, X_train_s, y_train, X_test_s, y_test, 'L2')

    df_results = pd.DataFrame([res_l1, res_l2])
    print("\n  Comparison Table:")
    print(df_results.to_string(index=False, float_format='%.4f'))

    # --- D3: Performance + sparsity plot ---
    metrics = ['accuracy', 'recall', 'roc_auc', 'log_loss']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Performance comparison
    x_pos = np.arange(len(metrics))
    width = 0.35
    vals_l1 = [res_l1[m] for m in metrics]
    vals_l2 = [res_l2[m] for m in metrics]
    axes[0].bar(x_pos - width/2, vals_l1, width, label='L1', color='steelblue')
    axes[0].bar(x_pos + width/2, vals_l2, width, label='L2', color='coral')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(['Accuracy', 'Recall', 'ROC-AUC', 'Log Loss'])
    axes[0].set_ylabel('Metric Value')
    axes[0].set_title('Performance Comparison: L1 vs L2')
    axes[0].legend()

    # Sparsity comparison
    coefs_l1 = best_l1.coef_[0]
    coefs_l2 = best_l2.coef_[0]
    idx = np.arange(len(coefs_l1))
    axes[1].bar(idx - 0.2, np.abs(coefs_l1), 0.4, label='L1 |coef|', color='steelblue', alpha=0.8)
    axes[1].bar(idx + 0.2, np.abs(coefs_l2), 0.4, label='L2 |coef|', color='coral', alpha=0.8)
    axes[1].set_xlabel('Feature Index')
    axes[1].set_ylabel('|Coefficient|')
    axes[1].set_title('Coefficient Magnitudes: L1 vs L2')
    axes[1].legend()
    axes[1].set_xticks(idx[::2])

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "task_d_regularization.png"), dpi=150)
    plt.close(fig)
    print("  Saved task_d_regularization.png")

    # --- Write regularization_report.md ---
    report = f"""# Task D: Regularization — L1 vs L2 Logistic Regression

## D1. 数据构造

- **样本量**: {n}
- **特征数**: {p_total}
  - Informative 特征: {p_informative} 个（真正影响类别概率）
  - Correlated 特征: {p_correlated} 个（与 informative 特征线性相关 + 噪声）
  - Noise 特征: {p_noise} 个（纯噪声）
- 存在明显的多重共线性（corr_* 是 info_* 的线性组合加噪声）

## D2. 模型对比

| 模型 | Accuracy | Recall | ROC-AUC | Log Loss | 非零系数数 |
|------|----------|--------|---------|----------|-----------|
| L1 | {res_l1['accuracy']:.4f} | {res_l1['recall']:.4f} | {res_l1['roc_auc']:.4f} | {res_l1['log_loss']:.4f} | {res_l1['n_nonzero']} |
| L2 | {res_l2['accuracy']:.4f} | {res_l2['recall']:.4f} | {res_l2['roc_auc']:.4f} | {res_l2['log_loss']:.4f} | {res_l2['n_nonzero']} |

- L1 最优超参数 C = {grid_l1.best_params_['C']}
- L2 最优超参数 C = {grid_l2.best_params_['C']}

## D3. 对比图

**图 task_d_regularization.png**：

左图 — 性能对比：
- 横轴：评估指标（Accuracy, Recall, ROC-AUC, Log Loss）
- 纵轴：指标值
- 蓝色柱：L1 正则化
- 红色柱：L2 正则化

右图 — 系数大小分布：
- 横轴：特征索引
- 纵轴：|系数绝对值|
- 蓝色柱：L1 的系数
- 红色柱：L2 的系数

## D4. 核心比较问题

### Q1. L1 和 L2 的预测表现差很多吗？
两者的预测表现（Accuracy, Recall, ROC-AUC, Log Loss）非常接近，差异不大。这说明在预测能力上，L1 和 L2 没有本质区别。

### Q2. 哪一个模型更稀疏？
**L1 更稀疏**。L1 正则化会将不重要的特征系数压缩到 exactly 0，实现自动特征选择。L2 只会将系数缩小但不会变为 0。因此 L1 的非零系数数量明显少于 L2。

### Q3. 哪一个模型更适合"给出一个更短的变量名单"？
**L1 更适合**。因为 L1 的稀疏性，它可以自动筛选出最重要的特征，给出一个更短、更可解释的变量名单。这在需要向业务方解释"哪些变量最重要"时非常有用。

### Q4. 如果业务方更在意模型稳定性而不是变量筛选，你更偏向哪一个？
**更偏向 L2**。L2 正则化对所有特征一视同仁地收缩系数，不会像 L1 那样因为数据的微小变化而改变哪些系数为 0、哪些不为 0。L2 的系数估计更稳定，更适合需要稳定预测的场景。此外，当特征之间存在共线性时，L2 的表现通常更稳健。
"""
    report_path = os.path.join(RESULTS_DIR, "regularization_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")


# =====================================================================
#  Task E: Titanic real data
# =====================================================================

def task_e():
    print("\n" + "=" * 60)
    print("Task E: Titanic Real Data Challenge")
    print("=" * 60)

    # Load Titanic data
    titanic_path = os.path.join(DATA_DIR, "Titanic_train.csv")
    df = pd.read_csv(titanic_path)
    print(f"  Loaded Titanic data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Survived rate: {df['Survived'].mean():.3f}")

    # --- Data cleaning ---
    # Select useful features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[['Survived'] + features].copy()

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # One-hot encode
    df['Sex_male'] = (df['Sex'] == 'male').astype(float)
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(float)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(float)
    df = df.drop(columns=['Sex', 'Embarked'])

    feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    X = df[feature_cols].values
    y = df['Survived'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --- Train logistic regression ---
    logr = LogisticRegression(max_iter=1000)
    logr.fit(X_train_s, y_train)
    prob_test = logr.predict_proba(X_test_s)[:, 1]

    # --- Threshold analysis ---
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    for t in thresholds:
        pred_t = (prob_test >= t).astype(int)
        results.append({
            'threshold': t,
            'accuracy': accuracy_score(y_test, pred_t),
            'precision': precision_score(y_test, pred_t, zero_division=0),
            'recall': recall_score(y_test, pred_t),
            'f1': f1_score(y_test, pred_t),
        })

    df_thresh = pd.DataFrame(results)
    roc_auc = roc_auc_score(y_test, prob_test)

    # Confusion matrix at 0.5
    pred_05 = (prob_test >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred_05)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, pred_05)
    prec = precision_score(y_test, pred_05)
    rec = recall_score(y_test, pred_05)
    f1 = f1_score(y_test, pred_05)

    print(f"\n  Confusion Matrix (threshold=0.5): TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

    # --- L1 vs L2 comparison ---
    C_values = [0.01, 0.1, 1, 10]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_l2 = GridSearchCV(
        LogisticRegression(l1_ratio=0, solver='saga', max_iter=5000),
        param_grid={'C': C_values}, cv=cv, scoring='roc_auc'
    )
    grid_l2.fit(X_train_s, y_train)

    grid_l1 = GridSearchCV(
        LogisticRegression(l1_ratio=1, solver='saga', max_iter=5000),
        param_grid={'C': C_values}, cv=cv, scoring='roc_auc'
    )
    grid_l1.fit(X_train_s, y_train)

    best_l1 = grid_l1.best_estimator_
    best_l2 = grid_l2.best_estimator_

    def eval_model(model, name):
        prob = model.predict_proba(X_test_s)[:, 1]
        pred = model.predict(X_test_s)
        return {
            'model': name,
            'accuracy': accuracy_score(y_test, pred),
            'recall': recall_score(y_test, pred),
            'roc_auc': roc_auc_score(y_test, prob),
            'log_loss': log_loss(y_test, prob),
            'n_nonzero': np.sum(np.abs(model.coef_[0]) > 1e-6),
        }

    res_l1 = eval_model(best_l1, 'L1')
    res_l2 = eval_model(best_l2, 'L2')
    df_reg = pd.DataFrame([res_l1, res_l2])
    print("\n  L1 vs L2 on Titanic:")
    print(df_reg.to_string(index=False, float_format='%.4f'))

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Threshold curves
    axes[0].plot(df_thresh['threshold'], df_thresh['accuracy'], 'g-o', label='Accuracy')
    axes[0].plot(df_thresh['threshold'], df_thresh['precision'], 'b-s', label='Precision')
    axes[0].plot(df_thresh['threshold'], df_thresh['recall'], 'r-^', label='Recall')
    axes[0].plot(df_thresh['threshold'], df_thresh['f1'], 'k-D', label='F1')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Metric Value')
    axes[0].set_title('Titanic: Threshold Analysis')
    axes[0].legend()

    # Feature importance (coefficients)
    coefs = logr.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))
    axes[1].barh(np.array(feature_cols)[sorted_idx], coefs[sorted_idx], color='steelblue')
    axes[1].set_xlabel('Coefficient')
    axes[1].set_title('Titanic: Feature Coefficients (LogisticRegression)')
    axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "task_e_titanic.png"), dpi=150)
    plt.close(fig)
    print("  Saved task_e_titanic.png")

    # --- Write real_data_report.md ---
    report = f"""# Task E: Titanic Real Data Report

## 数据概况

- 数据集：Titanic 训练集
- 样本量：{df.shape[0]}
- 目标变量：Survived（是否生还）
- 正类比例：{y.mean():.3f}
- 特征：Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S

## 基础指标 (threshold=0.5)

| 指标 | 值 |
|------|-----|
| TP | {tp} |
| TN | {tn} |
| FP | {fp} |
| FN | {fn} |
| Accuracy | {acc:.4f} |
| Precision | {prec:.4f} |
| Recall | {rec:.4f} |
| F1 | {f1:.4f} |
| ROC-AUC | {roc_auc:.4f} |

## Threshold 扫描

{df_thresh.to_markdown(index=False, floatfmt='.4f')}

## L1 vs L2 对比

{df_reg.to_markdown(index=False, floatfmt='.4f')}

## E3. 真实业务问题

### Q1. 单看 accuracy 会不会误导判断？
**会**。Titanic 数据中，存活率约为 {y.mean():.1%}，有一定的类别不平衡。如果模型把所有人都预测为"未存活"，accuracy 也能达到约 {1-y.mean():.1%}。但这样的模型完全没有识别存活者的能力。因此，单看 accuracy 会掩盖模型在少数类（存活者）上的表现，容易产生误导。

### Q2. 你最后更信任哪个指标？
**更信任 ROC-AUC 和 Recall**。ROC-AUC 衡量的是模型在所有阈值下区分正负类的综合能力，不受阈值选择的影响。Recall 衡量的是模型能捕获多少真正的存活者。在 Titanic 这个场景中（虽然不像疾病初筛那么极端），我们仍然希望模型能够识别出可能存活的人，而不是简单地预测多数类。

### Q3. 如果你要向业务方解释模型输出，你会强调"类别"还是"概率"？
**强调"概率"**。原因：(1) 概率提供了更多信息——告诉业务方"这个乘客有 80% 的存活概率"比简单地说"会存活"更有价值；(2) 概率允许业务方根据自己的偏好设定阈值——如果他们更保守或更激进，可以自行调整；(3) 概率可以用于排序——即使不做二分类，也可以按存活概率排序，优先救助概率最低的人。
"""
    report_path = os.path.join(RESULTS_DIR, "real_data_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")


# =====================================================================
#  Task F: Summary
# =====================================================================

def task_f():
    print("\n" + "=" * 60)
    print("Task F: Summary")
    print("=" * 60)

    report = """# Week 15 Summary

## 1. 为什么逻辑回归不是"线性回归后面接一个 sigmoid"这么简单？

逻辑回归和线性回归在三个层面有本质区别：

1. **概率模型不同**：线性回归假设 $Y|X \\sim N(X\\beta, \\sigma^2)$，逻辑回归假设 $Y|X \\sim Bernoulli(\\sigma(X\\beta))$。这不是简单的函数变换，而是对数据生成机制的根本不同假设。

2. **损失函数不同**：线性回归最小化 MSE，逻辑回归最大化 Bernoulli 似然（等价于最小化 log loss）。两者的优化目标来自不同的统计原理。

3. **输出解释不同**：线性回归的输出是条件均值 $E[Y|X]$，逻辑回归的输出是条件概率 $P(Y=1|X)$。前者是连续值的期望，后者是 Bernoulli 分布的参数。

因此，逻辑回归不是简单的"线性回归 + sigmoid"，而是一个完整的概率建模框架，从分布假设到参数估计到输出解释都有自己的逻辑。

## 2. sigmoid、Bernoulli likelihood、log loss 三者之间是什么关系？

三者构成了逻辑回归的完整链条：

- **sigmoid** 是连接函数（link function），将线性组合 $X\\beta$ 映射到 (0,1) 区间，使其可以解释为概率。它解决了"如何从线性预测值得到概率"的问题。

- **Bernoulli likelihood** 是概率模型，假设给定概率 p 时，Y 服从 Bernoulli(p) 分布。它解决了"如何用概率描述二分类数据"的问题。

- **log loss** 是训练目标，是 Bernoulli likelihood 取负对数后的形式。它解决了"如何优化模型参数"的问题。

三者的关系：sigmoid 产生概率 → Bernoulli likelihood 描述概率与标签的关系 → log loss 是 likelihood 的可优化形式。

## 3. 为什么分类模型不能只看 accuracy？

1. **类别不平衡问题**：当正负类比例悬殊时，即使模型把所有样本都预测为多数类，accuracy 也可能很高。这样的模型没有实际价值。

2. **阈值敏感性**：accuracy 依赖于固定的分类阈值，但最优阈值因业务场景而异。只看 accuracy 会忽略模型在不同阈值下的表现。

3. **错误代价不对称**：在很多场景中，FP 和 FN 的代价不同。比如疾病漏诊（FN）的代价远高于误报（FP）。accuracy 对 FP 和 FN 一视同仁，无法反映这种不对称性。

4. **应该结合 Precision、Recall、F1、ROC-AUC 等指标**，从不同角度评估模型的表现。

## 4. L1 和 L2 逻辑回归分别更适合什么目标？

- **L1（Lasso）更适合特征选择**：L1 会将不重要的特征系数压缩到 exactly 0，实现自动特征选择。当业务方需要一个"更短的变量名单"、需要可解释性、或者特征维度很高时，L1 是更好的选择。

- **L2（Ridge）更适合稳定预测**：L2 对所有特征系数进行均匀收缩，不会产生稀疏解。当业务方更在意预测稳定性、特征之间存在共线性、或者不需要特征选择时，L2 是更好的选择。

## 5. 为什么逻辑回归仍然是一个很强的 baseline？

如果业务方要的是"一个能输出稳定概率、还能解释变量方向"的模型，逻辑回归有以下优势：

1. **输出是概率**：逻辑回归直接输出 $P(Y=1|X)$，天然适合需要概率估计的场景（如风险评分、推荐系统）。

2. **系数可解释**：每个特征的系数方向和大小直接反映了该特征对结果的影响方向和强度。正系数意味着该特征增加正类概率，负系数意味着减少。

3. **训练稳定**：逻辑回归的损失函数是凸函数，有全局最优解，不存在局部最优问题。

4. **计算高效**：相比深度学习等复杂模型，逻辑回归训练速度快，部署简单。

5. **正则化支持**：通过 L1/L2 正则化，逻辑回归可以处理高维数据和共线性问题。

6. **校准良好**：逻辑回归的输出通常具有良好的概率校准性，即输出的概率与实际频率一致。

因此，在很多实际业务场景中，逻辑回归作为 baseline 模型，不仅简单易用，而且在可解释性和稳定性上都有很好的表现。
"""
    report_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved {report_path}")


# =====================================================================
#  Main entry point
# =====================================================================

def main():
    print("=" * 60)
    print("Week 15: Logistic Regression and Binary Classification")
    print("=" * 60)

    # Task A
    df_syn, X_test_a, y_test_a, logr_a, logr_prob_a = task_a()

    # Task B
    task_b()

    # Task C (uses Task A's test results)
    task_c(X_test_a, y_test_a, logr_prob_a)

    # Task D
    task_d()

    # Task E
    task_e()

    # Task F
    task_f()

    print("\n" + "=" * 60)
    print("All tasks completed! Check results/ for reports and figures.")
    print("=" * 60)


if __name__ == "__main__":
    main()
