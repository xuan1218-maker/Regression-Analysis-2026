"""Week15：逻辑回归、分类指标与阈值权衡。

运行入口：
    uv run week15/main.py

为了兼容老师官方说明，本作业还提供了 ``src/week15/main.py`` 包装入口。
主数据、结果和报告按照用户指定结构保存在 ``week15/`` 目录下。
"""
from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

# 让 week15/main.py 能导入同级上层 src/utils。
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.metrics import (  # noqa: E402
    binary_log_loss_manual,
    scan_thresholds,
    sigmoid,
    summarize_binary_classification,
)
from utils.transformers import CustomStandardScaler  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)
warnings.filterwarnings("ignore")

WEEK_DIR = ROOT / "week15"
DATA_DIR = WEEK_DIR / "data"
RESULTS_DIR = WEEK_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass
class ModelResult:
    """保存单个分类模型的核心结果，方便统一生成报告。"""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    nonzero_coefficients: int | None = None
    best_C: float | None = None


def ensure_dirs() -> None:
    """创建 Week15 所需目录。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    """统一保存图片，避免图像边界被裁剪。"""
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_synthetic_binary(n_samples: int = 720, n_features: int = 32) -> pd.DataFrame:
    """生成带 Bernoulli 概率结构的二分类模拟数据。

    设计思路：
    1. 前 6 个变量是主要解释变量；
    2. corr_1 到 corr_6 是一组明显相关变量；
    3. 其余变量大多是噪声，用于正则化实验；
    4. y 不是硬阈值生成，而是先计算概率 p，再从 Bernoulli(p) 抽样。
    """
    latent_risk = RNG.normal(0, 1, n_samples)
    latent_behavior = RNG.normal(0, 1, n_samples)

    x_signal_1 = latent_risk + RNG.normal(0, 0.35, n_samples)
    x_signal_2 = 0.85 * latent_risk + RNG.normal(0, 0.35, n_samples)
    x_protective_1 = RNG.normal(0, 1, n_samples)
    x_behavior_1 = latent_behavior + RNG.normal(0, 0.40, n_samples)
    x_behavior_2 = 0.75 * latent_behavior + RNG.normal(0, 0.45, n_samples)
    x_noise_base = RNG.normal(0, 1, n_samples)

    corr_block = {
        f"corr_{i}": 0.9 * x_signal_1 + RNG.normal(0, 0.18 + 0.02 * i, n_samples)
        for i in range(1, 7)
    }
    noise = {f"noise_{j}": RNG.normal(0, 1, n_samples) for j in range(1, n_features - 12 + 1)}

    eta = (
        -0.25
        + 1.60 * x_signal_1
        - 1.25 * x_protective_1
        + 1.10 * x_behavior_1
        - 0.80 * x_behavior_2
        + 0.65 * x_signal_2
        + 0.35 * x_noise_base
    )
    true_probability = sigmoid(eta)
    y = RNG.binomial(1, true_probability)

    data = pd.DataFrame(
        {
            "x_signal_1": x_signal_1,
            "x_signal_2": x_signal_2,
            "x_protective_1": x_protective_1,
            "x_behavior_1": x_behavior_1,
            "x_behavior_2": x_behavior_2,
            "x_noise_base": x_noise_base,
            **corr_block,
            **noise,
            "true_probability": true_probability,
            "y": y,
        }
    )
    data.to_csv(DATA_DIR / "synthetic_binary.csv", index=False)
    return data


def prepare_train_test(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """划分训练/测试集，并使用自定义标准化器完成无泄露标准化。"""
    feature_cols = [c for c in df.columns if c not in {"y", "true_probability"}]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    scaler = CustomStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


def evaluate_probabilities(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    nonzero: int | None = None,
    best_C: float | None = None,
) -> ModelResult:
    """统一计算分类指标。"""
    summary = summarize_binary_classification(y_true, y_prob, threshold=threshold)
    return ModelResult(
        name=name,
        accuracy=float(summary["accuracy"]),
        precision=float(summary["precision"]),
        recall=float(summary["recall"]),
        f1=float(summary["F1"]),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        log_loss=float(binary_log_loss_manual(y_true, y_prob)),
        nonzero_coefficients=nonzero,
        best_C=best_C,
    )


def model_results_to_markdown(results: list[ModelResult]) -> str:
    """把模型结果转成 Markdown 表格。"""
    rows = [
        "| 模型 | accuracy | precision | recall | F1 | ROC-AUC | log loss | 非零系数数 | best C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        nz = "" if r.nonzero_coefficients is None else str(r.nonzero_coefficients)
        c = "" if r.best_C is None else f"{r.best_C:.4g}"
        rows.append(
            f"| {r.name} | {r.accuracy:.4f} | {r.precision:.4f} | {r.recall:.4f} | "
            f"{r.f1:.4f} | {r.roc_auc:.4f} | {r.log_loss:.4f} | {nz} | {c} |"
        )
    return "\n".join(rows)




def df_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    """不依赖 tabulate，把小型 DataFrame 转成 Markdown 表格。"""
    if df.empty:
        return "（空表）"
    headers = list(df.columns)
    lines = ["| " + " | ".join(map(str, headers)) + " |"]
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def plot_linear_vs_logistic(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> tuple[ModelResult, ModelResult, dict[str, float]]:
    """训练 LinearRegression 和 LogisticRegression，并画出输出行为差别。"""
    linear_model = LinearRegression()
    logistic_model = LogisticRegression(max_iter=2000, penalty=None, solver="lbfgs")
    linear_model.fit(X_train, y_train)
    logistic_model.fit(X_train, y_train)

    linear_raw = linear_model.predict(X_test)
    linear_prob_clipped = np.clip(linear_raw, 0, 1)
    logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

    linear_result = evaluate_probabilities("LinearRegression（截断到[0,1]后当概率）", y_test, linear_prob_clipped)
    logistic_result = evaluate_probabilities("LogisticRegression", y_test, logistic_prob)

    # 为了画一维曲线，只改变 x_signal_1，其他变量固定在训练均值 0。
    idx = feature_cols.index("x_signal_1")
    grid = np.linspace(X_test[:, idx].min() - 0.5, X_test[:, idx].max() + 0.5, 240)
    X_grid = np.zeros((grid.size, X_train.shape[1]))
    X_grid[:, idx] = grid
    linear_curve = linear_model.predict(X_grid)
    logistic_curve = logistic_model.predict_proba(X_grid)[:, 1]

    plt.figure(figsize=(8.2, 5.2))
    jitter = RNG.normal(0, 0.025, size=y_test.shape[0])
    plt.scatter(X_test[:, idx], y_test + jitter, alpha=0.35, s=18, label="True labels y (jittered)")
    plt.plot(grid, linear_curve, linewidth=2.2, label="LinearRegression 输出")
    plt.plot(grid, logistic_curve, linewidth=2.2, label="LogisticRegression 概率")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axhline(1, linestyle="--", linewidth=1)
    plt.xlabel("x_signal_1（标准化后）")
    plt.ylabel("model output / predicted probability")
    plt.title("LinearRegression vs LogisticRegression outputs")
    plt.legend()
    savefig(FIGURES_DIR / "linear_vs_logistic_output.png")

    output_stats = {
        "linear_min": float(linear_raw.min()),
        "linear_max": float(linear_raw.max()),
        "linear_outside_rate": float(np.mean((linear_raw < 0) | (linear_raw > 1))),
        "logistic_min": float(logistic_prob.min()),
        "logistic_max": float(logistic_prob.max()),
    }
    return linear_result, logistic_result, output_stats


def plot_loss_curves() -> None:
    """画 squared error 与 log loss 随预测概率变化的曲线。"""
    p = np.linspace(0.001, 0.999, 800)
    se_y1 = (1 - p) ** 2
    se_y0 = p**2
    log_y1 = -np.log(p)
    log_y0 = -np.log(1 - p)

    plt.figure(figsize=(8.2, 5.2))
    plt.plot(p, se_y1, label="Squared error, y=1")
    plt.plot(p, log_y1, label="Log loss, y=1")
    plt.plot(p, se_y0, linestyle="--", label="Squared error, y=0")
    plt.plot(p, log_y0, linestyle="--", label="Log loss, y=0")
    plt.ylim(0, 7.2)
    plt.xlabel("predicted probability p")
    plt.ylabel("loss value")
    plt.title("Loss curves: log loss penalizes confident mistakes")
    plt.legend()
    savefig(FIGURES_DIR / "loss_curves.png")


def plot_confusion_matrix(counts: dict[str, int], filename: str, title: str) -> None:
    """画 2x2 混淆矩阵热力图。"""
    matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    plt.figure(figsize=(5.2, 4.4))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="count")
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12)
    plt.title(title)
    savefig(FIGURES_DIR / filename)


def threshold_experiment(y_test: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """扫描 0.1 到 0.9 的分类阈值，并画指标曲线。"""
    thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
    rows = scan_thresholds(y_test, y_prob, thresholds)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "threshold_scan.csv", index=False)

    plt.figure(figsize=(8.2, 5.2))
    for metric in ["accuracy", "precision", "recall", "F1"]:
        plt.plot(df["threshold"], df[metric], marker="o", label=metric)
    plt.xlabel("classification threshold")
    plt.ylabel("metric value")
    plt.ylim(0, 1.05)
    plt.title("Classification metrics across thresholds")
    plt.legend()
    savefig(FIGURES_DIR / "threshold_metrics.png")
    return df


def regularization_experiment(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
) -> tuple[list[ModelResult], pd.DataFrame]:
    """比较 L1 与 L2 正则化逻辑回归。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    c_grid = np.logspace(-2, 2, 13)

    estimators = {
        "L1 Logistic": LogisticRegression(penalty="l1", solver="liblinear", max_iter=3000, random_state=SEED),
        "L2 Logistic": LogisticRegression(penalty="l2", solver="liblinear", max_iter=3000, random_state=SEED),
    }
    results: list[ModelResult] = []
    coef_rows: list[dict[str, float | str]] = []
    for name, estimator in estimators.items():
        grid = GridSearchCV(
            estimator,
            param_grid={"C": c_grid},
            scoring="neg_log_loss",
            cv=cv,
            n_jobs=None,
        )
        grid.fit(X_train, y_train)
        best_model: LogisticRegression = grid.best_estimator_
        prob = best_model.predict_proba(X_test)[:, 1]
        coef = best_model.coef_.ravel()
        nonzero = int(np.sum(np.abs(coef) > 1e-6))
        results.append(
            evaluate_probabilities(
                name,
                y_test,
                prob,
                threshold=0.5,
                nonzero=nonzero,
                best_C=float(grid.best_params_["C"]),
            )
        )
        for col, value in zip(feature_cols, coef):
            coef_rows.append({"model": name, "feature": col, "coefficient": float(value)})

    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(RESULTS_DIR / "regularized_coefficients.csv", index=False)

    # 性能对比图。
    metric_names = ["accuracy", "recall", "roc_auc", "log_loss"]
    x = np.arange(len(metric_names))
    width = 0.35
    plt.figure(figsize=(8.2, 5.2))
    for i, result in enumerate(results):
        values = [getattr(result, m) for m in metric_names]
        offset = (i - 0.5) * width
        plt.bar(x + offset, values, width=width, label=result.name)
    plt.xticks(x, ["accuracy", "recall", "ROC-AUC", "log loss"])
    plt.ylabel("metric value")
    plt.title("L1 vs L2 Logistic: metric comparison")
    plt.legend()
    savefig(FIGURES_DIR / "regularization_metrics.png")

    # 非零系数数量对比图。
    plt.figure(figsize=(6.6, 4.8))
    plt.bar([r.name for r in results], [r.nonzero_coefficients for r in results])
    plt.ylabel("number of nonzero coefficients")
    plt.title("L1 tends to produce a sparse variable list")
    savefig(FIGURES_DIR / "nonzero_coefficients.png")

    # 画绝对系数最大的若干项。
    top_features = (
        coef_df.assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(18)
    )
    plt.figure(figsize=(8.4, 6.2))
    labels = top_features["model"] + ": " + top_features["feature"]
    plt.barh(np.arange(top_features.shape[0]), top_features["coefficient"])
    plt.yticks(np.arange(top_features.shape[0]), labels)
    plt.axvline(0, linewidth=1)
    plt.xlabel("standardized coefficient")
    plt.title("Major coefficients in L1/L2 logistic models")
    plt.gca().invert_yaxis()
    savefig(FIGURES_DIR / "l1_l2_coefficients.png")

    return results, coef_df


def real_data_experiment() -> tuple[pd.DataFrame, list[ModelResult], pd.DataFrame]:
    """使用 sklearn 乳腺癌数据完成真实二分类任务。"""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]
    # sklearn 原始 target 中 0=malignant，1=benign；这里把 malignant 设为正类 1，更贴近疾病初筛。
    df["is_malignant"] = 1 - df["target"]
    df = df.drop(columns=["target"])
    df.to_csv(DATA_DIR / "real_binary_breast_cancer.csv", index=False)

    feature_cols = [c for c in df.columns if c != "is_malignant"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["is_malignant"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    scaler = CustomStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=3000, random_state=SEED)
    base_model.fit(X_train_s, y_train)
    base_prob = base_model.predict_proba(X_test_s)[:, 1]
    base_result = evaluate_probabilities("真实数据 L2 Logistic", y_test, base_prob)

    threshold_df = pd.DataFrame(scan_thresholds(y_test, base_prob, np.round(np.arange(0.1, 1.0, 0.1), 1)))
    threshold_df.to_csv(RESULTS_DIR / "real_threshold_scan.csv", index=False)
    plt.figure(figsize=(8.2, 5.2))
    for metric in ["accuracy", "precision", "recall", "F1"]:
        plt.plot(threshold_df["threshold"], threshold_df[metric], marker="o", label=metric)
    plt.xlabel("classification threshold")
    plt.ylabel("metric value")
    plt.ylim(0, 1.05)
    plt.title("Real breast cancer data: metrics across thresholds")
    plt.legend()
    savefig(FIGURES_DIR / "real_threshold_metrics.png")

    counts = summarize_binary_classification(y_test, base_prob, 0.5)
    plot_confusion_matrix(counts, "real_confusion_matrix.png", "Real data logistic confusion matrix (threshold=0.5)")

    reg_results, _ = regularization_real_data(X_train_s, X_test_s, y_train, y_test)
    return df, [base_result, *reg_results], threshold_df


def regularization_real_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[ModelResult], pd.DataFrame]:
    """真实数据上简要比较 L1 和 L2。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    c_grid = np.logspace(-2, 2, 9)
    rows = []
    results = []
    for name, penalty in [("真实数据 L1 Logistic", "l1"), ("真实数据 L2 Logistic", "l2")]:
        grid = GridSearchCV(
            LogisticRegression(penalty=penalty, solver="liblinear", max_iter=3000, random_state=SEED),
            param_grid={"C": c_grid},
            scoring="neg_log_loss",
            cv=cv,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        prob = model.predict_proba(X_test)[:, 1]
        nonzero = int(np.sum(np.abs(model.coef_.ravel()) > 1e-6))
        result = evaluate_probabilities(
            name,
            y_test,
            prob,
            nonzero=nonzero,
            best_C=float(grid.best_params_["C"]),
        )
        results.append(result)
        rows.append(result.__dict__)
    return results, pd.DataFrame(rows)


def write_synthetic_report(
    df: pd.DataFrame,
    linear_result: ModelResult,
    logistic_result: ModelResult,
    output_stats: dict[str, float],
) -> None:
    """生成 synthetic_report.md。"""
    positive_rate = df["y"].mean()
    corr_12 = df[["x_signal_1", "x_signal_2"]].corr().iloc[0, 1]
    report = f"""# Week15 模拟二分类报告：LinearRegression vs LogisticRegression

## 1. 数据生成机制（DGP）

本报告对应 Task A。模拟数据保存在：

```text
week15/data/synthetic_binary.csv
```

样本量为 **{df.shape[0]}**，特征数为 **{df.shape[1] - 2}**，目标变量为 `y`。`y` 不是通过硬阈值手写出来的，而是先生成真实概率 `true_probability`，再从 Bernoulli 分布中抽样得到。

真实线性得分为：

```text
eta = -0.25
      + 1.60 * x_signal_1
      - 1.25 * x_protective_1
      + 1.10 * x_behavior_1
      - 0.80 * x_behavior_2
      + 0.65 * x_signal_2
      + 0.35 * x_noise_base
```

然后通过 sigmoid 转成概率：

$$
p = \\frac{{1}}{{1 + e^{{-\\eta}}}}
$$

最后生成：

$$
Y \\sim Bernoulli(p)
$$

其中 `x_signal_1`、`x_behavior_1`、`x_signal_2`、`x_noise_base` 会提高正类概率；`x_protective_1` 和 `x_behavior_2` 会降低正类概率。模拟数据正类比例为 **{positive_rate:.3f}**。此外，`x_signal_1` 与 `x_signal_2` 的相关系数为 **{corr_12:.3f}**，后面的 `corr_*` 变量也构成一组相关特征，用于正则化实验。

## 2. LinearRegression 与 LogisticRegression 对比

{model_results_to_markdown([linear_result, logistic_result])}

`LinearRegression` 可以勉强通过 0.5 阈值做分类，但它的输出不是概率。测试集上，它的原始输出最小值为 **{output_stats['linear_min']:.3f}**，最大值为 **{output_stats['linear_max']:.3f}**，其中有 **{output_stats['linear_outside_rate']:.1%}** 的预测值落在 `[0, 1]` 之外。`LogisticRegression` 的输出则天然落在 `[0, 1]`，本次测试集概率范围为 **{output_stats['logistic_min']:.3f}** 到 **{output_stats['logistic_max']:.3f}**。

## 3. 核心图：输出行为差别

![LinearRegression 与 LogisticRegression 输出行为对比](figures/linear_vs_logistic_output.png)

这张图的横轴是标准化后的 `x_signal_1`，纵轴是模型输出或预测概率。散点表示True 0/1 标签，蓝色/橙色曲线分别表示 LinearRegression 输出和 LogisticRegression 预测概率。图中最想说明的是：线性回归输出没有概率边界，可能小于 0 或大于 1；逻辑回归通过 sigmoid 函数把输出压到 0 到 1，更适合解释为“正类发生概率”。

## 4. Task A 核心问题回答

**1）LinearRegression 在这个任务里最不自然的地方是什么？**

最不自然的是它的输出本质上是连续数值，不是概率。即使可以用 0.5 阈值硬分类，它仍然可能输出负数或大于 1 的值，因此不能直接解释为“事件发生概率”。

**2）为什么逻辑回归的输出更容易解释成概率？**

逻辑回归先计算线性得分，再通过 sigmoid 映射到 `[0,1]`。这个输出可以直接看作 Bernoulli 分布中的参数 `p`，也就是正类发生概率。

**3）关键区别是“能不能分类”，还是“输出是否有概率意义”？**

关键区别不是能不能分类。LinearRegression 也可以硬切成 0/1；真正区别是 LogisticRegression 的输出有明确概率意义，并且训练目标也与 Bernoulli likelihood 和 log loss 相匹配。
"""
    (RESULTS_DIR / "synthetic_report.md").write_text(report, encoding="utf-8")


def write_threshold_report(
    y_test: np.ndarray,
    logistic_prob: np.ndarray,
    threshold_df: pd.DataFrame,
) -> None:
    """生成 threshold_report.md。"""
    basic = summarize_binary_classification(y_test, logistic_prob, threshold=0.5)
    plot_confusion_matrix(basic, "confusion_matrix_logistic.png", "Synthetic logistic confusion matrix (threshold=0.5)")

    table = df_to_markdown(threshold_df[["threshold", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "F1"]])
    best_f1_row = threshold_df.loc[threshold_df["F1"].idxmax()]
    report = f"""# Week15 阈值、混淆矩阵与 log loss 报告

## 1. Bernoulli、likelihood 与 log loss

### 1.1 Bernoulli 分布

$$
Y \\sim Bernoulli(p)
$$

在二分类任务中，目标变量 `Y` 只有 0 和 1 两种取值。`p` 表示 `Y=1` 的概率，`1-p` 表示 `Y=0` 的概率。逻辑回归输出的不是普通连续值，而是这个 Bernoulli 概率参数 `p`。

### 1.2 单样本 likelihood

$$
L(p;y)=p^y(1-p)^{{1-y}}
$$

当真实标签 `y=1` 时，likelihood 变成 `p`；当真实标签 `y=0` 时，likelihood 变成 `1-p`。所以这个公式把两种类别统一写在一个表达式里。最大似然估计的思想就是让模型给真实结果分配尽可能高的概率。

### 1.3 单样本负对数似然 / log loss

$$
\\ell(p;y)=-\\left[y\\log(p)+(1-y)\\log(1-p)\\right]
$$

对 likelihood 取负对数后，就得到单样本 log loss。模型如果给真实类别很低概率，log loss 会迅速变大。因此 log loss 很适合训练概率模型，尤其适合惩罚“错得很自信”的预测。

## 2. 损失函数图

![损失函数随预测概率变化](figures/loss_curves.png)

这张图的横轴是预测为正类的概率 `p`，纵轴是 loss value。实线表示真实标签 `y=1`，虚线表示真实标签 `y=0`；图中同时比较 squared error 和 log loss。可以看到，当模型错得很自信时，比如 `y=1` 但 `p` 接近 0，或者 `y=0` 但 `p` 接近 1，log loss 的惩罚会远大于 squared error。

这说明二分类中概率预测不能只关心最后类别，也要关心模型对概率的自信程度。log loss 不是凭空指定的损失函数，而是来自 Bernoulli likelihood 的负对数形式。

## 3. 阈值 0.5 下的混淆矩阵和基础指标

![Logistic 混淆矩阵](figures/confusion_matrix_logistic.png)

| 指标 | 数值 |
|---|---:|
| TP | {int(basic['TP'])} |
| TN | {int(basic['TN'])} |
| FP | {int(basic['FP'])} |
| FN | {int(basic['FN'])} |
| accuracy | {basic['accuracy']:.4f} |
| precision | {basic['precision']:.4f} |
| recall | {basic['recall']:.4f} |
| F1 | {basic['F1']:.4f} |

## 4. Threshold 扫描结果

{table}

![Threshold 指标曲线](figures/threshold_metrics.png)

这张图的横轴是 classification threshold，纵轴是 metric value。四条曲线分别表示 accuracy、precision、recall 和 F1。一般来说，阈值升高时，模型更保守，预测为正类的样本减少，因此 precision 可能上升，但 recall 往往下降；阈值降低时，模型更容易报正类，recall 会提高，但 FP 也可能增加。

本次模拟数据中，F1 最高的阈值是 **{best_f1_row['threshold']:.1f}**，对应 F1 为 **{best_f1_row['F1']:.4f}**。这说明默认 0.5 并不一定是最适合所有业务目标的阈值。

## 5. 业务场景解释：疾病初筛

如果这是疾病初筛，我会更重视 **recall**。原因是漏掉真正有病的人（FN）通常比让健康人进一步复查（FP）更危险。因此，如果业务方目标是尽可能减少漏诊，我会建议选择比 0.5 更低的阈值，例如 0.3 或 0.4，并向业务方说明：这样会提高召回率，但也会带来更多假阳性，需要后续复查环节承接。

如果业务方的复查成本非常高，则可以把阈值调高，牺牲一部分 recall 来换更高 precision。因此阈值不是纯技术参数，而是业务成本权衡。
"""
    (RESULTS_DIR / "threshold_report.md").write_text(report, encoding="utf-8")


def write_regularization_report(results: list[ModelResult], coef_df: pd.DataFrame) -> None:
    """生成 regularization_report.md。"""
    table = model_results_to_markdown(results)
    l1 = next(r for r in results if "L1" in r.name)
    l2 = next(r for r in results if "L2" in r.name)
    sparse_name = "L1 Logistic" if (l1.nonzero_coefficients or 0) < (l2.nonzero_coefficients or 0) else "L2 Logistic"
    top_l1 = (
        coef_df[coef_df["model"] == "L1 Logistic"]
        .assign(abs_coef=lambda d: d["coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .head(10)[["feature", "coefficient"]]
    )
    top_l1 = df_to_markdown(top_l1)
    report = f"""# Week15 正则化逻辑回归报告：L1 vs L2

## 1. 实验设计

本任务对应 Task D。模拟数据中共有 32 个特征，满足“特征数不少于 20”的要求。其中 `x_signal_1`、`x_signal_2` 和 `corr_1` 到 `corr_6` 构成明显相关变量组，另外还加入了多个噪声变量。

所有模型都先使用自定义 `CustomStandardScaler` 在训练集上拟合标准化参数，然后在测试集上 transform，避免数据泄露。`L1` 和 `L2` 逻辑回归都通过 5 折交叉验证选择超参数 `C`，评分标准为负 log loss。

## 2. 测试集表现与模型复杂度

{table}

![L1 与 L2 性能对比](figures/regularization_metrics.png)

这张图的横轴是分类指标，纵轴是metric value；不同柱子代表 L1 和 L2 逻辑回归。它展示两类正则化模型在 accuracy、recall、ROC-AUC 和 log loss 上的差异。

![非零系数数量对比](figures/nonzero_coefficients.png)

这张图的横轴是模型，纵轴是number of nonzero coefficients。L1 正则化会把部分系数压成 0，因此更容易得到稀疏模型；L2 正则化会缩小系数，但通常保留大多数变量。

![L1/L2 主要系数](figures/l1_l2_coefficients.png)

这张图展示绝对值较大的标准化系数。横轴是系数大小，纵轴是模型和变量名。它帮助我们观察 L1/L2 对重要变量的保留方式。

## 3. L1 模型保留的前 10 个主要变量

{top_l1}

## 4. 核心问题回答

**1）L1 和 L2 的预测表现差很多吗？**

本次实验中二者预测表现通常不会差很多，主要差异体现在模型复杂度和解释方式上。L2 更倾向于稳定利用全部变量，L1 更倾向于压缩出更短变量名单。

**2）哪一个模型更稀疏？**

本次结果中更稀疏的是 **{sparse_name}**。从number of nonzero coefficients图可以看出，L1 的非零系数数量通常少于 L2。

**3）哪一个更适合给出更短变量名单？**

如果业务方希望得到一个更短、更容易解释的变量名单，我更倾向于 L1。它能直接把一部分变量系数压为 0，因此天然具有变量选择作用。

**4）如果业务方更在意模型稳定性而不是变量筛选，更偏向哪一个？**

如果业务方更在意稳定概率输出，我更偏向 L2。原因是 L2 不会在相关变量之间过于激进地只保留某一个变量，而是把相关变量的系数整体压小，因此在高共线性数据中往往更稳定。
"""
    (RESULTS_DIR / "regularization_report.md").write_text(report, encoding="utf-8")


def write_real_data_report(
    df: pd.DataFrame,
    results: list[ModelResult],
    threshold_df: pd.DataFrame,
) -> None:
    """生成 real_data_report.md。"""
    positive_rate = df["is_malignant"].mean()
    table = model_results_to_markdown(results)
    best_recall = threshold_df.loc[threshold_df["recall"].idxmax()]
    report = f"""# Week15 真实二分类数据报告：乳腺癌筛查

## 1. 数据说明

真实数据保存在：

```text
week15/data/real_binary_breast_cancer.csv
```

这份数据来自 sklearn 自带的 breast cancer 数据集，共 **{df.shape[0]}** 行、**{df.shape[1] - 1}** 个数值特征。原始数据的目标变量区分 malignant 和 benign。本作业把 `is_malignant=1` 设为正类，即“恶性肿瘤”，更符合疾病初筛场景。

正类比例为 **{positive_rate:.3f}**。这说明它不是完全平衡的数据；如果只看 accuracy，可能会忽略漏诊风险。

## 2. 模型结果

{table}

![真实数据混淆矩阵](figures/real_confusion_matrix.png)

这张图展示默认阈值 0.5 下的混淆矩阵。横轴是预测类别，纵轴是真实类别。对于疾病初筛，最需要关注的是 FN，也就是真实恶性但模型预测为非恶性的样本。

## 3. 真实数据 threshold 分析

![真实数据阈值曲线](figures/real_threshold_metrics.png)

这张图横轴是 classification threshold，纵轴是metric value。四条曲线分别表示 accuracy、precision、recall 和 F1。阈值降低时，模型更容易预测为恶性，因此 recall 往往提高；阈值升高时，模型更保守，precision 可能提高，但 recall 可能下降。

在疾病初筛里，我更信任 recall，因为漏诊的成本通常高于误报。若业务方要求尽可能减少漏诊，可以选择 recall 较高的阈值，例如本次扫描中 recall 最高的阈值是 **{best_recall['threshold']:.1f}**。

## 4. 回答真实业务问题

**1）单看 accuracy 会不会误导判断？**

会。疾病筛查中，如果模型 accuracy 很高，但漏掉恶性样本，业务风险仍然很大。因此 accuracy 不能单独作为最终指标。

**2）最后更信任哪个指标？为什么？**

我更重视 recall，同时参考 precision 和 F1。recall 直接反映模型能找回多少真正恶性样本，更贴近初筛场景的安全目标。

**3）向业务方解释模型输出时，强调类别还是概率？**

我会强调概率。类别是由阈值切出来的结果，阈值可以根据业务成本调整；概率能告诉业务方风险程度，更适合排序、复查资源分配和人工审核。
"""
    (RESULTS_DIR / "real_data_report.md").write_text(report, encoding="utf-8")


def write_summary() -> None:
    """生成 Week15 总结报告。"""
    report = """# Week15 总结报告

## 1. 为什么逻辑回归不是“线性回归后面接一个 sigmoid”这么简单？

逻辑回归确实使用线性得分和 sigmoid，但它的统计含义不只是“套一个函数”。它假设目标变量服从 Bernoulli 分布，模型输出的是 `P(Y=1|X)`，训练目标对应 Bernoulli likelihood 的最大化，也就是最小化 log loss。因此逻辑回归从模型输出、概率解释到优化目标都是为二分类问题设计的。

## 2. sigmoid、Bernoulli likelihood、log loss 三者之间是什么关系？

sigmoid 把线性得分映射成 0 到 1 的概率 `p`；Bernoulli likelihood 用这个 `p` 给真实标签分配概率；log loss 则是 Bernoulli likelihood 的负对数形式。三者连起来就是：线性特征组合 -> 概率输出 -> 最大似然估计 -> log loss 优化。

## 3. 为什么分类模型不能只看 accuracy？

accuracy 只看总体分类正确率，但不同错误类型的业务成本可能差别很大。比如疾病初筛中，FN 的代价通常高于 FP；信用违约中，FP 和 FN 也对应不同的资金损失和机会成本。因此还要看 precision、recall、F1、ROC-AUC、log loss，并结合阈值分析。

## 4. L1 和 L2 逻辑回归分别更适合什么目标？

L1 更适合变量筛选和得到较短变量名单，因为它能把一部分系数压到 0。L2 更适合稳定建模和缓解共线性，因为它通常保留全部变量但缩小系数。如果业务方想要稀疏解释，可以优先考虑 L1；如果更看重稳定概率输出，可以优先考虑 L2。

## 5. 为什么逻辑回归仍然是一个很强的 baseline？

逻辑回归训练快、结果稳定、输出概率、系数方向容易解释，还能通过 L1/L2 正则化适应高维分类任务。即使在工业场景中，它也常被用作强基线模型，尤其适合需要概率解释、阈值调整和业务沟通的二分类任务。

## 6. 本周结果文件

- `synthetic_report.md`：模拟数据、LinearRegression 与 LogisticRegression 对比；
- `threshold_report.md`：Bernoulli/log loss、混淆矩阵与阈值扫描；
- `regularization_report.md`：L1/L2 正则化逻辑回归；
- `real_data_report.md`：真实乳腺癌筛查数据；
- `figures/`：所有报告嵌入图片。
"""
    (RESULTS_DIR / "summary.md").write_text(report, encoding="utf-8")


def write_readme() -> None:
    """更新中文 README。"""
    text = """# Week15 作业：逻辑回归与二分类

本作业按用户指定结构组织：

```text
students/07_nc/
├── pyproject.toml
├── src/
│   └── utils/
└── week15/
    ├── main.py
    ├── data/
    └── results/
```

运行方式：

```bash
cd students/07_nc
uv run week15/main.py
```

为了兼容老师官方说明，也提供包装入口：

```bash
uv run src/week15/main.py
```

`src/utils/` 是在 Week10-Week14 基础上继续维护的，没有删减前几周功能；Week15 在 `metrics.py` 中新增了二分类混淆矩阵、classification metrics、log loss 和 threshold scan 工具。
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """执行 Week15 全部流程。"""
    ensure_dirs()
    write_readme()

    # Task A：模拟二分类数据与 Linear/Logistic 对比。
    synthetic_df = generate_synthetic_binary()
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(synthetic_df)
    linear_result, logistic_result, output_stats = plot_linear_vs_logistic(
        X_train, X_test, y_train, y_test, feature_cols
    )

    # 重新训练逻辑回归，用于阈值实验。
    logistic_model = LogisticRegression(max_iter=2000, penalty=None, solver="lbfgs")
    logistic_model.fit(X_train, y_train)
    logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

    # Task B/C：损失曲线、混淆矩阵和 threshold 扫描。
    plot_loss_curves()
    threshold_df = threshold_experiment(y_test, logistic_prob)

    # Task D：L1/L2 正则化。
    reg_results, coef_df = regularization_experiment(X_train, X_test, y_train, y_test, feature_cols)

    # Task E：真实数据选做。
    real_df, real_results, real_threshold_df = real_data_experiment()

    # 生成报告。
    write_synthetic_report(synthetic_df, linear_result, logistic_result, output_stats)
    write_threshold_report(y_test, logistic_prob, threshold_df)
    write_regularization_report(reg_results, coef_df)
    write_real_data_report(real_df, real_results, real_threshold_df)
    write_summary()

    print("Week15 作业已完成。结果保存在：", RESULTS_DIR)


if __name__ == "__main__":
    main()
