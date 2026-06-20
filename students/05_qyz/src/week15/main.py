"""
Week15 - Logistic Regression (A+ 标准实验流程)
功能：完整实现逻辑回归全套对比实验
实验模块：
  Task A 仿真数据 + 线性回归/逻辑回归输出对比
  Task B 损失函数对比 (LogLoss vs MSE)
  Task C 分类阈值遍历 + 指标权衡分析
  Task D L1 / L2 正则化效果对比
  Task E 真实数据集(乳腺癌)建模
  Task F ROC 曲线 & PR 曲线绘制
  自动生成 CSV 指标文件 + Markdown 实验报告
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入模型与评估工具
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)

# =========================
# 路径与文件夹初始化
# =========================
# 获取当前脚本所在目录
ROOT = Path(__file__).resolve().parent
# 上级目录（备用路径）
SRC = ROOT.parent
sys.path.insert(0, str(SRC))

# 定义三类文件夹：数据、实验报告、图片
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"

# 批量创建文件夹
for p in [DATA_DIR, RESULT_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# 通用工具函数
# =========================
def savefig(name):
    """
    统一保存图片
    :param name: 图片文件名
    """
    plt.tight_layout()  # 自动调整布局，防止标签截断
    plt.savefig(FIG_DIR / name, dpi=160)
    plt.close()  # 关闭画布，释放内存


def write_md(name, content):
    """
    统一写入Markdown报告文件
    :param name: 报告文件名
    :param content: 报告文本内容
    """
    (RESULT_DIR / name).write_text(content, encoding="utf-8")


# =========================
# TASK A - 生成二分类仿真数据 (DGP 数据生成过程)
# =========================
def generate_data(n=600):
    """
    生成符合逻辑回归假设的二分类仿真数据集
    :param n: 样本总量
    :return: 包含特征+标签的DataFrame
    原理：线性组合 -> Sigmoid映射概率 -> 伯努利采样标签
    """
    np.random.seed(42)  # 固定随机种子，保证实验可复现

    # 生成4维标准正态特征
    X = np.random.randn(n, 4)
    # 自定义特征权重 (线性项系数)
    beta = np.array([2.0, -1.5, 1.2, 0.8])

    # 线性组合 logit = X·β
    logits = X @ beta
    # Sigmoid 函数映射为 0~1 概率
    p = 1 / (1 + np.exp(-logits))
    # 根据概率采样二分类标签 y ∈ {0,1}
    y = np.random.binomial(1, p)

    # 构造数据表
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    df["y"] = y

    # 保存仿真数据到csv
    df.to_csv(DATA_DIR / "synthetic_binary.csv", index=False)
    return df


# =========================
# TASK A - 线性回归 vs 逻辑回归 输出对比
# =========================
def task_a(df):
    """
    对比普通线性回归(OLS)与逻辑回归的预测输出
    核心考点：线性回归输出无界，逻辑回归输出为概率(0~1)
    :param df: 仿真数据集
    :return: 测试集真实标签, 逻辑回归预测概率
    """
    X = df[["x1", "x2", "x3", "x4"]].values
    y = df["y"].values

    # 划分训练集 / 测试集
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    # 1. 普通线性回归 (OLS) 基线模型
    from sklearn.linear_model import LinearRegression

    ols = LinearRegression()
    ols.fit(Xtr, ytr)
    ols_pred = ols.predict(Xte)  # OLS输出：无界连续值

    # 2. 逻辑回归模型
    logit = LogisticRegression(max_iter=2000)
    logit.fit(Xtr, ytr)
    prob = logit.predict_proba(Xte)[:, 1]  # 取类别1的预测概率

    # 绘图：单特征维度下两类模型输出对比
    plt.figure()
    plt.scatter(Xte[:, 0], yte, alpha=0.3, label="true")  # 真实标签
    plt.scatter(Xte[:, 0], ols_pred, alpha=0.3, label="OLS")  # 线性回归输出
    plt.scatter(Xte[:, 0], prob, alpha=0.3, label="Logistic")  # 逻辑回归概率
    plt.legend()
    plt.title("Model Comparison")
    savefig("A_compare.png")

    return yte, prob


# =========================
# TASK B - 损失函数对比：LogLoss(对数损失) vs MSE(均方误差)
# =========================
def task_b():
    """
    绘制两种损失函数曲线
    LogLoss对"自信错误"惩罚远大于MSE，更适配概率分类任务
    """
    # 预测概率取值范围 0.001 ~ 0.999 (避开0/1防止log无意义)
    p = np.linspace(0.001, 0.999, 300)

    # 真实标签 y=1 时的损失
    log_y1 = -np.log(p)
    mse_y1 = (1 - p) ** 2

    # 真实标签 y=0 时的损失
    log_y0 = -np.log(1 - p)
    mse_y0 = p**2

    plt.figure()
    plt.plot(p, log_y1, label="logloss y=1")
    plt.plot(p, log_y0, label="logloss y=0")
    plt.plot(p, mse_y1, "--", label="mse y=1")
    plt.plot(p, mse_y0, "--", label="mse y=0")

    plt.title("Loss Landscape: LogLoss vs MSE")
    plt.legend()
    savefig("B_loss.png")


# =========================
# TASK C - 分类阈值遍历 & 指标权衡分析
# =========================
def task_c(y_true, prob):
    """
    遍历不同分类阈值，计算全套分类指标
    考点：阈值 → 精确率/召回率/F1 此消彼长的权衡关系
    :param y_true: 真实标签
    :param prob: 模型预测概率
    :return: 各阈值对应的指标表格
    """
    # 阈值范围 0.1 ~ 1.0，步长0.1
    thresholds = np.arange(0.1, 1.0, 0.1)

    rows = []
    for t in thresholds:
        # 以当前阈值划分0/1分类
        pred = (prob >= t).astype(int)

        # 计算混淆矩阵四大指标 TN, FP, FN, TP
        TN, FP, FN, TP = confusion_matrix(y_true, pred).ravel()

        # 计算多分类指标
        rows.append(
            [
                t,
                TP,
                TN,
                FP,
                FN,
                accuracy_score(y_true, pred),
                precision_score(y_true, pred),
                recall_score(y_true, pred),
                f1_score(y_true, pred),
            ]
        )

    # 构造指标表格并保存
    df = pd.DataFrame(
        rows,
        columns=[
            "threshold",
            "TP",
            "TN",
            "FP",
            "FN",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ],
    )
    df.to_csv(DATA_DIR / "threshold_metrics.csv", index=False)

    # 绘制指标-阈值变化曲线
    plt.figure()
    for m in ["accuracy", "precision", "recall", "f1"]:
        plt.plot(df["threshold"], df[m], label=m)

    plt.legend()
    plt.title("Threshold Tradeoff")
    savefig("C_threshold.png")

    return df


# =========================
# TASK D - L1 / L2 正则化对比
# =========================
def task_d(X, y):
    """
    对比L1、L2正则逻辑回归
    考点：
      L1(Lasso): 产生稀疏解，自动特征筛选
      L2(Ridge): 系数平滑收缩，提升模型稳定性
    :param X: 特征矩阵
    :param y: 标签
    :return: 正则对比指标表
    """
    # 划分训练/测试集
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    # 特征标准化（正则化对量纲敏感，必须归一）
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # 固定正则强度C，分别定义L1、L2正则逻辑回归
    l1 = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000)
    l2 = LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, max_iter=2000)

    l1.fit(Xtr, ytr)
    l2.fit(Xtr, ytr)

    # 封装评估函数：返回全套指标 + 非零系数个数(判断稀疏性)
    def eval(model, name):
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        return [
            name,
            accuracy_score(yte, pred),
            recall_score(yte, pred),
            roc_auc_score(yte, prob),
            log_loss(yte, prob),
            np.sum(np.abs(model.coef_) > 1e-6),  # 统计非零系数数量
        ]

    # 评估并汇总结果
    df = pd.DataFrame(
        [eval(l1, "L1"), eval(l2, "L2")],
        columns=["model", "accuracy", "recall", "roc_auc", "log_loss", "non_zero"],
    )
    df.to_csv(DATA_DIR / "l1_l2_metrics.csv", index=False)
    return df


# =========================
# TASK E - 真实数据集实验 (乳腺癌数据集)
# =========================
def task_e():
    """
    使用sklearn内置乳腺癌真实数据集完成分类
    数据集特点：医学二分类、样本轻微不平衡，贴近实际业务
    :return: 测试集真实标签, 模型预测概率
    """
    from sklearn.datasets import load_breast_cancer

    # 加载乳腺癌数据集
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 划分数据集
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    # 训练基础逻辑回归
    model = LogisticRegression(max_iter=3000)
    model.fit(Xtr, ytr)

    prob = model.predict_proba(Xte)[:, 1]
    return yte, prob


# =========================
# TASK F - ROC 曲线 & PR 曲线绘制
# =========================
def task_f(y_true, prob):
    """
    绘制两类主流分类评估曲线
    1. ROC: FPR(横轴) - TPR(纵轴)，通用分类评估
    2. PR: Recall(横轴) - Precision(纵轴)，更适合不平衡数据
    :param y_true: 真实标签
    :param prob: 预测概率
    """
    # -------- 绘制 ROC 曲线 --------
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")  # 随机猜测参考线
    plt.title("ROC Curve")
    plt.legend()
    savefig("F_ROC.png")

    # -------- 绘制 PR 曲线 --------
    prec, rec, _ = precision_recall_curve(y_true, prob)
    pr_auc = auc(rec, prec)

    plt.figure()
    plt.plot(rec, prec, label=f"AUC={pr_auc:.3f}")
    plt.title("PR Curve")
    plt.legend()
    savefig("F_PR.png")


# =========================
# 自动生成Markdown实验报告
# =========================
def write_reports(y_true, prob, df_c, df_reg):
    """
    汇总所有实验结果，生成结构化MD报告
    :param y_true: 真实标签
    :param prob: 预测概率
    :param df_c: 阈值指标表
    :param df_reg: 正则对比指标表
    """
    # 选取F1分数最大的最优阈值
    best = df_c.loc[df_c["f1"].idxmax()]

    # 1. 仿真数据报告
    write_md(
        "synthetic_report.md",
        f"""
# Synthetic Report

## Best threshold = {best["threshold"]:.2f}

TP={int(best["TP"])}, TN={int(best["TN"])}, FP={int(best["FP"])}, FN={int(best["FN"])}

Accuracy={best["accuracy"]:.4f}
Precision={best["precision"]:.4f}
Recall={best["recall"]:.4f}
F1={best["f1"]:.4f}

![A](figures/A_compare.png)
""",
    )

    # 2. 阈值分析报告
    write_md(
        "threshold_report.md",
        """
# Threshold Analysis

![C](figures/C_threshold.png)

Threshold controls precision-recall tradeoff in classification.
""",
    )

    # 3. 正则化对比报告 (表格形式)
    write_md("regularization_report.md", df_reg.to_markdown(index=False))

    # 4. 真实数据报告
    write_md(
        "real_data_report.md",
        """
# Real Data

ROC and PR curves show ranking ability and imbalance robustness.
![ROC](figures/F_ROC.png)
![PR](figures/F_PR.png)
""",
    )

    # 5. 全局实验总结
    write_md(
        "summary.md",
        """
# Summary

Logistic Regression = sigmoid + Bernoulli likelihood + MLE

Threshold = decision boundary

L1 = sparsity selection
L2 = stability regularization
""",
    )


# =========================
# 主执行入口：串联全部实验流程
# =========================
def main():
    print("Week15 A+ PIPELINE START")

    # 1. 生成仿真数据 + 模型对比
    df = generate_data()
    y, prob = task_a(df)

    # 2. 损失函数对比实验
    task_b()

    # 3. 阈值遍历分析
    df_c = task_c(y, prob)

    # 4. L1/L2 正则对比
    X = df[["x1", "x2", "x3", "x4"]].values
    df_reg = task_d(X, df["y"].values)

    # 5. 真实数据集实验
    y2, p2 = task_e()

    # 6. ROC & PR 曲线
    task_f(y2, p2)

    # 7. 自动生成所有报告
    write_reports(y2, p2, df_c, df_reg)

    print("Week15 A+ PIPELINE DONE")


if __name__ == "__main__":
    main()
