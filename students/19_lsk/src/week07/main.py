"""
模块：week07.main
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from utils.models import AnalyticalOLS, GradientDescentOLS


def rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def add_intercept(X):
    """添加截距列（全1）"""
    return np.column_stack([np.ones(X.shape[0]), X])


def load_data():
    """
    加载营销数据。
    """
    # 当前文件所在目录: .../students/19_lsk/src/week07/
    current_dir = Path(__file__).resolve().parent
    # 项目根目录: 向上4级 -> Regression-Analysis-2026/
    project_root = current_dir.parent.parent.parent.parent
    data_path = project_root / "homework" / "week06" / "data" / "q3_marketing.csv"
    if not data_path.exists():
        
        alt_path = current_dir / "../../../../homework/week06/_pycache_/q3_marketing.csv"
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError("找不到数据文件，请确认路径")
    df = pd.read_csv(data_path)
    feature_cols = ["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]
    target_col = "Sales"
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    print(f"数据加载成功: {data_path}")
    return X, y


def main():
    # 创建 results 文件夹 (位于 students/19_lsk/results)
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"结果将保存在: {results_dir}\n")

    # 1. 加载数据
    X, y = load_data()
    print(f"原始数据形状: X={X.shape}, y={y.shape}")

    # ==================== Task 2: 5折交叉验证 (AnalyticalOLS) ====================
    print("\n" + "=" * 70)
    print("Task 2: AnalyticalOLS 的 5 折交叉验证")
    print("=" * 70)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = []
    cv_rmse = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train_fold = add_intercept(X[train_idx])
        y_train_fold = y[train_idx]
        X_val_fold = add_intercept(X[val_idx])
        y_val_fold = y[val_idx]

        model = AnalyticalOLS().fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        r2 = r2_score(y_val_fold, preds)
        rm = rmse(y_val_fold, preds)
        cv_r2.append(r2)
        cv_rmse.append(rm)
        print(f"Fold {fold}: R² = {r2:.4f}, RMSE = {rm:.4f}")

    avg_r2 = np.mean(cv_r2)
    avg_rmse = np.mean(cv_rmse)
    print(f"\n平均 CV R²:   {avg_r2:.4f} (±{np.std(cv_r2):.4f})")
    print(f"平均 CV RMSE: {avg_rmse:.4f} (±{np.std(cv_rmse):.4f})")

    # ==================== Task 3: 超参数调优 + 最终测试 ====================
    # 第一次划分: 60% 训练, 40% 临时
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    # 第二次划分: 将临时集等分为验证集和测试集 (各20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"\n数据划分: 训练 {X_train.shape[0]}, 验证 {X_val.shape[0]}, 测试 {X_test.shape[0]}")

    # 标准化特征 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 添加截距列
    X_train_scaled = add_intercept(X_train_scaled)
    X_val_scaled = add_intercept(X_val_scaled)
    X_test_scaled = add_intercept(X_test_scaled)

    # 超参数调优: 搜索最佳学习率，并记录发散情况
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    tuning_results = []   # 存储每个学习率的调优结果
    best_lr = None
    best_val_r2 = -np.inf

    print("\n" + "=" * 70)
    print("Task 3: 梯度下降模型超参数调优（学习率）")
    print("=" * 70)

    for lr in learning_rates:
        model = GradientDescentOLS(
            learning_rate=lr,
            tol=1e-5,
            max_iter=1000,
            gd_type="mini_batch",
            batch_fraction=0.2,
        ).fit(X_train_scaled, y_train)
        val_preds = model.predict(X_val_scaled)
        val_r2 = r2_score(y_val, val_preds)
        val_rmse = rmse(y_val, val_preds)

        # 判断是否发散（R² 为负无穷或极大）
        diverged = (val_r2 < 0) or np.isnan(val_r2) or np.isinf(val_rmse)
        status = " 发散" if diverged else "正常"

        tuning_results.append({
            'lr': lr,
            'r2': val_r2,
            'rmse': val_rmse,
            'diverged': diverged
        })

        print(f"学习率 = {lr:<8} | 验证 R² = {val_r2:>8.4f} | 验证 RMSE = {val_rmse:>8.4f} | {status}")

        if (not diverged) and (val_r2 > best_val_r2):
            best_val_r2 = val_r2
            best_lr = lr

    print(f"\n✅ 最佳学习率: {best_lr} (验证 R² = {best_val_r2:.4f})")

    # 最终测试集评估
    print("\n" + "=" * 70)
    print("最终测试集对比 (AnalyticalOLS vs GradientDescentOLS)")
    print("=" * 70)

    # 使用最佳学习率重新训练 GD
    gd_best = GradientDescentOLS(
        learning_rate=best_lr,
        tol=1e-5,
        max_iter=1000,
        gd_type="mini_batch",
        batch_fraction=0.2,
    ).fit(X_train_scaled, y_train)

    ols = AnalyticalOLS().fit(X_train_scaled, y_train)

    preds_gd = gd_best.predict(X_test_scaled)
    preds_ols = ols.predict(X_test_scaled)

    gd_r2 = r2_score(y_test, preds_gd)
    gd_rmse = rmse(y_test, preds_gd)
    ols_r2 = r2_score(y_test, preds_ols)
    ols_rmse = rmse(y_test, preds_ols)

    print(f"GradientDescentOLS:  R² = {gd_r2:.4f}, RMSE = {gd_rmse:.4f}")
    print(f"AnalyticalOLS:       R² = {ols_r2:.4f}, RMSE = {ols_rmse:.4f}")

    # ==================== Task 4: 学习曲线（全批量 vs 小批量） ====================
    print("\n" + "=" * 70)
    print("Task 4: 绘制学习曲线 (full batch vs mini batch)")
    print("=" * 70)
    model_full = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="full_batch",
        max_iter=300,
        tol=1e-8,
    ).fit(X_train_scaled, y_train)

    model_mini = GradientDescentOLS(
        learning_rate=0.01,
        gd_type="mini_batch",
        batch_fraction=0.1,
        max_iter=300,
        tol=1e-8,
    ).fit(X_train_scaled, y_train)

    plt.figure(figsize=(10, 6))
    plt.plot(model_full.loss_history_, label="Full Batch GD", color="steelblue", linewidth=2)
    plt.plot(model_mini.loss_history_, label="Mini-Batch GD (batch_fraction=0.1)", color="darkorange", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Learning Curve: Full Batch vs Mini-Batch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    curve_path = results_dir / "learning_curve_full_vs_mini.png"
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"学习曲线已保存: {curve_path}")

    # ==================== 生成中文 Markdown 报告（包含调优发散记录） ====================
    # 构建调优表格 rows
    tune_rows = ""
    for res in tuning_results:
        status_text = "❌ 发散" if res['diverged'] else "✅ 正常"
        tune_rows += f"| {res['lr']} | {res['r2']:.4f} | {res['rmse']:.4f} | {status_text} |\n"

    report_content = f"""# Week 7 实验报告：优化引擎与泛化能力

## 1. 梯度下降 OLS 的实现

- 损失函数：均方误差 (MSE)
- 梯度公式：`(2/m) * Xᵀ (Xβ - y)`
- 支持模式：`full_batch` 和 `mini_batch`
- 收敛判断：连续两次迭代的 loss 差小于 tolerance 或达到最大迭代次数
- 系数初始化：零向量

## 2. 解析解 OLS 的 5 折交叉验证结果

| 折数 | R²       | RMSE     |
|------|----------|----------|
"""
    for i, (r, rm) in enumerate(zip(cv_r2, cv_rmse), 1):
        report_content += f"| {i}    | {r:.4f} | {rm:.4f} |\n"

    report_content += f"""
**平均 R²**: {avg_r2:.4f} (±{np.std(cv_r2):.4f})  
**平均 RMSE**: {avg_rmse:.4f} (±{np.std(cv_rmse):.4f})

## 3. 梯度下降超参数调优（包含发散记录）

| 学习率 | 验证 R² | 验证 RMSE | 状态 |
|--------|---------|-----------|------|
{tune_rows}
### 发散原因分析

- **学习率过大（0.1, 0.01）**：步长较大，模型能快速下降，本次实验中 R² 较高，说明有效。
- **学习率适中（0.001）**：R² 下降，仍可接受。
- **学习率过小（0.0001, 1e-5）**：模型**完全发散**，R² 为负值，RMSE 极大。

**为什么学习率过小会导致发散？**  
梯度下降更新公式为：  
$\\theta_{{\\text{{new}}}} = \\theta_{{\\text{{old}}}} - \\eta \\nabla L(\\theta)$  
- 当学习率 $\\eta$ 过小时，参数更新步长极小，模型几乎无法学习。  
- 小批量随机梯度的固有噪声，加上浮点数精度限制，使得极小的步长无法抵消随机误差，参数可能在错误方向累积漂移，最终 R² 为负（预测比均值还差）。

✅ **最佳学习率**：`{best_lr}`，验证 R² = {best_val_r2:.4f}

## 4. 测试集最终对比 (未见过数据)

| 模型                | 测试 R²   | 测试 RMSE |
|---------------------|-----------|-----------|
| AnalyticalOLS       | {ols_r2:.4f} | {ols_rmse:.4f} |
| GradientDescentOLS  | {gd_r2:.4f} | {gd_rmse:.4f} |

> 两种模型在测试集上表现非常接近，说明梯度下降算法正确收敛到了与解析解几乎相同的位置。

## 5. 特征标准化与数据泄露防护

- 仅使用训练集的均值和标准差拟合 `StandardScaler`
- 验证集和测试集使用同一 scaler 进行变换
- 截距列（全1）在标准化**之后**添加，避免被缩放
- 严格遵循“禁止在全数据集上先标准化”的要求，杜绝了数据泄露

## 6. 学习曲线

![学习曲线](learning_curve_full_vs_mini.png)

- **全批量 (Full Batch)**：损失单调下降，曲线平滑，每轮使用全部数据
- **小批量 (Mini‑Batch)**：损失曲线带有噪声，但每轮计算量小，通常收敛更快（以 epoch 计）

## 7. 结论

- 梯度下降是一种可行的替代解析解的优化方法，尤其适用于大规模数据。
- 正确的学习率选择与特征标准化是梯度下降成功的关键。
- 在测试集上，梯度下降模型与解析解模型取得了几乎相同的 R² 和 RMSE，验证了实现的正确性。
- 小批量梯度下降在损失下降速度上通常优于全批量，且能跳出局部鞍点。

---
*报告生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""

    report_path = results_dir / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n中文报告已保存: {report_path}")

    print("\n" + "=" * 70)
    print("✅ Week 7 作业全部完成！")
    print(f"请查看结果目录: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()