import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True, precision=4)

# 导入我们写的模块
from engine import CustomOLS
from evalutor import evaluate_model

# ==============================
# Task 4: 自动创建/清空 results 文件夹
# ==============================
def setup_results_dir() -> Path:
    results_dir = Path(__file__).parent / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    return results_dir

# ==============================
# Task 3A: 合成数据测试
# ==============================
def scenario_A_synthetic(results_dir: Path):
    # 生成数据
    np.random.seed(42)
    n = 1000
    X = np.hstack([np.ones((n, 1)), np.random.randn(n, 3)])
    beta_true = np.array([10, 2.5, -1.3, 4.0])
    y = X @ beta_true + np.random.randn(n) * 1.5

    # 划分训练测试
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型
    model_custom = CustomOLS()
    model_sklearn = LinearRegression(fit_intercept=False)

    # 评估
    line1 = evaluate_model(model_custom, X_train, y_train, X_test, y_test, "CustomOLS")
    line2 = evaluate_model(model_sklearn, X_train, y_train, X_test, y_test, "Sklearn-LR")

    # 保存报告
    path = results_dir / "synthetic_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Synthetic Data Test\n\n")
        f.write("| Model            | Time     | R²    |\n")
        f.write("|------------------|----------|-------|\n")
        f.write(line1 + "\n")
        f.write(line2 + "\n")

# ==============================
# Task 3B: 真实市场数据（NA / EU）
# ==============================
def scenario_B_real_world(results_dir: Path):
    # 读取数据
    csv_file = Path(__file__).parent / "q3_marketing.csv"
    df = pd.read_csv(csv_file, na_filter=False, keep_default_na=False)


    # 特征 + 截距
    X = df[["TV_Budget", "Radio_Budget", "SocialMedia_Budget", "Is_Holiday"]].values
    X = np.hstack([np.ones((len(X), 1)), X])
    y = df["Sales"].values
    region = df["Region"].values

    # 拆分
    mask_na = (region == "NA")
    mask_eu = (region == "EU")

    X_na, y_na = X[mask_na], y[mask_na]
    X_eu, y_eu = X[mask_eu], y[mask_eu]

    # 建模（OOP 多实例）
    model_na = CustomOLS().fit(X_na, y_na)
    model_eu = CustomOLS().fit(X_eu, y_eu)

    # F 检验：广告系数是否全为 0
    C = np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
    ])
    f_na = model_na.f_test(C)
    f_eu = model_eu.f_test(C)

    # 保存报告
    path = results_dir / "real_world_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Real World Marketing Report\n\n")
        f.write(f"北美市场样本数: {X_na.shape[0]}\n")
        f.write(f"欧洲市场样本数: {X_eu.shape[0]}\n\n")

        f.write("## 北美模型 F 检验（广告是否有效）\n")
        f.write(f"F统计量: {f_na['f_stat']:.4f}\n")
        f.write(f"P值: {f_na['p_value']:.4f} → {'显著' if f_na['p_value'] < 0.05 else '不显著'}\n\n")

        f.write("## 欧洲模型 F 检验（广告是否有效）\n")
        f.write(f"F统计量: {f_eu['f_stat']:.4f}\n")
        f.write(f"P值: {f_eu['p_value']:.4f} → {'显著' if f_eu['p_value'] < 0.05 else '不显著'}\n\n")

        f.write("## 模型系数\n")
        f.write(f"NA 系数: {np.round(model_na.coef_, 4)}\n")
        f.write(f"EU  系数: {np.round(model_eu.coef_, 4)}\n")

    # 画图
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.scatter(model_na.predict(X_na), y_na, s=5)
    plt.title(f"NA: R²={model_na.score(X_na, y_na):.3f}")
    plt.xlabel("Predict")
    plt.ylabel("True")

    plt.subplot(1,2,2)
    plt.scatter(model_eu.predict(X_eu), y_eu, s=5)
    plt.title(f"EU: R²={model_eu.score(X_eu, y_eu):.3f}")
    plt.xlabel("Predict")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(results_dir / "market_comparison.png", dpi=150)
    plt.close()

# ==============================
# 主运行
# ==============================
if __name__ == "__main__":
    print("🚀 运行 OLS 回归项目...")
    results_dir = setup_results_dir()

    print("📊 生成合成数据测试...")
    scenario_A_synthetic(results_dir)

    print("🌍 真实市场数据分析...")
    scenario_B_real_world(results_dir)

    print("✅ 全部完成！结果保存在 results/ 文件夹")