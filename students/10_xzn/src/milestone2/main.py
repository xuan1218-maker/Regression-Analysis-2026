import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.transformers import CustomStandardScaler
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.models import GradientDescentOLS

# 设置输出路径：students/10_xzn/src/results
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 指向 src 目录
OUTPUT_DIR = os.path.join(BASE_DIR, "results")


def find_data_file():
    possible_paths = [
        "homework/week09/data/dirty_marketing.csv",
        "../homework/week09/data/dirty_marketing.csv",
        "../../homework/week09/data/dirty_marketing.csv",
        "../../../homework/week09/data/dirty_marketing.csv",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    for root, dirs, files in os.walk("."):
        if "dirty_marketing.csv" in files:
            return os.path.abspath(os.path.join(root, "dirty_marketing.csv"))
    raise FileNotFoundError("找不到 dirty_marketing.csv")


def clean_results_folder():
    """创建/清空 results 文件夹（位于 students/10_xzn/src/results）"""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"✓ results 文件夹已准备就绪: {OUTPUT_DIR}")


def bad_cross_validation(data_path):
    print("\n" + "="*60)
    print("Task 3: bad_cross_validation (存在数据泄露)")
    print("="*60)
    
    df = pd.read_csv(data_path)
    
    # 处理分类变量
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"数据形状: {df.shape}")
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # ❌ 危险：全局预处理
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    scaler = CustomStandardScaler()
    X = scaler.fit_transform(X)
    
    print("⚠️ 警告：已对全量数据执行全局预处理（存在数据泄露）")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = GradientDescentOLS(learning_rate=0.01, n_iterations=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = calculate_rmse(y_val, y_pred)
        rmse_scores.append(rmse)
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}")
    
    avg_rmse = np.mean(rmse_scores)
    print(f"\n平均 RMSE: {avg_rmse:.4f}")
    return avg_rmse


def good_cross_validation(data_path):
    print("\n" + "="*60)
    print("Task 4: good_cross_validation (无数据泄露)")
    print("="*60)
    
    df = pd.read_csv(data_path)
    
    # 处理分类变量
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"数据形状: {df.shape}")
    
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # ✅ 正确：只用训练集填补缺失值
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train_raw)
        X_val = imputer.transform(X_val_raw)
        
        # ✅ 正确：只用训练集 fit 标准化器
        scaler = CustomStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = GradientDescentOLS(learning_rate=0.01, n_iterations=500)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        rmse = calculate_rmse(y_val, y_pred)
        rmse_scores.append(rmse)
        print(f"  Fold {fold+1}: RMSE={rmse:.4f}")
    
    avg_rmse = np.mean(rmse_scores)
    print(f"\n平均 RMSE: {avg_rmse:.4f}")
    return avg_rmse


def save_comparison(bad_rmse, good_rmse):
    """保存对比结果到 students/10_xzn/src/results/evaluation_comparison.md"""
    content = f"""# 交叉验证结果对比

## 一、评估指标对比

| 方法 | RMSE |
|------|------|
| 存在数据泄露 (bad_cross_validation) | {bad_rmse:.4f} |
| 无数据泄露 (good_cross_validation) | {good_rmse:.4f} |
| **差异** | {abs(bad_rmse - good_rmse):.4f} |

## 二、核心结论

存在数据泄露的 RMSE 更小（更好看），但这是**虚假的**，因为模型在训练过程中偷看了验证集的信息。

## 三、为什么"好看"是致命的？

1. **虚假的信心**：模型看起来表现很好，但实际上没有学到真正的规律
2. **上线崩盘**：部署到生产环境后，模型对新数据的预测能力远差于验证结果
3. **业务损失**：基于虚假的评估结果做出错误的业务决策

## 四、正确的做法（Leakage-Free Pipeline）

在 `good_cross_validation` 中，我实现了严格的数据隔离：
- 只对训练集执行 `.fit()`
- 用训练集的参数对验证集执行 `.transform()`
- 确保验证集在预处理阶段完全未见

## 五、业务解读

基于 `good_cross_validation` 的 RMSE = {good_rmse:.4f}：

> 这个模型上线后，预测误差大约为 {good_rmse:.2f} 个单位。

**为什么要给老板看 Task 4 的"差成绩"？**

因为 Task 3 的"好成绩"是虚假的、不可信的。只有 Task 4 的"差成绩"才代表模型在真实世界中的实际表现。

---

*报告生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    output_path = os.path.join(OUTPUT_DIR, "evaluation_comparison.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✓ 结果已保存到 {output_path}")


def main():
    print("="*60)
    print("Milestone 2: 工业流水线与无泄漏的泛化评估")
    print("="*60)
    
    clean_results_folder()
    
    try:
        data_path = find_data_file()
        print(f"✓ 找到数据文件: {data_path}")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    bad_rmse = bad_cross_validation(data_path)
    good_rmse = good_cross_validation(data_path)
    save_comparison(bad_rmse, good_rmse)
    
    print("\n" + "="*60)
    print("最终结论")
    print("="*60)
    print(f"有泄露 RMSE: {bad_rmse:.4f} ← 虚假好成绩（不可信）")
    print(f"无泄露 RMSE: {good_rmse:.4f} ← 真实水平")
    print(f"\n差异: {abs(bad_rmse - good_rmse):.4f}")
    print("\n✅ 作业完成！")
    print(f"📁 请检查 {OUTPUT_DIR}/evaluation_comparison.md 查看完整报告")


if __name__ == "__main__":
    main()