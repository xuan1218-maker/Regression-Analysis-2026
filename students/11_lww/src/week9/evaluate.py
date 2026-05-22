from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif

def main():
    # 定义数据路径
    clean_data_path = Path(__file__).parent.parent.parent / "data" / "clean_marketing.csv"

    # ========== 1. 运行数据清洗CLI脚本 ==========
    print("=== 开始数据清洗 ===")
    import subprocess
    subprocess.run([
        sys.executable,
        str(Path(__file__).parent / "data_prep.py"),
        "--input", str(Path(__file__).parent.parent.parent / "data" / "dirty_marketing.csv"),
        "--output", str(clean_data_path)
    ], check=True)

    # 2. 读取清洗后的数据
    df = pd.read_csv(clean_data_path)
    target_col = "Sales"
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()
    # 添加截距项
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # ========== 3. 多重共线性VIF诊断 ==========
    print("\n=== 多重共线性VIF诊断结果 ===")
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    # 计算VIF时跳过截距项
    vif_scores = calculate_vif(X)

    for feat, vif in zip(feature_names, vif_scores):
        if vif > 10:
            # VIF>10用红色字体输出警告
            print(f"\033[91m⚠️  严重警告：特征 [{feat}] 的VIF值为 {vif} > 10，存在严重多重共线性\033[0m")
        else:
            print(f"✅ 特征 [{feat}] 的VIF值为 {vif}")

    # ========== 4. 5折交叉验证评估基线模型 ==========
    print("\n=== 5折交叉验证（AnalyticalOLS） ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_with_intercept), start=1):
        X_train, X_val = X_with_intercept[train_idx], X_with_intercept[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = AnalyticalOLS().fit(X_train, y_train)
        y_pred = model.predict(X_val)
        fold_r2 = r2_score(y_val, y_pred)
        r2_scores.append(fold_r2)
        
        print(f"Fold {fold}: R² = {fold_r2:.4f}")

    avg_r2 = np.mean(r2_scores)
    print(f"\n📊 5折交叉验证平均 R²: {avg_r2:.4f}")

    # ========== 课堂讨论问题提示 ==========
    print("\n💡 课堂思考问题：")
    print("用全量数据的中位数填充缺失值会造成**数据泄露**，验证集提前获取了训练集的统计信息，")
    print("导致交叉验证的R²结果虚高，高估了模型的真实泛化能力。")

if __name__ == "__main__":
    main()