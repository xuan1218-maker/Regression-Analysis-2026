import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.models import AnalyticalOLS
from src.utils.diagnostics import calculate_vif

def print_red(text: str):
    """红色输出"""
    print(f"\033[91m{text}\033[0m")

def main():
    # 默认读取清洗后的数据（可改成命令行参数，但作业不要求）
    clean_data_path = "data/clean_marketing.csv"
    if not Path(clean_data_path).exists():
        print(f"❌ 未找到清洗数据: {clean_data_path}")
        print("请先运行: python src/week9/data_prep.py -i homework/week09/data/dirty_marketing.csv -o data/clean_marketing.csv")
        sys.exit(1)

    df = pd.read_csv(clean_data_path)
    print(f"📂 加载清洗数据: {clean_data_path}, 形状: {df.shape}")

    # 分离特征和目标
    target = "Sales"
    if target not in df.columns:
        target = df.columns[-1]
        print(f"⚠️ 未找到 'Sales'，使用最后一列 '{target}' 作为目标")
    X = df.drop(columns=[target]).values.astype(np.float64)
    y = df[target].values.astype(np.float64)
    feature_names = df.drop(columns=[target]).columns.tolist()

    # ---------- 多重共线性诊断 ----------
    print("\n🔬 多重共线性检测 (VIF 阈值 = 10)")
    print("检查 X 是否有 NaN:", np.any(np.isnan(X)))
    print("检查 X 是否有 inf:", np.any(np.isinf(X)))
    print("X 的统计:", np.min(X), np.max(X))
    vifs = calculate_vif(X)
    has_high_vif = False
    for name, vif in zip(feature_names, vifs):
        if np.isinf(vif) or vif > 10:
            print_red(f"⚠️  {name}: VIF = {vif:.2f} > 10 (严重共线性)")
            has_high_vif = True
        elif vif > 5:
            print(f"⚠️  {name}: VIF = {vif:.2f} (中等共线性)")
        else:
            print(f"✅ {name}: VIF = {vif:.2f}")
    if not has_high_vif:
        print("✅ 未检测到严重多重共线性 (所有 VIF ≤ 10)")

    # ---------- 5折交叉验证 ----------
    print("\n📊 5折交叉验证 (AnalyticalOLS)")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # 添加截距列
        X_train_ic = np.column_stack([np.ones(len(X_train)), X_train])
        X_val_ic   = np.column_stack([np.ones(len(X_val)), X_val])

        model = AnalyticalOLS().fit(X_train_ic, y_train)
        y_pred = model.predict(X_val_ic)

        # 防止 NaN 预测值（理论上不会，但保险）
        if np.any(np.isnan(y_pred)):
            print(f"  Fold {fold}: 预测值包含 NaN，跳过")
            continue

        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        print(f"  Fold {fold}: R² = {r2:.4f}, RMSE = {rmse:.2f}")

    if len(r2_scores) > 0:
        mean_r2 = np.mean(r2_scores)
        mean_rmse = np.mean(rmse_scores)
        print(f"\n✅ 平均 R²  = {mean_r2:.4f} ± {np.std(r2_scores):.4f}")
        print(f"✅ 平均 RMSE = {mean_rmse:.2f}")
    else:
        print("❌ 交叉验证失败，所有折均无效")

    # 输出思考题提示
    print("\n💭 思考题：")
    print("   '既然在 data_prep.py 里用全量数据的均值填补了缺失值，")
    print("    那么在 5 折交叉验证时，验证集数据真的算是完全未见过的陌生数据吗？'")
    print("   答案：不是，因为均值计算时用了全量数据（包括验证集），导致数据泄露。")

if __name__ == "__main__":
    main()