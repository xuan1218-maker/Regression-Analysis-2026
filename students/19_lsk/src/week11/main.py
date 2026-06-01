#!/usr/bin/env python3
"""
Week 11: Dual Inference Sprint — Synthetic-to-Real Regression Workflow
=====================================================================
单一入口：uv run src/week11/main.py

流程:
  Task A: 模拟数据生成 → 清洗 → 预处理 → 自定义模型训练 → CV评估 → VIF诊断 → 推测
  Task B: Kaggle真实数据 → 清洗 → 编码 → 预处理 → 自定义模型训练 → CV评估 → VIF诊断 → 推测
  Task C: 对照总结报告

复用组件: src/utils/{models, metrics, transformers, diagnostics}.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# 确保 src/ 在路径中
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC_DIR)

from utils.models import AnalyticalOLS, GradientDescentOLS
from utils.metrics import calculate_rmse, calculate_mae, calculate_mape
from utils.transformers import CustomStandardScaler
from utils.diagnostics import calculate_vif, print_vif_report

# sklearn 仅用于辅助: KFold, train_test_split, baseline 对比
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler as SklearnScaler  # baseline 用

warnings.filterwarnings("ignore")

# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_regression.csv")
KAGGLE_PATH = os.path.join(DATA_DIR, "kaggle_california_housing.csv")


# ============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        Task A: 模拟数据                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ============================================================================

def generate_synthetic_data(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成带业务含义的模拟回归数据。

    场景: 广告预算与销售额分析
    ============================
    DGP (Data Generating Process):
      sales = β₀ + β₁·tv_budget + β₂·social_budget + β₃·online_budget
              + α_{region} + γ·is_holiday + ε

    其中:
      - tv_budget: 电视广告预算 (万元), 范围 [0, 100]
      - social_budget: 社交媒体预算, 与 tv_budget 高度相关:
                        social_budget = 0.85 * tv_budget + N(0, 10)
      - online_budget: 线上广告预算 (万元), 独立生成, 范围 [0, 50]
      - region: 地区 (North / South / East), 类别变量
      - is_holiday: 是否节假日促销期 (0/1)
      - ε ~ N(0, σ²)

    真实系数:
      β₀ = 50.0 (截距)
      β₁ = 3.2  (tv_budget 正向)
      β₂ = 0.5  (social_budget 正向但效果弱, 且因共线性难识别)
      β₃ = 1.8  (online_budget 正向)
      α_North = 0  (基准)
      α_South = -15 (负向)
      α_East = 8    (正向)
      γ = 20.0 (节假日正向)

    主动注入的问题:
      1. 缺失值: tv_budget 中有 5% 的 NaN
      2. 异常值: social_budget 中有 2% 的极端值 (放大 3 倍)
      3. 共线性: tv_budget 与 social_budget 高度相关 (ρ ≈ 0.95)
      4. 量纲差异: tv_budget [0,100] vs online_budget [0,50] vs binary is_holiday
    """
    rng = np.random.default_rng(seed)

    # --- 生成连续特征 ---
    tv_budget = rng.uniform(0, 100, n_samples)
    # social_budget 与 tv_budget 高度相关
    social_budget = 0.85 * tv_budget + rng.normal(0, 10, n_samples)
    online_budget = rng.uniform(0, 50, n_samples)

    # --- 生成类别变量 ---
    region_cats = rng.choice(["North", "South", "East"], size=n_samples, p=[0.4, 0.3, 0.3])
    is_holiday = rng.choice([0, 1], size=n_samples, p=[0.75, 0.25])

    # --- 构造目标变量 y (显式 DGP) ---
    # 基准噪声
    epsilon = rng.normal(0, 15, n_samples)

    # 系数
    beta_0 = 50.0
    beta_tv = 3.2
    beta_social = 0.5
    beta_online = 1.8
    gamma_holiday = 20.0

    # Region 效应 (One-Hot 形式)
    region_effect = np.zeros(n_samples)
    region_effect[region_cats == "South"] = -15.0
    region_effect[region_cats == "East"] = 8.0

    # 目标变量
    sales = (
        beta_0
        + beta_tv * tv_budget
        + beta_social * social_budget
        + beta_online * online_budget
        + region_effect
        + gamma_holiday * is_holiday
        + epsilon
    )

    # 组装 DataFrame
    df = pd.DataFrame({
        "tv_budget": tv_budget,
        "social_budget": social_budget,
        "online_budget": online_budget,
        "region": region_cats,
        "is_holiday": is_holiday.astype(int),
        "sales": sales,
    })

    # --- 主动注入问题 1: 缺失值 (tv_budget 约 5%) ---
    nan_idx = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[nan_idx, "tv_budget"] = np.nan

    # --- 主动注入问题 2: 异常值 (social_budget 约 2% 极端放大) ---
    outlier_idx = rng.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    df.loc[outlier_idx, "social_budget"] *= 3.0

    return df


def clean_synthetic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗模拟数据:
      1. 缺失值: 用中位数填充 (tv_budget)
      2. 异常值: Winsorization, 用 1% 和 99% 分位数截断 (social_budget)
      3. 类别变量编码: One-Hot Encoding (region)
    """
    df = df.copy()

    # 缺失值: 中位数填充
    for col in ["tv_budget", "social_budget", "online_budget"]:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [缺失值] {col}: 填充了 {df[col].isna().sum()} 个缺失值 (中位数={median_val:.2f})")

    # 异常值: Winsorization (1%-99%)
    for col in ["social_budget"]:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        n_lower = (df[col] < lower).sum()
        n_upper = (df[col] > upper).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"  [异常值] {col}: 下界截断 {n_lower} 个, 上界截断 {n_upper} 个 (分位数 [{lower:.2f}, {upper:.2f}])")

    # 类别变量: One-Hot Encoding (drop first)
    df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=float)

    return df


# ============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        Task B: Kaggle 真实数据                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ============================================================================

def load_kaggle_data() -> pd.DataFrame:
    """
    加载 California Housing 数据集 (来自 Kaggle/sklearn)。
    优先尝试在线加载 sklearn 版本, 失败则回退到本地 housing.csv。

    数据集信息:
      - 样本量: 20,640
      - 特征数: 8 (数值型)
      - 目标变量: MedHouseVal (中位房价, 单位: $100k)
      - 业务含义: 每条样本代表加州一个街区组 (block group) 的房屋信息

    特征 (sklearn 版本):
      MedInc       - 中位收入
      HouseAge     - 房龄中位数
      AveRooms     - 平均房间数
      AveBedrms    - 平均卧室数
      Population   - 人口
      AveOccup     - 平均入住人数
      Latitude     - 纬度
      Longitude    - 经度
    """
    print("\n📥 加载 California Housing 数据集...")

    # 优先尝试在线加载
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        print(f"   ✅ 在线加载成功 (sklearn 版本)")
        print(f"   样本量: {len(df)}")
        print(f"   特征数: {len(data.feature_names)}")
        print(f"   目标变量: MedHouseVal")
        return df
    except Exception as e:
        print(f"   ⚠️ 在线加载失败 ({e})")
        print(f"   🔄 回退到本地文件: {KAGGLE_PATH}")

    # 回退: 从本地 housing.csv 加载
    local_path = os.path.join(DATA_DIR, "housing.csv")
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"本地文件 {local_path} 不存在, 且在线加载失败。"
            f"请手动下载 California Housing 数据集。"
        )

    df_raw = pd.read_csv(local_path)
    print(f"   本地文件行数: {len(df_raw)}")

    # 映射到 sklearn 一致的列名格式
    # 本地 CSV 列: longitude,latitude,housing_median_age,total_rooms,total_bedrooms,
    #              population,households,median_income,median_house_value,ocean_proximity
    # 需要构造: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup,
    #           Latitude, Longitude, MedHouseVal
    df = pd.DataFrame()
    df["MedInc"] = df_raw["median_income"]
    df["HouseAge"] = df_raw["housing_median_age"]
    df["AveRooms"] = df_raw["total_rooms"] / df_raw["households"]
    df["AveBedrms"] = df_raw["total_bedrooms"] / df_raw["households"]
    df["Population"] = df_raw["population"]
    df["AveOccup"] = df_raw["population"] / df_raw["households"]
    df["Latitude"] = df_raw["latitude"]
    df["Longitude"] = df_raw["longitude"]
    df["MedHouseVal"] = df_raw["median_house_value"] / 100000.0  # 转换为 $100k 单位

    print(f"   ✅ 本地加载成功 (已映射为 sklearn 兼容格式)")
    print(f"   样本量: {len(df)}")
    print(f"   特征数: 8")
    print(f"   目标变量: MedHouseVal (单位: $100k)")

    return df


def clean_kaggle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗 Kaggle 数据:
      1. 创建类别变量 (基于 Latitude 分区) 以满足"至少一个非数值特征"要求
      2. 缺失值检查与处理
      3. 异常值处理 (Winsorization)
      4. One-Hot 编码
    """
    df = df.copy()

    # --- 创建类别变量: 基于纬度划分区域 ---
    # 加州大致分为: 南加州 (Lat < 35.0), 中加州 (35.0 <= Lat < 37.0), 北加州 (Lat >= 37.0)
    conditions = [
        df["Latitude"] < 35.0,
        (df["Latitude"] >= 35.0) & (df["Latitude"] < 37.0),
        df["Latitude"] >= 37.0,
    ]
    choices = ["SouthCA", "CentralCA", "NorthCA"]
    df["region_ca"] = np.select(conditions, choices, default="CentralCA")

    # --- 缺失值检查 ---
    missing_report = df.isna().sum()
    missing_cols = missing_report[missing_report > 0]
    if len(missing_cols) > 0:
        print(f"  [缺失值] 发现缺失列: {dict(missing_cols)}")
        for col in missing_cols.index:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print("  [缺失值] 无缺失值, 跳过处理")

    # --- 异常值处理: Winsorization 对连续变量 ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除目标变量和经纬度
    outlier_cols = [c for c in numeric_cols if c not in ["MedHouseVal", "Latitude", "Longitude"]]
    for col in outlier_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower, upper)
            print(f"  [异常值] {col}: 截断 {n_outliers} 个异常值 (分位数 [{lower:.4f}, {upper:.4f}])")

    # --- 类别变量编码 ---
    df = pd.get_dummies(df, columns=["region_ca"], drop_first=True, dtype=float)

    return df


# ============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        通用 Pipeline 函数                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ============================================================================

def prepare_features_target(df: pd.DataFrame, target_col: str) -> tuple:
    """分离特征和目标变量, 返回 X (DataFrame) 和 y (Series)"""
    feature_df = df.drop(columns=[target_col])
    target = df[target_col].values
    return feature_df, target


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    无泄露 5 折交叉验证。

    每一折:
      1. 用训练集 fit scaler
      2. transform 训练集和验证集 (用训练集的 mean/std)
      3. 训练模型
      4. 预测并计算指标

    返回: {"rmse": [...], "mae": [...], "mape": [...], "r2": [...]}
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    metrics = {"rmse": [], "mae": [], "mape": [], "r2": []}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- 无泄露预处理: fit on train, transform both ---
        scaler = CustomStandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        # 添加截距列
        X_train_design = np.column_stack([np.ones(X_train.shape[0]), X_train])
        X_val_design = np.column_stack([np.ones(X_val.shape[0]), X_val])

        # --- 训练模型 ---
        model = model_class(**model_kwargs)
        model.fit(X_train_design, y_train)

        # --- 预测与评估 ---
        y_pred = model.predict(X_val_design)

        metrics["rmse"].append(calculate_rmse(y_val, y_pred))
        metrics["mae"].append(calculate_mae(y_val, y_pred))
        metrics["mape"].append(calculate_mape(y_val, y_pred))

        # R² 计算
        sse = np.sum((y_val - y_pred) ** 2)
        sst = np.sum((y_val - np.mean(y_val)) ** 2)
        r2 = 1 - sse / sst if sst != 0 else 0.0
        metrics["r2"].append(r2)

    return metrics


def run_sklearn_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    sklearn baseline: LinearRegression + StandardScaler, 同样无泄露 CV
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    metrics = {"rmse": [], "mae": [], "mape": [], "r2": []}

    for train_idx, val_idx in kf.split(X):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = SklearnScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics["rmse"].append(calculate_rmse(y_val, y_pred))
        metrics["mae"].append(calculate_mae(y_val, y_pred))
        metrics["mape"].append(calculate_mape(y_val, y_pred))

        sse = np.sum((y_val - y_pred) ** 2)
        sst = np.sum((y_val - np.mean(y_val)) ** 2)
        metrics["r2"].append(1 - sse / sst if sst != 0 else 0.0)

    return metrics


def fmt_metric(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


# ============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         报告生成                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ============================================================================

def generate_synthetic_report(
    df_clean: pd.DataFrame,
    feature_names: list,
    target_name: str,
    cv_metrics: dict,
    coef_names: list,
    coefficients: np.ndarray,
    vif_values: list,
) -> str:
    """生成 Task A 模拟数据报告"""

    lines = []
    lines.append("# 📊 Task A: Synthetic Regression Report")
    lines.append("")
    lines.append("## 1. 数据生成机制 (DGP)")
    lines.append("")
    lines.append("### 场景: 广告预算与销售额回归分析")
    lines.append("")
    lines.append("目标变量 `sales` (销售额, 万元) 由以下公式生成:")
    lines.append("")
    lines.append("```")
    lines.append("sales = 50.0 + 3.2 × tv_budget + 0.5 × social_budget + 1.8 × online_budget")
    lines.append("        + α_region + 20.0 × is_holiday + ε")
    lines.append("```")
    lines.append("")
    lines.append("其中 `ε ~ N(0, 15²)`")
    lines.append("")
    lines.append("### 变量含义与预期方向:")
    lines.append("")
    lines.append("| 变量 | 类型 | 真实系数 | 预期方向 | 说明 |")
    lines.append("|------|------|----------|----------|------|")
    lines.append("| tv_budget | 连续 | +3.2 | 正向 ↑ | 电视广告预算 (万元) |")
    lines.append("| social_budget | 连续 | +0.5 | 正向 ↑ | 社交媒体预算, 与 tv_budget 高度相关 |")
    lines.append("| online_budget | 连续 | +1.8 | 正向 ↑ | 线上广告预算 (万元) |")
    lines.append("| region_South | 类别 | -15.0 | 负向 ↓ | 南部地区相对北部基准 |")
    lines.append("| region_East | 类别 | +8.0 | 正向 ↑ | 东部地区相对北部基准 |")
    lines.append("| is_holiday | 类别 | +20.0 | 正向 ↑ | 节假日促销效应 |")
    lines.append("")
    lines.append("### 构造的高度相关特征:")
    lines.append("")
    lines.append("`social_budget = 0.85 × tv_budget + N(0, 10²)`, 相关系数约 0.95。")
    lines.append("这种设计使得 `tv_budget` 和 `social_budget` 在统计上难以分离各自贡献,")
    lines.append("这正是我们在真实数据中经常遇到的多重共线性问题。")
    lines.append("")

    lines.append("### 注入的数据问题:")
    lines.append("")
    lines.append("| 问题类型 | 变量 | 详情 |")
    lines.append("|----------|------|------|")
    lines.append("| 缺失值 | tv_budget | 5% 的值为 NaN |")
    lines.append("| 异常值 | social_budget | 2% 的值被放大 3 倍 |")
    lines.append("| 共线性 | tv_budget ↔ social_budget | r ≈ 0.95 |")
    lines.append("| 量纲差异 | tv_budget [0,100] vs online_budget [0,50] vs is_holiday {0,1} | 需标准化 |")
    lines.append("")

    lines.append("## 2. 数据清洗")
    lines.append("")
    lines.append("- **缺失值**: 用中位数填充 `tv_budget` 的缺失值")
    lines.append("- **异常值**: 对 `social_budget` 使用 1%-99% Winsorization 截断")
    lines.append("- **编码**: `region` 类别变量使用 One-Hot 编码 (drop first)")
    lines.append("")

    lines.append("## 3. 交叉验证评估 (5 折, 无泄露)")
    lines.append("")
    lines.append("使用自定义 `GradientDescentOLS` 模型 + `CustomStandardScaler`。")
    lines.append("每一折: fit scaler 在训练集 → transform 训练集 & 验证集 → 训练模型 → 评估。")
    lines.append("")
    lines.append("| 指标 | 均值 ± 标准差 |")
    lines.append("|------|---------------|")
    for metric_name in ["rmse", "mae", "mape", "r2"]:
        vals = cv_metrics[metric_name]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        label = {"rmse": "RMSE", "mae": "MAE", "mape": "MAPE (%)", "r2": "R²"}[metric_name]
        lines.append(f"| {label} | {fmt_metric(mean_val, std_val)} |")
    lines.append("")

    lines.append("## 4. 模型系数与推测")
    lines.append("")
    lines.append("使用全量数据训练 `AnalyticalOLS` 得到的系数:")
    lines.append("")
    lines.append("| 变量 | 估计系数 | 真实系数 | 是否一致 |")
    lines.append("|------|----------|----------|----------|")
    true_coefs = {
        "const": 50.0, "tv_budget": 3.2, "social_budget": 0.5,
        "online_budget": 1.8, "region_South": -15.0,
        "region_East": 8.0, "is_holiday": 20.0,
    }
    for name, coef in zip(coef_names, coefficients):
        true_val = true_coefs.get(name, None)
        match = "✅" if true_val is not None and abs(coef - true_val) < 3 * np.std(cv_metrics["rmse"]) else "⚠️"
        if true_val is not None:
            lines.append(f"| {name} | {coef:.4f} | {true_val:.1f} | {match} |")
        else:
            lines.append(f"| {name} | {coef:.4f} | — | — |")
    lines.append("")

    lines.append("### 推测分析:")
    lines.append("")
    lines.append("1. **变量方向一致性**: 从系数符号看, 模型估计的方向与 DGP 大致一致。")
    lines.append("   但 `social_budget` 由于与 `tv_budget` 高度共线, 其系数估计可能偏离真实值。")
    lines.append("2. **共线性的影响**: `tv_budget` 和 `social_budget` 的相关系数约 0.95,")
    lines.append("   导致 VIF 极高, 两个变量的系数标准误会增大, 各自效应难以稳定识别。")
    lines.append("3. **难以稳定识别的变量**: `social_budget` 本身贡献较小 (真实系数仅 0.5),")
    lines.append("   又与 `tv_budget` 高度共线, 因此在噪声干扰下最容易被误估甚至方向反转。")
    lines.append("")

    lines.append("## 5. 多重共线性诊断 (VIF)")
    lines.append("")
    lines.append("| 变量 | VIF 值 | 严重程度 |")
    lines.append("|------|--------|----------|")
    for name, vif_val in zip(coef_names[1:], vif_values):  # 跳过截距
        if vif_val < 5:
            sev = "✅ 正常"
        elif vif_val < 10:
            sev = "⚠️ 中等"
        else:
            sev = "❌ 严重!"
        lines.append(f"| {name} | {vif_val:.2f} | {sev} |")
    lines.append("")
    lines.append("> ⚠️ `tv_budget` 和 `social_budget` 预期会显示严重共线性 (VIF >> 10),")
    lines.append("> 这正是我们设计 DGP 时有意构造的。")
    lines.append("")

    return "\n".join(lines)


def generate_kaggle_report(
    df_clean: pd.DataFrame,
    feature_names: list,
    target_name: str,
    cv_metrics_gd: dict,
    cv_metrics_baseline: dict,
    coef_names: list,
    coefficients: np.ndarray,
    vif_values: list,
) -> str:
    """生成 Task B Kaggle 数据报告"""

    lines = []
    lines.append("# 🌍 Task B: Kaggle Real-World Regression Report")
    lines.append("")
    lines.append("## 1. 数据集信息")
    lines.append("")
    lines.append("- **数据集名称**: California Housing")
    lines.append("- **来源**: sklearn.datasets.fetch_california_housing (源自 1990 年美国人口普查)")
    lines.append("- **样本量**: 20,640")
    lines.append("- **特征数**: 8 (数值型)")
    lines.append("- **预测目标**: MedHouseVal — 加州街区组中位房价 (单位: $100,000)")
    lines.append("")
    lines.append("### 业务含义:")
    lines.append("每条样本代表加州一个街区组 (block group) 的房屋信息。")
    lines.append("目标是通过收入、房龄、地理位置等特征预测该区域的中位房价。")
    lines.append("")
    lines.append("### 为什么选择这个数据集:")
    lines.append("1. 它是真实人口普查数据, 不是\"演示型\"教学数据, 存在真实的数据噪声和分布特征;")
    lines.append("2. 包含多种数值型特征, 且经纬度可以构造类别变量;")
    lines.append("3. 样本量足够大 (20,640), 适合交叉验证;")
    lines.append("4. 房价预测是一个经典的回归问题, 业务上易于理解。")
    lines.append("")

    lines.append("## 2. 数据清洗与预处理")
    lines.append("")
    lines.append("- **类别变量构造**: 基于 `Latitude` 将加州划分为三个区域:")
    lines.append("  - `SouthCA` (Lat < 35.0)")
    lines.append("  - `CentralCA` (35.0 ≤ Lat < 37.0)")
    lines.append("  - `NorthCA` (Lat ≥ 37.0)")
    lines.append("- **缺失值**: California Housing 数据集本身无缺失值, 但流程中仍进行了检查")
    lines.append("- **异常值**: 对所有连续数值特征 (除目标变量和经纬度) 使用 1%-99% Winsorization")
    lines.append("- **编码**: `region_ca` 使用 One-Hot 编码 (drop first)")
    lines.append("- **标准化**: 使用自定义 `CustomStandardScaler`")
    lines.append("")

    lines.append("## 3. 模型评估 (5 折无泄露 CV)")
    lines.append("")
    lines.append("### 3.1 自定义 GradientDescentOLS")
    lines.append("")
    lines.append("| 指标 | 均值 ± 标准差 |")
    lines.append("|------|---------------|")
    for metric_name in ["rmse", "mae", "mape", "r2"]:
        vals = cv_metrics_gd[metric_name]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        label = {"rmse": "RMSE", "mae": "MAE", "mape": "MAPE (%)", "r2": "R²"}[metric_name]
        lines.append(f"| {label} | {fmt_metric(mean_val, std_val)} |")
    lines.append("")

    lines.append("### 3.2 sklearn LinearRegression Baseline")
    lines.append("")
    lines.append("| 指标 | 均值 ± 标准差 |")
    lines.append("|------|---------------|")
    for metric_name in ["rmse", "mae", "mape", "r2"]:
        vals = cv_metrics_baseline[metric_name]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        label = {"rmse": "RMSE", "mae": "MAE", "mape": "MAPE (%)", "r2": "R²"}[metric_name]
        lines.append(f"| {label} | {fmt_metric(mean_val, std_val)} |")
    lines.append("")

    lines.append("### 3.3 模型对比")
    lines.append("")
    lines.append("自定义 `GradientDescentOLS` 与 `sklearn.LinearRegression` 理论上应得到相似的系数")
    lines.append("(在收敛良好的情况下)。差异主要来自梯度下降的收敛精度 vs 解析解。")
    lines.append("")

    lines.append("## 4. 模型系数")
    lines.append("")
    lines.append("使用全量数据训练 `AnalyticalOLS` 得到的系数:")
    lines.append("")
    lines.append("| 变量 | 系数 | 解释 |")
    lines.append("|------|------|------|")
    for name, coef in zip(coef_names, coefficients):
        direction = "↑ 正向" if coef > 0 else "↓ 负向"
        lines.append(f"| {name} | {coef:.6f} | {direction} |")
    lines.append("")

    lines.append("## 5. 多重共线性诊断 (VIF)")
    lines.append("")
    lines.append("| 变量 | VIF 值 | 严重程度 |")
    lines.append("|------|--------|----------|")
    has_severe = False
    for name, vif_val in zip(coef_names[1:], vif_values):
        if vif_val < 5:
            sev = "✅ 正常"
        elif vif_val < 10:
            sev = "⚠️ 中等"
        else:
            sev = "❌ 严重!"
            has_severe = True
        lines.append(f"| {name} | {vif_val:.2f} | {sev} |")
    lines.append("")

    if has_severe:
        lines.append("> ⚠️ 检测到部分变量存在共线性问题, 可能影响系数稳定性。")
    lines.append("")

    lines.append("## 6. 推测分析")
    lines.append("")
    lines.append("### 6.1 最稳定的变量")
    lines.append("`MedInc` (中位收入) 通常与房价关系最稳定且正向最强,")
    lines.append("这与经济学直觉 (收入越高, 房价越高) 一致。")
    lines.append("")
    lines.append("### 6.2 不稳定的变量")
    lines.append("`AveRooms` 和 `AveBedrms` 可能由于尺度或相关性问题而不稳定。")
    lines.append("`Population` 和 `AveOccup` 的效应方向在实际中可能依赖于具体区域。")
    lines.append("")
    lines.append("### 6.3 共线性问题")
    lines.append("房间数和卧室数 (`AveRooms` ↔ `AveBedrms`) 往往相关, 可能导致 VIF 偏高。")
    lines.append("")
    lines.append("### 6.4 业务解释")
    lines.append(f"模型的平均 RMSE 约为 ${np.mean(cv_metrics_gd['rmse']):.4f} (单位: $100k),")
    lines.append(f"即预测误差约 ${np.mean(cv_metrics_gd['rmse']) * 100000:.0f} 美元。")
    lines.append("在业务上这意味着: 对一栋中位房价的房屋, 模型预测误差在可接受范围内。")
    lines.append("")
    lines.append("### 6.5 上线风险")
    lines.append("1. **地理漂移**: 数据集是 1990 年的, 当前加州房价结构已发生显著变化;")
    lines.append("2. **分布外推**: 对极端收入区域 (极富/极贫) 预测可能不准;")
    lines.append("3. **遗漏变量**: 利率、学区质量等重要因素未在数据中;")
    lines.append("4. **共线性**: 部分变量相关性可能导致政策解读出错。")
    lines.append("")

    return "\n".join(lines)


def generate_summary_comparison() -> str:
    """生成 Task C 对照总结"""

    lines = []
    lines.append("# 🧠 Task C: Synthetic vs Real — Summary Comparison")
    lines.append("")
    lines.append("## 1. 为什么模拟数据中\"推测\"相对容易?")
    lines.append("")
    lines.append("在模拟数据中, 我们**明确知道 DGP (Data Generating Process)**——")
    lines.append("包括每个变量的真实系数、噪声分布、共线性结构和异常值位置。")
    lines.append("这使得我们可以:")
    lines.append("- 直接比较估计系数与真实系数;")
    lines.append("- 精确判断模型偏差来自哪里 (噪声? 共线性? 异常值?);")
    lines.append("- 主动设计实验: \"如果我的清洗流程正确, 参数应该恢复到什么程度?\"")
    lines.append("")
    lines.append("简而言之, 模拟数据给了我们一个\"标准答案\", 推测=对答案。")
    lines.append("")

    lines.append("## 2. 为什么真实数据中即使分数还可以, 解释也更困难?")
    lines.append("")
    lines.append("真实数据 (如 California Housing) 中:")
    lines.append("- **没有标准答案**: 我们不知道\真正的\系数是多少;")
    lines.append("- **遗漏变量偏差 (OVB)**: 重要变量 (利率、学区、政策) 可能完全缺失;")
    lines.append("- **反向因果**: 高房价可能反过来吸引高收入人群, 系数不是单向的;")
    lines.append("- **测量误差**: 人口普查数据本身有抽样和汇总误差;")
    lines.append("- **混杂因素**: 多个变量同时影响 X 和 Y, 而我们无法完全控制。")
    lines.append("")
    lines.append("所以, 即使 R² = 0.6 (看起来不错), 我们也不能轻易说\"X 导致 Y\",")
    lines.append("只能谨慎地说\"在控制其他变量后, X 与 Y 存在统计关联\"。")
    lines.append("")

    lines.append("## 3. 共线性、缺失值、异常值在两类数据上的影响")
    lines.append("")
    lines.append("| 问题 | 模拟数据 | 真实数据 |")
    lines.append("|------|----------|----------|")
    lines.append("| **共线性** | 我们主动构造, 知道哪两个变量相关。清理时可以从容处理。 | 共线性是\"意外发现\", 可能需要多次探索才知道哪些变量相关。 |")
    lines.append("| **缺失值** | 我们控制缺失机制 (MCAR), 填充效果好。 | 缺失机制未知 (可能是 MAR 或 MNAR), 简单填充可能引入偏差。 |")
    lines.append("| **异常值** | 我们知道哪些是异常值 (因为我们注入的), 可以精确评估处理效果。 | 不知道\"正常\"边界在哪, 可能误将重要信号当作异常值删除。 |")
    lines.append("")

    lines.append("## 4. 为什么\"无泄露交叉验证\"在真实数据上尤其重要?")
    lines.append("")
    lines.append("真实数据中, 无泄露 CV 是**防止自欺欺人**的最后防线:")
    lines.append("")
    lines.append("1. **预处理泄露**: 如果用全量数据 fit scaler, 验证集信息\"泄露\"进了训练过程,")
    lines.append("   CV 分数会被高估, 但上线后表现会差很多。")
    lines.append("2. **缺失值填充泄露**: 用全量数据的中位数填充 → 训练集\"看到了\"验证集的信息。")
    lines.append("3. **特征选择泄露**: 基于全量数据选择特征 → 选中的特征在 CV 中看起来总是好的。")
    lines.append("")
    lines.append("模拟数据中我们至少知道\"真实模型\"是什么, 可以交叉验证。")
    lines.append("真实数据中, CV 分数往往是我们唯一可信的性能指标。")
    lines.append("如果 CV 有泄露, 整个评估体系就崩塌了。")
    lines.append("")

    lines.append("## 5. 自己维护的 `utils/` 组件帮助省下了哪些重复劳动?")
    lines.append("")
    lines.append("这周 `utils/` 组件让我省下了以下重复劳动:")
    lines.append("")
    lines.append("| 组件 | 省下的工作 |")
    lines.append("|------|------------|")
    lines.append("| `models.py` | 不需要每次手写 OLS 正规方程或梯度下降; AnalyticalOLS 自带 summary 和 F-test |")
    lines.append("| `metrics.py` | RMSE/MAE/MAPE 一行调用, 不用每次写 numpy 公式 |")
    lines.append("| `transformers.py` | Scaler/Imputer 有统一的 fit/transform 接口, 在 CV 循环中无泄露使用 |")
    lines.append("| `diagnostics.py` | VIF 计算 + 格式化报告一键生成, 不用手动做回归 |")
    lines.append("")
    lines.append("核心价值: **接口一致性**。所有组件遵循 `fit()` → `transform()` / `predict()` 模式,")
    lines.append("使得在交叉验证中无缝切换预处理和模型步骤, 避免因接口不一致导致的泄露风险。")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            主入口                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# ============================================================================

def main():
    print("=" * 70)
    print("  Week 11: Dual Inference Sprint — Synthetic-to-Real Workflow")
    print("=" * 70)

    # ========================================================================
    # TASK A: 模拟数据
    # ========================================================================
    print("\n" + "─" * 70)
    print("  Task A: 模拟数据 — 广告预算与销售额回归分析")
    print("─" * 70)

    # A1. 生成模拟数据
    print("\n📊 生成模拟数据 (n=500, 4 features + 1 类别变量)...")
    df_synthetic = generate_synthetic_data(n_samples=500, seed=42)
    df_synthetic.to_csv(SYNTHETIC_PATH, index=False)
    print(f"✅ 模拟数据已保存到: {SYNTHETIC_PATH}")
    print(f"   样本量: {len(df_synthetic)}, 特征数: {len(df_synthetic.columns)-1}")
    print(f"   缺失值统计: tv_budget={df_synthetic['tv_budget'].isna().sum()}")
    print(f"   相关系数(tv_budget, social_budget): {df_synthetic[['tv_budget','social_budget']].corr().iloc[0,1]:.4f}")

    # A2. 清洗模拟数据
    print("\n🧹 清洗模拟数据...")
    df_synthetic_clean = clean_synthetic_data(df_synthetic)

    # A3. 准备数据
    synthetic_feature_df, synthetic_target = prepare_features_target(df_synthetic_clean, "sales")
    synthetic_feature_names = list(synthetic_feature_df.columns)
    synthetic_X = synthetic_feature_df.values.astype(np.float64)

    print(f"\n   清洗后特征 ({len(synthetic_feature_names)}): {synthetic_feature_names}")

    # A4. 无泄露 5 折交叉验证 (GradientDescentOLS)
    print("\n🔄 5 折无泄露交叉验证 (GradientDescentOLS)...")
    cv_synthetic = run_cross_validation(
        synthetic_X, synthetic_target,
        model_class=GradientDescentOLS,
        model_kwargs={"learning_rate": 0.01, "max_iter": 2000, "gd_type": "full_batch"},
        n_folds=5,
    )

    print(f"   RMSE: {fmt_metric(np.mean(cv_synthetic['rmse']), np.std(cv_synthetic['rmse']))}")
    print(f"   MAE:  {fmt_metric(np.mean(cv_synthetic['mae']), np.std(cv_synthetic['mae']))}")
    print(f"   MAPE: {fmt_metric(np.mean(cv_synthetic['mape']), np.std(cv_synthetic['mape']))}")
    print(f"   R²:   {fmt_metric(np.mean(cv_synthetic['r2']), np.std(cv_synthetic['r2']))}")

    # A5. 全量数据训练 AnalyticalOLS (用于解读系数)
    print("\n📈 全量数据训练 AnalyticalOLS (用于系数解读)...")
    # 标准化 + 截距
    scaler_syn = CustomStandardScaler()
    X_syn_scaled = scaler_syn.fit_transform(synthetic_X)
    X_syn_design = np.column_stack([np.ones(X_syn_scaled.shape[0]), X_syn_scaled])
    syn_coef_names = ["const"] + synthetic_feature_names

    ols_syn = AnalyticalOLS()
    ols_syn.fit(X_syn_design, synthetic_target, feature_names=syn_coef_names)
    print(ols_syn.summary())

    # A6. VIF 诊断
    print("\n🔍 VIF 诊断...")
    vif_syn = calculate_vif(X_syn_scaled)
    print_vif_report(synthetic_feature_names, vif_syn)

    # A7. 生成报告
    synthetic_report = generate_synthetic_report(
        df_synthetic_clean, synthetic_feature_names, "sales",
        cv_synthetic,
        syn_coef_names, ols_syn.coef_,
        vif_syn,
    )
    report_path = os.path.join(RESULTS_DIR, "synthetic_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(synthetic_report)
    print(f"\n✅ 模拟数据报告已保存到: {report_path}")

    # ========================================================================
    # TASK B: Kaggle 真实数据
    # ========================================================================
    print("\n\n" + "─" * 70)
    print("  Task B: Kaggle 真实数据 — California Housing")
    print("─" * 70)

    # B1. 加载数据
    df_kaggle = load_kaggle_data()
    df_kaggle.to_csv(KAGGLE_PATH, index=False)
    print(f"✅ Kaggle 数据已保存到: {KAGGLE_PATH}")

    # B2. 清洗数据
    print("\n🧹 清洗 Kaggle 数据...")
    df_kaggle_clean = clean_kaggle_data(df_kaggle)

    # B3. 准备数据
    kaggle_feature_df, kaggle_target = prepare_features_target(df_kaggle_clean, "MedHouseVal")
    kaggle_feature_names = list(kaggle_feature_df.columns)
    kaggle_X = kaggle_feature_df.values.astype(np.float64)

    print(f"\n   清洗后特征 ({len(kaggle_feature_names)}): {kaggle_feature_names}")

    # B4. 无泄露 5 折交叉验证 (GradientDescentOLS)
    print("\n🔄 5 折无泄露交叉验证 (自定义 GradientDescentOLS)...")
    cv_kaggle_gd = run_cross_validation(
        kaggle_X, kaggle_target,
        model_class=GradientDescentOLS,
        model_kwargs={"learning_rate": 0.01, "max_iter": 3000, "gd_type": "full_batch"},
        n_folds=5,
    )

    print(f"   RMSE: {fmt_metric(np.mean(cv_kaggle_gd['rmse']), np.std(cv_kaggle_gd['rmse']))}")
    print(f"   MAE:  {fmt_metric(np.mean(cv_kaggle_gd['mae']), np.std(cv_kaggle_gd['mae']))}")
    print(f"   MAPE: {fmt_metric(np.mean(cv_kaggle_gd['mape']), np.std(cv_kaggle_gd['mape']))}")
    print(f"   R²:   {fmt_metric(np.mean(cv_kaggle_gd['r2']), np.std(cv_kaggle_gd['r2']))}")

    # B5. sklearn baseline 对比
    print("\n📊 sklearn LinearRegression Baseline (5 折无泄露 CV)...")
    cv_kaggle_sk = run_sklearn_baseline(kaggle_X, kaggle_target, n_folds=5)

    print(f"   RMSE: {fmt_metric(np.mean(cv_kaggle_sk['rmse']), np.std(cv_kaggle_sk['rmse']))}")
    print(f"   MAE:  {fmt_metric(np.mean(cv_kaggle_sk['mae']), np.std(cv_kaggle_sk['mae']))}")
    print(f"   MAPE: {fmt_metric(np.mean(cv_kaggle_sk['mape']), np.std(cv_kaggle_sk['mape']))}")
    print(f"   R²:   {fmt_metric(np.mean(cv_kaggle_sk['r2']), np.std(cv_kaggle_sk['r2']))}")

    # B6. 全量数据训练 AnalyticalOLS (系数解读)
    print("\n📈 全量数据训练 AnalyticalOLS (用于系数解读)...")
    scaler_kg = CustomStandardScaler()
    X_kg_scaled = scaler_kg.fit_transform(kaggle_X)
    X_kg_design = np.column_stack([np.ones(X_kg_scaled.shape[0]), X_kg_scaled])
    kg_coef_names = ["const"] + kaggle_feature_names

    ols_kg = AnalyticalOLS()
    ols_kg.fit(X_kg_design, kaggle_target, feature_names=kg_coef_names)
    print(ols_kg.summary())

    # B7. VIF 诊断
    print("\n🔍 VIF 诊断...")
    vif_kg = calculate_vif(X_kg_scaled)
    print_vif_report(kaggle_feature_names, vif_kg)

    # B8. 生成报告
    kaggle_report = generate_kaggle_report(
        df_kaggle_clean, kaggle_feature_names, "MedHouseVal",
        cv_kaggle_gd, cv_kaggle_sk,
        kg_coef_names, ols_kg.coef_,
        vif_kg,
    )
    report_path = os.path.join(RESULTS_DIR, "kaggle_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(kaggle_report)
    print(f"\n✅ Kaggle 报告已保存到: {report_path}")

    # ========================================================================
    # TASK C: 对照总结
    # ========================================================================
    print("\n\n" + "─" * 70)
    print("  Task C: 对照总结")
    print("─" * 70)

    summary_report = generate_summary_comparison()
    summary_path = os.path.join(RESULTS_DIR, "summary_comparison.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_report)
    print(f"\n✅ 对照总结已保存到: {summary_path}")

    # ========================================================================
    # 完成
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("  ✅ Week 11 全部任务完成!")
    print("=" * 70)
    print(f"  生成文件:")
    print(f"    {SYNTHETIC_PATH}")
    print(f"    {KAGGLE_PATH}")
    print(f"    {os.path.join(RESULTS_DIR, 'synthetic_report.md')}")
    print(f"    {os.path.join(RESULTS_DIR, 'kaggle_report.md')}")
    print(f"    {os.path.join(RESULTS_DIR, 'summary_comparison.md')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
