"""
模块：utils.diagnostics
用途：模型诊断工具箱，包含多重共线性检测（VIF）、条件数、系数稳定性分析等函数
"""

import numpy as np
import pandas as pd
from utils.models import AnalyticalOLS


def calculate_vif(X: np.ndarray) -> list:
    """
    计算每个特征的方差膨胀因子（VIF）
    
    原理：VIF_j = 1 / (1 - R_j²)
    其中 R_j² 是将第 j 个特征作为因变量，对其他所有特征做回归得到的决定系数。
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵（应该已经包含截距列，或者不包含都可以，VIF只对特征计算）
        注意：VIF 计算时不应包含截距列（全1列），否则会严重膨胀
        
    Returns
    -------
    vif_values : list
        每个特征的 VIF 值，顺序与 X 的列顺序一致
    """
    # ========== 关键修复：确保 X 是数值类型 ==========
    # 如果 X 是 DataFrame，先转换为 numpy 数组
    if hasattr(X, 'values'):
        X = X.values
    
    # 强制转换为 float64
    try:
        X = X.astype(np.float64)
    except (ValueError, TypeError) as e:
        print(f"\n❌ 错误：特征矩阵包含无法转换为数值的列")
        print(f"   请确保数据已完成 One-Hot 编码且没有字符串列")
        print(f"   当前数据类型: {X.dtype if hasattr(X, 'dtype') else 'unknown'}")
        raise e
    
    n_samples, n_features = X.shape
    
    # 检查是否有缺失值
    if np.any(np.isnan(X)):
        print("\n⚠️ 警告：特征矩阵存在缺失值，VIF 计算可能不准确")
        # 简单处理：删除包含 NaN 的行（仅用于 VIF 计算）
        nan_rows = np.any(np.isnan(X), axis=1)
        X = X[~nan_rows]
        print(f"   删除了 {np.sum(nan_rows)} 行包含缺失值的数据")
        n_samples, n_features = X.shape
    
    if n_features < 2:
        return [1.0] * n_features
    
    vif_values = []
    
    print(f"\n计算 VIF 中... (共 {n_features} 个特征)")
    
    for i in range(n_features):
        y_col = X[:, i]
        X_others = np.delete(X, i, axis=1)
        
        # 为自变量添加截距列
        X_with_const = np.column_stack([np.ones(X_others.shape[0]), X_others])
        
        try:
            # 使用解析解 OLS 进行回归
            model = AnalyticalOLS()
            model.fit(X_with_const, y_col)
            
            # 计算 R²
            y_pred = model.predict(X_with_const)
            sse = np.sum((y_col - y_pred) ** 2)
            sst = np.sum((y_col - np.mean(y_col)) ** 2)
            r_squared = 1 - sse / sst if sst != 0 else 0.0
            
            # 计算 VIF = 1 / (1 - R²)
            if r_squared >= 0.999:
                vif = float('inf')
            else:
                vif = 1.0 / (1.0 - r_squared)
            
            vif_values.append(vif)
            
        except np.linalg.LinAlgError:
            vif_values.append(float('inf'))
            print(f"   ⚠️ 警告：计算第 {i} 个特征的 VIF 时矩阵奇异")
    
    return vif_values


def print_vif_report(feature_names: list, vif_values: list) -> None:
    """
    打印格式化的 VIF 报告，高亮显示严重共线性
    """
    print("\n" + "=" * 70)
    print("多重共线性诊断报告 (VIF - Variance Inflation Factor)")
    print("=" * 70)
    print(f"{'特征名称':<25} {'VIF值':>12} {'严重程度':>20}")
    print("-" * 70)
    
    has_severe = False
    
    for name, vif in zip(feature_names, vif_values):
        if vif < 5:
            severity = "✅ 正常"
            print(f"{name:<25} {vif:>12.2f} {severity:>20}")
        elif vif < 10:
            severity = "⚠️ 中等"
            print(f"{name:<25} {vif:>12.2f} {severity:>20}")
        else:
            severity = "❌ 严重!"
            has_severe = True
            # 红色字体
            print(f"\033[91m{name:<25} {vif:>12.2f} {severity:>20}\033[0m")
    
    print("=" * 70)
    
    if has_severe:
        print("\n\033[91m⚠️ 警告：检测到严重多重共线性 (VIF > 10)!")
        print("   建议：考虑删除高度相关的特征，或使用岭回归(Ridge Regression)等正则化方法。\033[0m")
    else:
        print("\n✅ 未检测到严重多重共线性 (所有 VIF < 10)")
    
    print("=" * 70)


def calculate_vif_dataframe(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    针对 DataFrame 计算 VIF，返回结果 DataFrame
    """
    X = df[feature_cols].values
    vif_values = calculate_vif(X)

    vif_df = pd.DataFrame({
        '特征': feature_cols,
        'VIF': vif_values
    }).sort_values('VIF', ascending=False)

    return vif_df


def diagnose_vif_from_dataframe(df: pd.DataFrame, target_col: str,
                                cat_cols: list,
                                preprocess_fn=None) -> tuple:
    """
    从 DataFrame 出发，完成预处理后计算 VIF。

    Parameters
    ----------
    df : pd.DataFrame
        包含目标列的完整数据
    target_col : str
        目标变量列名（会被排除）
    cat_cols : list
        类别特征列名列表（用于 One-Hot 编码）
    preprocess_fn : callable, optional
        预处理函数，签名同 transformers.py 中的 preprocess_features。
        如果提供，调用 preprocess_fn(df, target_col, cat_cols, fit_mode=True)
        返回 (X, y, feature_names, stats_dict)，其中 X 为标准化后的数据。
        如果不提供，使用内置的简单预处理（仅填补+编码，不做标准化）。

    Returns
    -------
    (feature_names, vif_values) : tuple
        特征名称列表和对应的 VIF 值列表
    """
    if preprocess_fn is not None:
        # 使用外部预处理函数（但不做标准化，VIF 是尺度无关的）
        # 调用方需传入一个只做填补+编码的版本
        X, _, feature_names, _ = preprocess_fn(
            df, target_col, cat_cols, fit_mode=True)
    else:
        # 内置简单预处理
        df = df.copy()
        if target_col in df.columns:
            df = df.drop(columns=[target_col])

        # 布尔/整数列转 float
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(float)
        int_cols = df.select_dtypes(include=['int']).columns
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype(float)

        # 缺失值填补
        num_cols = [c for c in df.columns if c not in cat_cols]
        for c in num_cols:
            if df[c].isnull().any():
                df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            if df[c].isnull().any():
                mode_val = df[c].mode()[0] if not df[c].mode().empty else 'unknown'
                df[c] = df[c].fillna(mode_val)

        # One-Hot 编码
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

        feature_names = list(df.columns)
        X = df.values.astype(np.float64)

    vif_values = calculate_vif(X)
    return feature_names, vif_values


def compute_vif_on_fold(df: pd.DataFrame, target_col: str, cat_cols: list) -> tuple:
    """
    在全量数据上做预处理后计算 VIF（用于诊断，不用于模型评估）。
    与 diagnose_vif_from_dataframe 相同，提供更直观的函数名。

    Parameters
    ----------
    df : pd.DataFrame
        包含目标列的完整数据
    target_col : str
        目标变量列名
    cat_cols : list
        类别特征列名列表

    Returns
    -------
    (feature_names, vif_values) : tuple
    """
    return diagnose_vif_from_dataframe(df, target_col, cat_cols)


def calculate_condition_number(X: np.ndarray) -> float:
    """
    计算矩阵 X 的条件数 (condition number)。

    条件数 = σ_max / σ_min，其中 σ 是奇异值。
    条件数越大，矩阵越接近奇异，OLS 系数估计越不稳定。

    Parameters
    ----------
    X : np.ndarray
        特征矩阵

    Returns
    -------
    float
        条件数
    """
    if hasattr(X, 'values'):
        X = X.values.astype(np.float64)
    S = np.linalg.svd(X, compute_uv=False)
    S = S[S > 1e-12]  # 排除数值零
    if len(S) < 2:
        return np.inf
    return S[0] / S[-1]


def calculate_matrix_rank(X: np.ndarray, tol: float = 1e-10) -> int:
    """
    计算矩阵的数值秩 (numerical rank)。

    Parameters
    ----------
    X : np.ndarray
        特征矩阵
    tol : float
        奇异值阈值，低于此值视为零

    Returns
    -------
    int
        数值秩
    """
    if hasattr(X, 'values'):
        X = X.values.astype(np.float64)
    S = np.linalg.svd(X, compute_uv=False)
    return int(np.sum(S > tol))


def coefficient_stability_analysis(
    X: np.ndarray, y: np.ndarray, n_splits: int = 50,
    feature_indices: list = None, test_size: float = 0.3,
    random_state: int = 42
) -> dict:
    """
    系数稳定性分析：多次随机切分，收集 OLS 系数并计算波动。

    Parameters
    ----------
    X : np.ndarray
        特征矩阵（不含截距列）
    y : np.ndarray
        目标变量
    n_splits : int
        随机切分次数
    feature_indices : list
        要追踪的特征索引列表，None 则取前5个
    test_size : float
        测试集比例
    random_state : int
        基础随机种子

    Returns
    -------
    dict
        {
            'coef_matrix': (n_splits, n_features_tracked),
            'feature_indices': list,
            'coef_std': array,
            'coef_mean': array,
        }
    """
    from sklearn.model_selection import train_test_split

    if feature_indices is None:
        feature_indices = list(range(min(5, X.shape[1])))

    n_tracked = len(feature_indices)
    coef_matrix = np.zeros((n_splits, n_tracked))

    for i in range(n_splits):
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=random_state + i
        )
        # 添加截距列
        X_train_const = np.column_stack([np.ones(X_train.shape[0]), X_train])
        model = AnalyticalOLS()
        model.fit(X_train_const, y_train)
        coef_matrix[i, :] = model.coef_[feature_indices]

    return {
        'coef_matrix': coef_matrix,
        'feature_indices': feature_indices,
        'coef_std': np.std(coef_matrix, axis=0),
        'coef_mean': np.mean(coef_matrix, axis=0),
    }