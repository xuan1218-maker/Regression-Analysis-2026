"""从 Week10 到 Week14 持续维护的回归模型和模型选择工具。

本文件保留前几周已经实现的 OLS、梯度下降和前向选择等工具，
并在 Week14 增加 PCR 工作流、矩阵条件数和稳定性比较函数。
在作业允许的位置使用 sklearn 组件，但核心实验逻辑仍放在自己的 utils 包中。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from utils.metrics import calculate_rmse
from utils.transformers import CustomStandardScaler


class AnalyticalOLS:
    """用最小二乘方式求解普通线性回归。

    该类不会自动添加截距项；如果需要截距，请在拟合前自行添加一列 1。
    """

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AnalyticalOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if np.isclose(sst, 0.0):
            return 0.0
        return float(1.0 - sse / sst)


class GradientDescentOLS:
    """使用梯度下降求解 OLS；保留自前几周作业。"""

    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-8,
        max_iter: int = 10000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.25,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if gd_type not in {"full_batch", "mini_batch"}:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")
        if not 0 < batch_fraction <= 1:
            raise ValueError("batch_fraction must be in (0, 1]")

        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "GradientDescentOLS":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        n_samples, n_features = X.shape
        batch_size = n_samples
        if self.gd_type == "mini_batch":
            batch_size = max(1, int(round(n_samples * self.batch_fraction)))

        self.coef_ = np.zeros(n_features, dtype=float)
        self.loss_history_ = []
        rng = np.random.default_rng(seed)
        previous_loss = np.inf

        for _ in range(self.max_iter):
            if self.gd_type == "mini_batch":
                idx = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
            else:
                X_batch = X
                y_batch = y

            error_batch = X_batch @ self.coef_ - y_batch
            gradient = (2.0 / X_batch.shape[0]) * (X_batch.T @ error_batch)
            self.coef_ -= self.learning_rate * gradient

            full_error = X @ self.coef_ - y
            loss = float(np.mean(full_error**2))
            self.loss_history_.append(loss)
            if abs(previous_loss - loss) < self.tol:
                break
            previous_loss = loss

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if np.isclose(sst, 0.0):
            return 0.0
        return float(1.0 - sse / sst)


CustomOLS = AnalyticalOLS


class OrdinaryLeastSquares:
    """自动在内部添加截距项的便捷 OLS 回归器。"""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinaryLeastSquares":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        X_design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        self.coef_ = np.linalg.lstsq(X_design, y_arr, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The model has not been fitted yet")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        X_design = np.column_stack([np.ones(X_arr.shape[0]), X_arr])
        return X_design @ self.coef_


LinearRegressionWithIntercept = OrdinaryLeastSquares


@dataclass
class SelectionStep:
    """前向选择搜索过程中的一条历史记录。"""

    step: int
    added_feature: str
    cv_rmse: float
    selected_features: list[str]


class ForwardSelectorCV:
    """基于 K 折交叉验证评价的贪心前向变量选择。"""

    def __init__(self, max_features: int = 5, cv: int = 5, random_state: int = 42) -> None:
        if max_features <= 0:
            raise ValueError("max_features must be positive")
        if cv < 2:
            raise ValueError("cv must be at least 2")
        self.max_features = max_features
        self.cv = cv
        self.random_state = random_state
        self.selected_indices_: list[int] | None = None
        self.selected_features_: list[str] | None = None
        self.history_: list[SelectionStep] = []
        self.estimator_: LinearRegression | None = None

    def _cv_rmse(self, X: np.ndarray, y: np.ndarray, indices: list[int]) -> float:
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        rmses: list[float] = []
        for train_idx, val_idx in splitter.split(X):
            model = LinearRegression()
            model.fit(X[train_idx][:, indices], y[train_idx])
            pred = model.predict(X[val_idx][:, indices])
            rmses.append(calculate_rmse(y[val_idx], pred))
        return float(np.mean(rmses))

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "ForwardSelectorCV":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X must be 2-D")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X_arr.shape[1])]
        if len(feature_names) != X_arr.shape[1]:
            raise ValueError("feature_names length must match X columns")

        selected: list[int] = []
        remaining = list(range(X_arr.shape[1]))
        self.history_ = []

        for step in range(1, min(self.max_features, X_arr.shape[1]) + 1):
            scores: list[tuple[float, int]] = []
            for candidate in remaining:
                score = self._cv_rmse(X_arr, y_arr, selected + [candidate])
                scores.append((score, candidate))
            best_score, best_feature_idx = min(scores, key=lambda pair: pair[0])
            selected.append(best_feature_idx)
            remaining.remove(best_feature_idx)
            self.history_.append(
                SelectionStep(
                    step=step,
                    added_feature=feature_names[best_feature_idx],
                    cv_rmse=best_score,
                    selected_features=[feature_names[i] for i in selected],
                )
            )

        self.selected_indices_ = selected
        self.selected_features_ = [feature_names[i] for i in selected]
        self.estimator_ = LinearRegression().fit(X_arr[:, selected], y_arr)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.selected_indices_ is None or self.estimator_ is None:
            raise RuntimeError("ForwardSelectorCV must be fitted before predict")
        X_arr = np.asarray(X, dtype=float)
        return self.estimator_.predict(X_arr[:, self.selected_indices_])

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "step": [row.step for row in self.history_],
                "added_feature": [row.added_feature for row in self.history_],
                "cv_rmse": [row.cv_rmse for row in self.history_],
                "selected_features": [", ".join(row.selected_features) for row in self.history_],
            }
        )


class PCRRegressor:
    """主成分回归 PCR：标准化 -> PCA -> 线性回归。

    标准化器和 PCA 都只在训练数据上拟合。验证集/测试集只用已拟合对象转换，
    从而避免数据泄露。
    """

    def __init__(self, n_components: int) -> None:
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = int(n_components)
        self.scaler = CustomStandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.regressor = LinearRegression()
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> "PCRRegressor":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        X_scaled = self.scaler.fit_transform(X_arr)
        Z = self.pca.fit_transform(X_scaled)
        self.regressor.fit(Z, y_arr)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        Z = self.transform(X)
        return self.regressor.predict(Z)


def pcr_cv_rmse(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    n_components: int,
    cv: int = 5,
    random_state: int = 42,
) -> float:
    """计算 PCR 的交叉验证 RMSE；每折内部重新完成全部预处理。"""
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rmses: list[float] = []
    for train_idx, val_idx in splitter.split(X_arr):
        model = PCRRegressor(n_components=n_components)
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[val_idx])
        rmses.append(calculate_rmse(y_arr[val_idx], pred))
    return float(np.mean(rmses))


def design_rank_and_condition(X: np.ndarray, tol: float = 1e-10) -> tuple[int, float]:
    """返回矩阵秩，以及显式考虑奇异性的条件数。

    在高维 OLS 中，X 的列数可能大于行数。此时即使 X 满行秩，X'X 也会奇异，
    因此只要 rank < 特征列数，本函数就返回 ``inf``。
    """
    X_arr = np.asarray(X, dtype=float)
    rank = int(np.linalg.matrix_rank(X_arr, tol=tol))
    if rank < X_arr.shape[1]:
        return rank, float("inf")
    singular_values = np.linalg.svd(X_arr, compute_uv=False)
    positive = singular_values[singular_values > tol]
    if positive.size == 0:
        return rank, float("inf")
    return rank, float(positive.max() / positive.min())


def prediction_stability_score(prediction_matrix: np.ndarray) -> float:
    """计算重复预测矩阵逐样本标准差的平均值。

    数值越低，表示同一批锚点样本在不同训练切分下得到的预测越稳定。
    """
    arr = np.asarray(prediction_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("prediction_matrix must be 2-D: repeats x anchor_points")
    return float(np.mean(np.std(arr, axis=0)))
