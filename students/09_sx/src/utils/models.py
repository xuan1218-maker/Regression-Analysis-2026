"""
模块：工具.模型
用途：核心机器学习估计器
包含：解析解普通最小二乘法、梯度下降普通最小二乘法
"""

import numpy as np
from typing import List, Optional  # 添加这行导入

class AnalyticalOLS:
    """解析解普通最小二乘法"""
    
    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept
        self.coef_ = None
        self.sigma2_ = None
        self.df_resid_ = None
        self.X_design_ = None
        
    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def fit(self, X, y):
        if self.add_intercept:
            self.X_design_ = self._add_intercept(X)
        else:
            self.X_design_ = X
        
        n, k = self.X_design_.shape
        self.df_resid_ = n - k
        
        XTX = self.X_design_.T @ self.X_design_
        XTy = self.X_design_.T @ y
        self.coef_ = np.linalg.solve(XTX + 1e-10 * np.eye(k), XTy)
        
        y_pred = self.predict(X)
        rss = np.sum((y - y_pred) ** 2)
        self.sigma2_ = rss / self.df_resid_ if self.df_resid_ > 0 else rss
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()")
        if self.add_intercept:
            X = self._add_intercept(X)
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 0
        return 1 - sse / sst


class GradientDescentOLS:
    """梯度下降普通最小二乘法"""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
        add_intercept: bool = True,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction
        self.add_intercept = add_intercept
        
        self.coef_ = None
        self.loss_history_ = []
        self.X_design_ = None
        
    def _add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def _compute_mse(self, X, y, coef):
        y_pred = X @ coef
        return np.mean((y - y_pred) ** 2)
    
    def fit(self, X, y, seed: int = 42):
        if self.add_intercept:
            self.X_design_ = self._add_intercept(X)
        else:
            self.X_design_ = X
        
        n_samples, n_features = self.X_design_.shape
        
        self.coef_ = np.zeros(n_features)
        self.loss_history_ = []
        
        rng = np.random.default_rng(seed)
        
        if self.gd_type == "full_batch":
            batch_size = n_samples
        elif self.gd_type == "mini_batch":
            batch_size = max(1, int(n_samples * self.batch_fraction))
        else:
            raise ValueError("gd_type must be 'full_batch' or 'mini_batch'")
        
        for epoch in range(self.max_iter):
            if self.gd_type == "mini_batch":
                indices = rng.choice(n_samples, size=batch_size, replace=False)
                X_batch = self.X_design_[indices]
                y_batch = y[indices]
            else:
                X_batch = self.X_design_
                y_batch = y
            
            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)
            
            self.coef_ -= self.learning_rate * gradient
            
            mse = self._compute_mse(self.X_design_, y, self.coef_)
            self.loss_history_.append(mse)
            
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break
        
        return self
    
    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("请先调用 fit()")
        if self.add_intercept:
            X = self._add_intercept(X)
        return X @ self.coef_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 0
        return 1 - sse / sst
    


# 在 utils/models.py 末尾添加以下内容

class ForwardSelector:
    """前向选择 - 基于交叉验证的特征选择"""
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'neg_mean_squared_error', max_features: int = None):
        """
        参数:
            cv_folds: 交叉验证折数
            scoring: 评估指标
            max_features: 最多选择的特征数
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.max_features = max_features
        self.selected_indices_ = None
        self.selected_names_ = None
        self.cv_scores_ = []
        
    def select(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> List[int]:
        """
        执行前向选择
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
            feature_names: 特征名称列表
            
        返回:
            selected_indices: 被选中的特征索引列表
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        
        n_samples, n_features = X.shape
        if self.max_features is None:
            self.max_features = min(n_features, 10)
        
        remaining = list(range(n_features))
        selected = []
        self.cv_scores_ = []
        
        print(f"开始前向选择，共{n_features}个特征，最多选择{self.max_features}个...")
        
        for k in range(self.max_features):
            if not remaining:
                break
                
            best_score = -np.inf
            best_feature = None
            
            for feature in remaining:
                current_set = selected + [feature]
                X_subset = X[:, current_set]
                
                model = LinearRegression()
                scores = cross_val_score(model, X_subset, y, cv=self.cv_folds, 
                                        scoring=self.scoring)
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_feature = feature
            
            if best_feature is not None:
                selected.append(best_feature)
                remaining.remove(best_feature)
                self.cv_scores_.append(best_score)
                name = feature_names[best_feature] if feature_names else str(best_feature)
                print(f"  第{k+1}步: 选中特征 {name}, CV分数={best_score:.4f}")
            else:
                break
        
        self.selected_indices_ = selected
        if feature_names:
            self.selected_names_ = [feature_names[i] for i in selected]
        
        print(f"前向选择完成，共选中{len(selected)}个特征")
        return selected


class BackwardEliminator:
    """后向剔除 - 基于交叉验证的特征选择"""
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'neg_mean_squared_error', min_features: int = 1):
        """
        参数:
            cv_folds: 交叉验证折数
            scoring: 评估指标
            min_features: 最少保留的特征数
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.min_features = min_features
        self.selected_indices_ = None
        self.selected_names_ = None
        self.cv_scores_ = []
        
    def select(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> List[int]:
        """
        执行后向剔除
        
        参数:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标变量 (n_samples,)
            feature_names: 特征名称列表
            
        返回:
            selected_indices: 被选中的特征索引列表
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        
        n_features = X.shape[1]
        current_features = list(range(n_features))
        self.cv_scores_ = []
        
        print(f"开始后向剔除，共{n_features}个特征，最少保留{self.min_features}个...")
        
        # 先计算全模型的分数
        model = LinearRegression()
        full_score = np.mean(cross_val_score(model, X, y, cv=self.cv_folds, scoring=self.scoring))
        self.cv_scores_.append(full_score)
        print(f"  全模型CV分数: {full_score:.4f}")
        
        step = 1
        while len(current_features) > self.min_features:
            worst_score = -np.inf
            worst_feature = None
            
            for feature in current_features:
                temp_features = [f for f in current_features if f != feature]
                X_subset = X[:, temp_features]
                
                model = LinearRegression()
                scores = cross_val_score(model, X_subset, y, cv=self.cv_folds, 
                                        scoring=self.scoring)
                avg_score = np.mean(scores)
                
                if avg_score > worst_score:
                    worst_score = avg_score
                    worst_feature = feature
            
            # 移除最差特征
            if worst_feature is not None:
                current_features.remove(worst_feature)
                self.cv_scores_.append(worst_score)
                name = feature_names[worst_feature] if feature_names else str(worst_feature)
                print(f"  第{step}步: 剔除特征 {name}, CV分数={worst_score:.4f}")
                step += 1
            else:
                break
        
        self.selected_indices_ = current_features
        if feature_names:
            self.selected_names_ = [feature_names[i] for i in current_features]
        
        print(f"后向剔除完成，共保留{len(current_features)}个特征")
        return current_features