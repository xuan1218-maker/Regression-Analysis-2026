# src/utils/models.py
"""
模块：工具.模型
用途：核心机器学习估计器
包含：解析解普通最小二乘法、梯度下降普通最小二乘法、PCR、稳定性分析
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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


class PCR:
    """
    主成分回归 (Principal Component Regression)
    
    流程：标准化 -> PCA -> 保留前k个主成分 -> 线性回归
    """
    
    def __init__(self, n_components: int = None, variance_ratio: float = None):
        """
        参数:
            n_components: 保留的主成分个数
            variance_ratio: 累计解释方差比例（与n_components二选一）
        """
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.scaler = StandardScaler()
        self.pca = None
        self.regressor = LinearRegression()
        self._is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PCR':
        X_scaled = self.scaler.fit_transform(X)
        
        if self.variance_ratio is not None:
            self.pca = PCA(n_components=self.variance_ratio)
        else:
            self.pca = PCA(n_components=self.n_components)
        
        Z = self.pca.fit_transform(X_scaled)
        self.regressor.fit(Z, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit()")
        X_scaled = self.scaler.transform(X)
        Z = self.pca.transform(X_scaled)
        return self.regressor.predict(Z)
    
    def get_original_coefficients(self) -> np.ndarray:
        """
        获取原始空间中的系数（近似）
        返回: beta_original = V @ beta_pc
        """
        if not self._is_fitted:
            raise RuntimeError("请先调用 fit()")
        V = self.pca.components_.T
        beta_pc = self.regressor.coef_
        return V @ beta_pc
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_
    
    def get_cumulative_variance_ratio(self) -> np.ndarray:
        return np.cumsum(self.pca.explained_variance_ratio_)
    
    def select_components_by_variance(self, X: np.ndarray, target_ratio: float = 0.9) -> int:
        """根据目标解释方差比例选择主成分个数"""
        X_scaled = self.scaler.fit_transform(X)
        pca_temp = PCA()
        pca_temp.fit(X_scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n = np.argmax(cumsum >= target_ratio) + 1
        return n


class CoefficientStabilityAnalyzer:
    """系数稳定性分析器"""
    
    def __init__(self, model_class, model_params: dict = None, n_splits: int = 50, test_size: float = 0.3):
        """
        参数:
            model_class: 模型类（如 LinearRegression, LassoCV, PCR）
            model_params: 模型参数字典
            n_splits: 随机切分次数
            test_size: 测试集比例
        """
        self.model_class = model_class
        self.model_params = model_params or {}
        self.n_splits = n_splits
        self.test_size = test_size
        
    def analyze(self, X: np.ndarray, y: np.ndarray, random_seed_base: int = 0) -> Dict:
        """
        分析模型在不同随机切分下的稳定性
        
        返回:
            dict: 包含系数矩阵、误差记录、稳定性指标
        """
        n_features = X.shape[1]
        coeffs_history = []
        train_errors = []
        test_errors = []
        
        for split_id in range(self.n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=random_seed_base + split_id
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = self.model_class(**self.model_params)
            
            if self.model_class.__name__ == 'PCR':
                model.fit(X_train_scaled, y_train)
                coeffs = model.get_original_coefficients()
            else:
                model.fit(X_train_scaled, y_train)
                coeffs = model.coef_
            
            if len(coeffs) == n_features:
                coeffs_history.append(coeffs)
            
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            train_errors.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
            test_errors.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        
        coeffs_array = np.array(coeffs_history)
        
        return {
            'coeffs': coeffs_array,
            'coeff_mean': np.mean(coeffs_array, axis=0) if len(coeffs_array) > 0 else np.array([]),
            'coeff_std': np.std(coeffs_array, axis=0) if len(coeffs_array) > 0 else np.array([]),
            'train_rmse_mean': np.mean(train_errors),
            'train_rmse_std': np.std(train_errors),
            'test_rmse_mean': np.mean(test_errors),
            'test_rmse_std': np.std(test_errors)
        }
    
    def get_stability_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算稳定性分数（系数标准差的平均值，越小越稳定）"""
        results = self.analyze(X, y)
        if len(results['coeff_std']) > 0:
            return np.mean(results['coeff_std'])
        return np.inf


class LassoPCRComparator:
    """Lasso与PCR比较器"""
    
    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.results = {}
    
    def compare(self, X_train: np.ndarray, X_test: np.ndarray, 
                y_train: np.ndarray, y_test: np.ndarray,
                pcr_components: int = 10) -> Dict:
        """
        比较Lasso和PCR在给定数据上的表现
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Lasso
        lasso = LassoCV(cv=self.cv_folds, random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        
        # PCR
        pcr = PCR(n_components=pcr_components)
        pcr.fit(X_train_scaled, y_train)
        y_pred_pcr = pcr.predict(X_test_scaled)
        
        self.results = {
            'Lasso': {
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
                'test_mae': np.mean(np.abs(y_test - y_pred_lasso)),
                'n_nonzero': np.sum(np.abs(lasso.coef_) > 1e-6),
                'alpha': lasso.alpha_,
                'predictions': y_pred_lasso
            },
            'PCR': {
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_pcr)),
                'test_mae': np.mean(np.abs(y_test - y_pred_pcr)),
                'n_components': pcr_components,
                'predictions': y_pred_pcr
            }
        }
        
        return self.results
    
    def print_comparison(self):
        """打印比较结果"""
        print("\n" + "="*50)
        print("Lasso vs PCR 比较结果")
        print("="*50)
        print(f"{'指标':<20} {'Lasso':<20} {'PCR':<20}")
        print("-"*60)
        print(f"{'Test RMSE':<20} {self.results['Lasso']['test_rmse']:<20.4f} {self.results['PCR']['test_rmse']:<20.4f}")
        print(f"{'Test MAE':<20} {self.results['Lasso']['test_mae']:<20.4f} {self.results['PCR']['test_mae']:<20.4f}")
        print(f"{'模型复杂度':<20} {self.results['Lasso']['n_nonzero']:<20} {self.results['PCR']['n_components']:<20}")
        print("="*50)


class ForwardSelector:
    """前向选择 - 基于交叉验证的特征选择"""
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'neg_mean_squared_error', max_features: int = None):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.max_features = max_features
        self.selected_indices_ = None
        self.selected_names_ = None
        self.cv_scores_ = []
        
    def select(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> List[int]:
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
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.min_features = min_features
        self.selected_indices_ = None
        self.selected_names_ = None
        self.cv_scores_ = []
        
    def select(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> List[int]:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        
        n_features = X.shape[1]
        current_features = list(range(n_features))
        self.cv_scores_ = []
        
        print(f"开始后向剔除，共{n_features}个特征，最少保留{self.min_features}个...")
        
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