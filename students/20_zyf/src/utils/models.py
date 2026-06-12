"""
Module: utils.models
Purpose: Core machine learning estimators.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


class AnalyticalOLS:
    """Analytical solution for OLS using normal equation."""
    
    def __init__(self):
        self.coef_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model using normal equation: beta = (X'X)^{-1}X'y
    
        Uses lstsq (not solve) to gracefully handle near-singular matrices.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.coef_, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 1.0 if sse == 0 else 0.0
        return 1 - sse / sst


class GradientDescentOLS:
    """Linear regression using gradient descent optimization."""
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        tol: float = 1e-5,
        max_iter: int = 1000,
        gd_type: str = "full_batch",
        batch_fraction: float = 0.1,
    ):
        self.learning_rate = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.gd_type = gd_type
        self.batch_fraction = batch_fraction

        self.coef_ = None
        self.loss_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        """Fit the model using gradient descent."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features, dtype=np.float64)
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
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            y_pred_batch = X_batch @ self.coef_
            error_batch = y_pred_batch - y_batch
            gradient = (2 / len(X_batch)) * (X_batch.T @ error_batch)

            self.coef_ -= self.learning_rate * gradient

            # Record full loss for comparison
            y_pred_full = X @ self.coef_
            mse = np.mean((y - y_pred_full) ** 2)
            self.loss_history_.append(mse)

            # Check convergence
            if epoch > 0:
                delta = abs(self.loss_history_[-1] - self.loss_history_[-2])
                if delta < self.tol:
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        y_pred = self.predict(X)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 1.0 if sse == 0 else 0.0
        return 1 - sse / sst


class ForwardSelectionRegressor:
    """Forward selection variable selection using cross-validation."""
    
    def __init__(self, k_features: int = None, cv_folds: int = 5):
        """
        Parameters:
        -----------
        k_features : int or None
            Number of features to select. If None, select based on best CV score.
        cv_folds : int
            Number of folds for cross-validation.
        """
        self.k_features = k_features
        self.cv_folds = cv_folds
        self.selected_features_ = None
        self.cv_scores_history_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Perform forward selection.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target variable
        feature_names : list, optional
            Names of features for tracking
        
        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(n_features)]
        
        self.feature_names_ = feature_names
        self.selected_features_ = []
        remaining_features = list(range(n_features))
        self.cv_scores_history_ = []
        
        k_max = self.k_features if self.k_features is not None else n_features
        
        for _ in range(min(k_max, n_features)):
            best_feature = None
            best_score = -np.inf
            best_idx_in_remaining = None
            
            # Try adding each remaining feature
            for idx, feat_id in enumerate(remaining_features):
                candidate_features = self.selected_features_ + [feat_id]
                
                # Cross-validation score
                cv_score = self._cv_score(X[:, candidate_features], y)
                
                if cv_score > best_score:
                    best_score = cv_score
                    best_feature = feat_id
                    best_idx_in_remaining = idx
            
            # Add best feature
            if best_feature is not None:
                self.selected_features_.append(best_feature)
                remaining_features.pop(best_idx_in_remaining)
                self.cv_scores_history_.append(best_score)
        
        return self
    
    def _cv_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate cross-validation R² score."""
        n_samples = X.shape[0]
        fold_size = n_samples // self.cv_folds
        scores = []
        
        for fold in range(self.cv_folds):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < self.cv_folds - 1 else n_samples
            
            # Split into train/test
            test_idx = np.arange(test_start, test_end)
            train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit OLS
            model = AnalyticalOLS()
            model.fit(X_train, y_train)
            
            # Score
            score = model.score(X_test, y_test)
            scores.append(score)
        
        return np.mean(scores)
    
    def get_selected_feature_names(self) -> list:
        """Return names of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Model must be fitted first.")
        return [self.feature_names_[i] for i in self.selected_features_]


class PCR:
    """
    Principal Component Regression (PCR).
    
    Workflow: Standardize -> PCA -> keep k PCs -> LinearRegression on PC scores.
    
    This class wraps sklearn's PCA and LinearRegression, but the workflow
    logic (how k is chosen, how CV is performed) is explicitly managed
    by the caller or via the built-in CV method.
    """
    
    def __init__(self, n_components: int = None):
        """
        Parameters:
        -----------
        n_components : int or None
            Number of principal components to retain. If None, all components
            are kept (equivalent to OLS on standardized data).
        """
        self.n_components = n_components
        self.pca_ = None
        self.regressor_ = None
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit PCR: standardize X, apply PCA, then fit linear regression on PC scores.
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        
        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        
        # Standardize
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid division by zero
        X_scaled = (X - self.mean_) / self.std_
        
        # PCA
        n_components = min(self.n_components, X.shape[1]) if self.n_components is not None else X.shape[1]
        self.pca_ = PCA(n_components=n_components)
        Z = self.pca_.fit_transform(X_scaled)
        
        # Linear regression on PC scores
        self.regressor_ = LinearRegression()
        self.regressor_.fit(Z, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted PCR model."""
        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self.mean_) / self.std_
        Z = self.pca_.transform(X_scaled)
        return self.regressor_.predict(Z)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R-squared."""
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        if sst == 0:
            return 1.0 if sse == 0 else 0.0
        return 1 - sse / sst


def compute_matrix_rank(X: np.ndarray) -> int:
    """Compute the numerical rank of a matrix using SVD tolerance."""
    X = np.asarray(X, dtype=np.float64)
    return np.linalg.matrix_rank(X)


def compute_condition_number(X: np.ndarray) -> float:
    """Compute the condition number of X (ratio of largest to smallest singular value)."""
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.svd(X, compute_uv=False)
    s_pos = s[s > 1e-12]
    if len(s_pos) == 0:
        return np.inf
    return float(s_pos[0] / s_pos[-1])


def compute_coefficient_stability(X: np.ndarray, y: np.ndarray, n_splits: int = 50,
                                   selected_indices: list = None) -> dict:
    """
    Compute the stability of OLS coefficients across multiple random splits.
    
    Repeatedly splits data into train sets (80%), fits OLS, and collects coefficients.
    Returns statistics on coefficient variation.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_splits : int
        Number of random splits
    selected_indices : list, optional
        Which feature indices to track. If None, tracks first 5.
    
    Returns:
    --------
    dict with keys: 'coef_trajectories', 'coef_stds', 'selected_indices'
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n_samples, n_features = X.shape
    
    if selected_indices is None:
        selected_indices = list(range(min(5, n_features)))
    
    coef_trajectories = {idx: [] for idx in selected_indices}
    
    for split_i in range(n_splits):
        n_train = int(0.8 * n_samples)
        indices = np.random.RandomState(split_i).permutation(n_samples)
        train_idx = indices[:n_train]
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        
        model = AnalyticalOLS()
        model.fit(X_train, y_train)
        
        for idx in selected_indices:
            if idx < len(model.coef_):
                coef_trajectories[idx].append(float(model.coef_[idx]))
    
    coef_stds = {idx: float(np.std(coef_trajectories[idx])) for idx in selected_indices}
    
    return {
        'coef_trajectories': coef_trajectories,
        'coef_stds': coef_stds,
        'selected_indices': selected_indices
    }