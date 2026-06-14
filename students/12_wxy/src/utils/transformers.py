import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
# ====================== A1 作业要求：生成低秩隐因子高维数据 ======================
def generate_highdim_latent_data(
    n_samples: int,
    n_features: int,
    n_latent: int,
    noise_std: float = 0.3,
    random_seed: int = 42
):
    """
    生成带隐因子的高维冗余数据（满足A1：p>n、低秩隐因子结构）
    :param n_samples: 样本量 ≥120
    :param n_features: 特征数 ≥60
    :param n_latent: 隐因子数量（远小于p）
    :return: X, y, latent_factors
    """
    rng = np.random.default_rng(random_seed)
    # 1.生成少量隐因子 latent factors
    latent = rng.normal(size=(n_samples, n_latent))
    # 2.载荷矩阵：隐因子线性组合生成全部原始特征（制造多重共线性）
    loadings = rng.normal(size=(n_latent, n_features))
    X = latent @ loadings + noise_std * rng.normal(size=(n_samples, n_features))
    # 3.y仅由隐因子线性生成，不由原始变量独立驱动
    beta_latent = rng.normal(size=n_latent)
    y = latent @ beta_latent + noise_std * rng.normal(size=n_samples)
    return X, y, latent

# ====================== C1场景1：稀疏真值数据生成 ======================
def generate_sparse_truth_data(
    n_samples: int,
    n_features: int,
    n_signal_feats: int,
    noise_std: float = 0.3,
    random_seed: int = 42
):
    """稀疏场景：仅少数原始变量有效，其余纯噪声"""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(size=(n_samples, n_features))
    beta = np.zeros(n_features)
    signal_idx = rng.choice(np.arange(n_features), size=n_signal_feats, replace=False)
    beta[signal_idx] = rng.normal(size=n_signal_feats)
    y = X @ beta + noise_std * rng.normal(size=n_samples)
    return X, y, signal_idx
