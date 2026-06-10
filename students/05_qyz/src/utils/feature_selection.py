import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


class ForwardSelector:
    """
    前向特征选择器（Forward Selection）
    使用交叉验证的 RMSE 作为选择标准，逐步添加特征
    """

    def __init__(self, k_features=None, cv=5):
        """
        初始化前向选择器

        Parameters:
        -----------
        k_features : int or None
            要选择的特征数量。如果为 None，则选择全部特征
        cv : int
            交叉验证折数（默认 5 折）
        """
        self.k_features = k_features  # 目标特征数
        self.cv = cv  # 交叉验证折数

        self.selected_features_ = []  # 最终选中的特征索引列表
        self.history_ = []  # 每一步的选择历史（特征集合 + 对应 RMSE）

    def fit(self, X, y):
        """
        执行前向特征选择

        Parameters:
        -----------
        X : 二维数组，形状 (n_samples, n_features)
            特征矩阵
        y : 一维数组，形状 (n_samples,)
            目标变量

        Returns:
        --------
        self : ForwardSelector
            已训练的选择器实例
        """

        # 初始化：所有特征的索引列表
        remaining = list(range(X.shape[1]))
        # 已选中的特征索引列表，初始为空
        selected = []

        # 确定要选择的特征数量
        if self.k_features is None:
            k_features = X.shape[1]  # 未指定时，选择全部特征
        else:
            k_features = self.k_features

        # 前向选择主循环：逐步添加特征，直到达到目标数量
        while len(selected) < k_features:
            best_score = np.inf  # 初始化最优 RMSE 为无穷大
            best_feature = None  # 初始化最优特征索引

            # 遍历所有剩余特征，尝试加入当前集合
            for feature in remaining:
                # 构造临时特征集合：已选特征 + 当前尝试的特征
                trial = selected + [feature]

                # 使用线性回归模型评估该特征集合的性能
                model = LinearRegression()

                # 交叉验证：计算每折的负 RMSE（因为 sklearn 的 scoring 越小越好）
                scores = -cross_val_score(
                    model,
                    X[:, trial],  # 使用当前尝试的特征子集
                    y,
                    scoring="neg_root_mean_squared_error",  # 以 RMSE 为评估指标
                    cv=self.cv,
                )

                # 取交叉验证 RMSE 的平均值
                rmse = scores.mean()

                # 如果当前特征带来更低的 RMSE（更好的性能），更新最优解
                if rmse < best_score:
                    best_score = rmse
                    best_feature = feature

            # 将最优特征加入已选集合，从剩余列表中移除
            selected.append(best_feature)
            remaining.remove(best_feature)

            # 记录当前步的选择结果（特征集合 + 交叉验证 RMSE）
            self.history_.append((selected.copy(), best_score))

        # 保存最终选中的特征列表
        self.selected_features_ = selected

        return self
