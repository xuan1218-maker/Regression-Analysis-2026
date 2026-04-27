import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['axes.unicode_minus'] = False

def setup_results_dir():
    path = Path(__file__).parent / "results"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    (path / "figures").mkdir()
    return path

def load_data():
    data_path = Path(__file__).parent / "q3_marketing.csv"
    df = pd.read_csv(data_path)
    print(f"数据加载成功，总样本量: {len(df)}")
    print(f"Region列唯一值: {df['Region'].unique()}")
    print(f"NA数量（空值）: {len(df[df['Region'].isna()])}")
    print(f"EU数量: {len(df[df['Region'] == 'EU'])}")
    return df

def split_region(df):
    # 欧洲市场：Region == 'EU'
    df_eu = df[df['Region'] == 'EU'].copy()
    # 北美市场：Region 为空值（nan）
    df_na = df[df['Region'].isna()].copy()
    
    print(f"北美市场样本量: {len(df_na)}")
    print(f"欧洲市场样本量: {len(df_eu)}")
    return df_na, df_eu

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def save_residual_plot(model, X, y, name, path):
    y_pred = model.predict(X)
    resid = y - y_pred
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(y_pred, resid, alpha=0.6)
    ax[0].axhline(0, color='r', linestyle='--')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Residuals')
    ax[0].set_title(f'{name} - Residual Plot')
    ax[0].grid(True, alpha=0.3)
    stats.probplot(resid, plot=ax[1])
    ax[1].set_title(f'{name} - Q-Q Plot')
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"已保存图表: {path}")