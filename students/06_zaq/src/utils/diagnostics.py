"""
Module: utils.diagnostics
Purpose: VIF calculation
"""
import math


def calculate_vif(X):
    """Calculate VIF for each feature (handles None values)"""
    # 先填补缺失值
    n = len(X)
    p = len(X[0]) if X else 1
    
    # 用每列均值填补 None
    X_clean = [[0.0]*p for _ in range(n)]
    for j in range(p):
        # 收集有效值
        valid_vals = []
        for i in range(n):
            val = X[i][j]
            if val is not None:
                valid_vals.append(float(val))
        mean_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
        
        for i in range(n):
            val = X[i][j]
            X_clean[i][j] = float(val) if val is not None else mean_val
    
    X = X_clean
    n = len(X)
    p = len(X[0])
    
    vifs = []
    
    for j in range(p):
        y = [row[j] for row in X]
        X_other = [[row[k] for k in range(p) if k != j] for row in X]
        
        # Add intercept
        X_other = [[1.0] + row for row in X_other]
        
        # Solve using normal equations
        n2, p2 = len(X_other), len(X_other[0])
        XTX = [[0.0]*p2 for _ in range(p2)]
        for i in range(p2):
            for k in range(p2):
                s = 0.0
                for m in range(n2):
                    s += X_other[m][i] * X_other[m][k]
                XTX[i][k] = s
        
        XTy = [0.0]*p2
        for i in range(p2):
            s = 0.0
            for m in range(n2):
                s += X_other[m][i] * y[m]
            XTy[i] = s
        
        # Gaussian elimination
        aug = [[0.0]*(2*p2) for _ in range(p2)]
        for i in range(p2):
            for k in range(p2):
                aug[i][k] = XTX[i][k]
            aug[i][p2+i] = 1.0
        
        for i in range(p2):
            pivot = aug[i][i]
            for k in range(2*p2):
                aug[i][k] /= pivot
            for k in range(p2):
                if k != i:
                    factor = aug[k][i]
                    for m in range(2*p2):
                        aug[k][m] -= factor * aug[i][m]
        
        coef = [aug[i][p2+i] for i in range(p2)]
        
        # R²
        y_pred = [sum(coef[k] * X_other[m][k] for k in range(p2)) for m in range(n2)]
        y_mean = sum(y) / n2
        ss_res = sum((y[m] - y_pred[m])**2 for m in range(n2))
        ss_tot = sum((y[m] - y_mean)**2 for m in range(n2))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        
        vif = 1/(1 - r2) if r2 < 0.999 else float('inf')
        vifs.append(vif)
    
    return vifs


def print_vif_warning(vif_values, feature_names):
    print("\n" + "="*60)
    print("VIF (Variance Inflation Factor) 诊断报告")
    print("="*60)
    print(f"{'Feature':<20} | {'VIF':<10} | {'Status'}")
    print("-" * 45)
    
    for name, vif in zip(feature_names, vif_values):
        if vif == float('inf'):
            status = "❌ 严重 (完全共线性)"
        elif vif > 10:
            status = "⚠️ 严重"
        elif vif > 5:
            status = "⚠️ 注意"
        else:
            status = "✅ 正常"
        vif_str = "∞" if vif == float('inf') else f"{vif:.2f}"
        print(f"{name:<20} | {vif_str:<10} | {status}")
    print("-" * 45)
