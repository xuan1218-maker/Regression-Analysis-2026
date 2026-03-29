import os
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from functions import generate_data, formula_estimation, sklearn_estimation, statsmodels_estimation

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  

def main():
    X, Y = generate_data()
    
    b0_f, b1_f, var_b1 = formula_estimation(X, Y)
    print(f"Formula: beta0={b0_f:.4f}, beta1={b1_f:.4f}, Var(beta1)={var_b1:.6f}")
    
    b0_sk, b1_sk = sklearn_estimation(X, Y)
    print(f"sklearn: beta0={b0_sk:.4f}, beta1={b1_sk:.4f}")
    
    b0_sm, b1_sm, model = statsmodels_estimation(X, Y)
    print(f"statsmodels: beta0={b0_sm:.4f}, beta1={b1_sm:.4f}")
    
    print(f"Bias: beta0={b0_f-1:.4f}, beta1={b1_f-2:.4f}")
    print(f"Hypothesis test p-value: {model.pvalues[1]:.6f}")
    print(f"R²={model.rsquared:.4f}, F={model.fvalue:.4f}, F p-value={model.f_pvalue:.6f}")
    
    plt.scatter(X, Y, alpha=0.6, label='Sample data')
    x_line = np.linspace(X.min(), X.max(), 100)
    plt.plot(x_line, b0_f + b1_f * x_line, 'r-', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Simple Linear Regression')
    plt.legend()
    plt.savefig('regression_plot.png')  
    plt.show()

if __name__ == "__main__":
    main()