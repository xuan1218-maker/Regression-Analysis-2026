import time
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name: str) -> str:
    """
    This function doesn't care if `model` is CustomOLS or sklearn.LinearRegression.
    As long as it has .fit() and .predict() methods, it works!
    """
    start_time = time.perf_counter()
    
    # 1. Train the model
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start_time
    
    # 2. Predict
    y_pred = model.predict(X_test)
    
    # 3. Calculate R² manually
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 4. Format result
    result_str = f"| {model_name} | {fit_time:.5f} sec | {r2_score:.4f} |\n"
    
    return result_str