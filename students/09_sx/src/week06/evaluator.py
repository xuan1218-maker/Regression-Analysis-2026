import time

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    r2 = model.score(X_test, y_test)
    return f"| {model_name} | {fit_time:.5f} sec | {r2:.4f} |\n"