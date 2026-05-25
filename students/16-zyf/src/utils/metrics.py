import numpy as np


def calculate_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )


def calculate_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(
        np.abs(y_true - y_pred)
    )


def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    epsilon = 1e-8

    denominator = np.where(
        np.abs(y_true) < epsilon,
        epsilon,
        y_true
    )

    return np.mean(
        np.abs(
            (y_true - y_pred)
            / denominator
        )
    ) * 100