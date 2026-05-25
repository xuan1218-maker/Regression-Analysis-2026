# Week 12 Example Summary

## Three key conclusions
1. Lower training error does not guarantee better generalization.
2. High-complexity models can become unstable across different training samples.
3. RMSE reacts more strongly than MAE when one prediction error becomes very large.

## Candidate model comparison
| degree | train_rmse | test_rmse |
| --- | --- | --- |
| 1 | 0.638 | 0.679 |
| 4 | 0.372 | 0.387 |
| 15 | 0.3 | 0.338 |

A reasonable first guess for deployment among the three candidates is `degree=15` because it has the best test-side behavior among the shown models.

## Full complexity sweep
Best test RMSE occurs at `degree=9`.
Largest generalization gap occurs at `degree=18`.

| degree | train_rmse | test_rmse | generalization_gap |
| --- | --- | --- | --- |
| 1 | 0.638 | 0.679 | 0.042 |
| 2 | 0.635 | 0.686 | 0.051 |
| 3 | 0.377 | 0.377 | 0.0 |
| 4 | 0.372 | 0.387 | 0.014 |
| 5 | 0.318 | 0.314 | -0.003 |
| 6 | 0.317 | 0.313 | -0.004 |
| 7 | 0.308 | 0.306 | -0.003 |
| 8 | 0.307 | 0.308 | 0.0 |
| 9 | 0.307 | 0.304 | -0.003 |
| 10 | 0.307 | 0.305 | -0.002 |

## Variance demo summary
| degree | mean_prediction_std | max_prediction_std |
| --- | --- | --- |
| 2 | 0.254 | 0.546 |
| 15 | 19.26 | 803.395 |

The higher-degree model has larger prediction dispersion, which is the visible signature of high variance.

## RMSE vs MAE under an outlier
| scenario | RMSE | MAE |
| --- | --- | --- |
| clean prediction | 1.0 | 1.0 |
| one large outlier | 6.083 | 3.0 |

RMSE is hit harder because it squares large errors, while MAE grows linearly.

## Natural transition to next week
If high model complexity creates unstable models, then regularization becomes a natural next step: it intentionally constrains the model to trade a little bias for lower variance.
