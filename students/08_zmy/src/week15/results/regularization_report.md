# Regularization Report (Task D)

## D1. High-dimensional data with collinearity
- Samples: 300, Features: 20
- Collinearity: features 5-9 are correlated with features 0-4 (noise added)
- True relevant features: first 5
- Target generated via Bernoulli(sigmoid(Xβ))

## D2. L1 vs L2 Comparison (best C chosen by cross-validation)

| Model | Accuracy | Recall | ROC-AUC | Log Loss | Non-zero coeffs |
|-------|----------|--------|---------|----------|-----------------|
| L1 (Lasso) | 0.7778 | 0.7959 | 0.9044 | 0.4114 | 7 |
| L2 (Ridge) | 0.7889 | 0.7959 | 0.8905 | 0.4458 | 20 |

## D3. Coefficient plot (fig_d3.png)
- X-axis: Features
- Y-axis: Coefficient value
- Blue bars: L1 coefficients
- Orange bars: L2 coefficients

**Observation**: L1 produces many zero coefficients (sparse), while L2 shrinks coefficients but rarely to zero.

## D4. Core questions

1. **Prediction performance**: Usually similar when signal is strong; here they are close.
2. **Which is sparser?** L1 is much sparser (7 vs 20 non-zero).
3. **Which gives a shorter variable list?** L1. It performs feature selection, easier to explain.
4. **Stability over variable selection**: If business cares more about stability, L2 is better because it does not perform hard selection and is more robust.
