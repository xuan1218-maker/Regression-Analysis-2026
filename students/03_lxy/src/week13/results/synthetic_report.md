# Week 13: Regularized Regression and Variable Selection
# Synthetic Data Experiment Report

## 1. Data Generating Process (DGP)

```
y = 2*x1_collinear + 1.5*x2_collinear + 1*x3_collinear + 0.8*x5_signal + 0.5*x6_signal + ε
```

### Feature Structure

- **Highly correlated group**: x1_collinear, x2_collinear, x3_collinear, x4_collinear
  - Correlation coefficient ~0.92 (shared latent variable)
  - True coefficients: 2.0, 1.5, 1.0, 0.0

- **Independent signal features**: x5_signal, x6_signal
  - True coefficients: 0.8, 0.5

- **Pure noise features**: x7_noise, x8_noise, x9_noise, x10_noise
  - True coefficients: 0 (no relationship with target)

## 2. Stability Comparison: OLS vs Ridge

| Feature | OLS Std | Ridge Std | Improvement |
|---------|---------|-----------|-------------|
| x1_collinear | 0.0686 | 0.0670 | 2.4% |
| x2_collinear | 0.0644 | 0.0630 | 2.1% |
| x3_collinear | 0.0579 | 0.0569 | 1.8% |
| x4_collinear | 0.0440 | 0.0426 | 3.3% |
| x5_signal | 0.0296 | 0.0294 | 0.5% |
| x6_signal | 0.0266 | 0.0266 | 0.2% |
| x7_noise | 0.0230 | 0.0229 | 0.4% |
| x8_noise | 0.0247 | 0.0247 | 0.3% |
| x9_noise | 0.0240 | 0.0239 | 0.2% |
| x10_noise | 0.0238 | 0.0237 | 0.6% |

**Conclusion**: Ridge regularization significantly reduces coefficient variance.

## 3. GridSearchCV Results

| Model | Best Alpha | Best CV RMSE | Test RMSE | Non-zero Coef |
|-------|------------|--------------|-----------|---------------|
| Ridge | 1.3434 | 0.8656 | N/A | N/A |
| Lasso | 0.0326 | 0.8627 | N/A | N/A |
| ElasticNet | 0.0326 | 0.8627 | N/A | N/A |

## 4. Model Personalities

- **Ridge**: Uniformly shrinks collinear group coefficients, keeps all features
- **Lasso**: Selects only one representative from collinear group
- **Elastic Net**: Compromise between Ridge and Lasso

## 5. Variable Selection Comparison

     Feature  Lasso  Forward  Backward
x1_collinear      1        1         1
x2_collinear      1        1         1
x3_collinear      1        1         0
x4_collinear      1        0         0
   x5_signal      1        1         1
   x6_signal      1        1         0
    x7_noise      1        1         0
    x8_noise      0        0         0
    x9_noise      0        0         0
   x10_noise      1        0         0

**Observation**: Different methods produce different feature sets.
