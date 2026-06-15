# Task A — Synthetic Correlated Data Report

## A1 & A2: Data Generating Process

**Sample size:** n = 300
**Features:** ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

**True DGP:**
```
y = 3*x1 + 1.5*x2 - 2*x3 + 2*x6 + ε,  ε ~ N(0, 0.5)
```

**Highly correlated group:** `x1, x2, x3`
- `x2 = x1 + N(0, 0.1)`
- `x3 = 2*x1 + N(0, 0.1)`

**Pure noise features:** `x4, x5, x7, x8` — coefficients = 0.

**Additional useful independent feature:** `x6` — coefficient = 2.

---

## A3: Model Comparison & Regularisation

### Why standardisation is essential
Regularised models penalise coefficient magnitude. If features are on different
scales, those with larger raw values are penalised more heavily for a given
coefficient, distorting the optimisation. `StandardScaler` centres each feature
to zero mean and unit variance so that the penalty is applied fairly.

### Test-set performance (best models from GridSearchCV)

| Model | RMSE | MAE | Non-zero coefs |
|-------|------|-----|----------------|
| OLS | 1.0621 | 0.8392 | 8 |
| Ridge | 1.0718 | 0.8387 | 8 |
| Lasso | 1.0713 | 0.8406 | 3 |
| ElasticNet | 1.0720 | 0.8414 | 2 |

### Coefficient behaviour on correlated features
Because `x1, x2, x3` are nearly collinear, OLS assigns extreme and unstable
coefficients across splits (see boxplot).  Ridge shrinks all coefficients and
stabilises them; Lasso tends to select one or two from the group and zero the
rest.  Elastic Net balances the two behaviours via the `l1_ratio` mixing
parameter.

---

## A4: Forward Selection vs Lasso

**Forward Selection selected features**: ['x6', 'x2']

**Lasso non-zero features**: ['x2', 'x6', 'x8']

**Comparison**: Forward selection adds features greedily based on OLS CV RMSE;
it tends to include one representative from the correlated group plus the
independent signal (`x6`).  Lasso, driven by the L1 penalty, also selects a
sparse subset but may choose slightly different features because the L1 path
trades off all coefficients simultaneously rather than greedily.

