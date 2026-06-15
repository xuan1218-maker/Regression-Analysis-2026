# Task B — Real-World Dataset Report (Boston Housing)

## Data Source & Business Context

**Dataset**: Boston Housing (originally from the 1978 UCI ML Repository, also
available on Kaggle).  Downloaded from a public GitHub mirror:
`https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv`.

**Target variable**: `medv` — median value of owner-occupied homes in $1000s.
**Features**: 13 (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO,
B, LSTAT).

**Business goal**: Predict home prices to support valuation, investment, or
policy decisions.

## B1 & B2: Modelling Results

| Model | Test RMSE | Test MAE | Best Params | Zeroed Features |
|-------|-----------|----------|-------------|-----------------|
| OLS   | 4.6387 | 3.1627 | N/A | None |
| Ridge | 4.6810 | 3.1505 | {'alpha': 13.89495494373136} | [] |
| Lasso | 4.6574 | 3.1534 | {'alpha': 0.026826957952797246} | [] |
| ElasticNet | 4.6786 | 3.1508 | {'alpha': 0.0379269019073225, 'l1_ratio': 0.1} | [] |

## B3: Interpretation

### 1. Did regularisation significantly improve over OLS?
Ridge RMSE differed by only -0.042 from OLS, indicating the model is not severely over-fitting and/or the dataset is large enough for OLS to be stable. Lasso RMSE differed by only -0.019 from OLS, so the sparsity benefit came at little or no cost in predictive accuracy.

### 2. What did Lasso remove, and is it reasonable?
Lasso zeroed: **[]**.
No features were zeroed by Lasso.

### 3. Top 5 key factors
If a business stakeholder asked for the five most important predictors, I would
use the **Lasso** results (or Elastic Net, if Lasso is too aggressive) because:
- Lasso performs automatic feature selection with a principled CV-tuned
  threshold.
- The non-zero Lasso coefficients offer a parsimonious, interpretable list.
- OLS includes every variable regardless of relevance; forward selection can
  be greedy and unstable.  Lasso's simultaneous shrinkage provides a more
  robust variable-importance ranking.

---
