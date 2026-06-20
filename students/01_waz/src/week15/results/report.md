# Week 15 Report: Logistic Regression (GLM)

## Task A: Why OLS Fails for Binary Classification

| Model | Min prediction | Max prediction | Below 0 | Above 1 |
|---|---|---|---|---|
| OLS | -0.266 | 1.153 | 20 | 1 |
| Logistic | 0.010 | 0.983 | 0 | 0 |

OLS predicts values outside [0,1], which have no natural probability interpretation.
Logistic regression's sigmoid output always stays in (0,1).

## Task B: Sigmoid Function

  sigma(eta) = 1 / (1 + exp(-eta))

- Input eta (linear predictor) is unbounded
- Output is always in (0,1) -> can be interpreted as a probability
- eta = 0 -> p = 0.5 (equal odds for both classes)

## Task C: Bernoulli MLE & Log Loss

For binary response Y in {0,1} with Y ~ Bernoulli(p),
the likelihood is p^y * (1-p)^(1-y).

Taking negative log gives the log loss: -[y * log(p) + (1-y) * log(1-p)].

Log loss penalizes **confident mistakes** much more severely than MSE:
- If y=1 but model predicts p=0.01 -> log loss ~ 4.6 (very high)
- The same mistake under MSE would be (1-0.01)^2 ~ 0.98 (moderate)

## Task D: Log-odds and Coefficient Interpretation

The logistic regression model is linear in **log-odds**:

  log(p / (1-p)) = X * beta

Therefore:
- beta_j > 0 -> odds increase with x_j (positive association)
- beta_j < 0 -> odds decrease with x_j (negative association)
- exp(beta_j) = odds ratio: multiplicative change in odds per unit increase in x_j

## Task E: Classification Metrics

- **Accuracy**: fraction of correct predictions
- **Precision**: TP / (TP + FP) -- how many predicted positives are real
- **Recall**: TP / (TP + FN) -- how many actual positives are found
- **F1**: harmonic mean of precision and recall
- **Log Loss**: measures probability quality, not just hard classification

The threshold controls the tradeoff: raising it favors precision, lowering it favors recall.

## Task F: Regularized Logistic Regression

| Model | Test Acc. | Log Loss | Non-zero coefs |
|---|---|---|---|
| None (no reg) | 0.8200 | 0.4045 | 4 |
| L2 (C=0.1) | 0.8133 | 0.4088 | 4 |
| L2 (C=1.0) | 0.8200 | 0.4005 | 4 |
| L1 (C=0.1) | 0.8200 | 0.3975 | 3 |
| L1 (C=1.0) | 0.8200 | 0.4004 | 3 |

- **L1** drives coefficients to exactly zero -> variable selection
- **L2** shrinks all coefficients toward zero -> stabilization
- The noise feature x_noise has true coefficient = 0; L1 at C=0.1 should be most effective at zeroing it out
