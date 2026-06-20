# Threshold & Loss Function Report (Task B & C)

## Task B: Bernoulli Likelihood and Log Loss

### B1. Three Essential Formulas

1. **Bernoulli distribution**:
   \[
   Y \sim \text{Bernoulli}(p)
   \]
   *Explanation*: Y takes value 1 with probability p and 0 with probability 1-p. Natural for binary outcomes.

2. **Single-sample likelihood**:
   \[
   L(p;y) = p^{y}(1-p)^{1-y}
   \]
   *Explanation*: When y=1, likelihood = p; when y=0, likelihood = 1-p.

3. **Negative log-likelihood (Log Loss)**:
   \[
   -\ln L(p;y) = -[y\ln p + (1-y)\ln(1-p)]
   \]
   *Explanation*: Minimizing this is equivalent to maximizing likelihood.

### B2. Loss Comparison Figure (fig_b2.png)

- X-axis: Predicted probability p
- Y-axis: Loss value
- Red solid: Log Loss (y=1); Red dashed: Log Loss (y=0)
- Blue solid: MSE (y=1); Blue dashed: MSE (y=0)

**Key insight**: When confidently wrong (e.g., predict p≈0 but true y=1), Log Loss → ∞, while MSE remains bounded. Log Loss heavily penalizes confident mistakes.

### B3. Discussion

1. **Why heavily penalize confident mistakes?** In critical decisions (medical diagnosis, fraud detection), confident errors can be catastrophic.
2. **Log loss from Bernoulli likelihood**: It derives directly from the negative logarithm of the Bernoulli likelihood.
3. **Why Log Loss over MSE?** MSE assumes Gaussian errors and symmetric loss; Log Loss respects [0,1] domain and Bernoulli nature.

## Task C: Confusion Matrix and Threshold Trade-offs

### C1. Basic Metrics (threshold = 0.5)

| Metric | Value |
|--------|-------|
| TP     | 65 |
| TN     | 62 |
| FP     | 10 |
| FN     | 13 |
| Accuracy | 0.8467 |
| Precision | 0.8667 |
| Recall | 0.8333 |
| F1     | 0.8497 |

### C2 & C3. Threshold Scan

Thresholds from 0.1 to 0.9, step 0.1.

**Figure (fig_c3.png)**:
- X-axis: Classification threshold
- Y-axis: Metric value
- Lines: Accuracy, Precision, Recall, F1

**Observations**:
- As threshold increases, recall decreases (fewer positives predicted).
- Precision often increases (higher confidence in predicted positives).
- Accuracy may peak at an intermediate threshold.
- F1 balances precision and recall.

**Trade-off**: Lower threshold → higher recall but lower precision; higher threshold → opposite.

### C4. Business Scenario: Credit Default Prediction

**Scenario**: Predict whether a borrower will default.

**Most important metric**: Recall (or F1). Missing a defaulter (FN) causes direct financial loss; false alarms (FP) cause customer dissatisfaction but lower cost. Accuracy can be misleading if default rate is low.

**Recommended threshold**: Choose threshold where recall is high enough (e.g., 80%) while maintaining acceptable precision. Use cost-benefit analysis: cost of FN (loss amount) vs cost of FP (lost business).
