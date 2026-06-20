# Summary: Logistic Regression and Binary Classification

## 1. Why Logistic Regression is not just "linear regression + sigmoid"

While mathematically it can be viewed that way, the crucial difference lies in the **objective function**. Linear regression minimizes squared error, which is inappropriate for binary data. Logistic regression maximizes Bernoulli likelihood (minimizes log loss), which respects the probabilistic nature of the output.

## 2. Relationship between sigmoid, Bernoulli likelihood, and log loss

- **Sigmoid**: Maps linear predictor η = Xβ to probability p = σ(η), ensuring p ∈ (0,1).
- **Bernoulli likelihood**: Describes the data-generating process for binary outcomes.
- **Log loss**: Negative log of Bernoulli likelihood; minimizing it = maximizing likelihood.

## 3. Why accuracy alone is insufficient

Accuracy treats all misclassifications equally and ignores class imbalance. A model that always predicts the majority class can have high accuracy but zero recall for the minority class. Precision, recall, F1, and ROC-AUC give a more nuanced view.

## 4. When to use L1 vs L2 logistic regression

- **L1 (Lasso)**: When feature selection is desired, sparse interpretable model.
- **L2 (Ridge)**: When prediction stability and handling multicollinearity are important.

## 5. Why Logistic Regression remains a strong baseline

- **Probabilistic outputs**: Well-calibrated probabilities for decision-making.
- **Interpretability**: Coefficients indicate direction and magnitude of feature influence.
- **Stability**: With L2 regularization, handles collinearity well.
- **Efficiency**: Fast to train and deploy.
- **Performance**: Often competitive with complex models as a baseline.
