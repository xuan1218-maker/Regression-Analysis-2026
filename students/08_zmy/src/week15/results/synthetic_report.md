# Synthetic Data Report (Task A)

## A1. Data Generation Process (DGP)

- Sample size: 500
- Number of features: 4
- Effective features: x0 (positive effect, beta=2.0), x1 (negative effect, beta=-1.5)
- Noise features: x2, x3 (beta=0)
- DGP: 
  1. Linear predictor η = Xβ
  2. Probability p = 1/(1+exp(-η))
  3. y ~ Bernoulli(p)

## A3. Model Comparison

**LinearRegression output issues**: 
- Predictions range from negative to positive, not bounded in [0,1].
- Cannot be interpreted as probability.
- Hard thresholding at 0.5 ignores uncertainty.

**LogisticRegression output**: 
- Naturally bounded between 0 and 1.
- Directly interpretable as P(y=1|X).

## A4. Figure Explanation (fig_a4.png)

- X-axis: Standardized feature x0
- Y-axis: Model output (Linear Regression) or probability (Logistic Regression)
- Gray dots: True binary labels
- Blue line: Linear Regression predictions (unbounded)
- Red line: Logistic Regression probabilities (bounded in [0,1])

**Key observation**: Linear Regression produces values outside [0,1] and does not fit the binary pattern; Logistic Regression outputs smooth S-curve probabilities.

## A5. Core Questions

1. **Most unnatural aspect of Linear Regression**: Output is not constrained to [0,1]; assumes constant variance.
2. **Why Logistic Regression output is interpretable as probability**: Uses sigmoid to map any real number to (0,1) and is trained via maximum likelihood for Bernoulli data.
3. **Key distinction**: Not about "ability to classify" but **probabilistic meaning**. Logistic Regression gives calibrated probabilities; Linear Regression gives arbitrary scores.
