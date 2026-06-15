# Task C — Theory & Practice Summary

## 1. Lasso's coefficient shrinkage and correlated groups

When features are highly correlated, Lasso tends to arbitrarily pick one
feature from the group and zero out the others.  In a business setting, this
can be risky: a manager reviewing the model might conclude that the dropped
features are irrelevant, when in fact they carry near-identical predictive
information.  For example, if both "years of education" and "degree level" are
candidates, Lasso might keep only one, obscuring a deeper behavioural
relationship.

**Elastic Net** mitigates this by mixing an L2 penalty with the L1 penalty.
The L2 component encourages coefficients within a correlated group to be
similar and shared, so Elastic Net is less likely to drop all but one.  The
`l1_ratio` parameter controls the balance, giving practitioners a knob to
trade off sparsity and stability.

---

## 2. GridSearchCV “best” vs subjective goals

`GridSearchCV` selects the hyper-parameters that minimise cross-validated
error (e.g., lowest RMSE).  This is an objective, data-driven criterion.
However, "sparser is better" or "more stable coefficients are better" are
often subjective goals driven by interpretability or deployment constraints.

The grid-search optimum may yield a model with many small non-zero
coefficients, whereas a business requirement for a 3-factor model would prefer
a higher alpha (more regularisation) at a small cost in accuracy.  Similarly,
the most stable model in repeated splits may not coincide with the lowest CV
error.  These trade-offs must be discussed with stakeholders and validated in
the deployment context.

---

## 3. Forward selection / backward elimination vs Lasso

**Computational efficiency**: Forward selection fits OLS repeatedly (O(k·p)
models for k selected features), while Lasso solves a convex optimisation once
over a path of alpha values.  For large p, Lasso is substantially faster
because its coordinate-descent solver scales well and does not require
refitting from scratch for each candidate.

**Final results**: Forward selection produces a hard inclusion/exclusion
decision, while Lasso yields continuous shrinkage.  Because forward selection
uses greedy OLS fits, it can be unstable under collinearity — small data
perturbations may change the feature order.  Lasso's simultaneous
regularisation is more principled for correlated data, but its arbitrary
selection within groups can be misleading.  In practice, I prefer Lasso (or
Elastic Net) for screening, then validate the selected feature set with domain
experts.

---
