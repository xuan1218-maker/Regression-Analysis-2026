# Week 13: Theory Summary

## 1. Business Risk of Lasso with Highly Correlated Features

**Risk**: Lasso randomly selects one feature from a correlated group, 
leading to unstable results and misleading business interpretation.

**Elastic Net Solution**: Combines L1 and L2 penalties:
- L2 part encourages group effect (similar coefficients)
- L1 part still enables selection at group level

## 2. GridSearchCV vs Subjective Preferences

| Goal | GridSearchCV | Subjective |
|------|--------------|------------|
| Prediction accuracy | ✅ Direct | ❌ Indirect |
| Sparsity | ❌ No | ✅ Yes |
| Stability | ✅ Implicit | ✅ Yes |

## 3. Traditional Selection vs Lasso

| Aspect | Forward/Backward | Lasso |
|--------|------------------|-------|
| Efficiency | O(k²p) | O(kp) |
| Scalability | p < 1000 | p >> n |
| Stability | Path-dependent | Global optimum |
| Interpretability | Transparent process | Automatic |
