# Antithetic cross-validation

## Goal

Given a Gaussian or asymptotically Gaussian vector $T\sim N(\mu,\Sigma)$, create $K$ pairs of train-test data. Assume $\Sigma$ is known, $\mu$ is unknown.

## Method

Generate external random vectors $\omega^1,\ldots,\omega^K$, which are jointly Gaussian, with $\omega^k \sim N(0,\Sigma) $, $Cov(\omega^j, \omega^k)=\frac{-1}{K-1} \Sigma$ for $j\neq k$. This ensures that $\sum_{k=1}^K \omega^k = 0$.

For each $k=1,\ldots,K$, the $k$-th pair of train-test data is constructed as
```math
T^{\text{train},k}= T + \alpha \omega^k,\quad T^{\text{test}, k} = T-\frac{1}{\sqrt\alpha} \omega^k.
```

It's easy to see that
- $T^{\text{train},k}$ is independent of $T^{\text{test}, k} $ marginally.
- Averaging $T^{\text{train},k}$ (or $T^{\text{test}, k}$) over $k$ recoverse the original data $T$.

Then we can use these train-test pairs to do cross-validation: fit the predictor/estimator on training data, and evaluate on the test data, and average the test errors over the $K$ pairs.
