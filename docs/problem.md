# Selective inference with cross-validated hyperparameters

## Motivating example: Lasso with cross-validated $\lambda$

We use CV to select a regularization parameter $\lambda$, and then run lasso to select a variable set $M$. We want to do inference for the selected variables $\beta_j,j\in M$. We do inference bsaed on the distribution of $\hat\beta_M=(X_M^\top X_M)^{-1} X_M^\top y$ conditional on the selection event $\{\hat\lambda=\lambda, \hat M^{\lambda} = M \}$. Here, $\hat\lambda$ denote the output of CV, and $\hat M^{\lambda}$ denote the output of lasso (signed) variable selection with fixed $\lambda$. Additionally, we need to condition on $A=X_{-M}^\top (I_n- X_M (X_M^\top X_M)^{-1} X_M^\top) y $ to get rid of nuisance parameters.

Under $\beta_0$, the conditional density of $\hat\beta_M$ is proportional to
$$
p_{\beta_0}(\hat\beta_M\mid \lambda,M,A) \propto \varphi(\hat\beta_M;\beta_0,\Sigma_M)\cdot P^{cv}(\hat\lambda=\lambda\mid \hat\beta_M,A ) \cdot P^{lasso}(\hat M^{\lambda}=M\mid \hat\beta_M,A).
$$

- $\varphi(\hat\beta_M;\beta_0,\Sigma_M)$ is the density of $N(\beta_0,\Sigma_M=\sigma^2(X_M^\top X_M)^{-1} )$
- P^{cv} is the probability of selecting $\lambda$ using CV
- P^{lasso} is the lasso selection probability.

There are efficient ways to compute $P^{lasso}$, since lasso selection event is a polyhedron.

So the main challenge is to sample or approximate the distribution 
$$
p_{\beta_0}(\hat\beta_M\mid \hat\lambda=\lambda,A) \propto \varphi(\hat\beta_M;\beta_0,\Sigma_M)\cdot P^{cv}(\hat\lambda=\lambda\mid \hat\beta_M ,A).
$$

## Previous approach: rejection sampling

Assume $\hat\lambda$ is a black-box selection procedure and we can repeatedly run it on generated data. We generate synthetic data $y^{(b)}$ from the null (conditioned on the linear constraint for $A$), let $\lambda^{(b)}=\hat\lambda(y^{(b)})$ and $\hat\beta_M^{(b)}=(X_M^\top X_M)^{-1} X_M^\top y^{(b)}$. If $\lambda^{(b)}=\lambda$, accept $\hat\beta_M^{(b)}$ as a sample. Once we get enough samples, we fit a density estimation algorithm on them.

## Proposed new sampling approach: MCMC

Rejection sampling can be inefficient when the probability of selecting $\lambda$ is low. We propose to use MCMC to sample $p_{\beta_0}(\hat\beta_M\mid \hat\lambda=\lambda, A)$. Because this is proportional to the product of a Gaussian density and $P^{cv}(\hat\lambda=\lambda\mid \hat\beta_M, A)$, a suitable MCMC algorithm is the elliptical slice sampling; see @docs/ESS.md

## Description of $P^{cv}$

### Antithetic CV

We are assuming $\hat\lambda$ is a function of $X^\top y$, which can be recovered from $\hat\beta_M,A$. So we consider the antithetic CV in @docs/antithetic_cv.md 

It uses the normality (or asymptotic normality if not exactly Gaussian linear model) to construct train-test pairs. For each $\lambda$, it outputs the estimated prediction error. 

### Exponential mechanism

Suppose $\Lambda$ is the set of candidate of values of $\lambda$. For each $\lambda\in\Lambda$, we use antithetic CV and get a score (i.e. prediction error) $\psi(\lambda)=\psi(\lambda; \hat\beta_M, A)$.

We then select $\lambda$ with probability equal to $\text{SoftMax}(\psi; \tau)$, i.e.
$$
P^{cv}(\hat\lambda=\lambda \mid \hat\beta_M) = \frac{\exp[-\frac{1}{\tau} \psi(\lambda) ]}{\sum_{\lambda'\in\Lambda}\exp[-\frac{1}{\tau} \psi(\lambda') ]}
$$

Thus, $P^{cv}$ can be evaluated as a function of $\hat\beta_M$.

