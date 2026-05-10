# Project plan

## Step 1:

- [x] Read and understand files in docs/
- [x] Understand the previous github repo https://github.com/liusf15/transport_selinf which uses rejection sampling to sample $\hat\beta_M\mid \hat\lambda=\lambda$; our goal is to replace rejection sampling with elliptical slice sampling (ESS).

## Step 2: Implement ESS

- [x] Implement the ESS algorithm following `docs/ess.md`
- [x] Test on Gaussian targets to verify the correctness

## Step 3: Setup the lasso cross-validation experiment

- [x] Write functions to generate (X, y); run lasso with fixed lambda. Follow the same setup as in https://github.com/liusf15/transport_selinf/blob/main/experiments/lasso/
- [x] Implement antithetic CV for the lasso problem following `docs/antithetic_cv.md`; output $\lambda$ following the exponential mechanism described in `docs/problem.md`
- [x] Incorporate these into one selector class: input (X, y), output selected lambda and selected variable set M.

## Step 4: Sampling $p_{\beta_0}(\hat\beta_M\mid \hat\lambda=\lambda)$

- [x] Implement rejection sampling as described in `docs/problem.md`
- [x] Run ESS to sample this distribution as described in `docs/problem.md`
- [x] Compare the samples obtained by rejection sampling and ESS: the samples should have the same distribution.

## Step 5: Incorporating lasso selection probability

- [ ] Compute $P^{lasso}$ using the separation-of-variable method. 


