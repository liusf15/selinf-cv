# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project description
Developing an efficient sampling algorithm (Elliptical Slice Sampling) to conduct post-selection inference for Lasso with cross-validated hyperparameters.

## Research Problem
See `docs/problem.md` for the full problem formulation and motivating example. See `docs/antithetic_cv.md` and `docs/ESS.md` for algorithmic details.

## Current Task
See `docs/tasks.md` for the ordered implementation plan and current status.

## Architecture

```
src/
├── ess.py              # EllipticalSliceSampler: MCMC for N(mu,Sigma)*f(theta)
├── lasso_gram.py       # lasso_gram(): coordinate descent using G=X^TX and T=X^Ty
├── selector.py         # LassoCVSelector: antithetic CV + exponential mechanism
├── rejection_sampler.py# RejectionSampler: baseline for comparison
└── data_generation.py  # gaussian_instance(): synthetic regression data
```

## Key design
- Selection operates on T=X^Ty, which can be reconstructed from (β̂_M, A_obs), where A_obs is the conditioned auxiliary variable. 
- The Lasso solver works directly from the Gram matrix X^TX and moment vector X^Ty, avoiding y-space operations.
- Antithetic perturbation omegas used during inference is **fixed**.

