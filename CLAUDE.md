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
├── antithetic_cv.py    # generate_antithetic_perturbations(), reconstruct_T(),
│                       # antithetic_cv_prediction_error()
├── selector.py         # LassoCVSelector: antithetic CV + exponential mechanism
├── rejection_sampler.py# RejectionSampler: baseline for comparison
└── data_generation.py  # gaussian_instance(): synthetic regression data
```

Key design: CV operates on β̂_M (d-dimensional sufficient statistic), reconstructing T=X^Ty from (β̂_M, A_obs) where A_obs is the conditioned auxiliary variable. The Lasso solver works directly from the Gram matrix X^TX and moment vector X^Ty, avoiding y-space operations.

## Commands

```bash
# Setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Run tests
.venv/bin/python -m pytest tests/ -v

# Run experiment
.venv/bin/python experiments/run_lassocv.py --seed 0
```

## Key Invariants

- Antithetic perturbation omegas used during ESS must be **fixed** (deterministic P^cv) — generated with `np.random.default_rng(0)` after selection. Stochastic omegas break MCMC detailed balance.
- Temperature τ is adaptive: `tau = c_tau * sum(psi(lambda))` with default `c_tau=2`.
- Alpha_cv defaults to 0.1 (controls train-test perturbation scale in antithetic CV).
