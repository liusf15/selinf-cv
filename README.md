# Selective Inference with Cross-Validated Lasso

Efficient sampling via Elliptical Slice Sampling (ESS) for post-selection inference with Lasso using cross-validated hyperparameters.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the main experiment:

```bash
python experiments/run_lassocv.py --seed 0
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Repository Structure

```
src/
├── ess.py               # Elliptical Slice Sampler (MCMC for Gaussian * likelihood)
├── lasso_gram.py        # Coordinate descent Lasso using Gram matrix X^TX
├── selector.py          # LassoCVSelector: antithetic CV + exponential mechanism
├── rejection_sampler.py # Rejection sampler baseline for comparison
└── data_generation.py   # Synthetic Gaussian regression data generation

experiments/
└── run_lassocv.py       # Main experiment script

tests/
├── test_ess.py          # Tests for elliptical slice sampler
├── test_antithetic_cv.py# Tests for antithetic CV components
└── test_comparison.py   # Comparison tests between methods

docs/                    # Problem formulation and algorithm details
```
