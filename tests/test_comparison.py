"""
End-to-end test: ESS and rejection sampling should produce
samples from the same distribution.
"""

import numpy as np
from scipy import stats
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import gaussian_instance
from src.selector import LassoCVSelector
from src.ess import EllipticalSliceSampler
from src.rejection_sampler import RejectionSampler


def _setup_experiment(seed=42):
    """Set up a small experiment for testing."""
    rng = np.random.default_rng(seed)
    n, p, s = 100, 20, 5
    sigma = 1.0
    rho = 0.3
    signal = np.sqrt(2.0 * 2 * np.log(p))

    X, y, beta = gaussian_instance(rng, n, p, s, sigma, rho, signal)

    lambdas = np.logspace(-2, np.log10(5), 10) * np.sqrt(np.log(p)) / n
    selector = LassoCVSelector(
        X, y, sigma, lambdas, c_tau=2.0, alpha_cv=0.1, K_cv=5, rng=rng
    )
    lam, E, beta_hat_M = selector.select()
    d = E.sum()

    if d == 0:
        pytest.skip("No variables selected; cannot test.")

    return selector, lam, beta_hat_M, d, rng


class TestESSvsRejection:
    """ESS and rejection sampling should target the same distribution."""

    @pytest.mark.slow
    def test_marginal_distributions_match(self):
        """2-sample KS test on each coordinate."""
        selector, lam, beta_hat_M, d, rng = _setup_experiment(seed=42)
        beta_null = np.zeros(d)

        def log_lik(beta):
            return selector.log_selection_probability(beta, lam)

        # ESS samples
        ess = EllipticalSliceSampler(
            mu=beta_null,
            Sigma=selector.Sigma_M,
            log_likelihood=log_lik,
            rng=np.random.default_rng(1),
        )
        ess_samples = ess.sample(beta_hat_M, n_samples=2000, n_burnin=500, thin=2)

        # Rejection samples
        rej = RejectionSampler(selector)
        rej_samples, n_attempts = rej.sample(
            beta_null=beta_null,
            n_samples=500,
            max_attempts=50000,
            rng=np.random.default_rng(2),
        )

        n_accepted = rej_samples.shape[0]
        print(f"\nRejection: {n_accepted}/{n_attempts} accepted "
              f"(rate = {n_accepted / max(n_attempts, 1):.4f})")

        if n_accepted < 30:
            pytest.skip(f"Only {n_accepted} rejection samples; not enough for KS test.")

        # KS test on each coordinate
        for j in range(d):
            stat, pval = stats.ks_2samp(ess_samples[:, j], rej_samples[:, j])
            print(f"  Coord {j}: KS stat={stat:.4f}, p={pval:.4f}")
            # Use a lenient threshold since both are finite samples from MCMC
            assert pval > 0.001, (
                f"KS test failed for coord {j}: stat={stat:.4f}, p={pval:.4f}"
            )

    def test_ess_produces_valid_samples(self):
        """Basic sanity: ESS samples have finite values and reasonable spread."""
        selector, lam, beta_hat_M, d, rng = _setup_experiment(seed=42)
        beta_null = np.zeros(d)

        def log_lik(beta):
            return selector.log_selection_probability(beta, lam)

        ess = EllipticalSliceSampler(
            mu=beta_null,
            Sigma=selector.Sigma_M,
            log_likelihood=log_lik,
            rng=np.random.default_rng(1),
        )
        samples = ess.sample(beta_hat_M, n_samples=200, n_burnin=100)

        assert np.all(np.isfinite(samples)), "ESS produced non-finite samples"
        assert samples.shape == (200, d)
        # Samples should have nonzero variance
        for j in range(d):
            assert samples[:, j].std() > 1e-6, f"Coord {j} has zero variance"

    def test_rejection_sampler_produces_valid_samples(self):
        """Basic sanity for rejection sampler."""
        selector, lam, beta_hat_M, d, rng = _setup_experiment(seed=42)
        beta_null = np.zeros(d)

        rej = RejectionSampler(selector)
        rej_samples, n_attempts = rej.sample(
            beta_null=beta_null,
            n_samples=50,
            max_attempts=20000,
            rng=np.random.default_rng(2),
        )

        n_accepted = rej_samples.shape[0]
        print(f"\nRejection: {n_accepted}/{n_attempts} accepted")

        if n_accepted == 0:
            pytest.skip("No rejection samples obtained.")

        assert np.all(np.isfinite(rej_samples))
