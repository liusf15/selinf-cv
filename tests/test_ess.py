import numpy as np
from scipy import stats
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ess import EllipticalSliceSampler


class TestESSPureGaussian:
    """Target is pure Gaussian (L(x)=1, log L=0). ESS should recover N(mu, Sigma)."""

    def test_1d_mean_variance(self):
        mu = np.array([3.0])
        Sigma = np.array([[4.0]])
        sampler = EllipticalSliceSampler(
            mu=mu,
            Sigma=Sigma,
            log_likelihood=lambda x: 0.0,
            rng=np.random.default_rng(42),
        )
        samples = sampler.sample(x0=mu, n_samples=5000, n_burnin=500)
        assert abs(samples.mean() - 3.0) < 0.2
        assert abs(samples.var() - 4.0) < 0.5

    def test_2d_mean_covariance(self):
        mu = np.array([1.0, -2.0])
        Sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
        sampler = EllipticalSliceSampler(
            mu=mu,
            Sigma=Sigma,
            log_likelihood=lambda x: 0.0,
            rng=np.random.default_rng(42),
        )
        samples = sampler.sample(x0=mu, n_samples=5000, n_burnin=500)
        mean_err = np.abs(samples.mean(axis=0) - mu)
        assert np.all(mean_err < 0.2)
        cov_err = np.abs(np.cov(samples.T) - Sigma)
        assert np.all(cov_err < 0.3)

    def test_ks_marginals(self):
        mu = np.array([0.0, 0.0])
        Sigma = np.eye(2)
        sampler = EllipticalSliceSampler(
            mu=mu,
            Sigma=Sigma,
            log_likelihood=lambda x: 0.0,
            rng=np.random.default_rng(42),
        )
        samples = sampler.sample(x0=mu, n_samples=5000, n_burnin=500, thin=3)
        for j in range(2):
            stat, pval = stats.kstest(samples[:, j], "norm")
            assert pval > 0.005, f"KS test failed for marginal {j}: p={pval:.4f}"


class TestESSTruncatedGaussian:
    """Target is N(0,1) truncated to [0, inf). Known distribution."""

    def test_truncated_1d(self):
        mu = np.array([0.0])
        Sigma = np.array([[1.0]])

        def log_lik(x):
            return 0.0 if x[0] >= 0 else -np.inf

        sampler = EllipticalSliceSampler(
            mu=mu,
            Sigma=Sigma,
            log_likelihood=log_lik,
            rng=np.random.default_rng(42),
        )
        samples = sampler.sample(x0=np.array([1.0]), n_samples=5000, n_burnin=500)
        assert np.all(samples >= 0), "Samples should be non-negative"

        # Compare to truncated normal: mean = sqrt(2/pi) ~ 0.798, var = 1 - 2/pi ~ 0.363
        expected_mean = np.sqrt(2 / np.pi)
        expected_var = 1 - 2 / np.pi
        assert abs(samples.mean() - expected_mean) < 0.1
        assert abs(samples.var() - expected_var) < 0.1

        # KS test against truncnorm
        stat, pval = stats.kstest(
            samples[:, 0], stats.truncnorm(a=0, b=np.inf, loc=0, scale=1).cdf
        )
        assert pval > 0.01, f"KS test failed: p={pval:.4f}"


class TestESSGaussianTimesGaussian:
    """
    Prior N(0, I), likelihood proportional to N(mu2, Sigma2).
    Posterior is Gaussian with known closed-form mean and covariance.
    """

    def test_2d_posterior(self):
        d = 2
        mu_prior = np.zeros(d)
        Sigma_prior = np.eye(d)

        mu_lik = np.array([3.0, -1.0])
        Sigma_lik = np.array([[2.0, 0.3], [0.3, 1.5]])
        Sigma_lik_inv = np.linalg.inv(Sigma_lik)

        # Posterior: Sigma_post = (I + Sigma_lik_inv)^{-1}, mu_post = Sigma_post @ Sigma_lik_inv @ mu_lik
        Sigma_post = np.linalg.inv(np.eye(d) + Sigma_lik_inv)
        mu_post = Sigma_post @ Sigma_lik_inv @ mu_lik

        # Bounded log-likelihood: subtract the max so that L(x) <= 1
        log_norm_const = -0.5 * mu_lik @ Sigma_lik_inv @ mu_lik

        def log_lik(x):
            return -0.5 * x @ Sigma_lik_inv @ x + x @ Sigma_lik_inv @ mu_lik + log_norm_const

        sampler = EllipticalSliceSampler(
            mu=mu_prior,
            Sigma=Sigma_prior,
            log_likelihood=log_lik,
            rng=np.random.default_rng(42),
        )
        samples = sampler.sample(x0=mu_post, n_samples=10000, n_burnin=1000)

        mean_err = np.abs(samples.mean(axis=0) - mu_post)
        assert np.all(mean_err < 0.1), f"Mean error: {mean_err}"

        cov_err = np.abs(np.cov(samples.T) - Sigma_post)
        assert np.all(cov_err < 0.15), f"Cov error:\n{cov_err}"
