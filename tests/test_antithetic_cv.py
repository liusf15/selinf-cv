import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from legacy.antithetic_cv import (
    generate_antithetic_perturbations,
    antithetic_cv_prediction_error,
    reconstruct_T,
)
from src.selector import LassoCVSelector
from src.data_generation import gaussian_instance
from src.lasso_gram import lasso_gram


class TestAntitheticPerturbations:
    """Verify the antithetic perturbation vectors."""

    def test_sum_to_zero(self):
        K, d = 5, 3
        Sigma = np.eye(d)
        rng = np.random.default_rng(42)
        omegas = generate_antithetic_perturbations(K, d, Sigma, rng)
        assert omegas.shape == (K, d)
        np.testing.assert_allclose(omegas.sum(axis=0), 0, atol=1e-12)

    def test_sum_to_zero_nontrivial_cov(self):
        K, d = 10, 4
        A = np.random.default_rng(0).standard_normal((d, d))
        Sigma = A @ A.T + 0.1 * np.eye(d)
        rng = np.random.default_rng(42)
        omegas = generate_antithetic_perturbations(K, d, Sigma, rng)
        np.testing.assert_allclose(omegas.sum(axis=0), 0, atol=1e-10)

    def test_marginal_covariance(self):
        """Empirical covariance of omega^k should approximate Sigma."""
        K, d = 5, 3
        Sigma = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]])
        n_reps = 2000
        samples = []
        for i in range(n_reps):
            rng = np.random.default_rng(i)
            omegas = generate_antithetic_perturbations(K, d, Sigma, rng)
            samples.append(omegas[0])
        samples = np.array(samples)
        empirical_cov = np.cov(samples.T)
        np.testing.assert_allclose(empirical_cov, Sigma, atol=0.2)

    def test_cross_covariance(self):
        """Cov(omega^j, omega^k) should approximate -1/(K-1) * Sigma."""
        K, d = 5, 2
        Sigma = np.eye(d)
        n_reps = 3000
        samples_0 = []
        samples_1 = []
        for i in range(n_reps):
            rng = np.random.default_rng(i)
            omegas = generate_antithetic_perturbations(K, d, Sigma, rng)
            samples_0.append(omegas[0])
            samples_1.append(omegas[1])
        samples_0 = np.array(samples_0)
        samples_1 = np.array(samples_1)
        cross_cov = np.cov(samples_0.T, samples_1.T)[:d, d:]
        expected = -1 / (K - 1) * Sigma
        np.testing.assert_allclose(cross_cov, expected, atol=0.15)


class TestLassoGram:
    """Test the Gram-based Lasso solver."""

    def test_matches_sklearn(self):
        """lasso_gram should produce the same solution as sklearn Lasso."""
        from sklearn.linear_model import Lasso

        rng = np.random.default_rng(42)
        n, p = 100, 10
        X = rng.standard_normal((n, p))
        beta_true = np.array([1.0, -0.5, 0, 0, 0.3, 0, 0, 0, 0, 0])
        y = X @ beta_true + 0.5 * rng.standard_normal(n)

        G = X.T @ X
        T = X.T @ y
        lam = 0.1

        beta_gram = lasso_gram(G, T, lam, n)

        sk_lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=5000, tol=1e-6)
        sk_lasso.fit(X, y)

        np.testing.assert_allclose(beta_gram, sk_lasso.coef_, atol=1e-3)


class TestReconstructT:
    """Test T reconstruction from (beta_hat_M, A_obs)."""

    def test_reconstruction_matches_direct(self):
        """T reconstructed from (beta_hat_M, A_obs) should equal X^T y."""
        rng = np.random.default_rng(42)
        n, p = 50, 10
        X = rng.standard_normal((n, p))
        beta_true = np.zeros(p)
        beta_true[:3] = [1.0, -0.5, 0.3]
        y = X @ beta_true + 0.5 * rng.standard_normal(n)

        # Suppose M = {0, 1, 2}
        E = np.zeros(p, dtype=bool)
        E[:3] = True
        X_M = X[:, E]
        XtX_M = X_M.T @ X_M
        Xt_minus_M_X_M = X[:, ~E].T @ X_M

        beta_hat_M = np.linalg.solve(XtX_M, X_M.T @ y)
        residual = y - X_M @ beta_hat_M
        A_obs = X[:, ~E].T @ residual

        T_direct = X.T @ y
        T_reconstructed = reconstruct_T(beta_hat_M, A_obs, XtX_M, Xt_minus_M_X_M, E, p)

        np.testing.assert_allclose(T_reconstructed, T_direct, atol=1e-10)


class TestAntitheticCVPredictionError:
    """Test the CV prediction error computation with T reconstruction."""

    def test_runs_without_error(self):
        """Basic smoke test."""
        rng = np.random.default_rng(42)
        n, p = 50, 10
        X = rng.standard_normal((n, p))
        beta_true = np.zeros(p)
        beta_true[:3] = [1.0, -0.5, 0.3]
        y = X @ beta_true + 0.5 * rng.standard_normal(n)

        E = np.zeros(p, dtype=bool)
        E[:3] = True
        d = E.sum()
        X_M = X[:, E]
        XtX_M = X_M.T @ X_M
        Xt_minus_M_X_M = X[:, ~E].T @ X_M
        G = X.T @ X

        beta_hat_M = np.linalg.solve(XtX_M, X_M.T @ y)
        residual = y - X_M @ beta_hat_M
        A_obs = X[:, ~E].T @ residual

        Sigma_M = 0.25 * np.linalg.inv(XtX_M)
        omegas = generate_antithetic_perturbations(5, d, Sigma_M, rng)

        err = antithetic_cv_prediction_error(
            beta_hat_M=beta_hat_M,
            A_obs=A_obs,
            G=G,
            XtX_M=XtX_M,
            Xt_minus_M_X_M=Xt_minus_M_X_M,
            E=E,
            lam=0.1,
            alpha=0.1,
            omegas=omegas,
            n=n,
        )
        assert np.isfinite(err)


class TestExponentialMechanism:
    """Test the softmax selection probabilities."""

    def test_probabilities_sum_to_one(self):
        """Exponential mechanism probabilities should sum to 1."""
        rng = np.random.default_rng(42)
        n, p, s = 100, 20, 5
        signal = np.sqrt(0.6 * 2 * np.log(p))
        X, y, beta = gaussian_instance(rng, n, p, s, sigma=1.0, rho=0.5, signal=signal)
        lambdas = np.logspace(-2, np.log10(5), 10) * np.sqrt(np.log(p)) / n

        selector = LassoCVSelector(X, y, sigma=1.0, lambdas=lambdas, rng=rng)
        selector.select()

        log_probs = []
        for lam in lambdas:
            lp = selector.log_selection_probability(selector.beta_hat_M, lam)
            log_probs.append(lp)
        log_probs = np.array(log_probs)

        probs = np.exp(log_probs)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=0.05)

    def test_all_log_probs_nonpositive(self):
        """Log probabilities should be <= 0 (probabilities <= 1)."""
        rng = np.random.default_rng(42)
        n, p, s = 100, 20, 5
        signal = np.sqrt(0.6 * 2 * np.log(p))
        X, y, beta = gaussian_instance(rng, n, p, s, sigma=1.0, rho=0.5, signal=signal)
        lambdas = np.logspace(-2, np.log10(5), 10) * np.sqrt(np.log(p)) / n

        selector = LassoCVSelector(X, y, sigma=1.0, lambdas=lambdas, rng=rng)
        selector.select()

        for lam in lambdas:
            lp = selector.log_selection_probability(selector.beta_hat_M, lam)
            assert lp <= 0 + 1e-10, f"log prob = {lp} > 0 for lambda={lam}"

    def test_selector_selects_variables(self):
        """Selector should select at least one variable with strong signal."""
        rng = np.random.default_rng(42)
        n, p, s = 100, 20, 5
        signal = np.sqrt(2.0 * 2 * np.log(p))
        X, y, beta = gaussian_instance(rng, n, p, s, sigma=1.0, rho=0.3, signal=signal)
        lambdas = np.logspace(-2, np.log10(5), 10) * np.sqrt(np.log(p)) / n

        selector = LassoCVSelector(X, y, sigma=1.0, lambdas=lambdas, rng=rng)
        lam, E, beta_hat_M = selector.select()

        assert E.sum() > 0, "Should select at least one variable"
        assert beta_hat_M.shape[0] == E.sum()
        assert selector.Sigma_M.shape == (E.sum(), E.sum())
        assert selector.A_obs is not None
        assert selector.A_obs.shape[0] == (~E).sum()
