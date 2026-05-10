import numpy as np
from typing import Callable, Optional


class EllipticalSliceSampler:
    """
    Elliptical Slice Sampling (Murray, Adams, MacKay 2010).

    Samples from p(x) proportional to N(x; mu, Sigma) * L(x),
    where L(x) is a bounded likelihood (0 <= L(x) <= 1).

    Parameters
    ----------
    mu : np.ndarray, shape (d,)
        Mean of the Gaussian prior.
    Sigma : np.ndarray, shape (d, d)
        Covariance of the Gaussian prior.
    log_likelihood : callable
        Function mapping x (shape (d,)) to log L(x). Must satisfy log L(x) <= 0.
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.
    """

    def __init__(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        log_likelihood: Callable[[np.ndarray], float],
        rng: Optional[np.random.Generator] = None,
    ):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.log_likelihood = log_likelihood
        self.rng = rng if rng is not None else np.random.default_rng()
        self._chol = np.linalg.cholesky(self.Sigma)

    def _draw_prior(self) -> np.ndarray:
        """Draw nu ~ N(mu, Sigma)."""
        z = self.rng.standard_normal(self.mu.shape[0])
        return self.mu + self._chol @ z

    def step(self, x: np.ndarray) -> tuple:
        """
        One ESS iteration from current state x.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Current state.

        Returns
        -------
        x_new : np.ndarray, shape (d,)
            Next state (always accepted).
        n_evaluations : int
            Number of likelihood evaluations in this iteration (includes rejects).
        """
        nu = self._draw_prior()
        u = self.rng.uniform()
        log_y = self.log_likelihood(x) + np.log(u)

        theta = self.rng.uniform(0, 2 * np.pi)
        theta_min = theta - 2 * np.pi
        theta_max = theta

        x_centered = x - self.mu
        nu_centered = nu - self.mu

        n_evals = 0
        while True:
            x_prime = x_centered * np.cos(theta) + nu_centered * np.sin(theta) + self.mu
            n_evals += 1
            if self.log_likelihood(x_prime) > log_y:
                return x_prime, n_evals
            # Shrink bracket
            if theta < 0:
                theta_min = theta
            else:
                theta_max = theta
            theta = self.rng.uniform(theta_min, theta_max)

    def sample(
        self,
        x0: np.ndarray,
        n_samples: int,
        n_burnin: int = 0,
        thin: int = 1,
        verbose: bool = False,
    ) -> tuple:
        """
        Run ESS chain and collect samples.

        Parameters
        ----------
        x0 : np.ndarray, shape (d,)
            Initial state.
        n_samples : int
            Number of post-burnin samples to collect.
        n_burnin : int
            Number of initial samples to discard.
        thin : int
            Keep every thin-th sample.
        verbose : bool
            If True, show progress bar during sampling.

        Returns
        -------
        samples : np.ndarray, shape (n_samples, d)
        n_evaluations : np.ndarray, shape (n_samples,)
            Number of likelihood evaluations for each collected sample.
        """
        x = np.array(x0, dtype=float)
        samples = np.empty((n_samples, x.shape[0]))
        n_evals_list = []

        # Burn-in
        if verbose:
            from tqdm import tqdm
            pbar_burnin = tqdm(range(n_burnin), desc="Burn-in", disable=n_burnin == 0)
            for _ in pbar_burnin:
                x, _ = self.step(x)
        else:
            for _ in range(n_burnin):
                x, _ = self.step(x)

        # Collect samples
        if verbose:
            from tqdm import tqdm
            pbar = tqdm(range(n_samples * thin), desc="Sampling")
            idx = 0
            for i in pbar:
                x, n_evals = self.step(x)
                if (i + 1) % thin == 0:
                    samples[idx] = x
                    n_evals_list.append(n_evals)
                    idx += 1
        else:
            idx = 0
            for i in range(n_samples * thin):
                x, n_evals = self.step(x)
                if (i + 1) % thin == 0:
                    samples[idx] = x
                    n_evals_list.append(n_evals)
                    idx += 1

        return samples, np.array(n_evals_list)
