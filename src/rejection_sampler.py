import numpy as np
from typing import Optional

from src.selector import LassoCVSelector


class RejectionSampler:
    """
    Rejection sampling for p(beta_hat_M | lambda_hat = lambda).

    Target: p(beta) proportional to N(beta; beta_null, Sigma_M) * P^cv(lambda | beta)

    Algorithm:
    1. Draw beta^* ~ N(beta_null, Sigma_M)
    2. Compute f = P^cv(lambda_selected | beta^*)  (in [0, 1])
    3. Draw u ~ Uniform(0, 1)
    4. If u < f, accept beta^*

    This uses the same deterministic P^cv (fixed omegas) as the ESS sampler,
    so both methods target the same distribution.

    Parameters
    ----------
    selector : LassoCVSelector
        Fitted selector (select() must have been called).
    """

    def __init__(self, selector: LassoCVSelector):
        self.selector = selector
        self.Sigma_M = selector.Sigma_M
        self._chol = np.linalg.cholesky(self.Sigma_M)

    def sample(
        self,
        beta_null: np.ndarray,
        n_samples: int,
        max_attempts: int = 100000,
        rng: Optional[np.random.Generator] = None,
        verbose: bool = False,
    ):
        """
        Collect n_samples via rejection sampling.

        Parameters
        ----------
        beta_null : np.ndarray, shape (d,)
            Null hypothesis value of beta_M.
        n_samples : int
            Number of accepted samples to collect.
        max_attempts : int
            Maximum number of proposals before stopping.
        rng : np.random.Generator, optional
            Random number generator. If None, uses default.
        verbose : bool
            If True, show progress bar during sampling.

        Returns
        -------
        samples : np.ndarray, shape (n_accepted, d)
            Accepted samples. May have fewer than n_samples rows
            if max_attempts is reached.
        n_attempts : int
            Total number of proposals made.
        """
        if rng is None:
            rng = np.random.default_rng()

        d = beta_null.shape[0]
        samples = []
        n_attempts = 0

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples, desc="Rejection sampling")

        while len(samples) < n_samples and n_attempts < max_attempts:
            # Propose from the prior: beta^* ~ N(beta_null, Sigma_M)
            z = rng.standard_normal(d)
            beta_star = beta_null + self._chol @ z

            # Compute acceptance probability: P^cv(lambda | beta^*)
            log_f = self.selector.log_selection_probability(
                beta_star, self.selector.lam_selected
            )
            f = np.exp(log_f)

            # Accept/reject
            u = rng.uniform()
            if u < f:
                samples.append(beta_star)
                if verbose:
                    pbar.update(1)

            n_attempts += 1

        if verbose:
            pbar.close()

        if len(samples) == 0:
            return np.empty((0, d)), n_attempts

        return np.array(samples), n_attempts
