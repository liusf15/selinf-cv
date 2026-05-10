import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import Lasso
from typing import Optional



class LassoCVSelector:
    """
    Selects lambda via antithetic CV with exponential mechanism,
    then runs Lasso to select variables.

    The selection proceeds as follows:
    1. For each candidate lambda, run Lasso(lambda) on (X, y) to get M(lambda).
       Compute beta_hat_M(lambda), A_obs(lambda), and run antithetic CV
       (perturbing beta_hat_M in d-dimensional space, reconstructing T = X^T y
       from (beta_hat_M, A_obs)) to get prediction error psi(lambda).
    2. Select lambda via exponential mechanism (softmax of -psi/tau).
    3. Fix A_obs and antithetic perturbations for deterministic P^cv
       evaluation during ESS inference.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Full design matrix.
    y : np.ndarray, shape (n,)
        Response vector.
    sigma : float
        Known noise standard deviation.
    lambdas : np.ndarray, shape (n_lambda,)
        Grid of candidate regularization parameters.
    c_tau : float
        Temperature scaling: tau = c_tau * sum(psi(lambda)).
    alpha_cv : float
        Splitting ratio for antithetic CV train-test construction.
    K_cv : int
        Number of antithetic CV folds.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sigma: float,
        lambdas: np.ndarray,
        c_tau: float = 2.0,
        alpha_cv: float = 0.1,
        K_cv: int = 10,
        rng: Optional[np.random.Generator] = None,
    ):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sigma = sigma
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.c_tau = c_tau
        self.alpha_cv = alpha_cv
        self.K_cv = K_cv
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n, self.p = self.X.shape

        # Precompute Gram matrix
        self.G = self.X.T @ self.X  # (p, p)

        # Generate fixed p-dimensional omegas for antithetic CV (used in both selection and inference)
        # These are generated ONCE with fixed seed and reused for all evaluations
        omega_rng = np.random.default_rng(0)
        self._omegas = self._generate_antithetic_omegas_p(omega_rng)  # shape (K_cv, p)

        # These are set after select() is called
        self.lam_selected = None
        self.E = None  # boolean active set indicator, shape (p,)
        self.beta_hat_M = None  # OLS on active set, shape (d,)
        self.X_M = None  # X[:, E], shape (n, d)
        self.A_obs = None  # auxiliary variable, shape (p - d,)
        self._Sigma_M = None
        self._XtX_M = None  # X_M^T X_M, shape (d, d)
        self._Xt_minus_M_X_M = None  # X_{-M}^T X_M, shape (p-d, d)

    def _generate_antithetic_omegas_p(self, rng):
        """
        Generate K_cv antithetic perturbation vectors in R^p (full dimension).

        These are p-dimensional perturbations that will be applied directly to T = X'y.
        Generated with fixed seed so that P^cv is deterministic.

        Parameters
        ----------
        rng : np.random.Generator
            RNG with fixed seed (typically seed=0)

        Returns
        -------
        omegas : np.ndarray, shape (K_cv, p)
            K_cv perturbation vectors in p-dimensional space
        """
        K = self.K_cv
        d = self.p

        # Generate K independent gaussian vectors, center, and rescale
        W = rng.standard_normal((K, d))
        W = W - W.mean(axis=0, keepdims=True)
        W = W * np.sqrt(K / (K - 1))

        return W

    def _reconstruct_T_general(self, beta_hat_M, A_obs, XtX_M, Xt_minus_M_X_M, E):
        """
        Reconstruct T = X'y from beta_hat_M and A_obs (general version).

        T[E]   = X_M^T X_M @ beta_hat_M
        T[~E]  = X_{-M}^T X_M @ beta_hat_M + A_obs

        Parameters
        ----------
        beta_hat_M : np.ndarray, shape (d,)
            Sufficient statistic (OLS estimate on active set)
        A_obs : np.ndarray, shape (p - d,)
            Observed auxiliary variable
        XtX_M : np.ndarray, shape (d, d)
            X_M^T X_M
        Xt_minus_M_X_M : np.ndarray, shape (p - d, d)
            X_{-M}^T X_M
        E : np.ndarray, shape (p,), bool
            Active set indicator

        Returns
        -------
        T : np.ndarray, shape (p,)
            Reconstructed X'y
        """
        T = np.zeros(self.p)
        T[E] = XtX_M @ beta_hat_M
        T[~E] = Xt_minus_M_X_M @ beta_hat_M + A_obs
        return T

    def _reconstruct_T(self, beta_hat_M):
        """
        Reconstruct T = X'y from beta_hat_M and A_obs (using selected M).

        Uses stored active set E, A_obs, XtX_M, Xt_minus_M_X_M from selection.

        Parameters
        ----------
        beta_hat_M : np.ndarray, shape (d,)
            Sufficient statistic (OLS estimate on active set)

        Returns
        -------
        T : np.ndarray, shape (p,)
            Reconstructed X'y
        """
        return self._reconstruct_T_general(
            beta_hat_M, self.A_obs, self._XtX_M, self._Xt_minus_M_X_M, self.E
        )

    def _antithetic_cv_T(self, T, lam):
        """
        Compute antithetic CV prediction error for a given T and lambda.

        Applies fixed p-dimensional perturbations directly to T.

        Parameters
        ----------
        T : np.ndarray, shape (p,)
            Moment vector X'y
        lam : float
            Regularization parameter for Lasso

        Returns
        -------
        prediction_error : float
            Average prediction error across K_cv folds
        """
        from src.lasso_gram import lasso_gram

        total_error = 0.0

        for k in range(self.K_cv):
            omega_k = self._omegas[k]  # shape (p,)

            # Perturb T directly
            T_train = T + self.alpha_cv * omega_k
            T_test = T - omega_k / np.sqrt(self.alpha_cv)

            # Fit Lasso using Gram matrix
            beta_fit = lasso_gram(self.G, T_train, lam, self.n)

            # Prediction error
            pred_error = (beta_fit @ self.G @ beta_fit - 2 * T_test @ beta_fit) / self.n
            total_error += pred_error

        return total_error / self.K_cv

    def select(self):
        """
        Run selection: pick lambda via exponential mechanism, run Lasso,
        identify active set M.

        Returns
        -------
        lam_selected : float
            Selected regularization parameter.
        E : np.ndarray, shape (p,), bool
            Active set indicator.
        beta_hat_M : np.ndarray, shape (d,)
            OLS estimate restricted to active set.
        """
        # Score each lambda using antithetic CV
        scores = self._compute_cv_scores_all_lambdas()

        # Adaptive temperature
        self.tau = self.c_tau * np.sum(scores)

        # Select lambda via exponential mechanism (softmax)
        log_probs = -scores / self.tau
        log_probs -= logsumexp(log_probs)
        probs = np.exp(log_probs)
        idx = self.rng.choice(len(self.lambdas), p=probs)
        self.lam_selected = self.lambdas[idx]

        # Set M, beta_hat_M, A_obs for the selected lambda
        self._set_active_set(self.lam_selected)

        return self.lam_selected, self.E, self.beta_hat_M

    def _set_active_set(self, lam):
        """Run Lasso with given lambda and compute M, beta_hat_M, A_obs."""
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=5000)
        lasso.fit(self.X, self.y)
        self.E = np.abs(lasso.coef_) > 1e-10
        d = self.E.sum()

        if d == 0:
            raise ValueError("Lasso selected no variables. Try smaller lambda values.")

        self.X_M = self.X[:, self.E]
        self._XtX_M = self.X_M.T @ self.X_M
        self._Xt_minus_M_X_M = self.X[:, ~self.E].T @ self.X_M

        # OLS on active set: beta_hat_M = (X_M^T X_M)^{-1} X_M^T y
        self.beta_hat_M = np.linalg.solve(self._XtX_M, self.X_M.T @ self.y)

        # Covariance: Sigma_M = sigma^2 * (X_M^T X_M)^{-1}
        self._Sigma_M = self.sigma ** 2 * np.linalg.inv(self._XtX_M)

        # Auxiliary variable: A = X_{-M}^T (I - P_M) y
        # Since (I - P_M) y = y - X_M beta_hat_M:
        residual = self.y - self.X_M @ self.beta_hat_M
        self.A_obs = self.X[:, ~self.E].T @ residual

    def _compute_cv_scores_all_lambdas(self):
        """
        Compute antithetic CV prediction errors for all lambdas.

        For each lambda:
        1. Run Lasso to get M(lambda)
        2. Compute beta_hat_M(lambda) and A_obs(lambda)
        3. Reconstruct T from (beta_hat_M, A_obs)
        4. Evaluate antithetic CV in T space using fixed omegas

        Returns
        -------
        scores : np.ndarray, shape (n_lambda,)
        """
        scores = np.zeros(len(self.lambdas))

        for i, lam in enumerate(self.lambdas):
            # Run Lasso with this lambda to get M(lambda)
            lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=5000, tol=1e-3)
            lasso.fit(self.X, self.y)
            E_lam = np.abs(lasso.coef_) > 1e-10
            d_lam = E_lam.sum()

            if d_lam == 0:
                # No variables selected: assign a large prediction error
                scores[i] = np.sum(self.y ** 2) / self.n
                continue

            X_M_lam = self.X[:, E_lam]
            XtX_M_lam = X_M_lam.T @ X_M_lam
            Xt_minus_M_X_M_lam = self.X[:, ~E_lam].T @ X_M_lam

            # OLS estimate and auxiliary variable for this M(lambda)
            beta_hat_M_lam = np.linalg.solve(XtX_M_lam, X_M_lam.T @ self.y)
            residual_lam = self.y - X_M_lam @ beta_hat_M_lam
            A_obs_lam = self.X[:, ~E_lam].T @ residual_lam

            # Reconstruct T from (beta_hat_M_lam, A_obs_lam) using this lambda's active set
            T_lam = self._reconstruct_T_general(
                beta_hat_M_lam, A_obs_lam, XtX_M_lam, Xt_minus_M_X_M_lam, E_lam
            )

            # Compute CV prediction error using fixed p-dimensional omegas
            scores[i] = self._antithetic_cv_T(T_lam, lam)

        return scores

    def compute_cv_scores(self, beta_hat_M):
        """
        Compute antithetic CV prediction errors for all lambdas,
        given beta_hat_M and the fixed A_obs.

        Reconstructs T from (beta_hat_M, A_obs) and evaluates CV scores
        using fixed p-dimensional omegas for deterministic evaluation.

        Parameters
        ----------
        beta_hat_M : np.ndarray, shape (d,)
            Sufficient statistic at the selected active set M

        Returns
        -------
        scores : np.ndarray, shape (n_lambda,)
        """
        # Reconstruct T from beta_hat_M using selected M and A_obs
        T = self._reconstruct_T(beta_hat_M)

        # Compute CV scores for all lambdas using the same fixed omegas
        scores = np.zeros(len(self.lambdas))
        for i, lam in enumerate(self.lambdas):
            scores[i] = self._antithetic_cv_T(T, lam)

        return scores

    def log_selection_probability(self, beta_hat_M, lam):
        """
        Compute log P^cv(lambda_hat = lam | beta_hat_M, A_obs).

        This is the f(theta) used as the ESS likelihood. It is bounded in [0, 1]
        because it's a probability.

        Parameters
        ----------
        beta_hat_M : np.ndarray, shape (d,)
        lam : float

        Returns
        -------
        log_prob : float
            log P^cv, satisfying log_prob <= 0.
        """
        scores = self.compute_cv_scores(beta_hat_M)

        # Adaptive tau based on current scores
        tau = self.c_tau * np.sum(scores)

        if tau <= 0 or not np.isfinite(tau):
            return -np.log(len(self.lambdas))

        idx = np.argmin(np.abs(self.lambdas - lam))
        log_numerator = -scores[idx] / tau
        log_denominator = logsumexp(-scores / tau)
        return log_numerator - log_denominator

    @property
    def Sigma_M(self):
        """Covariance of beta_hat_M: sigma^2 * (X_M^T X_M)^{-1}."""
        return self._Sigma_M
