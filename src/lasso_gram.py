"""
Lasso solver using the Gram matrix G = X^T X and the moment vector T = X^T y.

The Lasso problem min_beta (1/(2n)) ||y - X beta||^2 + lam ||beta||_1
has gradient -(T - G beta)/n, so coordinate descent only needs G and T.
"""

import numpy as np


def soft_threshold(z, t):
    """Soft-thresholding: sign(z) * max(|z| - t, 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_gram(G, T, lam, n, max_iter=5000, tol=1e-4):
    """
    Solve the Lasso problem using coordinate descent on the Gram matrix.

    Minimizes: (1/(2n)) beta^T G beta - (1/n) T^T beta + lam ||beta||_1

    This is equivalent to min_beta (1/(2n)) ||y - X beta||^2 + lam ||beta||_1
    when G = X^T X and T = X^T y.

    Parameters
    ----------
    G : np.ndarray, shape (p, p)
        Gram matrix X^T X.
    T : np.ndarray, shape (p,)
        Moment vector X^T y.
    lam : float
        Regularization parameter.
    n : int
        Sample size (for scaling).
    max_iter : int
        Maximum number of coordinate descent passes.
    tol : float
        Convergence tolerance on the maximum coefficient change.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        Lasso solution.
    """
    p = G.shape[0]
    beta = np.zeros(p)

    for iteration in range(max_iter):
        max_change = 0.0
        for j in range(p):
            # Partial residual in the gradient: T_j - sum_{k != j} G_{jk} beta_k
            # = T_j - G[j,:] @ beta + G[j,j] * beta[j]
            r_j = T[j] - G[j] @ beta + G[j, j] * beta[j]

            # Coordinate update: beta_j = S(r_j / n, lam) / (G_jj / n)
            # = S(r_j, n * lam) / G_jj
            beta_new = soft_threshold(r_j, n * lam) / G[j, j]

            change = abs(beta_new - beta[j])
            if change > max_change:
                max_change = change
            beta[j] = beta_new

        if max_change < tol:
            break

    return beta
