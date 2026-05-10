"""
Main experiment: compare ESS and rejection sampling for
p(beta_hat_M | lambda_hat = lambda).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_generation import gaussian_instance
from src.selector import LassoCVSelector
from src.ess import EllipticalSliceSampler
from src.rejection_sampler import RejectionSampler


def run_experiment(
    seed=0,
    n=100,
    p=20,
    s=5,
    sigma=1.0,
    rho=0.5,
    signal_fac=0.6,
    n_lambda=10,
    c_tau=2.0,
    alpha_cv=0.1,
    K_cv=10,
    n_ess_samples=2000,
    n_ess_burnin=500,
    n_rejection_samples=2000,
    max_rejection_attempts=100000,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    # 1. Generate data
    print(f"Generating data: n={n}, p={p}, s={s}, sigma={sigma}, rho={rho}")
    X, y, beta_true = gaussian_instance(
        rng, n, p, s, sigma, rho, signal, equicorrelated=True
    )

    # 2. Run selection
    lambdas = np.logspace(-2, np.log10(5), n_lambda) * np.sqrt(np.log(p)) / n
    selector = LassoCVSelector(
        X, y, sigma, lambdas, c_tau=c_tau, alpha_cv=alpha_cv, K_cv=K_cv, rng=rng
    )
    lam_selected, E, beta_hat_M = selector.select()
    d = E.sum()
    print(f"Selected lambda = {lam_selected:.4f}, |M| = {d}")
    print(f"Active set: {np.where(E)[0]}")
    print(f"beta_hat_M = {beta_hat_M}")

    # 3. ESS sampling
    print(f"\nRunning ESS: {n_ess_samples} samples, {n_ess_burnin} burn-in")
    beta_null = np.zeros(d)

    def log_lik(beta):
        return selector.log_selection_probability(beta, lam_selected)

    ess = EllipticalSliceSampler(
        mu=beta_null,
        Sigma=selector.Sigma_M,
        log_likelihood=log_lik,
        rng=rng,
    )
    ess_samples, n_evals = ess.sample(beta_hat_M, n_ess_samples, n_burnin=n_ess_burnin, verbose=verbose)
    print(f"ESS sample mean: {ess_samples.mean(axis=0)}")
    print(f"ESS sample std:  {ess_samples.std(axis=0)}")

    # Acceptance rates: 1 / n_evals
    acceptance_rates = 1.0 / n_evals
    print(f"\nESS Acceptance Rates:")
    print(f"  Mean acceptance rate: {acceptance_rates.mean():.4f}")
    print(f"  Std acceptance rate:  {acceptance_rates.std():.4f}")
    print(f"  Min acceptance rate:  {acceptance_rates.min():.4f}")
    print(f"  Max acceptance rate:  {acceptance_rates.max():.4f}")

    # 4. Rejection sampling
    print(f"\nRunning rejection sampling: target {n_rejection_samples} samples")
    rej = RejectionSampler(selector)
    rej_samples, n_attempts = rej.sample(
        beta_null=beta_null,
        n_samples=n_rejection_samples,
        max_attempts=max_rejection_attempts*n_rejection_samples,
        rng=rng,
        verbose=verbose,
    )
    n_accepted = rej_samples.shape[0]
    print(f"Rejection sampling: {n_accepted}/{n_attempts} accepted "
          f"(rate = {n_accepted / n_attempts:.4f})")
    if n_accepted > 0:
        print(f"Rejection sample mean: {rej_samples.mean(axis=0)}")
        print(f"Rejection sample std:  {rej_samples.std(axis=0)}")

    # 5. Compare
    results = compare_samples(ess_samples, rej_samples, d)
    results["ess_samples"] = ess_samples
    results["rej_samples"] = rej_samples
    results["n_evals"] = n_evals
    results["acceptance_rates"] = acceptance_rates
    results["rej_acceptance_rate"] = n_accepted / n_attempts if n_attempts > 0 else 0.0
    results["rej_n_accepted"] = n_accepted
    results["rej_n_attempts"] = n_attempts
    return results


def compare_samples(ess_samples, rej_samples, d):
    """Compare ESS and rejection samples via KS test and summary statistics."""
    results = {}
    n_rej = rej_samples.shape[0]

    if n_rej < 10:
        print("\nToo few rejection samples for comparison.")
        return results

    print(f"\n{'='*60}")
    print("Comparison: ESS vs Rejection Sampling")
    print(f"{'='*60}")
    print(f"ESS samples: {ess_samples.shape[0]}, Rejection samples: {n_rej}")

    results["ess_mean"] = ess_samples.mean(axis=0)
    results["rej_mean"] = rej_samples.mean(axis=0)
    results["ess_std"] = ess_samples.std(axis=0)
    results["rej_std"] = rej_samples.std(axis=0)

    return results


def plot_comparison(ess_samples, rej_samples, d, save_path=None):
    """QQ plots comparing ESS and rejection marginals."""
    fig, axes = plt.subplots(1, d, figsize=(4 * d, 4))
    if d == 1:
        axes = [axes]
    for j in range(d):
        ax = axes[j]
        ess_q = np.sort(ess_samples[:, j])
        rej_q = np.sort(rej_samples[:min(len(rej_samples), len(ess_samples)), j])
        n_q = min(len(ess_q), len(rej_q))
        ess_q = np.quantile(ess_samples[:, j], np.linspace(0, 1, n_q))
        rej_q = np.quantile(rej_samples[:, j], np.linspace(0, 1, n_q))
        ax.scatter(rej_q, ess_q, s=5, alpha=0.5)
        lims = [min(rej_q.min(), ess_q.min()), max(rej_q.max(), ess_q.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"Rejection (coord {j})")
        ax.set_ylabel(f"ESS (coord {j})")
        ax.set_title(f"QQ Plot: Coord {j}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESS vs Rejection Sampling for Lasso CV")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--signal_fac", type=float, default=0.6)
    parser.add_argument("--n_lambda", type=int, default=10)
    parser.add_argument("--c_tau", type=float, default=1.0)
    parser.add_argument("--alpha_cv", type=float, default=0.1)
    parser.add_argument("--K_cv", type=int, default=10)
    parser.add_argument("--n_ess", type=int, default=500)
    parser.add_argument("--n_rej", type=int, default=500)
    parser.add_argument("--max_attempts", type=int, default=100)
    parser.add_argument("--plot", action="store_true", help="Save QQ plot to results folder")
    parser.add_argument("--progress", action="store_true", help="Show progress bar during ESS sampling")
    args = parser.parse_args()

    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    results = run_experiment(
        seed=args.seed,
        n=args.n,
        p=args.p,
        s=args.s,
        sigma=args.sigma,
        rho=args.rho,
        signal_fac=args.signal_fac,
        n_lambda=args.n_lambda,
        c_tau=args.c_tau,
        alpha_cv=args.alpha_cv,
        K_cv=args.K_cv,
        n_ess_samples=args.n_ess,
        n_ess_burnin=0,
        n_rejection_samples=args.n_rej,
        max_rejection_attempts=args.max_attempts,
        verbose=args.progress,
    )

    # Save results if plot flag is set
    if args.plot and "ess_samples" in results and "rej_samples" in results:
        ess_samples = results["ess_samples"]
        rej_samples = results["rej_samples"]
        d = ess_samples.shape[1]

        plot_path = os.path.join(results_dir, f"qqplot_seed{args.seed}.png")
        plot_comparison(ess_samples, rej_samples, d, save_path=plot_path)

    # Save acceptance rate summary
    if "acceptance_rates" in results:
        summary_path = os.path.join(results_dir, f"acceptance_summary_seed{args.seed}.txt")
        with open(summary_path, "w") as f:
            f.write("Acceptance Rates: ESS vs Rejection Sampling\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Seed: {args.seed}\n\n")

            # ESS statistics
            f.write("Elliptical Slice Sampling (ESS)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Mean acceptance rate: {results['acceptance_rates'].mean():.6f}\n")
            f.write(f"Std acceptance rate:  {results['acceptance_rates'].std():.6f}\n")
            f.write(f"Min acceptance rate:  {results['acceptance_rates'].min():.6f}\n")
            f.write(f"Max acceptance rate:  {results['acceptance_rates'].max():.6f}\n")
            f.write(f"Number of samples:   {len(results['acceptance_rates'])}\n\n")

            # Rejection sampling statistics
            f.write("Rejection Sampling\n")
            f.write("-" * 60 + "\n")
            f.write(f"Acceptance rate:     {results.get('rej_acceptance_rate', 0.0):.6f}\n")
            f.write(f"Samples accepted:    {results.get('rej_n_accepted', 0)}\n")
            f.write(f"Total attempts:      {results.get('rej_n_attempts', 0)}\n")
        print(f"\nAcceptance rate summary saved to {summary_path}")
