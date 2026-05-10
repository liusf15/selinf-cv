# Elliptical Slice Sampling (ESS)

Elliptical Slice Sampling is an MCMC algorithm for sampling from targets of the form:

$$p(\theta) \propto \varphi(\theta; \mu, \Sigma) \cdot f(\theta)$$

where $\varphi(\theta; \mu, \Sigma)$ is a Gaussian density and $f(\theta)$ is a bounded likelihood term ($0 \leq f(\theta) \leq 1$).

In our case, $\theta$ is $\hat\beta_M$, $\mu=\beta_0,\Sigma=\sigma^2(X_M^\top X_M)^{-1},f(\theta)=P^{cv}(\hat\lambda=\lambda\mid \hat\beta_M=\theta)$.

**Key properties:**
- No tuning (no step sizes or scaling parameters)
- Exploits Gaussian structure via elliptical proposals
- Handles bounded likelihood naturally

## Initialization

Set the initial point to be the observed $\hat\beta_M$.

## One ESS Iteration

Given current state $\theta^{(t)}$, mean $\mu$, and covariance $\Sigma$:

1. Draw $\nu \sim \mathcal{N}(\mu, \Sigma)$ 
2. Draw $u \sim \text{Uniform}[0, 1]$
3. Set log-likelihood threshold $\log y \leftarrow \log L(\mathbf{x}) + \log u$
4. Draw an initial proposal $\theta \sim \text{Uniform}[0, 2\pi]$ 
5. Define a bracket $[\theta_{\min}, \theta_{\max}] \leftarrow [\theta - 2\pi, \theta]$ 
6. $\mathbf{x}' \leftarrow (\mathbf{x} - \mu)\cos\theta + (\nu - \mu)\sin\theta + \mu$
7. **if** $\log L(\mathbf{x}') > \log y$ **then**
8. $\quad$ **return** $\mathbf{x}'$  (accept)
9. **else** $\triangleright$ Shrink the bracket and try a new point
10. $\quad$ **if** $\theta < 0$ **then**
11. $\quad\quad$ $\theta_{\min} \leftarrow \theta$
12. $\quad$ **else**
13. $\quad\quad$ $\theta_{\max} \leftarrow \theta$
14. $\quad$ $\theta \sim \text{Uniform}[\theta_{\min}, \theta_{\max}]$
15. $\quad$ **go to** 6
