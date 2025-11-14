## NLME and SAEM implementation

### Notations

Consider a non-linear mixed effects model of the following dimensions

```math
\begin{align*}
&n \text{ individuals, indexed by }i, \\
& m_i \text{ observations for individual }i, \\
&p \text{ random effects}, \\
&q \text{ fixed effects}, \\
&r \text{ covariates}, \\
\end{align*}
```

Non-linear mixed effects model, for individual $i$, random effects $\theta_i$ and fixed effects $\psi_i$:

```math
\begin{align*}
\theta_i =& \mu_{pop} \times \exp( covariates_i \times coeffs) \times \exp(\eta_i), \\
\eta_i & \sim \mathcal{N}(0,\Omega): \text{ random effects of individual i, size } p, \\
\mu_{pop} :& \text{population means of random effects, size } p\\
\psi:& \text{ fixed effects, size } q \\
y_{i,j} = & f(\theta_{i,j}, \psi) + \epsilon_{i,j}, \epsilon_{i,j}\sim \mathcal{N}(0, \sigma^2)
\end{align*}
```

We write $\beta$ and $X_i$ as

```math
\begin{align*}
\beta =& (\log(\mu_1), \rho_{cov_1/mu_1}, \rho_{cov_2/mu_1}, ... , \log(\mu_2), \rho_{cov_1/mu_2}, \rho_{cov_2/mu_2}, ...) \text{ : size } p \times (1+r)  \\
X_i =& \begin{pmatrix} 1 & cov_{1,1} & cov_{2,1} & ... & 0 & ... & 0 \\ 0 &...& 0 & 1& cov_{1,i} & cov_{2,i} & ... & 0 \\ & & &  ... & & \\ \end{pmatrix} \text{ size } (p, p\times (1+r))
\end{align*}
```

such that

```math
\log(\theta_i) = X_i \times \beta + \eta_i
```

Note: The covariates effects are only described if $\neq 0$ (meaning that the design matrix in fact has _at most_ $1+r$ columns)

### SAEM iteration

At each iteration $k$, the following quantities are available

```math
\begin{align*}
\beta^k&\text{: current estimate of mean of random effects,} \\
\eta^k&\text{: current individual random effects,} \\
\Omega^k&\text{: current random effects covariance matrix,} \\
\psi^k&\text{: current fixed effects,} \\
X = \begin{pmatrix} X_1 \\ X_2 \\ .. \\ X_n \end{pmatrix} & \text{: full design matrix} \\
\end{align*}
```

1. Sample new random effects $\eta^{k+1}$ from the current posterior distribution, with the mean of the current estimate $\eta^k$ as a starting point. MCMC sampling.

```math
\eta^{k+1} \sim p(\eta | \beta^k, \Omega^k, \psi^k, \sigma^k, Y)
```

2. Compute the individual descriptors $\theta^{k+1}$ from these $\eta^{k+1}$

```math
\begin{align*}
\log(\theta^{k+1}) = X \times \beta^k + \eta^{k+1}
\end{align*}
```

3. Compute the residuals by simulating $f(\theta_{k+1}, \psi^k)$

In fact, these are already computed during the MCMC sampling step, and can just be stored during step 1. and retrieved here

4. Update the residual covariance (sum of squares)

```math

(\sigma^{k+1})^2 = \frac{1}{n} \sum_i \frac{1}{m_i} \sum_i (y_{i,j} - f(\theta^{k+1}, \psi^k))^2

```

(+ simulated annealing for certain steps)

5. Update random effects covariance

Use averaged MCMC samples if multiple samples were accepted.

```math
\Omega^{k+1} = \frac{1}{n} \sum_i  \eta_i^{k+1}, \eta_i^{k+1})^T = \frac{1}{n} (\eta^{k+1})^T \times \eta^{k+1}
```

6. Find the MI parameters, $\psi^{k+1}$ by maximizing the likelihood function over all observations (without changing the means, covariate effects nor random effects)

```math
\psi^{k+1} = \underset{\psi}{\arg \max} \sum_{i} \log p(y | \theta^{k+1}, \psi^{k}, \sigma^{k+1})
```

> Note: this is a crucially limiting step in the current implementation

7. Update the $\beta$ by solving generalized least squares on the average log parameters

The idea is that we want the conditional mean of (log) params $\log \bar{\theta}^{k+1} = \frac{1}{n}\sum_i \log \theta_i^{k+1}$ to depend linearly on $\bar{X} = \frac{1}{n}\sum_i X_i$, with the residuals having a covariance matrix $\Omega^{k+1}$.

In other words we are looking for $\hat{\beta}$ such that

```math
\log \bar{\theta}^{k+1} = \bar{X} \hat{\beta} + \epsilon, E[\epsilon | X]=0, cov[\epsilon | X] = \Omega^{k+1}
```

The solution $\beta^{k+1}$ is obtained by solving the associated normal equation

```math
\bar{X} (\Omega^{k+1})^{-1} \bar{X} \beta^{k+1} = \bar{X} (\Omega^{k+1})^{-1} \log \bar{\theta}^{k+1}
```

### References

- Tanner, M. A. (1993). Tools for statistical inference (Vol. 3). New York: Springer. https://doi.org/10.1007/978-1-4684-0192-9
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the royal statistical society: series B (methodological), 39(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
- Bernard Delyon. Marc Lavielle. Eric Moulines. "Convergence of a stochastic approximation version of the EM algorithm." Ann. Statist. 27 (1) 94 - 128, February 1999. https://doi.org/10.1214/aos/1018031103
- Lindstrom, M. J., & Bates, D. M. (1990). Nonlinear mixed effects models for repeated measures data. Biometrics, 673-687. https://doi.org/10.2307/2532087
