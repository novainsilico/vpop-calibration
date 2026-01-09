# SAEM implementation

This note is not intended to be a complete introduction to SAEM nor a mathematical proof of its convergence. The interested reader might refer to [^Delyon99], [^Tanner93], [^Dempster77], [^Lindstrom90].

## Pre-requisites

In order to run SAEM, the following elements are necessary

- an NLME model (implemented as described in the [dedicated note](./nlme_model.md))
- a data set, containing observations for each patient of the NLME model
- prior estimates, or guesses, regarding the distribution parameters of the NLME model

> [!TIP] Implementation:
> The `PySaem` class implements the optimizer that allows to run SAEM. The source code is available in [saem.py](../vpop_calibration/saem.py)

## SAEM overview

The SAEM iteration is composed of the following steps:

- E-step: new random effects are sampled by performing transitions of a MCMC kernel
- M-step: the sufficient statistics are updated, as well as the residual error variance, with stochastic approximation

The algorithm execution is divided in two phases: the **exploratory** phase, during which the learning rate is set to 1 (no stochastic approximation), and the **smoothing** phase, during which the learning rate gradually decays to 0, to ensure convergence of the Markov chains.

### One iteration

Throughout the iterations $k$, the following quantities are tracked

```math
\begin{align*}
\beta^k&\text{: current estimate of mean of random effects,} \\
\eta^k&\text{: current individual random effects,} \\
\Omega^k&\text{: current random effects covariance matrix,} \\
\phi^k & \text{: current PDU estimates,} \\
\psi^k&\text{: current fixed effects,} \\
S_1^k&\text{: current sufficient statistic 1 (cross-product)} \\
S_2^k&\text{: current sufficient statistic 2 (outer-product)}, \\
\gamma^k & \text{: current stochastic approximation learning rate}
\end{align*}
```

### 1. E-step:

Sample new random effects $\eta^{k+1}$ from the current posterior distribution, with the mean of the current estimate $\eta^k$ as a starting point.

In practice, a certain number of transitions from an MCMC (Monte-Carlo Markov Chain) kernel are performed. The Metropolis-Hastings algorithm is used to perform these transitions (see [nlme.py](../vpop_calibration/nlme.py): `mh_step`)

```math
\eta^{k+1} \sim p(\eta | \beta^k, \Omega^k, \psi^k, \sigma^k, \mathbf{y})
```

The $\phi$ PDU parameters are updated simultaneously as the random effects $\eta$, via the relationship

```math
\log \phi^{k+1} = X \times \beta^k + \eta^{k+1}
```

### 2. M-step:

#### 2.1. Residual error variance update

Using the residuals corresponding to the updated individual parameters, the residual error variance is updated. The target residual error variance is defined as

```math
\sigma_{target}^2 = \frac{1}{N_{obs}} \sum_i \sum_j (y_{i,j} - f(\theta^{k+1}_i, t_{i,j}))^2
```

The new residual error variance is the computed via stochastic approximation (smoothing):

```math
\sigma^{k+1} = \gamma^{k+1}\sigma_{target} +  (1-\gamma^{k+1})\sigma^k
```

During the exploratory phase, the residual error variance update also incorporates a simulated annealing step, restricting the rate at which the residual error variance may decay.

#### 2.2. Sufficient statistics update

The two sufficient statistics are updated as follows

```math
S_1^{target}= \sum_i {X_i}^T \log \phi_i^{k+1} \\
S_2^{target} = \sum_i (\log \phi_i^{k+1})^T\log \phi_i^{k+1}
```

The update is performed with stochastic approximation:

```math
S_1^{k+1} = \gamma^{k+1} S_1^{target} +  (1-\gamma^{k+1})S_1^k \\
S_2^{k+1} = \gamma^{k+1} S_2^{target} +  (1-\gamma^{k+1})S_2^k
```

#### 2.3. $\beta$ and $\Omega$ update

The fixed effects vector $\beta$ is updated by solving the linear system

```math
X^T X \beta^{k+1} = S_1^{k+1}
```

The covariance matrix $\Omega$ is updated as

```math
\Omega^{k+1} = \frac{1}{n_{patients}} \Big(S_2^{k+1} - (X\beta^{k+1})^T(X\beta^{k+1})\Big)
```

In practice, the eigenvalues of $\Omega$ are also clamped to a minimal value, ensuring numerical stability.

#### 2.4. Model intrinsic $\psi$ update

The MI parameters are updated by maximizing the likelihood function over all observations (without changing the means, covariate effects nor random effects)

```math
\psi^{k+1} = \underset{\psi}{\arg \max} \sum_{i} \log p(\mathbf{y} | \theta^{k+1}, \psi^{k}, \sigma^{k+1})
```

> [!TIP] Note:
> this is a crucially limiting step in the current implementation, as it involves calling an external optimizer.

### References

[^Tanner93]: Tanner, M. A. (1993). Tools for statistical inference (Vol. 3). New York: Springer. https://doi.org/10.1007/978-1-4684-0192-9
[^Dempster77]: Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the royal statistical society: series B (methodological), 39(1), 1-22. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
[^Delyon99]: Bernard Delyon. Marc Lavielle. Eric Moulines. "Convergence of a stochastic approximation version of the EM algorithm." Ann. Statist. 27 (1) 94 - 128, February 1999. https://doi.org/10.1214/aos/1018031103
[^Lindstrom90]: Lindstrom, M. J., & Bates, D. M. (1990). Nonlinear mixed effects models for repeated measures data. Biometrics, 673-687. https://doi.org/10.2307/2532087
