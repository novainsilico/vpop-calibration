# SAEM implementation

This note is not intended to be a complete introduction to SAEM nor a mathematical proof of its convergence. The interested reader might refer to [^Delyon99], [^Tanner93], [^Dempster77], [^Lindstrom90].

## Pre-requisites

In order to run SAEM, the following elements are necessary

- an NLME model (implemented as described in the [dedicated note](./nlme_model.md))
- a data set, containing observations for each patient of the NLME model
- prior estimates, or guesses, regarding the distribution parameters of the NLME model

> [!TIP] Implementation:
> The `PySaem` class implements the optimizer that allows to run SAEM. The source code is available in [optimizer.py](../vpop_calibration/saem/optimizer.py)

## SAEM overview

Each SAEM iteration $k$ is split into the following two steps:

- E-step: new random effects are sampled by performing transitions of a Markov Chain Monte-Carlo (MCMC) kernel
- M-step: the sufficient statistics are updated, as well as the residual error variance, with **stochastic approximation** using step size $\gamma_k$.

The algorithm execution is divided into three phases:
  * **burn-in**: runs only MCMC transitions while keeping population parameters fixed (M-step is skipped)
  * **learning**: updates both individual Gaussian parameters and population parameters (both E-step and M-step are run). The population parameters are updated with $\gamma_k = 1$ _i.e._ no averaging with their previous values.
  * **smoothing**: updates the population parameters using decaying step sizes to stabilize the stochastic approximation estimates. $\gamma_{k} = k^{-r}$ where $k$ here denotes the iteration number of the smoothing phase and $r$ is referred to as `learning_rate_power` in the SAEM config. Note that a necessary condition for convergence is that $\sum_k \gamma_{k} = \infty$ and $\sum_k \gamma_{k}^2 \lt \infty$ which means $\frac{1}{2} \lt r \leq 1$.


### SAEM iterations

Some notation reminders (see [nlme_model.md](./nlme_model.md)):

For patient $i$, the PDU parameters $\phi_i$ are assumed to take the following form:

```math
\log \phi_i = X_i \beta + \eta_i \tag{1}
```
```math
\eta_i \sim \mathcal{N}(0, \Omega)
```

SAEM keeps track of the following throughout its iterations, here $k$ denotes the iteration index:

```math
\begin{align*}
\beta_k&\text{: current estimate of population effects,} \\
\eta^{(k)}&\text{: current individual random effects,} \\
\Omega_k&\text{: current random effects covariance matrix,} \\
\phi^{(k)} & \text{: current PDU estimates,} \\
\psi_k&\text{: current fixed effects,} \\
\sigma_k^2&\text{: current estimate of the variance of the measurement error noise, later referred to as residual variance} \\
s_k^{\text{cross}}&\text{: current stochastic approximation of the expectation of the cross-product sufficient statistic,} \\
s_k^{\text{outer}}&\text{: current stochastic approximation of the expectation of the outer-product sufficient statistic}^{*}, \\
\gamma_k & \text{: current stochastic approximation step size}
\end{align*}
```

${}^{*}$: This is formally an approximation of a **centered** outer-product, not a sufficient statistic because it depends on the provisional $\hat\beta_{k+1}$ (see 1.2.1), itself computed from the running stochastic-approximation state. This is implementation-specific and does not match the SAEM algorithm outlined in [^KuhnLavielle05]. However, the term "sufficient statistic" is chosen here for the sake of simplicity and consistency with the cross-product.

### 1. E-step

SAEM is a stochastic-approximation version of standard EM. It replaces the usual E-step with two steps:

#### 1.1 Simulation step

The simulation step consists of drawing new random effects from their current conditional distribution:
```math
\eta^{(k+1)} \sim p(\cdot | \mathbf{y} ; \beta_k, \Omega_k, \psi_k, \sigma_k^2)
```
Because direct sampling is generally intractable, the implementation approximates it using an MCMC procedure that applies Metropolis-Hastings transitions, starting from $\eta^{(k)}$ and targeting this conditional distribution (see the `mh_step()` function in [metropolis_hastings.py](../vpop_calibration/metropolis_hastings.py)).

The PDU parameters $\phi^{(k+1)}$ corresponding to the sampled random effects are computed via (1) using $\beta_k$.

#### 1.2 Stochastic approximation

#### 1.2.1 Sufficient statistics

The stochastic approximation is first applied to the cross-product sufficient statistic:

```math
\begin{align*}
\tilde S_{k+1}^{\text{cross}} &= \sum_i {X_i}^T \log \phi_i^{(k+1)}\\
s_{k+1}^{\text{cross}} &= \gamma_k \tilde S_{k+1}^{\text{cross}} + (1-\gamma_k)s_{k}^{\text{cross}}
\end{align*}
```

We provisionally compute a target value for the population effects $\hat\beta_{k+1}$ by solving 

```math
\left(\sum_i {X_i}^T X_i\right)\hat\beta_{k+1} = s_{k+1}^{\text{cross}}
```

Finally, the stochastic approximation is applied to the outer-product sufficient statistic:
```math
\begin{align*}
\tilde S_{k+1}^{\text{outer}} &= \sum_i (\log \phi_i^{(k+1)} - X_i \hat\beta_{k+1}) (\log \phi_i^{(k+1)} - X_i \hat \beta_{k+1})^T\\
s_{k+1}^{\text{outer}} &= \gamma_k \tilde S_{k+1}^{\text{outer}} + (1-\gamma_k)s_{k}^{\text{outer}}
\end{align*}
```

> [!TIP] Note:
> This is implementation-specific, see the note in the "SAEM iterations" section.

The expressions for $\tilde S_{k+1}^{\text{cross}}$ and $\tilde S_{k+1}^{\text{outer}}$ assume a single MCMC chain for the sake of clarity. When multiple chains are involved (see the `nb_chains` option), they are further averaged over all chains.  

#### 1.2.2 Residual variance

For the sake of simplicity, we assume an additive error noise model in the following.

Contrary to textbook SAEM where the variance update is part of the M-step, this implementation applies stochastic approximation directly to its maximum-likelihood estimate:

```math
\hat \sigma_{k+1}^2 = \frac{1}{N_{obs}} \sum_i \sum_j (y_{i,j} - f(\theta^{(k+1)}_i, t_{i,j}))^2
```
During the learning phase, simulated annealing prevents the variance components from decreasing too quickly. For each component of the variance, the target is first replaced by

```math
\tilde \sigma_{k+1}^{2} = \max\left(\hat \sigma_{k+1}^2,\alpha \sigma_{k}^2\right).
```
where $\alpha$ is the annealing factor.
The stochastic-approximation update is then

```math
\sigma_{k+1}^{2}
=
\gamma_k\tilde \sigma_{k+1}^{2}
+
(1-\gamma_k)\sigma_{k}^{2}.
```

### 2. M-step

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

[^KuhnLavielle05]: E. Kuhn, M. Lavielle (2005). Maximum likelihood estimation in nonlinear mixed effects models. Computational Statistics & Data Analysis, Volume 49, Issue 4. https://doi.org/10.1016/j.csda.2004.07.002.
