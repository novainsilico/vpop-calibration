# Non-linear mixed effect models implementation

This note is not intended to provide an exhaustive introduction to non-linear mixed effects (NLME) modeling. The goal is to introduce some notation and link it with the implemented code to support the user's understanding of the algorithms. The interested reader could refer to [^Lindstrom90] for a complete introduction.

## Statistical model

Consider a data set $\bf{y}$ of longitudinal observations for $N$ individuals. For each individual $i$, $y_{i,j}, j=1\dots m_i$ observations have been collected at times $t_j$. An NLME model associates a structural model $f$ and an error model $g$ to predict the observations as follows

```math
y_{i,j} = f(\theta_i,t_{i,j}) + g(\theta_i, t_{i,j}, \sigma) \varepsilon_{i,j}, \\
\varepsilon \sim  \mathcal{N}(0,1)
```

The observations $y_{i,j}$ may be vector-valued, in the case where multiple outcomes are measured.

The structural model is the simulation model, either a QSP model or a surrogate model in our context, see [Structural model](#structural-model).

The structural model inputs are population parameters which follow a certain distribution - see [Patient descriptors](#patient-descriptors).

### Structural model

The structural model $f$ represents the simulation model. It deterministically maps patient descriptors and observation time to model outcomes.

> [!TIP] Implementation:
> The `StructuralModel` class is implemented to interface with different types of simulation models. It is implemented in [structural_model.py](../vpop_calibration/structural_model.py)

For the moment, two instances can be used:

```python
# From a gaussian process model
struct_model_gp = StructuralGp(myGP)
# From an ODE model
struct_model_ode = StructuralOdeModel(ode_model, protocol_design, initial_conditions)
```

### Patient descriptors

The patient descriptors $\theta_i$ are divided in 4 groups:

- PDK: patient descriptors known, denoted as $\nu_i$. These are parameters informed by the data set, which are used by the structural model directly. Their value is known for each patient.
- PDU: patient descriptors unknown, denoted as $\phi_i$. These are parameters with inter-individual variability. Their distribution is unknown (to be calibrated).
- Covariates: similarly to PDK, these are informed by the data. However, they are not used by the structural model, instead their effect is incorporated inside the statistical model.
- MI: Model intrinsic, denoted as $\psi_i$. These are parameters which are assumed to be constant in the population, but their value is unknown (to be calibrated).

#### PDUs

The PDUs are assumed to follow a multivariate log-normal distribution

```math
\log \phi_i = \beta X_i + \eta_i, \\
\eta \sim \mathcal{N}(0, \Omega)
```

where $\beta$ is the vector of population parameters, containing the means and the covariate coefficients, and $X_i$ is the design matrix of individual $i$, containing the covariate values.

##### Covariates

Considering that $n_{PDU}$ PDU parameters are described, and for each PDU $k$, the covariates $c_{k,1}, \dots, c_{k,n_k}$ have an influence, the total number of $\beta$ values is $n_\beta = \sum_{k=1}^{n_{PDU}} n_k$. For each covariate $c_{k,j}$ which has an influence on PDU $k$, the associated covariation coefficient is denoted as $\rho_{c_{k,j} \rightarrow \mu_k}$ (coefficient of the influence of $c_{k,j}$ on $\mu_k$). The vector of fixed effects $\beta$ and the design matrices $X_i$ are thus formulated as:

```math
\beta = \begin{pmatrix}
\log(\mu_1) \\
\rho_{c_{1,1} \rightarrow\mu_1} \\
\dots \\
\rho_{c_{1,n_1} \rightarrow\mu_1} \\
\dots \\
\log(\mu_2) \\
\rho_{c_{1,2} \rightarrow \mu_2} \\
\dots
\end{pmatrix} \\
X_i = \begin{pmatrix}
1 & c_{1,1}^i & \dots & c_{1,n_{1}}^i & 0 & \dots \\
0 & 0 & 0 & 0 & 1 & c_{2,1} & \dots & c_{2,n_2} & 0 \\
\dots
\end{pmatrix}
```

> [!NOTE]
> For the moment, only exponential covariates are implemented in the NLME, which implies that only continuous values may be taken into account in the covariate values. Different models are possible, and will be considered in future releases.

##### Random effects

The inter-individual variability of PDUs is described via the random effects vectors $\eta_i$, which follow a centered multivariate normal distribution with covariance matrix $\Omega$ (size $n_{PDU} \times n_{PDU}$).

> [!NOTE]
> For the moment, only log-normal multivariate distribution is implemented, but other transformations of the parameter space are possible. The log transformation is a natural choice for QSP models, as it ensures all parameter values (structural model inputs) $\theta_i$ remain positive.

#### Implementation

Parameter descriptors are defined when initiating an NLME model:

```python
# Define model intrinsic parameters
init_log_MI = {"k_a": 0.5}
# Define PDU parameters
init_log_PDU = {
    "k_21": {"mean": 0.1, "sd": 0.5},
    "k_12": {"mean": -0.8, "sd": 0.5},
}
# Define a covariate map
init_covariate_map = {
    "k_21": {"foo": {"coef": "cov_foo_k12", "value": 0.5}},
    "k_12": {},
}
# With this covariate map, one value of `foo` must appear for each patient in the supplied data set

```

### Error model

Different residual error models may be used to represent the unexplained variability.

#### Additive

If the selected model is additive, the NLME predictions are written as:

```math
y_{i,j} = f(\theta_i, t_{i,j}) + \sigma \varepsilon_{i,j}, \\
\mathbf{E}[y_{i,j} | \theta_i] = f(\theta_i, t_{i,j}), Var[y_{i,j} | \theta_i] = \sigma^2,
```

#### Proportional

If the selected error model is proportional, the NLME predictions are written as

```math
y_{i,j} = f(\theta_i, t_{i,j}) (1 + \sigma \varepsilon_{i,j}), \\
\mathbf{E}[y_{i,j}|\theta_i] = f(\theta_i, t_{i,j}), Var[y_{i,j} | \theta_i] = (\sigma f(\theta_i, t_{i,j}))^2
```

> [!NOTE]
> For the moment, only `additive` and `proportional` error models are implemented. Future releases will add different options.

## Likelihood maximization

Optimizing an NLME model with respect to an observation data set can be expressed as a maximum likelihood problem. In fact this formulation only applies to the framework of optimizing the PDU distributions, and handling MI (fixed effects) is a specific issue. We assume now that all parameters are PDUs ($\theta_i = \phi_i$).

### Total likelihood formulation

The objective is to maximimze the total log-likelihood of observations, with respect to the (NLME) model parameters. The full list of parameters is hereby denoted as $\Theta$ and contains:

- $\beta$ (list of population parameters),
- $\Omega$ (covariance matrix of individual random effects),
- $\sigma^2$ (residual variance), one per model output.

The objective that we wish to maximize is the log-likelihood of NLME model parameters given observations $\mathbf{y}$:

```math
\begin{align*}
LL(\Theta ; \mathbf{y}) &= \sum_i \log p(y_i; \Theta) = \sum_i \log p(y_i | \theta_i ; \sigma) p(\theta_i | \Theta) \\
&= \sum_i \log p (y_i | \theta_i; \sigma) + \sum_i \log p(\theta_i | \beta, \Omega)
\end{align*}
```

This expression assumes that all patients are independent.

### Likelihood of observations

For an individual patient $i$, the likelihood of the individual parameters $\theta_i$ given the observation $y_i$ is expressed as

```math
\log p(y_i | \theta_i; \sigma) = -\frac{1}{2} \sum_{j=1}^{m_i} \Big(  \log (2\pi g(\theta_i, t_{i,j}, \sigma)^2) + \frac{(f(\theta_i, t_{i,j}) - y_{i,j})^2}{g(\theta_i, t_{i,j}, \sigma)^2} \Big)
```

> [!TIP]
> This is implemented in the `NlmeModel` class, as the method `log_likelihood_observation` - see [nlme.py](../vpop_calibration/nlme.py)

### Likelihood of individual parameters

The total likelihood balances the likelihood of observations (data) with that of individual parameters (patient descriptors). This second contribution can be expressed as

```math
\log \theta_i = \beta X_i +\eta_i, \eta_i \sim \mathcal{N}(0, \Omega)
```

allows to write their contribution to the likelihood as

```math
\log p(\theta_i | \Theta) = \log p(\eta_i | \Omega) = -\frac{1}{2} (n_{PDU}  \log(2 \pi) + \log|\Omega| + \eta_i^T \Omega^{-1} \eta_i)
```

> [!TIP]
> This is implemented in the `NlmeModel` class, as the method `_log_prior_etas` - see [nlme.py](../vpop_calibration/nlme.py)

[^Lindstrom90]: Lindstrom, M. J., & Bates, D. M. (1990). Nonlinear Mixed Effects Models for Repeated Measures Data. Biometrics, 46(3), 673–687. https://doi.org/10.2307/2532087
