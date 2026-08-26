# Kuhn and Lavielle (2005): notation to code mapping

This table maps the notation in Kuhn and Lavielle, *Maximum likelihood
estimation in nonlinear mixed effects models* (2005), to the closest variables
in the current implementation. The implementation generalizes the model in the
article, so some mappings cover several code variables.

| Article notation | Meaning in the article | Code equivalent |
|---|---|---|
| $y_{ij}$ | Observation $j$ for individual $i$ | `model.data.full_obs.obs_values` |
| $x_{ij}$ | Known observation condition, such as time | `model.data.full_obs.obs_index.time` (with protocol information in the other `DataIndex` fields) |
| $g(\phi_i, \beta, x_{ij})$ | Structural-model prediction | `model.predict_all_patients(...)` / the configured `StructuralModel` |
| $h(\phi_i, \beta, x_{ij})$ | Observation-error scale | `model.residual_var.variance(...)` |
| $\sigma^2$ | Within-individual residual variance | `model.residual_var` (`sigma_add` and `sigma_prop` store variance components) |
| $\phi_i$ | Individual parameter vector, with $\phi_i=A_i\mu+\eta_i$ | `mh_state.gaussian_params` |
| $A_i$ | Known individual design matrix | `model.design_matrices[id]`; stacked as `model.full_design_matrix` |
| $\mu$ | Population means and covariate coefficients | `model.population_betas` |
| $\eta_i$ | Centered individual random effects | `mh_state.etas` |
| $\Gamma$ | Random-effects covariance matrix | `model.omega_pop` |
| $\beta$ | Fixed parameters that do not enter $\phi_i$ | Primarily `model.log_mi` and `model.surv_coeffs` |
| $\theta=(\beta,\mu,\Gamma,\sigma^2)$ | Complete population-level parameter vector | The fields assembled in `NlmeModelState` / `PopEstimates` |
| $\widetilde S(y,\phi)$ and $s_k$ | Complete-data sufficient statistics and their running approximation | `optimizer.sufficient_statistics`, especially `cross_product` and `outer_product`; residual-error statistics are updated separately in `PySaem.step()` |
| $\gamma_k$ | Stochastic-approximation step size | `optimizer.scheduler.stochastic_approximation_rate` |
| $\Pi_{\theta_k}$ and $M$ | MCMC transition kernel and number of transitions per SAEM iteration | Repeated calls to `mh_step(...)`; `config.nb_mcmc_transitions` |
| $p(\phi\mid y;\theta_k)$ | Conditional target distribution used in the simulation step | `model.log_posterior_etas_all_patients(...)`, expressed in the equivalent centered `etas` coordinates |
| $f(y,\phi;\theta)$ | Complete-data density | `mh_state.log_prob` is its log-density, up to constants and expressed through `etas` |
| $q(y;\theta)$ and $l(\theta)=\log q(y;\theta)$ | Marginal likelihood and observed-data log-likelihood | `ImportanceSampler.log_lik`, computed by `compute_log_likelihood_importance_sampling(...)` |

The article's $\theta$ should not be confused with tensors named `theta` in the
code: those tensors contain individual physical model parameters. Likewise,
the article models $\phi_i$ directly as Gaussian, whereas the implementation
calls this representation `gaussian_params` and may transform it to constrained
physical PDU values before evaluating the structural model.

Reference: E. Kuhn and M. Lavielle (2005), *Computational Statistics & Data
Analysis* 49, 1020–1038, <https://doi.org/10.1016/j.csda.2004.07.002>.
