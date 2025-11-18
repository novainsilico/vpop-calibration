# Vpop calibration

## Description

A set of Python tools to allow for virtual population calibration, using a non-linear mixed effects (NLME) model approach, combined with surrogate models in order to speed up the simulation of QSP models.

### Currently available features

- Surrogate modeling using gaussian processes, implemented using [GPyTorch](https://github.com/cornellius-gp/gpytorch)
- Synthetic data generation using ODE models. The current implementation uses [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), parallelized with [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- Non-linear mixed effect models:
  - Log-distributed parameters
  - Additive or multiplicative error model
  - Covariates handling
  - Known individual patient descriptors (i.e. covariates with no effect on other descriptors outside of the structural model)
- SAEM: see the [dedicated doc](./docs/saem_implementation.md) for more details

## Getting started

- [Tutorial](./examples/saem_gp_model.ipynb): this notebook demonstrates step-by-step how to create and train a surrogate model, using a reference ODE model and a GP surrogate model. It then showcases how to optimize the surrogate model on synthetic data using SAEM
- Other available examples:
  - [Data generation using Sobol sequences](./examples/generate_data_ranges.ipynb)
  - [Data generation using a reference NLME model](./examples/generate_data_nlme.ipynb)
  - [Training and exporting a GP using synthetic data](./examples/train_gp.ipynb)
  - [Running SAEM on a reference ODE model](./examples/saem_ode_model.ipynb). Note: the current implementation is notably under-optimized for running SAEM directly on an ODE structural model. This is implemented for testing purposes mostly
  - [Training a GP with a deep kernel](./examples/train_deep_kernel.ipynb)

## Support

For any issue or comments, please reach out to paul.lemarre@novainsilico.ai, or feel free to open an issue in the repo directly.

## Authors and acknowledgment

- Paul Lemarre
- Eléonore Dravet
- Adeline Leclerq-Sampson

## Roadmap

- NLME:
  - Support additional error models (additive-multiplicative, power, etc...)
  - Support additional covariate models (categorical covariates)
  - Add residual diagnostic methods (weighted residuals computation and visualization)
- Structural models:
  - Integrate with SBML models (Roadrunner)
- Surrogate models:
  - Support additional surrogate models in PyTorch
- Optimizer:
  - Add SVGP for surrogate model optimization

## References

- [Delyon et al. 99](https://doi.org/10.1214/aos/1018031103): Bernard Delyon. Marc Lavielle. Eric Moulines. "Convergence of a stochastic approximation version of the EM algorithm." Ann. Statist. 27 (1) 94 - 128, February 1999. https://doi.org/10.1214/aos/1018031103
- [Grenier et al. 2018](https://doi.org/10.1007/s40314-016-0337-5): Grenier, E., Helbert, C., Louvet, V. et al. Population parametrization of costly black box models using iterations between SAEM algorithm and kriging. Comp. Appl. Math. 37, 161–173 (2018). https://doi.org/10.1007/s40314-016-0337-5
