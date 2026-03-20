# Vpop calibration

## Description

A set of Python tools to allow for virtual population calibration, using a non-linear mixed effects (NLME) model approach, combined with surrogate models in order to speed up the simulation of QSP models.

The approach was mainly inspired from [^Grenier2018].

### Currently available features

- Surrogate modeling using gaussian processes, implemented using [GPyTorch](https://github.com/cornellius-gp/gpytorch)
- Synthetic data generation using ODE models. The current implementation uses [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), parallelized with [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- Non-linear mixed effect models, see the [dedicated doc](./docs/nlme_model.md):
  - Log-distributed parameters
  - Additive or multiplicative error model
  - Covariates handling
  - Known individual patient descriptors (i.e. covariates with no effect on other descriptors outside of the structural model)
- SAEM: see the [dedicated doc](./docs/saem_implementation.md)
  - Optimization of random and fixed effects using repeated longitudinal data
- Individual parameter estimation
  - Conditional distribution sampling: the posterior conditional distribution may directly be sampled using Metropolis-Hastings algorithm
  - Empirical Bayesian Estimators (EBEs) or Maximum A Posteriori (MAP) estimates: these values represent the mode of the conditional distribution for each parameter and each patient. The current implementation is sub-optimal for `torch` surrogate models as it does not leverage the gradient predictions.
- Diagnostics:
  - Post-inference diagnostics are available, and different type sof weighted residuals may be plotted (IWRES, PWRES, NPDE)

## Getting started

- [Tutorial](./examples/saem/saem_gp_model.ipynb): this notebook demonstrates step-by-step how to create and train a surrogate model, using a reference ODE model and a GP surrogate model. It then showcases how to optimize the surrogate model on synthetic data using SAEM

### In-depth examples

- Surrogate model training
  - [Data generation using Sobol sequences](./examples/surrogate_model/generate_data_ranges.ipynb)
  - [Data generation using a reference NLME model](./examples/surrogate_model/generate_data_nlme.ipynb)
  - [Training and exporting a GP using synthetic data](./examples/surrogate_model/train_gp.ipynb)
  - [Training a GP with a deep kernel](./examples/surrogate_model/train_deep_kernel.ipynb)
- Running SAEM without surrogate model
  - [Running SAEM on a reference ODE model](./examples/saem/saem_ode_model.ipynb). Note: the current implementation is notably under-optimized for running SAEM directly on an ODE structural model. This is implemented for testing purposes mostly
- Benchmarking examples
  - [Orange trees](./examples/benchmarking/benchmark_orange_trees.ipynb)
  - [Theophylline](./examples/benchmarking/benchmark_theophylline.ipynb)

## Support

For any issue or comments, please reach out to <paul.lemarre@novainsilico.ai>, or feel free to open an issue in the repo directly.

## Authors

- Paul Lemarre
- Eléonore Dravet
- Hugo Alves

## Acknowledgements

- Adeline Leclercq-Samson
- Eliott Tixier
- Louis Philippe

## Roadmap

- NLME:
  - Support additional error models (additive-multiplicative, power, etc...)
  - Support additional covariate models (categorical covariates)
  - Compute likelihood via importance sampling following population parameters optimization
- Structural models:
  - Integrate with SBML models (Roadrunner)
- Surrogate models:
  - Support additional surrogate models in PyTorch
- Optimizer:
  - Add preconditioned Stochastic-Gradient-Descent (SGD) method for surrogate model optimization

## QSPC26 poster

This work was presented at QSPC2026 in Leiden, and all the corresponding material is introduced in [this document](./qspc26/tmdd_benchmark/tmdd_benchmark.md).
The benchmark notebooks for standard data sets are available directly for [orange trees](./examples/benchmarking/benchmark_orange_trees.ipynb) and [theophylline](./examples/benchmarking/benchmark_theophylline.ipynb).

The poster itself will be made available in the repository as soon as it is ready.

## References

[^Grenier2018]: [Grenier et al. 2018](https://doi.org/10.1007/s40314-016-0337-5): Grenier, E., Helbert, C., Louvet, V. et al. Population parametrization of costly black box models using iterations between SAEM algorithm and kriging. Comp. Appl. Math. 37, 161–173 (2018). <https://doi.org/10.1007/s40314-016-0337-5>
