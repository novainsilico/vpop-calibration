# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.1] - 2026-01-14
### Modified
- Consolidate pytest pipelibe structure

## [2.4.0] - 2026-01-08
### Modified
- NLME: Add support for multiple parallel MCMC chains per patient

### Added
- Examples: two benchmarking examples for comparison with `saemix` results. Theophylline and orange trees data sets

## [2.3.1] - 2025-12-15

### Modified

- GP: training and validation losses are now plotted live during the training.
- GP: early stopping criterion added. The algorithm now stops if the loss (validation loss, if available, otherwise training loss) stabilizes for at least a certain number of iterations. Stabilization parametrized with `min_delta` parameter, number of iterations parametrized by `patience` parameter
- GP: all plotting methods can now be configured with a `fig_scaling` parameter, giving the `(width, height)` tuple that will define the size of each individual facet in the plots

### Removed

- GP: `plot_loss` method deprecated

## [2.3.0] - 2025-12-12

### Modified

- Torch: support device management and use cuda if available
- Torch: refactor implementation with better tensor manipulation, allowing actual scaling on GPU. Marginal speed-up observed on CPU as well, via the use of torch.compile wherever possible

## [2.2.0] - 2025-12-05

### Added

- Live plots during SAEM iterations: PDU mean, variance, MI value, residual variance, number of out of bounds patients, convergence criterion
- Convergence criterion: complete log likelihood from MCMC step
- Validation of predictive variance: if var > threshold, patients are flagged. Number of patients with high predictive variance is plotted in dashboard during iterations
- Diagnostics: posterior validation of maximum a posteriori estimates with respect to surrogate model training ranges

### Fixed

- Correct square in initial PDU standard deviation
- Data generation: properly add residual error to generated data using NLME model

## [2.1.2] - 2025-12-04

### Added

- Include multithreading opt-out in OdeModel class, useful when `multiprocessing` is buggy

### Modified

- Consolidate example notebooks and ensure they run consistently

## [2.1.1] - 2025-11-27

### Fix

- GP: Handle NaN values in GP RMSE output
- Correct column naming for Vpop generation

### Modified

- Tests: set random state / seed for tests
- GP: plot of individual solutions now uses ±2 \* pred variance for confidence interval

## [2.1.0] - 2025-11-26

### Modified

- Structural model: Simulate GP in wide format and use tensor manipulation to recover task values, runtime improvement for SAEM estimation
- Vpop: use natural log to define parameter ranges, aligned with implementation choice in the rest of the module

## [2.0.1] - 2025-11-25

### Modified

- GP: allow for NaN values to appear in the training data frame. This enables to train a GP on an incomplete data set

## [2.0.0] - 2025-11-24

### Modified

- SAEM: corrected E-step updates. Drastic speed-up in runtime achieved
- SAEM: refactored M-step, implemented sufficient statistics with stochastic estimation

## [1.1.1] - 2025-11-18

### Added

- Support for deep feature extractor with all kernels
- Support for Matern5/2 kernel
- Nested process bars in batch training for GP

## [1.1.0] - 2025-11-18

### Added

- Tests for data loading and processing
- Support for other surrogates: separate data and plot functionalities into modules
- Add new kernel `Deep-RBF`: a feature extracting neural network is added inside the kernel, to help with the flexibility of the GP

### Removed

- All plotting and data processing methods remove from the core of the `GP` class

## [1.0.0] - 2025-11-14

### Added

- Add support for local ODE simulations
- Add GP surrogates
- Add nlme models
- Add SAEM optimizer
- Add examples notebooks and docs
- Add tests for GP and SAEM
