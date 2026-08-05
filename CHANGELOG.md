# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.7.0] - 2026-08-20

### Added

- Time to event data is now supported, allowing to fit joint-survival models. The pre-requisites are the following: the ODE model must output an instantaneous `log_hazard` and `cumulative_hazard`, which is integrated by the ODE solver. This allows to compute log-likelihood of a survival event. Survival coefficients may be defined in the input parameters, allowing to optimize gaussian parameters directly (no transformation). This is an experimental feature.

## [4.6.9] - 2026-08-24

### Modified

- Fixed effect optimization: bug fix. Summing likelihood in the objective function is now done over all patients rather than all samples, ensuring that the finite difference gradient has the correct shape. An assertion test is added to catch this error

## [4.6.3] - 2026-08-18

### Modified

- `pyproject.toml`: relax torch dependency to accept `>=2.10.0`

## [4.6.3] - 2026-08-18

### Modified

- `pyproject.toml`: relax torch dependency to accept `>=2.10.0`

## [4.6.2] - 2026-08-17

### Modified

- For notebook users, the main module to import is now `vpop_calibration.api`

## [4.6.1] - 2026-08-04

### Modified

- Fixed effects optimization: replace Nelder-Mead call with a fixed number of iterations of Adams, using forward finite differences to estimate the gradient. New diagnostic endpoint in the SAEM live plot: `fixed_effects_loss`

## [4.5.0] - 2026-08-04

### Modified

- Error moel: add support for combined error model. The specific `sigma_add, sigma_prop` parameters are optimized via torch BFGS.

## [4.4.8] - 2026-08-03

### Modified

- Conditional co-distribution plots: x-axis labels formatted in scientific format

## [4.4.7] - 2026-07-29

### Modified

- Separate EBE and MAP
  - Maximum a posteriori (MAP) estimates: these parameter estimates correspond to the mode of the conditional distribution for each patient. They are estimated in post-processing, by sampling from the conditional distribution.
  - Empirical Bayesian Estimators (EBE): these parameter estimates correspond to the mean of the conditional distribution for each patient. They are estimated throughout the iterations, by averaging over the physical parameter estimates and using stochastic approximation for smoothing.

## [4.4.6] - 2026-07-29

### Modified

- Visual predictive checks: median is now identified by a specific color, and out of CI empirical quantiles are marked in red

## [4.4.5] - 2026-07-23

### Added

- Initial estimates distribution visualization tool: call `nlme_model.initial_estimates.plot_distribution()` to visualize the parameter distributions in physical space
- Golden tests for SAEM on an analytical model
- SBML support via libroadrunner -> `StructuralSbml`

## [4.4.4] - 2026-07-10

### Added

- Save/load methods for the `NlmeModel` interface class. Quickly export the current state of the model and optimizer to a json file. Restore it, by providing the json snapshot and the initial data + structural model.

## [4.4.3] - 2026-07-10

### Added

- Co-distribution plots: visualize marginal distribution histograms for conditional distribution samples and EBEs

## [4.4.2] - 2026-07-09

### Added

- SDK: new module exposing functions to be used in a CLI tool

## [4.4.1] - 2026-07-08

### Added

- Visual predictive checks: call `nlme_model.plot.vpc()` to get a visualization of the predictive quantiles estimated from the conditional distribution

## [4.4.0] - 2026-07-08

### Modified

- New EBE estimation logic: empirical bayesian estimators are now computed during the sampling of the conditional distribution. Convergence plots are featured to show the improvement of the likelihood for each individual patient and on average.
- The `compute_ebe()` method is now deprecated. One should call `sample_conditional_distribution` in order to obtain samples and EBEs.
- `sample_conditional_distribution` method is interruptable, and a maximum cache size is defined to stored samples.

## [4.3.0] - 2026-07-03

### Modified

- SAEM optimizer: heavy refactoring of the optimizer, separate scheduler and business logic functions (m_step). Stream iterations via a generator, allowing to interrupt and restart the optimizer seamlessly.
- Warning: A few configuration options are renamed
- Simwork model: build executable only once for test pipeline - marginal speedup for pytest pipeline

## [4.2.0] - 2026-06-22

### Modified

- EBE estimation: simplification of the empirical bayesian estimates calculation. This is now performed by sampling the conditional distribution and retrieving the sample with highest posterior log-likelihood

## [4.1.0] - 2026-06-17

### Added

- Support for Simwork models in the StructuralModel class. See StructuralSimwork class.
- Important: the current implementation is dramatically under-performing for model intrinsic parameter optimization, as well as EBE estimates. To be investigated and fixed.

### Modified

- Scipy.optimize calls: switched implementation to Nelder-Mead algorithm for model intrinsic and EBE estimates.

## [4.0.0] - 2026-06-11

### Breaking

- ODE simulations are not supported in Python anymore. All example cases are now refactored to use analytical models written in torch. The end users are expected to come with their own simulation software
- GP training ranges cannot be used a NLME parameter constraints out of the box anymore. This will be fixed in a future release

### Modified

- Re-structuration of the NLME interface, see examples for details on how to use the API
- Typing enforced everywhere possible in the NLME and structural model classes. The whole platform is now stricter about typing. Most of these changes should be silent to the user, except for the parameter and configuration of NLME models

## [3.0.4] - 2026-03-23

- Diagnostics: fix PWRES calculation, use variance of _observations_

## [3.0.3] - 2026-03-20

- Structural ode model: add support for patient-specific initial assignments (initial conditions)
- Scientific dissemination: add QSPC26 poster, and supplementary material

## [3.0.2] - 2026-03-17

### Modified

- Structural model: implemented support for protocol designs in StructuralAnalytical()

## [3.0.1] - 2026-02-20

### Added

- NLME: remove torch.compile on data handling functions (graph breaks)

## [3.0.0] - 2026-02-19

### Added

- NLME: add true MAP computation, through conditional distribution sampling and optimization
- NLME: new plotting function to visualize conditional distribution and MAP

## Modified

- NLME: all MAP-related functions now use the true maximum of the distribution, including individual MAP plots and residuals computation
- NLME: functions allowing to go from etas to outputs can now handle single-patient case

## [2.6.0] - 2026-01-29

### Added

- NLME: add weighted residual computations (IWRES, PWRES, NPDE) and plotting

## [2.5.1] - 2026-01-19

### Modified

- NLME: add diagnostic plots for post optimization visual check (see `diagnostics.py`)

## [2.5.0] - 2026-01-14

### Modified

- NLME: add support for parameter constraints using sigmoidal transformation in the computation of physical parameters

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
