# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
