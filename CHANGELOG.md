# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
