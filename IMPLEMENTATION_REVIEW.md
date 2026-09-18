# Implementation review

Date: 2026-09-18

Scope: `vpop_calibration`, excluding GP-specific and Simwork-specific code and findings.

The review identified **11 numerical correctness bugs** and **two configuration/runtime issues**. No implementation files were changed.

Priority labels:

- **P1:** High-priority correctness issue that can corrupt predictions, likelihoods, or calibration under the stated conditions.
- **P2:** Diagnostic, estimation, configuration, or runtime issue that should be corrected.

## Numerical correctness findings

### 1. [P1] Unsorted patient IDs mix up parameters and observations

Location: [pynlme/data.py](vpop_calibration/pynlme/data.py), line 24; [pynlme/indexing.py](vpop_calibration/pynlme/indexing.py), line 67.

`ObsData.patients` preserves first-seen patient order, while observation indices sort IDs. Patient parameter tensors follow the former ordering and prediction indices follow the latter.

**Reproduction:** For patients `['b', 'a']` with known offsets `[20, 10]`, an analytical model computing `offset + gain`, with gain equal to 1, returned predictions `[11, 21]`. Individual evaluations correctly returned `[21, 11]`.

**Impact:** Observations are paired with another patient's parameters, corrupting predictions, likelihoods, and patient-wise Metropolis–Hastings updates.

**Suggested correction:** Use a consistent patient ordering across parameter tensors and full observation indices, while retaining local indices for individual evaluations.

### 2. [P1] Zero residual variance removes an output from the likelihood

Location: [pynlme/residuals.py](vpop_calibration/pynlme/residuals.py), lines 224–226.

`compute_normal_likelihood()` identifies continuous outputs using the numerical values of `sigma_add` and `sigma_prop`, rather than the flags identifying active error components. Zero variance is accepted as a prior and can also arise from an exact residual fit.

**Reproduction:** With additive variance zero, both exact predictions and predictions of `100` contributed `0` to the log-likelihood, despite `min_variance=1e-6`.

**Impact:** An output can silently stop influencing calibration when its estimated residual variance reaches zero.

**Suggested correction:** Select continuous outputs using the active-component flags and apply the configured variance floor to their likelihood.

### 3. [P1] Fixed-effect optimization can move in the wrong direction with multiple chains

Location: [saem/optimizer.py](vpop_calibration/saem/optimizer.py), lines 291–293.

The fixed-effect objective averages Gaussian latent parameters across chains before evaluating the nonlinear model. This differs from averaging the complete-data log-likelihood contributions across sampled chains.

**Reproduction:** For `y = m * p`, observations equal to 1, chain samples `p = 0.5` and `p = 2`, and `m = 0.8`, the implementation produced a gradient of approximately `−0.32`. Averaging the chain likelihoods produced approximately `+0.72`.

**Impact:** The fixed-effect update can move in the opposite direction from the intended objective.

**Suggested correction:** Evaluate the objective for each sampled chain and average likelihood contributions instead of latent parameters.

### 4. [P1] Population regression ignores random-effect correlations

Location: [saem/m_step.py](vpop_calibration/saem/m_step.py), lines 118–122.

The population regression solves ordinary least squares using `XᵀX`. With correlated PDUs and different covariate sets per PDU, the Gaussian likelihood requires covariance weighting using `XᵀΩ⁻¹X`.

**Reproduction:** With three patients, an intercept and slope for the first PDU, and only an intercept for the second, the code returned slope `0`. The likelihood gradient for that slope was `6` under the covariance returned by the same update. A covariance-weighted slope of `−0.25` reduced deviance by `0.75`.

**Impact:** The returned population coefficients need not be stationary for the Gaussian likelihood.

**Suggested correction:** Account for the fitted covariance in the population-coefficient update for the supported covariate structures.

### 5. [P1] SBML predictions shift when observations start after time zero

Location: [structural_model/sbml.py](vpop_calibration/structural_model/sbml.py), lines 26–28.

After resetting initial conditions, the implementation calls `rr.simulate(times=time_steps)`. When the first requested time is positive, RoadRunner assigns the initial conditions to that time instead of integrating from time zero first.

**Reproduction:** For the repository SBML model with `A0(t) = 0.5 * exp(-0.5 * t)`, requesting times `[1, 2]` returned approximately `[0.5, 0.3033]`. Requesting `[0, 1, 2]` established the correct values at times 1 and 2 as `[0.3033, 0.1839]`.

**Impact:** Predictions and calibration depend incorrectly on whether time zero appears in the observation schedule.

**Suggested correction:** Integrate from the model's initial time before extracting predictions at the requested observation times.

### 6. [P1] `simulate_from_df()` can select the wrong output or protocol

Location: [structural_model/base.py](vpop_calibration/structural_model/base.py), lines 43–44.

`simulate_from_df()` creates alphabetically indexed observations without remapping output, protocol, and task indices into the structural model's ordering.

**Reproduction:** A model with outputs `['z', 'a']` producing `[2, 20]` returned `[20, 2]`. Requesting only arm `'a'` from model arms `['z', 'a']`, with doses `[10, 100]`, produced `20` instead of `200`.

**Impact:** Direct dataframe simulations and data-generation workflows using this method can silently use the wrong output or protocol.

**Suggested correction:** Remap observation indices to the model's reference order before simulation.

### 7. [P1] Analytical protocol overrides depend on dataframe column order

Location: [structural_model/analytical.py](vpop_calibration/structural_model/analytical.py), lines 53–59.

Override values are stored in dataframe column order, but the argument mapping interprets them in function-argument order.

**Reproduction:** For `f(k, t, a, b) = k + 10*a + b`, `k = 2`, and dataframe columns ordered as `b = 2, a = 3`, the prediction was `25` instead of `34`.

**Impact:** Reordering protocol-design columns changes model predictions without changing the parameter names or values.

**Suggested correction:** Select override columns explicitly in the order specified by `self.protocol_parameters`.

### 8. [P2] Importance sampling uses variance as Student-t scale

Location: [pynlme/importance_sampling.py](vpop_calibration/pynlme/importance_sampling.py), lines 60–65.

`fit_student_t_proposal()` passes `torch.var(etas, 0)` directly as the Student-t `scale`. Variance and scale have different units.

**Reproduction:** Conditional samples with standard deviation approximately `0.01007` produced a proposal standard deviation of approximately `0.000131`, about 77 times narrower. For a normalized Gaussian target with true log-integral `0`, estimates using 10,000 proposals were approximately `−2.13`, `−2.59`, and `−2.14` for three seeds.

**Impact:** The importance-weight formula is correct, but the incorrect proposal width can severely destabilize finite-sample likelihood estimates.

**Suggested correction:** Take the square root of the estimated variance, with any intended Student-t variance adjustment.

### 9. [P2] PWRES uses observed-value variance instead of predictive covariance

Location: [pynlme/diagnostics.py](vpop_calibration/pynlme/diagnostics.py), line 198.

`compute_pwres()` uses `torch.cov(obs_patient.T)`. Since `obs_patient` is one-dimensional, this produces a scalar variance across observed values instead of the covariance of the patient's predictive observations.

**Reproduction:** Constant observations produced infinite weighted residuals. A patient with one observation caused a validation error. Changing the residual variance from 1 to 100 left the computed PWRES unchanged.

**Impact:** PWRES is incorrectly scaled and can become invalid for ordinary observation schedules.

**Suggested correction:** Estimate predictive covariance across simulated replicates, including residual error, and handle the single-observation case explicitly.

### 10. [P2] NPDE ignores observation noise

Location: [pynlme/diagnostics.py](vpop_calibration/pynlme/diagnostics.py), lines 240–247.

`compute_npde()` compares observations against noise-free structural predictions without adding residual observation noise.

**Reproduction:** For a constant structural prediction of 1, observations of `0.9` and `1.1`, and unit residual variance, the method returned approximately `±3.09` with 1,000 samples. The corresponding normal quantiles should be near `±0.1`. Changing residual variance did not change the output.

**Impact:** Small observation errors can appear as extreme model discrepancies.

**Suggested correction:** Generate predictive observations including the configured residual error before computing prediction discrepancies.

### 11. [P2] VPC prediction intervals omit observation noise

Location: [pynlme/diagnostics.py](vpop_calibration/pynlme/diagnostics.py), line 326.

`compute_vpc()` uses deterministic conditional predictions from `total_samples_predictions_df`, without residual observation noise.

**Reproduction:** In a constant-output model with nonzero residual variance, the predicted 10th, 50th, and 90th percentiles and all their intervals collapsed to 1.

**Impact:** The plotted prediction intervals do not represent the variability of observed data and can falsely indicate lack of fit.

**Suggested correction:** Generate replicated observation datasets with the intended population sampling and residual-error model before computing VPC quantiles.

## Configuration and runtime findings

### 12. [P2] Mode-dependent SAEM defaults do not follow the selected mode

Location: [saem/config.py](vpop_calibration/saem/config.py), lines 34–44.

The `NamedTuple` field defaults are evaluated once when the class is defined. Setting `mode` during construction does not recompute the dependent defaults.

**Reproduction:** All three modes produced `live_plot=True`, `logging=False`, and `progress_bars=True`.

**Impact:** CLI mode still plots and debug mode does not enable logging unless users explicitly override the individual fields.

**Suggested correction:** Resolve mode-dependent defaults when constructing the configuration.

### 13. [P2] SAEM plotting can crash when optional display dependencies are absent

Location: [saem/plot.py](vpop_calibration/saem/plot.py), lines 58–59 and 69–70; also line 45 for missing matplotlib.

`self.handle` is initialized only when IPython display support is available, but later updates access it unconditionally. Similarly, the constructor accesses `self.axes` even when matplotlib was unavailable and no axes were created.

**Reproduction:** With the module's `display` set to `None`, a plot update raised `AttributeError` because `self.handle` did not exist.

**Impact:** Plot-enabled SAEM runs can fail in environments lacking optional plotting dependencies. The mode-default issue above makes this relevant to CLI configurations as well.

**Suggested correction:** Initialize optional plotting state consistently and disable or guard plotting when its dependencies are unavailable.

## Validation and limitations

- The existing non-Simwork test suite passed: **116 tests and 48 subtests**, in approximately 20 seconds, using its configured smoke-test behavior. This run included GP tests before the review scope was narrowed further; GP findings are excluded from this report.
- Test files were copied to a temporary directory to avoid the package-wide Simwork build in `vpop_calibration/test/__init__.py`. Repository tests and implementation files were not modified.
- Focused standalone Python reproductions demonstrated the numerical failures described above, despite the existing suite passing.
- CPU execution was used. GPU-specific behavior was not verified.
- This is a review of confirmed findings, not a guarantee that all implementation defects have been identified.

Standalone reproduction scripts were retained at these temporary local paths. They are not committed artifacts and may be removed when temporary storage is cleaned:

- `/tmp/audit_pynlme_repro.py`: patient ordering, zero residual variance, and importance proposal scaling.
- `/tmp/saem_audit_repro.py`: population regression, multiple-chain fixed-effect objective, and mode defaults.
- `/tmp/audit_core_structural_repro.py`: output/protocol indexing, analytical override order, and SBML timing.
- `/tmp/repro_review_diagnostics.py`: PWRES, NPDE, and VPC.
