# vpop_calibration bug findings

Date: 2026-09-18

Scope: full `vpop_calibration/` package, produced by a parallel multi-agent code
review (5 reviewers, one per module area), followed by a targeted investigation
of patient-ID ordering.

**Note:** `IMPLEMENTATION_REVIEW.md` (untracked, already present in this repo
from a separate review pass) independently found several of the same bugs with
*executed* reproduction scripts, and additionally found bugs this review
missed (fixed-effect multi-chain objective, population regression ignoring
random-effect correlations, SBML time-zero shift, `simulate_from_df` index
remapping, NPDE/VPC ignoring observation noise). The two reviews corroborate
each other on every overlapping item. Read both; this file does not restate
`IMPLEMENTATION_REVIEW.md`'s unique findings in full detail.

Priority labels:
- **P1** — corrupts predictions, likelihoods, or calibration results silently.
- **P2** — crashes, dead code, or a latent/edge-case defect.

## P1 — Silent numerical corruption

### 1. Unsorted patient IDs swap patients' parameters and observations

**Confirmed independently, with an executed reproduction (see below), and
corroborated by `IMPLEMENTATION_REVIEW.md` finding #1.**

- `vpop_calibration/pynlme/data.py:24` — `ObsData.patients` is built via
  `.drop_duplicates()` → **first-appearance order**.
- `vpop_calibration/pynlme/indexing.py:65` — `DataIndex.from_dataframe` builds
  `ref_values` via `.drop_duplicates().sort_values()` → **sorted order**,
  applied to the `id` field too.
- `vpop_calibration/pynlme/model.py:145-149` (`remap_all_indexings`) remaps
  `output_name`/`protocol_arm`/`task` but **never `id`**, so the two orderings
  are never reconciled.
- Every structural-model backend builds its patient-axis tensor in
  `self.patients` (first-appearance) order, then gathers per-observation
  predictions using `prediction_index.id.index_values` (indices into the
  *sorted* list), treating the two as positionally equivalent:
  `structural_model/analytical.py:115-141`,
  `structural_model/simwork.py:283-286,336,382-389`,
  `structural_model/sbml.py:145-146,220-228`,
  `structural_model/gp.py:45-56`.
- The same root bug independently recurs in
  `metropolis_hastings.py:117-119` (`accept_mask.index_select(1, ...id.index_values)`),
  applying one patient's MH accept/reject decision to another patient's state.
- The one code path that does correct ID-keyed (dict) lookups,
  `StatisticalModel.single_patient_likelihood_factory`
  (`pynlme/model.py:643-691`), is never called anywhere — dead code, provides
  no protection.

**Reproduction (executed):** two patients `"p2"`, `"p1"`, in that row order in
the input dataframe (`self.patients == ["p2", "p1"]`, but sorted
`ref_values == ["p1", "p2"]`). Ground-truth per-patient prediction: p1→10.0,
p2→1484.13. Batched `predict_all_patients()` returned `[10.0, 1484.13]` for
observation rows `[p2, p1]` — i.e. **exactly swapped**: p2's observations get
scored against p1's simulated trajectory and vice versa. No error, no shape
mismatch — it's a silent positional swap. Script:
`/tmp/nix-shell.2z974x/claude-1000/-home-eliott-tixier-git-vpop-calibration/b5268590-9d01-4fe4-bb0d-551c681c109d/scratchpad/repro_patient_mismatch.py`.

**Why untested:** every test fixture and the golden benchmark CSV uses patient
IDs that happen to already be in sorted order (`"p1","p2",...` or zero-padded
`"patient_0000"`), so first-appearance order and sorted order coincide by
construction and the bug never triggers.

**Impact:** corrupts the likelihood, MCMC accept/reject, SAEM parameter
estimation, and all diagnostics (IWRES/PWRES/NPDE) across every structural
model backend, for any dataset where patient rows aren't already in ID-sorted
order in the file (e.g. sorted by visit date/site, or unpadded numeric IDs
like `"patient_10"` sorting before `"patient_2"`).

**Suggested fix:** make `DataIndex.from_dataframe`'s `ref_values` for the `id`
field use first-appearance order (drop the `.sort_values()` for that field) so
it matches `ObsData.patients` — lower-risk than patching every one of the 5
call sites that assume positional alignment.

### 2. PWRES uses variance of observations instead of predictive variance

`pynlme/diagnostics.py:198` (`compute_pwres`): `torch.cov(obs_patient.T)`
should be `torch.cov(this_patient_data.T)` (the Monte-Carlo simulated
predictions). `obs_patient` is 1-D, so `torch.cov` collapses to a scalar over
time points — statistically meaningless — and the multivariate/Cholesky
branch becomes dead code. (Matches `IMPLEMENTATION_REVIEW.md` #9.)

### 3. Importance-sampling Student-t proposal uses variance where it needs std-dev

`pynlme/importance_sampling.py:60-65`: `sigma = torch.var(etas, 0)` is passed
directly as `scale=` to `dist.StudentT`, which expects a std-dev-like scale,
not a variance. Distorts importance weights and the estimated marginal
log-likelihood whenever `var(etas) != 1`. (Matches
`IMPLEMENTATION_REVIEW.md` #8, which measured proposals ~77x too narrow and
likelihood estimates off by ~2 log-units.)

### 4. Continuous-output likelihood mask uses raw sigma values, not the boolean flags

`pynlme/residuals.py:224-226`: `torch.logical_or(residual_error.sigma_add,
residual_error.sigma_prop)` treats any nonzero float as "on" instead of using
the purpose-built `additive_output`/`proportional_output` flags. If a sigma is
exactly `0.0` (a valid prior, or reached via `clamp_min(0.0)`), that output's
entire log-likelihood contribution silently becomes 0 regardless of fit
quality. (Matches `IMPLEMENTATION_REVIEW.md` #2.)

### 5. `analytical.py` structural model can feed the wrong protocol-override column into user equations

`structural_model/analytical.py:53-62`: unlike `sbml.py`/`simwork.py`, doesn't
filter `protocol_design` down to `self.protocol_parameters` before building
the override tensor, so extra/reordered override columns shift the index
mapping and equations get called with mismatched argument values — no error,
just wrong simulation output. (Matches `IMPLEMENTATION_REVIEW.md` #7.)

### 6. NaN poisoning of a censored patient's entire log-posterior

`pynlme/residuals.py:192`: `ll_surv = event_status * log_hz_predicted -
cumulative_hz_predicted`. For a censored subject (`event_status == 0`), if the
predicted log-hazard is exactly `-inf` (hazard = 0), IEEE arithmetic gives
`0 * -inf = NaN` instead of the intended `0`, corrupting that patient's entire
log-posterior (continuous outputs included), which then breaks MH
accept/reject and SAEM updates for that patient. Medium confidence — depends
on the hazard model reaching exactly zero.

## P2 — Crashes, dead code, and latent issues

### 7. `mode="cli"`/`mode="debug"` SAEM config knobs are no-ops

`saem/config.py:32-44`: `live_plot`, `logging`, `progress_bars` are computed
as `NamedTuple` class-body expressions evaluated once at class-definition
time (against the literal default `"notebook"`), not per-instance from the
`mode` actually passed in. `SaemConfigDict(mode="cli")` (used as the SDK's
default, `sdk/config.py:9`) still has `live_plot=True, progress_bars=True`.
(Matches `IMPLEMENTATION_REVIEW.md` #12.)

### 8. `OptimizerPlot` crashes with `AttributeError` when IPython isn't installed

`saem/plot.py:56-70`: `self.handle` is only ever assigned inside
`if display is not None:`. If matplotlib is present but IPython isn't (more
likely to occur because of bug #7 above), the next `update()` call hits
`if self.handle is not None:` on an attribute that was never set. (Matches
`IMPLEMENTATION_REVIEW.md` #13.)

### 9. `compute_iwres` asserts the wrong tensor shape when a survival submodel is present

`pynlme/diagnostics.py:88-93`: the assertion on
`map_physical_params_samples.shape` omits `+ self.model.nb_surv_coeffs` from
the expected last dimension. Any model with a `time_to_event` block raises
`AssertionError` on every call to `compute_iwres()`.

### 10. `load_nlme_model`'s "no plots/progress bars" promise is never implemented

`sdk/model.py:126`: comment says the output mode is overridden for loaded
models, but no code does it — a model exported interactively
(`live_plot=True`) and reloaded in a headless/batch context still tries to
plot / render progress bars.

### 11. `get_data_inputs` can `KeyError` or silently mutate the caller's DataFrame

`model/data.py:371-405`: when `protocol_arm` is missing, the default
`"identity"` value is computed into a local variable but never assigned back
to `new_data`, so the subsequent `.pivot(columns=["output_name",
"protocol_arm"])` raises `KeyError`. Separately, `new_data = data_set` aliases
the caller's frame (no `.copy()`); when `value` is missing,
`new_data["value"] = 1.0` permanently mutates the caller's own DataFrame, so a
second call on the same object silently skips generating a fresh dummy
column.

### 12. `normalize_inputs_tensor` mutates the caller's tensor in place

`model/data.py:296-304`, reached via `structural_model/gp.py:42-43`
(`StructuralGp.simulate`): `vpop_calibration/config.py` calls
`torch.set_default_device(device)`, so `X = inputs.to(device)` is a no-op
(returns `self`) when `inputs` is already on that device. The subsequent
`X[:, self.log_inputs_indices] = torch.log(...)` therefore overwrites the
caller's original tensor's log-scaled columns in place as a side effect of
what looks like a pure prediction call.

### 13. No floor before `log()` at prediction time (inconsistent with training time)

`model/data.py`: training-time log-scaling floors with
`np.log(np.maximum(val, self.log_lower_limit))`; `normalize_inputs_tensor`
(prediction time) does a bare `torch.log(...)` with no floor. A sampled
parameter reaching ≤ 0 produces `-inf`/`NaN`, silently freezing/corrupting any
downstream MH chain that depends on it.

### 14. `consecutive_converged_iters`/`patience` bookkeeping is dead

`saem/optimizer.py:59-67,395-416`, `saem/config.py:29`
(`patience: int = 5`): convergence is tracked per iteration but never read
against `patience` anywhere in the repo to stop early or warn — changing
`patience` has zero effect on run behavior.

### 15. `reproducible_uuid4` permanently reseeds the global RNG

`utils.py:33-36`: when called with a `seed`, reseeds the process-global
`random` module and never restores prior state, corrupting any other code's
`random` stream in the same process. Currently dormant — the only call site
(`pynlme/conditional_distribution.py:325`) never passes `seed` — but a latent
bug in the function's contract.

### 16. Shape-inconsistent aggregation in `total_samples`

`pynlme/conditional_distribution.py:279-290`: `eta`/`physical` use
`torch.cat` (merging the leading singleton batch dim) while `pred`/`log_prob`
use `torch.stack` (adding a new dim), so the four fields of the returned
object aren't shape-consistent with each other. Not currently triggered
because existing consumers only touch one field at a time, but breaks the
object's contract for any future caller indexing across fields together.

## Regression tests

The following bugs are now covered by failing tests under
`vpop_calibration/test/known_bugs/`, so they show up as `FAILED` in a normal
`pytest` run until fixed:

- Patient ID ordering (#1): `test_patient_id_ordering.py`
- PWRES/NPDE/VPC ignoring residual noise (#2, and the two items below carried
  over from `IMPLEMENTATION_REVIEW.md`): `test_diagnostics_noise.py`
- Importance-sampling variance-as-scale (#3): `test_importance_sampling_scale.py`
- Zero-sigma removes an output from the likelihood (#4): `test_residuals_zero_variance.py`
- Analytical protocol-override column order (#5): `test_analytical_protocol_overrides.py`
- `mode="cli"`/`mode="debug"` config no-ops (#7): `test_saem_config_mode.py`
- `simulate_from_df()` wrong output/protocol selection (`IMPLEMENTATION_REVIEW.md` #6): `test_structural_simulate_from_df.py`
- SBML initial-time shift (`IMPLEMENTATION_REVIEW.md` #5): `test_sbml_initial_time.py`
- Fixed-effects multi-chain objective sign error (`IMPLEMENTATION_REVIEW.md` #3): `test_saem_fixed_effects_chains.py`
- Population regression ignoring random-effect correlations (`IMPLEMENTATION_REVIEW.md` #4): `test_saem_mstep_covariance.py`

Not yet covered by a test: #6 NaN poisoning for censored survival subjects,
and all the P2 crashes/dead-code items (#8-#16), since those are either
crash-on-call (already self-evident) or dead-code/latent issues rather than
silent wrong-number bugs.

## Not independently flagged here but see `IMPLEMENTATION_REVIEW.md`

- #3 Fixed-effect optimization moving in the wrong direction with multiple MH chains (`saem/optimizer.py:291-293`)
- #4 Population regression ignoring random-effect correlations (`saem/m_step.py:118-122`)
- #5 SBML predictions shifting when observations start after time zero (`structural_model/sbml.py:26-28`)
- #6 `simulate_from_df()` selecting the wrong output/protocol (`structural_model/base.py:43-44`)
- #10 NPDE ignoring observation noise (`pynlme/diagnostics.py:240-247`)
- #11 VPC prediction intervals omitting observation noise (`pynlme/diagnostics.py:326`)

These were confirmed with executed reproductions in that file and are not
re-derived here.
