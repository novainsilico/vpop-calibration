Yes. The document needs a substantive revision, not just copy-editing. I found three major mathematical errors plus significant drift from the current implementation. I did not modify any files.

## Major correctness issues

1. **The observed-data likelihood is wrong.** At [nlme_model.md:152](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:152), the latent individual parameter is not integrated out. It should be

$$
\ell(\Theta;\mathbf y)
=\sum_i\log\int
p(\mathbf y_i\mid\eta_i;\Theta)
p(\eta_i\mid\Omega)\,d\eta_i.
$$

The displayed product is instead the complete-data joint density:

$$
\ell_c
=\sum_i\left[
\log p(\mathbf y_i\mid\eta_i;\Theta)
+\log p(\eta_i\mid\Omega)
\right].
$$

These two quantities should be distinguished explicitly; SAEM uses the complete-data formulation to target the marginal likelihood.

2. **The transformed density is missing a Jacobian.** At [nlme_model.md:181](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:181),

$$
p(\theta_i\mid\Theta)\neq p(\eta_i\mid\Omega)
$$

when $\theta_i=\exp(X_i\beta+\eta_i)$. On the physical scale,

$$
\log p(\theta_i\mid\beta,\Omega)
=\log p(\eta_i\mid\Omega)-\sum_k\log\theta_{i,k}.
$$

The cleanest fix is to formulate the complete-data likelihood directly in terms of the latent $\eta_i$, matching [model.py:368](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/model.py:368).

3. **The meaning of `sigma` disagrees with the code.** The equations at [nlme_model.md:118](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:118) treat $\sigma$ as a standard deviation:

$$
V_{\mathrm{add}}=\sigma^2,\qquad
V_{\mathrm{prop}}=\sigma^2f^2.
$$

The implementation treats configured `sigma` values directly as variance coefficients:

$$
V=\texttt{sigma\_add}+\texttt{sigma\_prop}f^2,
$$

as shown in [residuals.py:79](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/residuals.py:79) and [error_estimation.py:60](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/error_estimation.py:60). Thus configured `sigma=0.1` currently means variance $0.1$, not $0.01$. Either the implementation must square SD inputs, or the document should use $y=f+\sqrt{\sigma}\varepsilon$.

The conditional expectation and variance equations themselves are otherwise correct under the document’s SD convention.

## Mathematical and modeling inconsistencies

4. At [nlme_model.md:50](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:50) and line 175, the matrix multiplication is backwards. Given the displayed dimensions, it must be

$$
\log\phi_i=X_i\beta+\eta_i,
$$

which also matches [model.py:231](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/model.py:231).

5. The parameter count at [nlme_model.md:58](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:58) omits one intercept per PDU:

$$
n_\beta=n_{\mathrm{PDU}}+\sum_k n_k.
$$

The same display has $c_{1,2}$ where the second block should begin with $c_{2,1}$, and the second row’s covariates lack the individual superscript $i$.

6. The document says observations can be vector-valued at [nlme_model.md:14](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:14), but every subsequent equation is scalar. The implementation also stores scalar observations in long format and assumes independent output errors; see [schemas.py:5](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/schemas.py:5). Either introduce an output index $k$, or state that the equations apply componentwise with conditional independence.

7. The independence assumptions needed for the likelihood are missing. It should state that residuals are independent across observations and outputs, independent of $\eta_i$, and that random effects are independent across individuals. Merely writing $\varepsilon_{ij}\sim\mathcal N(0,1)$ does not establish these facts.

8. The definition of $\theta_i$ is inconsistent around [nlme_model.md:38](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:38). Covariates are listed as components of the patient descriptors but are explicitly not structural-model inputs. A clearer decomposition is

$$
\theta_i=(\nu_i,\phi_i,\psi),
$$

with covariates represented separately in $X_i$. Since MI parameters are population-shared, use $\psi$, not $\psi_i$.

9. Calling $\mu_k$ a “mean” is misleading. Under a lognormal model, $\mu_k=\exp(\beta_k)$ is the conditional median/geometric typical value, while

$$
\operatorname E[\phi_{i,k}\mid X_i]
=\exp\left((X_i\beta)_k+\frac12\Omega_{kk}\right).
$$

10. The claim that exponential effects imply continuous covariates at [nlme_model.md:79](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:79) is false: binary indicator variables work in a log-linear model. The actual implementation restriction is that covariates must be numeric.

11. The likelihood terminology is reversed in several places:

- [Line 148](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:148) should say “likelihood of the observations as a function of $\Theta$,” not parameters given observations.
- [Line 161](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:161) describes observations given $\theta_i$, not $\theta_i$ given observations.
- Maximum likelihood also applies to shared MI parameters; only their optimization procedure differs.

## Stale implementation documentation

- [Lines 24–34](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:24) link to nonexistent `structural_model.py` and use nonexistent `StructuralOdeModel`. Current exported implementations are `StructuralGp`, `StructuralAnalytical`, `StructuralSimwork`, and `StructuralSbml`; see [api/__init__.py:7](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/api/__init__.py:7).

- The configuration example at [line 90](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:90) cannot validate against the current API. The current schema uses `model_intrinsic`, `pdu`, `pdk`, `error_model`, `prior`, `prior_omega`, and nested `covariates`; see [params.py:46](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/params.py:46). Its `cov_foo_k12` name is also attached to `k_21`.

- “Only log-normal” is outdated. Shifted-log and bounded-logit transforms are supported in [params.py:11](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/params.py:11). A general formulation $h(\phi_i)=X_i\beta+\eta_i$ would match the implementation better.

- [Line 134](/home/eliott.tixier/git/vpop-calibration/docs/nlme_model.md:134) incorrectly says only additive and proportional errors exist. Combined errors and survival likelihoods are now supported.

- Both `nlme.py` links are broken. `log_likelihood_observation` is now in [residuals.py:267](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/residuals.py:267), while `log_prior_etas` is in [model.py:368](/home/eliott.tixier/git/vpop-calibration/vpop_calibration/pynlme/model.py:368).

- The document omits protocol-arm dependence and does not mention that GP predictive variance is currently returned but discarded when computing the observation likelihood.

Smaller fixes include consistent $t_{i,j}$ notation, `\mathbf y` instead of `\bf y`, `\operatorname{Var}`, “divided into,” “covariate coefficient,” and the `maximimze` typo. The Lindstrom–Bates citation and DOI are valid according to the [indexed publication record](https://pubmed.ncbi.nlm.nih.gov/2242409/).

Overall, I would rewrite the likelihood and implementation sections and revise the remaining sections around a consistent transformed-Gaussian notation.
