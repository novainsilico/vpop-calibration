# FIM implementation in fim.py



**Notation**

- $i = 1..N$: patients; $j$: observation times; $o$: outputs.
- $m = 1..n$: conditional samples (iterations × chains).
- $\mathcal T$: transform to the gaussian scale (log, or logit if bounded). We have $\mathcal T(\phi_i) = X_i\beta + \eta_i$ and $\Omega = LL^T$.
- $\psi$: model intrinsic parameters (MI); $\kappa$: survival coefficients.
- $v_{ijo} = a_o^2 + b_o^2 f_o^2$: residual variance.
- $h_i$: hazard; $\Lambda_i$: cumulative hazard; $\delta_i$: event indicator.
- $g_i$: score; $H_i$: hessian.
- $\Delta_q$: finite-difference step; $G_k$: link function of a displayed parameter.

---

## Constants

$$
\texttt{IDENTITY\_LINK} = (x \mapsto x,\ y \mapsto y)
\qquad
\texttt{LOG\_LINK} = (\log,\ \exp)
\qquad
\texttt{ATANH\_LINK} = (\operatorname{atanh},\ \tanh)
$$

Each pair is $(G, G^{-1})$.

---

## `constraint_link(constraint)`

Returns $(G, G^{-1}) = (\mathcal T, \mathcal T^{-1})$ for a constraint $(\text{low}, \text{high})$, with $\text{shift} = \text{low}$ and $\text{scale} = \text{high} - \text{low}$.

- No upper bound:
  $$
  G(x) = \log(x - \text{shift}),\qquad G^{-1}(y) = \text{shift} + e^{y}
  $$
- With an upper bound:
  $$
  G(x) = \operatorname{logit}\Big(\frac{x - \text{shift}}{\text{scale}}\Big),\qquad G^{-1}(y) = \text{shift} + \text{scale}\cdot\operatorname{sigmoid}(y)
  $$

---


## `class FimAccumulator`

### `__init__(model, finite_difference_step)`



$$
\tilde{\Theta}
=
\left(
\beta,\,
(\log L_{kk})_k,\,
(L_{kl})_{k>l},\,
p_{pdu}=n_\beta+\frac{n_{pdu}(n_{pdu}+1)}{2},\,
\mathcal{T}(\psi),\,
\kappa,\,
n_{mi}+n_\kappa,\,
\log a_o^2,\,
\log b_o^2
\right)
$$

We write $p = \dim\tilde\Theta$ and $\varepsilon = \texttt{finite\_difference\_step}$.



### `pdu_derivatives(gaussian_params)`

**Method: automatic differentiation (autograd).** This block is a closed-form gaussian density that does not involve the structural model, so its score and hessian are computed exactly.

Input: $\mathcal T(\phi_i^{(m)})$ for each sample $m$ and each patient $i$.

`log_likelihood_pdu(parameters, gaussian_row, design_row)`: for one pair $(m, i)$, with $(\beta, L)$ read from `parameters`:

$$
\eta_i(\beta) = \mathcal T(\phi_i) - X_i\beta,\qquad
z_i = L^{-1}\eta_i(\beta)
$$

$$
\log p(\phi_i;\ \beta, \Omega) = -\frac{n_{pdu}}{2}\log 2\pi - \sum_k \log L_{kk} - \frac12\lVert z_i\rVert^2 + \text{Constant}
$$

The constant is $\sum_k \log|\mathcal T_k'(\phi_{ik})|$. It depends neither on $\beta$ nor on $\Omega$, so it does not contribute to $g_i$ or $H_i$, and the code omits it.

Output, computed by autograd for each pair $(m, i)$:

$$
g_i^{pdu} = \nabla_{(\beta,\log L_{kk},L_{kl})}\log p(\phi_i;\beta,\Omega),\qquad
H_i^{pdu} = \nabla^2_{(\beta,\log L_{kk},L_{kl})}\log p(\phi_i;\beta,\Omega)
$$

### `observation_derivatives(gaussian_params, predictions)`

**Method: finite differences.** $\psi$ and $\kappa$ act through the structural model $f$, which is generally not differentiable by autograd (e.g. external ODE solvers), so the score and hessian of this block are approximated numerically.

Inputs: $\mathcal T(\phi_i^{(m)})$ and the stored predictions $f_o(\theta_i^{(m)}, t_{ij})$.

Block parameters and steps:

$$
\tilde\Theta^{obs} = \big(\mathcal T(\psi),\ \kappa,\ \log a_o^2,\ \log b_o^2\big),\qquad
\Delta_q = \varepsilon\,\max\big(|\tilde\Theta^{obs}_q|,\ 1\big)
$$

**Perturbations.** To compute derivatives by finite differences, the likelihood is evaluated at slightly shifted parameter values $\vartheta = \tilde\Theta^{obs} + \text{perturbation}$, where the perturbation is one of

$$
0,\qquad \pm\Delta_q e_q\ \ (\text{one coordinate } q \text{ shifted}),\qquad \pm\Delta_q e_q \pm \Delta_r e_r\ \ (\text{two coordinates } q, r \text{ shifted})
$$

with $e_q$ the $q$-th basis vector. A perturbation "touches" $\psi$ or $\kappa$ when it has a non-zero component on one of their coordinates.

`log_likelihood_perturbated_observations(perturbation)`: for $\vartheta = \tilde\Theta^{obs} + \text{perturbation}$, it proceeds in three steps.

1. **Predictions.**
   - If the perturbation touches $\psi$ or $\kappa$, $f_o(\theta_i, t_{ij})$ is re-simulated with $\theta_i = \big(PDK_i,\ \phi_i,\ \psi,\ \kappa\big)$, where $\psi, \kappa$ are read from $\vartheta$. 
   - Otherwise the stored predictions are reused: perturbing a variance does not change $f$.
2. **Residual error model.** A copy of `model.residual_var` is made in which the active variances are replaced by the perturbed ones, $a_o^2 = e^{\vartheta_{\log a_o^2}}$ and $b_o^2 = e^{\vartheta_{\log b_o^2}}$. Inactive components are left untouched.
3. **Call to `log_likelihood_observation`** (from `residuals.py`):

   ```python
   log_likelihood_observation(
       model.data.full_obs,                  # observations y_ijo, with patient / output / time indices
       current_predictions,                  # f_o(θ_i, t_ij), shape (samples, n_obs)
       residual_var._replace(...),           # a_o², b_o² from step 2
       model.config.residual_min_variance,   # floor on v_ijo
   )
   ```

   It returns one value per (sample, patient), shape (samples, N), as the sum of two parts:
   - `compute_normal_likelihood`, over continuous outputs only:
     $$
     \sum_{j,o}\Big[-\frac12\log(2\pi\,v_{ijo}) - \frac{(y_{ijo} - f_o(\theta_i, t_{ij}))^2}{2\,v_{ijo}}\Big],\qquad v_{ijo} = \max\big(a_o^2 + b_o^2 f_o^2,\ v_{min}\big)
     $$
     Observation rows are summed per patient.
   - `compute_survival_likelihood`: $\delta_i\log h_i(T_i) - \Lambda_i(T_i)$, read from the `log_<hazard>` and `cumulative_<hazard>` outputs of the predictions. It is zero if the model has no survival output.

   This gives:

$$
\ell_i(\vartheta) = \log p(y_i \mid \phi_i;\ \psi,\kappa,\sigma^2) = \sum_{j,o}\Big[-\frac12\log(2\pi\,v_{ijo}) - \frac{(y_{ijo} - f_o(\theta_i, t_{ij}))^2}{2\,v_{ijo}}\Big] + \delta_i\log h_i(T_i) - \Lambda_i(T_i)
$$

Evaluations, one per perturbation listed above:

$$
\ell_0 = \ell(\tilde\Theta^{obs}),\qquad
\ell_q^{\pm} = \ell(\tilde\Theta^{obs} \pm \Delta_q e_q),\qquad
\ell_{qr}^{\pm\pm} = \ell(\tilde\Theta^{obs} \pm \Delta_q e_q \pm \Delta_r e_r)
$$

Output, for each pair $(m, i)$:

$$
(g_i^{obs})_q = \frac{\ell_q^+ - \ell_q^-}{2\Delta_q},\qquad
(H_i^{obs})_{qq} = \frac{\ell_q^+ - 2\ell_0 + \ell_q^-}{\Delta_q^2},\qquad
(H_i^{obs})_{qr} = (H_i^{obs})_{rq} = \frac{\ell_{qr}^{++} - \ell_{qr}^{+-} - \ell_{qr}^{-+} + \ell_{qr}^{--}}{4\Delta_q\Delta_r}
$$

### `accumulate(eta_samples, predictions)`

**Why $H_i$ is block diagonal.** With $\phi_i$ fixed, the complete likelihood factorizes into two terms that share no parameter:

$$
p(y_i, \phi_i;\ \tilde\Theta) = p(y_i \mid \phi_i;\ \psi, \kappa, \sigma^2)\; p(\phi_i;\ \beta, \Omega)
$$

so its log is a sum of a function of $(\psi, \kappa, \sigma^2)$ and a function of $(\beta, \Omega)$. All cross second derivatives vanish:

$$
H_i = \nabla^2_{\tilde\Theta}\log p(y_i, \phi_i;\ \tilde\Theta) =
\begin{pmatrix} \nabla^2_{(\beta,\Omega)}\log p(\phi_i;\beta,\Omega) & 0 \\ 0 & \nabla^2_{(\psi,\kappa,\sigma^2)}\log p(y_i \mid \phi_i;\psi,\kappa,\sigma^2) \end{pmatrix}
$$

For each sample $\eta^{(m)}$ of the batch:

$$
\mathcal T(\phi_i^{(m)}) = X_i\hat\beta + \eta_i^{(m)}
$$

$$
g_i^{(m)} = \begin{pmatrix} g_i^{pdu} \\ g_i^{obs} \end{pmatrix},\qquad
H_i^{(m)} = \begin{pmatrix} H_i^{pdu} & 0 \\ 0 & H_i^{obs} \end{pmatrix}
$$

Sums are updated as:

$$
n \mathrel{+}= \#\text{samples},\qquad
S_H \mathrel{+}= \sum_m\sum_i H_i^{(m)},\qquad
S_{gg} \mathrel{+}= \sum_m\sum_i g_i^{(m)} g_i^{(m)T},\qquad
S_{g,i} \mathrel{+}= \sum_m g_i^{(m)}
$$

The total score of each sample, $G^{(m)} = \sum_i g_i^{(m)}$, is also stored (`total_score_per_sample`), to follow the convergence score as samples accumulate.

### `observed_fisher_information()`

Louis' formula:

$$
\bar g_i = \frac{S_{g,i}}{n},\qquad
\hat I = -\Big(\frac{S_H}{n} + \frac{S_{gg}}{n} - \sum_{i=1}^N \bar g_i\,\bar g_i^T\Big),\qquad
\hat I \leftarrow \frac12\big(\hat I + \hat I^T\big)
$$

---

## `get_displayed_parameters(model)`

`from_link(link)`: $(G, G^{-1}) \mapsto (G, G^{-1}, G^{-1})$. The link serves both for the confidence interval and for mapping $\tilde\Theta_k$ to $\Theta_{aff,k}$.

For each component of $\Theta_{aff}$, returns the triple (name, $(G_k, G_k^{-1})$, map $\tilde\Theta_k \mapsto \Theta_{aff,k}$):

| Block | Name | $G_k$ | $\tilde\Theta_k \mapsto \Theta_{aff,k}$ |
|---|---|---|---|
| `betas` (intercept) | $\mu_k$ | $\mathcal T$ | $\mathcal T^{-1}(\beta_k)$ |
| `betas` (covariate) | coefficient | identity | $\beta_k$ |
| `omega_sd` | $\omega_k$ | $\log$ | computed in `display_transformation` |
| `omega_corr` | $\mathrm{corr}_{kl}$, $k > l$ | $\operatorname{atanh}$ | computed in `display_transformation` |
| `mis` | $\psi$ | $\mathcal T$ | $\mathcal T^{-1}(\mathcal T(\psi))$ |
| `surv_coeffs` | $\kappa$ | identity | $\kappa$ |
| `residual_sds` | $a_o$, $b_o$ | $\log$ | $e^{\frac12\log a_o^2}$, $e^{\frac12\log b_o^2}$ |

---

## `compute_estimates_uncertainty(accumulator, model)`

Internal covariance, via Cholesky if $\hat I$ is positive definite and via pseudo-inverse otherwise:

$$
C_{\tilde\Theta} = \hat I^{-1}
$$

`display_transformation(parameters)`: $\tilde\Theta \mapsto \Theta_{aff}$,

$$
\Omega = LL^T,\qquad
\omega_k = \sqrt{\Omega_{kk}},\qquad
\mathrm{corr}_{kl} = \frac{\Omega_{kl}}{\omega_k\,\omega_l}
$$

The other components are mapped element-wise (see the table above).

Covariance of the estimates on the displayed scale:

$$
J = \frac{\partial\Theta_{aff}}{\partial\tilde\Theta}\Big|_{\hat{\tilde\Theta}},\qquad
C_{aff} = J\,C_{\tilde\Theta}\,J^T
$$

Standard errors and intervals:

$$
SE_k = \sqrt{(C_{aff})_{kk}},\qquad
RSE_k = 100\,\frac{SE_k}{|\hat\Theta_k|}
$$

$$
CI95_k = G_k^{-1}\Big(G_k(\hat\Theta_k) \pm 1.96\,SE_k\,\big|G_k'(\hat\Theta_k)\big|\Big)
$$

**Convergence score.** At the maximum likelihood the gradient of the observed log-likelihood vanishes. By Fisher's identity it is the expected complete score, estimated by the mean over the first $n$ samples:

$$
\nabla\ell(\hat\Theta) = \sum_i \mathbb E\big[g_i \mid y_i\big] \approx \bar G(n) = \frac1n\sum_{m=1}^{n} G^{(m)}
$$

The Newton step $J\,C_{\tilde\Theta}\,\bar G(n)$ estimates how far each displayed parameter is from the maximum likelihood; dividing by $SE_k$ expresses it in standard errors:

$$
\texttt{convergence\_score}_k(n) = \frac{\big(J\,C_{\tilde\Theta}\,\bar G(n)\big)_k}{SE_k}
$$

$J$, $C_{\tilde\Theta}$ and $SE_k$ are the final ones; only $\bar G(n)$ varies with $n$. If SAEM converged, the score fluctuates around 0 and shrinks as $n$ grows (Monte-Carlo noise); a plateau away from 0 means the parameter is not at the maximum likelihood.

Outputs:

- `fisher_information` = $I_{aff} = J^{-T}\,\hat I\,J^{-1}$, the FIM on the reported scale (since $C_{aff} = J C_{\tilde\Theta} J^T$ and $C = I^{-1}$), with the same labels as `covariance`;
- `normalized_fisher_information` = $r_{qr} = (I_{aff})_{qr}/\sqrt{(I_{aff})_{qq}(I_{aff})_{rr}}$ and `correlation` = $R_{qr} = (C_{aff})_{qr}/\sqrt{(C_{aff})_{qq}(C_{aff})_{rr}}$, both computed by `normalize_matrix`;
- `covariance` = $C_{aff}$;
- `standard_errors` = $\big(\hat\Theta_k,\ SE_k,\ RSE_k,\ CI95_k,\ \texttt{convergence\_score}_k(n_{tot})\big)_k$;
- `convergence_score_history` = $\big(\texttt{convergence\_score}_k(n)\big)_{n,k}$, plotted by `PlottingUtility.convergence_score()`.
