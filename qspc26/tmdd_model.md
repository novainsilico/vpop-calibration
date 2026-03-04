# TMDD Benchmarking

This benchmark is designed to demonstrate the benefits of using a surrogate model + SAEM approach as opposed to the standard implementation. The "standard implementation" is hereby represented as the R package [nlmixr](https://nlmixrdevelopment.github.io/nlmixr/index.html).

## Data generation

The model is a target-mediated drug disposition (TMDD) model, described for instance in [Dua et al. (2015)](https://doi.org/10.1002/psp4.41), that contains 3 ODEs:

```math
\begin{align*}
\frac{dL}{dt} =& -k_{eL} L -k_{on} LR + k_{off} P, \\
\frac{dR}{dt} =& k_{syn} - k_{def} R -k_{on} LR +k_{off} P, \\
\frac{dP}{dt} =& k_{on} LR + k_{off} P - k_{eP} P, \\
DV = & \log(\frac{L}{V_c})\\
&L(0) = 0, R(0) = R_0 = \frac{k_{syn}}{k_{deg}}, P(0) = 0, \\
&\frac{k_{off}}{k_{on}} = K_D
\end{align*}
```

The synthetic reference data is generated in [`generate_synthetic_data.R`](./generate_synthetic_data.R), using the [`rxode2`](https://nlmixr2.github.io/rxode2/index.html) package for efficient implementation of ODE system solving.

## Reference parameter values

The following parameters are considered as known and fixed:

| Parameter | Description | Reference value |
| --- | --- | --- |
| $ `k_{off}` $ | Unbinding constant rate | $ `1\ h^{-1}` $ |
| $ `k_{D}` $ | Dissociation constant | $ `0.05\ nM` $ |
| $ `k_{on}` $ | On-rate constant | $ `\frac{k_{off}}{k_D}` $ |
| $ `inj` $ | Administered concentration | $ `100\ nanomole` $ |
| $ `k_{deg}` $ | Receptor elimination rate | $ `0.5\ h^{-1}` $ |
| $ `k_{syn}` $ | Receptor synthesis rate | $ `R_0 \times k_{deg}` $ |
| $ `k_{eP}` $ | Bound antibody elimination rate  | $ `0.2\ h^{-1}` $ |

The following parameters are unknwon, and their target population values are described below:

| Parameter | Description | Reference value | Spread (log) |
| --- | --- | --- | --- |
| $ `V_c` $ | Distribution volume | $ `3L` $ | $ `\omega^2 = 0.5` $ |
| $ `R_0` $ | receptor amount | $ `1\ nanomole` $ | $ `\omega^2 = 0.15` $ |
| $ `k_{eL}` $ | Free antibody elimination rate | $ `0.5\ h^{-1}` $ | $ `\omega^2 = 0.2` $ |
