# TMDD Benchmarking

## Data generation

This notebook benchmarks the PySaem implementation on reference data.

The model is a target-mediated drug disposition (TMDD) model, described for instance in [Dua et al. (2015)](https://doi.org/10.1002/psp4.41), that contains 3 ODEs:

```math
\begin{align*}
\frac{dL}{dt} =& -k_{eL} L -k_{on} LR + k_{off} P, \\
\frac{dR}{dt} =& k_{syn} - k_{def} R -k_{on} LR +k_{off} P, \\
\frac{dP}{dt} =& k_{on} LR + k_{off} P - k_{eP} P, \\
&L(0) = L_0, R(0) = R_0 = \frac{k_{syn}}{k_{deg}}, P(0) = 0, \\
&\frac{k_{off}}{k_{on}} = K_D
\end{align*}
```

## Reference parameter values

The parameters' reference values and distributions are gathered in the table below. Distributed parameters follow a log-normal distribution.

| Parameter | Description | Assumption | Reference value | Spread (log) |
| --- | --- | --- | --- | --- |
| $`k_{off}`$ | Unbinding constant rate | Known | $`1 h^{-1}`$ | None |
| $`k_{D}`$ | Dissociation constant | Known | $`1 nM`$ | None |
| $`k_{on}`$ | On-rate constant | Known | $`\frac{k_{off}}{k_D}`$ | None |
| $`inj`$ | Administered concentration | Known | $`10 nanomole`$ | None |
| $`V_c`$ | Distribution volume | Unknown | $`3L`$ | $`\omega^2 = 0.1`$ |
| $`L_0`$ | Initial antibody concentration | Known | $`10 nM`$ | None |
| $`k_{deg}`$ | Receptor elimination rate | Known | $`1e-2 h^{-1}`$ | None |
| $`R_0`$ | Total receptor concentration | Known | $`3 nM`$ | None |
| $`k_{syn}`$ | Receptor synthesis rate | Known | $`R_0 \times k_{deg}`$ | None |
| $`k_{eL}`$ | Free antibody elimination rate | Unknown | $`5e-2 h^{-1}`$ | $`\omega^2 = 0.5`$ |
| $`k_{eP}`$ | Bound antibody elimination rate | Unknown | $`1e-1 h^{-1}`$ | $` \omega^2 = 0.1 `$ |
