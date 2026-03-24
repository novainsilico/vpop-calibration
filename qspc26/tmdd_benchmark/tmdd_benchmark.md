# TMDD Benchmarking

This benchmark is designed to demonstrate the benefits of using a surrogate model + SAEM approach as opposed to the standard implementation. The "standard implementation" is hereby represented as the R package [nlmixr](https://nlmixrdevelopment.github.io/nlmixr/index.html). The figures introduced here are part of the poster presented by Nova In Silico at [QSPC 2026, Leiden](https://www.qspc.eu/qspc2026/home). The full poster itself is available in [this document](./outputs/2026_QSPC_Vpop_calibration.pdf).

## Data generation

The model is a target-mediated drug disposition (TMDD) model, described for instance in [Dua et al. (2015)](https://doi.org/10.1002/psp4.41), that contains 3 ODEs:

```math
\begin{align*}
\frac{dL}{dt} =& -k_{eL} L -k_{on} LR + k_{off} P, \\
\frac{dR}{dt} =& k_{syn} - k_{deg} R -k_{on} LR +k_{off} P, \\
\frac{dP}{dt} =& k_{on} LR + k_{off} P - k_{eP} P, \\
DV = & \log(\frac{L}{V_c})\\
&L(0) = 0, R(0) = R_0 = \frac{k_{syn}}{k_{deg}}, P(0) = 0, \\
&\frac{k_{off}}{k_{on}} = K_D
\end{align*}
```

The synthetic reference data is generated in [`generate_synthetic_data.R`](./generate_synthetic_data.R), using the [`rxode2`](https://nlmixr2.github.io/rxode2/index.html) package for efficient implementation of ODE system solving.

### Reference parameter values

The following parameters are considered as known and fixed:

| Parameter | Description | Reference value |
| --- | --- | --- |
| $`k_{off}`$ | Unbinding constant rate | $`1\ h^{-1}`$ |
| $`k_{D}`$ | Dissociation constant | $`0.5\ nM`$ |
| $`k_{on}`$ | On-rate constant | $`\frac{k_{off}}{k_D}`$ |
| $`k_{deg}`$ | Receptor elimination rate | $`0.5\ h^{-1}`$ |
| $`k_{syn}`$ | Receptor synthesis rate | $`R_0 \times k_{deg}`$ |
| $`k_{eP}`$ | Bound antibody elimination rate  | $`0.2\ h^{-1}`$ |

The following parameters are unknwon, and their target population values are described below:

| Parameter | Description | Reference value (mean) | Spread (log) |
| --- | --- | --- | --- |
| $`V_c`$ | Distribution volume | $`3L`$ | $`\omega^2 = 0.5`$ |
| $`R_0`$ | receptor amount | $`1\ nanomole`$ | $`\omega^2 = 0.15`$ |
| $`k_{eL}`$ | Free antibody elimination rate | $`0.5\ h^{-1}`$ | $`\omega^2 = 0.2`$ |

### Dosing regimen

The antibody administration is represented as an IV bolus, with two dosings separated by `12 h`. The synthetic data is exported in two versions

- `1dose`: a single dose of `100 nanomole` is injected for all patients
- `2dose`: half of the patients receive `100 nmol`, half of them receive `10 nmol`

## Convergence benchmark

The convergence of the algorithm is tested against a reference data set simulated with the TMDD model. The comparison can be performed by running

- [`benchmark_pysaem_convergence.ipynb`](./benchmark_pysaem_convergence.ipynb)
- [`benchmark_nlmixr_convergence.R`](./benchmark_nlmixr2_convergence.R)

The results are summarized in `convergence_nlmixr.csv` and `convergence_pysaem.csv`, which are used by the [`generate_comparison_plots.ipynb`](./generate_comparison_plots.ipynb) notebook.

## Runtime benchmark

The comparison of runtimes for increasing sizes of data sets is performed in

- [`benchmark_pysaem_runtime.ipynb`](./benchmark_pysaem_runtime.ipynb)
- [`benchmark_nlmixr_runtime.R`](./benchmark_nlmixr2_runtime.R)

The results are aggregated in `performance_nlmixr.csv` and `performance_pysaem.csv`, which are used by the [`generate_comparison_plots.ipynb`](./generate_comparison_plots.ipynb) notebook.

The performance of PySaem is evaluated by accounting for both surrogate model training, which accounts for an incompressible time of ~15s, cumulated with the runtime of SAEM on the surrogate model. In this very specific setup, the use of a surrogate model begins to be beneficial for data sets of >500 patients. For very large QSP models, this threshold is most likely much lower.

## Benchmark plots

The latest benchmarking plots, as included in the QSPC poster, are shown below

![full benchmark](./outputs/tmdd_benchmark.png)
