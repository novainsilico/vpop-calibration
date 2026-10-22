# Tensor shape notation

To keep the code concise, tensor shapes are written with single-letter
dimension names, e.g. as a comment next to a tensor:

```python
etas = mh_state.etas  # (C, P, D)
best = samples.log_prob.argmax(dim=0)  # (P,)
```

| Symbol | Dimension | Code equivalent |
|---|---|---|
| `C` | Number of MCMC chains | `model.nb_chains` |
| `P` | Number of patients | `model.nb_patients` |
| `D` | Number of PDUs | `model.nb_pdu` |
| `F` | Number of fixed effects (MI + survival coeffs) | `model.num_fixed_effects()` |
| `Y` | Number of observations | `data.nb_total_observations` |
| `N` | Number of samples | `nb_samples` |
