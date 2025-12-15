# Non-linear mixed effect models implementation

This note is not intended to provide an exhaustive introduction to non-linear mixed effects (NLME) modeling. The goal is to introduce some notation and link it with the implemented code to support the user's understanding of the algorithms. The interested reader could refer to [^Lindstrom90].

## Statistical model

Consider a data set $\bf{y}$ of longitudinal observations for $N$ individuals. For each individual $i$, $y_{i,j}, j=1\dots m_j$ observations have been collected at times $t_j$. An NLME model associates a structural model $f$ and an error model $g$ to predict the observations as follows

```math
y_{i,j} = f(\theta_i,t_{i,j}) + g(\theta_i, t_{i,j}, \sigma) \varepsilon_{i,j}, \\
\varepsilon \sim  \mathcal{N}(0,1)
```

The observations $y_{i,j}$ may be vector-valued, in the case where multiple outcomes are measured.

### Structural model

The structural model $f$ represents the simulation model. It deterministically maps patient descriptors and observation time to model outcomes.

> [!TIP] Implementation:

The `StructuralModel` class is implemented to interface with different types of simulation models. For the moment, two instances can be used:

```python
struct_model_gp = StructuralGp(myGP)
struct_model_ode = StructuralOdeModel(ode_model, protocol_design, initial_conditions)
```

[^Lindstrom90]: Lindstrom, M. J., & Bates, D. M. (1990). Nonlinear Mixed Effects Models for Repeated Measures Data. Biometrics, 46(3), 673–687. https://doi.org/10.2307/2532087
