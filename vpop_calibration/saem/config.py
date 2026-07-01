from typing import NamedTuple, Literal


class SaemConfigDict(NamedTuple):
    ## Schedule
    nb_iter_burn_in: int = 5
    nb_iter_learning: int = 100
    nb_iter_smoothing: int | None = None

    ## E-step parameters
    nb_mcmc_transitions: int = 1
    # Metropolis-Hastings step size. Stick to the 0.1 - 1 range
    init_step_size_unscaled: float = 0.5  # to be divided by sqrt(nb_pdu)
    init_step_adaptation: float = 0.5

    ## M-step parameters
    # Stochastic-approximation learning rate decay power
    learning_rate_power: float = 0.8
    # Simulated annealing factor
    annealing_factor: float = 0.95

    ## General parameters
    convergence_threshold: float = 1e-4
    patience: int = 5
    mode: Literal["test", "debug", "cli", "notebook"] = "notebook"
    optim_max_fun: int = 50  # for MI optimization
    plot_frames: int = 20
    plot_columns: int = 3
