from typing import NamedTuple, Literal


class SaemConfigDict(NamedTuple):
    ## Schedule
    nb_iter_burnin: int = 0
    nb_iter_learning: int = 100
    nb_iter_smoothing: int = 100

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
    optim_max_fun: int = 50  # for MI optimization

    # Convergence parameters
    convergence_threshold: float = 1e-4
    patience: int = 5

    # Output mode selector
    mode: Literal["test", "debug", "cli", "notebook"] = "notebook"

    # Mode dependent config options
    live_plot: bool = mode in ["notebook", "debug"]
    plot_frames: int = 20
    plot_columns: int = 3
    facet_size: tuple[float, float] = (2.0, 1.2)

    logging: bool = mode == "debug"
    logging_frequency: int = 10
    column_width: int = 12

    progress_bars: bool = mode == "notebook"
