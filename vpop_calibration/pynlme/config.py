from typing import NamedTuple


class NlmeConfigDict(NamedTuple):
    nb_chains: int = 1
    live_plot: bool = True
    plot_frequency: int = 5
    progress_bar: bool = True
