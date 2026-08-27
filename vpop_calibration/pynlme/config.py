from typing import NamedTuple, Any


class NlmeConfigDict(NamedTuple):
    nb_chains: int = 1
    live_plot: bool = True
    plot_frequency: int = 5
    progress_bar: bool = True
    max_samples: int = 1000
    residual_min_variance: float = 1e-6

    importance_sampling_df: float = 5.0

    fim_burn_in: int = 50
    fim_accumulation_decay_power: float = 0.7
    fim_finite_differences_eps: float = 1e-2

    def get_state_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self._asdict().items()}

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any]) -> "NlmeConfigDict":
        return cls(**state_dict)
