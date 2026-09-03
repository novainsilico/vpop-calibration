try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    from IPython.display import display
except ImportError:
    display = None
import pandas as pd
import numpy as np


from vpop_calibration.config import smoke_test


class OptimizerPlot:
    def __init__(
        self,
        history: pd.DataFrame | dict[str, list],
        nb_tot_iter: int,
        facet_size: tuple[float, float],
        nb_cols: int = 3,
    ):
        self.nb_cols = nb_cols
        self.facet_size = facet_size
        self.nb_tot_iter = nb_tot_iter
        self.headers = [k for k in history.keys() if k != "iteration"]
        self.nb_plots = len(self.headers)

        self.nb_rows = int(np.ceil(self.nb_plots / self.nb_cols))
        if plt is not None:
            self.fig, self.axes = plt.subplots(
                nrows=self.nb_rows,
                ncols=self.nb_cols,
                figsize=(
                    self.nb_cols * self.facet_size[0],
                    self.nb_rows * self.facet_size[1],
                ),
                squeeze=False,
                sharex="all",
            )
        self.traces = {}
        for plot_idx, header in enumerate(self.headers):
            row, col = plot_idx // self.nb_cols, plot_idx % self.nb_cols
            ax = self.axes[row, col]
            ax.set_xlim(0, self.nb_tot_iter)
            (tr,) = ax.plot(history["iteration"], history[header])
            ax.set_title(header)
            ax.grid(True)
            self.traces.update({header: tr})
        for plot_idx in range(self.nb_plots, self.nb_rows * self.nb_cols):
            row, col = plot_idx // self.nb_cols, plot_idx % self.nb_cols
            ax = self.axes[row, col]
            ax.axis("off")

        if not smoke_test and plt is not None:
            plt.tight_layout()
            if display is not None:
                self.handle = display(self.fig, display_id=True)

    def update(self, history: pd.DataFrame):
        for header in self.headers:
            self.traces[header].set_data(history["iteration"], history[header])

        if not smoke_test:
            for ax in self.axes.flatten():
                ax.autoscale_view(scaley=True, scalex=False)
                ax.relim()
            if self.handle is not None:
                self.handle.update(self.fig)

    def close(self):
        if plt is not None:
            plt.close(self.fig)
