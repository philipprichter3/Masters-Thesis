import math

import matplotlib.pyplot as plt
from IPython.display import display

from logs.log import Log


class Plot:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, logger: Log, ncols=3):

        self.ncols = ncols

        self.fig = None
        self.handle = None

        self.axes = {}
        self.lines = {}

        logger.updated.connect(self.update, weak=False)

    def _setup(self, sender: Log):

        keys = (
            [("metric", k) for k in sender.history]
            + [("snapshot", k) for k in sender.snapshots]
        )

        n = len(keys)

        ncols = min(self.ncols, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3 * nrows),
            squeeze=False
        )

        self.fig = fig
        self.axes = {}
        self.lines = {}

        for ax, (kind, key) in zip(axes.flat, keys):

            line, = ax.plot([], [], lw=1.5)

            ax.set_title(key)
            ax.set_xlabel(
                "Training step" if kind == "metric" else "x"
            )
            ax.grid(True, alpha=0.3)

            self.axes[(kind, key)] = ax
            self.lines[(kind, key)] = line

        for ax in axes.flat[n:]:
            ax.set_visible(False)

        fig.tight_layout()


# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def update(
        self,
        sender: Log,
        new_keys=(),
        new_snapshots=()
    ):

        if not sender.history and not sender.snapshots:
            return

        if self.fig is None or new_keys or new_snapshots:
            self._setup(sender)

        for (kind, key), line in self.lines.items():

            ax = self.axes[(kind, key)]

            if kind == "metric":

                x = sender.steps
                y = sender.history[key]

            else:

                _, x, y = sender.snapshots[key]
                ax.set_title(key)

            line.set_data(x, y)

            ax.relim()
            ax.autoscale_view()

        if self.handle is None:
            self.handle = display(self.fig, display_id=True)
        else:
            self.handle.update(self.fig)

        plt.close(self.fig)