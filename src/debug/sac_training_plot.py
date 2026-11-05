import math

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


class LiveTrainingPlot:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, ncols = 3):

        self.ncols = ncols

        self.steps = []
        self.history = {}

        self.fig = None
        self.handle = None

    def _setup(self):

        keys = list(self.history.keys())
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

        for ax, key in zip(axes.flat, keys):

            line, = ax.plot([], [], lw=1.5)

            ax.set_title(key)
            ax.set_xlabel("Training step")
            ax.grid(True, alpha=0.3)

            self.axes[key] = ax
            self.lines[key] = line

        for ax in axes.flat[n:]:

            ax.set_visible(False)

        fig.tight_layout()

        if self.handle is None:

            self.handle = display(fig, display_id=True)

        else:

            self.handle.update(fig)

        plt.close(fig)

# ======================================================================
# ============================= LOGGING ============================
# ======================================================================

    def log(self, step, params):

        # Detect new metrics
        new_keys = [k for k in params if k not in self.history]

        for key in new_keys:

            self.history[key] = [np.nan] * len(self.steps)

        self.steps.append(step)

        for key in self.history:

            self.history[key].append(
                float(params[key])
            )

        if new_keys:

            self._setup()

        if len(self.steps) >= 1:

            self.update()

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def update(self):

        if self.fig == None:

            return

        for key, line in self.lines.items():

            line.set_data(self.steps, self.history[key])

            ax = self.axes[key]
            ax.relim()
            ax.autoscale_view()

        self.handle.update(self.fig)

