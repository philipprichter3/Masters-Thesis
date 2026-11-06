import math

import matplotlib.pyplot as plt
import numpy as np
from blinker import Signal
from IPython.display import display

# ======================================================================
# ============================= LOG CLASS ============================
# ======================================================================

class Log:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, create_plot = False, update_every = 200):

        self.steps = []
        self.history = {}

        self.update_every = update_every

        self.updated = Signal()

        self.plot = Plot(self) if create_plot else None

# ======================================================================
# ============================= LOGGING ============================
# ======================================================================

    def log(self, step, **metrics):

        if step % self.update_every != 0:

            return

        new_keys = [k for k in metrics if k not in self.history]

        for key in new_keys:
        
            self.history[key] = [np.nan] * len(self.steps)

        self.steps.append(step)

        for key in self.history:

            self.history[key].append(
                float(metrics.get(key, np.nan))
            )

        self.updated.send(self, new_keys = new_keys)





# ======================================================================
# ============================= PLOT CLASS ============================
# ======================================================================

class Plot:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, logger : Log, ncols = 3):

        self.ncols = ncols

        self.fig = None
        self.handle = None

        logger.updated.connect(self.update, weak = False)

    def _setup(self, sender : Log):

        keys = list(sender.history.keys())
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
# ============================= RUNTIME ============================
# ======================================================================

    def update(self, sender: Log, new_keys):

        if not sender.history:
            return

        if self.fig is None or new_keys:
            self._setup(sender)

        for key, line in self.lines.items():
            line.set_data(sender.steps, sender.history[key])

            ax = self.axes[key]
            ax.relim()
            ax.autoscale_view()

        self.handle.update(self.fig)