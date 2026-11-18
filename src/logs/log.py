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

    def __init__(self, create_plot = False, update_every = 200, key_filters = None):

        self.steps = []

        self.history = {}
        self.snapshots = {}

        self.update_every = update_every
        self.key_filters = key_filters

        self.new_log_keys = []
        self.new_snapshot_keys = []

        self.updated = Signal()

        self.plot = Plot(self) if create_plot else None

    def _filter(self, data):
        if not self.key_filters:
            return data

        return {
            key: value
            for key, value in data.items()
            if key in self.key_filters
        }

# ======================================================================
# ============================= LOGGING ============================
# ======================================================================

    def log(self, step, **metrics):

        if step % self.update_every != 0:

            return

        metrics = self._filter(metrics)

        new_keys = [k for k in metrics if k not in self.history]

        for key in new_keys:

            if self.key_filters == {} or key in self.key_filters:
        
                self.history[key] = [np.nan] * len(self.steps)

        self.steps.append(step)

        for key in self.history:

            self.history[key].append(
                float(metrics.get(key, np.nan))
            )

        self.new_log_keys.extend(new_keys)

# ======================================================================
# ============================= SNAPSHOTTING ============================
# ======================================================================

    def snapshot(self, step, **snapshots):

        snapshots = self._filter(snapshots)

        new_keys = [k for k in snapshots if k not in self.snapshots]

        for key, data in snapshots.items():

            if self.key_filters == {} or key in self.key_filters:

                if isinstance(data, tuple):
                    x, y = data
                else:
                    y = data
                    x = np.arange(len(y))

                self.snapshots[key] = (
                    step,
                    np.asarray(x).copy(),
                    np.asarray(y).copy()
                )

        self.new_snapshot_keys.extend(new_keys)

# ======================================================================
# ============================= UPDATING ============================
# ======================================================================

    def update(self, step):

        if step % self.update_every != 0:

            return

        self.updated.send(
            self, 
            new_keys = self.new_log_keys.copy(), 
            new_snapshots = self.new_snapshot_keys.copy()
        )

        self.new_log_keys.clear()
        self.new_snapshot_keys.clear()




# ======================================================================
# ============================= PLOT CLASS ============================
# ======================================================================

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