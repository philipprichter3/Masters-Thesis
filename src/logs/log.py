
import numpy as np
from blinker import Signal

from logs.log_plotter import Plot


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