
import numpy as np
from blinker import Signal

from logs.log_plotter import Plot


class Log:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, create_plot = False, key_filters = None):

        self.steps = []

        self.history = {}
        self.snapshots = {}

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

        metrics = self._filter(metrics)

        if not metrics:
            return

        new_keys = [k for k in metrics if k not in self.history]

        for key in new_keys:
            self.history[key] = [np.nan] * len(self.steps)

        if not self.steps or self.steps[-1] != step:

            self.steps.append(step)

            for key in self.history:
                self.history[key].append(np.nan)

        for key, value in metrics.items():
            self.history[key][-1] = float(value)

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

        self.updated.send(
            self, 
            new_keys = self.new_log_keys.copy(), 
            new_snapshots = self.new_snapshot_keys.copy()
        )

        self.new_log_keys.clear()
        self.new_snapshot_keys.clear()

# ======================================================================
# ============================= SERIALIZATION ==========================
# ======================================================================

    def state_dict(self):

        return {
            "steps": self.steps,
            "history": self.history,
            "snapshots": self.snapshots,
            "key_filters": self.key_filters,
        }

    def load_state_dict(self, state):

        self.steps = state["steps"]
        self.history = state["history"]
        self.snapshots = state["snapshots"]
        self.key_filters = state["key_filters"]

        self.new_log_keys = list(self.history)
        self.new_snapshot_keys = list(self.snapshots)