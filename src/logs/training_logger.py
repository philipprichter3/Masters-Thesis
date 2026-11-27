
import numpy as np
import torch

from logs.log import Log


class TrainingLogger:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, hyper_params):

        self._log = (
            Log(
                create_plot=hyper_params["PLOT_LOG"],
                key_filters=hyper_params["PLOT_LOG_FILTER"]
            )
            if hyper_params["USE_LOG"]
            else None
        )

        self.update_every = hyper_params["LOG_EVERY"]

        self._snapshot = {}
        self._episode_history = []

        self._update_metrics = {}

# ======================================================================
# ============================= DATA RECORDING ============================
# ======================================================================

    def record_for_snapshot(self, info):

        if self._log is None:
            return

        self._episode_history.append(info)

    def record_step(self, metrics):

        if self._log is None:
            return

        self._update_metrics = metrics

# ======================================================================
# ============================= SNAPSHOT EPISODES ============================
# ======================================================================

    def finalize_snapshot(self, step):

        if self._log is None:
            return

        snapshot = {}

        for key in self._episode_history[0]:

            values = np.asarray([
                record[key]
                for record in self._episode_history
            ])

            if values.ndim == 1:
                snapshot[key] = values

            elif values.ndim == 2:
                for i in range(values.shape[1]):
                    snapshot[f"{key}_{i}"] = values[:, i]


        self._snapshot = snapshot.copy()
        
        self._episode_history.clear()

# ======================================================================
# ============================= UPDATE ============================
# ======================================================================

    def update(self, step):

        if self._log is None:
            return

        if step % self.update_every != 0:
            return

        self._log.log(step, **self._update_metrics)

        if self._snapshot:
            self._log.snapshot(step, **self._snapshot)
            self._snapshot.clear()

        self._log.update(step)

        self._update_metrics.clear()

    def redraw(self):

        if self._log is None:
            return

        self._log.update(None)

# ======================================================================
# ============================= SERIALIZATION ==========================
# ======================================================================

    def state_dict(self):

        return {
            "log": (
                self._log.state_dict()
                if self._log is not None
                else None
            ),
            "update_every": self.update_every,
            "snapshot": self._snapshot,
            "episode_history": self._episode_history,
            "update_metrics": self._update_metrics,
        }

    def load_state_dict(self, state):

        self.update_every = state["update_every"]

        self._snapshot = state["snapshot"]
        self._episode_history = state["episode_history"]
        self._update_metrics = state["update_metrics"]

        if state["log"] is not None:

            if self._log is None:
                self._log = Log()

            self._log.load_state_dict(state["log"])

        else:
            self._log = None

    def save(self, path):

        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, hyper_params):

        logger = cls(hyper_params)

        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        logger.load_state_dict(state)

        return logger