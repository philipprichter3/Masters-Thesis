
import numpy as np

from debug.log import Log


class TrainingLogger:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, hyper_params):

        self._log = (
            Log(
                update_every=hyper_params["LOG_EVERY"],
                create_plot=hyper_params["PLOT_LOG"],
                key_filters=hyper_params["PLOT_LOG_FILTER"]
            )
            if hyper_params["USE_LOG"]
            else None
        )

        self._snapshot = {}
        self._episode_history = []

        self._update_metrics = {}

# ======================================================================
# ============================= DATA RECORDING ============================
# ======================================================================

    def record_step(self, info):

        if self._log is None:
            return

        self._episode_history.append(info)

    def record_update(self, metrics):

        if self._log is None:
            return

        self._update_metrics = metrics

# ======================================================================
# ============================= SNAPSHOT EPISODES ============================
# ======================================================================

    def end_episode(self, step):

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

        self._log.log(step, **self._update_metrics)

        if self._snapshot:
            self._log.snapshot(step, **self._snapshot)
            self._snapshot.clear()

        self._log.update(step)

        self._update_metrics.clear()
