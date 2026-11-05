import matplotlib.pyplot as plt
import numpy as np


def plot_results(history):

    keys = list(dict.fromkeys(
        key for entry in history for key in entry if key != "time"
    ))

    times = [entry["time"] for entry in history]
    n = len(keys)

    _, axes = plt.subplots(
        n, 1, figsize=(10, 3 * n),
        sharex=True, squeeze=False
    )

    for ax, key in zip(axes.flat, keys):

        values = [entry.get(key, np.nan) for entry in history]

        ax.step(times, values, label=key)
        ax.set_ylabel(key)
        ax.grid(True)
        ax.legend()

    axes[-1, 0].set_xlabel("Time")

    plt.tight_layout()
    plt.show()