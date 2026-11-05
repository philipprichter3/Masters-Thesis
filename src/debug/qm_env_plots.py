
import math

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results, ncols=3):

    keys = list(dict.fromkeys(
        key for entry in results for key in entry if key != "time"
    ))

    times = [entry["time"] for entry in results]
    n = len(keys)

    if n == 0:
        return

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    _, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=True,
        squeeze=False
    )

    for ax, key in zip(axes.flat, keys):
        values = [entry.get(key, np.nan) for entry in results]

        ax.plot(times, values)
        ax.set_title(key)
        ax.set_xlabel("Time")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
