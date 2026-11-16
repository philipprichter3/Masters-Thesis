import math

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results_array, ncols=3):

    keys = list(dict.fromkeys(
        key
        for results in results_array
        for entry in results
        for key in entry
        if key != "time"
    ))

    times = np.array([
        entry["time"] for entry in results_array[0]
    ])

    n = len(keys)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    _, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        sharex=True,
        squeeze=False,
    )

    for ax, key in zip(axes.flat, keys):

        values = np.array([
            [entry.get(key, np.nan) for entry in results]
            for results in results_array
        ], dtype=float)

        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)

        ax.plot(times, mean, label="mean", ls = "--")

        ax.fill_between(
            times,
            mean - std,
            mean + std,
            alpha=0.25,
            label="std",
            color = "red"
        )

        ax.set_title(key)
        ax.set_xlabel("Time")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()