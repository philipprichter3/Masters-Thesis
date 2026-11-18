
import numpy as np


def objective_weights(
        step : int, 
        c_start,
        c_end,
        c_width,
        c_mean
    ) -> np.ndarray:

    start = np.asarray(c_start, dtype=np.float32)
    end = np.asarray(c_end, dtype=np.float32)

    width = float(c_width)
    mean = float(c_mean)

    return \
        end + (start - end) / (1.0 + np.exp((step - mean) / width))


def target_entropy(
        step: int, 
        h_start,
        h_end,
        h_decay
    ) -> float:

    decay = float(h_decay)

    return float(
        h_end
        + (h_start - h_end)
        * np.exp(-step / decay)
    )
