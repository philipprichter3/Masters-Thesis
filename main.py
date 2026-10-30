from matplotlib import pyplot as plt

from qm_env_lindblad import Lindblad_Environment
from qm_env_montecarlo import MonteCarlo_Environment


def main():

    #hello

    lambda_c = 0.05      #<---output by RL SAC in future
    
    params = {
        "wc": 1.0,
        "wb": 1.0,
        "Nc": 6,
        "Nb": 6,
        "Nc_max": 24,
        "kappa": 0.2,
        "dt": 0.05,
        "tau": 10.0,
        "ntraj": 300,
        "seed": 42
    }

    lindblad_history = Lindblad_Environment(
        params
    ).run(lambda_c)

    montecarlo_history = MonteCarlo_Environment(
        params
    ).run(lambda_c)

    key = "ergotropy_full"

    plt.plot(
        [h["time"] for h in lindblad_history], 
        [h[key] for h in lindblad_history],
        c = "blue",
        label = "Lindblad Groundtruth"
    )

    plt.plot(
        [h["time"] for h in montecarlo_history], 
        [h[key] for h in montecarlo_history],
        c = "green",
        ls = "-.",
        label = "Ergotropy"
    )

    plt.plot(
        [h["time"] for h in montecarlo_history], 
        [h["daemonic_ergotropy_full"] for h in montecarlo_history],
        c = "red",
        ls = "--",
        label = "Daemonic Ergotropy"
    )  

    plt.legend()
    plt.grid()

    plt.xlabel("Time")
    plt.ylabel("Ergotopy")

    plt.show()


if __name__ == "__main__":
    main()