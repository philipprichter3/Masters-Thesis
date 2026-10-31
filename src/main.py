from matplotlib import pyplot as plt

from src.environments.qm_env_lindblad import Lindblad_Environment
from src.environments.qm_env_montecarlo import MonteCarlo_Environment

LAMBDA_C = 0.15      #<---output by RL SAC in future

PARAMS = {
    #general
    "wc": 1.0,
    "wb": 1.0,
    "Nc": 6,
    "Nb": 6,
    "Nc_max": 24,
    "kappa": 0.2,
    "dt": 0.05,
    "tau": 10.0,

    #for monte carlo
    "ntraj": 150,
    "seed": 42

    #for SSE
}

def run(env, u):

    history = []

    while env.t < env.tau:
        result = env.step(u)
        history.append(result)

    return history

def main():

    lindblad_history = run(
        Lindblad_Environment(PARAMS), 
        LAMBDA_C
    )

    montecarlo_history = run(
        MonteCarlo_Environment(PARAMS), 
        LAMBDA_C
    )

    key = "ergotropy_full"

    plt.plot(
        [h["time"] for h in lindblad_history], 
        [h[key] for h in lindblad_history],
        c = "blue",
        label = "Lindblad ergotropy"
    )

    plt.plot(
        [h["time"] for h in montecarlo_history], 
        [h[key] for h in montecarlo_history],
        c = "green",
        ls = "-.",
        label = "MC ergotropy"
    )

    plt.plot(
        [h["time"] for h in montecarlo_history], 
        [h["daemonic_ergotropy_full"] for h in montecarlo_history],
        c = "red",
        ls = "--",
        label = "MC daemonic ergotropy"
    )  

    plt.legend()
    plt.grid()

    plt.xlabel("Time")
    plt.ylabel("Ergotopy")

    plt.show()


if __name__ == "__main__":
    main()