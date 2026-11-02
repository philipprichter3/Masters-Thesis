from matplotlib import pyplot as plt

from environments.qm_env_lindblad import Lindblad_Environment
from environments.qm_env_montecarlo import MonteCarlo_Environment

LAMBDA_C = 0.15      #<---output by RL SAC in future

Nb = 6

ENV_PARAMS = {
    #general
    "wc": 1.0,
    "wb": 1.0,

    "Nc": 6,
    "Nb": Nb,

    "Nc_max": 24,

    "kappa": 0.2,
    
    "dt": 0.05,
    "tau": 10.0,

    #for monte carlo
    "ntraj": 16,
    "seed": 42,

    #for SSE

    #for SAC
    "min_u": -0.3,
    "max_u": 0.3,
    "objective": "ergotropy_full",
    "reward_coeff": 1.0
}

TRAINING_HYPERPARAMS = {
    "BATCH_SIZE": 256,            #batch size

    "LR": 0.001,                  #learning rate for Q and Pi loss
    "ALPHA_LR": 0.003,            #learning rate to tune the temperature parameters alpha 

    "H_START": 0.72,              #initial target entropy of the policy
    "H_END": -3.,                 #final target entropy of the policy
    "H_DECAY": 200000,            #exponential decay of the target entropy of the policy

    "C_START": [1./Nb, 0.],       #initial weights for energy and ergotropy to compute the reward (Eqs. S38 - S39)
    "C_END": [0., 1.],            #final weights for energy and ergotropy to compute the reward (Eqs. S38 - S39)
    "C_MEAN": 40000,              #timestep number where the weights are half way between start and end
    "C_WIDTH": 20000,             #width in timesteps to transition from start and end weights        

    "REPLAY_MEMORY_SIZE": 180000, #size of the replay buffer

    "POLYAK": 0.995,              #polyak coefficient

    "LOG_STEPS": 1000,            #save logs and display training every number of steps

    "GAMMA": 0.993,               #RL discount factor
    "RETURN_GAMMA": 0.9,          #exponential averaging of the return when logging during training
    "LOSS_GAMMA": 0.995,          #exponential averaging of the loss functions when logging during training

    "HIDDEN_SIZES": (512,256),    #size of hidden layers of the neural networks

    "SAVE_STATE_STEPS": 480000,   #saves complete state of trainig every number of steps

    "INITIAL_RANDOM_STEPS": 5000, #number of initial uniformly random steps
    "UPDATE_AFTER": 1000,         #start minimizing loss function after initial steps
    "UPDATE_EVERY": 50,           #performs this many updates every this many steps

    "USE_CUDA": True,             #use cuda for computation

    "MIN_COV_EIGEN": 1.e-7,       #security parameter for the covariance matrix of the policy

    "DONT_SAVE_MEMORY": True      #if true, it won't save the memory buffer, so cannot resume training from file
}

def run(env, u):

    history = []

    while env.t < env.tau:
        result = env.step(u)
        history.append(result)

    return history

def main():

    lindblad_history = run(
        Lindblad_Environment(ENV_PARAMS), 
        LAMBDA_C
    )

    montecarlo_history = run(
        MonteCarlo_Environment(ENV_PARAMS), 
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