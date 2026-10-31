import numpy as np
import torch
from torch import nn


class WeightScheduler(nn.Module):
    """
    Module that returns a 1D array with weights to assign to each objective. It is used to schedule the weight
    between objectives during training. The scheduling interpolates by a set of initial weights 
    training_hyperparams["C_START"] to some final weights training_hyperparams["C_END"] using a Fermi distribution
    with mean training_hyperparams["C_MEAN"] and typical width training_hyperparams["C_WIDTH"]

    Args:
        training_hyperparams(dict): dictionary containing
            "C_START"(np.array): 1D array, with same length as objectives, representing the initial weight for each objective
            "C_END"(np.array): 1D array, with same length as objectives, representing the final weight for each objective
            "C_MEAN"(float): mean point of the fermi distribution in units of steps
            "C_WIDTH"(float): typical width of the Fermi distribution in units of steps
            If these are missing, it will simply return a (1,0...,0) vector with no scheduling
        obj_num(int): number of objectives. If the previous hyperparameters are not passed, this must be specified, otherwise
            in gives an error
    """

    def __init__(self, training_hyperparams, obj_num=None):
        super().__init__()
        
        #create the main tensors that will produce the weights
        if "C_START" in training_hyperparams:
            self.register_buffer("c_start", torch.tensor(training_hyperparams["C_START"], dtype=torch.float32))
            self.register_buffer("c_end", torch.tensor(training_hyperparams["C_END"], dtype=torch.float32))
            self.register_buffer("c_mean", torch.tensor(training_hyperparams["C_MEAN"], dtype=torch.float32))
            self.register_buffer("c_width", torch.tensor(training_hyperparams["C_WIDTH"], dtype=torch.float32))
        else:
            assert obj_num is not None
            self.register_buffer("c_start", torch.zeros((obj_num,), dtype=torch.float32))
            self.register_buffer("c_end", torch.zeros((obj_num,), dtype=torch.float32))
            self.register_buffer("c_mean", torch.tensor(1., dtype=torch.float32))
            self.register_buffer("c_width", torch.tensor(1., dtype=torch.float32))
            self.c_start[0] = 1.
            self.c_end[0] = 1.

        #save their cpu version
        self.c_start_cpu = self.c_start.cpu().numpy()
        self.c_end_cpu = self.c_end.cpu().numpy()
        self.c_mean_cpu = self.c_mean.cpu().numpy()
        self.c_width_cpu = self.c_width.cpu().numpy()
        
    def forward(self, step, return_numpy_cpu = False):
        """
        Return the weights for the multi-objectives. 
        Args:
            step(int): current training step number
            return_numpy_cpu(bool): whether the output should by a np.array or torch.tensor 
        """
        if return_numpy_cpu:
            return self.return_schedule(step, self.c_start_cpu, self.c_end_cpu, self.c_mean_cpu, self.c_width_cpu, np)
        else:
            return self.return_schedule(step, self.c_start, self.c_end, self.c_mean, self.c_width, torch)

    def return_schedule(self, step, start, end, mean, width, module):
        """return the Fermi distribution computed in the current step"""
        return end + (start - end)/(1. + module.exp( (step-mean)/width ))