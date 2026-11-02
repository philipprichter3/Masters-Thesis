from torch import nn


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    return a sequential net of fully connected layers

    Args:
        sizes(tuple(int)): sizes of all the layers
        activation: activation function for all layers except for the output layer
        output_activation: activation to use for output layer only

    Returns:
        stacked fully connected layers
    """

    layers = []

    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        
    return nn.Sequential(*layers)




