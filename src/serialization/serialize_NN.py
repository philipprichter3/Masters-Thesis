import torch
from torch import nn

#currently not used

class Serializable__NN(nn.Module):

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path, map_location="cpu", strict=True):

        state_dict = torch.load(
            path,
            map_location=map_location,
            weights_only=True
        )

        self.load_state_dict(state_dict, strict=strict)
        
        return self