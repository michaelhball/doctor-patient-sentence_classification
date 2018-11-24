import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class BasicClassifier(nn.Module):
    def __init__(self, layers, drops=None):
        """
        A basic classifier - affine operations + nonlinearity
        Args:
            layers (list(int)): List of layer sizes
            drops (list(float)): List of dropout probs for layers
        """
        super().__init__()
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])
    
    def forward(self, input):
        x = input
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x
