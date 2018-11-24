import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock

class RNNPoolingClassifier(nn.Module):
    """
    Class for a pooling linear classifier that concatenates the final output of the RNN,
        the average pool and the max pool as its input.
    Args:
        layers (list): list of dimensions from which to construct a block of linear layers
        drops (list): A list of values containing dropouts for these layers.
    """
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, batch_size, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (1,)).view(batch_size, -1)
    
    def forward(self, input):
        output = input[-1]
        sl, batch_size, _ = output.size()
        avgpool = self.pool(output, batch_size, False)
        maxpool = self.pool(output, batch_size, True)
        x = torch.cat([output[-1], maxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)

        return l_x