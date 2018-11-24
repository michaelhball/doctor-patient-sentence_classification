import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


from torch.autograd import Variable


class PoolingClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.view(1, x.size()[1], x.size()[0]), 1).view(-1)
    
    def forward(self, input):
        max_pool = self.pool(input[1], True)
        avg_pool = self.pool(input[1], False)
        x = torch.cat([input[0], max_pool, avg_pool]).view(1, -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        
        return l_x
