import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class PoolingClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i+1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, is_max):
        print(x.size())
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        print(f(x).size())
        
        return f(x)
        # return f(x.permute(1, 2, 0), (1,)).view(1, -1)
    
    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            source, sentence = input
            max_pool = self.pool(sentence, True)
            avg_pool = self.pool(sentence, False)
            x = torch.cat([source, max_pool, avg_pool])
            
            for l in self.layers:
                l_x = l(x)
                x = F.relu(l_x)
            
            return l_x
