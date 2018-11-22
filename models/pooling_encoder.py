import torch
import torch.nn as nn

from torch.autograd import Variable


class PoolingEncoder(nn.Module):
    def __init__(self, pooling_type="max"):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, input):
        x = Variable(torch.tensor(input, dtype=torch.float), requires_grad=True)
        if self.pooling_type == "max":
            output, _ = torch.max(x, 0)
        elif self.pooling_type == "ave":
            output, _ = torch.mean(x, 0)
        
        return output