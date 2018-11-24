import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            return input
