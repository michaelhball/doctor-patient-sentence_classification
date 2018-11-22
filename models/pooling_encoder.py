import torch
import torch.nn as nn

from torch.autograd import Variable


class PoolingEncoder(nn.Module):
    def __init__(self, pooling_type="max"):
        """
        Encodes sentences by pooling the word embeddings of 
            the tokens in the sentence using a given pooling
            method.
        Args:
            pooling_type (str): specifies which type of pooling
        """
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, input):
        if self.pooling_type == "max":
            output, _ = torch.max(input, 0)
        elif self.pooling_type == "ave":
            output, _ = torch.mean(input, 0)
        
        return output.reshape(1, -1)