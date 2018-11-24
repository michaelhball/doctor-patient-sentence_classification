import numpy as np
import torch
import torch.nn as nn

from collections import Counter
from torch.autograd import Variable


class BoWEncoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        
    def forward(self, input):
        sent = [input[0].item()] + [t.item() for t in input[1]]
        with torch.set_grad_enabled(self.training):
            freq = Counter(sent)
            bow_rep = np.zeros((len(self.vocab)))
            for k, v in freq.items():
                bow_rep[k] = v
            
            return torch.tensor(bow_rep, dtype=torch.float).view(1, -1)
