import torch
import torch.nn as nn
from torch.autograd import Variable

# from .regularisation import LockedDropout, EmbeddingDropout, WeightDrop


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim): # consider including dropout here too
        super().__init__()
        self.batch_size = 1
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstms = nn.ModuleList([nn.LSTM(embedding_dim, hidden_dim), nn.LSTM(hidden_dim, embedding_dim)])
        self.num_layers = len(self.lstms)
        self.reset()
    
    def forward(self, input):
        with torch.set_grad_enabled(self.training):
            new_hidden, outputs = [], []
            output = input[1].view(input[1].size()[0], self.batch_size, self.embedding_dim)
            for l, lstm in enumerate(self.lstms):
                output, new_h = lstm(output, self.hidden[l])
                new_hidden.append(new_h)
                outputs.append(output)

            self.hidden = self.repackage_hidden(new_hidden)
        
        return outputs

    def repackage_hidden(self, h):
        """
        Repackages a variable to allow it to forget its history.
        """
        return h.detach() if type(h) == torch.Tensor else tuple(self.repackage_hidden(v) for v in h)

    def one_hidden(self, l):
        """
        Resets one hidden layer
        """
        num_hidden = (self.hidden_dim if l != self.num_layers - 1 else self.embedding_dim)
        return Variable(self.weights.new(1, self.batch_size, num_hidden).zero_())

    def reset(self):
        """
        Resets the networks hidden layers for the next iteration of training.
        """
        self.weights = next(self.parameters()).data
        self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.num_layers)]
