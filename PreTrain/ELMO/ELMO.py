import torch
from torch import nn


class ELMO(nn.Module):
    def __init__(self, vocab_size, run_layer, **kwargs):
        super(ELMO, self).__init__(**kwargs)
        self.rnn = run_layer
        self.num_hiddens = self.rnn.hidden_size
        self.embedding = nn.Embedding(vocab_size, self.num_hiddens)
