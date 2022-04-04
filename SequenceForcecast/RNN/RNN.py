import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, run_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__()
        self.rnn = run_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, input, state):
        X = F.one_hot(input.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层⾸先将Y的形状改为(时间步数*批量⼤⼩,隐藏单元数)
        # 它的输出形状是(时间步数*批量⼤⼩,词表⼤⼩)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))
