import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, run_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
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
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))


def getParams(vocab_size, num_hiddens, device):
    num_inputs = num_output = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def tree():
        return (
            normal((num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = tree()  # 输入门
    W_xo, W_ho, b_o = tree()  # 输出门
    W_xf, W_hf, b_f = tree()  # 遗忘门
    W_xc, W_hc, b_c = tree()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_output))
    b_q = torch.zeros(size=num_hiddens, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros(size=(batch_size, num_hiddens), device=device),
            torch.zeros(size=(batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state

    outputs = []
    for X in inputs:
        I = torch.sigmoid((W_xi @ X) + (W_hi @ H) + b_i)
        F = torch.sigmoid((W_xf @ X) + (W_hf @ H) + b_f)
        O = torch.sigmoid((W_xo @ X) + (W_ho @ H) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = C * F + C_tilda * I
        H = O * torch.tanh(C)
        Y = W_hq @ H + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
