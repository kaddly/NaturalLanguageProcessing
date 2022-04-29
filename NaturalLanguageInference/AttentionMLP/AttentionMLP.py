import torch
from torch import nn
from torch.nn import functional as F


def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)


class Attend(nn.Module):
    """计算假设（beta）与输⼊前提A的软对⻬以及前提（alpha）与输⼊假设B的软对⻬"""
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens)

    def forward(self, A, B):
        # A/B的形状：（批量⼤⼩，序列A/B的词元数，embed_size）
        # f_A/f_B的形状：（批量⼤⼩，序列A/B的词元数，num_hiddens）
        f_A = self.f(A)
        f_B = self.f(B)
        # e的形状：（批量⼤⼩，序列A的词元数，序列B的词元数）
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # beta的形状：（批量⼤⼩，序列A的词元数，embed_size），
        # 意味着序列B被软对⻬到序列A的每个词元(beta的第1个维度)
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # beta的形状：（批量⼤⼩，序列B的词元数，embed_size），
        # 意味着序列A被软对⻬到序列B的每个词元(alpha的第1个维度)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha

