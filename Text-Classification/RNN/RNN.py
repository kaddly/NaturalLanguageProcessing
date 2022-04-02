import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config():
    def __init__(self, dataset):
        pass


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.fc = torch.nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def forward(self, x):
        pass
