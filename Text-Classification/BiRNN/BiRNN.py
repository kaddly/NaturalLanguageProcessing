import torch
from torch import nn


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layer, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layer, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是（批量⼤⼩，时间步数）
        # 因为⻓短期记忆⽹络要求其输⼊的第⼀个维度是时间维，
        # 所以在获得词元表⽰之前，输⼊会被转置。
        # 输出形状为（时间步数，批量⼤⼩，词向量维度）
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # 返回上⼀个隐藏层在不同时间步的隐状态，
        # outputs的形状是（时间步数，批量⼤⼩，2*隐藏单元数）
        outputs, _ = self.encoder(embeddings)
        # 连结初始和最终时间步的隐状态，作为全连接层的输⼊，
        # 其形状为（批量⼤⼩，4*隐藏单元数）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
