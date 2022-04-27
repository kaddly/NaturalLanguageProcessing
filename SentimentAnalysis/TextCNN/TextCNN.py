import torch
from torch import nn


# 一维卷积
def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y


# 多通道卷积
def corr1d_multi_in(X, K):
    # ⾸先，遍历'X'和'K'的第0维（通道维）。然后，把它们加在⼀起
    return sum(corr1d(x, k) for x, k in zip(X, K))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 这个嵌⼊层不需要训练
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 最⼤时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # 沿着向量维度将两个嵌⼊层连结起来，
        # 每个嵌⼊层的输出形状都是（批量⼤⼩，词元数量，词元向量维度）连结起来
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # 根据⼀维卷积层的输⼊格式，重新排列张量，以便通道作为第2维
        embeddings = embeddings.permute(0, 2, 1)
        # 每个⼀维卷积层在最⼤时间汇聚层合并后，获得的张量形状是（批量⼤⼩，通道数，1）
        # 删除最后⼀个维度并沿通道维度连结
        encoding = torch.cat([torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1) for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

