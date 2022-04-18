import torch
from torch import nn


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


class Word2vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        self.v_embedding = nn.Embedding(vocab_size, embed_size)
        self.u_embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, center, context_negative):
        return skip_gram(center, context_negative, self.v_embedding, self.u_embedding)
