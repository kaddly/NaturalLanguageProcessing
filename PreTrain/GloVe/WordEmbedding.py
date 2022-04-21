import os
import torch
from torch import nn


class TokenEmbedding:
    """GloVe嵌入"""

    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = './WordEmbedding/' + embedding_name
        # GloVe⽹站：https://nlp.stanford.edu/projects/glove/
        # fastText⽹站：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r', encoding='UTF-8') as f:
            for line in f:
                elems = line.strip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的⾸⾏
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


# 词嵌入
print("词嵌入")
glove_6b50d = TokenEmbedding('glove.6b.50d')
print(len(glove_6b50d))
print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])


# 词相似度
def knn(W, x, k):
    # 增加1e-9以获得数值稳定性
    cos = torch.mv(W, x.reshape(-1)) / (torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) * torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]


def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):
        print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')

print("词相似度")
get_similar_tokens('chip', 3, glove_6b50d)
get_similar_tokens('baby', 3, glove_6b50d)
get_similar_tokens('beautiful', 3, glove_6b50d)


# 词类比
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 删除未知词


print("词类比")
print(get_analogy('man', 'woman', 'son', glove_6b50d))
print(get_analogy('beijing', 'china', 'tokyo', glove_6b50d))
print(get_analogy('bad', 'worst', 'big', glove_6b50d))
print(get_analogy('do', 'did', 'go', glove_6b50d))
