import os
import matplotlib.pyplot as plt
import random
import pickle
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset, DataLoader
from utils.token_utils import BytePairEncoding
from utils.sample_utils import RandomGenerator, Poisson


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrice) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrice.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()


def _read_wiki(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)
    contexts = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if len(line.split(' . ')) < 2:
                continue
            contexts.extend(line.strip().lower().split(' . '))
    return contexts


def BPE_Encoding(train_sentences, val_sentences, test_sentences, num_merge):
    BPE = BytePairEncoding(train_sentences, num_merge, ['<unk>', '</w>', '<masked>', '<sep>'], min_freq=5)
    if not os.path.exists(f'./data/BPE_Decoding_token{num_merge}/'):
        train_tokens, val_tokens, test_tokens = Parallel(n_jobs=3)(
            delayed(BPE.segment_BPE)(sentences) for sentences in [train_sentences, val_sentences, test_sentences])
        os.mkdir(f'./data/BPE_Decoding_token{num_merge}/')
        with open(f'./data/BPE_Decoding_token{num_merge}/train_tokens.plk', 'wb') as f:
            pickle.dump(train_tokens, f)
        with open(f'./data/BPE_Decoding_token{num_merge}/val_tokens.plk', 'wb') as f:
            pickle.dump(val_tokens, f)
        with open(f'./data/BPE_Decoding_token{num_merge}/test_tokens.plk', 'wb') as f:
            pickle.dump(test_tokens, f)
    else:
        with open(f'./data/BPE_Decoding_token{num_merge}/train_tokens.plk', 'rb') as f:
            train_tokens = pickle.load(f)
        with open(f'./data/BPE_Decoding_token{num_merge}/val_tokens.plk', 'rb') as f:
            val_tokens = pickle.load(f)
        with open(f'./data/BPE_Decoding_token{num_merge}/test_tokens.plk', 'rb') as f:
            test_tokens = pickle.load(f)
    return train_tokens, val_tokens, test_tokens, BPE


# ⽣成下⼀句预测任务的数据
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输⼊序列的词元及其⽚段索引"""
    tokens = ['<bos>'] + tokens_a + ['<sep>']
    if tokens_b is not None:
        tokens += tokens_b + ['<eos>']
    return tokens


def load_data_wiki(batch_size, max_len, reconstruct_ways, num_merge=10000):
    data_dir = './data/wikitext-2'
    train_sentences = _read_wiki(data_dir, 'wiki.train.tokens')
    val_sentences = _read_wiki(data_dir, 'wiki.valid.tokens')
    test_sentences = _read_wiki(data_dir, 'wiki.test.tokens')
    if 'Sentence Permutation' in reconstruct_ways:
        pass
    train_tokens, val_tokens, test_tokens, BPE = BPE_Encoding(train_sentences, val_sentences, test_sentences, num_merge)
    return
