import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import collections


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


# 词表
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按照频率统计出现的次数
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元索引为0
        self.idx_to_token = ['<UNK>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        # self.idx_to_token, self.token_to_idx = [], dict()
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1d或者2d列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def read_data_nmt():
    with open('data/fra-eng/fra.txt', 'r', encoding='UTF-8') as F:
        return F.read()


def preprocess_nmt(text):
    """预处理数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使⽤空格替换不间断空格
    # 使⽤⼩写字⺟替换⼤写字⺟
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插⼊空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    _, _, patches = plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)


class My_Dataset(Dataset):
    def __init__(self, min_freq=2, num_steps=10, num_examples=600):
        self.num_steps = num_steps
        self.source, self.target = tokenize_nmt(preprocess_nmt(read_data_nmt()), num_examples=num_examples)
        self.src_vocab = Vocab(self.source, min_freq=min_freq, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        self.tgt_vocab = Vocab(self.target, min_freq=min_freq, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        self.src_lines = [self.src_vocab[l] + [self.src_vocab['<eos>']] for l in self.source]
        self.tgt_lines = [self.tgt_vocab[l] + [self.tgt_vocab['<eos>']] for l in self.target]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        src, tgt = self.src_lines[item], self.tgt_lines[item]
        src_len, tgt_len = len(src), len(tgt)
        if self.num_steps > src_len:
            src += [self.src_vocab['<pad>']] * (self.num_steps - src_len)
        else:
            src = src[:self.num_steps]
            src_len = self.num_steps
        if self.num_steps > tgt_len:
            tgt += [self.tgt_vocab['<pad>']] * (self.num_steps - tgt_len)
        else:
            tgt = tgt[:self.num_steps]
            tgt_len = self.num_steps
        return torch.tensor(src), torch.tensor(int(src_len)), torch.tensor(tgt), torch.tensor(int(tgt_len))

    def get_vocab(self):
        return self.src_vocab, self.tgt_vocab
