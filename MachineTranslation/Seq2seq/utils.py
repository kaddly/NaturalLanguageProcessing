from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import collections


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


"""载入数据&预处理"""


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


# source, target = tokenize_nmt(preprocess_nmt(read_data_nmt()))
# show_list_len_pair_hist(['source', 'target'], '# tokens per sequence', 'count', source, target)
# plt.show()
def truncate_pad(line, num_steps, padding_token):
    """截断或填充⽂本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


class My_Dataset(Dataset):
    def __init__(self, min_freq=2, num_steps=10, num_examples=600):
        self.num_steps = num_steps
        self.source, self.target = tokenize_nmt(preprocess_nmt(read_data_nmt()), num_examples=num_examples)
        self.src_vocab = Vocab(self.source, min_freq=min_freq, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        self.tgt_vocab = Vocab(self.target, min_freq=min_freq, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        self.src_lines = [self.src_vocab[l] + [self.src_vocab['<eos>']] for l in self.source]
        self.tgt_lines = [self.src_vocab[l] + [self.src_vocab['<eos>']] for l in self.target]

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        src, tgt = self.src_lines[item], self.tgt_lines[item]
        src = truncate_pad(src, self.num_steps, self.src_vocab['<pad>'])
        src_len = (torch.tensor(src) != self.src_vocab['<pad>']).type(torch.int32).sum()
        tgt = truncate_pad(src, self.num_steps, self.tgt_vocab['<pad>'])
        tgt_len = (torch.tensor(tgt) != self.tgt_vocab['<pad>']).type(torch.int32).sum()
        return torch.tensor(src), src_len, torch.tensor(tgt), tgt_len

    def get_vocab(self):
        return self.src_vocab, self.tgt_vocab

