import os
import pickle
import collections
import jieba
from tqdm import tqdm


def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    elif token == 'ChineseWord':
        return [jieba.lcut(line, cut_all=False) for line in lines]
    else:
        print("未知类型：" + token)


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_machine_translation`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


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


class BytePairEncoding:
    def __init__(self, lines, num_merges, reserved_tokens=None) -> None:
        self.tokens = tokenize(lines, 'word')
        raw_token_freqs = count_corpus(self.tokens)
        self.token_freqs = {}
        for token, freq in raw_token_freqs.items():
            self.token_freqs[' '.join(list(token))] = raw_token_freqs[token]
        if reserved_tokens is None:
            reserved_tokens = ['<unk>', '</w>']
        self.symbols = reserved_tokens + [chr(i) for i in range(97, 123)]
        self.token_to_idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        if not os.path.exists('./data/BPE'):
            os.mkdir('./data/BPE')
        if not os.path.exists('./data/BPE/symbols.plk'):
            for i in tqdm(range(num_merges), desc="BPE Encoding"):
                pairs = self.get_max_freq_pair()
                self.token_freqs = self.merge_symbols(pairs)
            with open('./data/BPE/symbols.plk', 'wb') as f:
                pickle.dump(self.symbols, f)
            with open('./data/BPE/token_to_idx.plk', 'wb') as f:
                pickle.dump(self.token_to_idx, f)
        else:
            with open('./data/BPE/symbols.plk', 'rb') as f:
                self.symbols = pickle.load(f)
            with open('./data/BPE/token_to_idx.plk', 'rb') as f:
                self.token_to_idx = pickle.load(f)

    def get_max_freq_pair(self):
        pairs = collections.defaultdict(int)
        for token, freq in self.token_freqs.items():
            symbols = token.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return max(pairs, key=pairs.get)

    def merge_symbols(self, max_freq_pair):
        self.symbols.append(''.join(max_freq_pair))
        self.token_to_idx[''.join(max_freq_pair)] = len(self.symbols) - 1
        new_token_freqs = dict()
        for token, freq in self.token_freqs.items():
            new_token = token.replace(' '.join(max_freq_pair), ''.join(max_freq_pair))
            new_token_freqs[new_token] = self.token_freqs[token]
        return new_token_freqs

    def segment_BPE_tokens(self, tokens):
        output = []
        for token in tokens:
            start, end = 0, len(token)
            cur_output = []
            # 具有符号中可能最⻓⼦字的词元段
            while start < len(token) and start < end:
                if token[start:end] in self.symbols:
                    cur_output.append(token[start:end])
                    start = end
                    end = len(token)
                else:
                    end -= 1
            if start < len(token):
                cur_output.append("<unk>")
            cur_output.append('</w>')
            output.extend(cur_output)
        return output

    def segment_BPE(self, sentences):
        all_tokens = tokenize(sentences, 'word')
        return [self.segment_BPE_tokens(tokens) for tokens in tqdm(all_tokens, desc='BPE Decoding')]

    def __len__(self):
        return len(self.symbols)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.symbols[indices]
        return [self.symbols[index] for index in indices]

    @property
    def get_symbols(self):
        return self.symbols

    @property
    def get_token_freqs(self):
        return self.token_freqs

    @property
    def unk(self):  # 未知词元的索引为0
        return 0
