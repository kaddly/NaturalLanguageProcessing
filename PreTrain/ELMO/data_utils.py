import random
import torch
from torch.utils.data import DataLoader, Dataset
import re
from token_utils import Vocab, tokenize


def _read_wiki(data_dir):
    with open(data_dir, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    # ⼤写字⺟转换为⼩写字⺟
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def _seq_data_cut(corpus, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(1, num_subseqs * num_steps, num_steps))

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的子序列
        return corpus[pos:num_steps + pos]

    seqs = []
    seqs_fw = []
    seqs_bw = []
    for pos in initial_indices:
        if pos + num_steps + 1 > len(corpus):
            continue
        seqs.append(data(pos))
        seqs_fw.append(data(pos + 1))
        seqs_bw.append(data(pos - 1))
    return seqs, seqs_fw, seqs_bw


class _WikiTextDataset(Dataset):
    def __init__(self, lines, num_steps, vocab=None, max_tokens=-1):
        tokens = tokenize(lines, token='word')
        if vocab is None:
            self.vocab = Vocab(tokens, min_freq=5)
        else:
            self.vocab = vocab
        self.num_steps = num_steps
        corpus = [self.vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        seqs, seqs_fw, seqs_bw = _seq_data_cut(corpus, self.num_steps)
        self.seqs, self.seqs_fw, self.seqs_bw = torch.tensor(seqs, dtype=torch.long), torch.tensor(seqs_fw,
                                                                                                   dtype=torch.long), torch.tensor(
            seqs_bw, dtype=torch.long)

    def __getitem__(self, item):
        return self.seqs[item], self.seqs_fw[item], self.seqs_bw[item]

    def __len__(self):
        return len(self.seqs)


def load_WikiTextDataset(bach_size, num_steps, use_random_iter=False, max_tokens=-1):
    train_lines = _read_wiki('./data/wikitext-2/wiki.train.tokens')
    valid_lines = _read_wiki('./data/wikitext-2/wiki.valid.tokens')
    test_lines = _read_wiki('./data/wikitext-2/wiki.test.tokens')
    train_dataset = _WikiTextDataset(train_lines, num_steps, max_tokens=max_tokens)
    valid_dataset = _WikiTextDataset(valid_lines, num_steps, train_dataset.vocab, max_tokens=max_tokens)
    test_dataset = _WikiTextDataset(test_lines, num_steps, train_dataset.vocab, max_tokens=max_tokens)
    train_iter = DataLoader(train_dataset, batch_size=bach_size, shuffle=use_random_iter, drop_last=True)
    valid_iter = DataLoader(valid_dataset, batch_size=bach_size, shuffle=use_random_iter, drop_last=True)
    test_iter = DataLoader(test_dataset, batch_size=bach_size, shuffle=use_random_iter, drop_last=True)
    return train_iter, valid_iter, test_iter, train_dataset.vocab
