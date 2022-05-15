import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from token_utils import Vocab, tokenize, truncate_pad


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    # ⼤写字⺟转换为⼩写字⺟
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


class _WikiTextDataset(Dataset):
    def __init__(self, paragraphs, num_steps):
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5)
        self.num_steps = num_steps
        corpus = [self.vocab[token] for line in sentences for token in line]
        self.corpus = corpus[random.randint(0, self.num_steps - 1):]
        num_subseqs = (len(self.corpus) - 1) // self.num_steps
        self.initial_indices = list(range(0, num_subseqs * self.num_steps, self.num_steps))
        random.shuffle(self.initial_indices)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def data(self, pos):
        # 返回从pos位置开始的长度为num_steps的子序列
        return self.corpus[pos:self.num_steps + pos]
