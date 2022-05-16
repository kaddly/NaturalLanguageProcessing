import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from token_utils import Vocab, tokenize


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    # ⼤写字⺟转换为⼩写字⺟
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _seq_data_cut(corpus, num_steps):
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的子序列
        return corpus[pos:num_steps + pos]

    seqs = []
    seqs_fw = []
    seqs_bw = []
    for pos in initial_indices:
        seqs.append(data(pos))
        seqs_fw.append(data(pos+1))
        seqs_bw.append(data(pos-1))
    return seqs, seqs_fw, seqs_bw


class _WikiTextDataset(Dataset):
    def __init__(self, paragraphs, num_steps):
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5)
        self.num_steps = num_steps
        corpus = [self.vocab[token] for line in sentences for token in line]
        self.seqs, self.seq_fw, self.seq_bw= _seq_data_cut(corpus, self.num_steps)

    def __getitem__(self, item):
        return self.seqs[item], self.seq_fw[item], self.seq_bw[item]

    def __len__(self):
        return len(self.seqs)
