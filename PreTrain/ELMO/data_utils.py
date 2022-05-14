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


class _WikiTextDataset(Dataset):
    def __init__(self):
        pass
    def __getitem__(self, item):
        pass
    def __len__(self):
        pass
