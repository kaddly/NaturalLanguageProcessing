import torch
from torch.utils.data import TensorDataset, DataLoader
from token_utils import Vocab, tokenize, truncate_pad
import re
import os


def _text_standardize(text):
    text = re.sub(r'—', '-', text)
    text = re.sub(r'–', '-', text)
    text = re.sub(r'―', '-', text)
    return text.strip()


def read_MPRCData(data_dir, is_train=True):
    is_next_labels, contents = [], []
    with open(os.path.join(data_dir, 'msr_paraphrase_train.txt' if is_train else 'msr_paraphrase_test.txt'), encoding='UTF-8') as f:
        for line in f.readlines()[1:]:
            line = line.split('\t')
            is_next_labels.append(line[0])
            contents.append((_text_standardize(line[-2]), _text_standardize(line[-1])))
    return contents, is_next_labels
