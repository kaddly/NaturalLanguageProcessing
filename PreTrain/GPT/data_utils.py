import torch
from torch.utils.data import TensorDataset, DataLoader
from token_utils import Vocab, tokenize, truncate_pad


def read_MPRCData(data_dir, is_train=False):
    with open('./data/MSRParaphraseCorpus/msr_paraphrase_train.txt') as f:
        pass
