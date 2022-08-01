import os
import torch
from torch.utils.data import Dataset, DataLoader


def read_data(data_dir, file_name):
    with open(os.path.join(data_dir, file_name), 'rb', encoding='UTF-8') as f:
        f.readlines()
