import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from token_utils import Vocab, tokenize, truncate_pad


def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集⽂本序列和标签"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


def load_data_imdb(batch_size, num_steps=500):
    """返回数据迭代器和IMDB评论数据集的词表"""
    data_dir = '../data/aclImdb'
    train_data = read_imdb(data_dir, is_train=True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token='word')
    test_tokens = tokenize(test_data[0], token='word')
    vocab = Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_dataset = TensorDataset(*(train_features, torch.tensor(train_data[1])))
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(*(test_features, torch.tensor(test_data[1])))
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter, vocab
