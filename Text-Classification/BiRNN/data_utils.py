import torch
from torch.utils.data import TensorDataset, DataLoader
from token_utils import Vocab, tokenize, truncate_pad


def read_YHUCNews(data_dir, is_train):
    contents, targets = [], []
    with open(data_dir, 'r', encoding='UTF-8') as f:
        for line in f:  # 读取每行的数据
            lin = line.strip()  # 去掉最后的\n符号
            if not lin:  # 如果是空的话，直接continue跳过
                continue
            contents.append(lin.split('\t')[0])
            targets.append(int(lin.split('\t')[1]))
    return contents, targets


def load_data_THUCNews(batch_size, num_steps=32):
    data_dir = '../data/THUCNews/data/'
    train_data = read_YHUCNews(data_dir + 'train.txt')
    dev_data = read_YHUCNews(data_dir + 'dev.txt')
    test_data = read_YHUCNews(data_dir + 'test.txt')
    train_tokens = tokenize(train_data[0], token='ChineseWord')
    dev_tokens = tokenize(dev_data[0], token='ChineseWord')
    test_tokens = tokenize(test_data[0], token='ChineseWord')
    vocab = Vocab(train_tokens, min_freq=1, reserved_tokens=['<pad>'])
    train_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    dev_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in dev_tokens])
    test_features = torch.tensor([truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_dataset = TensorDataset(*(train_features, torch.tensor(train_data[1])))
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataset = TensorDataset(*(dev_features, torch.tensor(dev_data[1])))
    dev_iter = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(*(test_features, torch.tensor(test_data[1])))
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter, dev_iter, vocab
