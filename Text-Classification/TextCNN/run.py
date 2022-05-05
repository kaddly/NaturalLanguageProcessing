# coding: UTF-8
import torch
import numpy as np
from train_eval import train, init_network
from utils import My_Dataset, build_vocab, train_test_split
from TextCNN import Config, Model
from torch.utils.data import DataLoader

if __name__ == '__main__':

    dataset = '../data/THUCNews'  # 数据集

    # train_test_split(dataset)

    config = Config(dataset)

    print("Loading data...")

    vocab = build_vocab(config)
    train_data = My_Dataset(config, config.train_path, vocab)
    dev_data = My_Dataset(config, config.dev_path, vocab)
    test_data = My_Dataset(config, config.test_path, vocab)

    train_iter = DataLoader(train_data, batch_size=config.batch_size)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size)
    test_iter = DataLoader(test_data, batch_size=config.batch_size)
    # train
    config.n_vocab = len(vocab)
    TextCNN_model = Model(config)
    # 模型放入到GPU中去
    TextCNN_model = TextCNN_model.to(config.device)
    print(TextCNN_model.parameters)
    train(config, TextCNN_model, train_iter, dev_iter, test_iter)
