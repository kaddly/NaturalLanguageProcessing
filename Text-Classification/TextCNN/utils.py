import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
import re

UNK, PAD = '<UNK>', '<PAD>'


def clean_str(string):
    # 清理数据替换无词义的符号
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub('[^A-Za-z]+', ' ', string)
    return string.strip().lower()


def train_test_split(dataset):
    positive_data_file = dataset + '/rt-polarity.pos'
    negative_data_file = dataset + '/rt-polarity.neg'
    # Load data from files
    # 加载数据
    positive_examples = list(open(positive_data_file, "r", encoding='UTF-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]  # 去空格
    negative_examples = list(open(negative_data_file, "r", encoding='UTF-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]  # 去空格
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]  # 字符过滤，实现函数见clean_str()
    # Generate labels
    # 生成标签
    positive_labels = [0 for _ in positive_examples]
    negative_labels = [1 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)  # 将两种label连在一起
    # shuffle
    data_len = len(y)
    shuffle_indices = np.random.permutation(np.arange(data_len))
    shuffle_data = []
    for i in shuffle_indices:
        shuffle_data.append((x_text[i], y[i]))
    # 划分训练集测试集验证集
    train_len = data_len // 10 * 7
    dev_len = data_len // 10 * 2
    test_len = data_len // 10
    train_f = open(dataset + '/data/train.txt', 'w')
    for i in range(train_len):
        train_f.write(shuffle_data[i][0] + '\t' + str(shuffle_data[i][1]) + '\n')
    dev_f = open(dataset + '/data/dev.txt', 'w')
    for i in range(train_len, (dev_len + train_len)):
        dev_f.write(shuffle_data[i][0] + '\t' + str(shuffle_data[i][1]) + '\n')
    test_f = open(dataset + '/data/test.txt', 'w')
    for i in range((train_len + dev_len), data_len):
        test_f.write(shuffle_data[i][0] + '\t' + str(shuffle_data[i][1]) + '\n')


def build_vocab(config):
    file_path = config.train_path
    tokenizer = config.tokenizer
    max_size = config.max_size
    min_freq = config.min_freq
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:  ## 读取数据文件：训练集还是验证集还是测试集
        for line in tqdm(f):  ## 读取每行的数据
            lin = line.strip()  ## 去掉最后的\n符号
            if not lin:  ##如果是空的话，直接continue跳过
                continue
            content = lin.split('\t')[0]  ## 文本数据用\t进行分割，取第一个[0]是文本，第二个是【1】是标签数据
            for word in tokenizer(content):  ##用字进行切割，tokenizer(content)看函数得到的是一个列表对吧。
                vocab_dic[word] = vocab_dic.get(word, 0) + 1  ##生成词表字典，这个字典的get就是有这个元素就返回结果，数量在这里原始值加1，如果没有返回默认值为0，数量加1；
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
                     :max_size]  ##首先是根据频次筛选，然后sort一下降序，然后取词表最大
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}  ##从词表字典中找到我们需要的那些就可以了
        vocab_dic.update({config.UNK: len(vocab_dic), config.PAD: len(vocab_dic) + 1})  ##然后更新两个字符，一个是unk字符，一个pad字符
    print(f"Vocab size: {len(vocab_dic)}")

    return vocab_dic


def get_time_dif(start_time):
    end_time = time.time()

    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class My_Dataset(Dataset):
    def __init__(self, config, path, vocab_file):
        self.config = config
        file = open(path, 'r', encoding='utf-8')
        self.contents = []
        self.labels = []
        for line in file.readlines():
            line = line.strip().split('\t')
            content = line[0]
            label = line[1]
            self.contents.append(content)
            self.labels.append(label)
        self.pad_size = config.pad_size
        self.tokenizer = config.tokenizer
        self.vocab = vocab_file

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, item):
        content, label = self.contents[item], self.labels[item]
        token = self.tokenizer(content)
        seq_len = len(token)
        words_line = []

        if len(token) < self.pad_size:
            token.extend([PAD] * (self.pad_size - len(token)))
        else:
            token = token[:self.pad_size]
            seq_len = self.pad_size
        for word in token:
            words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
        data = torch.Tensor(words_line).long().to(self.config.device)
        label = torch.tensor([int(label)]).squeeze().to(self.config.device)
        seq_len = torch.tensor([int(seq_len)]).squeeze().to(self.config.device)
        return (data, seq_len), label
