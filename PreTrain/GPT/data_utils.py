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
    with open(os.path.join(data_dir, 'msr_paraphrase_train.txt' if is_train else 'msr_paraphrase_test.txt'),
              encoding='UTF-8') as f:
        for line in f.readlines()[1:]:
            line = line.split('\t')
            is_next_labels.append(line[0])
            contents.append((_text_standardize(line[-2]), _text_standardize(line[-1])))
    return contents, is_next_labels


# ⽣成下⼀句预测任务的数据
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输⼊序列的词元及其⽚段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记⽚段A和B
    segment = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segment += [1] * (len(tokens_b) + 1)
    return tokens, segment


def _pad_GPT_input(examples, max_len, vocab):
    pass
