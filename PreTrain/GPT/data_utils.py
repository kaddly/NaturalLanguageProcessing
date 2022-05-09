import torch
from torch.utils.data import Dataset, DataLoader
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
            is_next_labels.append(int(line[0]))
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


def _get_nsp_data(contents, is_next_labels, max_len):
    nsp_data = []
    for content, is_next in zip(contents, is_next_labels):
        if len(content[0]) + len(content[-1]) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(content[0], content[1])
        nsp_data.append((tokens, segments, is_next))
    return nsp_data


def _pad_GPT_input(examples, max_len, vocab):
    all_token_ids, all_segments, valid_lens, = [], [], []
    nsp_labels = []
    for (token_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(vocab[token_ids] + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, nsp_labels)


class MSRPC_dataset(Dataset):
    def __init__(self, contents, is_next_labels, max_len, vocab=None):
        contents = [tokenize(content, token='word') for content in contents]
        sentences = [sentence for content in contents for sentence in content]
        if vocab is None:
            self.vocab = Vocab(sentences, min_freq=1, reserved_tokens=['<pad>', '<cls>', '<sep>'])
        else:
            self.vocab = vocab
        # 获取下⼀句⼦预测任务的数据
        examples = _get_nsp_data(contents, is_next_labels, max_len)
        self.all_token_ids, self.all_segments, self.valid_lens, self.nsp_labels = _pad_GPT_input(examples, max_len,
                                                                                                 self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_MSRPC(batch_size, max_len):
    """加载MSPRC数据集"""

    data_dir = './data/MSRParaphraseCorpus'
    train_contents, train_is_next_labels = read_MPRCData(data_dir, is_train=True)
    test_contents, test_is_next_labels = read_MPRCData(data_dir, is_train=False)
    train_dataset = MSRPC_dataset(train_contents, train_is_next_labels, max_len)
    test_dataset = MSRPC_dataset(test_contents, test_is_next_labels, max_len, train_dataset.vocab)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter, train_dataset.vocab
