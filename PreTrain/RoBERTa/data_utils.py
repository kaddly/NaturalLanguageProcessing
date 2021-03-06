import os
import random
import pickle
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset, DataLoader
from token_utils import BytePairEncoding


def _read_wiki(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)
    contexts = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if len(line.split(' . ')) < 2:
                continue
            contexts.extend(line.strip().lower().split(' . '))
    return contexts


def BPE_Encoding(train_sentences, val_sentences, test_sentences, num_merge):
    BPE = BytePairEncoding(train_sentences, num_merge, ['<unk>', '</w>', '<masked>', '<sep>'], min_freq=5)
    if not os.path.exists(f'./data/BPE_Decoding_token{num_merge}/'):
        train_tokens, val_tokens, test_tokens = Parallel(n_jobs=3)(
            delayed(BPE.segment_BPE)(sentences) for sentences in [train_sentences, val_sentences, test_sentences])
        os.mkdir(f'./data/BPE_Decoding_token{num_merge}/')
        with open(f'./data/BPE_Decoding_token{num_merge}/train_tokens.plk', 'wb') as f:
            pickle.dump(train_tokens, f)
        with open(f'./data/BPE_Decoding_token{num_merge}/val_tokens.plk', 'wb') as f:
            pickle.dump(val_tokens, f)
        with open(f'./data/BPE_Decoding_token{num_merge}/test_tokens.plk', 'wb') as f:
            pickle.dump(test_tokens, f)
    else:
        with open(f'./data/BPE_Decoding_token{num_merge}/train_tokens.plk', 'rb') as f:
            train_tokens = pickle.load(f)
        with open(f'./data/BPE_Decoding_token{num_merge}/val_tokens.plk', 'rb') as f:
            val_tokens = pickle.load(f)
        with open(f'./data/BPE_Decoding_token{num_merge}/test_tokens.plk', 'rb') as f:
            test_tokens = pickle.load(f)
    return train_tokens, val_tokens, test_tokens, BPE


# ⽣成遮蔽语⾔模型任务的数据
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    # 为遮蔽语⾔模型的输⼊创建新的词元副本，其中输⼊可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱用于屏蔽语言模型任务中获取15%随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为"<masked>"词元
        if random.random() < 0.8:
            masked_token = '<masked>'
        else:
            # 10%的时间：词保持不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：⽤随机词替换该词
            else:
                masked_token = random.choice(vocab.symbols)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        # 在遮蔽语⾔模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语⾔模型任务中预测15%的随机词元
    num_mlm_pred = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_pred,
                                                                      vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _seq_data_cut(sentences_tokens, max_len):
    all_tokens = [token for tokens in sentences_tokens for token in tokens]
    all_tokens = all_tokens[random.randint(0, max_len - 1):]
    num_subseqs = (len(all_tokens) - 1) // max_len
    initial_indices = list(range(1, num_subseqs * max_len, max_len))

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的子序列
        return all_tokens[pos:max_len + pos]

    seqs = []
    for pos in initial_indices:
        if pos + max_len + 1 > len(all_tokens):
            continue
        seqs.append(data(pos))
    return seqs


class _WikiTextDataset(Dataset):
    def __init__(self, sentences_tokens, max_len, **kwargs):
        super(_WikiTextDataset, self).__init__(**kwargs)
        self.seqs = _seq_data_cut(sentences_tokens, max_len)

    def __getitem__(self, item):
        return self.seqs[item]

    def __len__(self):
        return len(self.seqs)


class collate_fn:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, data):
        batch_tokens = []
        batch_pred_positions = []
        batch_labels = []
        for seq in data:
            tokens, pred_positions, mlm_labels = _get_mlm_data_from_tokens(seq, self.vocab)
            batch_tokens.append(tokens)
            batch_pred_positions.append(pred_positions)
            batch_labels.append(mlm_labels)
        return (torch.tensor(batch_tokens, dtype=torch.long),
                torch.tensor(batch_pred_positions, dtype=torch.long)), torch.tensor(batch_labels, dtype=torch.long)


def load_wiki(batch_size, max_len, num_merge=10000):
    data_dir = './data/wikitext-2'
    train_sentences = _read_wiki(data_dir, 'wiki.train.tokens')
    val_sentences = _read_wiki(data_dir, 'wiki.valid.tokens')
    test_sentences = _read_wiki(data_dir, 'wiki.test.tokens')
    train_tokens, val_tokens, test_tokens, BPE = BPE_Encoding(train_sentences, val_sentences, test_sentences, num_merge)
    train_dataset = _WikiTextDataset(train_tokens, max_len)
    val_dataset = _WikiTextDataset(val_tokens, max_len)
    test_dataset = _WikiTextDataset(test_tokens, max_len)
    batchify = collate_fn(BPE)
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=batchify)
    val_iter = DataLoader(val_dataset, batch_size, shuffle=True, collate_fn=batchify)
    test_iter = DataLoader(test_dataset, batch_size, shuffle=True, collate_fn=batchify)
    return train_iter, val_iter, test_iter, BPE
