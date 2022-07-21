import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from token_utils import BytePairEncoding, Vocab, tokenize


def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    contexts = []
    with open(file_name, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if len(line.split(' . ')) < 2:
                continue
            contexts.extend(line.strip().lower().split(' . '))
    return contexts


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
                masked_token = random.choice(vocab.idx_to_token)
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


class _WikiTextDataset(Dataset):
    def __init__(self, **kwargs):
        super(_WikiTextDataset, self).__init__(**kwargs)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def load_wiki(batch_size, max_len):
    data_dir = './data/wikitext-2'
    sentences = _read_wiki(data_dir)
    BRE = BytePairEncoding(sentences, 5000, ['<unk>', '</w>', '<mask>', '<seq>'])
    tokens = BRE.segment_BPE(sentences)
    print(tokens)


load_wiki(32, 64)
