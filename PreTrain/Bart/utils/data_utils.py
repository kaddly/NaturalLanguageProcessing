import os
import matplotlib.pyplot as plt
import random
import pickle
from joblib import Parallel, delayed
import torch
from torch.utils.data import Dataset, DataLoader
from utils.token_utils import BytePairEncoding
from utils.sample_utils import RandomGenerator, Poisson


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrice) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrice.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()


def _read_wiki(data_dir, file_name):
    file_name = os.path.join(data_dir, file_name)
    with open(file_name, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
    # ⼤写字⺟转换为⼩写字⺟
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def BPE_Encoding(train_sentences, val_sentences, test_sentences, num_merge):
    sentences = [sentence for sentences in train_sentences for sentence in sentences]
    BPE = BytePairEncoding(sentences, num_merge, ['<unk>', '</w>', '<s>', '</s>', '<masked>', '<pad>'], min_freq=5)
    if not os.path.exists(f'../data/BPE_Decoding_token{num_merge}/'):
        train_tokens, val_tokens, test_tokens = Parallel(n_jobs=3)(
            delayed(BPE.segment_BPE)(sentences) for sentences in [train_sentences, val_sentences, test_sentences])
        os.mkdir(f'../data/BPE_Decoding_token{num_merge}/')
        with open(f'../data/BPE_Decoding_token{num_merge}/train_tokens.plk', 'wb') as f:
            pickle.dump(train_tokens, f)
        with open(f'../data/BPE_Decoding_token{num_merge}/val_tokens.plk', 'wb') as f:
            pickle.dump(val_tokens, f)
        with open(f'../data/BPE_Decoding_token{num_merge}/test_tokens.plk', 'wb') as f:
            pickle.dump(test_tokens, f)
    else:
        with open(f'../data/BPE_Decoding_token{num_merge}/train_tokens.plk', 'rb') as f:
            train_tokens = pickle.load(f)
        with open(f'../data/BPE_Decoding_token{num_merge}/val_tokens.plk', 'rb') as f:
            val_tokens = pickle.load(f)
        with open(f'../data/BPE_Decoding_token{num_merge}/test_tokens.plk', 'rb') as f:
            test_tokens = pickle.load(f)
    return train_tokens, val_tokens, test_tokens, BPE


def build_inputs_with_special_tokens(tokens_a, tokens_b=None):
    """获取输⼊序列的词元及其⽚段索引"""
    if tokens_b is None:
        return ['<s>'] + tokens_a + ['</s>']
    cls = ['<s>']
    sep = ['</s>']
    return cls + tokens_a + sep + sep + tokens_b + sep


def _get_nsp_data_from_paragraph(paragraph, max_len, is_sentence_permutation=True):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        if is_sentence_permutation:
            tokens_a, tokens_b = Sentence_Permutation(paragraph[i], paragraph[i + 1])
        else:
            tokens_a, tokens_b = Document_Rotation(paragraph[i], paragraph[i + 1])
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens = build_inputs_with_special_tokens(tokens_a, tokens_b)
        nsp_data_from_paragraph.append(tokens)
    return nsp_data_from_paragraph


# ⽣成遮蔽语⾔模型任务的数据
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab, reconstruct_way):
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


def _get_mlm_data_from_tokens(tokens, vocab, reconstruct_way):
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        # 在遮蔽语⾔模型任务中不会预测特殊词元
        if token in ['<s>', '</s>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语⾔模型任务中预测15%的随机词元
    num_mlm_pred = max(1, round(len(tokens) * 0.15) if reconstruct_way == 'Token_Masking' else round(len(tokens) * 0.3))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_pred,
                                                                      vocab, reconstruct_way)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def Token_Masking(tokens):
    pass


def Token_Deletion(tokens):
    pass


def Text_Infilling(tokens):
    pass


def Sentence_Permutation(token_a, token_b, replace_probability=0.5):
    if random.random() < replace_probability:
        return token_a, token_b
    else:
        return token_b, token_a


def Document_Rotation(token_a, token_b):
    rotation_idx = random.choice(range(len(token_a)))
    return token_a[rotation_idx:], token_b + token_a[:rotation_idx]


def reconstruct_tokens(paragraphs, reconstruct_ways, vocab, max_len):
    reconstruct_way = random.choices(reconstruct_ways, k=2)
    examples = []
    if 'Sentence_Permutation' in reconstruct_way:
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, max_len, is_sentence_permutation=True))
    elif 'Document_Rotation' in reconstruct_way:
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, max_len, is_sentence_permutation=False))
    if 'Token_Masking' in reconstruct_way:
        examples = [_get_mlm_data_from_tokens(example, vocab, 'Token_Masking') for example in examples]
    elif 'Token_Deletion' in reconstruct_way:
        examples = [_get_mlm_data_from_tokens(example, vocab, 'Token_Deletion') for example in examples]
    elif 'Text_Infilling' in reconstruct_way:
        examples = [_get_mlm_data_from_tokens(example, vocab, 'Text_Infilling') for example in examples]
    return examples


class _wiki_dataset(Dataset):
    def __init__(self, **kwargs):
        super(_wiki_dataset, self).__init__(**kwargs)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


def load_data_wiki(batch_size, max_len, reconstruct_ways=['Sentence_Permutation', 'Text_Infilling'], num_merge=10000):
    data_dir = '../data/wikitext-2'
    train_sentences = _read_wiki(data_dir, 'wiki.train.tokens')
    val_sentences = _read_wiki(data_dir, 'wiki.valid.tokens')
    test_sentences = _read_wiki(data_dir, 'wiki.test.tokens')
    train_tokens, val_tokens, test_tokens, BPE = BPE_Encoding(train_sentences, val_sentences, test_sentences, num_merge)
    reconstruct_tokens(train_tokens, reconstruct_ways, BPE, max_len)
    reconstruct_tokens(val_tokens, reconstruct_ways, BPE, max_len)
    reconstruct_tokens(test_tokens, reconstruct_ways, BPE, max_len)
    return
