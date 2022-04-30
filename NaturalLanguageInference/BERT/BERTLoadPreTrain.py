import json
import os
import torch
from torch import nn
from token_utils import Vocab
from BERT import BERTModel


def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = './PretrainedParams/' + pretrained_model
    # 定义空词表以加载预定义词表
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = BERTModel(len(vocab), num_hiddens, norm_shape=[256], ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                     num_heads=4, num_layers=2, dropout=0.2, max_len=max_len, key_size=256, query_size=256,
                     value_size=256, hid_in_features=256, mlm_in_features=256, nsp_in_features=256)
    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bert, vocab


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
