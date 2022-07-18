import torch
from torch import nn
from .BertLayer import EncoderBlock


class RoBERTaEncoder(nn.Module):
    """RoBERTa编码器"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
        super(RoBERTaEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}",
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                                              ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌⼊是可学习的，因此我们创建⼀个⾜够⻓的位置嵌⼊参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量⼤⼩，最⼤序列⻓度，num_hiddens）
        X = self.token_embedding(tokens)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    """BERT的掩蔽语⾔模型任务"""

    def __init__(self, vocab_size, num_hiddens, num_input=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_input, num_hiddens), nn.ReLU(), nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class RoBERTaModel(nn.Module):
    """RoBERTa模型"""

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, max_len=1000, key_size=768, query_size=768, value_size=768, hid_in_features=768,
                 mlm_in_features=768):
        super(RoBERTaModel, self).__init__()
        self.encoder = RoBERTaEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                                   num_layers, dropout, max_len, key_size, query_size, value_size)

        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)

    def forward(self, tokens, valid_len=None, pred_positions=None):
        encoded_X = self.encoder(tokens, valid_len)
        if pred_positions is not None:
            mlm_y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_y_hat = None
        return encoded_X, mlm_y_hat
