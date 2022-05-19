import torch
from torch import nn


class ELMO(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, **kwargs):
        super(ELMO, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # forward LSTM
        self.fs = nn.ModuleList([nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size, bidirectional=False)
                                 if i == 0 else
                                 nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=False)
                                 for i in range(num_layers)])
        self.f_dense = nn.Linear(self.hidden_size, vocab_size)
        # backward LSTM
        self.bs = nn.ModuleList([nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size, bidirectional=False)
                                 if i == 0 else
                                 nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=False)
                                 for i in range(num_layers)])
        self.b_dense = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, seqs, state_f, state_b):
        # (num_steps,batch_size,embedding_size)
        embedding = self.embedding(seqs.T)
        self.fxs = [embedding.permute(1, 0, 2)]
        self.bxs = [embedding.permute(1, 0, 2)]
        for fl, bl in zip(self.fs, self.bs):
            output_f, state_f = fl(embedding, state_f)
            self.fxs.append(output_f.permute(1, 0, 2))
            output_b, state_b = bl(torch.flip(embedding, dims=[0, ]), state_b)
            self.bxs.append(torch.flip(output_b, dims=[0, ]).permute(1, 0, 2))
        return self.f_dense(self.fxs[-1]), self.b_dense(self.bxs[-1]), state_f, state_b

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.fs[0], nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((1, batch_size, self.hidden_size), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (
                torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    @property
    def get_embedding(self, data, device):
        state_f = self.begin_state(batch_size=data.shape[0], device=device)
        state_b = self.begin_state(batch_size=data.shape[0], device=device)
        self(data, state_f, state_b)
        xs = [
                 torch.cat((self.fxs[0], self.bxs[0]), dim=2).cpu().data.numpy()
             ] + [
                 torch.cat((f, b), dim=2).cpu().data.numpy()
                 for f, b in zip(self.fxs[1:], self.bxs[1:])
             ]
        for x in xs:
            print("layers shape=", x.shape)
        return xs
