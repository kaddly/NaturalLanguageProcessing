import torch
from torch import nn


class ELMO(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, **kwargs):
        super(ELMO, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # forward LSTM
        self.fs = nn.ModuleList([nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=False)
                                 if i == 0 else
                                 nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=False)
                                 for i in range(num_layers)])
        self.f_dense = nn.Linear(hidden_size, vocab_size)
        # backward LSTM
        self.bs = nn.ModuleList([nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=False)
                                 if i == 0 else
                                 nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=False)
                                 for i in range(num_layers)])
        self.b_dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, seqs, state):
        # (num_steps,batch_size,embedding_size)
        embedding = self.embedding(seqs.T)
        self.fxs = [embedding.permute(1, 0, 3)]
        self.bxs = [embedding.permute(1, 0, 3)]
        for fl, bl in zip(self.fs, self.bs):
            output_f, state = fl(embedding, state)
            self.fxs.append(output_f.permute(1, 0, 3))
            output_b, state = bl(torch.flip(embedding, dims=[0, ]), state)
            self.bxs.append(torch.flip(output_b, dims=[0, ]).permute(1, 0, 3))
        return self.f_dense(self.fxs[-1].reshape(-1, self.fxs[-1].shape[-1])), self.b_dense(
            self.bxs[-1].reshape(-1, self.bxs[-1].shape[-1])), state

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
    def get_embedding(self):
        xs = [
                 torch.cat((self.fxs[0], self.bxs[0]), dim=2).cpu().data.numpy()
             ] + [
                 torch.cat((f, b), dim=2).cpu().data.numpy()
                 for f, b in zip(self.fxs[1:], self.bxs[1:])
             ]
        for x in xs:
            print("layers shape=", x.shape)
        return xs
