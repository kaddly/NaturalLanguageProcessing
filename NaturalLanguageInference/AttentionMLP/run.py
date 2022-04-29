import torch
from torch import nn
from data_utils import load_data_snli
from token_utils import TokenEmbedding
from AttentionMLP import DecomposableAttention
from train_eval import train_snli, predict_snli

if __name__ == '__main__':
    batch_size, num_steps = 256, 50
    train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)
    embed_size, num_hiddens = 100, 200
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    net = DecomposableAttention(vocab, embed_size, num_hiddens)
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    lr, num_epochs = 0.001, 4
    trainer = torch.optim.Adam(params=net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    train_snli(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    print(predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'], devices[0]))
