import torch
from torch import nn
from data_utils import load_data_imdb
from token_utils import Vocab, TokenEmbedding
from train_eval import train_sentiment, predict_sentiment
from TextCNN import TextCNN

if __name__ == '__main__':
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb(batch_size)
    embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.data.copy_(embeds)
    net.constant_embedding.weight.requires_grad = False
    lr, num_epochs = 0.001, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    train_sentiment(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    print(predict_sentiment(net, vocab, 'this movie is so great', devices[0]))
    print(predict_sentiment(net, vocab, 'this movie is so bad', devices[0]))


