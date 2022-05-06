import torch
from torch import nn
from BiRNN import BiRNN
from data_utils import load_data_THUCNews
from train_eval import train

if __name__ == '__main__':
    batch_size = 128
    num_step = 32
    num_epochs = 20
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    devices = devices if devices else [torch.device('cpu')]
    train_iter, test_iter, dev_iter, vocab = load_data_THUCNews(batch_size, num_step)
    loss = nn.CrossEntropyLoss()
    lr = 1e-3
    embed_size, num_hiddens, num_layers = 300, 128, 2
    model = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    train(model, train_iter, dev_iter, test_iter, loss, devices, lr, num_epochs)
